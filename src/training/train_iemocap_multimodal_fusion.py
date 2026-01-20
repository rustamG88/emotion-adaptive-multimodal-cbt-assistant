"""Full multimodal fusion training pipeline for IEMOCAP emotion classifier."""

from pathlib import Path
import sys
import os
from typing import Dict, Tuple, List

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchaudio.transforms as AT
import yaml

from tqdm import tqdm
from transformers import AutoTokenizer

from config.config import load_config, PROJECT_ROOT
from src.utils.seed_utils import set_global_seed
from src.utils.logging_utils import get_logger

from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.iemocap_multimodal_fusion_model import build_iemocap_fusion_model


def extract_mfcc_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
) -> torch.Tensor:
    """Extract MFCC features from audio waveform tensor."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mfcc_transform = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(waveform)

    if mfcc.shape[0] == 1:
        mfcc = mfcc.squeeze(0)

    return mfcc


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1.0
    weights = 1.0 / torch.clamp(counts, min=1.0)
    return weights


def multimodal_collate_fn(batch: List[Dict], tokenizer, sample_rate: int = 16000, n_mfcc: int = 40, device: torch.device = None):
    """
    Collate function for multimodal batches.

    Args:
        batch: List of samples from dataset
        tokenizer: Text tokenizer
        sample_rate: Audio sample rate
        n_mfcc: Number of MFCC coefficients
        device: Device to move tensors to

    Returns:
        Dictionary with batched tensors
    """
    texts = []
    audio_waveforms = []
    video_frames = []
    emotions = []

    for sample in batch:
        texts.append(sample.get("text", ""))
        audio_waveforms.append(sample.get("audio"))
        video_frames.append(sample.get("video"))
        emotions.append(sample.get("emotion"))

    # Tokenize text
    text_encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = text_encodings["input_ids"]
    attention_mask = text_encodings["attention_mask"]

    # Process MFCC features
    if audio_waveforms[0] is not None:
        # Check if they are already MFCC tensors (likely from dataset)
        if isinstance(audio_waveforms[0], torch.Tensor) and audio_waveforms[0].dim() == 2:
            # They are already (n_mfcc, time)
            padded_mfcc = []
            max_time = max(mfcc.shape[1] for mfcc in audio_waveforms if mfcc is not None)
            for mfcc in audio_waveforms:
                if mfcc is None:
                    mfcc = torch.zeros(n_mfcc, max_time)
                if mfcc.shape[1] < max_time:
                    padding = max_time - mfcc.shape[1]
                    mfcc = torch.nn.functional.pad(mfcc, (0, padding))
                padded_mfcc.append(mfcc)
            audio_mfcc = torch.stack(padded_mfcc)
        else:
            # Extract MFCC from audio waveforms (fallback)
            mfcc_features = []
            for waveform in audio_waveforms:
                if waveform is not None and isinstance(waveform, torch.Tensor):
                    waveform_cpu = waveform.cpu() if waveform.is_cuda else waveform
                    mfcc = extract_mfcc_from_waveform(waveform_cpu, sample_rate=sample_rate, n_mfcc=n_mfcc)
                    mfcc_features.append(mfcc)
                else:
                    mfcc_features.append(torch.zeros(n_mfcc, 100, dtype=torch.float32))

            max_time = max(mfcc.shape[1] for mfcc in mfcc_features)
            padded_mfcc = []
            for mfcc in mfcc_features:
                if mfcc.shape[1] < max_time:
                    padding = max_time - mfcc.shape[1]
                    mfcc = torch.nn.functional.pad(mfcc, (0, padding))
                padded_mfcc.append(mfcc)
            audio_mfcc = torch.stack(padded_mfcc)
    else:
        audio_mfcc = None

    # Stack video frames
    if video_frames[0] is not None:
        video_batch = torch.stack(video_frames)  # (batch, C, H, W)
    else:
        video_batch = None

    # Move to device if provided
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if audio_mfcc is not None:
            audio_mfcc = audio_mfcc.to(device)
        if video_batch is not None:
            video_batch = video_batch.to(device)

    return {
        "text_input_ids": input_ids,
        "text_attention_mask": attention_mask,
        "audio_mfcc": audio_mfcc,
        "video_frames": video_batch,
        "emotions": emotions,
    }


def create_fusion_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create multimodal dataloaders for IEMOCAP fusion training.

    Args:
        cfg: Project configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = get_logger("iemocap_fusion_data")

    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    fusion_cfg = raw_cfg.get("iemocap_fusion", {})
    video_cfg = raw_cfg.get("iemocap_video", {})
    audio_cfg = raw_cfg.get("iemocap_audio", {})

    image_size = video_cfg.get("image_size", 224)
    batch_size = fusion_cfg.get("batch_size", 4)  # Smaller batch for multimodal
    sample_rate = audio_cfg.get("audio_sample_rate", 16000)
    n_mfcc = audio_cfg.get("n_mfcc", 40)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)

    # Load datasets
    train_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["text", "audio", "video"],
        split="train",
        image_size=image_size,
        is_training=True,
    )

    val_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["text", "audio", "video"],
        split="val",
        image_size=image_size,
        is_training=False,
    )

    test_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["text", "audio", "video"],
        split="test",
        image_size=image_size,
        is_training=False,
    )

    if len(train_dataset) == 0:
        logger.warning("Train IEMOCAP multimodal dataset is empty. Cannot train fusion model.")
        return None, None, None

    # Get emotion labels and create mapping
    emotion_labels = train_dataset.emotion_labels
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    num_classes = len(emotion_labels)

    logger.info(f"Number of emotion classes: {num_classes}")
    logger.info(f"Emotion labels: {emotion_labels}")

    # Store label mapping in datasets
    train_dataset.emotion_to_id = emotion_to_id
    train_dataset.id_to_emotion = {v: k for k, v in emotion_to_id.items()}
    val_dataset.emotion_to_id = emotion_to_id
    val_dataset.id_to_emotion = {v: k for k, v in emotion_to_id.items()}
    test_dataset.emotion_to_id = emotion_to_id
    test_dataset.id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    # Create collate function with tokenizer
    def collate_fn(batch):
        return multimodal_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            device=cfg.device.device,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        )

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    logger,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Fusion Train]")
    for batch in progress_bar:
        # Extract inputs
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        audio_mfcc = batch["audio_mfcc"].to(device) if batch["audio_mfcc"] is not None else None
        video_frames = batch["video_frames"].to(device) if batch["video_frames"] is not None else None
        emotions = batch["emotions"]

        # Convert emotions to label IDs
        emotion_to_id = train_loader.dataset.emotion_to_id
        labels = torch.tensor([emotion_to_id[emotion] for emotion in emotions], dtype=torch.long).to(device)

        # Forward pass
        loss, logits = model(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_mfcc=audio_mfcc,
            video_frames=video_frames,
            labels=labels,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track loss and accuracy
        batch_size = text_input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "acc": correct / total_samples})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    # Per-class tracking
    class_correct = {}
    class_total = {}
    emotion_to_id = data_loader.dataset.emotion_to_id
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="[Val]")
        for batch in progress_bar:
            # Extract inputs
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            audio_mfcc = batch["audio_mfcc"].to(device) if batch["audio_mfcc"] is not None else None
            video_frames = batch["video_frames"].to(device) if batch["video_frames"] is not None else None
            emotions = batch["emotions"]

            # Convert emotions to label IDs
            labels = torch.tensor([emotion_to_id[emotion] for emotion in emotions], dtype=torch.long).to(device)

            # Forward pass
            loss, logits = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_mfcc=audio_mfcc,
                video_frames=video_frames,
                labels=labels,
            )

            # Track loss
            batch_size = text_input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()

            # Per-class metrics
            for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                emotion = id_to_emotion[label]
                if emotion not in class_total:
                    class_total[emotion] = 0
                    class_correct[emotion] = 0
                class_total[emotion] += 1
                if label == pred:
                    class_correct[emotion] += 1

            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    # Compute per-class accuracies
    per_class_metrics = {}
    for emotion in class_total:
        per_class_metrics[emotion] = class_correct[emotion] / class_total[emotion] if class_total[emotion] > 0 else 0.0

    return avg_loss, accuracy, per_class_metrics


def train_iemocap_fusion():
    """Main training function for IEMOCAP multimodal fusion."""
    cfg = load_config()
    logger = get_logger("fusion_training_iemocap")
    set_global_seed()

    logger.info("=" * 60)
    logger.info("IEMOCAP Multimodal Fusion Training")
    logger.info("=" * 60)

    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    fusion_cfg = raw_cfg.get("iemocap_fusion", {})
    batch_size = int(fusion_cfg.get("batch_size", 4))
    learning_rate = float(fusion_cfg.get("learning_rate", 1e-4))
    num_epochs = int(fusion_cfg.get("num_epochs", 15))

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Create dataloaders
    logger.info("Creating IEMOCAP multimodal dataloaders...")
    train_loader, val_loader, test_loader = create_fusion_dataloaders(cfg)

    if train_loader is None:
        logger.error("Failed to create fusion dataloaders. Exiting.")
        return

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.emotion_labels)
    logger.info(f"Number of emotion classes: {num_classes}")

    # Build fusion model
    logger.info("Building IEMOCAP fusion model...")
    try:
        model = build_iemocap_fusion_model(
            num_classes=num_classes,
            device=cfg.device.device,
        )
        logger.info(f"Model built and moved to {cfg.device.device}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise

    # Create optimizer (only optimize trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)
    logger.info(f"Optimizer: AdamW with lr={learning_rate}")

    # Compute total steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    # Create LambdaLR scheduler
    def lr_lambda(step: int) -> float:
        """Learning rate schedule with warmup and linear decay."""
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR scheduler: Linear warmup ({warmup_steps} steps) + decay")

    # Training setup
    best_val_loss = float("inf")
    best_model_path = cfg.paths.models_dir / "fusion" / "iemocap_multimodal_fusion_best.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Best model will be saved to: {best_model_path}")

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device.device,
            epoch=epoch,
            logger=logger,
        )
        logger.info(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Evaluate
        if val_loader is not None:
            val_loss, val_acc, per_class_metrics = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                device=cfg.device.device,
            )
            logger.info(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            logger.info("Per-class validation accuracies:")
            for emotion, acc in per_class_metrics.items():
                logger.info(f"  {emotion}: {acc:.4f} ({acc*100:.2f}%)")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"[SAVED] New best model (val_loss={val_loss:.4f}) to {best_model_path}")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

    # Final logging
    logger.info(f"\n{'='*50}")
    logger.info("IEMOCAP fusion training complete.")
    if val_loader is not None:
        logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    train_iemocap_fusion()

