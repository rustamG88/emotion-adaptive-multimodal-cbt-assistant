"""Full text-only training pipeline for DistilBERT emotion classifier on IEMOCAP dataset."""

from pathlib import Path
import sys
from typing import Tuple

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm

from config.config import load_config
from src.utils import get_logger, set_global_seed

from src.data.iemocap_text_dataset import create_iemocap_text_dataloaders
from src.models.text_distilbert import build_text_model


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    logger,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
    for batch in progress_bar:
        # Move tensors to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        loss, logits = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model: nn.Module, val_loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Tuple of (average validation loss, validation accuracy)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="[Val]")
        for batch in progress_bar:
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            loss, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_text_model() -> None:
    """
    Main training function for text-only emotion classification on IEMOCAP dataset.

    Loads configuration, creates dataloaders, builds model, and trains.
    """
    # Load configuration
    cfg = load_config()

    # Initialize logger
    logger = get_logger("text_training_iemocap")

    # Set random seed
    set_global_seed()
    logger.info(f"Random seed set to {cfg.seed}")

    # Check for debug/fast_dev_run mode
    debug_mode = getattr(cfg, "debug_mode", False)
    fast_dev_run = getattr(cfg, "fast_dev_run", False)

    if debug_mode or fast_dev_run:
        logger.info("⚠ DEBUG MODE: Using small dataset subset and limited epochs")

    # Load dataloaders
    logger.info("Loading IEMOCAP dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_iemocap_text_dataloaders(cfg)
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    # Limit dataset size in debug mode
    if debug_mode or fast_dev_run:
        from torch.utils.data import Subset
        max_train_samples = 100
        max_val_samples = 20
        train_indices = list(range(min(max_train_samples, len(train_loader.dataset))))
        val_indices = list(range(min(max_val_samples, len(val_loader.dataset))))
        train_loader.dataset = Subset(train_loader.dataset, train_indices)
        val_loader.dataset = Subset(val_loader.dataset, val_indices)
        logger.info(f"Debug mode: Using {len(train_indices)} train and {len(val_indices)} val samples")

    # Determine number of labels from dataset
    base_dataset = train_loader.dataset if not (debug_mode or fast_dev_run) else train_loader.dataset.dataset
    num_labels = base_dataset.num_labels
    logger.info(f"Number of emotion labels: {num_labels}")

    # Build model
    logger.info("Building model...")
    try:
        model = build_text_model(num_labels=num_labels, device=cfg.device.device)
        logger.info(f"Model built and moved to {cfg.device.device}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.text_model.learning_rate)
    logger.info(f"Optimizer: AdamW with lr={cfg.text_model.learning_rate}")

    # Define learning rate scheduler with warmup
    # Adjust epochs for debug mode
    num_epochs = 1 if fast_dev_run else cfg.text_model.num_epochs
    if debug_mode:
        num_epochs = min(2, num_epochs)  # Max 2 epochs in debug mode

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        """Learning rate schedule with warmup and linear decay."""
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR scheduler: Linear warmup ({warmup_steps} steps) + decay")

    # Training loop
    best_val_loss = float("inf")
    model_save_dir = cfg.paths.models_dir / "text"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = cfg.paths.models_dir / "text" / "distilbert_iemocap_best.pt"

    logger.info(f"Starting training for {cfg.text_model.num_epochs} epochs...")
    logger.info(f"Best model will be saved to: {best_model_path}")

    for epoch in range(cfg.text_model.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device.device,
            epoch=epoch,
            logger=logger,
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(cfg.device.device)
                attention_mask = batch["attention_mask"].to(cfg.device.device)
                labels = batch["labels"].to(cfg.device.device)

                # Forward pass with labels so model returns (loss, logits)
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Accumulate loss for epoch (per-batch mean)
                val_loss += loss.item()

                # Compute predictions and accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Average validation loss over number of batches (since loss is already mean per batch)
        val_loss /= len(val_loader)

        # Validation accuracy
        val_acc = correct / total if total > 0 else 0.0

        logger.info("Val loss: %.4f | Val acc: %.4f", val_loss, val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"[SAVED] Saved best model (val_loss={val_loss:.4f}) to {best_model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    logger.info(f"\n{'='*50}")
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    train_text_model()

