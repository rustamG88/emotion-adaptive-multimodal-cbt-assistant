"""Comprehensive evaluation pipeline for IEMOCAP multimodal emotion classification."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import functools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from config.config import load_config, PROJECT_ROOT, _load_yaml_config
from src.data.iemocap_multimodal_dataset import IEMOCAPMultimodalDataset
from src.models.text_distilbert import build_text_model
from src.models.audio_iemocap_cnn_lstm import build_iemocap_audio_model
from src.models.video_iemocap_resnet import build_iemocap_video_model
from src.models.iemocap_multimodal_fusion_model import build_iemocap_fusion_model
from src.utils.logging_utils import get_logger
from src.utils.seed_utils import set_global_seed

logger = get_logger(__name__)


def load_models(cfg_dict: Dict, device: torch.device) -> Dict:
    """Load all trained models."""
    models = {}

    # Load text model
    logger.info("Loading text model...")
    try:
        text_model_path = PROJECT_ROOT / cfg_dict["iemocap_fusion"]["text_model_path"]
        if text_model_path.exists():
            text_model = build_text_model(
                num_labels=cfg_dict["iemocap_fusion"]["num_classes"],
                device=device,
            )
            text_model.load_state_dict(torch.load(text_model_path, map_location=device))
            text_model.eval()
            models["text"] = text_model
            logger.info(f"Loaded text model from {text_model_path}")
        else:
            logger.warning(f"Text model not found at {text_model_path}")
    except Exception as e:
        logger.error(f"Failed to load text model: {e}")

    # Load audio model
    logger.info("Loading audio model...")
    try:
        audio_model_path = PROJECT_ROOT / cfg_dict["iemocap_fusion"]["audio_model_path"]
        if audio_model_path.exists():
            audio_model = build_iemocap_audio_model(
                num_classes=cfg_dict["iemocap_fusion"]["num_classes"],
                device=device,
            )
            audio_model.load_state_dict(torch.load(audio_model_path, map_location=device))
            audio_model.eval()
            models["audio"] = audio_model
            logger.info(f"Loaded audio model from {audio_model_path}")
        else:
            logger.warning(f"Audio model not found at {audio_model_path}")
    except Exception as e:
        logger.error(f"Failed to load audio model: {e}")

    # Load video model
    logger.info("Loading video model...")
    try:
        video_model_path = PROJECT_ROOT / cfg_dict["iemocap_fusion"]["video_model_path"]
        if video_model_path.exists():
            video_model = build_iemocap_video_model(
                num_classes=cfg_dict["iemocap_fusion"]["num_classes"],
                device=device,
            )
            video_model.load_state_dict(torch.load(video_model_path, map_location=device))
            video_model.eval()
            models["video"] = video_model
            logger.info(f"Loaded video model from {video_model_path}")
        else:
            logger.warning(f"Video model not found at {video_model_path}")
    except Exception as e:
        logger.error(f"Failed to load video model: {e}")

    # Load fusion model
    logger.info("Loading fusion model...")
    try:
        fusion_model = build_iemocap_fusion_model(
            num_classes=cfg_dict["iemocap_fusion"]["num_classes"],
            device=device,
        )
        fusion_model_path = PROJECT_ROOT / cfg_dict["evaluation"]["fusion_model_path"]
        if fusion_model_path.exists():
            fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
            fusion_model.eval()
            models["fusion"] = fusion_model
            logger.info(f"Loaded fusion model from {fusion_model_path}")
        else:
            logger.warning(f"Fusion model not found at {fusion_model_path}")
    except Exception as e:
        logger.error(f"Failed to load fusion model: {e}")

    return models


def collate_multimodal_batch(batch, modalities: List[str]):
    """Custom collate function for multimodal batches."""
    texts = []
    audio_mfccs = []
    video_frames = []
    labels = []

    for item in batch:
        if "text" in modalities and item["text"] is not None:
            texts.append(item["text"])
        if "audio" in modalities and item["audio"] is not None:
            audio_mfccs.append(item["audio"])
        if "video" in modalities and item["video"] is not None:
            video_frames.append(item["video"])
        labels.append(item["emotion_id"])

    result = {"labels": torch.tensor(labels, dtype=torch.long)}

    if texts:
        from transformers import AutoTokenizer
        cfg = load_config()
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.text_model.max_seq_len,
            return_tensors="pt",
        )
        result["text_input_ids"] = encoded["input_ids"]
        result["text_attention_mask"] = encoded["attention_mask"]

    if audio_mfccs:
        # Pad audio MFCCs to same length
        max_time = max(mfcc.shape[1] for mfcc in audio_mfccs)
        padded_audio = []
        for mfcc in audio_mfccs:
            if mfcc.shape[1] < max_time:
                padding = max_time - mfcc.shape[1]
                mfcc = torch.nn.functional.pad(mfcc, (0, padding))
            elif mfcc.shape[1] > max_time:
                mfcc = mfcc[:, :max_time]
            padded_audio.append(mfcc)
        result["audio_mfcc"] = torch.stack(padded_audio)

    if video_frames:
        result["video_frames"] = torch.stack(video_frames)

    return result


def evaluate_unimodal(
    model,
    dataloader: DataLoader,
    device: torch.device,
    modality: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a unimodal model."""
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)

            if modality == "text":
                input_ids = batch["text_input_ids"].to(device)
                attention_mask = batch["text_attention_mask"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            elif modality == "audio":
                audio_mfcc = batch["audio_mfcc"].to(device)
                logits = model(audio_mfcc)
            elif modality == "video":
                video_frames = batch["video_frames"].to(device)
                logits = model(video_frames)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_multimodal(
    fusion_model,
    dataloader: DataLoader,
    device: torch.device,
    modalities: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a multimodal fusion model (bimodal or trimodal)."""
    all_preds = []
    all_labels = []

    fusion_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)

            # Prepare inputs based on modalities
            text_input_ids = None
            text_attention_mask = None
            audio_mfcc = None
            video_frames = None

            if "text" in modalities:
                text_input_ids = batch["text_input_ids"].to(device)
                text_attention_mask = batch["text_attention_mask"].to(device)
            if "audio" in modalities:
                audio_mfcc = batch["audio_mfcc"].to(device)
            if "video" in modalities:
                video_frames = batch["video_frames"].to(device)

            # Forward pass
            _, logits = fusion_model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_mfcc=audio_mfcc,
                video_frames=video_frames,
                labels=None,
            )

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Compute comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist(),
    }


def run_evaluation(
    cfg_dict: Dict,
    output_dir: Path,
    class_names: List[str],
) -> Dict:
    """Run comprehensive evaluation across all modality combinations."""
    device = torch.device(cfg_dict.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    seed = cfg_dict.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    # Load models
    models = load_models(cfg_dict, device)

    # Load test dataset
    index_path = PROJECT_ROOT / cfg_dict["evaluation"]["test_index_path"]
    logger.info(f"Loading test dataset from {index_path}")

    results = {}

    # Define all modality combinations to evaluate
    modality_configs = [
        (["text"], "text-only"),
        (["audio"], "audio-only"),
        (["video"], "video-only"),
        (["text", "audio"], "text+audio"),
        (["text", "video"], "text+video"),
        (["audio", "video"], "audio+video"),
        (["text", "audio", "video"], "text+audio+video"),
    ]

    for modalities, config_name in modality_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_name}")
        logger.info(f"{'='*60}")

        try:
            # Create dataset
            dataset = IEMOCAPMultimodalDataset(
                index_path=index_path,
                modalities=modalities,
                split="test",
            )

            if len(dataset) == 0:
                logger.warning(f"No samples found for {config_name}, skipping...")
                continue

            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=cfg_dict["evaluation"].get("batch_size", 8),
                shuffle=False,
                num_workers=cfg_dict["evaluation"].get("num_workers", 2),
                collate_fn=functools.partial(collate_multimodal_batch, modalities=modalities),
            )

            # Run evaluation
            if len(modalities) == 1:
                # Unimodal
                modality = modalities[0]
                if modality not in models:
                    logger.warning(f"Model for {modality} not available, skipping...")
                    continue
                y_pred, y_true = evaluate_unimodal(
                    models[modality],
                    dataloader,
                    device,
                    modality,
                )
            else:
                # Multimodal (bimodal or trimodal)
                if "fusion" not in models:
                    logger.warning("Fusion model not available, skipping...")
                    continue
                y_pred, y_true = evaluate_multimodal(
                    models["fusion"],
                    dataloader,
                    device,
                    tuple(modalities),
                )

            # Compute metrics
            metrics = compute_metrics(y_true, y_pred, class_names)
            results[config_name] = metrics

            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {config_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    return results


def save_results(
    results: Dict,
    output_dir: Path,
    class_names: List[str],
    timestamp: str,
):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON metrics
    json_path = output_dir / f"metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    # Save CSV summary table
    csv_rows = []
    for config_name, metrics in results.items():
        csv_rows.append({
            "Configuration": config_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Weighted F1": f"{metrics['weighted_f1']:.4f}",
        })

    df_summary = pd.DataFrame(csv_rows)
    csv_path = output_dir / f"summary_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False)
    logger.info(f"Saved summary table to {csv_path}")

    # Save detailed per-class metrics CSV
    detailed_rows = []
    for config_name, metrics in results.items():
        for class_name, class_metrics in metrics["per_class"].items():
            detailed_rows.append({
                "Configuration": config_name,
                "Class": class_name,
                "Precision": f"{class_metrics['precision']:.4f}",
                "Recall": f"{class_metrics['recall']:.4f}",
                "F1": f"{class_metrics['f1']:.4f}",
                "Support": class_metrics["support"],
            })

    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv_path = output_dir / f"per_class_metrics_{timestamp}.csv"
    df_detailed.to_csv(detailed_csv_path, index=False)
    logger.info(f"Saved per-class metrics to {detailed_csv_path}")

    # Save confusion matrices (for plotting script)
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for config_name, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"])
        np.save(cm_dir / f"cm_{config_name.replace('+', '_')}_{timestamp}.npy", cm)

    logger.info(f"Saved confusion matrices to {cm_dir}")


def evaluate_daic_woz(
    cfg_dict: Dict,
    output_dir: Path,
    timestamp: str,
) -> Optional[Dict]:
    """Evaluate DAIC-WOZ depression detection model (binary classification)."""
    logger.info("\n" + "="*60)
    logger.info("Evaluating DAIC-WOZ Depression Model")
    logger.info("="*60)

    try:
        from src.data.daic_woz_dataset import create_daic_woz_dataloaders
        from src.models.audio_daic_woz import build_daic_woz_model

        device = torch.device(cfg_dict.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        # Load model
        model_path = PROJECT_ROOT / cfg_dict["evaluation"]["daic_woz_model_path"]
        if not model_path.exists():
            logger.warning(f"DAIC-WOZ model not found at {model_path}, skipping...")
            return None

        model = build_daic_woz_model(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Load test dataloader
        cfg = load_config()
        _, _, test_loader = create_daic_woz_dataloaders(cfg)

        # Evaluate
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                audio_features = batch["audio_features"].to(device)
                labels = batch["label"].to(device)

                logits = model(audio_features)
                probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
                preds = (probs > 0.5).long() if logits.shape[1] == 1 else logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute binary metrics
        from sklearn.metrics import roc_auc_score, roc_curve

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

        # ROC AUC
        if len(np.unique(all_labels)) > 1:
            if all_probs.ndim > 1 and all_probs.shape[1] > 1:
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                roc_auc = roc_auc_score(all_labels, all_probs.flatten())
        else:
            roc_auc = 0.0

        metrics = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

        logger.info(f"DAIC-WOZ Accuracy: {accuracy:.4f}")
        logger.info(f"DAIC-WOZ F1: {f1:.4f}")
        logger.info(f"DAIC-WOZ ROC AUC: {roc_auc:.4f}")

        # Save to file
        json_path = output_dir / f"daic_woz_metrics_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved DAIC-WOZ metrics to {json_path}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to evaluate DAIC-WOZ: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def evaluate_meld(
    cfg_dict: Dict,
    output_dir: Path,
    timestamp: str,
) -> Optional[Dict]:
    """Evaluate MELD text emotion classification model."""
    logger.info("\n" + "="*60)
    logger.info("Evaluating MELD Text Model")
    logger.info("="*60)

    try:
        from src.data.meld_text_dataset import create_dataloaders
        from src.models.text_distilbert import build_text_model

        device = torch.device(cfg_dict.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        # Load model
        model_path = PROJECT_ROOT / cfg_dict["evaluation"]["meld_model_path"]
        if not model_path.exists():
            logger.warning(f"MELD model not found at {model_path}, skipping...")
            return None

        cfg = load_config()
        _, _, test_loader = create_dataloaders(cfg)

        # Get number of classes from test loader
        # This is a bit hacky, but we need to know num_classes
        # For MELD, it's typically 7 emotions
        num_classes = 7  # joy, sadness, anger, fear, surprise, disgust, neutral
        meld_class_names = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

        model = build_text_model(num_classes=num_classes, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Evaluate
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, meld_class_names)

        logger.info(f"MELD Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"MELD Macro F1: {metrics['macro_f1']:.4f}")

        # Save to file
        json_path = output_dir / f"meld_metrics_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved MELD metrics to {json_path}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to evaluate MELD: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def main():
    """Main evaluation entry point."""
    # Load config
    cfg = load_config()
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_dict = _load_yaml_config(cfg_path)

    # Get evaluation config
    eval_cfg = cfg_dict.get("evaluation", {})

    # Output directory
    output_dir = PROJECT_ROOT / eval_cfg.get("output_dir", "reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class names
    class_names = eval_cfg.get("class_names", [
        "anger", "happy", "sadness", "neutral", "excited",
        "frustration", "disgust", "fear", "surprise"
    ])

    # Timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log evaluation start
    logger.info("="*60)
    logger.info("Starting IEMOCAP Multimodal Evaluation")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {timestamp}")

    # Run IEMOCAP evaluation
    results = run_evaluation(cfg_dict, output_dir, class_names)

    # Save results
    save_results(results, output_dir, class_names, timestamp)

    # Optional: Evaluate DAIC-WOZ
    if eval_cfg.get("evaluate_daic_woz", False):
        daic_results = evaluate_daic_woz(cfg_dict, output_dir, timestamp)
        if daic_results:
            results["daic_woz"] = daic_results

    # Optional: Evaluate MELD
    if eval_cfg.get("evaluate_meld", False):
        meld_results = evaluate_meld(cfg_dict, output_dir, timestamp)
        if meld_results:
            results["meld"] = meld_results

    logger.info("="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for config_name, metrics in results.items():
        if config_name in ["daic_woz", "meld"]:
            # Special handling for binary/other metrics
            if "roc_auc" in metrics:
                print(f"\n{config_name.upper()}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            else:
                print(f"\n{config_name.upper()}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"\n{config_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

