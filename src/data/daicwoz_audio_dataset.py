"""DAIC-WOZ audio dataset adapter using BaseAudioDataset."""

import math
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from config.config import ProjectConfig, load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger

from .audio_base_dataset import BaseAudioDataset

# Load DAIC-WOZ config and define global paths
_cfg = load_config()
# Load raw YAML to access daic_woz section
cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
_cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
with open(_cfg_path, 'r') as f:
    _raw_cfg = yaml.safe_load(f)
_daic_cfg = _raw_cfg.get("daic_woz", {})

SESSIONS_DIR = PROJECT_ROOT / _daic_cfg.get("sessions_dir", "data/raw/daic_woz/sessions")
SPLITS_DIR = PROJECT_ROOT / _daic_cfg.get("splits_dir", "data/raw/daic_woz/splits")

logger = get_logger(__name__)


def find_audio_file_for_participant(participant_id: str) -> Optional[Path]:
    """
    Given a participant ID like '300', search in SESSIONS_DIR/300 for a .wav file.

    Prefer files that contain 'AUDIO' in their name if multiple exist.
    Return the Path if found, else None.
    """
    session_dir = SESSIONS_DIR / participant_id
    if not session_dir.exists():
        logger.warning(f"Session directory not found for participant {participant_id}: {session_dir}")
        return None

    wav_files = list(session_dir.rglob("*.wav"))
    if not wav_files:
        logger.warning(f"No .wav files found for participant {participant_id} in {session_dir}")
        return None

    # Prefer files with 'AUDIO' in the name, else fallback to the first wav
    preferred = [p for p in wav_files if "AUDIO" in p.name.upper()]
    if preferred:
        return preferred[0]

    return wav_files[0]


class DAICWOZAudioDataset(BaseAudioDataset):
    """DAIC-WOZ audio dataset for depression detection."""

    dataset_name = "daicwoz"

    def __init__(
        self,
        cfg: ProjectConfig,
        csv_path: Path,
        split_name: str = "train",
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        max_duration: float = 10.0,
        window_size: Optional[float] = 10.0, # in seconds
        hop_size: Optional[float] = 5.0,     # in seconds for overlap
        feature_type: str = "mfcc",
    ):
        """
        Initialize DAIC-WOZ audio dataset.

        Args:
            cfg: Project configuration
            csv_path: Path to metadata CSV file
            split_name: Dataset split name (train, dev, test)
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            max_duration: Maximum duration in seconds (ignored if window_size is set)
            window_size: Size of sliding window segments in seconds
            hop_size: Hop size for sliding window in seconds
            feature_type: Type of features to return ("mfcc" or "raw")
        """
        self.logger = get_logger(__name__)
        self.window_size = window_size
        self.hop_size = hop_size

        # Load CSV metadata
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.logger.error(f"Failed to load CSV from {csv_path}: {e}")
            raise

        # Identify participant ID column (case-insensitive)
        id_col = None
        for candidate in ["Participant_ID", "participant_ID", "ID", "id"]:
            if candidate in df.columns:
                id_col = candidate
                break

        if id_col is None:
            self.logger.warning(f"Participant ID column not found. Available columns: {list(df.columns)}")
            raise ValueError(f"Participant ID column not found. Available columns: {list(df.columns)}")

        # For train/dev, look for label column, prefer PHQ8_Binary, else PHQ8_Score
        label_col = None
        for candidate in ["PHQ8_Binary", "PHQ8_Score", "PHQ8", "phq8_score", "phq8", "PHQ-8", "label"]:
            if candidate in df.columns:
                label_col = candidate
                break

        # For test split, there may be no label column; in that case we can create dummy labels or return a dataset without labels.
        if label_col is None:
            self.logger.info(f"No label column found for {split_name}. Proceeding without labels.")

        records = []
        for idx, row in df.iterrows():
            try:
                raw_id = row[id_col]

                # Normalise participant ID:
                # - If it's a float like 303.0 -> "303"
                # - If it's an int -> "303"
                # - Else -> stripped string without trailing ".0"
                if isinstance(raw_id, (int,)):
                    pid = str(raw_id)
                elif isinstance(raw_id, float):
                    if math.isfinite(raw_id):
                        pid = str(int(raw_id))
                    else:
                        continue
                else:
                    pid_str = str(raw_id).strip()
                    if pid_str.endswith(".0"):
                        pid_str = pid_str[:-2]
                    pid = pid_str

                if not pid:
                    continue

                audio_path = find_audio_file_for_participant(pid)
                if audio_path is None:
                    continue

                if label_col is not None:
                    label_val = row[label_col]
                    # If it's PHQ8_Score, convert to binary (>= 10)
                    if label_col in ["PHQ8_Score", "PHQ8", "phq8_score", "phq8", "PHQ-8"]:
                        try:
                            phq_score = float(label_val)
                            label_val = 1 if phq_score >= 10 else 0
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Row {idx}: Invalid PHQ-8 score '{label_val}': {e}, skipping")
                            continue
                    # If it's already PHQ8_Binary or label, use as-is (but convert to int if needed)
                    else:
                        try:
                            label_val = int(label_val)
                        except (ValueError, TypeError):
                            # If it's already a valid label, keep it
                            pass
                else:
                    label_val = None

                records.append(
                    {
                        "participant_id": pid,
                        "audio_path": str(audio_path),
                        "label": label_val,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Row {idx}: Error processing row: {e}, skipping")
                continue

        if not records:
            self.logger.warning(f"No usable records for split {split_name} after resolving audio paths.")
            raise ValueError(f"No usable records for split {split_name} after resolving audio paths.")

        # Convert records to samples format expected by BaseAudioDataset
        samples: List[Dict] = []
        for record in records:
            if self.window_size is not None:
                # Basic segmentation: we don't know audio duration yet,
                # but we can get it from the file.
                try:
                    info = torchaudio.info(record["audio_path"])
                    total_duration = info.num_frames / info.sample_rate

                    start = 0.0
                    while start + self.window_size <= total_duration:
                        sample = {
                            "audio_path": Path(record["audio_path"]),
                            "label": record["label"],
                            "start_sec": start,
                            "duration_sec": self.window_size,
                            "participant_id": record["participant_id"]
                        }
                        samples.append(sample)
                        start += self.hop_size
                except Exception as e:
                    self.logger.warning(f"Failed to get info for {record['audio_path']}: {e}")
                    # Fallback to single full-length or fixed-length sample
                    samples.append({
                        "audio_path": Path(record["audio_path"]),
                        "label": record["label"],
                        "participant_id": record["participant_id"]
                    })
            else:
                sample = {
                    "audio_path": Path(record["audio_path"]),
                    "label": record["label"],
                }
                if "participant_id" in record:
                    sample["participant_id"] = record["participant_id"]
                samples.append(sample)

        # Initialize base class
        super().__init__(
            cfg=cfg,
            samples=samples,
            dataset_name=self.dataset_name,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_duration=max_duration if window_size is None else window_size,
            split_name=split_name,
            feature_type=feature_type,
        )


def create_daicwoz_datasets(
    cfg: Optional[ProjectConfig] = None,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    max_duration: float = 10.0,
) -> Tuple[Optional[DAICWOZAudioDataset], Optional[DAICWOZAudioDataset], Optional[DAICWOZAudioDataset]]:
    """
    Create DAIC-WOZ train, validation, and test datasets.

    Args:
        cfg: Project configuration. If None, loads from default config.
        sample_rate: Target sample rate for audio
        n_mfcc: Number of MFCC coefficients
        max_duration: Maximum duration in seconds

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
        Missing datasets are returned as None.
    """
    if cfg is None:
        cfg = load_config()

    logger = get_logger(__name__)

    # Look for split-specific CSV files (try AVEC2017 names first, then simple names)
    train_csv_avec = SPLITS_DIR / "train_split_Depression_AVEC2017.csv"
    dev_csv_avec = SPLITS_DIR / "dev_split_Depression_AVEC2017.csv"
    test_csv_avec = SPLITS_DIR / "test_split_Depression_AVEC2017.csv"

    train_csv = train_csv_avec if train_csv_avec.exists() else SPLITS_DIR / "train.csv"
    dev_csv = dev_csv_avec if dev_csv_avec.exists() else SPLITS_DIR / "dev.csv"
    test_csv = test_csv_avec if test_csv_avec.exists() else SPLITS_DIR / "test.csv"

    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Check if split files exist
    if train_csv.exists() and dev_csv.exists() and test_csv.exists():
        logger.info("Found train/dev/test split files")
        try:
            train_dataset = DAICWOZAudioDataset(
                cfg=cfg,
                csv_path=train_csv,
                split_name="train",
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                max_duration=max_duration,
            )
        except Exception as e:
            logger.error(f"Failed to create train dataset: {e}")

        try:
            val_dataset = DAICWOZAudioDataset(
                cfg=cfg,
                csv_path=dev_csv,
                split_name="dev",
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                max_duration=max_duration,
            )
        except Exception as e:
            logger.error(f"Failed to create dev dataset: {e}")

        try:
            test_dataset = DAICWOZAudioDataset(
                cfg=cfg,
                csv_path=test_csv,
                split_name="test",
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                max_duration=max_duration,
            )
        except Exception as e:
            logger.error(f"Failed to create test dataset: {e}")

    else:
        # Look for CSV files in splits directory
        csv_paths = sorted(SPLITS_DIR.glob("*.csv"))
        if not csv_paths:
            logger.warning(f"No CSV files found in {SPLITS_DIR}")
            return None, None, None
        elif len(csv_paths) == 1:
            logger.info(f"Found single metadata file: {csv_paths[0]}, treating as 'train'")
            try:
                train_dataset = DAICWOZAudioDataset(
                    cfg=cfg,
                    csv_path=csv_paths[0],
                    split_name="train",
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    max_duration=max_duration,
                )
            except Exception as e:
                logger.error(f"Failed to create train dataset: {e}")
        else:
            logger.warning(
                f"Multiple CSV files found in {SPLITS_DIR}: {csv_paths}. "
                "Please organize into train.csv, dev.csv, test.csv or specify manually."
            )

    return train_dataset, val_dataset, test_dataset


__all__ = ["DAICWOZAudioDataset", "create_daicwoz_datasets"]

