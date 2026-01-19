"""IEMOCAP multimodal dataset loader supporting text, audio, and video modalities."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset
import cv2

from config.config import ProjectConfig, load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.data.video_transforms import load_and_transform_video_frame
from src.data.audio_base_dataset import load_audio_mono, extract_mfcc


class IEMOCAPMultimodalDataset(Dataset):
    """IEMOCAP multimodal dataset supporting text, audio, and video modalities."""

    def __init__(
        self,
        cfg: Optional[ProjectConfig] = None,
        index_path: Optional[Path] = None,
        modalities: List[str] = None,
        split: Optional[str] = None,
        audio_sample_rate: int = 16000,
        audio_max_duration: float = 10.0,
        video_frame_idx: int = 0,  # Which frame to extract from video (deprecated, uses middle frame)
        image_size: int = 224,  # Image size for video frames
        is_training: bool = False,  # Whether to apply training transforms
    ):
        """
        Initialize IEMOCAP multimodal dataset.

        Args:
            cfg: Project configuration (if None, loads from default)
            index_path: Path to the multimodal index CSV (if None, uses default)
            modalities: List of modalities to load ["text", "audio", "video"]
            split: Dataset split to use ("train", "val", "test", or None for all)
            audio_sample_rate: Target sample rate for audio
            audio_max_duration: Maximum audio duration in seconds
            video_frame_idx: Frame index to extract from video (deprecated, uses middle frame)
            image_size: Image size for video frames (default 224)
            is_training: Whether to apply training transforms with augmentation (default False)
        """
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        self.logger = get_logger(__name__)

        # Set default modalities
        if modalities is None:
            modalities = ["text", "audio", "video"]
        self.modalities = [m.lower() for m in modalities]

        # Validate modalities
        valid_modalities = {"text", "audio", "video"}
        for mod in self.modalities:
            if mod not in valid_modalities:
                raise ValueError(f"Invalid modality: {mod}. Must be one of {valid_modalities}")

        # Set up paths
        if index_path is None:
            index_path = PROJECT_ROOT / cfg.paths.processed_dir / "iemocap_multimodal_index.csv"

        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # Load index
        self.df = pd.read_csv(self.index_path)

        # Normalize audio_path and video_path columns to strings
        if "audio_path" in self.df.columns:
            self.df["audio_path"] = self.df["audio_path"].fillna("").astype(str)
        if "video_path" in self.df.columns:
            self.df["video_path"] = self.df["video_path"].fillna("").astype(str)

        # Create emotion label to ID mapping from FULL dataset to ensure consistency
        self.emotion_labels = sorted(self.df["emotion"].unique().tolist())
        self.label_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.logger.info(f"Emotion mapping: {self.label_to_id}")

        # Filter by split if specified
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            self.logger.info(f"Filtered to {split} split: {len(self.df)} samples")

        # Debug logging before modality filtering
        self.logger.debug(f"Total rows after split filter: {len(self.df)}")
        if "audio_path" in self.df.columns:
            non_empty_audio = (self.df["audio_path"].str.strip() != "").sum()
            self.logger.debug(f"Rows with non-empty audio_path: {non_empty_audio}")
        if "video_path" in self.df.columns:
            non_empty_video = (self.df["video_path"].str.strip() != "").sum()
            self.logger.debug(f"Rows with non-empty video_path: {non_empty_video}")
        if "audio_path" in self.df.columns and "video_path" in self.df.columns:
            both_non_empty = ((self.df["audio_path"].str.strip() != "") &
                             (self.df["video_path"].str.strip() != "")).sum()
            self.logger.debug(f"Rows with both audio_path and video_path: {both_non_empty}")

        # Filter samples that have all requested modalities
        self._filter_by_modalities()

        # Audio/video processing parameters
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_duration = audio_max_duration
        # Align with DAIC/AudioCNNLSTM defaults
        self.n_mfcc = 40
        self.video_frame_idx = video_frame_idx  # Kept for backward compatibility
        self.image_size = image_size
        self.is_training = is_training

        self.logger.info(
            f"Loaded {len(self.df)} samples with modalities: {self.modalities}"
        )

    def _filter_by_modalities(self):
        """Filter samples to only include those with all requested modalities."""
        initial_count = len(self.df)

        # Filter based on requested modalities
        if "text" in self.modalities:
            self.df = self.df[self.df["text"].notna() & (self.df["text"] != "")].reset_index(drop=True)

        if "audio" in self.modalities:
            # Check for non-empty audio_path (using normalized string column)
            audio_mask = self.df["audio_path"].str.strip() != ""

            # Verify files exist for non-empty paths
            def check_audio_path(path_str):
                if not path_str or str(path_str).strip() == "":
                    return False
                try:
                    audio_path = Path(path_str)
                    # Handle relative paths - paths in CSV are relative to IEMOCAP_full_release
                    if not audio_path.is_absolute():
                        from config.config import PROJECT_ROOT, load_config
                        cfg = load_config()
                        # Paths are relative to IEMOCAP_full_release root
                        iemocap_root = PROJECT_ROOT / cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
                        audio_path_abs = iemocap_root / audio_path
                        if not audio_path_abs.exists():
                            # Fallback to project root
                            audio_path_abs = PROJECT_ROOT / audio_path
                        audio_path = audio_path_abs
                    return audio_path.exists()
                except:
                    return False

            # Check existence for all rows
            audio_exists = self.df["audio_path"].apply(check_audio_path)
            # Combine mask: non-empty AND exists
            final_audio_mask = audio_mask & audio_exists
            self.df = self.df[final_audio_mask].reset_index(drop=True)

        if "video" in self.modalities:
            # Check for non-empty video_path (using normalized string column)
            video_mask = self.df["video_path"].str.strip() != ""

            # Verify files exist for non-empty paths
            def check_video_path(path_str):
                if not path_str or str(path_str).strip() == "":
                    return False
                try:
                    video_path = Path(path_str)
                    # Handle relative paths - paths in CSV are relative to IEMOCAP_full_release
                    if not video_path.is_absolute():
                        from config.config import PROJECT_ROOT, load_config
                        cfg = load_config()
                        # Paths are relative to IEMOCAP_full_release root
                        iemocap_root = PROJECT_ROOT / cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
                        video_path_abs = iemocap_root / video_path
                        if not video_path_abs.exists():
                            # Fallback to project root
                            video_path_abs = PROJECT_ROOT / video_path
                        video_path = video_path_abs
                    return video_path.exists()
                except:
                    return False

            # Check existence for all rows
            video_exists = self.df["video_path"].apply(check_video_path)
            # Combine mask: non-empty AND exists
            final_video_mask = video_mask & video_exists
            self.df = self.df[final_video_mask].reset_index(drop=True)

        filtered_count = len(self.df)
        if filtered_count < initial_count:
            self.logger.info(
                f"Filtered {initial_count - filtered_count} samples missing requested modalities"
            )

        self.logger.info(f"Loaded {len(self.df)} samples with modalities: {self.modalities}")

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load audio and return MFCC features shaped (n_mfcc, time)."""
        try:
            waveform = load_audio_mono(audio_path, sample_rate=self.audio_sample_rate)
            # Crop or pad waveform to fixed duration
            max_samples = int(self.audio_max_duration * self.audio_sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                waveform = torch.nn.functional.pad(
                    torch.tensor(waveform, dtype=torch.float32), (0, max_samples - len(waveform))
                ).numpy()
            # Compute MFCC -> (time, n_mfcc)
            mfcc = extract_mfcc(
                waveform=waveform,
                sample_rate=self.audio_sample_rate,
                n_mfcc=self.n_mfcc,
            )
            # Return torch tensor (n_mfcc, time)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).transpose(0, 1)
            return mfcc_tensor
        except Exception as e:
            self.logger.warning(f"Failed to load/process audio {audio_path}: {e}")
            # Fallback: zero tensor shaped (n_mfcc, approx_time)
            n_frames = int(self.audio_max_duration * self.audio_sample_rate / 512)
            return torch.zeros((self.n_mfcc, n_frames), dtype=torch.float32)

    def _load_video_frame(self, video_path: Path) -> torch.Tensor:
        """Load a single frame from video with ImageNet transforms.

        Returns:
            Image tensor of shape (C, H, W) with ImageNet normalization
        """
        # Use the new video transforms
        frame_tensor = load_and_transform_video_frame(
            video_path=video_path,
            image_size=self.image_size,
            is_training=self.is_training,
        )

        if frame_tensor is None:
            self.logger.warning(f"Failed to load video frame {video_path}, using zero tensor")
            # Return zero-filled tensor with ImageNet normalization
            # Mean-subtracted zero tensor
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # Normalized zero: (0 - mean) / std = -mean / std
            normalized_zero = -mean / std
            return normalized_zero.expand(3, self.image_size, self.image_size)

        return frame_tensor

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - "text": str (if text modality requested)
                - "audio": torch.Tensor (if audio modality requested)
                - "video": torch.Tensor (if video modality requested)
                - "emotion": str
                - "utterance_id": str
        """
        row = self.df.iloc[idx]

        sample = {
            "emotion": str(row["emotion"]),
            "utterance_id": str(row["utterance_id"]),
            "emotion_id": self.label_to_id[str(row["emotion"])],
        }

        # Load requested modalities
        if "text" in self.modalities:
            sample["text"] = str(row["text"])

        if "audio" in self.modalities:
            audio_path_str = str(row["audio_path"]).strip()
            # Use same "non-empty" definition as filtering: str.strip() != ""
            if audio_path_str:
                audio_path = Path(audio_path_str)
                # Handle both absolute and relative paths
                if not audio_path.is_absolute():
                    # Paths in CSV are relative to IEMOCAP_full_release root
                    iemocap_root = PROJECT_ROOT / self.cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
                    audio_path_abs = iemocap_root / audio_path
                    if not audio_path_abs.exists():
                        # Fallback to project root
                        audio_path_abs = PROJECT_ROOT / audio_path
                    audio_path = audio_path_abs
                # Verify path exists
                if audio_path.exists():
                    sample["audio"] = self._load_audio(audio_path)

        if "video" in self.modalities:
            video_path_str = str(row["video_path"]).strip()
            # Use same "non-empty" definition as filtering: str.strip() != ""
            if video_path_str:
                video_path = Path(video_path_str)
                # Handle both absolute and relative paths
                if not video_path.is_absolute():
                    # Paths in CSV are relative to IEMOCAP_full_release root
                    iemocap_root = PROJECT_ROOT / self.cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
                    video_path_abs = iemocap_root / video_path
                    if not video_path_abs.exists():
                        # Fallback to project root
                        video_path_abs = PROJECT_ROOT / video_path
                    video_path = video_path_abs
                # Verify path exists
                if video_path.exists():
                    sample["video"] = self._load_video_frame(video_path)

        return sample

    @property
    def num_labels(self) -> int:
        """Return number of unique emotion labels."""
        return len(self.emotion_labels)


def load_iemocap_multimodal_dataset(
    cfg: Optional[ProjectConfig] = None,
    modalities: List[str] = None,
    split: Optional[str] = None,
    image_size: int = 224,
    is_training: bool = False,
) -> IEMOCAPMultimodalDataset:
    """
    Convenience function to load IEMOCAP multimodal dataset.

    Args:
        cfg: Project configuration
        modalities: List of modalities to load
        split: Dataset split ("train", "val", "test", or None)
        image_size: Image size for video frames (default 224)
        is_training: Whether to apply training transforms with augmentation (default False)

    Returns:
        IEMOCAPMultimodalDataset instance
    """
    return IEMOCAPMultimodalDataset(
        cfg=cfg,
        modalities=modalities,
        split=split,
        image_size=image_size,
        is_training=is_training,
    )


__all__ = ["IEMOCAPMultimodalDataset", "load_iemocap_multimodal_dataset"]

