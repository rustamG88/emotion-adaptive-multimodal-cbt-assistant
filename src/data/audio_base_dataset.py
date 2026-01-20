"""Reusable audio dataset utilities for MFCC-based models."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from torch.utils.data import Dataset

from config.config import ProjectConfig, load_config
from src.utils import get_logger, load_numpy, save_numpy


def load_audio_mono(path: Path, sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio file as mono waveform.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Audio waveform as 1D numpy array

    Raises:
        RuntimeError: If audio loading fails
    """
    try:
        # Load audio: waveform shape is (channels, time)
        waveform, sr = torchaudio.load(str(path))

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=sample_rate)

        # Return as 1D numpy array
        return waveform.squeeze(0).numpy()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}")


def extract_mfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Extract MFCC features from audio waveform.

    Args:
        waveform: Audio waveform (1D numpy array)
        sample_rate: Sample rate of the waveform
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT (ignored, kept for compatibility)
        n_fft: FFT window size (ignored, kept for compatibility)

    Returns:
        MFCC features array of shape (time, n_mfcc)
    """
    # Convert to tensor: (1, time)
    tensor_waveform = torch.from_numpy(waveform).unsqueeze(0)

    # Create MFCC transform
    mfcc_transform = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)

    # Extract MFCC: (1, n_mfcc, time)
    mfcc = mfcc_transform(tensor_waveform)

    # Reshape to (time, n_mfcc)
    mfcc = mfcc.squeeze(0).transpose(0, 1)

    return mfcc.numpy()


class BaseAudioDataset(Dataset):
    """Base class for audio datasets with MFCC feature extraction and caching."""

    def __init__(
        self,
        cfg: ProjectConfig,
        samples: List[Dict],
        dataset_name: str,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        max_duration: float = 10.0,
        split_name: str = "train",
        subset_ratio: float = 0.8,
        feature_type: str = "mfcc", # "mfcc" or "raw"
    ):
        """
        Initialize base audio dataset.

        Args:
            cfg: Project configuration
            samples: List of sample dictionaries, each with:
                - "audio_path": Path to audio file
                - "label": int label
                - Any other metadata
            dataset_name: Name of the dataset (for cache organization)
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            max_duration: Maximum duration in seconds (crop/pad to this)
            split_name: Dataset split name (train, dev, test)
            subset_ratio: Fraction of samples to use for training split (default: 0.8)
            feature_type: Type of features to return ("mfcc" or "raw")
        """
        self.cfg = cfg
        self.samples = samples
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.split_name = split_name
        self.feature_type = feature_type

        # Setup cache directory
        self.cache_dir = cfg.paths.processed_dir / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Apply subsetting for training split
        if split_name == "train" and subset_ratio < 1.0:
            subset_size = int(len(self.samples) * subset_ratio)
            import random

            random.seed(cfg.seed)
            self.samples = random.sample(self.samples, subset_size)
            get_logger(__name__).info(
                f"Using {len(self.samples)}/{len(samples)} samples "
                f"({subset_ratio * 100:.1f}%) for {split_name}"
            )

        self.logger = get_logger(__name__)

    def _get_cache_path(self, idx: int) -> Path:
        """
        Get cache file path for a sample.

        Args:
            idx: Sample index

        Returns:
            Path to cached .npy file
        """
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        # Use filename stem as unique identifier
        filename_stem = audio_path.stem

        # Create cache filename
        cache_filename = (
            f"{self.dataset_name}_{self.split_name}_{filename_stem}_"
            f"sr{self.sample_rate}_mfcc{self.n_mfcc}_max{self.max_duration:.1f}s.npy"
        )

        return self.cache_dir / cache_filename

    def _load_or_compute_mfcc(self, idx: int) -> np.ndarray:
        """
        Load MFCC features from cache or compute and cache them.

        Args:
            idx: Sample index

        Returns:
            MFCC features array of shape (time, n_mfcc)
        """
        cache_path = self._get_cache_path(idx)

        # Try to load from cache
        if cache_path.exists():
            try:
                return load_numpy(cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

        # Compute MFCC
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        try:
            # Load audio
            waveform = load_audio_mono(audio_path, self.sample_rate)

            # Crop or pad to max_duration
            max_samples = int(self.max_duration * self.sample_rate)
            if len(waveform) > max_samples:
                # Crop
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                # Pad with zeros
                waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode="constant")

            # Extract MFCC
            mfcc = extract_mfcc(
                waveform=waveform,
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
            )

            # Save to cache
            try:
                save_numpy(mfcc, cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to save cache to {cache_path}: {e}")

            return mfcc

        except Exception as e:
            self.logger.warning(f"Failed to process audio {audio_path}: {e}. Skipping sample.")
            # Return zero-filled array as fallback
            # Estimate time frames based on max_duration
            # Using approximate frame calculation for torchaudio MFCC
            n_frames = int(self.max_duration * self.sample_rate / 512)  # approximate
            return np.zeros((n_frames, self.n_mfcc), dtype=np.float32)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    @property
    def num_labels(self) -> int:
        """Return number of unique labels in the dataset."""
        if not self.samples:
            return 0
        unique_labels = set(sample["label"] for sample in self.samples)
        return len(unique_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - "features": torch.Tensor of shape (n_mfcc, time) for MFCC, or (time,) for raw
                - "label": int label
        """
        if self.feature_type == "mfcc":
            mfcc = self._load_or_compute_mfcc(idx)
            sample = self.samples[idx]
            label = int(sample["label"])

            # Convert to torch tensor and transpose to (n_mfcc, time)
            # mfcc from extract_mfcc is (time, n_mfcc)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
            features = mfcc_tensor.transpose(0, 1)  # (n_mfcc, time)
        else:
            # Load raw waveform
            sample = self.samples[idx]
            audio_path = sample["audio_path"]
            label = int(sample["label"])

            waveform = load_audio_mono(audio_path, self.sample_rate)

            # Crop or pad to max_duration
            max_samples = int(self.max_duration * self.sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode="constant")

            features = torch.tensor(waveform, dtype=torch.float32)

        return {
            "features": features,
            "label": torch.tensor(label, dtype=torch.long),
        }


__all__ = ["load_audio_mono", "extract_mfcc", "BaseAudioDataset"]

