from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


@dataclass
class PathsConfig:
    data_root: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    cache_dir: Path
    logs_dir: Path


@dataclass
class TextModelConfig:
    pretrained_name: str
    max_seq_len: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    train_subset_fraction: float
    class_balance_strategy: str


@dataclass
class DeviceConfig:
    device_str: str

    @property
    def device(self) -> torch.device:
        if self.device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.device_str == "mps":
             if torch.backends.mps.is_available():
                 return torch.device("mps")
             else:
                 return torch.device("cpu")
        else:
            return torch.device(self.device_str)


@dataclass
class ProjectConfig:
    name: str
    seed: int
    debug_mode: bool
    fast_dev_run: bool
    paths: PathsConfig
    text_model: TextModelConfig
    device: DeviceConfig


def load_config() -> ProjectConfig:
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"

    raw_cfg = _load_yaml_config(cfg_path)

    paths = raw_cfg["paths"]
    paths_cfg = PathsConfig(
        data_root=PROJECT_ROOT / paths["data_root"],
        raw_dir=PROJECT_ROOT / paths["raw_dir"],
        processed_dir=PROJECT_ROOT / paths["processed_dir"],
        models_dir=PROJECT_ROOT / paths["models_dir"],
        cache_dir=PROJECT_ROOT / paths["cache_dir"],
        logs_dir=PROJECT_ROOT / paths["logs_dir"],
    )

    text_cfg_raw = raw_cfg["text_model"]
    text_cfg = TextModelConfig(
        pretrained_name=text_cfg_raw["pretrained_name"],
        max_seq_len=int(text_cfg_raw["max_seq_len"]),
        batch_size=int(text_cfg_raw["batch_size"]),
        num_epochs=int(text_cfg_raw["num_epochs"]),
        learning_rate=float(text_cfg_raw["learning_rate"]),
        train_subset_fraction=float(text_cfg_raw["train_subset_fraction"]),
        class_balance_strategy=str(text_cfg_raw["class_balance_strategy"]),
    )

    # Handle device configuration
    # If device is a dict (old format), extract prefer_gpu
    # If device is a string (new format), use it directly
    device_raw = raw_cfg.get("device", "cuda")
    if isinstance(device_raw, dict):
        # Backward compatibility: convert old format to new format
        prefer_gpu = device_raw.get("prefer_gpu", True)
        device_str = "cuda" if prefer_gpu else "cpu"
    else:
        # New format: device is a string
        device_str = str(device_raw)

    device_cfg = DeviceConfig(device_str=device_str)

    return ProjectConfig(
        name=str(raw_cfg["project"]["name"]),
        seed=int(raw_cfg["project"]["seed"]),
        debug_mode=bool(raw_cfg["project"].get("debug_mode", False)),
        fast_dev_run=bool(raw_cfg["project"].get("fast_dev_run", False)),
        paths=paths_cfg,
        text_model=text_cfg,
        device=device_cfg,
    )
