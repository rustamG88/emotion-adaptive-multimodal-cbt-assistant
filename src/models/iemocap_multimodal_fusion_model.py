"""Trainable multimodal fusion model for IEMOCAP combining text, audio, and video."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import yaml

from config.config import load_config, PROJECT_ROOT
from src.models.text_distilbert import DistilBERTEmotionClassifier, DistilBERTConfig
from src.models.audio_iemocap_cnn_lstm import IEMOCAPAudioCNNLSTM, IEMOCAPAudioModelConfig
from src.models.video_iemocap_resnet import IEMOCAPVideoResNet, IEMOCAPVideoModelConfig
from src.utils.logging_utils import get_logger


@dataclass
class IEMOCAPFusionConfig:
    """Configuration for IEMOCAP multimodal fusion model."""

    # Embedding dimensions (will be inferred from models if not provided)
    text_embed_dim: int = 768  # DistilBERT hidden size
    audio_embed_dim: int = 256  # 2 * LSTM hidden size (128 * 2)
    video_embed_dim: int = 512  # ResNet18 feature dim (512 for ResNet18, 2048 for ResNet50)

    # Fusion MLP
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    num_classes: int = 9

    # Freeze settings
    freeze_text: bool = True
    freeze_audio: bool = True
    freeze_video: bool = True


class IEMOCAPMultimodalFusionModel(nn.Module):
    """Trainable multimodal fusion model for IEMOCAP emotion classification."""

    def __init__(
        self,
        text_model: DistilBERTEmotionClassifier,
        audio_model: IEMOCAPAudioCNNLSTM,
        video_model: IEMOCAPVideoResNet,
        fusion_cfg: IEMOCAPFusionConfig,
    ):
        """
        Initialize IEMOCAP multimodal fusion model.

        Args:
            text_model: Pretrained text model (DistilBERT)
            audio_model: Pretrained audio model (CNN+LSTM)
            video_model: Pretrained video model (ResNet)
            fusion_cfg: Fusion configuration
        """
        super().__init__()
        self.fusion_cfg = fusion_cfg

        # Store unimodal models
        self.text_model = text_model
        self.audio_model = audio_model
        self.video_model = video_model

        # Freeze unimodal models if requested
        if fusion_cfg.freeze_text:
            for param in self.text_model.parameters():
                param.requires_grad = False

        if fusion_cfg.freeze_audio:
            for param in self.audio_model.parameters():
                param.requires_grad = False

        if fusion_cfg.freeze_video:
            for param in self.video_model.parameters():
                param.requires_grad = False

        # Fusion MLP: concatenate embeddings -> hidden -> output
        total_embed_dim = fusion_cfg.text_embed_dim + fusion_cfg.audio_embed_dim + fusion_cfg.video_embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_embed_dim, fusion_cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_cfg.dropout),
            nn.Linear(fusion_cfg.fusion_hidden_dim, fusion_cfg.num_classes),
        )

    def extract_text_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract text embedding from DistilBERT (before classifier).

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask

        Returns:
            Text embedding tensor of shape (batch, text_embed_dim)
        """
        # Pass through transformer
        outputs = self.text_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling (same as in DistilBERTEmotionClassifier)
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (last_hidden_state * attention_mask_expanded).sum(dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1)
            pooled = sum_hidden / (sum_mask + 1e-9)
        else:
            pooled = last_hidden_state.mean(dim=1)

        # Apply dropout (same as in original model)
        pooled = self.text_model.dropout(pooled)

        return pooled

    def extract_audio_embedding(self, mfcc_features: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding from CNN+LSTM (before classifier).

        Args:
            mfcc_features: MFCC features of shape (batch, n_mfcc, time)

        Returns:
            Audio embedding tensor of shape (batch, audio_embed_dim)
        """
        # Conv block 1
        x = self.audio_model.conv1(mfcc_features)
        x = self.audio_model.bn1(x)
        x = self.audio_model.relu1(x)
        x = self.audio_model.pool1(x)

        # Conv block 2
        x = self.audio_model.conv2(x)
        x = self.audio_model.bn2(x)
        x = self.audio_model.relu2(x)
        x = self.audio_model.pool2(x)

        # Permute for LSTM
        x = x.permute(0, 2, 1)

        # BiLSTM
        outputs, (h_n, c_n) = self.audio_model.lstm(x)

        # Extract hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]

        # Concatenate
        h_cat = torch.cat([h_forward, h_backward], dim=-1)

        # Apply dropout
        h_cat = self.audio_model.dropout(h_cat)

        return h_cat

    def extract_video_embedding(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Extract video embedding from ResNet (before classifier).

        Args:
            video_frames: Video frames of shape (batch, C, H, W)

        Returns:
            Video embedding tensor of shape (batch, video_embed_dim)
        """
        # Extract features through backbone
        features = self.video_model.backbone(video_frames)
        # Squeeze spatial dimensions
        features = features.squeeze(-1).squeeze(-1)

        # Apply dropout
        features = self.video_model.dropout(features)

        return features

    def forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_mfcc: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through fusion model.

        Args:
            text_input_ids: Tokenized text input IDs
            text_attention_mask: Text attention mask
            audio_mfcc: MFCC features for audio
            video_frames: Video frames
            labels: Ground truth labels (optional)

        Returns:
            Tuple of (loss, logits). Loss is None if labels are not provided.
        """
        embeddings = []

        # Extract text embedding
        if text_input_ids is not None:
            text_emb = self.extract_text_embedding(text_input_ids, text_attention_mask)
            embeddings.append(text_emb)

        # Extract audio embedding
        if audio_mfcc is not None:
            audio_emb = self.extract_audio_embedding(audio_mfcc)
            embeddings.append(audio_emb)

        # Extract video embedding
        if video_frames is not None:
            video_emb = self.extract_video_embedding(video_frames)
            embeddings.append(video_emb)

        if not embeddings:
            raise ValueError("At least one modality must be provided")

        # Concatenate embeddings
        fused_embedding = torch.cat(embeddings, dim=-1)  # (batch, total_embed_dim)

        # Pass through fusion MLP
        logits = self.fusion_mlp(fused_embedding)  # (batch, num_classes)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def load_pretrained_models(
    text_model_path: Optional[Path] = None,
    audio_model_path: Optional[Path] = None,
    video_model_path: Optional[Path] = None,
    num_classes: int = 9,
    device: Optional[torch.device] = None,
) -> Tuple[DistilBERTEmotionClassifier, IEMOCAPAudioCNNLSTM, IEMOCAPVideoResNet]:
    """
    Load pretrained unimodal models.

    Args:
        text_model_path: Path to text model weights
        audio_model_path: Path to audio model weights
        video_model_path: Path to video model weights
        num_classes: Number of emotion classes
        device: Device to load models on

    Returns:
        Tuple of (text_model, audio_model, video_model)
    """
    logger = get_logger(__name__)
    cfg = load_config()

    if device is None:
        device = cfg.device.device

    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Load text model
    logger.info("Loading text model...")
    text_cfg = DistilBERTConfig(
        pretrained_name=cfg.text_model.pretrained_name,
        num_labels=num_classes,
    )
    text_model = DistilBERTEmotionClassifier(text_cfg).to(device)
    if text_model_path and text_model_path.exists():
        text_model.load_state_dict(torch.load(text_model_path, map_location=device))
        logger.info(f"Loaded text model weights from {text_model_path}")
    else:
        logger.warning(f"Text model path not found: {text_model_path}, using pretrained weights only")

    # Load audio model
    logger.info("Loading audio model...")
    audio_cfg_raw = raw_cfg.get("iemocap_audio", {})
    n_mfcc = audio_cfg_raw.get("n_mfcc", 40)
    audio_cfg = IEMOCAPAudioModelConfig(num_mfcc=n_mfcc, num_classes=num_classes)
    audio_model = IEMOCAPAudioCNNLSTM(audio_cfg).to(device)
    if audio_model_path and audio_model_path.exists():
        audio_model.load_state_dict(torch.load(audio_model_path, map_location=device))
        logger.info(f"Loaded audio model weights from {audio_model_path}")
    else:
        logger.warning(f"Audio model path not found: {audio_model_path}, using random weights")

    # Load video model
    logger.info("Loading video model...")
    video_cfg_raw = raw_cfg.get("iemocap_video", {})
    backbone = video_cfg_raw.get("backbone", "resnet18")
    video_cfg = IEMOCAPVideoModelConfig(
        backbone=backbone,
        num_classes=num_classes,
    )
    video_model = IEMOCAPVideoResNet(video_cfg).to(device)
    if video_model_path and video_model_path.exists():
        video_model.load_state_dict(torch.load(video_model_path, map_location=device))
        logger.info(f"Loaded video model weights from {video_model_path}")
    else:
        logger.warning(f"Video model path not found: {video_model_path}, using pretrained ResNet weights")

    return text_model, audio_model, video_model


def build_iemocap_fusion_model(
    text_model_path: Optional[Path] = None,
    audio_model_path: Optional[Path] = None,
    video_model_path: Optional[Path] = None,
    num_classes: Optional[int] = None,
    fusion_hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    freeze_text: Optional[bool] = None,
    freeze_audio: Optional[bool] = None,
    freeze_video: Optional[bool] = None,
    device: Optional[torch.device] = None,
) -> IEMOCAPMultimodalFusionModel:
    """
    Build IEMOCAP multimodal fusion model.

    Args:
        text_model_path: Path to text model weights
        audio_model_path: Path to audio model weights
        video_model_path: Path to video model weights
        num_classes: Number of emotion classes (if None, reads from config)
        fusion_hidden_dim: Hidden dimension for fusion MLP (if None, reads from config)
        dropout: Dropout rate (if None, reads from config)
        freeze_text: Whether to freeze text model (if None, reads from config)
        freeze_audio: Whether to freeze audio model (if None, reads from config)
        freeze_video: Whether to freeze video model (if None, reads from config)
        device: Device to load models on

    Returns:
        Initialized fusion model
    """
    cfg = load_config()
    if device is None:
        device = cfg.device.device

    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    fusion_cfg_raw = raw_cfg.get("iemocap_fusion", {})

    # Get config values
    if num_classes is None:
        num_classes = fusion_cfg_raw.get("num_classes", 9)
    if fusion_hidden_dim is None:
        fusion_hidden_dim = fusion_cfg_raw.get("fusion_hidden_dim", 256)
    if dropout is None:
        dropout = fusion_cfg_raw.get("dropout", 0.3)
    if freeze_text is None:
        freeze_text = fusion_cfg_raw.get("freeze_text", True)
    if freeze_audio is None:
        freeze_audio = fusion_cfg_raw.get("freeze_audio", True)
    if freeze_video is None:
        freeze_video = fusion_cfg_raw.get("freeze_video", True)

    # Get model paths from config if not provided
    if text_model_path is None:
        text_path_str = fusion_cfg_raw.get("text_model_path", "")
        text_model_path = PROJECT_ROOT / text_path_str if text_path_str else None
    if audio_model_path is None:
        audio_path_str = fusion_cfg_raw.get("audio_model_path", "")
        audio_model_path = PROJECT_ROOT / audio_path_str if audio_path_str else None
    if video_model_path is None:
        video_path_str = fusion_cfg_raw.get("video_model_path", "")
        video_model_path = PROJECT_ROOT / video_path_str if video_path_str else None

    # Load pretrained models
    text_model, audio_model, video_model = load_pretrained_models(
        text_model_path=text_model_path,
        audio_model_path=audio_model_path,
        video_model_path=video_model_path,
        num_classes=num_classes,
        device=device,
    )

    # Determine embedding dimensions
    # Text: DistilBERT hidden size
    text_embed_dim = text_model.config.hidden_size

    # Audio: 2 * LSTM hidden size
    audio_embed_dim = 2 * audio_model.cfg.lstm_hidden

    # Video: ResNet feature dimension
    if video_model.cfg.backbone == "resnet18":
        video_embed_dim = 512
    elif video_model.cfg.backbone == "resnet50":
        video_embed_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {video_model.cfg.backbone}")

    # Create fusion config
    fusion_cfg = IEMOCAPFusionConfig(
        text_embed_dim=text_embed_dim,
        audio_embed_dim=audio_embed_dim,
        video_embed_dim=video_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        freeze_text=freeze_text,
        freeze_audio=freeze_audio,
        freeze_video=freeze_video,
    )

    # Create fusion model
    fusion_model = IEMOCAPMultimodalFusionModel(
        text_model=text_model,
        audio_model=audio_model,
        video_model=video_model,
        fusion_cfg=fusion_cfg,
    )

    return fusion_model.to(device)


__all__ = [
    "IEMOCAPFusionConfig",
    "IEMOCAPMultimodalFusionModel",
    "load_pretrained_models",
    "build_iemocap_fusion_model",
]

