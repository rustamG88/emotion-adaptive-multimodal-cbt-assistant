import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple

class AttentionPooling(nn.Module):
    """Attention pooling layer to weigh time steps of audio features."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, dim)
            mask: (batch, time) optional boolean mask
        Returns:
            pooled: (batch, dim)
        """
        weights = self.attention(x).squeeze(-1) # (batch, time)
        if mask is not None:
            weights = weights.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(weights, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # (batch, dim)
        return pooled

class AudioWav2Vec2Attention(nn.Module):
    """Wav2Vec2 encoder with Attention Pooling for depression detection."""
    def __init__(self, num_labels: int = 2, freeze_encoder: bool = True):
        super().__init__()
        # Load pre-trained bundle
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feature_dim = 768 # Standard for base
        self.pooling = AttentionPooling(self.feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (batch, time) raw waveform at 16kHz
        """
        # Features from Wav2Vec2: (batch, time, dim)
        features, _ = self.encoder(x)

        # Pooled representation
        pooled = self.pooling(features)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits

def build_wav2vec_attention_model(num_labels: int = 2, device: str = "cpu") -> nn.Module:
    model = AudioWav2Vec2Attention(num_labels=num_labels)
    return model.to(device)
