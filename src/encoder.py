import torch
from typing import Dict, Any, Optional, Union

from .utils import getActivationLayer
from .config import ModelConfig
from .constants import ACTIVATIONS
from .position_encodings import RotaryPositionalEncoding, PositionalEncoding

class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        modelConfig: ModelConfig,
        sharedPosEncoder: Union[PositionalEncoding, torch.nn.Module],
        vocabSize: int
    ):
        super().__init__()
        self.config = modelConfig
        self.nLayers = self.config.encoderConfig.nLayers
        self.nHeads = self.config.nHeads
        self.embedDim = self.config.embedDim
        self.ffMult = self.config.encoderConfig.ffMult
        self.dropout = self.config.encoderConfig.dropout
        self.activation = self.config.encoderConfig.activation
        self.peType = self.config.peType
        self.vocabSize = vocabSize

        self.layers = torch.nn.ModuleList()
        self.tokenEmbedding = torch.nn.Embedding(vocabSize, self.config.embedDim)
        self.posEncoder = sharedPosEncoder

        for _ in range(self.nLayers):
            self.layers.append(EncoderLayer(
                nHeads = self.nHeads,
                embedDim = self.embedDim,
                ffMult = self.ffMult,
                activation = self.activation,
                dropout = self.dropout,
                posEncoder = self.posEncoder if self.peType != "rope" else None,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.min() >= 0 and x.max() < self.vocabSize, f"{x.min()}, {x.max()}, {self.vocabSize}"
        x = self.tokenEmbedding(x)
        if self.peType != "rope" and self.posEncoder is not None:
            x = self.posEncoder(x)
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        nHeads: int,
        embedDim: int,
        ffMult: int,
        dropout: float,
        activation: ACTIVATIONS,
        posEncoder: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        self.nHeads = nHeads
        self.embedDim = embedDim
        self.ffMult = ffMult
        
        self.attnDropout = torch.nn.Dropout(dropout)
        self.ffnDropout = torch.nn.Dropout(dropout)

        self.posEncoder = posEncoder

        self.qkvProj = torch.nn.Linear(embedDim, 3 * embedDim)
        self.scale: float = embedDim ** -0.5 
        self.softmax = torch.nn.Softmax(dim = -1)
        self.norm1 = torch.nn.LayerNorm(embedDim)
        self.norm2 = torch.nn.LayerNorm(embedDim)
        # Calculate q, k & v matrices
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedDim, ffMult * embedDim),
            getActivationLayer(activation, embedDim = ffMult * embedDim),
            torch.nn.Linear(ffMult * embedDim, embedDim),
            getActivationLayer(activation, embedDim = embedDim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        # x: [B, N, embedDim]
        B, N, D = x.size()
        qkv: torch.Tensor = self.qkvProj(x) # [B, N, 3 * embedDim]
        q = qkv[..., :self.embedDim] # [B, N, embedDim]
        k = qkv[..., self.embedDim: 2 * self.embedDim] # [B, N, embedDim]
        v = qkv[..., 2 * self.embedDim: ] # [B, N, embedDim]

        # MHA (Apply RoPE if available)
        q = q.view(B, N, self.nHeads, D // self.nHeads).transpose(1, 2) # [B, H, N, Dh]  
        k = k.view(B, N, self.nHeads, D // self.nHeads).transpose(1, 2)
        v = v.view(B, N, self.nHeads, D // self.nHeads).transpose(1, 2)
        if self.posEncoder is not None and isinstance(self.posEncoder, RotaryPositionalEncoding):
            q, k = self.posEncoder.forwardRope(q, k)

        attnWeights = (q @ k.transpose(-1, -2)) * self.scale # [B, H, N, N]
        z: torch.Tensor = self.softmax(attnWeights) # [B, H, N, N]

        attnScores = z @ v # [B, H, N, embedDim // H]

        # Merge heads
        attnScores = attnScores.transpose(1, 2).reshape(B, N, D) # [B, N, D]

        attnScores = self.norm1(attnScores + x) # [B, N, embedDim]
        attnScores = self.attnDropout(attnScores)
        out = self.ffn(attnScores) # [B, N, embedDim]
        out = self.norm2(out + attnScores) # [B, N, embedDim]
        out = self.ffnDropout(out)
        return out