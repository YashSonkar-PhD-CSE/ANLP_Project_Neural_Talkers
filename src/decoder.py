from typing import Optional, Union
import torch

from src.utils import getActivationLayer

from .encoder import EncoderLayer
from .constants import ACTIVATIONS
from .position_encodings import PositionalEncoding, getPositionEncoder
from .config import ModelConfig

class TextDecoder(torch.nn.Module):
    def __init__(
        self,
        modelConfig: ModelConfig,
        sharedEmbedding: torch.nn.Embedding,
        sharedPosEncoder: Union[PositionalEncoding, torch.nn.Module],
        vocabSize: int
    ):
        super().__init__()
        self.config = modelConfig
        self.embedDim = modelConfig.embedDim
        self.nHeads = modelConfig.nHeads
        self.nLayers = modelConfig.decoderConfig.nLayers
        self.ffMult = modelConfig.decoderConfig.ffMult
        self.dropout = modelConfig.decoderConfig.dropout
        self.activation = modelConfig.decoderConfig.activation
        self.peType = modelConfig.peType

        self.tokenEmbedding = sharedEmbedding
        self.posEncoder = sharedPosEncoder

        self.layers = torch.nn.ModuleList([
            DecoderLayer(
                embedDim = self.embedDim,
                nHeads = self.nHeads,
                ffMult = self.ffMult,
                dropout = self.dropout,
                activation = self.activation,
                posEncoder = self.posEncoder if self.peType == "rope" else None
            ) for _ in range(self.nLayers)
        ])

        self.outputProjection = torch.nn.Linear(self.embedDim, vocabSize)
        self.outputProjection.weight = self.tokenEmbedding.weight

    def forward(
        self,
        tgtTokens: torch.Tensor,
        encoderOut: torch.Tensor
    ) -> torch.Tensor:
        # tgtTokens: [B, T] (token ids)
        x = self.tokenEmbedding(tgtTokens) # [B, T, D]
        if self.peType != "rope" and self.posEncoder is not None:
            x = self.posEncoder(x)

        for layer in self.layers:
            x = layer(x, encoderOut)
        logits = self.outputProjection(x) # [B, T, vocabSize]
        return logits
    
class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        embedDim: int,
        nHeads: int,
        ffMult: int,
        dropout: float,
        activation: ACTIVATIONS,
        posEncoder: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        self.selfAttn = EncoderLayer( # Re-using encoder layers for self-attention
            nHeads = nHeads,
            embedDim = embedDim,
            ffMult = ffMult,
            dropout = dropout,
            activation = activation,
            posEncoder = posEncoder
        )

        self.crossAttn = CrossAttentionLayer(
            embedDim = embedDim,
            nHeads = nHeads,
            dropout = dropout
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedDim, ffMult * embedDim),
            getActivationLayer(activation, embedDim = embedDim),
            torch.nn.Linear(ffMult * embedDim, embedDim),
            getActivationLayer(activation, embedDim = embedDim)
        )

        self.norm = torch.nn.LayerNorm(embedDim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoderOut: torch.Tensor
    ) -> torch.Tensor:
        x = self.selfAttn(x)
        x = self.crossAttn(x, encoderOut)
        x = self.norm(self.ffn(x) + x)
        return x
    
class CrossAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        embedDim: int,
        nHeads: int,
        dropout: float
    ):
        super().__init__()
        self.embedDim = embedDim
        self.nHeads = nHeads
        self.dropout = torch.nn.Dropout(dropout)

        self.qProj = torch.nn.Linear(embedDim, embedDim)
        self.kProj = torch.nn.Linear(embedDim, embedDim)
        self.vProj = torch.nn.Linear(embedDim, embedDim)
        self.outProj = torch.nn.Linear(embedDim, embedDim)

        self.scale: float = embedDim ** -0.5
        self.softmax = torch.nn.Softmax(dim = -1)
        self.norm = torch.nn.LayerNorm(embedDim)

    def forward(
        self,
        x: torch.Tensor,
        encoderOut: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, T, D]: decoder input
        # encoderOut: [B, S, D]: encoder output / context
        B, T, D = x.size()

        q: torch.Tensor = self.qProj(x).view(B, T, self.nHeads, D // self.nHeads).transpose(1, 2) # [B, H, T, D // H]
        k: torch.Tensor = self.kProj(encoderOut).view(B, -1, self.nHeads, D // self.nHeads).transpose(1, 2) # [B, H, S, D // H]
        v: torch.Tensor = self.kProj(encoderOut).view(B, -1, self.nHeads, D // self.nHeads).transpose(1, 2) # [B, H, S, D // H]

        # Cross-Attention
        attnScores = (q @ k.transpose(-1, -2)) * self.scale # [B, H, T, S]
        attnScores: torch.Tensor = self.softmax(attnScores) # [B, H, T, S]
        z = attnScores @ v # [B, H, T, D // H]

        # Merge heads
        z = z.transpose(1, 2).reshape(B, T, D) # [B, T, D]
        
        out = self.outProj(z)
        return self.norm(out + x)