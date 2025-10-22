import torch
from typing import Union, Optional

from ..config import ModelConfig
from .position_encodings import PositionalEncoding
from .encoder import EncoderLayer

class NARTextDecoder(torch.nn.Module):
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
            EncoderLayer( 
                nHeads=self.nHeads,
                embedDim=self.embedDim,
                ffMult=self.ffMult,
                dropout=self.dropout,
                activation=self.activation,
                posEncoder=self.posEncoder if self.peType == "rope" else None
            ) for _ in range(self.nLayers)
        ])

        self.outputProjection = torch.nn.Linear(self.embedDim, vocabSize)
        self.outputProjection.weight = self.tokenEmbedding.weight

        # Optional: learn to predict output length
        self.lengthPredictor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(self.embedDim, 1)
        )

    def forward(
        self,
        encoderOut: torch.Tensor,
        tgtTokens: Optional[torch.Tensor],  # Can be None during inference
        maxLen: Optional[int] = None
    ) -> torch.Tensor:
        B, S, D = encoderOut.size()

        if tgtTokens is not None:
            # Training mode: use provided target tokens
            x = self.tokenEmbedding(tgtTokens)
            if self.peType != "rope" and self.posEncoder is not None:
                x = self.posEncoder(x)
        else:
            # Inference mode: predict length and decode in parallel
            if maxLen is None:
                predictedLen = self.lengthPredictor(encoderOut.transpose(1, 2)).squeeze(-1).round().long()
                maxLen = predictedLen.max().item()
            assert maxLen is not None
            x = torch.arange(0, 1, maxLen, device=encoderOut.device).unsqueeze(0).expand(B, -1)  # [B, T]
            x = self.tokenEmbedding(x)
            if self.peType != "rope" and self.posEncoder is not None:
                x = self.posEncoder(x)

        for layer in self.layers:
            x = layer(x, encoderOut)

        return self.outputProjection(x)  # [B, T, V]
