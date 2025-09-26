from dataclasses import dataclass, field, fields
from typing import Tuple, Literal
import torch

from .constants import ACTIVATIONS, PE_TYPES

@dataclass
class DataConfig:
    languages: Tuple[str, str]
    name: str
    tokenizer: torch.nn.Module
    dataRoot: str

# @dataclass
# class TrainConfig:
#     dataConfig: DataConfig = DataConfig(
#         languages = ("en", "fr"),
#         name = "En_Fr",
#         tokenizer = torch.nn.Identity(),
#         dataRoot = "../data/"
#     )

@dataclass
class EncoderConfig:
    nLayers: int = 4
    ffMult: int = 4
    dropout: float = 0.05
    activation: ACTIVATIONS = "relu"

@dataclass
class DecoderConfig:
    nLayers: int = 4
    ffMult: int = 4
    dropout: float = 0.05
    activation: ACTIVATIONS = "relu"

@dataclass
class ModelConfig:
    embedDim: int = 512
    peType: PE_TYPES = "none"
    maxSeqLen: int = 5000
    nHeads: int = 8
    encoderConfig: EncoderConfig = field(default_factory=EncoderConfig)
    decoderConfig: DecoderConfig = field(default_factory=DecoderConfig)
    languages: Tuple[str, str] = field(default_factory=Tuple)
    vocabSize: int = 50000