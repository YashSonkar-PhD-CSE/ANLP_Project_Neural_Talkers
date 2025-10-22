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

@dataclass
class EncoderConfig:
    nLayers: int = 6
    ffMult: int = 4
    dropout: float = 0.05
    activation: ACTIVATIONS = "swiglu"

@dataclass
class DecoderConfig:
    nLayers: int = 6
    ffMult: int = 4
    dropout: float = 0.05
    activation: ACTIVATIONS = "swiglu"

@dataclass
class ModelConfig:
    useNAR: bool = False
    embedDim: int = 512
    peType: PE_TYPES = "rope"
    maxSeqLen: int = 5000
    nHeads: int = 8
    encoderConfig: EncoderConfig = field(default_factory=EncoderConfig)
    decoderConfig: DecoderConfig = field(default_factory=DecoderConfig)
    languages: Tuple[str, str] = tuple()
    vocabSize: int = 50000
    startToken: int = 1
    padToken: int = 0


def getModelConfig(
    configName: str,
    languages: Tuple[str, str],
    vocabSize: int
) -> ModelConfig:
    return ModelConfig(
        useNAR = "nar" in configName.lower(),
        languages = languages,
        vocabSize = vocabSize,
    )
