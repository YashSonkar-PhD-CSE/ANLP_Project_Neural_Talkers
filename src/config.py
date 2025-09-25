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
    nHeads: int = 8
    embedDim: int = 512
    ffMult: int = 4
    dropout: float = 0.05
    activation: ACTIVATIONS = "relu"
    peType: PE_TYPES = "none"