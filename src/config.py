from dataclasses import dataclass, field, fields
from typing import Tuple, Literal
import torch

@dataclass
class DataConfig:
    langauges: Tuple[str, str]
    name: str
    tokenizer: torch.nn.Module
    dataRoot: str

@dataclass
class TrainConfig:
    dataConfig: DataConfig = DataConfig(
        langauges = ("en", "fr"),
        name = "En_Fr",
        tokenizer = torch.nn.Identity(),
        dataRoot = "../data/"
    )