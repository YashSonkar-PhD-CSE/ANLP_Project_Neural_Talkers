from typing import Literal

ACTIVATIONS = Literal["relu", "swiglu", "gelu", "none"]

PE_TYPES = Literal["rope", "learned", "sinusoidal", "none"]

FORWARD_MODES = Literal["reconstruct", "translate"]

DATA_SPLITS = Literal["train", "valid", "test"]

LANGUAGES = Literal["en", "la", "hi"]

TOKENIZERS = Literal["bert", "bpe", "sentencepiece"]

EVALUATION_TYPES = Literal["translation", "embedding"]