from typing import Literal

ACTIVATIONS = Literal["relu", "swiglu", "gelu", "none"]

PE_TYPES = Literal["rope", "learned", "sinusoidal", "none"]