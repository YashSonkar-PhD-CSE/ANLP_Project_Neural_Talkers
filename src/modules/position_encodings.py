import math
from typing import Dict, Any, Optional, Type, Tuple, Union
import torch

from ..constants import PE_TYPES

class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        peType: PE_TYPES,
        **kwargs: Any,
    ):
        super().__init__()
        self.peType = peType
        self.config = kwargs
        self.embedDim = self.config.get("embedDim", 512)
        self.maxSeqLen = self.config.get("maxSeqLen", 5000)
        self.nHeads = self.config.get("nHeads", 8)
        self.headDim = self.embedDim // self.nHeads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"forward method not implemented for {self.peType} positional encoding")

class RotaryPositionalEncoding(PositionalEncoding):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__("rope", **kwargs)
        self.base: int = self.config.get("base", 10_000)

        # Precomputing frequencies
        invFreq = 1.0 / (self.base ** (torch.arange(0, self.headDim, 2).float() / self.headDim))
        pos = torch.arange(self.maxSeqLen, dtype = torch.float)

        freqs = torch.einsum("i,j->ij", pos, invFreq) # [N, headDim / 2]

        # the comment type: torch.Tensor is for the IDE to recognize cos and sin.
        self.register_buffer("cos", freqs.cos().repeat_interleave(2, dim=-1)) # type: torch.Tensor
        self.register_buffer("sin", freqs.sin().repeat_interleave(2, dim=-1)) # type: torch.Tensor
        # [N, headDim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("When using rope, call forwardRope instead of forward")

    def forwardRope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqOffset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, N, D] # D = headDim, H = numHeads, N: seqLen

        seqLen = q.size(2)
        assert q.size(-1) == self.headDim, "headDim mismatch"
        # Only for the IDE
        assert isinstance(self.cos, torch.Tensor)
        assert isinstance(self.sin, torch.Tensor)
        assert seqLen + seqOffset <= self.maxSeqLen, f"Seq len ({seqLen}) + offset ({seqOffset}) exceeds max seq len ({self.maxSeqLen})"
        cos = self.cos[seqOffset:seqOffset + seqLen].unsqueeze(0).unsqueeze(0) # [1, 1, N, D]
        sin = self.sin[seqOffset:seqOffset + seqLen].unsqueeze(0).unsqueeze(0) # [1, 1, N, D]

        def apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., ::2], x[..., 1::2] # Splitting into even/odd

            return torch.cat([
                x1 * cos[..., ::2] - x2 * sin[..., ::2],
                x2 * cos[..., ::2] + x1 * sin[..., ::2]
            ], dim=-1)

        return apply(q), apply(k)
    
class SinusoidalPositionalEncoding(PositionalEncoding):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__("sinusoidal", **kwargs)

        position = torch.arange(self.maxSeqLen + 1).unsqueeze(1) # [maxSeqLen + 1, 1]
        den = torch.exp(torch.arange(0, self.embedDim, 2) * (-math.log(10_000.0) / self.embedDim)) # [embedDim / 2]

        pe = torch.zeros(self.maxSeqLen + 1, self.embedDim)
        pe[:, 0::2] = torch.sin(position * den)
        pe[:, 1::2] = torch.cos(position * den)

        self.register_buffer("pe", pe) # type: torch.Tensor
        # [maxSeqLen, embedDim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]

        # The assert is only for the syntax checker in the IDE
        assert isinstance(self.pe, torch.Tensor)
        seqLen = x.size(-2) 
        pe = self.pe[:seqLen].unsqueeze(0) # [1, N, D]
        return x + pe.to(x.device)
    
class LearnablePositionalEncoding(PositionalEncoding):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__("learned", **kwargs)

        self.pe = torch.nn.Parameter(torch.zeros(self.maxSeqLen, self.embedDim))
        torch.nn.init.normal_(self.pe, mean = 0.0, std = 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        seqLen = x.size(-2)
        pe = self.pe[:seqLen].unsqueeze(0) # [1, N, D]
        return x + pe.to(x.device)
    
CLS_MAP: Dict[PE_TYPES, Optional[Type[PositionalEncoding]]] = {
    "rope": RotaryPositionalEncoding,
    "sinusoidal": SinusoidalPositionalEncoding,
    "learned": LearnablePositionalEncoding,
    "none": None
}
    
def getPositionEncoder(peType: PE_TYPES, **kwargs: Any) -> Union[PositionalEncoding, torch.nn.Module]:
    cls = CLS_MAP.get(peType)
    return cls(**kwargs) if cls is not None else torch.nn.Identity()
