import torch
from .constants import ACTIVATIONS

class SwiGLU(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear1 = torch.nn.Linear(dimension,dimension)
        self.linear2 = torch.nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear2(x)

        return swiglu
    
def getActivationLayer(actType: ACTIVATIONS, **kwargs) -> torch.nn.Module:
    if actType == "relu":
        return torch.nn.ReLU()
    elif actType == "swiglu":
        return SwiGLU(dimension = kwargs.get("embedDim"))
    elif actType == "gelu":
        return torch.nn.GELU()
    elif actType == "none":
        return torch.nn.Identity()
    else:
        raise NotImplementedError(f"activation type {actType} not implemented yet")
    

# Positional encodings
# TODO: RoPE, Sinusoidal and Learnable