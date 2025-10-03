import torch
import random
import argparse
from typing import Literal, Optional

from .tokenizers import TokenizerModule
from .constants import ACTIVATIONS, LANGUAGES, TOKENIZERS

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
    

def maskInput(
    tokens: torch.Tensor,
    padToken: int = 0,
    maskFraction: float = 0.2
) -> torch.Tensor:
    # Drop {maskFraction}% of non-pad tokens
    mask = (tokens != padToken)
    masked = tokens.clone()
    for i in range(tokens.size(0)):
        valid = mask[i].nonzero(as_tuple = True)[0]
        dropCount = int(maskFraction * len(valid))
        if dropCount > 0:
            dropIndices = random.sample(valid.tolist(), dropCount)
            masked[i, dropIndices] = padToken
    return masked

def makeTrainParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type = str,
        default = './data/'
    )
    parser.add_argument(
        "--train-phase",
        choices = ["autoencoder", "backtranslation"]
    )
    parser.add_argument(
        "--num-epochs",
        type = int,
        default = 50,
    )
    parser.add_argument(
        "--src-language",
        choices = LANGUAGES.__args__
    )
    parser.add_argument(
        "--tgt-language",
        choices = LANGUAGES.__args__
    )
    parser.add_argument(
        "--model-config",
        type = str,
        default = "base"
    )
    parser.add_argument(
        "--checkpoint-path",
        type = str,
        default = "./checkpoints"
    )
    parser.add_argument(
        "--log",
        action = "store_true"
    )
    parser.add_argument(
        '--tokenizer',
        choices = TOKENIZERS.__args__,
        default = "bert"
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 32
    )
    parser.add_argument(
        "--save-interval",
        type = int,
        default = 10,
    )
    parser.add_argument(
        "--autoencoder-checkpoint", 
        type = str,
        default = None,
        help = "Path to pretrained autoencoder checkpoint"
    )
    return parser

# Typed namepsace to enable syntax highlighting and auto-completes with IDEs when using args
class TrainArgs(argparse.Namespace):
    data_root: str
    train_phase: Literal["autoencoder", "backtranslation"]
    num_epochs: int
    src_language: LANGUAGES
    tgt_language: LANGUAGES
    checkpoint_path: str
    model_config: str
    log: bool
    batch_size: int
    tokenizer: TOKENIZERS
    save_interval: int
    autoencoder_checkpoint: Optional[str]

def getTokenizer(tokenizerType: TOKENIZERS, maxLength: int = 512) -> TokenizerModule:
    tokenizerName = ""
    if tokenizerType == "bert":
        tokenizerName = "bert-multilingual"
    elif tokenizerType == "bpe":
        tokenizerName = "bpe"
    elif tokenizerType == "sentencepiece":
        tokenizerName = "sentencepiece"
    else:
        raise NotImplementedError(f"Tokenizer {tokenizerType} hasn't been implemented yet")
    return TokenizerModule(
        tokenizer_type = tokenizerName,
        max_length = maxLength
    )