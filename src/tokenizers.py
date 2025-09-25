import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    GPT2TokenizerFast,
    AutoTokenizer
)

class TokenizerModule(nn.Module):
    def __init__(self, tokenizer_type="bert-multilingual", max_length=128):
        
        super().__init__()
        self.max_length = max_length

        if tokenizer_type == "bert-multilingual":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        elif tokenizer_type == "bpe":
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
              self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        elif tokenizer_type == "sentencepiece":
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer_type = tokenizer_type

    def forward(self, texts):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encodings
