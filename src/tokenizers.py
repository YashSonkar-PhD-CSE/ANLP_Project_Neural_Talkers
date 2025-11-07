import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    GPT2TokenizerFast,
    AutoTokenizer
)

class TokenizerModule(nn.Module):
    def __init__(self, tokenizer_type="bert-multilingual", max_length=512):
        
        super().__init__()
        self.max_length = max_length

        if tokenizer_type == "bert-multilingual":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            self.tokenizer.bos_token_id = getattr(self.tokenizer, "cls_token_id", None)
            # BERT doesn't have a BOS and EOS tokens so we use CLS as a 
            # substitute for BOS and SEP as substitute for EOS
            self.tokenizer.eos_token_id = getattr(self.tokenizer, "sep_token_id", None)
        elif tokenizer_type == "bpe":
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
              self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            #   self.tokenizer.add_special_tokens({''})

        elif tokenizer_type == "sentencepiece":
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        #self.vocab_size = self.tokenizer.vocab_size
        self.vocab_size = len(self.tokenizer)
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
        
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode a single sequence or batch of token IDs back to string(s).
        """
        # Handle both single example (1D tensor) and batch (2D tensor)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # If input is list of lists → batch decode
        if isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self, return_dict=False):
        """Return vocab size or full vocab dictionary."""
        if return_dict:
            return self.tokenizer.get_vocab()
        return len(self.tokenizer)

    def get_special_token_ids(self):
        """Return a dictionary of all relevant special token IDs."""
        t = self.tokenizer
        return {
            "pad_token_id": getattr(t, "pad_token_id", None),
            "bos_token_id": getattr(t, "bos_token_id", None),
            "eos_token_id": getattr(t, "eos_token_id", None),
            "cls_token_id": getattr(t, "cls_token_id", None),
            "sep_token_id": getattr(t, "sep_token_id", None),
            "unk_token_id": getattr(t, "unk_token_id", None),
            "mask_token_id": getattr(t, "mask_token_id", None),
        }



