import torch
from typing import Any, Literal, Tuple, List, Dict
from dataclasses import dataclass
import os
import logging
from tqdm import tqdm

from src.constants import DATA_SPLITS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class EvaluationDataItem:
    texts: Dict[str, str]
    tokenIds: Dict[str, torch.Tensor]
    id: int

class EvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset for evaluation that returns paired samples across two languages.
    Only supports 'valid' or 'test' splits.
    Assumes files are paired by filename across languages.
    """
    def __init__(
        self,
        languages: Tuple[str, str] = ("en", "la"),
        dataRoot: str = "./data/",
        split: Literal["valid", "test"] = "valid",
        tokenizer: torch.nn.Module = torch.nn.Identity(),
        name: str = "",
    ):
        super().__init__()
        assert split in ["valid", "test"], "EvaluationDataset only supports 'valid' or 'test' splits"
        self.name = name
        self.tokenizer = tokenizer
        self.languages = languages
        self.dataPath = os.path.join(dataRoot, split)
        self.langToId = {lang: i for i, lang in enumerate(languages)}
        self.data: List[EvaluationDataItem] = []

        # Load paired files
        lang_paths = {lang: os.path.join(self.dataPath, lang) for lang in languages}
        file_sets = [set(os.listdir(lang_paths[lang])) for lang in languages]
        common_files = sorted(set.intersection(*file_sets))

        logger.info(f"Found {len(common_files)} paired files for split '{split}'")

        for idx, file in enumerate(tqdm(common_files, desc="Loading paired samples", leave=False)):
            texts = {}
            tokenIds = {}
            for lang in languages:
                file_path = os.path.join(lang_paths[lang], file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    tokenized = self.tokenizer(text)
                    texts[lang] = text
                    tokenIds[lang] = tokenized["input_ids"].squeeze(0)
            self.data.append(EvaluationDataItem(texts=texts, tokenIds=tokenIds, id=idx))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> EvaluationDataItem:
        return self.data[idx]

    def getLanguages(self) -> Tuple[str, str]:
        return self.languages

    def getLangId(self, lang: str) -> int:
        return self.langToId[lang]

    def collateFn(self, batch: List[EvaluationDataItem], padTokenIdx: int = 0) -> Dict[str, Any]:
        lang1, lang2 = self.languages
        tokens_lang1 = [item.tokenIds[lang1] for item in batch]
        tokens_lang2 = [item.tokenIds[lang2] for item in batch]

        padded_lang1 = torch.nn.utils.rnn.pad_sequence(tokens_lang1, batch_first=True, padding_value=padTokenIdx)
        padded_lang2 = torch.nn.utils.rnn.pad_sequence(tokens_lang2, batch_first=True, padding_value=padTokenIdx)

        return {
            "tokens": {
                lang1: padded_lang1,
                lang2: padded_lang2
            },
            "ids": torch.LongTensor([item.id for item in batch])
        }
