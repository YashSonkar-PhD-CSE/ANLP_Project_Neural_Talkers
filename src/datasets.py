import torch
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass
import os
import random

from .constants import DATA_SPLITS

@dataclass
class DataItem:
    text: str
    tokenIds: torch.Tensor
    id: int
    language: str

class BaseDataset(torch.utils.data.Dataset):
    """
    A base dataset class that will serve as parent class for all other dataset classes.
    This class will define basic methods that are common for all other classes.
    Expected dataset folder structure:
    - data:
        - lang1:
            - 1.txt
            - 2.txt
            ...
        - lang2:
            - 1.txt
            - 2.txt
        ...

    Note: The txt files need not be paired
    
    Args:
        - languages: Tuple(str, str): A tuple containing language codes for required languages. Supported languages will be mentioned in README
        - dataRoot: str: Path to the data folder
        - tokenizer: torch.nn.Module: Tokenizer class to use for tokenizing text in both languages. Ideally, one tokenizer should be used for both languages but since
        a module is being accepted, it can be implemented to use different tokenizers for each language.
        - name: str: Name of the dataset class (used in log statements for debugging purposes).
    """
    def __init__(
        self,
        languages: Tuple[str, str] = ("en", "fr"),
        dataRoot: str = "../data/",
        split: DATA_SPLITS = "valid",
        tokenizer: torch.nn.Module = torch.nn.Identity(),
        name: str = "",
    ):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.dataPath = os.path.join(dataRoot, split)
        self.languages = languages
        self.langData: Dict[str, List[DataItem]] = {lang: [] for lang in languages}
        self.data: List[DataItem] = []
        self.langToId = {lang: i for i, lang in enumerate(languages)}
        for lang in languages:
            langPath = os.path.join(self.dataPath, lang)
            for file in os.listdir(langPath):
                if not file.endswith('.txt'):
                    continue
                with open(os.path.join(langPath, file), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    tokenized = self.tokenizer(text)
                    item = DataItem(
                        text = text,
                        tokenIds = tokenized['input_ids'].squeeze(0),
                        id = len(self.langData[lang]),
                        language = lang
                    )
                    self.langData[lang].append(item)
                    self.data.append(item)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> DataItem:
        return self.data[idx]
    
    def getLengthPerLanguage(self) -> Tuple[int, int]:
        return len(self.langData[self.languages[0]]), len(self.langData[self.languages[1]])
    
    def getSampleByLanguage(self, lang: str, idx: int) -> DataItem:
        assert lang in self.languages, f"language ({lang}) not in dataset (available langauges: {self.languages})"
        return self.langData[lang][idx]
    
    def getRandomSample(self, lang: str) -> DataItem:
        assert lang in self.languages, f"language ({lang}) not in dataset (available langauges: {self.languages})"
        return random.choice(self.langData[lang])
    
    def getLanguageBatches(self, lang: str, batchSize: int) -> List[List[DataItem]]:
        assert lang in self.languages, f"language ({lang}) not in dataset (available langauges: {self.languages})"
        items = self.langData[lang]
        return [items[i: i + batchSize] for i in range(0, len(items), batchSize)]

    def getLangId(self, lang: str) -> int:
        return self.langToId[lang]
    
    def getLanguages(self) -> Tuple[str, str]:
        return self.languages
    
    def collateFn(self, batch: List[DataItem], padTokenIdx: int = 0) -> Dict[str, torch.Tensor]:
        tokenIds = [item.tokenIds for item in batch]
        padded = torch.nn.utils.rnn.pad_sequence(
            tokenIds, 
            batch_first=True, 
            padding_value = padTokenIdx
        )
        langIds = torch.LongTensor([self.langToId[item.language] for item in batch])
        return {"tokens": padded, "languages": langIds}
