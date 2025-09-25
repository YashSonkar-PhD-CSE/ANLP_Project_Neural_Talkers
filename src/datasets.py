import torch
from typing import Tuple, List, Dict
from dataclasses import dataclass
import os

@dataclass
class DataItem:
    text: str
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
        tokenizer: torch.nn.Module = torch.nn.Identity(),
        name: str = "",
    ):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.languages = languages
        self.langData: Dict[str, List[DataItem]] = {languages[0]: list(), languages[1]: list()}
        for lang in languages:
            for file in os.listdir(os.path.join(dataRoot, lang)):
                if not file.endswith('.txt'):
                    continue
                with open(os.path.join(dataRoot, lang, file), "r", encoding = "utf-8") as f:
                    self.langData[lang].append(DataItem(
                        text = self.tokenizer(f.read().strip()),
                        id = len(self.langData[lang]),
                        language = lang
                    ))
    
    def __len__(self) -> int:
        return sum(self.getLengthPerLangauge())
    
    def __getitem__(self, idx: int) -> DataItem:
        return self.langData[self.languages[idx % 2]][idx]
    
    def getLengthPerLangauge(self) -> Tuple[int, int]:
        return len(self.langData[self.languages[0]]), len(self.langData[self.languages[1]])
    
    def getSampleByLanguage(self, lang: str, idx: int) -> DataItem:
        assert lang in self.languages, f"language ({lang}) not in dataset (available langauges: {self.languages})"
        return self.langData[lang][idx]