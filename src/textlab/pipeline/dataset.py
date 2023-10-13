"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
from torch import Tensor

from torch.utils.data import Dataset

from textlab import Config


class LabDataset(Dataset):
    """Mini version of WikiText2."""

    def __init__(self, data_dir: Path = Config.DATAPATH, block_size: int = 35, download: bool = True) -> None:
        super().__init__()
        self.path = Path(data_dir) / "wikitext-2.txt"
        if download:
            self.download(self.path)
        self.data, self.dictionary = tokenize(self.path)
        self.block_size = block_size

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return inputs, target

    @staticmethod
    def download(destination: Path) -> None:
        os.makedirs(destination.parent, exist_ok=True)
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        if os.path.exists(destination):
            return
        with open(destination, "w") as f:
            f.write(requests.get(url).text)


class Dictionary:
    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def tokenize(path: Path) -> Tuple[Tensor, Dictionary]:
    dictionary = Dictionary()

    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, encoding="utf8") as f:
        for line in f:
            words = line.split() + ["<eos>"]
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, encoding="utf8") as f:
        idss: List[Tensor] = []
        for line in f:
            words = line.split() + ["<eos>"]
            ids: List[int] = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))

    return torch.cat(idss), dictionary
