import multiprocessing
import os
from pathlib import Path

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from textlab import Config
from textlab.pipeline import LabDataset

NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class LabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int = NUMWORKERS,
        data_dir: Path = Config.DATAPATH,
        block_size: int = 35,
        download: bool = True,
        train_size: float = 0.8,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.download = download
        self.num_workers = num_workers
        self.train_size = train_size
        self.dataset = None

    def prepare_data(self) -> None:
        self.dataset = LabDataset(data_dir=self.data_dir, block_size=self.block_size, download=self.download)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            train_size = int(len(self.dataset) * self.train_size)
            test_size = len(self.dataset) - train_size
            self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.val_data

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, num_workers=self.num_workers)
