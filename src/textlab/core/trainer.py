# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.profilers import Profiler, PyTorchProfiler

from textlab import Config


class LabTrainer(pl.Trainer):
    def __init__(
        self,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = None,
        callbacks: Optional[List] = None,
        plugins: Optional[List] = None,
        set_seed: bool = True,
        **kwargs: Dict[str, Any]
    ) -> None:
        # SET SEED
        if set_seed:
            seed_everything(Config.GLOBALSEED, workers=True)
        if callbacks and not isinstance(callbacks, list):
            callbacks = [callbacks]
        if not callbacks:
            callbacks = []
        super().__init__(
            logger=logger or TensorBoardLogger(Config.LOGSPATH, name="tensorboard"),
            profiler=profiler or PyTorchProfiler(dirpath=Config.TORCHPROFILERPATH, filename="profiler"),
            callbacks=callbacks + [ModelCheckpoint(dirpath=Config.CHKPTSPATH, filename="model")],
            plugins=plugins,
            **kwargs
        )

    def persist_predictions(self, predictions_dir: Optional[Union[str, Path]] = Config.PREDSPATH) -> None:
        self.test(ckpt_path="best", datamodule=self.datamodule)
        predictions = self.predict(self.model, self.datamodule.val_dataloader())
        torch.save(predictions, predictions_dir)
