import pytorch_lightning as pl
from pytorch_lightning.demos.transformer import Transformer


class LabModule(pl.LightningModule):
    """a custom PyTorch Lightning LightningModule"""

    def __init__(self):
        super().__init__()
        self.model = Transformer()

    def forward(self, x):
        pass

    def training_step(self, batch):
        pass

    def test_step(self, batch, *args):
        pass

    def validation_step(self, batch, *args):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        pass
