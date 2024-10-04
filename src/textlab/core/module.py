"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import MultiheadAttention

if hasattr(MultiheadAttention, "_reset_parameters") and not hasattr(MultiheadAttention, "reset_parameters"):
    # See https://github.com/pytorch/pytorch/issues/107909
    MultiheadAttention.reset_parameters = MultiheadAttention._reset_parameters


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int, ninp: int = 200, nhead: int = 2, nhid: int = 200, nlayers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.embedding = nn.Embedding(vocab_size, ninp)
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(ninp, vocab_size)

        self.ninp = ninp
        self.vocab_size = vocab_size
        self.src_mask = None

    def forward(self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, t = inputs.shape

        # we assume target is already shifted w.r.t. inputs
        if mask is None:
            mask = torch.tril(torch.ones(t, t, device=inputs.device)) == 1
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        src = self.pos_encoder(self.embedding(inputs) * math.sqrt(self.ninp))
        target = self.pos_encoder(self.embedding(target) * math.sqrt(self.ninp))
        output = self.transformer(src, target, tgt_mask=mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.view(-1, self.vocab_size)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.max_len = max_len

        pe = self._init_pos_encoding()
        # workaround, can't use buffer, see https://github.com/pytorch/pytorch/issues/68407
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def reset_parameters(self) -> None:
        self.pe.copy_(self._init_pos_encoding())  # type: ignore[operator]

    def forward(self, x: Tensor) -> Tensor:
        x + self.pe[: x.size(0), :]  # type: ignore[index]
        return self.dropout(x)

    def _init_pos_encoding(self) -> Tensor:
        pe = torch.zeros(self.max_len, self.dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe


class LabModule(pl.LightningModule):
    def __init__(self, vocab_size: int = 33278):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx = 0):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("training-loss", loss)
        return loss

    def validation_step(self, batch, batch_idx = 0):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("val-loss", loss)

    def test_step(self, batch, batch_idx = 0):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("test-loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
