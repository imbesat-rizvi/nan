import torch
from torch import nn

from nan.encoding import encode_nums, encode_aux

from ..utils.model_utils import create_fcn


class Reconstructor(nn.Module):
    def __init__(
        self,
        embedder=lambda x: torch.hstack((encode_nums(x), encode_aux(x))),
        num_layers=2,
        hidden_size=64,
        dropout=0.2,
        non_linearity="ReLU",
    ):

        super().__init__()
        self.embedder = embedder

        if hasattr(embedder, "hidden_size"):
            emb_size = embedder.hidden_size
        else:
            emb_size = embedder(0).shape[-1]

        self.fcn = create_fcn(
            in_size=emb_size,
            num_outputs=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            non_linearity=non_linearity,
        )

    def forward(self, x):
        return self.fcn(self.embedder(x))
