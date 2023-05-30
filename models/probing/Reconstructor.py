import torch
from torch import nn

from nan.model_utils import create_fcn

from .probe_utils import get_embedder
from .. import LitModel


LitReconstructor = LitModel


class Reconstructor(nn.Module):
    def __init__(
        self,
        emb_name="NANEmbedder",
        emb_kwargs=dict(
            emb_net=None,
            nums_kwargs={},
            aux_kwargs={},
            use_aux="also",
            random_state=42,
        ),
        num_layers=2,
        hidden_size=64,
        dropout=0.2,
        non_linearity="ReLU",
    ):

        super().__init__()
        self.embedder = get_embedder(emb_name, emb_kwargs)

        if hasattr(self.embedder, "hidden_size"):
            emb_size = self.embedder.hidden_size
        else:
            emb_size = self.embedder(torch.tensor([0])).shape[-1]

        self.fcn = create_fcn(
            in_size=emb_size,
            num_outputs=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            non_linearity=non_linearity,
        )

    def forward(self, x):
        return self.fcn(self.embedder(x)).flatten()
