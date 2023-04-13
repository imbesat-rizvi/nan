from torch import nn

from .probe_utils import get_embedder
from .. import LitModel
from ..utils.model_utils import create_fcn


LitReconstructor = LitModel


class Reconstructor(nn.Module):
    def __init__(
        self,
        emb_name="encoding",
        emb_args=dict(use_aux=True, nums={}, aux={}),
        num_layers=2,
        hidden_size=64,
        dropout=0.2,
        non_linearity="ReLU",
    ):

        super().__init__()
        self.embedder = get_embedder(emb_name, emb_args)

        if hasattr(self.embedder, "hidden_size"):
            emb_size = self.embedder.hidden_size
        else:
            emb_size = self.embedder(0).shape[-1]

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
