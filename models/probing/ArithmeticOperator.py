import torch
from torch import nn

from .probe_utils import get_embedder
from .. import LitModel
from ..utils.model_utils import create_fc_task_nets


class ArithmeticOperator(nn.Module):
    def __init__(
        self,
        emb_name="encoding",
        emb_args=dict(use_aux=True, nums={}, aux={}),
        num_ops=1,
        ops_kwargs=dict(
            num_layers=2,
            hidden_size=64,
            dropout=0.2,
            non_linearity="ReLU",
        ),
    ):
        r"""Arithmetic Operator for 2 numbers"""

        super().__init__()
        self.embedder = get_embedder(emb_name, emb_args)

        if hasattr(self.embedder, "hidden_size"):
            emb_size = self.embedder.hidden_size
        else:
            emb_size = self.embedder(0).shape[-1]

        self.ops_heads = create_fc_task_nets(
            in_size=2 * emb_size,
            num_tasks=num_ops,
            task_out_sizes=1,
            head_kwargs=ops_kwargs,
        )

    def forward(self, x):
        x = torch.cat((self.embedder(x[:, 0]), self.embedder(x[:, 1])), axis=-1)
        x = [op_head(x).flatten() for op_head in self.ops_heads]
        return x


class LitArithmeticOperator(LitModel):
    def __init__(self, task_weights=1, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(task_weights, int):
            assert len(task_weights) == len(self.neural_net.ops_heads)
            task_weights = torch.tensor(task_weights)

        self.task_weights = task_weights
        self.loss_func = self.ops_loss

    def ops_loss(self, x, y):
        out = self.neural_net(x)
        out = torch.vstack(out).T

        if y.shape != out.shape:
            y = y.reshape(out.shape)

        task_mse = nn.functional.mse_loss(out, y, reduction="none").mean(axis=0)
        task_rmse = torch.sqrt(task_mse + 1e-8)  # 1e-8 as eps for nan gradient
        weighted_rmse = torch.sum(self.task_weights * task_rmse)

        return weighted_rmse
