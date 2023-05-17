import torch
from torch import nn

from nan.model_utils import create_fc_task_nets

from .probe_utils import get_embedder
from .. import LitModel


class ArithmeticOperator(nn.Module):
    def __init__(
        self,
        emb_name="encoding",
        emb_kwargs=dict(use_aux=True, nums={}, aux={}),
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
        self.embedder = get_embedder(emb_name, emb_kwargs)

        if hasattr(self.embedder, "hidden_size"):
            emb_size = self.embedder.hidden_size
        else:
            emb_size = self.embedder(torch.tensor([0])).shape[-1]

        self.ops_heads = create_fc_task_nets(
            in_size=2 * emb_size,
            num_tasks=num_ops,
            task_out_sizes=1,
            head_kwargs=ops_kwargs,
        )

    def forward(self, x):
        x = torch.cat((self.embedder(x[:, 0]), self.embedder(x[:, 1])), axis=-1)
        out = [op_head(x).flatten() for op_head in self.ops_heads]
        out = torch.vstack(out).T
        return out


class LitArithmeticOperator(LitModel):
    def __init__(self, neural_net, task_weights=1, **kwargs):
        super().__init__(neural_net, **kwargs)

        if not isinstance(task_weights, int):
            assert len(task_weights) == len(self.neural_net.ops_heads)
            task_weights = torch.tensor(task_weights)

        self.task_weights = task_weights
        self.loss_func = self.ops_loss

    def ops_loss(self, y_pred, y):
        if y.shape != y_pred.shape:
            y = y.reshape(y_pred.shape)

        task_mse = nn.functional.mse_loss(y_pred, y, reduction="none").mean(axis=0)
        weighted_mse = torch.sum(self.task_weights * task_mse)

        return weighted_mse
