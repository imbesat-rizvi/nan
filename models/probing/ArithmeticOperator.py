import torch
from torch import nn

from nan.model_utils import create_fc_task_nets, digit_polarity_loss

from .probe_utils import get_embedder
from .. import LitModel


class ArithmeticOperator(nn.Module):
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
        ops=("add", "sub", "abs_diff", "mul", "div", "max", "argmax"),
        model_type="classifier",
        # output_sizes specified for classifier only as regressor output_sizes will be 1
        # output_sizes will be the size of numbers to be predicted
        # e.g. 1 for polarity + 12 for int_decimals + 7 for frac_decimals
        output_sizes=20,
        ops_kwargs=dict(
            num_layers=2,
            hidden_size=64,
            dropout=0.2,
            non_linearity="ReLU",
        ),
    ):
        r"""Arithmetic Operator for 2 numbers"""

        super().__init__()

        assert model_type in ("classifier", "regressor")
        self.model_type = model_type

        self.embedder = get_embedder(emb_name, emb_kwargs)

        if hasattr(self.embedder, "hidden_size"):
            emb_size = self.embedder.hidden_size
        else:
            emb_size = self.embedder(torch.tensor([0.0])).shape[-1]

        self.ops = ops
        self.ops_heads = nn.ModuleDict()
        for op in ops:
            if model_type == "regressor" or op == "argmax":
                task_out_sizes = 1
            else:
                # number prediction, output_sizes-1 times digit probas
                # and 1 time sign proba
                task_out_sizes = (output_sizes - 1) * 10 + 1

            self.ops_heads[op] = create_fc_task_nets(
                in_size=2 * emb_size,
                num_tasks=1,
                task_out_sizes=task_out_sizes,
                head_kwargs=ops_kwargs,
            )

    def forward(self, x):
        x = torch.cat((self.embedder(x[:, 0]), self.embedder(x[:, 1])), axis=-1)
        out = {op: op_head(x) for op, op_head in self.ops_heads.items()}
        return out


class LitArithmeticOperator(LitModel):
    def __init__(self, neural_net, task_weights=1, **kwargs):
        super().__init__(neural_net, **kwargs)

        if not isinstance(task_weights, int):
            assert len(task_weights) == len(self.neural_net.ops_heads)
        self.task_weights = task_weights
        self.loss_func = self.ops_loss

    def ops_loss(self, y_pred, y):
        loss = 0
        y = torch.atleast_2d(y)

        for i, op in enumerate(self.neural_net.ops):

            if op == "argmax":
                output = y_pred[op].squeeze()
                loss_func = nn.functional.binary_cross_entropy_with_logits
            elif self.neural_net.model_type == "regressor":
                output = y_pred[op].squeeze()
                loss_func = nn.functional.mse_loss
            else:
                output = y_pred[op]
                loss_func = digit_polarity_loss

            weight = 1
            if isinstance(self.task_weights, dict):
                weight = self.task_weights.get(op, 1)
            loss += weight * loss_func(output, y[:, i])

        return loss
