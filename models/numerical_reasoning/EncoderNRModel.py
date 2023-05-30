# Encoder-based Numerical Reasoning (NR) Model

import torch
from torch import nn
from functools import partial

from transformers import AutoModel

from nan import NANModel
from nan.model_utils import (
    create_fcn,
    digit_polarity_loss,
    scientific_notation_loss,
    decode_digit_polarity_output,
    decode_scientific_notation_output,
)

from .. import LitModel


class EncoderNRModel(nn.Module):
    def __init__(
        self,
        net="bert-base-uncased",
        embedder="NANEmbedder",  # Embedder object in which case embedder_kwargs is ignored
        embedder_kwargs=dict(
            emb_net="fcn",
            emb_kwargs=dict(
                output_size=768,  # emb size of the upstream model e.g. BERT
                num_layers=1,
                dropout=0.2,  # immaterial for 1 layer NN
                non_linearity="ReLU",  # immaterial for 1 layer NN
            ),
            nums_kwargs=dict(
                digit_kwargs=dict(int_decimals=12, frac_decimals=7, scale=True)
            ),
            aux_kwargs={},  # default encode_aux kwargs
            use_aux=False,
            random_state=42,
        ),
        head_type="classifier",
        # output_sizes specified for classifier or scientific notation only
        # as regressor output_sizes will be 1
        # output_sizes will be the size of numbers to be predicted
        # e.g. for classifier: 1 for polarity + 12 for int_decimals + 7 for frac_decimals
        # or for scientific notation: 1 for mantissa and +12 to -7 i.e. 19 for exponent
        output_sizes=20,
        head_kwargs=dict(
            num_layers=2,
            hidden_size=64,
            dropout=0.2,
            non_linearity="ReLU",
        ),
    ):

        super().__init__()

        self.net = net
        if isinstance(net, str):
            self.net = AutoModel.from_pretrained(net)

        # assumes the model output dim to head is same as embedding dim
        in_size = self.net.embeddings.word_embeddings.embedding_dim

        if embedder:
            self.net = NANModel(
                self.net, embedder=embedder, embedder_kwargs=embedder_kwargs
            )

        assert head_type in ("classifier", "regressor", "scientific_notation")
        self.head_type = head_type

        if head_type == "regressor":
            out_size = 1
        elif head_type == "classifier":
            # number prediction, output_sizes-1 times digit probas
            # and 1 time sign proba
            out_size = (output_sizes - 1) * 10 + 1
        else:
            # scientific notation prediction, 1 for mantissa
            # output_sizes - 1 for exponent
            out_size = output_sizes

        self.head_net = create_fcn(in_size=in_size, num_outputs=out_size, **head_kwargs)

    def forward(self, **kwargs):
        out = self.net(**kwargs)
        out = self.head_net(out["pooler_output"])
        return out


class LitEncoderNRModel(LitModel):
    def __init__(
        self,
        neural_net,
        target_names="output_answer",
        int_decimals=12,
        exp_ub=12,
        **kwargs,
    ):
        super().__init__(neural_net, target_names=target_names, **kwargs)

        if self.neural_net.head_type == "regressor":
            self.y_pred_transform = lambda x: x.squeeze()
            self.loss_fn = nn.functional.mse_loss

        else:
            self.y_pred_transform = lambda x: x

            if self.neural_net.head_type == "classifier":
                self.loss_fn = partial(digit_polarity_loss, int_decimals=int_decimals)
                self.output_decoder = partial(
                    decode_digit_polarity_output, int_decimals=int_decimals
                )

            elif self.neural_net.head_type == "scientific_notation":
                self.loss_fn = partial(scientific_notation_loss, exp_ub=exp_ub)
                self.output_decoder = partial(
                    decode_scientific_notation_output, exp_ub=exp_ub
                )

        self.loss_func = self._loss

    def _loss(self, y_pred, y):
        loss = self.loss_fn(self.y_pred_transform(y_pred), y)
        return loss
