import torch
from torch import nn, optim

from functools import partial

try:
    import lightning.pytorch as pl  # latest
except ModuleNotFoundError:
    import pytorch_lightning as pl  # older

from nan.encoding import encode_nums, encode_aux

from ..utils.model_utils import create_fcn


def get_embedder(emb_name="encoding", emb_args=dict(use_aux=True, nums={}, aux={})):
    if emb_name == "encoding":
        if emb_args.get("use_aux", False):
            num_encoder = partial(encode_nums, **emb_args["nums"])
            embedder = lambda x: torch.hstack((num_encoder(x), encode_aux(x, num_encoder=num_encoder, **emb_args["aux"])))
        else:
            embedder = lambda x: encode_nums(x, **emb_args["nums"])
    
    return embedder


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


class LitReconstructor(pl.LightningModule):
    def __init__(
        self,
        neural_net,
        optimizer_name="AdamW",
        optimizer_kwargs=dict(lr=1e-3),
        scheduler_name="ReduceLROnPlateau",
        scheduler_kwargs=dict(patience=5),
        scheduler_config=dict(monitor="val_loss", mode="min", frequency=1),
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["neural_net"])

        self.neural_net = neural_net
        self.loss_func = nn.functional.mse_loss

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config

    def training_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch), batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch), batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch), batch)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.neural_net(batch)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer_name)(
            self.parameters(),
            **self.optimizer_kwargs,
        )

        scheduler = getattr(optim.lr_scheduler, self.scheduler_name)(
            optimizer,
            **self.scheduler_kwargs,
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(scheduler=scheduler, **self.scheduler_config),
        )
