import torch
from torch import nn, optim

try:
    import lightning.pytorch as pl  # latest
except ModuleNotFoundError:
    import pytorch_lightning as pl  # older


def rmse_loss(x, y):
    return torch.sqrt(nn.functional.mse_loss(x, y))


class LitModel(pl.LightningModule):
    def __init__(
        self,
        neural_net,
        loss_func=rmse_loss,
        optimizer_name="AdamW",
        optimizer_kwargs=dict(lr=1e-3),
        scheduler_name="ReduceLROnPlateau",
        scheduler_kwargs=dict(patience=5),
        scheduler_config=dict(monitor="val_loss", mode="min", frequency=1),
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["neural_net"])

        self.neural_net = neural_net
        self.loss_func = loss_func

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config

    def training_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch[0]), batch[1])
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch[0]), batch[1])
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss_func(self.neural_net(batch[0]), batch[1])
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            return self.neural_net(batch[0])
        else:
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
