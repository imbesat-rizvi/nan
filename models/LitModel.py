import torch
from torch import nn, optim

try:
    import lightning.pytorch as pl  # latest
except ModuleNotFoundError:
    import pytorch_lightning as pl  # older


class LitModel(pl.LightningModule):
    def __init__(
        self,
        neural_net,
        target_names="labels",
        loss_func=nn.functional.mse_loss,
        optimizer_name="AdamW",
        optimizer_kwargs=dict(lr=1e-3),
        scheduler_name="ReduceLROnPlateau",
        scheduler_kwargs=dict(patience=5),
        scheduler_config=dict(monitor="val_loss", mode="min", frequency=1),
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["neural_net"])

        self.neural_net = neural_net

        self.target_names = target_names
        if isinstance(target_names, str):
            self.target_names = [target_names]

        self.loss_func = loss_func
        self.output_decoder = lambda x: x  # override for processing predict_step

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs

        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config

    def forward(self, batch):
        if isinstance(batch, dict):
            return self.neural_net(**batch)
        return self.neural_net(batch)

    def split_target_from_batch(self, batch):
        target = None
        if isinstance(batch, dict):
            if len(self.target_names) == 1:
                target = batch.pop(self.target_names[0], None)
            else:
                target = {k: batch.pop(k) for k in self.target_names}
        elif isinstance(batch, list):
            batch, target = batch[0], batch[1]
        return batch, target

    def training_step(self, batch, batch_idx):
        batch, target = self.split_target_from_batch(batch)
        loss = self.loss_func(self(batch), target)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, target = self.split_target_from_batch(batch)
        loss = self.loss_func(self(batch), target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch, target = self.split_target_from_batch(batch)
        loss = self.loss_func(self(batch), target)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        batch, _ = self.split_target_from_batch(batch)
        return self.output_decoder(self(batch))

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
