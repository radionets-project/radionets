from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING

import torch
from lightning import LightningModule

from radionets.evaluation.utils import apply_symmetry, get_ifft

if TYPE_CHECKING:
    from radionets.io.eval_config import EvaluationMethodsConfig


class TrainModule(LightningModule):
    def __init__(
        self,
        train_config: dict,
        train_length: int | None = None,
        eval_methods: EvaluationMethodsConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_config = train_config
        self.model = train_config["model"]["arch_name"]()
        self.optimizer = train_config["training"]["optimizer"]["optimizer"]
        self.loss_fn = train_config["training"]["loss"]["loss_func"]()
        self.train_length = train_length
        self.num_epochs = train_config["training"]["num_epochs"]
        self.batch_size = train_config["dataloader"]["batch_size"]

        self.eval_methods = eval_methods

        if self.eval_methods:
            for field in self.eval_methods:
                if hasattr(field[1], "met_cls"):
                    setattr(self, field[0], field[1].met_cls)

    def forward(self, inputs):
        """Main forward feed call to the model."""
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = self._extract_inputs_targets(batch)

        logits = self(inputs)["pred"]
        loss = self.loss_fn(logits, targets)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = self._extract_inputs_targets(batch)

        logits = self(inputs)["pred"]
        loss = self.loss_fn(logits, targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = self._extract_inputs_targets(batch)
        preds = self(inputs)["pred"]

        if targets is not None:
            loss = self.loss_fn(preds, targets)
            self.log("test_loss", loss, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        inputs, targets = self._extract_inputs_targets(batch)
        preds = self(inputs)["pred"]

        if targets is None:
            return preds

        preds_ifft = get_ifft(apply_symmetry(preds))
        targets_ifft = get_ifft(apply_symmetry(targets))

        preds_ifft = preds_ifft.reshape(-1, *preds_ifft.shape[-2:])
        targets_ifft = targets_ifft.reshape(-1, *targets_ifft.shape[-2:])

        if self.eval_methods:
            # Fields are tuples w/ format (name, value)
            for field in self.eval_methods:
                if hasattr(field[1], "met_cls"):
                    getattr(self, field[0]).update(preds_ifft, targets_ifft)

        return torch.stack((preds, targets), dim=1)

    def _extract_inputs_targets(self, batch):
        if isinstance(batch, dict):
            inputs = batch["inputs"]
            targets = batch.get("target", None)
        elif isinstance(batch, list | tuple):
            if len(batch) >= 2 and hasattr(batch[1], "__array__"):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch[0], None
        else:
            inputs, targets = batch, None

        return inputs, targets

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.train_config["training"]["optimizer"]["lr"],
        )

        sched_config = self.train_config["training"]["lr_scheduling"]
        if sched_config:
            sched_sig_set = set(signature(sched_config["scheduler"]).parameters.keys())

            if "epochs" not in sched_config:
                sched_config["epochs"] = self.num_epochs

            if "steps_per_epoch" not in sched_config:
                sched_config["steps_per_epoch"] = (
                    int(self.train_length) // self.batch_size
                )

            sched_config_set = set(sched_config)
            sched_keys = sched_config_set.intersection(sched_sig_set)
            sched_kwargs = dict(zip(sched_keys, map(sched_config.get, sched_keys)))

            scheduler = sched_config["scheduler"](optimizer, **sched_kwargs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_config["monitor"],
                    "interval": sched_config["interval"],
                    "frequency": sched_config["frequency"],
                    "strict": sched_config["strict"],
                },
            }

        return {"optimizer": optimizer}
