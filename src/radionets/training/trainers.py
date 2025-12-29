import torch
from lightning import LightningModule


class TrainModule(LightningModule):
    def __init__(self, train_config: dict, train_length: int = None):
        super().__init__()
        self.save_hyperparameters()

        self.train_config = train_config
        self.model = train_config["model"]["arch_name"]()
        self.optimizer = train_config["training"]["optimizer"]["optimizer"]
        self.loss_fn = train_config["training"]["loss"]["loss_func"]()
        self.train_length = train_length

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idinputs):
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

    def test_step(self, batch, batch_idx):
        inputs, targets = self._extract_inputs_targets(batch)
        preds = self(inputs)["pred"]

        if targets is not None:
            loss = self.loss_fn(preds, targets)
            self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        return preds, targets

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _ = self._extract_inputs_targets(batch)
        preds = self(inputs)["pred"]

        return preds

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.train_config["training"]["optimizer"]["lr"],
        )

        return optimizer
