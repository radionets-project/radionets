from inspect import signature

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
        self.num_epochs = train_config["training"]["num_epochs"]
        self.batch_size = train_config["training"]["batch_size"]

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.1, patience=10, threshold=1e-5
        # )

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
