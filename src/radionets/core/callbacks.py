import warnings
from abc import ABC
from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)
from lightning.pytorch.callbacks import Callback as LightningCallback
from lightning.pytorch.loggers import CometLogger, MLFlowLogger
from matplotlib.colors import PowerNorm
from pydantic import BaseModel

from radionets.evaluation.metrics import IntensityRatio, SourceAreaRatio
from radionets.evaluation.utils import apply_symmetry, get_ifft
from radionets.plotting.utils import get_vmin_vmax, set_cbar

__all__ = [
    "Callbacks",
    "CometCallback",
    "LogAdditionalParamsCallback",
    "MLFlowCallback",
    "MLFlowCodeCarbonCallback",
    "PlottingCallbackABC",
]


class Callbacks:
    @classmethod
    def get_callbacks(cls, train_config: BaseModel) -> list:
        default_callback = RichProgressBar()
        callbacks = [default_callback]

        if train_config.callbacks.model_checkpoint:
            model_checkpoint = ModelCheckpoint(
                **train_config.callbacks.model_checkpoint.model_dump()
            )
            callbacks.append(model_checkpoint)

        if train_config.callbacks.batch_size_finder:
            batch_size_finder = BatchSizeFinder(
                **train_config.callbacks.batch_size_finder.model_dump()
            )
            callbacks.append(batch_size_finder)

        if train_config.callbacks.early_stopping:
            early_stopping = EarlyStopping(
                **train_config.callbacks.early_stopping.model_dump()
            )
            callbacks.append(early_stopping)

        if train_config.callbacks.lr_monitor:
            lr_monitor = LearningRateMonitor(
                **train_config.callbacks.lr_monitor.model_dump()
            )
            callbacks.append(lr_monitor)

        if train_config.callbacks.device_stats_monitor:
            callbacks.append(DeviceStatsMonitor())

        if train_config.callbacks.timer:
            timer = Timer(**train_config.callbacks.timer.model_dump())
            callbacks.append(timer)

        if train_config.logging.comet_ml:
            matplotlib.use("Agg")
            callbacks.append(CometCallback(train_config))

        if train_config.logging.mlflow:
            matplotlib.use("Agg")
            callbacks.append(MLFlowCallback(train_config))

            if train_config.logging.codecarbon:
                callbacks.append(MLFlowCodeCarbonCallback(train_config))

            callbacks.append(LogAdditionalParamsCallback(train_config))

        return callbacks


class PlottingCallbackABC(ABC, LightningCallback):
    def __init__(self, train_config, *args, **kwargs):
        super().__init__()
        self.train_config = train_config
        self.amp_phase = train_config.dataloader.amp_phase
        self.scale = train_config.logging.scale

        self.cached_batch = None

        data_types = ["Amplitude", "Phase"] if self.amp_phase else ["Real", "Imaginary"]
        results = [" Prediction", " Ground Truth"]
        self.pred_plot_titles = [t + r for r in results for t in data_types]

    def plot_val_pred(self, predictions, targets, current_epoch: int):
        self.fig, self.axs = plt.subplots(
            2, 2, figsize=(12, 8.5), layout="constrained", sharex=True, sharey=True
        )
        self.axs = self.axs.flatten()

        limits_0 = get_vmin_vmax(targets[0, 0])  # Limits for amp/real
        limits_1 = get_vmin_vmax(targets[0, 1])  # Limits for phase/imaginary

        im0 = self.axs[0].imshow(
            predictions[0, 0],
            cmap="radionets.PuOr",
            vmin=-limits_0,
            vmax=limits_0,
            origin="lower",
        )
        im1 = self.axs[1].imshow(
            predictions[0, 1],
            cmap="radionets.PuOr",
            vmin=-limits_1,
            vmax=limits_1,
            origin="lower",
        )
        im2 = self.axs[2].imshow(
            targets[0, 0],
            cmap="radionets.PuOr",
            vmin=-limits_0,
            vmax=limits_0,
            origin="lower",
        )
        im3 = self.axs[3].imshow(
            targets[0, 1],
            cmap="radionets.PuOr",
            vmin=-limits_1,
            vmax=limits_1,
            origin="lower",
        )

        for ax, im, title in zip(
            self.axs,
            [im0, im1, im2, im3],
            self.pred_plot_titles,
        ):
            set_cbar(self.fig, ax, im, title=title, phase="Phase" in title)

        self.axs[0].set(ylabel="Frequels")
        self.axs[2].set(xlabel="Frequels", ylabel="Frequels")
        self.axs[3].set(xlabel="Frequels")

    def plot_val_fft(self, predictions, targets, current_epoch):
        ifft_pred = get_ifft(
            predictions,
            amp_phase=self.amp_phase,
            scale=self.scale,
        )
        ifft_truth = get_ifft(targets, amp_phase=self.amp_phase, scale=self.scale)

        self.fig, self.axs = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

        im0 = self.axs[0].imshow(
            ifft_pred,
            norm=PowerNorm(0.25, vmax=ifft_truth.max()),
            cmap="inferno",
            origin="lower",
        )
        im1 = self.axs[1].imshow(
            ifft_truth,
            norm=PowerNorm(0.25),
            cmap="inferno",
            origin="lower",
        )

        limits = get_vmin_vmax(ifft_pred - ifft_truth)
        im2 = self.axs[2].imshow(
            ifft_pred - ifft_truth,
            cmap="radionets.PuOr",
            vmin=-limits,
            vmax=limits,
            origin="lower",
        )

        for ax, im, title in zip(
            self.axs,
            [im0, im1, im2],
            ["Prediction", "Truth", "Difference"],
        ):
            set_cbar(self.fig, ax, im, title="FFT " + title)

        self.axs[0].set(
            ylabel="Pixels",
            xlabel="Pixels",
        )
        self.axs[1].set_xlabel("Pixels")
        self.axs[2].set_xlabel("Pixels")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        """Log predictions at validation epoch end."""

        if self.cached_batch is None:
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))

            # cache only one sample
            self.cached_batch = (
                batch[0][0][None, ...].cpu(),
                batch[1][0][None, ...].cpu(),
            )

        if (trainer.current_epoch + 1) % self.train_config.logging.plot_n_epochs == 0:
            batch = (
                self.cached_batch[0].to(pl_module.device),
                self.cached_batch[1].to(pl_module.device),
            )

            results = pl_module.predict_step(batch, batch_idx=0).cpu()
            predictions = results[:, 0].cpu()
            targets = results[:, 1].cpu()

            # check if images are half or full
            if predictions.shape[-2] != predictions.shape[-1]:
                predictions = apply_symmetry(predictions)
                targets = apply_symmetry(targets)

            self.plot_val_pred(
                predictions,
                targets,
                current_epoch=trainer.current_epoch,
            )

            self.plot_val_fft(
                predictions,
                targets,
                current_epoch=trainer.current_epoch,
            )


class CometCallback(PlottingCallbackABC):
    def __init__(self, train_config, *args, **kwargs):
        super().__init__(train_config, *args, **kwargs)
        self.experiment = None

    def plot_val_pred(self, predictions, targets, current_epoch: int) -> None:
        super().plot_val_pred(predictions, targets, current_epoch)

        self.experiment.log_figure(
            figure=self.fig,
            figure_name=f"fourier_pred_{current_epoch:0>4}",
        )

        plt.close(self.fig)

    def plot_val_fft(self, predictions, targets, current_epoch: int) -> None:
        super().plot_val_fft(predictions, targets, current_epoch)

        self.experiment.log_figure(
            figure=self.fig,
            figure_name=f"fft_pred_{current_epoch:0>4}",
        )
        plt.close(self.fig)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        """Log predictions at validation epoch end."""
        if self.experiment is None:
            try:
                self.experiment = next(
                    logger.experiment
                    for logger in trainer.loggers
                    if isinstance(logger, CometLogger)
                )
            except StopIteration as e:
                raise ValueError(
                    f"Could not find a CometLogger instance in {trainer.loggers}."
                ) from e

        super().on_validation_epoch_end(trainer, pl_module)


class MLFlowCallback(PlottingCallbackABC):
    def __init__(self, train_config, *args, **kwargs):
        super().__init__(train_config, *args, **kwargs)

        self.experiment = None

    def plot_val_pred(self, predictions, targets, current_epoch: int) -> None:
        super().plot_val_pred(predictions, targets, current_epoch)

        artifact_file = f"fourier_pred_{current_epoch:0>4}.png"

        self.experiment.log_figure(
            figure=self.fig,
            artifact_file=artifact_file,
            run_id=self.logger._run_id,
        )

        plt.close(self.fig)

    def plot_val_fft(self, predictions, targets, current_epoch: int) -> None:
        super().plot_val_fft(predictions, targets, current_epoch)

        artifact_file = f"fft_pred_{current_epoch:0>4}.png"

        self.experiment.log_figure(
            figure=self.fig,
            artifact_file=artifact_file,
            run_id=self.logger._run_id,
        )
        plt.close(self.fig)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        """Log predictions at validation epoch end."""
        if self.experiment is None:
            try:
                self.logger = next(
                    logger
                    for logger in trainer.loggers
                    if isinstance(logger, MLFlowLogger)
                )
                self.experiment = self.logger.experiment

                self.base_dir = (
                    self.train_config.paths.model_path / f"mlflow/{self.logger._run_id}"
                )
                if trainer.is_global_zero:
                    self.base_dir.mkdir(parents=True, exist_ok=True)

            except StopIteration as e:
                raise ValueError(
                    f"Could not find a MLFlowLogger instance in {trainer.loggers}."
                ) from e

        super().on_validation_epoch_end(trainer, pl_module)


class MLFlowCodeCarbonCallback(LightningCallback):
    def __init__(self, train_config, *args, **kwargs):
        self.train_config = train_config

        self.experiment = None

    def on_fit_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)
            self.num_samples = trainer.datamodule.train_length
            self.num_samples += trainer.datamodule.valid_length

        try:
            self._log_metrics()
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"{e}. No emissions were logged.", stacklevel=2)

        self._log_params()

    def on_test_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)
            self.num_samples = trainer.datamodule.test_length

        try:
            self._log_metrics()
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"{e}. No emissions were logged.", stacklevel=2)

        self._log_params()

    def on_predict_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)
            self.num_samples = trainer.datamodule.predict_length

        try:
            self._log_metrics()
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"{e}. No emissions were logged.", stacklevel=2)

        self._log_params()

    def _set_up_experiment(self, trainer):
        try:
            self.logger = next(
                logger for logger in trainer.loggers if isinstance(logger, MLFlowLogger)
            )
            trainer.carbontracker.tracker.stop()
            self.experiment = self.logger.experiment
            self.task = trainer.radionets_task

        except StopIteration as e:
            raise ValueError(
                f"Could not find a MLFlowLogger instance in {trainer.loggers}."
            ) from e

    def _log_metrics(self):
        emission_file = Path(
            self.train_config.logging.codecarbon.output_dir + "/emissions.csv"
        )
        emission_data = pd.read_csv(emission_file).to_dict()

        eval_res = dict(
            running_time_total=emission_data["duration"][0],
            running_time=emission_data["duration"][0] / self.num_samples,
            power_draw_total=emission_data["energy_consumed"][0] * 3.6e6,
            power_draw=emission_data["energy_consumed"][0] * 3.6e6 / self.num_samples,
        )

        for key, val in eval_res.items():
            self.experiment.log_metric(
                key=key,
                value=val,
                run_id=self.logger._run_id,
            )

        self.architecture = emission_data["gpu_model"][0]

        # Remove file after logging all important metrics to mlflow.
        # This prevents codecarbon from creating 'emissions.csv_%d.bak'
        # files in the save directory
        if emission_file.is_file():
            emission_file.unlink()

    def _log_params(self):
        dataset = self.train_config.paths.data_path.name
        dataset += (
            "_amp_phase" if self.train_config.dataloader.amp_phase else "_real_imag"
        )

        model = "Radionets"
        model += "_" + str(self.train_config.model.arch_name().__class__.__name__)
        model += "_" + str(self.train_config.training.optimizer.optimizer.__name__)

        if self.train_config.training.lr_scheduling:
            model += "_" + str(
                self.train_config.training.lr_scheduling.scheduler.__name__
            )

        params_dict = dict(
            model=model,
            dataset=dataset,
            task=self.task,
            architecture=self.architecture,
        )
        for key, val in params_dict.items():
            self.experiment.log_param(
                key=key,
                value=val,
                run_id=self.logger._run_id,
            )


class LogAdditionalParamsCallback(LightningCallback):
    def __init__(self, train_config, *args, **kwargs):
        self.train_config = train_config
        self.amp_phase = train_config.dataloader.amp_phase

        self.experiment = None

        self.source_area_ratio = SourceAreaRatio()
        self.intensity_ratio = IntensityRatio()

    def on_fit_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)

        self._log_metrics(
            dataloader=trainer.datamodule.val_dataloader(),
            pl_module=pl_module,
        )

    def on_test_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)

        self._log_metrics(
            dataloader=trainer.datamodule.test_dataloader(),
            pl_module=pl_module,
        )

    def on_predict_end(self, trainer, pl_module):
        if self.experiment is None:
            self._set_up_experiment(trainer)

        self._log_metrics(
            dataloader=trainer.datamodule.predict_dataloader(),
            pl_module=pl_module,
        )

    def _set_up_experiment(self, trainer):
        try:
            self.logger = next(
                logger for logger in trainer.loggers if isinstance(logger, MLFlowLogger)
            )
            self.experiment = self.logger.experiment

        except StopIteration as e:
            raise ValueError(
                f"Could not find a MLFlowLogger instance in {trainer.loggers}."
            ) from e

    def _log_metrics(self, dataloader, pl_module):
        from radionets.evaluation.utils import _method_factory
        from radionets.io.eval_config import EvaluationMethodsConfig

        eval_methods = EvaluationMethodsConfig(
            save_images=False,
            viewing_angle=False,
            dynamic_range=False,
            intensity=True,
            area=dict(mode="pixel"),
            mean_diff=False,
        )
        _method_factory(eval_methods)

        for batch in dataloader:
            pl_module.predict_step(
                batch,
                batch_idx=0,
                eval_methods=eval_methods,
            ).detach().cpu()

        metrics = {}
        for field in eval_methods:
            if hasattr(field[1], "met_cls"):
                metrics[field[0]] = field[1].met_cls.compute()

        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )

        additional_metrics = dict(
            num_trainable_parameters=trainable_params,
            mean_area_ratio=np.abs(
                1.0 - np.mean(metrics["area"]["source_area"].numpy())
            ),
            mean_integrated_flux=np.abs(
                1.0 - np.mean(metrics["intensity"]["integrated_flux"].numpy())
            ),
            mean_peak_flux=np.abs(
                1.0 - np.mean(metrics["intensity"]["peak_flux"].numpy())
            ),
        )

        for key, val in additional_metrics.items():
            self.experiment.log_metric(
                key=key,
                value=val,
                run_id=self.logger._run_id,
            )
