"""Tests for src/radionets/core/callbacks.py"""

from pathlib import Path
from unittest import mock

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)

from radionets.core.callbacks import (
    Callbacks,
    CometCallback,
    LogAdditionalParamsCallback,
    MLFlowCallback,
    MLFlowCodeCarbonCallback,
    PlottingCallbackABC,
)

try:
    from lightning.pytorch.loggers import CometLogger
except ImportError:
    CometLogger = None  # ty:ignore[invalid-assignment]

try:
    from lightning.pytorch.loggers import MLFlowLogger
except ImportError:
    MLFlowLogger = None  # ty:ignore[invalid-assignment]

skip_no_comet = pytest.mark.skipif(CometLogger is None, reason="comet_ml not installed")
skip_no_mlflow = pytest.mark.skipif(MLFlowLogger is None, reason="mlflow not installed")


def _make_train_config(**overrides):
    config = mock.MagicMock()

    config.callbacks.model_checkpoint = overrides.get("model_checkpoint")
    config.callbacks.batch_size_finder = overrides.get("batch_size_finder")
    config.callbacks.early_stopping = overrides.get("early_stopping")
    config.callbacks.lr_monitor = overrides.get("lr_monitor")
    config.callbacks.device_stats_monitor = overrides.get("device_stats_monitor")
    config.callbacks.timer = overrides.get("timer")

    config.logging.comet_ml = overrides.get("comet_ml")
    config.logging.mlflow = overrides.get("mlflow")
    config.logging.codecarbon = overrides.get("codecarbon", False)
    config.logging.scale = overrides.get("scale", 1.0)
    config.logging.plot_n_epochs = overrides.get("plot_n_epochs", 1)

    config.dataloader.amp_phase = overrides.get("amp_phase", True)

    config.paths = mock.MagicMock()
    config.paths.model_path = Path("/tmp/test_model")

    return config


@pytest.fixture
def mock_trainer():
    trainer = mock.MagicMock(spec=L.Trainer)
    trainer.current_epoch = 0

    val_dataloader = mock.MagicMock()
    fake_pred = torch.randn(10, 2, 32, 32)
    fake_target = torch.randn(10, 2, 32, 32)

    val_dataloader.__iter__ = mock.MagicMock(
        return_value=iter([(fake_pred, fake_target)])
    )

    trainer.datamodule = mock.MagicMock()
    trainer.datamodule.val_dataloader = mock.MagicMock(return_value=val_dataloader)

    trainer.loggers = []

    return trainer


@pytest.fixture
def mock_pl_module():
    pl_module = mock.MagicMock()
    pl_module.device = torch.device("cpu")
    pl_module.predict_step = mock.MagicMock(return_value=torch.randn(1, 2, 32, 32))

    return pl_module


@pytest.fixture
def plotting_callback():
    train_config = _make_train_config()

    return PlottingCallbackABC(train_config)


class TestGetCallbacks:
    """Tests for Callbacks.get_callbacks()."""

    def test_returns_list(self):
        """get_callbacks should return a list."""
        config = _make_train_config()
        result = Callbacks.get_callbacks(config)

        assert isinstance(result, list)

    def test_always_returns_rich_progress_bar(self):
        """Should always include RichProgressBar."""
        config = _make_train_config()
        callbacks = Callbacks.get_callbacks(config)
        rich_callbacks = [c for c in callbacks if isinstance(c, RichProgressBar)]

        assert len(rich_callbacks) == 1

    def test_model_checkpoint_added(self):
        checkpoint_cfg = mock.MagicMock()
        checkpoint_cfg.model_dump.return_value = {
            "dirpath": "/tmp/checkpoints",
            "save_top_k": 1,
            "monitor": "val_loss",
        }

        config = _make_train_config(model_checkpoint=checkpoint_cfg)

        callbacks = Callbacks.get_callbacks(config)
        checkpoint_callbacks = [c for c in callbacks if isinstance(c, ModelCheckpoint)]

        assert len(checkpoint_callbacks) == 1

    def test_batch_size_finder_added(self):
        bs_cfg = mock.MagicMock()
        bs_cfg.model_dump.return_value = {}

        config = _make_train_config(batch_size_finder=bs_cfg)

        callbacks = Callbacks.get_callbacks(config)
        bs_callbacks = [c for c in callbacks if isinstance(c, BatchSizeFinder)]

        assert len(bs_callbacks) == 1

    def test_early_stopping_added(self):
        es_cfg = mock.MagicMock()
        es_cfg.model_dump.return_value = {"monitor": "val_loss", "patience": 3}

        config = _make_train_config(early_stopping=es_cfg)

        callbacks = Callbacks.get_callbacks(config)
        es_callbacks = [c for c in callbacks if isinstance(c, EarlyStopping)]

        assert len(es_callbacks) == 1

    def test_lr_monitor_added(self):
        lr_cfg = mock.MagicMock()
        lr_cfg.model_dump.return_value = {"logging_interval": "epoch"}

        config = _make_train_config(lr_monitor=lr_cfg)

        callbacks = Callbacks.get_callbacks(config)
        lr_callbacks = [c for c in callbacks if isinstance(c, LearningRateMonitor)]

        assert len(lr_callbacks) == 1

    def test_device_stats_monitor_added(self):
        config = _make_train_config(device_stats_monitor=True)

        callbacks = Callbacks.get_callbacks(config)
        ds_callbacks = [c for c in callbacks if isinstance(c, DeviceStatsMonitor)]

        assert len(ds_callbacks) == 1

    def test_timer_added(self):
        timer_cfg = mock.MagicMock()
        timer_cfg.model_dump.return_value = {
            "duration": {"days": 0, "hours": 1, "minutes": 0, "seconds": 0},
            "interval": "epoch",
            "verbose": False,
        }

        config = _make_train_config(timer=timer_cfg)

        callbacks = Callbacks.get_callbacks(config)
        timer_callbacks = [c for c in callbacks if isinstance(c, Timer)]

        assert len(timer_callbacks) == 1

    def test_comet_callback_added(self):
        comet_cfg = mock.MagicMock()
        config = _make_train_config(comet_ml=comet_cfg)

        with mock.patch(
            "radionets.core.callbacks.CometCallback",
            spec=True,
        ) as mock_cb_cls:
            mock_cb_cls.return_value = mock.MagicMock()
            Callbacks.get_callbacks(config)

            mock_cb_cls.assert_called_once_with(config)

    def test_mlflow_callback_added(self):
        mlflow_cfg = mock.MagicMock()
        config = _make_train_config(mlflow=mlflow_cfg)

        with mock.patch(
            "radionets.core.callbacks.MLFlowCallback",
            spec=True,
        ) as mock_cb_cls:
            mock_cb_cls.return_value = mock.MagicMock()
            Callbacks.get_callbacks(config)

            mock_cb_cls.assert_called_once_with(config)

    def test_mlflow_codecarbon_callback_added(self):
        mlflow_cfg = mock.MagicMock()
        config = _make_train_config(mlflow=mlflow_cfg, codecarbon=True)

        with (
            mock.patch("radionets.core.callbacks.MLFlowCallback"),
            mock.patch("radionets.core.callbacks.MLFlowCodeCarbonCallback") as mock_cb,
        ):
            Callbacks.get_callbacks(config)

            mock_cb.assert_called_once_with(config)

    def test_mlflow_log_additional_params_added(self):
        mlflow_cfg = mock.MagicMock()
        config = _make_train_config(mlflow=mlflow_cfg)

        with (
            mock.patch("radionets.core.callbacks.MLFlowCallback"),
            mock.patch(
                "radionets.core.callbacks.LogAdditionalParamsCallback"
            ) as mock_cb,
        ):
            Callbacks.get_callbacks(config)

            mock_cb.assert_called_once_with(config)


class TestPlottingCallbackABC:
    """Tests for PlottingCallbackABC."""

    def test_init(self):
        config = _make_train_config(amp_phase=True, scale=1.5)
        cb = PlottingCallbackABC(config)

        assert cb.train_config == config

    def test_pred_plot_titles_amp_phase(self):
        config = _make_train_config(amp_phase=True)
        cb = PlottingCallbackABC(config)

        assert len(cb.pred_plot_titles) == 4
        assert "Amplitude Prediction" in cb.pred_plot_titles
        assert "Phase Ground Truth" in cb.pred_plot_titles

    def test_pred_plot_titles_real_imag(self):
        config = _make_train_config(amp_phase=False)
        cb = PlottingCallbackABC(config)

        assert "Real Prediction" in cb.pred_plot_titles
        assert "Imaginary Ground Truth" in cb.pred_plot_titles

    def test_plot_val_pred_creates_figure(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_pred(predictions, targets, current_epoch=0)

        assert plotting_callback.fig is not None
        assert plotting_callback.axs is not None
        assert len(plotting_callback.axs) == 4

    def test_plot_val_pred_creates_subplots(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_pred(predictions, targets, current_epoch=0)

        axs = plotting_callback.axs
        for ax in axs:
            assert len(ax.images) > 0

    def test_plot_val_pred_sets_labels(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_pred(predictions, targets, current_epoch=0)

        assert plotting_callback.axs[0].get_ylabel() == "Frequels"
        assert plotting_callback.axs[2].get_ylabel() == "Frequels"
        assert plotting_callback.axs[2].get_xlabel() == "Frequels"
        assert plotting_callback.axs[3].get_xlabel() == "Frequels"

    def test_plot_val_pred_vmin_vmax_symmetry(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_pred(predictions, targets, current_epoch=0)

        for ax in plotting_callback.axs.flatten():
            norm = ax.images[0].norm
            assert norm.vmin == -norm.vmax

    def test_plot_val_fft_creates_figure(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_fft(predictions, targets, current_epoch=0)

        assert plotting_callback.fig is not None
        assert len(plotting_callback.axs) == 3

    def test_plot_val_fft_three_subplots(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_fft(predictions, targets, current_epoch=0)

        # Each subplot should have an image
        for ax in plotting_callback.axs:
            assert len(ax.images) > 0

    def test_plot_val_fft_labels(self, plotting_callback):
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_fft(predictions, targets, current_epoch=0)

        axs = plotting_callback.axs
        assert axs[0].get_ylabel() == "Pixels"
        assert axs[0].get_xlabel() == "Pixels"
        assert axs[1].get_xlabel() == "Pixels"
        assert axs[2].get_xlabel() == "Pixels"

    def test_plot_val_fft_uses_inferno_cmap(self, plotting_callback):
        """First two FFT subplots should use inferno colormap."""
        predictions = torch.randn(1, 2, 32, 32)
        targets = torch.randn(1, 2, 32, 32)

        plotting_callback.plot_val_fft(predictions, targets, current_epoch=0)

        assert plotting_callback.axs[0].images[0].cmap.name == "inferno"
        assert plotting_callback.axs[1].images[0].cmap.name == "inferno"
        assert plotting_callback.axs[2].images[0].cmap.name == "radionets.PuOr"

    def test_on_validation_epoch_end_caches_first_batch(
        self, mock_trainer, mock_pl_module
    ):
        config = _make_train_config(plot_n_epochs=1)
        cb = PlottingCallbackABC(config)
        assert cb.cached_batch is None

        class DummyImage:
            cmap = mock.MagicMock()
            cmap.colorbar_extend = "both"
            cmap.name = "radionets.PuOr"
            vmin = 1.0
            vmax = -1.0
            norm = mock.MagicMock()
            norm.vmin = 1.0
            norm.vmax = -1.0
            norm._scale = "linear"
            get_clim = mock.MagicMock(return_value=(-1.0, 1.0))
            callbacks = mock.MagicMock()
            get_array = mock.MagicMock(return_value=None)

        im = DummyImage()
        with (
            mock.patch("matplotlib.pyplot.Axes.imshow", return_value=im),
            mock.patch("matplotlib.figure.Figure.colorbar"),
            mock.patch(
                "radionets.core.callbacks.get_ifft",
                return_value=torch.randn(1, 2, 32, 32),
            ),
        ):
            cb.on_validation_epoch_end(mock_trainer, mock_pl_module)

        assert cb.cached_batch is not None

        pred_batch, target_batch = cb.cached_batch

        assert len(cb.cached_batch) == 2
        assert pred_batch.shape == (1, 2, 32, 32)
        assert target_batch.shape == (1, 2, 32, 32)
        assert pred_batch.device.type == "cpu"
        assert target_batch.device.type == "cpu"

    def test_on_validation_epoch_end_plots(self, mock_trainer, mock_pl_module):
        config = _make_train_config(plot_n_epochs=1)
        cb = PlottingCallbackABC(config)

        class DummyImage:
            cmap = mock.MagicMock()
            cmap.colorbar_extend = "both"
            cmap.name = "radionets.PuOr"
            vmin = 1.0
            vmax = -1.0
            norm = mock.MagicMock()
            norm.vmin = 1.0
            norm.vmax = -1.0
            norm._scale = "linear"
            get_clim = mock.MagicMock(return_value=(-1.0, 1.0))
            callbacks = mock.MagicMock()
            get_array = mock.MagicMock(return_value=None)

        mock_axs = np.empty(4, dtype=object)
        for i in range(4):
            mock_axs[i] = mock.MagicMock()

        mock_fig = mock.MagicMock()
        im = DummyImage()

        with (
            mock.patch("matplotlib.pyplot.Axes.imshow", return_value=im),
            mock.patch("radionets.core.callbacks.set_cbar"),
            mock.patch(
                "radionets.core.callbacks.get_ifft",
                return_value=torch.randn(1, 2, 32, 32),
            ),
            mock.patch("radionets.core.callbacks.plt.subplots") as mock_subplots,
        ):
            mock_subplots.return_value = (mock_fig, mock_axs)
            mock_trainer.current_epoch = 0

            cb.cached_batch = (torch.randn(1, 2, 32, 32), torch.randn(1, 2, 32, 32))
            cb.on_validation_epoch_end(mock_trainer, mock_pl_module)

            assert mock_subplots.called


@skip_no_comet
class TestCometCallback:
    def test_init_sets_experiment_none(self):
        config = _make_train_config(comet_ml=mock.MagicMock())
        cb = CometCallback(config)

        assert cb.experiment is None

    def test_on_validation_epoch_end_finds_comet_logger(self, mock_trainer):
        config = _make_train_config(comet_ml=mock.MagicMock())
        cb = CometCallback(config)

        mock_logger = mock.MagicMock(spec=CometLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_trainer.loggers = [mock_logger]

        with mock.patch.object(PlottingCallbackABC, "on_validation_epoch_end"):
            cb.on_validation_epoch_end(mock_trainer, mock.MagicMock())

        assert cb.experiment is not None

    def test_on_validation_epoch_end_raises_without_comet_logger(self, mock_trainer):
        config = _make_train_config(comet_ml=mock.MagicMock())
        cb = CometCallback(config)
        mock_trainer.loggers = []

        # Mock parent to prevent plotting logic
        with (
            mock.patch.object(PlottingCallbackABC, "on_validation_epoch_end"),
            pytest.raises(ValueError, match="Could not find a CometLogger"),
        ):
            cb.on_validation_epoch_end(mock_trainer, mock.MagicMock())

    def test_log_figure_called_on_plot_val_pred(self):
        config = _make_train_config()
        cb = CometCallback(config)
        cb.experiment = mock.MagicMock()

        fig = mock.MagicMock()
        cb.fig = fig

        cb.plot_val_pred(
            torch.randn(1, 2, 32, 32),
            torch.randn(1, 2, 32, 32),
            current_epoch=42,
        )

        cb.experiment.log_figure.assert_called_once()
        call_kwargs = cb.experiment.log_figure.call_args

        assert call_kwargs[1]["figure_name"] == "fourier_pred_0042"


@skip_no_mlflow
class TestMLFlowCallback:
    def test_init_sets_experiment_none(self):
        config = _make_train_config(mlflow=mock.MagicMock())
        cb = MLFlowCallback(config)

        assert cb.experiment is None

    def test_on_validation_epoch_end_finds_mlflow_logger(self, mock_trainer):
        config = _make_train_config(mlflow=mock.MagicMock())
        cb = MLFlowCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run_42"
        mock_trainer.loggers = [mock_logger]

        config.paths.model_path.mkdir(parents=True, exist_ok=True)

        with (
            mock.patch.object(PlottingCallbackABC, "on_validation_epoch_end"),
            mock.patch.object(Path, "mkdir") as mock_mkdir,
        ):
            mock_mkdir.return_value = None
            cb.on_validation_epoch_end(mock_trainer, mock.MagicMock())

        assert cb.experiment is not None
        assert cb.logger is not None
        assert hasattr(cb, "base_dir")
        assert "test_run_42" in str(cb.base_dir)

    def test_on_validation_epoch_end_raises_without_mlflow_logger(self, mock_trainer):
        config = _make_train_config(mlflow=mock.MagicMock())
        cb = MLFlowCallback(config)
        mock_trainer.loggers = []

        with (
            mock.patch.object(PlottingCallbackABC, "on_validation_epoch_end"),
            pytest.raises(ValueError, match="Could not find a MLFlowLogger"),
        ):
            cb.on_validation_epoch_end(mock_trainer, mock.MagicMock())

    def test_log_figure_called_on_plot_val_pred(self):
        config = _make_train_config(mlflow=mock.MagicMock())
        cb = MLFlowCallback(config)
        cb.experiment = mock.MagicMock()
        cb.logger = mock.MagicMock()
        cb.logger._run_id = "test_run_42"
        fig = mock.MagicMock()
        cb.fig = fig

        cb.plot_val_pred(
            torch.randn(1, 2, 32, 32),
            torch.randn(1, 2, 32, 32),
            current_epoch=42,
        )

        cb.experiment.log_figure.assert_called_once()
        call_kwargs = cb.experiment.log_figure.call_args
        assert call_kwargs[1]["artifact_file"] == "fourier_pred_0042.png"


@skip_no_mlflow
class TestMLFlowCodeCarbonCallback:
    def test_init_sets_experiment_none(self):
        config = _make_train_config()
        cb = MLFlowCodeCarbonCallback(config)

        assert cb.experiment is None
        assert cb.train_config is config

    def test_on_fit_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = MLFlowCodeCarbonCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)

        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run_42"

        mock_trainer.loggers = [mock_logger]

        mock_trainer.datamodule.train_length = 100
        mock_trainer.datamodule.valid_length = 50

        mock_trainer.radionets_task = "training"

        tracker = mock.MagicMock()
        mock_trainer.carbontracker = tracker

        with (
            mock.patch.object(cb, "_log_metrics") as mock_log,
            mock.patch.object(cb, "_log_params"),
        ):
            cb.on_fit_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_test_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = MLFlowCodeCarbonCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)

        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run_42"

        mock_trainer.loggers = [mock_logger]

        mock_trainer.datamodule.test_length = 200
        mock_trainer.radionets_task = "testing"

        tracker = mock.MagicMock()
        mock_trainer.carbontracker = tracker

        with (
            mock.patch.object(cb, "_log_metrics") as mock_log,
            mock.patch.object(cb, "_log_params"),
        ):
            cb.on_test_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_predict_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = MLFlowCodeCarbonCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_trainer.loggers = [mock_logger]
        mock_trainer.datamodule.predict_length = 300
        mock_trainer.radionets_task = "inference"

        tracker = mock.MagicMock()
        mock_trainer.carbontracker = tracker

        with (
            mock.patch.object(cb, "_log_metrics") as mock_log,
            mock.patch.object(cb, "_log_params"),
        ):
            cb.on_predict_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_fit_end_stops_carbontracker(self, mock_trainer):
        """on_fit_end should stop the carbon tracker."""
        codecarbon_cfg = mock.MagicMock()
        mlflow_cfg = mock.MagicMock()
        config = _make_train_config(mlflow=mlflow_cfg, codecarbon=codecarbon_cfg)
        cb = MLFlowCodeCarbonCallback(config)

        mock_trainer.datamodule.train_length = 100
        mock_trainer.datamodule.valid_length = 50
        mock_trainer.radionets_task = "training"

        # Create a mock for _set_up_experiment that tracks calls and stops tracker
        setup_called = []
        tracker_mock = mock.MagicMock()

        def tracked(trainer):
            setup_called.append(True)
            cb.experiment = mock.MagicMock()
            cb.logger = mock.MagicMock()
            cb.logger._run_id = "test_run"
            cb.num_samples = trainer.datamodule.train_length
            cb.architecture = "test_gpu"
            tracker_mock.stop()

        with (
            mock.patch.object(cb, "_set_up_experiment", side_effect=tracked),
            mock.patch.object(cb, "_log_metrics"),
            mock.patch.object(cb, "_log_params"),
        ):
            cb.on_fit_end(mock_trainer, mock.MagicMock())

        assert len(setup_called) == 1
        tracker_mock.stop.assert_called_once()

    def test_on_fit_end_raises_without_mlflow_logger(self, mock_trainer):
        config = _make_train_config()
        cb = MLFlowCodeCarbonCallback(config)
        mock_trainer.loggers = []

        with pytest.raises(ValueError, match="Could not find a MLFlowLogger"):
            cb.on_fit_end(mock_trainer, mock.MagicMock())


class TestLogAdditionalParamsCallback:
    def test_init_stores_amp_phase(self):
        config = _make_train_config(amp_phase=True)
        cb = LogAdditionalParamsCallback(config)
        assert cb.amp_phase is True

    def test_init_creates_metrics(self):
        config = _make_train_config()
        cb = LogAdditionalParamsCallback(config)

        assert cb.experiment is None
        assert hasattr(cb, "source_area_ratio")
        assert hasattr(cb, "intensity_ratio")

    def test_on_fit_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = LogAdditionalParamsCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_trainer.loggers = [mock_logger]

        val_dl = mock.MagicMock()
        val_dl.__iter__ = mock.MagicMock(return_value=iter([]))
        mock_trainer.datamodule.val_dataloader = mock.MagicMock(return_value=val_dl)

        with mock.patch.object(cb, "_log_metrics") as mock_log:
            cb.on_fit_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_test_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = LogAdditionalParamsCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_trainer.loggers = [mock_logger]

        test_dl = mock.MagicMock()
        test_dl.__iter__ = mock.MagicMock(return_value=iter([]))
        mock_trainer.datamodule.test_dataloader = mock.MagicMock(return_value=test_dl)

        with mock.patch.object(cb, "_log_metrics") as mock_log:
            cb.on_test_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_predict_end_calls_set_up_and_log(self, mock_trainer):
        config = _make_train_config()
        cb = LogAdditionalParamsCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_trainer.loggers = [mock_logger]

        pred_dl = mock.MagicMock()
        pred_dl.__iter__ = mock.MagicMock(return_value=iter([]))
        mock_trainer.datamodule.predict_dataloader = mock.MagicMock(
            return_value=pred_dl
        )

        with mock.patch.object(cb, "_log_metrics") as mock_log:
            cb.on_predict_end(mock_trainer, mock.MagicMock())

        mock_log.assert_called_once()

    def test_on_fit_end_raises_without_mlflow_logger(self, mock_trainer):
        config = _make_train_config()
        cb = LogAdditionalParamsCallback(config)
        mock_trainer.loggers = []

        with pytest.raises(ValueError, match="Could not find a MLFlowLogger"):
            cb.on_fit_end(mock_trainer, mock.MagicMock())

    def test_log_metrics_calls_dataloader_iteration(self, mock_trainer, mock_pl_module):
        config = _make_train_config(amp_phase=True)
        cb = LogAdditionalParamsCallback(config)

        mock_logger = mock.MagicMock(spec=MLFlowLogger)
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_trainer.loggers = [mock_logger]

        cb.logger = mock_logger
        cb.experiment = mock_logger.experiment

        pred_tensor = torch.randn(1, 2, 32, 32)
        target_tensor = torch.randn(1, 2, 32, 32)

        dl = mock.MagicMock()
        dl.__iter__ = mock.MagicMock(return_value=iter([(pred_tensor, target_tensor)]))
        dl.__len__ = mock.MagicMock(return_value=1)

        # Create pl_module with known trainable params
        param1 = torch.nn.Parameter(torch.randn(10, 5, requires_grad=True))
        param2 = torch.nn.Parameter(torch.randn(5, 3, requires_grad=False))
        mock_pl_module.parameters = mock.MagicMock(return_value=iter([param1, param2]))

        with (
            mock.patch.object(cb, "source_area_ratio", return_value=[0.95]),
            mock.patch.object(cb, "intensity_ratio", return_value=([0.90], [0.85])),
        ):
            cb._log_metrics(dl, mock_pl_module)

        cb.experiment.log_metric.assert_called()
        calls = cb.experiment.log_metric.call_args_list

        # Should log num_trainable_parameters, mean_area_ratio,
        # mean_total_flux, mean_peak_flux
        assert len(calls) >= 4

    def test_log_metrics_computes_trainable_params(self, mock_trainer):
        config = _make_train_config(amp_phase=True)
        cb = LogAdditionalParamsCallback(config)

        mock_logger = mock.MagicMock()
        mock_logger.experiment = mock.MagicMock()
        mock_logger._run_id = "test_run"
        mock_logger.__class__ = MLFlowLogger
        mock_trainer.loggers = [mock_logger]

        cb.logger = mock_logger
        cb.experiment = mock_logger.experiment

        pred_tensor = torch.randn(1, 2, 32, 32)
        target_tensor = torch.randn(1, 2, 32, 32)

        dl = mock.MagicMock()
        dl.__iter__ = mock.MagicMock(return_value=iter([(pred_tensor, target_tensor)]))

        pl_module = mock.MagicMock()
        param1 = torch.nn.Parameter(torch.randn(10, 5, requires_grad=True))
        param2 = torch.nn.Parameter(torch.randn(5, 3, requires_grad=False))
        param1.requires_grad = True
        param2.requires_grad = False

        def mock_params():
            return iter([param1, param2])

        pl_module.parameters = mock_params

        with (
            mock.patch("radionets.core.callbacks.get_ifft") as mock_ifft,
            mock.patch.object(cb, "source_area_ratio", return_value=[0.95]),
            mock.patch.object(cb, "intensity_ratio", return_value=([0.90], [0.85])),
        ):
            mock_ifft.return_value = torch.randn(1, 2, 32, 32)
            cb._log_metrics(dl, pl_module)

        for call in cb.experiment.log_metric.call_args_list:
            if call[1].get("key") == "num_trainable_parameters":
                assert call[1].get("value") == 50
                break
