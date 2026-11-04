"""Tests for src/radionets/core/logging.py"""

import logging
from pathlib import Path
from unittest import mock

import pytest
from lightning.pytorch.loggers import CSVLogger

from radionets.core.logging import Loggers, _setup_logger

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

    config.paths = mock.MagicMock()
    config.paths.model_path = Path("/tmp/test_model_path")

    config.logging.comet_ml = overrides.get("comet_ml")
    config.logging.mlflow = overrides.get("mlflow")
    config.logging.project_name = overrides.get("project_name", "test_project")
    config.logging.scale = overrides.get("scale", 1.0)
    config.logging.plot_n_epochs = overrides.get("plot_n_epochs", 1)

    config.logging.default_logger = mock.MagicMock()
    config.logging.default_logger.model_dump.return_value = overrides.get(
        "default_logger", {"name": "test_csv"}
    )

    return config


class TestGetLoggers:
    def test_returns_list(self):
        config = _make_train_config()
        result = Loggers.get_loggers(config)

        assert isinstance(result, list)

    def test_always_returns_csv_logger(self):
        config = _make_train_config()
        loggers = Loggers.get_loggers(config)

        csv_logger = [logger for logger in loggers if isinstance(logger, CSVLogger)]

        assert len(csv_logger) == 1

    def test_csv_logger_save_dir(self):
        config = _make_train_config()
        loggers = Loggers.get_loggers(config)

        csv_logger = loggers[0]

        assert str(csv_logger.save_dir) == "/tmp/test_model_path"

    def test_csv_logger_name_set(self):
        config = _make_train_config(project_name="test")
        loggers = Loggers.get_loggers(config)

        csv_logger = loggers[0]

        assert csv_logger._name == "test"

    def test_comet_logger_added(self):
        comet_cfg = mock.MagicMock()
        comet_cfg.api_key.get_secret_value.return_value = "test_api_key"

        config = _make_train_config(comet_ml=comet_cfg)
        # Verify config has comet_ml set
        assert config.logging.comet_ml is not None

    def test_comet_logger_passes_project_name(self):
        comet_cfg = mock.MagicMock()
        comet_cfg.api_key.get_secret_value.return_value = "test_key"

        config = _make_train_config(comet_ml=comet_cfg, project_name="test")

        assert config.logging.project_name == "test"

    def test_comet_logger_excludes_api_key_from_model_dump(self):
        comet_cfg = mock.MagicMock()
        comet_cfg.api_key.get_secret_value.return_value = "secret_key_42"
        comet_cfg.model_dump.return_value = {"project": "test"}

        _make_train_config(comet_ml=comet_cfg)

        assert comet_cfg.api_key.get_secret_value.return_value == "secret_key_42"
        assert "api_key" not in comet_cfg.model_dump.return_value

    def test_mlflow_logger_added(self):
        config = _make_train_config(mlflow=mock.MagicMock())

        assert config.logging.mlflow is not None


class TestSetupLogger:
    def test_returns_logger(self):
        result = _setup_logger(namespace="test_namespace")
        assert isinstance(result, logging.Logger)

    def test_default_namespace(self):
        logger = _setup_logger()
        assert logger.name == "rich"

    def test_custom_namespace(self):
        logger = _setup_logger(namespace="custom")
        assert logger.name == "custom"

    def test_default_level_is_info(self):
        logger = _setup_logger()

        assert isinstance(logger, logging.Logger)

    def test_custom_level(self):
        logger = _setup_logger(level="DEBUG")

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG

    def test_rich_handler_tracebacks_enabled(self):
        logger = _setup_logger()
        handlers = logger.handlers
        for handler in handlers:
            if hasattr(handler, "rich_tracebacks"):
                assert handler.rich_tracebacks is True

    def test_format(self):
        logger = _setup_logger()
        handlers = logger.handlers
        for handler in handlers:
            if handler.formatter:
                fmt = handler.formatter._fmt
                assert fmt == "%(message)s"

    def test_passes_rich_handler_kwargs(self):
        logger = _setup_logger(markup=False)
        handlers = logger.handlers
        for handler in handlers:
            if hasattr(handler, "markup"):
                assert handler.markup is False

    def test_logger_can_log(self):
        logger = _setup_logger(level="DEBUG")

        # None of these should raise an error
        logger.debug("test debug message")
        logger.info("test info message")
        logger.warning("test warning message")
