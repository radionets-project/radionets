from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning.pytorch.loggers import CSVLogger
from rich.logging import RichHandler

if TYPE_CHECKING:
    from pydantic import BaseModel


__all__ = ["Loggers"]


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


class Loggers:
    @classmethod
    def get_loggers(cls, train_config: BaseModel) -> list:
        default_logger = CSVLogger(
            save_dir=train_config.paths.model_path,
            **train_config.logging.default_logger.model_dump(),
        )
        default_logger._name = train_config.logging.project_name

        loggers = [default_logger]

        if train_config.logging.comet_ml:
            try:
                from lightning.pytorch.loggers import CometLogger
            except ImportError as e:
                raise ModuleNotFoundError(
                    "'comet_ml' was set to 'true' in your training config but "
                    "radionets could not import 'CometLogger'. This usually "
                    "indicates that 'comet_ml' is missing from your environment. "
                    "You can install it using 'uv pip install comet_ml'."
                ) from e

            comet_logger = CometLogger(
                project=train_config.logging.project_name,
                api_key=train_config.logging.comet_ml.api_key.get_secret_value(),
                **train_config.logging.comet_ml.model_dump(exclude="api_key"),
            )
            loggers.append(comet_logger)

        if train_config.logging.mlflow:
            try:
                from lightning.pytorch.loggers import MLFlowLogger
            except ImportError as e:
                raise ModuleNotFoundError(
                    "'mlflow' was set to 'true' in your training config but "
                    "radionets could not import 'MLflowLogger'. This usually "
                    "indicates that 'mlflow' is missing from your environment. "
                    "You can install it using 'uv pip install mlflow'."
                ) from e

            mlflow_logger = MLFlowLogger(
                experiment_name=train_config.logging.project_name,
                save_dir=train_config.paths.model_path,
                **train_config.logging.mlflow.model_dump(),
            )
            loggers.append(mlflow_logger)

        return loggers


def _setup_logger(namespace="rich", level="INFO", **kwargs):
    """Basic logging setup. Uses :class:`~rich.logging.RichHandler`
    for formatting and highlighting of the log.

    Parameters
    ----------
    namespace : str, optional
        Namespace to use for the logger. Default: ``'rich'``
    level : str, optional
        Logging level. Default ``'INFO'``
    **kwargs
        Keyword arguments for :class:`~rich.logging.RichHandler`.

    Returns
    -------
    logging.Logger
        Logger object using :class:`~rich.logging.RichHandler`
        for formatting and highlighting.

    See Also
    --------
    :class:`~rich.logging.RichHandler` :
        Rich's builtin logging handler for more information on
        allowed keyword arguments.
    """
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, **kwargs)],
    )

    return logging.getLogger(namespace)
