from pathlib import Path
from typing import Literal

from pydantic import BaseModel

__all__ = [
    "BatchSizeFinderCallbackConfig",
    "EarlyStoppingCallbackConfig",
    "LearningRateMonitorCallbackConfig",
    "ModelCheckpointCallbackConfig",
    "TimerCallbackConfig",
]


class ModelCheckpointCallbackConfig(BaseModel):
    """Lightning ModelCheckpoint callback config"""

    dirpath: str | Path | None = None
    filename: str | Path | None = None
    monitor: str | None = None
    verbose: bool = False
    save_last: bool | Literal["link"] | None = None
    save_top_k: int = 1
    save_on_exception: bool = False
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: int | None = None
    every_n_epochs: int | None = None
    save_on_train_epoch_end: bool | None = None
    enable_version_counter: bool = True


class BatchSizeFinderCallbackConfig(BaseModel):
    """Lightning BatchSizeFinder callback config"""

    mode: Literal["power", "binsearch"] = "power"
    steps_per_trial: int = 3
    max_trials: int = 25


class EarlyStoppingCallbackConfig(BaseModel):
    """Lightning EarlyStopping callback config"""

    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "min"
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None
    check_on_train_epoch_end: bool | None = None
    log_rank_zero_only: bool = False


class LearningRateMonitorCallbackConfig(BaseModel):
    """Lightning LearningRateMonitor callback config"""

    logging_interval: str | None = "epoch"
    log_momentum: bool = False
    log_weight_decay: bool = False


class TimerCallbackConfig(BaseModel):
    """Lightning Timer callback config"""

    duration: str | None = "14:00:00:00"
    interval: Literal["epoch", "step"] = "epoch"
    verbose: bool = True
