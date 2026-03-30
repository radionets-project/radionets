import inspect
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radionets.architecture import archs

from . import data
from .training import (
    BatchSizeFinderCallbackConfig,
    CodeCarbonEmissionTrackerConfig,
    CometLoggerConfig,
    CSVLoggerConfig,
    DeepSpeedConfig,
    EarlyStoppingCallbackConfig,
    LearningRateMonitorCallbackConfig,
    LossConfig,
    LRSchedulerConfig,
    MLFlowLoggerConfig,
    ModelCheckpointCallbackConfig,
    OptimizerConfig,
    TimerCallbackConfig,
)


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_path: Path = Path("./example_data/")
    model_path: Path = Path("./build/example_model/")
    checkpoint: Path | None | Literal[False] = None

    @field_validator("data_path", "model_path", "checkpoint")
    @classmethod
    def expand_path(cls, v: Path) -> Path:
        """Expand and resolve paths."""

        if v in {None, False}:
            v = None
        else:
            v.expanduser().resolve()

        return v


class ModelConfig(BaseModel):
    arch_name: str | Callable = archs.SRResNet18
    amp_phase: bool = True
    normalize: bool = False

    @field_validator("arch_name")
    @classmethod
    def load_arch_instance(cls, arch: str):
        avail_archs = {}

        for member in inspect.getmembers(archs):
            if inspect.isclass(member[1]):
                avail_archs[member[0]] = member[1]

        try:
            arch = avail_archs[arch]
        except KeyError as e:
            raise ValueError(
                f"Unknown architecture: TrainConfig got {arch} but expected "
                f"one of {avail_archs.keys()}!"
            ) from e

        return arch


class TrainingConfig(BaseModel):
    """Hyperparameters configuration."""

    num_epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=100, gt=0)
    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    lr_scheduling: bool | LRSchedulerConfig = False

    @field_validator("loss", mode="after")
    @classmethod
    def validate_loss(cls, v):
        if isinstance(v, dict):
            return LossConfig(**v)

        return v

    @field_validator("optimizer", mode="after")
    @classmethod
    def validate_optimizer(cls, v):
        if isinstance(v, dict):
            return OptimizerConfig(**v)

        return v

    @field_validator("lr_scheduling", mode="after")
    @classmethod
    def validate_lr_scheduler(cls, v: bool | LRSchedulerConfig):
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return LRSchedulerConfig(**v)
        elif v is True:
            return LRSchedulerConfig()

        return v


class DeviceConfig(BaseModel):
    """Device configuration settings."""

    accelerator: str = "auto"
    num_devices: str | list | int = "auto"
    precision: str | int = "32-true"
    deepspeed: bool | str | DeepSpeedConfig = False
    strategy: str = "auto"

    @model_validator(mode="after")
    def check_device_count(self) -> Self:
        if self.accelerator in ["gpu", "tpu", "hpu"] and not torch.cuda.is_available():
            raise ValueError(
                f"'accelerator' is set to {self.accelerator} in the "
                "configuration but CUDA is not available. Please "
                "ensure CUDA is installed or set accelerator to 'cpu'."
            )

        if (
            self.accelerator in ["gpu", "tpu", "hpu"]
            and isinstance(self.num_devices, int) > torch.cuda.device_count()
        ):
            raise ValueError(
                f"'num_devices' exceeds the number of available {self.accelerator}s "
                f"({self.num_devices} > {torch.cuda.device_count})"
            )

        return self

    @field_validator("deepspeed", mode="after")
    @classmethod
    def validate_deepspeed(cls, v: bool | str | DeepSpeedConfig):
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return DeepSpeedConfig(**v)
        elif v is True:
            return DeepSpeedConfig()

        return v


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""

    module: str | Callable = data.H5DataModule
    num_workers: int = Field(default=10, gt=0)
    compressed: bool = False
    fourier: bool = True

    model_config = ConfigDict(extra="allow")

    @field_validator("module")
    @classmethod
    def load_data_module_instance(cls, name: str):
        if isinstance(name, type):
            return name

        avail_data_modules = {}

        for member in inspect.getmembers(data):
            if inspect.isclass(member[1]):
                avail_data_modules[member[0]] = member[1]

        try:
            if name.lower() in ["h5", "hdf5"]:
                data_module = avail_data_modules["H5DataModule"]
            elif name.lower() in ["wds", "webdataset"]:
                data_module = avail_data_modules["WebDatasetModule"]
            else:
                data_module = avail_data_modules[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown optimizer: TrainConfig got {name} but expected "
                f"one of {set(avail_data_modules)}!"
            ) from e

        return data_module


class CallbacksConfig(BaseModel):
    "Callbacks configuration."

    model_checkpoint: bool | ModelCheckpointCallbackConfig = False
    batch_size_finder: bool | BatchSizeFinderCallbackConfig = False
    early_stopping: bool | EarlyStoppingCallbackConfig = False
    lr_monitor: bool | LearningRateMonitorCallbackConfig = False
    timer: bool | TimerCallbackConfig = False
    device_stats_monitor: bool = False

    @field_validator("model_checkpoint", mode="after")
    @classmethod
    def validate_model_checkpoint(cls, v):
        if isinstance(v, dict):
            return ModelCheckpointCallbackConfig(**v)
        elif v is True:
            return ModelCheckpointCallbackConfig()  # Return defaults

        return v

    @field_validator("batch_size_finder", mode="after")
    @classmethod
    def validate_batch_size_finder(cls, v):
        if isinstance(v, dict):
            return BatchSizeFinderCallbackConfig(**v)
        elif v is True:
            return BatchSizeFinderCallbackConfig()  # Return defaults

        return v

    @field_validator("early_stopping", mode="after")
    @classmethod
    def validate_early_stopping(cls, v):
        if isinstance(v, dict):
            return EarlyStoppingCallbackConfig(**v)
        elif v is True:
            return EarlyStoppingCallbackConfig()  # Return defaults

        return v

    @field_validator("lr_monitor", mode="after")
    @classmethod
    def validate_lr_monitor(cls, v):
        if isinstance(v, dict):
            return LearningRateMonitorCallbackConfig(**v)
        elif v is True:
            return LearningRateMonitorCallbackConfig()  # Return defaults

        return v

    @field_validator("timer", mode="after")
    @classmethod
    def validate_timer(cls, v):
        if isinstance(v, dict):
            return TimerCallbackConfig(**v)
        elif v is True:
            return TimerCallbackConfig()  # Return defaults

        return v


class LoggingConfig(BaseModel):
    """Logging and experiment tracking configuration."""

    project_name: str = "Radionets"
    plot_n_epochs: int = Field(default=10, gt=0)
    scale: bool = True
    default_logger: CSVLoggerConfig = CSVLoggerConfig()
    comet_ml: bool | CometLoggerConfig = False
    mlflow: bool | MLFlowLoggerConfig = False
    codecarbon: bool | CodeCarbonEmissionTrackerConfig = False

    @field_validator("default_logger", mode="after")
    @classmethod
    def validate_default_logger(cls, v):
        if isinstance(v, dict):
            return CSVLoggerConfig(**v)

        return v

    @field_validator("comet_ml", mode="after")
    @classmethod
    def validate_comet_ml(cls, v):
        if isinstance(v, dict):
            return CometLoggerConfig(**v)
        elif v is True:
            return CometLoggerConfig()  # Return defaults

        return v

    @field_validator("mlflow", mode="after")
    @classmethod
    def validate_mlflow(cls, v):
        if isinstance(v, dict):
            return MLFlowLoggerConfig(**v)
        elif v is True:
            return MLFlowLoggerConfig()  # Return defaults

        return v

    @field_validator("codecarbon", mode="after")
    @classmethod
    def validate_codecarbon(cls, v: bool | CodeCarbonEmissionTrackerConfig):
        if isinstance(v, dict):
            return CodeCarbonEmissionTrackerConfig(
                **v, project_name=cls.logging.project_name
            )
        elif v is True:
            return CodeCarbonEmissionTrackerConfig(
                project_name=cls.logging.project_name
            )

        # NOTE: CometML automatically logs with codecarbon
        # if codecarbon is installed. This should ensure
        # that codecarbon is only used when set in the config
        os.environ["COMET_AUTO_LOG_CO2"] = "false"

        return v


class TrainConfig(BaseModel):
    """Main training configuration."""

    title: str = "Train configuration"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
