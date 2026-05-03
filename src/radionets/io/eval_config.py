import inspect
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Self

import torch
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from radionets.architecture import archs

from .train_config import DataLoaderConfig
from .training import DeepSpeedConfig


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_path: Path = Path("./example_data/")
    model_paths: list[Path] = Field(
        default=[Path("./path/to/model.ckpt")],
        min_length=1,
        max_length=2,
    )
    save_path: Path = Path("./build")

    @field_validator("data_path", "save_path")
    @classmethod
    def expand_path(cls, v: Path) -> Path:
        """Expand and resolve paths."""

        if v in {None, False}:
            v = None
        else:
            v.expanduser().resolve()

        return v

    @field_validator("model_paths")
    @classmethod
    def expand_model_paths(cls, v: Path | list[Path]) -> list[Path]:
        """Expand model paths"""
        if not isinstance(v, list):
            v = [v]

        v = [
            path.expanduser().resolve() if path not in {None, False} else None
            for path in v
        ]

        return v


class ModelConfig(BaseModel):
    arch_name: str | list[Callable] = Field(
        default=[archs.SRResNet18],
        min_length=1,
        max_length=2,
    )
    fourier: bool = True
    amp_phase: bool = False

    @field_validator("arch_name")
    @classmethod
    def load_arch_instance(cls, archs: str):
        avail_archs = {}

        for member in inspect.getmembers(archs):
            if inspect.isclass(member[1]):
                avail_archs[member[0]] = member[1]

        arch_list = []
        for arch in archs:
            try:
                if isinstance(arch, str):
                    arch_list.append(avail_archs[arch])
                else:
                    arch_list.append(arch)
            except KeyError as e:
                raise ValueError(
                    f"Unknown architecture: EvalConfig got {arch} but expected "
                    f"one of {avail_archs.keys()}!"
                ) from e

        return arch_list


class DeviceConfig(BaseModel):
    """Device configuration settings."""

    accelerator: list[str] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )
    num_devices: list[str | list | int] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )
    precision: list[str | int] = Field(
        default=["32-true"],
        min_length=1,
        max_length=2,
    )
    deepspeed: list[bool | str | DeepSpeedConfig] = Field(
        default=[False],
        min_length=1,
        max_length=2,
    )
    strategy: list[str] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )

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

        fields = []
        for field in self.__class__.model_fields:
            val = getattr(self, field)
            if not isinstance(val, list):
                setattr(self, field, [val, val])
            elif len(val) != 2:
                # Repeat list entry if val is not of length 2
                # so that we get a two-item list
                setattr(self, field, val * 2)

            fields.append(len(getattr(self, field)))

        if len(set(fields)) > 1:
            raise RuntimeError(
                "Expected all device config fields to be lists of length 2!"
            )

        return self

    @field_validator("deepspeed", mode="after")
    @classmethod
    def validate_deepspeed(cls, val: bool | str | DeepSpeedConfig):
        result = []
        for v in val:
            if isinstance(v, str):
                result.append(v)
            elif isinstance(v, dict):
                result.append(DeepSpeedConfig(**v))
            elif v is True:
                result.append(DeepSpeedConfig())
            else:
                result.append(v)

        return result


class GeneralConfig(BaseModel):
    batch_size: int = 20


class EvaluationComponentsConfig(BaseModel):
    viewing_angle: bool = True
    dynamic_range: bool = True
    ms_ssim: bool = False
    intensity: bool = True
    mean_diff: bool = True
    area: bool = True
    point: bool = False
    predict_grad: bool = False
    evaluate_gan: bool = True


class EvalConfig(BaseModel):
    """Main training configuration."""

    title: str = "Evaluation configuration"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    evaluation: EvaluationComponentsConfig = Field(
        default_factory=EvaluationComponentsConfig
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> "EvalConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
