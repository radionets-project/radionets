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
from pydantic_settings import BaseSettings

from radionets.architecture import archs

from .evaluation import (
    AreaConfig,
    DynamicRangeConfig,
    IntensityConfig,
    MeanDiffConfig,
    SaveImagesConfig,
    ViewingAngleConfig,
)
from .train_config import DataLoaderConfig
from .training import DeepSpeedConfig

__all__ = [
    "PathsConfig",
    "ModelConfig",
    "DeviceConfig",
    "DataLoaderConfig",
    "EvaluationMethodsConfig",
    "EvalConfig",
]


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_path: Path = Path("./example_data/")
    """Path to the directory containing the test dataset."""

    model_paths: list[Path] = Field(
        default=[Path("./path/to/model.ckpt")],
        min_length=1,
        max_length=2,
    )
    """Paths to the pretrained model checkpoints."""

    save_path: Path = Path("./build")
    """Path to the directory where evaluation results will be saved."""

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
    """The model configuration sets the architecture
    and data representation.
    """

    arch_name: str | list[str | Callable] = Field(
        default=[archs.SRResNet18],
        min_length=1,
        max_length=2,
    )
    """Name/callable or list of two names/callables of the architecture(s)
    to use for evaluation."""

    weights_only: bool | list[bool] = Field(
        default=[True],
        min_length=1,
        max_length=2,
    )
    """Whether PyTorch's unpickler should be restricted when loading checkpoints.
    See `torch.load with weights_only=True <https://docs.pytorch.org/docs/2.12/notes/serialization.html#weights-only>`_
    for more information.
    """

    @field_validator("arch_name")
    @classmethod
    def load_arch_instance(cls, arch_name: str | list) -> list:
        if isinstance(arch_name, str):
            arch_name = [arch_name]

        avail_archs = {}
        for member in inspect.getmembers(archs):
            if inspect.isclass(member[1]):
                avail_archs[member[0]] = member[1]

        arch_list = []
        for arch in arch_name:
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

    @field_validator("weights_only", mode="before")
    @classmethod
    def validate_weights_only(cls, weights_only: bool | list[bool]) -> list:
        if isinstance(weights_only, bool):
            weights_only = [weights_only]

        return weights_only


class DeviceConfig(BaseModel):
    """Device configuration settings.

    Allows setting the type and number of devices used for
    the evaluation, as well as strategies for distribution.
    """

    accelerator: str | list[str] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )
    """Type of accelerator to use (e.g., "auto", "gpu", "cpu"). See
    `PyTorch Lightning Accelerator
    <https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html>`_
    for more information.
    """

    num_devices: str | int | list[str | list | int] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )
    """Number of devices to use."""

    precision: str | int | list[str | int] = Field(
        default=["32-true"],
        min_length=1,
        max_length=2,
    )
    """Precision to use (e.g., "32-true", "16-mixed"). See
    `PyTorch Lightning N-bit Precision
    <https://lightning.ai/docs/pytorch/stable/common/precision.html>`_
    for more information
    """

    deepspeed: bool | str | DeepSpeedConfig | list[bool | str | DeepSpeedConfig] = (
        Field(
            default=[False],
            min_length=1,
            max_length=2,
        )
    )

    """Whether to use the deepspeed deep learning training optimization library.
    Set to ``True`` if you want to use the `default settings
    <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DeepSpeedStrategy.html>`_,
    or change specific settings yourself:

    .. code-block:: toml

        [devices.deepspeed]
        stage = 1

    You can also pass a string like ``"deepspeed_stage_1"`` to use the settings
    pre-defined by PyTorch Lightning. See `DeepSpeed
    <https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html>`_
    for more information.
    """

    strategy: str | list[str] = Field(
        default=["auto"],
        min_length=1,
        max_length=2,
    )
    """Select a strategy for model distribution during evaluation. See
    `What is a Strategy? <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>`_
    for more information.
    """

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

    @field_validator("deepspeed", mode="before")
    @classmethod
    def validate_deepspeed(cls, val: bool | str | DeepSpeedConfig | list):
        if isinstance(val, str | bool | DeepSpeedConfig):
            val = [val]

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


class EvaluationMethodsConfig(BaseModel):
    """Contains all settings for the evaluation methods defined in
    :mod:`~radionets.evaluation`.
    """

    save_images: bool | dict | SaveImagesConfig = Field(
        default=True, validate_default=True
    )
    """Whether to save images in .pt pickle files."""

    viewing_angle: bool = Field(default=True, validate_default=True)
    """Enable viewing angle evaluation of the source. ``radionets`` will
    use a PCA to estimate the relative source angle to the x-axis of the image
    and compare prediction and target angles.
    """

    dynamic_range: bool | dict | DynamicRangeConfig = Field(
        default=True, validate_default=True
    )
    """Enable dynamic range evaluation of the source flux. ``radionets`` will
    compute the RMS for both prediction and target and estimate the dynamic range
    of both images.
    """

    intensity: bool | dict | IntensityConfig = Field(
        default=True, validate_default=True
    )
    """Enable peak flux intensity and integrated flux intensity evaluation of the
    source. The peak flux intensity is the maximum flux emitted from the source.
    The integrated flux intensity is the sum of all pixels above a threshold
    (the default is ``0.05``) of the maximum target flux. To change the threshold,
    change the config as follows:

    .. code-block:: toml

        [evaluation.intensity]
        threshold = 0.05

    The ratios between predictions and targets are saved to a file `flux_intensity.csv`
    under the path specified in the ``save_path`` field. For more information, refer
    to :func:`~radionets.evaluation.contour.intensity_ratio` and
    :func:`~radionets.evaluation.contour.eval_intensity`.
    """

    mean_diff: bool | dict | MeanDiffConfig = Field(default=True, validate_default=True)
    """Enable mean difference computation between prediction and target.
    Sources are detected using a Laplacian of Gaussian approach
    (using skimage's :func:`~skimage.feature.blob_log`). The mean of the differences
    of the blobs is saved to a file `mean_diff.csv` under the path specified in the
    ``save_path`` field. All values are percentages. For more information, refer to
    :func:`~radionets.evaluation.feature.eval_mean_difference`.
    """

    area: bool | dict | AreaConfig = Field(default=True, validate_default=True)
    """Enable source area evaluation. The area is estimated using a matplotlib
    :class:`~matplotlib.contour.QuadContourSet`. All flux above a threshold
    (the default is ``0.05``) of the maximum target flux is considered signal
    and thus contributes to the area. To change the threshold,
    change the config as follows:


    .. code-block:: toml

        [evaluation.intensity]
        threshold = 0.05

    The ratios between predictions and targets are saved to a file `area_ratios.csv`
    under the path specified in the ``save_path`` field. For more information, refer
    to :func:`~radionets.evaluation.contour.source_area_ratio` and
    :func:`~radionets.evaluation.contour.eval_area`.
    """

    predict_grad: bool = True
    evaluate_gan: bool = True

    @field_validator("save_images", mode="after")
    @classmethod
    def validate_save_images_config(cls, v: bool | dict | SaveImagesConfig):
        if isinstance(v, dict):
            return SaveImagesConfig(**v)
        elif v is True:
            return SaveImagesConfig()  # Return defaults

        return v

    @field_validator("viewing_angle", mode="after")
    @classmethod
    def validate_viewing_angle_config(cls, v: bool | dict | ViewingAngleConfig):
        if isinstance(v, dict):
            return ViewingAngleConfig(**v)
        elif v is True:
            return ViewingAngleConfig()  # Return defaults

        return v

    @field_validator("dynamic_range", mode="after")
    @classmethod
    def validate_dynamic_range_config(cls, v: bool | dict | DynamicRangeConfig):
        if isinstance(v, dict):
            return DynamicRangeConfig(**v)
        elif v is True:
            return DynamicRangeConfig()  # Return defaults

        return v

    @field_validator("intensity", mode="after")
    @classmethod
    def validate_intensity_config(cls, v: bool | dict | IntensityConfig):
        if isinstance(v, dict):
            return IntensityConfig(**v)
        elif v is True:
            return IntensityConfig()  # Return defaults

        return v

    @field_validator("mean_diff", mode="after")
    @classmethod
    def validate_mean_diff_config(cls, v: bool | dict | MeanDiffConfig):
        if isinstance(v, dict):
            return MeanDiffConfig(**v)
        elif v is True:
            return MeanDiffConfig()  # Return defaults

        return v

    @field_validator("area", mode="after")
    @classmethod
    def validate_area_config(cls, v: bool | dict | AreaConfig):
        if isinstance(v, dict):
            return AreaConfig(**v)
        elif v is True:
            return AreaConfig()  # Return defaults

        return v


class EvalConfig(BaseSettings):
    """Main training configuration."""

    title: str = "Evaluation configuration"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    evaluation: EvaluationMethodsConfig = Field(default_factory=EvaluationMethodsConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "EvalConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
