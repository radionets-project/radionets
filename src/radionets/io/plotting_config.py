import sysconfig
import tomllib
from pathlib import Path

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
)
from pydantic_settings import BaseSettings

__all__ = [
    "PlottingConfig",
]


class PathsConfig(BaseSettings):
    """File paths configuration."""

    data_path: str | Path | list[str | Path] = Field(
        min_length=1,
        max_length=4,
    )
    """Path to the directory containing the evaluation data."""

    save_path: Path
    """Path to the directory where the plots will be saved."""

    @field_validator("data_path", mode="before")
    @classmethod
    def expand_data_paths(cls, v: str | Path | list[str | Path]) -> list[Path]:

        paths = []
        if not isinstance(v, list):
            paths.append(v)

        paths = [Path(path).expanduser().resolve() for path in paths]

        return paths

    @field_validator("save_path")
    @classmethod
    def expand_save_path(cls, v: Path) -> Path:
        """Expand and resolve paths."""
        return v.expanduser().resolve()


class RCParamsConfig(BaseSettings):
    model_config = ConfigDict(extra="allow")


class GeneralConfig(BaseSettings):
    display_names: None | list = Field(default=None)
    colors: None | list = Field(default=None)
    mplstyle: str | Path = Field(default="radionets")
    rcparams: dict | RCParamsConfig = Field(default_factory=RCParamsConfig)

    @field_validator("mplstyle")
    @classmethod
    def validate_mplstyle(cls, v):
        if v == "radionets":
            root = sysconfig.get_path("data", sysconfig.get_default_scheme())
            v = root + "/share/resources/radionets.mplstyle"

        return v

    @field_validator("rcparams", mode="before")
    @classmethod
    def validate_rcparams_config(cls, v: dict | RCParamsConfig):
        if isinstance(v, dict):
            return RCParamsConfig.model_validate({**v})
        elif isinstance(v, RCParamsConfig):
            return RCParamsConfig.model_validate(v)


class SubplotsConfig(BaseSettings):
    layout: str = "constrained"

    model_config = ConfigDict(extra="allow")


class FigSaveConfig(BaseSettings):
    bbox_iches: str = "tight"

    model_config = ConfigDict(extra="allow")


class FigConfig(BaseSettings):
    subplots: SubplotsConfig = Field(default_factory=SubplotsConfig)
    save: FigSaveConfig = Field(default_factory=FigSaveConfig)


class HistConfig(BaseSettings):
    bins: int = 60
    histtype: str = "step"
    linewidth: float = 1.5

    model_config = ConfigDict(extra="allow")


class LegendConfig(BaseSettings):
    legx: float = 0.55
    legy: float = 1.2
    ncols: int = 2
    loc: str = "upper center"
    frameon: bool = False

    model_config = ConfigDict(extra="allow")


class PlottingBase(BaseSettings):
    fig: dict | FigConfig = Field(default_factory=FigConfig)
    hist: dict | HistConfig = Field(default_factory=HistConfig)
    legend: dict | LegendConfig = Field(default_factory=LegendConfig)
    lower_bound: int = 0
    upper_bound: int = 2

    @field_validator("fig", mode="before")
    @classmethod
    def validate_fig_config(cls, v: dict | FigConfig):
        if isinstance(v, dict):
            return FigConfig.model_validate({**v})
        elif isinstance(v, FigConfig):
            return FigConfig.model_validate(v)

    @field_validator("hist", mode="before")
    @classmethod
    def validate_hist_config(cls, v: dict | HistConfig):
        if isinstance(v, dict):
            return HistConfig.model_validate({**v})
        elif isinstance(v, HistConfig):
            return HistConfig.model_validate(v)

    @field_validator("legend", mode="before")
    @classmethod
    def validate_legend_config(cls, v: dict | LegendConfig):
        if isinstance(v, dict):
            return LegendConfig.model_validate({**v})
        elif isinstance(v, LegendConfig):
            return LegendConfig.model_validate(v)


class PeakFluxPlotConfig(PlottingBase): ...


class IntegratedFluxPlotConfig(PlottingBase): ...


class AnglePlotConfig(BaseSettings):
    fig: FigConfig = Field(default_factory=FigConfig)
    legend: LegendConfig = Field(default_factory=LegendConfig)
    lower_bounds: tuple[int, int] = (-90, -20)
    upper_bounds: tuple[int, int] = (90, 20)
    hist0: dict | HistConfig = Field(default_factory=HistConfig)
    hist1: dict | HistConfig = Field(default_factory=HistConfig)

    @field_validator("hist0", mode="before")
    @classmethod
    def validate_hist0_config(cls, v: dict | HistConfig):
        if isinstance(v, dict):
            return HistConfig.model_validate({"bins": 50, **v})
        elif isinstance(v, HistConfig):
            user_fields = {
                key: val
                for key, val in v.model_dump().items()
                if key in v.model_fields_set
            }

            if "bins" not in user_fields:
                user_fields["bins"] = 50

            return HistConfig.model_validate({**user_fields})

    @field_validator("hist1", mode="before")
    @classmethod
    def validate_hist1_config(cls, v: dict | HistConfig):
        if isinstance(v, dict):
            return HistConfig.model_validate({"bins": 30, **v})
        elif isinstance(v, HistConfig):
            user_fields = {
                key: val
                for key, val in v.model_dump().items()
                if key in v.model_fields_set
            }

            if "bins" not in user_fields:
                user_fields["bins"] = 30

            return HistConfig.model_validate({**user_fields})

    @field_validator("fig", mode="before")
    @classmethod
    def validate_fig_config(cls, v: dict | FigConfig):
        if isinstance(v, dict):
            if "subplots" not in v:
                return FigConfig.model_validate({"subplots": {"figsize": (7, 3)}, **v})
            elif "figsize" not in v["subplots"]:
                v["subplots"].update({"figsize": (7, 3)})

            return FigConfig.model_validate({**v})
        elif isinstance(v, FigConfig):
            user_fields = {
                key: val
                for key, val in v.subplots.model_dump().items()
                if key in v.subplots.model_fields_set
            }

            fields = {}
            if "figsize" not in user_fields:
                fields["subplots"] = {"figsize": (7, 3)}

            return FigConfig.model_validate({**fields})


class MeanDiffPlotConfig(PlottingBase):
    lower_bound: int = -50
    lower_bound: int = 50

    @field_validator("hist", mode="before")
    @classmethod
    def validate_hist_config(cls, v: dict | HistConfig):
        if isinstance(v, dict):
            return HistConfig.model_validate({"bins": 50, **v})
        elif isinstance(v, HistConfig):
            user_fields = {
                key: val
                for key, val in v.model_dump().items()
                if key in v.model_fields_set
            }

            if "bins" not in user_fields:
                user_fields["bins"] = 50

            return HistConfig.model_validate({**user_fields})


class SourceAreaPlotConfig(PlottingBase):
    @field_validator("hist", mode="before")
    @classmethod
    def validate_hist_config(cls, v: dict | HistConfig):
        if isinstance(v, dict):
            return HistConfig.model_validate({"bins": 50, **v})
        elif isinstance(v, HistConfig):
            user_fields = {
                key: val
                for key, val in v.model_dump().items()
                if key in v.model_fields_set
            }

            if "bins" not in user_fields:
                user_fields["bins"] = 50

            return HistConfig.model_validate({**user_fields})

    @field_validator("legend", mode="before")
    @classmethod
    def validate_legend_config(cls, v: dict | LegendConfig):
        if isinstance(v, dict):
            return LegendConfig.model_validate({"legy": 1.35, **v})
        elif isinstance(v, LegendConfig):
            user_fields = {
                key: val
                for key, val in v.model_dump().items()
                if key in v.model_fields_set
            }

            if "legy" not in user_fields:
                user_fields["legy"] = 1.35

            return LegendConfig.model_validate({**user_fields})


class PlottingConfig(BaseSettings):
    """Main training configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    peak_flux: bool | PeakFluxPlotConfig = True
    integrated_flux: bool | IntegratedFluxPlotConfig = True
    angle: bool | dict | AnglePlotConfig = True
    mean_diff: bool | dict | MeanDiffPlotConfig = True
    area: bool | dict | SourceAreaPlotConfig = True
    debug: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "PlottingConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    @field_validator("peak_flux", mode="after")
    @classmethod
    def validate_peak_flux_config(cls, v: bool | dict | PeakFluxPlotConfig):
        if isinstance(v, dict):
            return PeakFluxPlotConfig(**v)
        elif v is True:
            return PeakFluxPlotConfig()  # Return defaults

        return v

    @field_validator("intensity_sum", mode="after")
    @classmethod
    def validate_intensity_sum_config(cls, v: bool | dict | IntegratedFluxPlotConfig):
        if isinstance(v, dict):
            return IntegratedFluxPlotConfig(**v)
        elif v is True:
            return IntegratedFluxPlotConfig()  # Return defaults

        return v

    @field_validator("angle", mode="after")
    @classmethod
    def validate_angle_config(cls, v: bool | dict | AnglePlotConfig):
        if isinstance(v, dict):
            return AnglePlotConfig(**v)
        elif v is True:
            return AnglePlotConfig()  # Return defaults

        return v

    @field_validator("mean_diff", mode="after")
    @classmethod
    def validate_mean_diff_config(cls, v: bool | dict | MeanDiffPlotConfig):
        if isinstance(v, dict):
            return MeanDiffPlotConfig(**v)
        elif v is True:
            return MeanDiffPlotConfig()  # Return defaults

        return v

    @field_validator("area", mode="after")
    @classmethod
    def validate_area_config(cls, v: bool | dict | SourceAreaPlotConfig):
        if isinstance(v, dict):
            return SourceAreaPlotConfig(**v)
        elif v is True:
            return SourceAreaPlotConfig()  # Return defaults

        return v
