from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AreaConfig",
    "DynamicRangeConfig",
    "IntensityConfig",
    "MeanDiffConfig",
    "ViewingAngleConfig",
    "SaveImagesConfig",
]


class SaveImagesConfig(BaseModel):
    split_size: int = Field(default=-1, gt=0)  # -1 will save all images in one file
    num_images: int | None = None
    random_sampling: int | bool = False

    model_config = ConfigDict(extra="allow")


class AreaConfig(BaseModel):
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    mode: Literal["contour", "pixel"] = "contour"

    model_config = ConfigDict(extra="allow")


class DynamicRangeConfig(BaseModel):
    sensitivity: float = Field(default=1e-6, gt=0.0)

    model_config = ConfigDict(extra="allow")


class IntensityConfig(BaseModel):
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")


class MeanDiffConfig(BaseModel):
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")


class ViewingAngleConfig(BaseModel):
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")
