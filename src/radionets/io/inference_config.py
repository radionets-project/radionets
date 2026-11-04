import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from .eval_config import DataLoaderConfig, DeviceConfig, ModelConfig, PathsConfig

__all__ = [
    "InferenceConfig",
]


class SaveImagesConfig(BaseSettings):
    split_size: int = Field(default=-1, gt=0)  # -1 will save all images in one file


class InferenceConfig(BaseSettings):
    """Main training configuration."""

    title: str = "Evaluation configuration"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    save_images: SaveImagesConfig = Field(default_factory=SaveImagesConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "InferenceConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
