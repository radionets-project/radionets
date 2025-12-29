import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator

__all__ = [
    "CSVLoggerConfig",
    "CometLoggerConfig",
    "MLFlowLoggerConfig",
    "CodeCarbonEmissionTrackerConfig",
]


class CSVLoggerConfig(BaseModel):
    """Lightning CSVLogger logging config"""

    name: str | None = "lightning_logs"
    version: int | str | None = None
    prefix: str = ""
    flush_logs_every_n_steps: int = 100


class CometLoggerConfig(BaseModel):
    """Lightning CometLogger logging config"""

    api_key: SecretStr = SecretStr(os.getenv("COMET_API_KEY"))
    workspace: str | None = None
    experiment_key: str | None = None
    mode: Literal["get_or_create", "get", "create"] | None = None
    online: bool | None = None
    prefix: str | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, key: SecretStr | None) -> SecretStr | None:
        key = SecretStr(key) if key else None

        return key


class MLFlowLoggerConfig(BaseModel):
    """Lightning MLFlowLogger logging config"""

    run_name: str | None = None
    tracking_uri: str | None = "http://127.0.0.1:5000"
    tags: dict | None = None
    log_model: Literal[True, False, "all"] = False
    prefix: str = ""
    artifact_location: str | None = None
    run_id: str | None = None
    synchronous: bool | None = None


class CodeCarbonEmissionTrackerConfig(BaseModel):
    """Codecarbon emission tracker configuration"""

    log_level: str | int = "error"
    country_iso_code: str = "DEU"
    output_dir: str | None = None

    @field_validator("output_dir", mode="after")
    @classmethod
    def expand_path(cls, v: str | Path) -> str:
        """Expand and resolve paths."""

        if not isinstance(v, Path):
            v = Path(v)

        if v in {None, False}:
            v = Path(os.getcwd())
        else:
            v.expanduser().resolve()

        if not v.exists():
            v.mkdir(parents=True)

        return str(v)
