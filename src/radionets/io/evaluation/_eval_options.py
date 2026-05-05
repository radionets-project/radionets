from pydantic import BaseModel, Field

__all__ = ["AreaConfig", "IntensityConfig"]


class IntensityConfig(BaseModel):
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class AreaConfig(BaseModel):
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)
