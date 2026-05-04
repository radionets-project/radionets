from pydantic import BaseModel, Field

__all__ = ["AreaConfig"]


class AreaConfig(BaseModel):
    level: float = Field(default=0.05, ge=0.0, le=1.0)
