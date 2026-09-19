"""Request/response schemas. Literal copy of backend/src/api/schemas.py."""
from pydantic import BaseModel, Field, field_validator


class BulletsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to convert to bullets")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        return v


class ErrorDetail(BaseModel):
    error_code: str
    message: str
    details: dict | None = None


class TokenLimitError(ErrorDetail):
    actual_tokens: int
    max_allowed_tokens: int
