"""Request and response schemas for API endpoints."""
from pydantic import BaseModel, Field, field_validator


class BulletsRequest(BaseModel):
    """Request body for bullet generation."""

    text: str = Field(..., min_length=1, description="Input text to convert to bullets")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Reject whitespace-only input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        return v


class ErrorDetail(BaseModel):
    """Structured error response."""

    error_code: str
    message: str
    details: dict | None = None


class TokenLimitError(ErrorDetail):
    """Error response for token limit exceeded."""

    actual_tokens: int
    max_allowed_tokens: int
