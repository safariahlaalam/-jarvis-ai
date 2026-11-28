"""Strict Pydantic request schemas."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be empty or whitespace")
        return value


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=100)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Project name must not be empty or whitespace")
        return value
