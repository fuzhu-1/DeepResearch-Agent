"""Tool call/response models for agent tools."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool invocation by an agent."""

    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default="")


class ToolResponse(BaseModel):
    """Response from a tool invocation."""

    tool_name: str
    result: Any = None
    error: Optional[str] = None
    call_id: str = Field(default="")
