"""Report model for storing generated research reports."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ReportSection(BaseModel):
    """A single section within a report."""

    title: str
    content: str
    subsections: List["ReportSection"] = Field(default_factory=list)


class ReportSource(BaseModel):
    """A source citation in the report."""

    url: str
    title: str
    snippet: str = Field(default="")


class Report(BaseModel):
    """Complete research report model."""

    task_id: str
    title: str = Field(default="")
    abstract: str = Field(default="")
    sections: List[ReportSection] = Field(default_factory=list)
    sources: List[ReportSource] = Field(default_factory=list)
    conclusion: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = Field(default=1)
