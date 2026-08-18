"""Services module for DeepResearch-Agent.

Provides ReportService, LLMService, MemoryService, and other
business-logic services. Task orchestration lives in TaskManager.
"""

from app.services.report_service import ReportService
from app.services.llm_service import LLMService

try:
    from app.memory.knowledge_memory import KnowledgeMemory
    from app.memory.session_memory import SessionMemory

    MemoryService = SessionMemory
except ImportError:
    MemoryService = None  # type: ignore[assignment,misc]

__all__ = [
    "ReportService",
    "LLMService",
    "MemoryService",
]
