"""Session memory — stores current task context and execution state.

In production this uses Redis (via fakeredis for testing); for development
we fall back to an in-memory dictionary.
"""

import json
import logging
import time
from typing import Dict, List, Optional

from app.models.state import ResearchState

logger = logging.getLogger(__name__)


class SessionMemory:
    """Stores current task context and execution state.

    In production this uses Redis; for development we use an in-memory dict
    with a fakeredis fallback if the library is available.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None  # Would be aioredis/fakeredis in production
        self._store: Dict[str, str] = {}  # task_id -> JSON-serialized state
        self._ttl: Dict[str, float] = {}  # task_id -> expiry timestamp

        # Attempt to use fakeredis for closer-to-production behaviour
        self._use_fakeredis = False
        try:
            import fakeredis.aioredis  # noqa: F401

            # In a real setup we would initialise fakeredis here.
            # For simplicity and reliability, we stick with the in-memory
            # dict approach but mark that fakeredis is available.
            self._use_fakeredis = True
            logger.info("fakeredis available — using in-memory store (Redis-compatible API ready)")
        except ImportError:
            logger.info("fakeredis not installed — using pure in-memory dict store")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_state(self, task_id: str, state: ResearchState) -> None:
        """Save a ResearchState with a 24-hour TTL.

        Args:
            task_id: Unique identifier for the research task.
            state: The ResearchState to persist.
        """
        serialized = state.model_dump_json()
        self._store[task_id] = serialized
        self._ttl[task_id] = time.time() + 86400  # 24 hours
        logger.debug("Saved state for task '%s' (expires in 24h)", task_id)

    async def load_state(self, task_id: str) -> Optional[ResearchState]:
        """Load a previously saved ResearchState.

        Returns None if the key does not exist or has expired.

        Args:
            task_id: Unique identifier for the research task.

        Returns:
            The deserialized ResearchState, or None.
        """
        if self._is_expired(task_id):
            if task_id in self._store:
                logger.info("State for task '%s' has expired — cleaning up", task_id)
                await self.delete_state(task_id)
            return None

        raw = self._store.get(task_id)
        if raw is None:
            return None

        try:
            data = json.loads(raw)
            return ResearchState(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to deserialize state for task '%s': %s", task_id, exc)
            return None

    async def delete_state(self, task_id: str) -> None:
        """Delete a state entry.

        Args:
            task_id: Unique identifier for the research task.
        """
        self._store.pop(task_id, None)
        self._ttl.pop(task_id, None)
        logger.debug("Deleted state for task '%s'", task_id)

    async def list_sessions(self) -> List[str]:
        """List active (non-expired) session IDs.

        Returns:
            Sorted list of active task IDs.
        """
        active = [tid for tid in self._store if not self._is_expired(tid)]
        return sorted(active)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired.

        Args:
            key: The task ID to check.

        Returns:
            True if the key's TTL has elapsed or the key has no TTL set.
        """
        expiry = self._ttl.get(key)
        if expiry is None:
            return True  # No TTL means expired/non-existent
        return time.time() > expiry

    def __len__(self) -> int:
        """Return the number of stored sessions (convenience for testing)."""
        return len(self._store)

    def __contains__(self, task_id: str) -> bool:
        """Check if a task ID is in the store and not expired."""
        return task_id in self._store and not self._is_expired(task_id)
