"""Tests for task-local event callback isolation."""

import asyncio

import pytest

from app.workflow.events import emit, set_event_callback


def test_emit_without_callback_is_noop():
    """Emitting with no active callback must not raise."""
    emit("agent_status", agent="X")


@pytest.mark.asyncio
async def test_callbacks_are_task_local():
    """Two concurrent tasks must only receive their own events."""
    received_a: list = []
    received_b: list = []

    async def worker_a():
        set_event_callback(lambda t, d: received_a.append((t, d)))
        emit("agent_status", agent="A")
        await asyncio.sleep(0.05)
        emit("agent_result", agent="A")

    async def worker_b():
        set_event_callback(lambda t, d: received_b.append((t, d)))
        emit("agent_status", agent="B")
        await asyncio.sleep(0.05)
        emit("agent_result", agent="B")

    await asyncio.gather(worker_a(), worker_b())

    assert [t for t, _ in received_a] == ["agent_status", "agent_result"]
    assert [t for t, _ in received_b] == ["agent_status", "agent_result"]
    assert all(d["agent"] == "A" for _, d in received_a)
    assert all(d["agent"] == "B" for _, d in received_b)


@pytest.mark.asyncio
async def test_callback_does_not_leak_across_tasks():
    """A callback set in one task must not be visible in another."""
    from app.workflow.events import _active_callback

    async def setter():
        set_event_callback(lambda t, d: None)

    async def checker():
        await asyncio.sleep(0.05)
        assert _active_callback.get() is None

    await asyncio.gather(setter(), checker())
