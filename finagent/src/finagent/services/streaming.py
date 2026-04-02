from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from langgraph.errors import GraphInterrupt

logger = logging.getLogger(__name__)

# Registry: thread_id → asyncio.Queue
_queues: dict[str, asyncio.Queue] = {}

# Status tracking: thread_id → status string
_status: dict[str, str] = {}

# Timestamp tracking for TTL-based cleanup: thread_id → creation time
_created_at: dict[str, float] = {}

# Maximum age of a completed/failed queue before pruning (1 hour)
_QUEUE_TTL_SECONDS = 3600


def create_stream_queue(thread_id: str) -> asyncio.Queue:
    """Create and register a new streaming queue for a thread."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _queues[thread_id] = q
    _status[thread_id] = "running"
    _created_at[thread_id] = time.monotonic()
    return q


def prune_stale_queues() -> int:
    """Remove completed/failed queues older than _QUEUE_TTL_SECONDS. Returns count pruned."""
    now = time.monotonic()
    stale = [
        tid for tid, ts in _created_at.items()
        if now - ts > _QUEUE_TTL_SECONDS and _status.get(tid) in ("completed", "failed", "awaiting_approval")
    ]
    for tid in stale:
        _queues.pop(tid, None)
        _status.pop(tid, None)
        _created_at.pop(tid, None)
    if stale:
        logger.info("Pruned %d stale stream queues", len(stale))
    return len(stale)


def get_stream_queue(thread_id: str) -> asyncio.Queue | None:
    return _queues.get(thread_id)


def get_status(thread_id: str) -> str | None:
    return _status.get(thread_id)


def set_status(thread_id: str, status: str) -> None:
    _status[thread_id] = status


async def run_graph_with_streaming(
    graph,
    initial_state: dict,
    config: dict,
    thread_id: str,
) -> None:
    """Run the LangGraph graph and push events into the asyncio.Queue."""
    q = _queues.get(thread_id)
    if q is None:
        logger.error("No queue found for thread_id=%s", thread_id)
        return

    try:
        set_status(thread_id, "running")
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            # Node completion events
            if event_type == "on_chain_end" and event_name in {
                "web_research", "financial_data", "sentiment", "report_writer", "supervisor"
            }:
                await q.put(json.dumps({
                    "type": "node_complete",
                    "node": event_name,
                }))

            # LLM token streaming
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    await q.put(json.dumps({
                        "type": "token",
                        "content": chunk.content,
                    }))

        # After the stream ends, check if the graph paused at an interrupt.
        # In LangGraph 1.0.10+, interrupt() sets snapshot.interrupts (not snapshot.next).
        # snapshot.next is empty after interrupt() because the node already started executing.
        if get_status(thread_id) == "running":
            try:
                snapshot = await graph.aget_state(config)
                logger.info(
                    "snapshot for thread_id=%s: next=%s interrupts=%s tasks=%d",
                    thread_id, snapshot.next, snapshot.interrupts, len(snapshot.tasks),
                )
                if snapshot and snapshot.interrupts:
                    set_status(thread_id, "awaiting_approval")
                    await q.put(json.dumps({
                        "type": "awaiting_approval",
                        "message": "Research complete. Awaiting human approval to finalize report.",
                    }))
                else:
                    set_status(thread_id, "completed")
                    await q.put(json.dumps({"type": "complete", "message": "Research completed"}))
            except Exception as snap_exc:
                logger.error(
                    "aget_state failed for thread_id=%s — sending complete as fallback: %s",
                    thread_id, snap_exc,
                )
                set_status(thread_id, "completed")
                await q.put(json.dumps({"type": "complete", "message": "Research completed"}))

    except GraphInterrupt:
        # Fallback: future LangGraph versions may propagate instead of suppress GraphInterrupt
        set_status(thread_id, "awaiting_approval")
        await q.put(json.dumps({
            "type": "awaiting_approval",
            "message": "Research complete. Awaiting human approval to finalize report.",
        }))
    except asyncio.CancelledError:
        logger.info("Graph streaming cancelled for thread_id=%s", thread_id)
        set_status(thread_id, "failed")
        raise
    except Exception as exc:
        logger.error("Graph streaming error for thread_id=%s: %s", thread_id, exc)
        set_status(thread_id, "failed")
        await q.put(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        # Sentinel to signal stream end
        await q.put(None)
        # Opportunistically prune old queues
        prune_stale_queues()


async def sse_generator(thread_id: str) -> AsyncIterator[dict]:
    """Yield Server-Sent Events from the queue. Sends heartbeat every 30s."""
    q = _queues.get(thread_id)
    if q is None:
        yield {"data": json.dumps({"type": "error", "message": "No stream found for this thread_id"})}
        return

    heartbeat_interval = 30.0

    while True:
        try:
            item = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
        except asyncio.TimeoutError:
            # Send heartbeat comment to keep connection alive
            yield {"comment": "heartbeat"}
            continue

        if item is None:
            # Sentinel: stream ended
            yield {"data": json.dumps({"type": "done"})}
            break

        yield {"data": item}
