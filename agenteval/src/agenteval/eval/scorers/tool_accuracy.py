"""Deterministic scorer: recall of expected tools actually called."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def score_tool_accuracy(
    expected_tools: list[str],
    trajectory: list[dict],
) -> tuple[float, str]:
    """Return (score 0.0-1.0, reason string).

    Score = |expected ∩ called| / |expected|
    If expected_tools is empty, return 1.0 (nothing expected, nothing missing).
    """
    if not expected_tools:
        return 1.0, "No expected tools specified."

    called_tools = {step["name"] for step in trajectory if step.get("type") == "tool_call"}
    expected_set = set(expected_tools)
    intersection = expected_set & called_tools
    score = len(intersection) / len(expected_set)

    missing = expected_set - called_tools
    extra = called_tools - expected_set
    reason_parts = [f"Called: {sorted(called_tools)}", f"Expected: {sorted(expected_set)}"]
    if missing:
        reason_parts.append(f"Missing: {sorted(missing)}")
    if extra:
        reason_parts.append(f"Extra: {sorted(extra)}")

    return round(score, 3), " | ".join(reason_parts)
