"""Deterministic scorer: trajectory efficiency (penalise extra steps)."""

import logging

logger = logging.getLogger(__name__)

PENALTY_PER_EXTRA_STEP = 0.1


def score_trajectory(
    trajectory: list[dict],
    min_steps: int,
) -> tuple[float, str]:
    """Return (score 0.0-1.0, reason string).

    Counts tool_call steps only.
    score = 1.0 - 0.1 × max(0, actual_steps - min_steps)
    Clamped to [0.0, 1.0].
    """
    actual_steps = sum(1 for s in trajectory if s.get("type") == "tool_call")
    extra = max(0, actual_steps - min_steps)
    score = max(0.0, 1.0 - PENALTY_PER_EXTRA_STEP * extra)
    reason = f"Tool calls: {actual_steps} (min expected: {min_steps}, extra: {extra})"
    return round(score, 3), reason
