from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

audit_logger = logging.getLogger("finagent.audit")
audit_logger.setLevel(logging.INFO)
# Bare-message formatter + no propagation: keeps each line valid standalone JSON,
# independent of the root logger's "%(asctime)s %(levelname)s ..." prefix format.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(_handler)
audit_logger.propagate = False


def log_research_request(
    *,
    thread_id: str,
    ticker: str,
    query_length: int,
    guard_flagged: bool,
    guard_reason: str = "",
) -> None:
    """Write one structured JSON audit line per research request."""
    audit_logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "thread_id": thread_id,
                "ticker": ticker,
                "query_length": query_length,
                "guard_flagged": guard_flagged,
                "guard_reason": guard_reason,
            }
        )
    )
