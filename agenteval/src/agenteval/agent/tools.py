"""SQL tools for the agent: list_tables, get_schema, run_query."""

import logging
import sqlite3
from typing import Optional

from langchain_core.tools import tool

from agenteval.config import get_settings

logger = logging.getLogger(__name__)

# Module-level DB path (overridable in tests)
_db_path: Optional[str] = None


def set_db_path(path: str) -> None:
    """Override DB path (used in tests)."""
    global _db_path
    _db_path = path


def _get_db_path() -> str:
    global _db_path
    return _db_path or get_settings().db_path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


@tool
def list_tables() -> str:
    """List all tables available in the database."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        tables = [r["name"] for r in rows]
        return "Tables: " + ", ".join(tables)
    except Exception as e:
        logger.error("list_tables error: %s", e)
        return f"Error listing tables: {e}"


@tool
def get_schema(table_name: str) -> str:
    """Get the CREATE TABLE schema for a specific table.

    Args:
        table_name: Name of the table to inspect.
    """
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
        if row is None:
            return f"Table '{table_name}' not found."
        return row["sql"]
    except Exception as e:
        logger.error("get_schema error: %s", e)
        return f"Error getting schema for '{table_name}': {e}"


@tool
def run_query(query: str) -> str:
    """Execute a read-only SQL SELECT query and return results as a string.

    Args:
        query: A valid SQLite SELECT statement.
    """
    # Strip leading whitespace and comments before checking query type
    import re as _re
    q_normalized = _re.sub(r"(--[^\n]*\n?|/\*.*?\*/)", "", query, flags=_re.DOTALL).strip().upper()
    if not q_normalized.startswith("SELECT") and not q_normalized.startswith("WITH"):
        return "Error: Only SELECT queries are allowed."
    try:
        with _connect() as conn:
            rows = conn.execute(query).fetchmany(50)
        if not rows:
            return "No results."
        cols = rows[0].keys()
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        logger.error("run_query error: %s", e)
        return f"SQL error: {e}"
