from __future__ import annotations

import logging

from langgraph.graph import StateGraph

from codesentinel.agents.quality_agent import quality_agent_node
from codesentinel.agents.security_agent import security_agent_node
from codesentinel.agents.state import ReviewState
from codesentinel.agents.summary_agent import summary_agent_node
from codesentinel.tools.diff_parser import get_changed_filenames, parse_diff

logger = logging.getLogger(__name__)


def parse_diff_node(state: ReviewState) -> dict:
    """Extract changed files from the diff for downstream agents."""
    file_diffs = parse_diff(state.get("diff", ""))
    changed_files = get_changed_filenames(file_diffs)
    logger.info(
        "parse_diff_node: %d files changed in %s PR #%d",
        len(changed_files),
        state["repo"],
        state["pr_number"],
    )
    return {"changed_files": changed_files}


def merge_findings_node(state: ReviewState) -> dict:
    """Barrier node — just passes through; LangGraph waits for both parallel agents."""
    total = len(state.get("security_findings", [])) + len(state.get("quality_findings", []))
    logger.info(
        "merge_findings_node: merged %d total findings for %s PR #%d",
        total,
        state["repo"],
        state["pr_number"],
    )
    # Return empty dict — just a synchronization barrier, no state changes needed
    return {}


def build_review_graph() -> StateGraph:
    """Build and compile the CodeSentinel review workflow."""
    builder = StateGraph(ReviewState)

    # Register nodes
    builder.add_node("parse_diff", parse_diff_node)
    builder.add_node("security_agent", security_agent_node)
    builder.add_node("quality_agent", quality_agent_node)
    builder.add_node("merge_findings", merge_findings_node)
    builder.add_node("summary_agent", summary_agent_node)

    # Entry point
    builder.set_entry_point("parse_diff")

    # parse_diff → parallel fan-out to both agents simultaneously
    builder.add_edge("parse_diff", "security_agent")
    builder.add_edge("parse_diff", "quality_agent")

    # Fan-in: both agents → merge_findings (LangGraph waits for both to complete)
    builder.add_edge("security_agent", "merge_findings")
    builder.add_edge("quality_agent", "merge_findings")

    # merge_findings → summary → END
    builder.add_edge("merge_findings", "summary_agent")
    builder.set_finish_point("summary_agent")

    graph = builder.compile()
    logger.info("CodeSentinel review graph compiled successfully")
    return graph


# Module-level singleton — build once
_review_graph = None


def get_review_graph():
    global _review_graph
    if _review_graph is None:
        _review_graph = build_review_graph()
    return _review_graph
