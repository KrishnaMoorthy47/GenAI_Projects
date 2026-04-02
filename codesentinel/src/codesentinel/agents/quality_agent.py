from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from codesentinel.agents.state import ReviewState
from codesentinel.config import get_llm
from codesentinel.tools.diff_parser import parse_diff, summarize_diff

logger = logging.getLogger(__name__)

QUALITY_SYSTEM = """You are a senior software engineer conducting a thorough code quality review.
Analyze the provided code diff for code quality issues.

Evaluate:
1. **Complexity**: Overly complex logic, deeply nested conditionals, long functions (>50 lines)
2. **Naming**: Unclear variable/function names, abbreviations, misleading names
3. **Error handling**: Missing try/except, bare except clauses, swallowed exceptions
4. **Code duplication**: DRY violations, copy-pasted blocks
5. **Testing**: Missing test coverage for new code, untested edge cases
6. **Documentation**: Missing docstrings for public APIs, unclear comments
7. **Performance**: Obvious N+1 queries, inefficient loops, missing indexes
8. **Maintainability**: Tight coupling, violation of single responsibility, god classes

For each finding, rate severity as: high | medium | low | info
Be constructive and specific. Include line numbers where possible.
If the code is well-written, acknowledge it."""


def quality_agent_node(state: ReviewState) -> dict:
    """Analyze diff for code quality issues using LLM."""
    logger.info("quality_agent: reviewing %s PR #%d", state["repo"], state["pr_number"])

    diff = state.get("diff", "")
    if not diff:
        return {"quality_findings": []}

    file_diffs = parse_diff(diff)
    diff_summary = summarize_diff(file_diffs)

    # Build per-file analysis context
    file_context_parts = []
    for fd in file_diffs:
        if fd.additions == 0:
            continue
        file_context_parts.append(
            f"**File: {fd.filename}** ({fd.language})\n"
            f"Added: +{fd.additions}, Removed: -{fd.deletions}\n"
        )
        # Include added lines for context
        for line_no, content in fd.added_lines[:40]:
            file_context_parts.append(f"  +{line_no}: {content}")
        if len(fd.added_lines) > 40:
            file_context_parts.append(f"  ... ({len(fd.added_lines) - 40} more lines)")

    file_context = "\n".join(file_context_parts)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=QUALITY_SYSTEM),
        HumanMessage(
            content=(
                f"Repository: {state['repo']}\n"
                f"PR #{state['pr_number']}\n"
                f"Changed files: {', '.join(state.get('changed_files', []))}\n\n"
                f"Diff Summary:\n{diff_summary}\n\n"
                f"Code Changes:\n{file_context}\n\n"
                "Please provide a comprehensive code quality review."
            )
        ),
    ])

    llm_analysis = response.content if hasattr(response, "content") else str(response)

    findings: list[dict] = []
    if llm_analysis.strip():
        findings.append({
            "id": "LLM-QUALITY-ANALYSIS",
            "category": "Code Quality Review",
            "severity": "info",
            "description": llm_analysis,
            "file": None,
            "line": None,
            "snippet": None,
            "remediation": "See description for specific recommendations.",
        })

    logger.info("quality_agent: completed review for PR #%d", state["pr_number"])
    return {"quality_findings": findings}
