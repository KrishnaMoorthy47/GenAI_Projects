from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from codesentinel.agents.state import ReviewState
from codesentinel.config import get_llm
from codesentinel.tools.diff_parser import parse_diff, summarize_diff
from codesentinel.tools.owasp_patterns import scan_diff_chunk

logger = logging.getLogger(__name__)

SECURITY_SYSTEM = """You are an expert application security engineer specializing in code review.
Your job is to identify security vulnerabilities in the provided code diff.

Focus on:
- OWASP Top 10 vulnerabilities (injection, broken auth, XSS, IDOR, security misconfig, etc.)
- Hardcoded secrets, API keys, or credentials
- Insecure cryptography (weak hashing, broken encryption)
- Input validation failures
- Insecure deserialization
- Dependency vulnerabilities (if package files changed)

For each finding, provide:
- severity: critical | high | medium | low | info
- description: Clear explanation of the vulnerability
- file and line (if identifiable)
- remediation: Specific fix recommendation

If no security issues are found, say so clearly. Be precise — avoid false positives."""


def security_agent_node(state: ReviewState) -> dict:
    """Scan diff for security vulnerabilities using OWASP patterns + LLM analysis."""
    logger.info("security_agent: reviewing %s PR #%d", state["repo"], state["pr_number"])

    diff = state.get("diff", "")
    if not diff:
        return {"security_findings": []}

    # Phase 1: Static regex-based OWASP scan (fast, deterministic)
    file_diffs = parse_diff(diff)
    static_findings: list[dict] = []
    for fd in file_diffs:
        chunk_findings = scan_diff_chunk(fd.diff_chunk, fd.filename)
        static_findings.extend(chunk_findings)

    # Phase 2: LLM-based semantic security analysis
    llm = get_llm()
    diff_summary = summarize_diff(file_diffs)

    # Truncate diff for LLM context (max ~8000 chars)
    _MAX_DIFF_CHARS = 8000
    if len(diff) > _MAX_DIFF_CHARS:
        diff_excerpt = diff[:_MAX_DIFF_CHARS]
        static_findings.append({
            "id": "DIFF-TRUNCATED",
            "category": "Review Coverage",
            "severity": "info",
            "description": (
                f"Diff truncated to {_MAX_DIFF_CHARS} characters "
                f"({len(diff) - _MAX_DIFF_CHARS} chars omitted). "
                "Review remaining files manually."
            ),
            "file": None,
            "line": None,
            "snippet": None,
            "remediation": "Review the full diff locally for any issues beyond the truncation point.",
        })
    else:
        diff_excerpt = diff

    response = llm.invoke([
        SystemMessage(content=SECURITY_SYSTEM),
        HumanMessage(
            content=(
                f"Repository: {state['repo']}\n"
                f"PR #{state['pr_number']}\n\n"
                f"Diff Summary:\n{diff_summary}\n\n"
                f"Full Diff (truncated if large):\n```diff\n{diff_excerpt}\n```\n\n"
                f"Static analysis already detected these potential issues:\n"
                + "\n".join(
                    f"- [{f['severity'].upper()}] {f['id']}: {f['description']} (File: {f['file']}, Line: {f['line']})"
                    for f in static_findings
                )
                + "\n\nPlease provide a comprehensive security review with any additional issues found."
            )
        ),
    ])

    llm_analysis = response.content if hasattr(response, "content") else str(response)

    # Merge static + LLM findings
    all_findings = list(static_findings)
    if llm_analysis.strip():
        all_findings.append({
            "id": "LLM-SECURITY-ANALYSIS",
            "category": "LLM Security Review",
            "severity": "info",
            "description": llm_analysis,
            "file": None,
            "line": None,
            "snippet": None,
            "remediation": "See description for specific recommendations.",
        })

    logger.info(
        "security_agent: found %d findings (%d static) for PR #%d",
        len(all_findings),
        len(static_findings),
        state["pr_number"],
    )

    return {"security_findings": all_findings}
