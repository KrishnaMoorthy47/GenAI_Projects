from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class OwaspPattern:
    id: str
    category: str
    severity: str  # "critical" | "high" | "medium" | "low"
    pattern: re.Pattern
    description: str
    remediation: str


# OWASP Top 10 regex patterns
OWASP_PATTERNS: list[OwaspPattern] = [
    # A01 — Injection
    OwaspPattern(
        id="OWASP-A01-SQL-001",
        category="A01:2021 – Injection",
        severity="critical",
        pattern=re.compile(
            r'(execute|query|cursor\.execute)\s*\(\s*[f"\'].*?(SELECT|INSERT|UPDATE|DELETE|DROP)',
            re.IGNORECASE,
        ),
        description="Potential SQL injection via string formatting/f-strings in query.",
        remediation="Use parameterized queries or an ORM. Never interpolate user input into SQL.",
    ),
    OwaspPattern(
        id="OWASP-A01-CMD-001",
        category="A01:2021 – Injection",
        severity="critical",
        pattern=re.compile(
            r'(os\.system|subprocess\.(call|run|Popen))\s*\([^)]*\+[^)]*\)',
            re.IGNORECASE,
        ),
        description="Potential command injection — dynamic string concatenation in shell command.",
        remediation="Use subprocess with a list of arguments (never shell=True with user input). Validate/sanitize all inputs.",
    ),
    # A02 — Cryptographic Failures
    OwaspPattern(
        id="OWASP-A02-HARDCODED-SECRET-001",
        category="A02:2021 – Cryptographic Failures",
        severity="critical",
        pattern=re.compile(
            r'(password|secret|api_key|apikey|token|passwd|pwd)\s*=\s*["\'][^"\']{6,}["\']',
            re.IGNORECASE,
        ),
        description="Potential hardcoded secret, password, or API key in source code.",
        remediation="Move secrets to environment variables or a secrets manager (e.g. HashiCorp Vault, AWS Secrets Manager).",
    ),
    OwaspPattern(
        id="OWASP-A02-WEAK-HASH-001",
        category="A02:2021 – Cryptographic Failures",
        severity="high",
        pattern=re.compile(r'\b(md5|sha1)\b', re.IGNORECASE),
        description="Use of weak hashing algorithm (MD5 or SHA-1).",
        remediation="Use SHA-256 or stronger. For passwords, use bcrypt, argon2, or scrypt.",
    ),
    # A03 — XSS
    OwaspPattern(
        id="OWASP-A03-XSS-001",
        category="A03:2021 – Injection (XSS)",
        severity="high",
        pattern=re.compile(r'innerHTML\s*=\s*[^"\';\n]*\+', re.IGNORECASE),
        description="Potential XSS — user-controlled data assigned to innerHTML.",
        remediation="Use textContent instead of innerHTML, or sanitize with DOMPurify before assignment.",
    ),
    # A05 — Security Misconfiguration
    OwaspPattern(
        id="OWASP-A05-DEBUG-001",
        category="A05:2021 – Security Misconfiguration",
        severity="medium",
        pattern=re.compile(r'DEBUG\s*=\s*True', re.IGNORECASE),
        description="Debug mode enabled — may expose stack traces and sensitive data in production.",
        remediation="Set DEBUG=False in production and use environment-specific configuration.",
    ),
    OwaspPattern(
        id="OWASP-A05-CORS-001",
        category="A05:2021 – Security Misconfiguration",
        severity="medium",
        pattern=re.compile(r'allow_origins\s*=\s*\[?\s*["\']?\*["\']?\s*\]?', re.IGNORECASE),
        description="CORS wildcard origin — allows any domain to make cross-origin requests.",
        remediation="Restrict CORS origins to known trusted domains.",
    ),
    # A07 — Identification and Authentication Failures
    OwaspPattern(
        id="OWASP-A07-AUTH-001",
        category="A07:2021 – Identification and Authentication Failures",
        severity="high",
        pattern=re.compile(
            r'verify\s*=\s*False',
            re.IGNORECASE,
        ),
        description="SSL/TLS certificate verification disabled.",
        remediation="Never disable SSL verification in production. Pass a CA bundle if using custom certs.",
    ),
    # A09 — Security Logging and Monitoring Failures
    OwaspPattern(
        id="OWASP-A09-LOGGING-001",
        category="A09:2021 – Security Logging and Monitoring Failures",
        severity="low",
        pattern=re.compile(
            r'(print|console\.log)\s*\([^)]*?(password|secret|token|key)[^)]*\)',
            re.IGNORECASE,
        ),
        description="Potential sensitive data logged via print/console.log.",
        remediation="Use a structured logger with log level control. Never log secrets or credentials.",
    ),
]


def scan_line(line: str, line_number: int, filename: str) -> list[dict]:
    """Scan a single line against all OWASP patterns. Returns matched findings."""
    findings = []
    for pattern in OWASP_PATTERNS:
        if pattern.pattern.search(line):
            findings.append({
                "id": pattern.id,
                "category": pattern.category,
                "severity": pattern.severity,
                "description": pattern.description,
                "remediation": pattern.remediation,
                "file": filename,
                "line": line_number,
                "snippet": line.strip()[:200],
            })
    return findings


def scan_diff_chunk(diff_chunk: str, filename: str) -> list[dict]:
    """Scan an entire diff chunk (added lines only) for OWASP patterns."""
    findings = []
    line_number = 0

    for line in diff_chunk.splitlines():
        # Track line numbers from diff hunks
        if line.startswith("@@"):
            # e.g., "@@ -10,6 +10,8 @@" — extract the target line start
            match = re.search(r'\+(\d+)', line)
            if match:
                line_number = int(match.group(1)) - 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            # This is an added line
            line_number += 1
            actual_line = line[1:]  # strip the leading "+"
            findings.extend(scan_line(actual_line, line_number, filename))
        elif not line.startswith("-"):
            line_number += 1

    return findings
