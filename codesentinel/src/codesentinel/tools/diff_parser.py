from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    filename: str
    old_filename: str
    status: str  # "added" | "modified" | "deleted" | "renamed"
    language: str
    additions: int
    deletions: int
    diff_chunk: str  # Raw diff text for this file
    added_lines: list[tuple[int, str]] = field(default_factory=list)  # (line_no, content)


def detect_language(filename: str) -> str:
    """Detect programming language from file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    language_map = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "java": "java",
        "go": "go",
        "rb": "ruby",
        "php": "php",
        "cs": "csharp",
        "cpp": "cpp",
        "c": "c",
        "h": "c",
        "rs": "rust",
        "kt": "kotlin",
        "swift": "swift",
        "sh": "shell",
        "bash": "shell",
        "yaml": "yaml",
        "yml": "yaml",
        "json": "json",
        "sql": "sql",
        "tf": "terraform",
        "dockerfile": "dockerfile",
    }
    if filename.lower() == "dockerfile":
        return "dockerfile"
    return language_map.get(ext, "unknown")


def parse_diff(raw_diff: str) -> list[FileDiff]:
    """Parse a raw git diff string into a list of FileDiff objects."""
    if not raw_diff or not raw_diff.strip():
        return []

    file_diffs: list[FileDiff] = []

    # Split on "diff --git" boundaries
    file_sections = re.split(r'(?=^diff --git )', raw_diff, flags=re.MULTILINE)

    for section in file_sections:
        if not section.strip() or not section.startswith("diff --git"):
            continue

        lines = section.splitlines()

        # Extract filenames from "diff --git a/foo b/bar"
        header_match = re.match(r'diff --git a/(.*?) b/(.*?)$', lines[0])
        if not header_match:
            continue

        old_filename = header_match.group(1)
        new_filename = header_match.group(2)

        # Determine status
        status = "modified"
        for line in lines[1:6]:
            if line.startswith("new file"):
                status = "added"
            elif line.startswith("deleted file"):
                status = "deleted"
            elif line.startswith("rename"):
                status = "renamed"

        # Count additions and deletions
        additions = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
        deletions = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))

        # Extract added lines with line numbers
        added_lines: list[tuple[int, str]] = []
        current_line = 0
        for line in lines:
            if line.startswith("@@"):
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1)) - 1
            elif line.startswith("+") and not line.startswith("+++"):
                current_line += 1
                added_lines.append((current_line, line[1:]))
            elif not line.startswith("-"):
                current_line += 1

        file_diff = FileDiff(
            filename=new_filename,
            old_filename=old_filename,
            status=status,
            language=detect_language(new_filename),
            additions=additions,
            deletions=deletions,
            diff_chunk=section,
            added_lines=added_lines,
        )
        file_diffs.append(file_diff)

    return file_diffs


def get_changed_filenames(file_diffs: list[FileDiff]) -> list[str]:
    return [fd.filename for fd in file_diffs if fd.status != "deleted"]


def summarize_diff(file_diffs: list[FileDiff]) -> str:
    """Create a human-readable summary of the diff for LLM context."""
    if not file_diffs:
        return "No files changed."

    total_additions = sum(fd.additions for fd in file_diffs)
    total_deletions = sum(fd.deletions for fd in file_diffs)

    lines = [
        f"Files changed: {len(file_diffs)}",
        f"Total additions: +{total_additions}, deletions: -{total_deletions}",
        "",
    ]

    for fd in file_diffs:
        lines.append(f"[{fd.status.upper()}] {fd.filename} ({fd.language}) +{fd.additions}/-{fd.deletions}")
        # Include up to 50 added lines per file for LLM context
        for line_no, content in fd.added_lines[:50]:
            lines.append(f"  +{line_no}: {content}")
        if len(fd.added_lines) > 50:
            lines.append(f"  ... ({len(fd.added_lines) - 50} more added lines)")

    return "\n".join(lines)
