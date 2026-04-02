from __future__ import annotations

import pytest

from codesentinel.tools.diff_parser import (
    FileDiff,
    detect_language,
    get_changed_filenames,
    parse_diff,
    summarize_diff,
)


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("app/main.py") == "python"

    def test_typescript(self):
        assert detect_language("src/components/Button.tsx") == "typescript"

    def test_javascript(self):
        assert detect_language("index.js") == "javascript"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "dockerfile"

    def test_unknown_extension(self):
        assert detect_language("somefile.xyz") == "unknown"

    def test_no_extension(self):
        assert detect_language("Makefile") == "unknown"


class TestParseDiff:
    def test_parse_empty_diff(self):
        assert parse_diff("") == []
        assert parse_diff("   ") == []

    def test_parse_single_file(self, sample_diff):
        result = parse_diff(sample_diff)
        assert len(result) == 1
        assert result[0].filename == "app/api/users.py"
        assert result[0].language == "python"
        assert result[0].status == "modified"
        assert result[0].additions > 0

    def test_parse_added_lines(self, sample_diff):
        result = parse_diff(sample_diff)
        file_diff = result[0]
        # Should detect the SQL injection line
        added_contents = [line for _, line in file_diff.added_lines]
        assert any("cursor.execute" in line for line in added_contents)
        assert any("hardcoded_secret" in line for line in added_contents)

    def test_added_file_status(self):
        diff = '''diff --git a/newfile.py b/newfile.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,3 @@
+def hello():
+    return "world"
+
'''
        result = parse_diff(diff)
        assert len(result) == 1
        assert result[0].status == "added"
        assert result[0].filename == "newfile.py"

    def test_get_changed_filenames(self, sample_diff):
        file_diffs = parse_diff(sample_diff)
        filenames = get_changed_filenames(file_diffs)
        assert "app/api/users.py" in filenames


class TestOwaspPatterns:
    def test_detects_sql_injection(self, sample_diff):
        from codesentinel.tools.owasp_patterns import scan_diff_chunk

        file_diffs = parse_diff(sample_diff)
        all_findings = []
        for fd in file_diffs:
            all_findings.extend(scan_diff_chunk(fd.diff_chunk, fd.filename))

        ids = [f["id"] for f in all_findings]
        assert "OWASP-A01-SQL-001" in ids

    def test_detects_hardcoded_secret(self, sample_diff):
        from codesentinel.tools.owasp_patterns import scan_diff_chunk

        file_diffs = parse_diff(sample_diff)
        all_findings = []
        for fd in file_diffs:
            all_findings.extend(scan_diff_chunk(fd.diff_chunk, fd.filename))

        ids = [f["id"] for f in all_findings]
        assert "OWASP-A02-HARDCODED-SECRET-001" in ids

    def test_clean_code_no_findings(self, clean_diff):
        from codesentinel.tools.owasp_patterns import scan_diff_chunk

        file_diffs = parse_diff(clean_diff)
        all_findings = []
        for fd in file_diffs:
            all_findings.extend(scan_diff_chunk(fd.diff_chunk, fd.filename))

        # Clean code should have no OWASP violations
        assert len(all_findings) == 0

    def test_finding_has_required_fields(self, sample_diff):
        from codesentinel.tools.owasp_patterns import scan_diff_chunk

        file_diffs = parse_diff(sample_diff)
        findings = []
        for fd in file_diffs:
            findings.extend(scan_diff_chunk(fd.diff_chunk, fd.filename))

        for f in findings:
            assert "id" in f
            assert "severity" in f
            assert "description" in f
            assert "remediation" in f
            assert "file" in f
            assert "line" in f


class TestSummarizeDiff:
    def test_empty(self):
        assert "No files changed" in summarize_diff([])

    def test_includes_filenames(self, sample_diff):
        file_diffs = parse_diff(sample_diff)
        summary = summarize_diff(file_diffs)
        assert "app/api/users.py" in summary

    def test_includes_stats(self, sample_diff):
        file_diffs = parse_diff(sample_diff)
        summary = summarize_diff(file_diffs)
        assert "Files changed" in summary
        assert "additions" in summary
