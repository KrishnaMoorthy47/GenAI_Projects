from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


class TestGraphCompilation:
    def test_review_graph_compiles(self):
        """Review graph should compile without errors."""
        from codesentinel.agents.graph import build_review_graph

        graph = build_review_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Verify all expected nodes are present."""
        from codesentinel.agents.graph import build_review_graph

        graph = build_review_graph()
        node_names = set(graph.nodes.keys())
        expected = {"parse_diff", "security_agent", "quality_agent", "merge_findings", "summary_agent"}
        assert expected.issubset(node_names)


class TestSecurityAgent:
    def test_returns_findings_for_vulnerable_code(self, sample_diff):
        """Security agent should detect OWASP violations in vulnerable diff."""
        mock_response = AIMessage(
            content="Found SQL injection vulnerability on line 7 and hardcoded credential on line 11."
        )

        with patch("codesentinel.agents.security_agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            from codesentinel.agents.security_agent import security_agent_node

            state = {
                "repo": "test/repo",
                "pr_number": 1,
                "diff": sample_diff,
                "changed_files": [],
                "security_findings": [],
                "quality_findings": [],
                "final_review": None,
                "pr_url": "https://github.com/test/repo/pull/1",
                "mode": "api",
                "github_token": "",
            }

            result = security_agent_node(state)

        assert "security_findings" in result
        # Should have static findings (SQL injection + hardcoded secret) + LLM analysis
        assert len(result["security_findings"]) >= 2

    def test_returns_empty_for_no_diff(self):
        """Security agent should return empty findings for empty diff."""
        from codesentinel.agents.security_agent import security_agent_node

        state = {
            "repo": "test/repo",
            "pr_number": 1,
            "diff": "",
            "changed_files": [],
            "security_findings": [],
            "quality_findings": [],
            "final_review": None,
            "pr_url": "",
            "mode": "api",
            "github_token": "",
        }

        result = security_agent_node(state)
        assert result["security_findings"] == []


class TestWebhookSignature:
    def test_valid_signature(self):
        import hashlib
        import hmac

        from codesentinel.services.github_service import verify_webhook_signature

        secret = "test-secret"
        payload = b'{"action": "opened"}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        with patch("codesentinel.services.github_service.get_settings") as mock_settings:
            mock_settings.return_value.github_webhook_secret = secret
            assert verify_webhook_signature(payload, sig) is True

    def test_invalid_signature(self):
        from codesentinel.services.github_service import verify_webhook_signature

        with patch("codesentinel.services.github_service.get_settings") as mock_settings:
            mock_settings.return_value.github_webhook_secret = "secret"
            assert verify_webhook_signature(b"payload", "sha256=invalidsig") is False

    def test_missing_prefix_rejected(self):
        from codesentinel.services.github_service import verify_webhook_signature

        with patch("codesentinel.services.github_service.get_settings") as mock_settings:
            mock_settings.return_value.github_webhook_secret = "secret"
            assert verify_webhook_signature(b"payload", "invalidsig") is False
