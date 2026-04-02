"""Integration tests for the eval runner and reporter."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agenteval.eval.reporter import aggregate_results, compute_overall
from agenteval.models.response import CaseResult, EvalReport, ScoreCard
from tests.conftest import make_mock_llm


# ──────────────────────────────────────────────
# Reporter tests
# ──────────────────────────────────────────────

class TestReporter:
    def _make_case(self, ts=0.8, ta=0.9, te=1.0, hall=0.9) -> CaseResult:
        scores = ScoreCard(
            task_success=ts,
            tool_accuracy=ta,
            trajectory_efficiency=te,
            hallucination=hall,
            overall=0.0,
        )
        scores.overall = compute_overall(scores)
        return CaseResult(
            id="tc_001",
            question="How many songs?",
            expected_answer="92",
            actual_answer="92 songs.",
            scores=scores,
            difficulty="easy",
            tags=["count"],
        )

    def test_compute_overall_weights(self):
        scores = ScoreCard(
            task_success=1.0,
            tool_accuracy=1.0,
            trajectory_efficiency=1.0,
            hallucination=1.0,
            overall=0.0,
        )
        assert compute_overall(scores) == 1.0

    def test_compute_overall_zero(self):
        scores = ScoreCard(
            task_success=0.0,
            tool_accuracy=0.0,
            trajectory_efficiency=0.0,
            hallucination=0.0,
            overall=0.0,
        )
        assert compute_overall(scores) == 0.0

    def test_aggregate_empty(self):
        report = aggregate_results("run-1", [])
        assert report.status == "completed"
        assert report.total_cases == 0
        assert report.aggregate_scores is None

    def test_aggregate_single_case(self):
        case = self._make_case(ts=1.0, ta=1.0, te=1.0, hall=1.0)
        report = aggregate_results("run-1", [case])
        assert report.aggregate_scores.task_success == 1.0
        assert report.aggregate_scores.overall == 1.0

    def test_aggregate_multiple_cases(self):
        cases = [
            self._make_case(ts=1.0, ta=1.0, te=1.0, hall=1.0),
            self._make_case(ts=0.0, ta=0.0, te=0.0, hall=0.0),
        ]
        report = aggregate_results("run-1", cases)
        assert abs(report.aggregate_scores.task_success - 0.5) < 0.01
        assert abs(report.aggregate_scores.overall - 0.5) < 0.01


# ──────────────────────────────────────────────
# Dataset loading tests
# ──────────────────────────────────────────────

class TestDatasetLoader:
    def test_load_cases_from_file(self, tmp_path):
        from agenteval.eval.dataset import load_cases

        cases = [
            {
                "id": "tc_001",
                "question": "How many songs?",
                "expected_answer": "92",
                "expected_tools": ["list_tables", "run_query"],
                "min_steps": 2,
                "difficulty": "easy",
                "tags": ["count"],
            }
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(cases))

        loaded = load_cases(str(path))
        assert len(loaded) == 1
        assert loaded[0]["id"] == "tc_001"

    def test_load_cases_missing_file(self):
        from agenteval.eval.dataset import load_cases

        with pytest.raises(FileNotFoundError):
            load_cases("/nonexistent/path/cases.json")


# ──────────────────────────────────────────────
# EvalRunner integration test
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_runner_full_run(mock_db, tmp_path):
    """Full eval run with mocked LLM and 2-case dataset."""
    import json as _json
    from agenteval.agent import tools as tools_module
    from agenteval.eval.runner import EvalRunner

    tools_module.set_db_path(mock_db)

    # Minimal 2-case dataset
    cases = [
        {
            "id": "tc_001",
            "question": "How many songs?",
            "expected_answer": "6",
            "expected_tools": ["list_tables", "run_query"],
            "min_steps": 2,
            "difficulty": "easy",
            "tags": ["count"],
        },
        {
            "id": "tc_002",
            "question": "How many composers?",
            "expected_answer": "3",
            "expected_tools": ["list_tables", "run_query"],
            "min_steps": 2,
            "difficulty": "easy",
            "tags": ["count"],
        },
    ]
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(_json.dumps(cases))

    # Mock LLM for agent + LLM-as-judge
    mock_llm = make_mock_llm(final_answer="There are 6 songs.")

    report_store = {}
    run_id = "test-run-001"
    report_store[run_id] = EvalReport(run_id=run_id, status="pending")

    runner = EvalRunner(
        report_store=report_store,
        dataset_path=str(dataset_path),
        llm=mock_llm,
    )

    # Patch LangSmith push to avoid real API call
    with patch("agenteval.eval.runner.push_to_langsmith", return_value=None):
        with patch("agenteval.eval.scorers.task_success.get_llm", return_value=mock_llm):
            with patch("agenteval.eval.scorers.hallucination.get_llm", return_value=mock_llm):
                await runner.run(run_id=run_id, push_langsmith=False)

    report = report_store[run_id]
    assert report.status == "completed"
    assert report.total_cases == 2
    assert report.completed_cases == 2
    assert report.aggregate_scores is not None
    assert 0.0 <= report.aggregate_scores.overall <= 1.0


@pytest.mark.asyncio
async def test_eval_runner_handles_agent_error(tmp_path):
    """EvalRunner should not crash if agent raises for one case."""
    import json as _json
    from agenteval.eval.runner import EvalRunner

    cases = [
        {
            "id": "tc_001",
            "question": "Will fail",
            "expected_answer": "N/A",
            "expected_tools": [],
            "min_steps": 1,
            "difficulty": "easy",
            "tags": [],
        }
    ]
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(_json.dumps(cases))

    # Mock LLM that raises
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

    report_store = {}
    run_id = "test-run-002"
    report_store[run_id] = EvalReport(run_id=run_id, status="pending")

    runner = EvalRunner(
        report_store=report_store,
        dataset_path=str(dataset_path),
        llm=mock_llm,
    )

    with patch("agenteval.eval.runner.push_to_langsmith", return_value=None):
        with patch("agenteval.eval.scorers.task_success.get_llm", return_value=mock_llm):
            with patch("agenteval.eval.scorers.hallucination.get_llm", return_value=mock_llm):
                await runner.run(run_id=run_id, push_langsmith=False)

    report = report_store[run_id]
    # Should still complete (error captured per-case)
    assert report.status in ("completed", "failed")


# ──────────────────────────────────────────────
# FastAPI endpoint tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint():
    from httpx import AsyncClient, ASGITransport
    from agenteval.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_agent_query_requires_auth():
    from httpx import AsyncClient, ASGITransport
    from agenteval.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/agent/query", json={"question": "How many songs?"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_run_eval_requires_auth():
    from httpx import AsyncClient, ASGITransport
    from agenteval.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/eval/run-eval", json={})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_run_eval_returns_run_id(tmp_path):
    """POST /eval/run-eval should return a run_id immediately."""
    import json as _json

    cases = [
        {
            "id": "tc_001",
            "question": "How many songs?",
            "expected_answer": "6",
            "expected_tools": ["run_query"],
            "min_steps": 1,
            "difficulty": "easy",
            "tags": [],
        }
    ]
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(_json.dumps(cases))

    from httpx import AsyncClient, ASGITransport
    from agenteval.main import app
    from agenteval.config import get_settings

    settings = get_settings()

    # Manually initialise lifespan state (lifespan doesn't run in ASGITransport)
    app.state.report_store = {}

    with patch("agenteval.eval.runner.EvalRunner.run", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/eval/run-eval",
                json={"dataset_path": str(dataset_path), "push_to_langsmith": False},
                headers={"x-api-key": settings.api_key},
            )

    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "pending"
