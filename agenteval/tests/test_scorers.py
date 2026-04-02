"""Unit tests for all 4 scorers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agenteval.eval.scorers.judge_model import JudgeResult
from agenteval.eval.scorers.tool_accuracy import score_tool_accuracy
from agenteval.eval.scorers.trajectory import score_trajectory


# ──────────────────────────────────────────────
# Deterministic scorers (no LLM needed)
# ──────────────────────────────────────────────

class TestToolAccuracyScorer:
    def _trajectory(self, tool_names: list[str]) -> list[dict]:
        return [{"type": "tool_call", "name": n, "input": "", "output": ""} for n in tool_names]

    def test_perfect_recall(self):
        expected = ["list_tables", "get_schema", "run_query"]
        traj = self._trajectory(["list_tables", "get_schema", "run_query"])
        score, reason = score_tool_accuracy(expected, traj)
        assert score == 1.0

    def test_partial_recall(self):
        expected = ["list_tables", "get_schema", "run_query"]
        traj = self._trajectory(["list_tables", "run_query"])  # missing get_schema
        score, reason = score_tool_accuracy(expected, traj)
        assert abs(score - 2 / 3) < 0.01
        assert "get_schema" in reason.lower() or "Missing" in reason

    def test_zero_recall(self):
        expected = ["run_query"]
        traj = self._trajectory(["list_tables"])
        score, reason = score_tool_accuracy(expected, traj)
        assert score == 0.0

    def test_empty_expected(self):
        score, reason = score_tool_accuracy([], [])
        assert score == 1.0

    def test_extra_tools_not_penalised(self):
        """Extra tools called beyond expected don't reduce score."""
        expected = ["run_query"]
        traj = self._trajectory(["list_tables", "get_schema", "run_query"])
        score, reason = score_tool_accuracy(expected, traj)
        assert score == 1.0


class TestTrajectoryScorer:
    def _trajectory(self, n_tool_calls: int) -> list[dict]:
        return [{"type": "tool_call", "name": "run_query", "input": "", "output": ""}] * n_tool_calls

    def test_exact_min_steps(self):
        traj = self._trajectory(3)
        score, reason = score_trajectory(traj, min_steps=3)
        assert score == 1.0

    def test_one_extra_step(self):
        traj = self._trajectory(4)
        score, reason = score_trajectory(traj, min_steps=3)
        assert abs(score - 0.9) < 0.01

    def test_many_extra_steps(self):
        traj = self._trajectory(13)
        score, reason = score_trajectory(traj, min_steps=3)
        assert score == 0.0  # clamped at 0

    def test_fewer_than_min_steps_not_penalised(self):
        """If agent uses fewer steps than min (edge case), score is still 1.0."""
        traj = self._trajectory(2)
        score, reason = score_trajectory(traj, min_steps=3)
        assert score == 1.0  # max(0, 2-3) = 0, so no penalty


# ──────────────────────────────────────────────
# LLM-as-judge scorers (mocked LLM)
# ──────────────────────────────────────────────

def make_judge_llm(score: float, reason: str = "test reason"):
    """Return a mock LLM that returns a structured JudgeResult via with_structured_output."""
    judge_result = JudgeResult(score=score, reason=reason)
    judge_mock = MagicMock()
    judge_mock.ainvoke = AsyncMock(return_value=judge_result)
    mock = MagicMock()
    mock.with_structured_output = MagicMock(return_value=judge_mock)
    return mock


class TestTaskSuccessScorer:
    @pytest.mark.asyncio
    async def test_correct_answer(self):
        from agenteval.eval.scorers.task_success import score_task_success
        llm = make_judge_llm(score=1.0, reason="Correct")
        score, reason = await score_task_success(
            question="How many tracks?",
            expected_answer="130",
            actual_answer="There are 130 tracks.",
            llm=llm,
        )
        assert score == 1.0
        assert "Correct" in reason

    @pytest.mark.asyncio
    async def test_wrong_answer(self):
        from agenteval.eval.scorers.task_success import score_task_success
        llm = make_judge_llm(score=0.0, reason="Completely wrong")
        score, reason = await score_task_success(
            question="How many tracks?",
            expected_answer="130",
            actual_answer="I don't know.",
            llm=llm,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_answer_returns_zero(self):
        from agenteval.eval.scorers.task_success import score_task_success
        score, reason = await score_task_success(
            question="q", expected_answer="x", actual_answer=None
        )
        assert score == 0.0
        assert "No answer" in reason

    @pytest.mark.asyncio
    async def test_score_clamped(self):
        """Scores outside [0, 1] should be clamped."""
        from agenteval.eval.scorers.task_success import score_task_success
        llm = make_judge_llm(score=1.5)  # out of range
        score, _ = await score_task_success("q", "x", "y", llm=llm)
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_error_returns_zero(self):
        from agenteval.eval.scorers.task_success import score_task_success
        judge_mock = MagicMock()
        judge_mock.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
        mock = MagicMock()
        mock.with_structured_output = MagicMock(return_value=judge_mock)
        score, reason = await score_task_success("q", "x", "y", llm=mock)
        assert score == 0.0
        assert "error" in reason.lower()


class TestHallucinationScorer:
    @pytest.mark.asyncio
    async def test_grounded_answer(self):
        from agenteval.eval.scorers.hallucination import score_hallucination
        llm = make_judge_llm(score=1.0, reason="Fully grounded")
        score, reason = await score_hallucination(
            actual_answer="There are 130 tracks.",
            sql_result="COUNT(*)\n130",
            llm=llm,
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_hallucinated_answer(self):
        from agenteval.eval.scorers.hallucination import score_hallucination
        llm = make_judge_llm(score=0.0, reason="Number not in SQL result")
        score, reason = await score_hallucination(
            actual_answer="There are 999 tracks.",
            sql_result="COUNT(*)\n130",
            llm=llm,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_answer_returns_one(self):
        from agenteval.eval.scorers.hallucination import score_hallucination
        score, reason = await score_hallucination(actual_answer=None, sql_result="anything")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_sql_result_returns_half(self):
        from agenteval.eval.scorers.hallucination import score_hallucination
        score, reason = await score_hallucination(actual_answer="some answer", sql_result=None)
        assert score == 0.5
