"""EvalRunner: orchestrates full evaluation run."""

import asyncio
import logging
from typing import Optional

_EVAL_CONCURRENCY = 5  # Max concurrent eval cases

from agenteval.agent.graph import run_sql_agent
from agenteval.eval.dataset import load_cases, push_to_langsmith
from agenteval.eval.reporter import aggregate_results, compute_overall
from agenteval.eval.scorers.hallucination import score_hallucination
from agenteval.eval.scorers.task_success import score_task_success
from agenteval.eval.scorers.tool_accuracy import score_tool_accuracy
from agenteval.eval.scorers.trajectory import score_trajectory
from agenteval.models.response import CaseResult, EvalReport, ScoreCard

logger = logging.getLogger(__name__)


async def _score_case(case: dict, llm=None) -> CaseResult:
    """Run agent on one test case and compute all 4 scores."""
    question = case["question"]
    expected_answer = case["expected_answer"]
    expected_tools = case.get("expected_tools", [])
    min_steps = case.get("min_steps", 1)

    actual_answer = None
    trajectory = []
    sql_result = None
    error = None

    try:
        state = await run_sql_agent(question, llm=llm)
        actual_answer = state.get("answer")
        trajectory = state.get("trajectory", [])
        sql_result = state.get("sql_result")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error("Agent error on case %s: %s", case["id"], e)
        error = str(e)

    # Score in parallel where possible
    task_score, _ = await score_task_success(question, expected_answer, actual_answer, llm=llm)
    hall_score, _ = await score_hallucination(actual_answer, sql_result, llm=llm)
    tool_score, _ = score_tool_accuracy(expected_tools, trajectory)
    traj_score, _ = score_trajectory(trajectory, min_steps)

    scores = ScoreCard(
        task_success=task_score,
        tool_accuracy=tool_score,
        trajectory_efficiency=traj_score,
        hallucination=hall_score,
        overall=0.0,
    )
    scores.overall = compute_overall(scores)

    return CaseResult(
        id=case["id"],
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        scores=scores,
        difficulty=case.get("difficulty", "unknown"),
        tags=case.get("tags", []),
        error=error,
    )


class EvalRunner:
    """Orchestrate full evaluation run."""

    def __init__(self, report_store: dict, dataset_path: Optional[str] = None, llm=None):
        self.report_store = report_store
        self.dataset_path = dataset_path
        self.llm = llm

    async def run(self, run_id: str, push_langsmith: bool = True, experiment_prefix: str = "sql-agent") -> None:
        """Run full eval. Updates report_store[run_id] in place."""
        report = self.report_store.get(run_id)
        if report:
            report.status = "running"

        try:
            cases = load_cases(self.dataset_path)
            n = len(cases)
            logger.info("Starting eval run %s with %d cases", run_id, n)

            if report:
                report.total_cases = n

            # Push dataset to LangSmith (non-blocking)
            langsmith_url = None
            if push_langsmith:
                langsmith_url = push_to_langsmith(cases)

            # Run cases concurrently (bounded by semaphore to avoid rate-limit hammering)
            sem = asyncio.Semaphore(_EVAL_CONCURRENCY)

            async def _run_with_sem(case: dict, idx: int) -> CaseResult:
                async with sem:
                    logger.info("Evaluating case %d/%d: %s", idx + 1, n, case["id"])
                    result = await _score_case(case, llm=self.llm)
                    if report:
                        report.completed_cases += 1
                    return result

            tasks = [_run_with_sem(case, i) for i, case in enumerate(cases)]
            results: list[CaseResult] = list(await asyncio.gather(*tasks))

            final_report = aggregate_results(run_id, results, langsmith_url=langsmith_url)
            self.report_store[run_id] = final_report

        except asyncio.CancelledError:
            logger.info("Eval run %s cancelled", run_id)
            raise
        except Exception as e:
            logger.error("Eval run %s failed: %s", run_id, e)
            if run_id in self.report_store:
                self.report_store[run_id].status = "failed"
                self.report_store[run_id].error = str(e)
            else:
                self.report_store[run_id] = EvalReport(
                    run_id=run_id, status="failed", error=str(e)
                )
