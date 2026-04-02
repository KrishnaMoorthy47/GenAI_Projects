"""Aggregate per-case scores into an EvalReport."""

import logging

from agenteval.models.response import CaseResult, EvalReport, ScoreCard

logger = logging.getLogger(__name__)

WEIGHTS = {
    "task_success": 0.40,
    "tool_accuracy": 0.20,
    "trajectory_efficiency": 0.20,
    "hallucination": 0.20,
}


def compute_overall(scores: ScoreCard) -> float:
    return round(
        scores.task_success * WEIGHTS["task_success"]
        + scores.tool_accuracy * WEIGHTS["tool_accuracy"]
        + scores.trajectory_efficiency * WEIGHTS["trajectory_efficiency"]
        + scores.hallucination * WEIGHTS["hallucination"],
        3,
    )


def aggregate_results(run_id: str, case_results: list[CaseResult], langsmith_url: str = None) -> EvalReport:
    """Compute aggregate scores across all case results."""
    if not case_results:
        return EvalReport(
            run_id=run_id,
            status="completed",
            total_cases=0,
            completed_cases=0,
            langsmith_url=langsmith_url,
        )

    n = len(case_results)
    agg = ScoreCard(
        task_success=round(sum(r.scores.task_success for r in case_results) / n, 3),
        tool_accuracy=round(sum(r.scores.tool_accuracy for r in case_results) / n, 3),
        trajectory_efficiency=round(sum(r.scores.trajectory_efficiency for r in case_results) / n, 3),
        hallucination=round(sum(r.scores.hallucination for r in case_results) / n, 3),
        overall=round(sum(r.scores.overall for r in case_results) / n, 3),
    )

    logger.info(
        "Eval complete — %d cases | overall=%.3f task=%.3f tool=%.3f traj=%.3f hall=%.3f",
        n,
        agg.overall,
        agg.task_success,
        agg.tool_accuracy,
        agg.trajectory_efficiency,
        agg.hallucination,
    )

    return EvalReport(
        run_id=run_id,
        status="completed",
        total_cases=n,
        completed_cases=n,
        aggregate_scores=agg,
        case_results=case_results,
        langsmith_url=langsmith_url,
    )
