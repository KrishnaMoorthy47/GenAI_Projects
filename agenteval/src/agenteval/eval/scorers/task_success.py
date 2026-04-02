"""LLM-as-judge scorer: did the agent answer match the expected answer?"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from agenteval.config import get_llm
from agenteval.eval.scorers.judge_model import JudgeResult

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
async def _invoke_judge(judge, prompt: str) -> "JudgeResult":
    return await judge.ainvoke([HumanMessage(content=prompt)])


TASK_SUCCESS_PROMPT = """You are an evaluation judge. Determine if the agent's answer correctly answers the question and matches the expected answer.

Question: {question}
Expected Answer: {expected_answer}
Agent's Answer: {actual_answer}

Score from 0.0 to 1.0 where:
- 1.0 = The agent's answer is correct and matches the expected answer (minor wording differences are OK)
- 0.7 = Partially correct (right concept, wrong number or missing details)
- 0.3 = Tangentially related but mostly wrong
- 0.0 = Completely wrong or no answer

Respond ONLY with valid JSON: {{"score": <float>, "reason": "<brief reason>"}}"""


async def score_task_success(
    question: str,
    expected_answer: str,
    actual_answer: Optional[str],
    llm=None,
) -> tuple[float, str]:
    """Return (score 0.0-1.0, reason string)."""
    if not actual_answer:
        return 0.0, "No answer produced."

    llm = llm or get_llm(temperature=0.0)
    judge = llm.with_structured_output(JudgeResult)
    prompt = TASK_SUCCESS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )

    try:
        result: JudgeResult = await _invoke_judge(judge, prompt)
        return max(0.0, min(1.0, result.score)), result.reason
    except Exception as e:
        logger.error("task_success scorer error: %s", e)
        return 0.0, f"Scorer error: {e}"
