"""LLM-as-judge scorer: is the agent's answer grounded in the SQL result?"""

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


HALLUCINATION_PROMPT = """You are an evaluation judge checking for hallucination.

The agent ran a SQL query and got this result:
--- SQL RESULT ---
{sql_result}
--- END SQL RESULT ---

The agent then produced this answer:
--- AGENT ANSWER ---
{actual_answer}
--- END AGENT ANSWER ---

Determine if the answer contains specific facts (numbers, names, dates) that are NOT present in the SQL result.

Score from 0.0 to 1.0 where:
- 1.0 = Fully grounded — every factual claim in the answer is supported by the SQL result
- 0.5 = Partially grounded — some facts supported, some potentially hallucinated
- 0.0 = Answer contains facts not present in the SQL result (hallucination detected)

Respond ONLY with valid JSON: {{"score": <float>, "reason": "<brief reason>"}}"""


async def score_hallucination(
    actual_answer: Optional[str],
    sql_result: Optional[str],
    llm=None,
) -> tuple[float, str]:
    """Return (score 0.0-1.0, reason string). Higher = less hallucination."""
    if not actual_answer:
        return 1.0, "No answer to check."
    if not sql_result:
        return 0.5, "No SQL result available to verify against."

    llm = llm or get_llm(temperature=0.0)
    judge = llm.with_structured_output(JudgeResult)
    prompt = HALLUCINATION_PROMPT.format(
        sql_result=sql_result,
        actual_answer=actual_answer,
    )

    try:
        result: JudgeResult = await _invoke_judge(judge, prompt)
        return max(0.0, min(1.0, result.score)), result.reason
    except Exception as e:
        logger.error("hallucination scorer error: %s", e)
        return 0.5, f"Scorer error: {e}"
