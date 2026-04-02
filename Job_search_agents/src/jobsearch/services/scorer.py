from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from jobsearch.config import get_llm

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_chain(chain, inputs: dict):
    return chain.invoke(inputs)


_SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert career advisor. Score how well a job matches a candidate's CV.
Return ONLY a JSON object with these fields:
- score: float between 0.0 and 1.0 (1.0 = perfect match)
- reason: one sentence explaining the score"""),
    ("human", """CV Summary:
{cv_text}

Job Title: {job_title}
Company: {company}
Job Description: {job_desc}

Return JSON only."""),
])


def score_job_fit(cv_text: str, job_title: str, company: str, job_desc: str = "") -> dict:
    """Score how well a job matches a CV. Returns {score, reason}."""
    import json
    try:
        llm = get_llm()
        chain = _SCORE_PROMPT | llm
        result = _invoke_chain(chain, {
            "cv_text": cv_text[:3000],  # truncate to avoid token limits
            "job_title": job_title,
            "company": company,
            "job_desc": job_desc[:500],
        })
        content = result.content
        j_start, j_end = content.find("{"), content.rfind("}")
        if j_start != -1 and j_end != -1:
            return json.loads(content[j_start: j_end + 1])
    except Exception as exc:
        logger.warning("Scoring failed: %s", exc)
    return {"score": 0.5, "reason": "Unable to score — using default"}
