"""Direct agent query endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from agenteval.agent.graph import run_sql_agent
from agenteval.models.request import AgentQueryRequest
from agenteval.models.response import AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def verify_api_key(request: AgentQueryRequest = None):
    """Placeholder — actual auth injected via dependency in main.py."""
    pass


@router.post("/query", response_model=AgentResponse)
async def query_agent(body: AgentQueryRequest):
    """Run the SQL agent for a single question and return its answer + trajectory."""
    try:
        state = await run_sql_agent(body.question)
        return AgentResponse(
            question=body.question,
            answer=state.get("answer"),
            trajectory=state.get("trajectory", []),
            sql_result=state.get("sql_result"),
            iterations=state.get("iterations", 0),
        )
    except Exception as e:
        logger.error("Agent query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
