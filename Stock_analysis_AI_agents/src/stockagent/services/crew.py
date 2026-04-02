from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from crewai import Agent, Crew, Process, Task

from stockagent.config import get_crewai_llm
from stockagent.tools.yfinance_tools import (
    get_basic_stock_info,
    get_fundamental_analysis,
    get_stock_news,
    get_stock_risk_assessment,
    get_technical_analysis,
)

logger = logging.getLogger(__name__)

# In-memory job store
_job_store: dict[str, dict[str, Any]] = {}


def get_job_store() -> dict[str, dict[str, Any]]:
    return _job_store


def create_job(query: str) -> str:
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "status": "pending",
        "query": query,
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    return job_id


def _run_crew_sync(query: str) -> str:
    """Build and run the CrewAI crew synchronously (called in thread executor)."""
    llm = get_crewai_llm()

    stock_researcher = Agent(
        llm=llm,
        role="Stock Researcher",
        goal="Identify the stock ticker from the query and retrieve basic stock information.",
        backstory="A junior stock researcher who gathers essential data about stocks, industries, and market positioning.",
        tools=[get_basic_stock_info],
        verbose=False,
        allow_delegation=False,
    )

    financial_analyst = Agent(
        llm=llm,
        role="Financial Analyst",
        goal="Perform fundamental, technical, and risk analysis on the identified stock.",
        backstory="A seasoned financial analyst with expertise in interpreting complex financial data.",
        tools=[get_technical_analysis, get_fundamental_analysis, get_stock_risk_assessment],
        verbose=False,
        allow_delegation=False,
    )

    news_analyst = Agent(
        llm=llm,
        role="News Analyst",
        goal="Fetch and analyse recent news for the stock and assess its potential impact.",
        backstory="A sharp news analyst who quickly digests market news and assesses stock implications.",
        tools=[get_stock_news],
        verbose=False,
        allow_delegation=False,
    )

    report_writer = Agent(
        role="Financial Report Writer",
        goal="Synthesize all analysis into a cohesive, professional stock report with a clear recommendation.",
        backstory="An experienced financial writer with a talent for clear, concise reporting.",
        tools=[],
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    collect_info = Task(
        description=f"""
Identify the stock ticker from the user query and retrieve basic stock info.
User query: {query}
""",
        expected_output="Stock name, ticker, sector, current price, and basic metrics.",
        agent=stock_researcher,
    )

    perform_analysis = Task(
        description=f"""
Conduct thorough fundamental, technical, and risk analysis based on the user query.
Focus on the most relevant metrics for: {query}
""",
        expected_output="Detailed financial and technical analysis with key metrics and interpretation.",
        agent=financial_analyst,
        context=[collect_info],
    )

    analyse_news = Task(
        description="Fetch recent news for the identified stock and assess sentiment and potential impact.",
        expected_output="Summary of recent news and overall market sentiment.",
        agent=news_analyst,
        context=[collect_info],
    )

    generate_report = Task(
        description=f"""
Synthesize all information into a professional stock report in Markdown format.
Include: Executive Summary, Key Metrics, Analysis, News Sentiment, Investment Recommendation (BUY/HOLD/SELL).
User query: {query}
""",
        expected_output="Complete Markdown stock report with investment recommendation.",
        agent=report_writer,
        context=[collect_info, perform_analysis, analyse_news],
    )

    crew = Crew(
        agents=[stock_researcher, financial_analyst, news_analyst, report_writer],
        tasks=[collect_info, perform_analysis, analyse_news, generate_report],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"query": query, "default_date": datetime.now().strftime("%Y-%m-%d")})
    return str(result)


async def run_analysis(job_id: str, query: str) -> None:
    store = _job_store[job_id]
    try:
        store["status"] = "running"
        logger.info("[%s] Starting stock analysis for: %s", job_id, query)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_crew_sync, query)
        store["status"] = "completed"
        store["result"] = result
        logger.info("[%s] Analysis complete", job_id)
    except Exception as exc:
        logger.exception("[%s] Analysis failed: %s", job_id, exc)
        store["status"] = "failed"
        store["error"] = str(exc)
