from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from jobsearch.config import get_llm, get_settings
from jobsearch.services.scorer import score_job_fit

logger = logging.getLogger(__name__)

# In-memory job store
_search_store: dict[str, dict[str, Any]] = {}

SUPPORTED_COMPANIES = ["Google", "Amazon", "Apple", "Microsoft", "LinkedIn", "Meta", "Netflix"]


def get_search_store() -> dict[str, dict[str, Any]]:
    return _search_store


def create_search(cv_text: str, companies: list[str], role: str) -> str:
    search_id = str(uuid.uuid4())
    _search_store[search_id] = {
        "status": "pending",
        "cv_length": len(cv_text),
        "companies": companies,
        "role": role,
        "progress": {},
        "jobs": [],
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    return search_id


async def _search_one_company(company: str, cv_text: str, role: str, search_id: str) -> list[dict]:
    """Run browser-use agent for one company and return list of jobs."""
    settings = get_settings()
    store = _search_store[search_id]
    store["progress"][company] = "running"

    try:
        from browser_use import Agent, Controller
        from browser_use.browser.browser import Browser, BrowserConfig

        controller = Controller()
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path=settings.chrome_path,
                disable_security=True,
                headless=settings.headless,
            )
        )

        found_jobs: list[dict] = []

        @controller.action("Save job listing with fit score")
        def save_job(title: str, link: str, location: str = "", salary: str = ""):
            fit = score_job_fit(cv_text, title, company)
            job = {
                "title": title,
                "company": company,
                "link": link,
                "location": location,
                "salary": salary,
                "fit_score": fit.get("score", 0.5),
                "fit_reason": fit.get("reason", ""),
            }
            found_jobs.append(job)
            return f"Saved: {title} (fit: {job['fit_score']:.2f})"

        @controller.action("Read CV for context")
        def read_cv():
            return cv_text[:3000]

        task = (
            f"Search {company}'s careers website for {role} job openings. "
            f"Use read_cv to understand the candidate profile. "
            f"Find up to 5 relevant positions and save each with save_job. "
            f"Focus on roles matching: {role}."
        )

        llm = get_llm()
        agent = Agent(task=task, llm=llm, controller=controller, browser=browser)
        await agent.run(max_steps=20)
        await browser.close()

        store["progress"][company] = "done"
        store["jobs"].extend(found_jobs)
        return found_jobs

    except Exception as exc:
        logger.exception("[%s] Browser agent failed for %s: %s", search_id, company, exc)
        store["progress"][company] = f"failed: {exc}"
        return []


async def run_job_search(search_id: str, cv_text: str, companies: list[str], role: str) -> None:
    store = _search_store[search_id]
    try:
        store["status"] = "running"
        logger.info("[%s] Starting job search: %s @ %s", search_id, role, companies)
        for company in companies:
            store["progress"][company] = "pending"

        # Run all companies in parallel
        await asyncio.gather(*[
            _search_one_company(company, cv_text, role, search_id)
            for company in companies
        ])

        # Sort by fit score descending
        store["jobs"].sort(key=lambda j: j.get("fit_score", 0), reverse=True)
        store["status"] = "completed"
        logger.info("[%s] Job search complete — %d jobs found", search_id, len(store["jobs"]))

    except Exception as exc:
        logger.exception("[%s] Job search failed: %s", search_id, exc)
        store["status"] = "failed"
        store["error"] = str(exc)
