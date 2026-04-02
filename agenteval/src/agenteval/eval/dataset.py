"""Load eval cases from JSON; push dataset to LangSmith."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "datasets/eval_cases.json"


def load_cases(dataset_path: Optional[str] = None) -> list[dict]:
    """Load test cases from JSON file."""
    path = Path(dataset_path or DEFAULT_DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        cases = json.load(f)
    logger.info("Loaded %d eval cases from %s", len(cases), path)
    return cases


def push_to_langsmith(cases: list[dict], dataset_name: str = "sql-agent-eval-v1") -> Optional[str]:
    """Push dataset to LangSmith. Returns dataset URL or None if LangSmith unavailable."""
    try:
        from langsmith import Client
        from agenteval.config import get_settings

        settings = get_settings()
        if not settings.langsmith_api_key:
            logger.warning("LANGSMITH_API_KEY not set — skipping dataset push")
            return None

        client = Client(api_key=settings.langsmith_api_key)

        # Create dataset (idempotent — skip if exists)
        existing = [d for d in client.list_datasets() if d.name == dataset_name]
        if existing:
            dataset = existing[0]
            logger.info("LangSmith dataset already exists: %s", dataset_name)
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="SQL agent evaluation dataset — Chinook music DB",
            )
            logger.info("Created LangSmith dataset: %s", dataset_name)

        # Push examples
        client.create_examples(
            inputs=[{"question": c["question"]} for c in cases],
            outputs=[{"expected_answer": c["expected_answer"]} for c in cases],
            dataset_id=dataset.id,
        )
        logger.info("Pushed %d examples to LangSmith", len(cases))
        return str(dataset.url) if hasattr(dataset, "url") else None

    except Exception as e:
        logger.error("LangSmith push failed: %s", e)
        return None
