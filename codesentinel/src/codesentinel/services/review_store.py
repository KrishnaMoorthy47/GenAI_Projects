from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from codesentinel.config import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ReviewRecord(Base):
    __tablename__ = "reviews"

    review_id = Column(String(36), primary_key=True)
    repo = Column(String(200), nullable=False)
    pr_number = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    security_findings = Column(JSON, default=list)
    quality_findings = Column(JSON, default=list)
    final_review = Column(Text, default="")
    files_reviewed = Column(JSON, default=list)
    total_findings = Column(Integer, default=0)
    critical_count = Column(Integer, default=0)
    high_count = Column(Integer, default=0)
    pr_comment_posted = Column(Integer, default=0)  # SQLite bool as int
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = None
_session_factory: async_sessionmaker | None = None


async def init_db() -> None:
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_async_engine(settings.database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Review store database initialized")


async def close_db() -> None:
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Review store database closed")


def _get_session() -> AsyncSession:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


async def create_review(repo: str, pr_number: int) -> str:
    """Create a new review record and return the review_id."""
    review_id = str(uuid.uuid4())
    async with _get_session() as session:
        record = ReviewRecord(
            review_id=review_id,
            repo=repo,
            pr_number=pr_number,
            status="pending",
        )
        session.add(record)
        await session.commit()
    return review_id


async def update_review(review_id: str, **fields) -> None:
    """Update fields on an existing review record."""
    async with _get_session() as session:
        result = await session.execute(
            select(ReviewRecord).where(ReviewRecord.review_id == review_id)
        )
        record = result.scalar_one_or_none()
        if record:
            for key, value in fields.items():
                setattr(record, key, value)
            await session.commit()


async def get_review(review_id: str) -> dict | None:
    """Retrieve a review record by ID."""
    async with _get_session() as session:
        result = await session.execute(
            select(ReviewRecord).where(ReviewRecord.review_id == review_id)
        )
        record = result.scalar_one_or_none()
        if not record:
            return None
        return {
            "review_id": record.review_id,
            "repo": record.repo,
            "pr_number": record.pr_number,
            "status": record.status,
            "security_findings": record.security_findings or [],
            "quality_findings": record.quality_findings or [],
            "final_review": record.final_review or "",
            "files_reviewed": record.files_reviewed or [],
            "total_findings": record.total_findings or 0,
            "critical_count": record.critical_count or 0,
            "high_count": record.high_count or 0,
            "pr_comment_posted": bool(record.pr_comment_posted),
            "error": record.error,
            "created_at": record.created_at,
        }
