from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "finagent"}


@router.get("/")
async def root():
    return {
        "service": "FinAgent",
        "description": "AI-powered stock research with LangGraph multi-agent system",
        "docs": "/docs",
    }
