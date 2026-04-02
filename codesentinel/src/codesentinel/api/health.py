from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "codesentinel"}


@router.get("/")
async def root():
    return {
        "service": "CodeSentinel",
        "description": "AI-powered GitHub PR security and quality review",
        "docs": "/docs",
    }
