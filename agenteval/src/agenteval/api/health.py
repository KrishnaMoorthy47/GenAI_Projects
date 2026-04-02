"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {"service": "agenteval", "status": "ok"}


@router.get("/health")
async def health():
    return {"status": "ok"}
