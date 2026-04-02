from __future__ import annotations

from pydantic import BaseModel, Field


class PreferencesRequest(BaseModel):
    """Meal preferences for the household."""

    likes: list[str] = Field(default_factory=list, description="Foods / cuisines the family enjoys")
    dislikes: list[str] = Field(default_factory=list, description="Foods / cuisines to avoid")
    hard_requirements: list[str] = Field(
        default_factory=list,
        description="Non-negotiable dietary rules (e.g. nut-free, vegetarian, halal)",
    )
