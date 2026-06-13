"""
src/schemas.py — Modèles Pydantic pour la validation des données Thumalien.
Apporté depuis le projet de référence (master).
"""
from datetime import datetime
from pydantic import BaseModel, Field


# ---------- Posts ----------

class PostBase(BaseModel):
    """Données brutes d'un post Bluesky."""
    uri: str
    author_handle: str
    author_display_name: str | None = None
    text: str
    lang: list[str | None] = Field(default_factory=list)
    created_at: str
    collected_at: str
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0


class PostProcessed(PostBase):
    """Post après prétraitement NLP."""
    clean_text: str
    tokens: list[str]
    detected_lang: str = "fr"


# ---------- Classification ----------

class CredibilityScores(BaseModel):
    fiable: float = Field(ge=0, le=1)
    douteux: float = Field(ge=0, le=1)
    fake: float = Field(ge=0, le=1)


class CredibilityResult(BaseModel):
    label: str = Field(pattern=r"^(fiable|douteux|fake)$")
    confidence: float = Field(ge=0, le=1)
    scores: CredibilityScores | None = None


# ---------- Émotions ----------

class EmotionResult(BaseModel):
    dominant_emotion: str
    confidence: float = Field(ge=0, le=1)
    scores: dict[str, float] = Field(default_factory=dict)


# ---------- Énergie ----------

class EnergyTask(BaseModel):
    task: str
    duration_seconds: float
    emissions_kg_co2: float = 0.0
    energy_kwh: float = 0.0


class EnergySummary(BaseModel):
    total_emissions_kg_co2: float = 0.0
    total_duration_seconds: float = 0.0
    total_energy_kwh: float = 0.0
    num_tasks: int = 0
    tasks: list[EnergyTask] = Field(default_factory=list)


# ---------- Pipeline ----------

class PipelineConfig(BaseModel):
    bluesky_handle: str
    bluesky_password: str
    query: str = "fake news"
    lang: str | None = None
    limit: int = Field(default=25, ge=1, le=500)


class PipelineOutput(BaseModel):
    timestamp: str
    query: str
    num_posts: int
    energy: EnergySummary
