"""Modèles SQLAlchemy ORM pour la base de données Thumalien."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class PostModel(Base):
    """Table des posts collectés depuis Bluesky."""
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uri = Column(String, unique=True, nullable=False)
    cid = Column(String)
    author_handle = Column(String, nullable=False)
    author_display_name = Column(String)
    text_content = Column(Text, nullable=False)
    clean_text = Column(Text)
    lang = Column(String(5))
    created_at = Column(DateTime(timezone=True))
    collected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    like_count = Column(Integer, default=0)
    repost_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)

    analyses = relationship("AnalysisModel", back_populates="post", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_posts_author", "author_handle"),
        Index("idx_posts_created", "created_at"),
    )


class AnalysisModel(Base):
    """Table des analyses (crédibilité + émotions)."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    credibility_label = Column(String(20), nullable=False)
    credibility_score = Column(Float, nullable=False)
    scores_detail = Column(JSON)
    dominant_emotion = Column(String(30))
    emotion_scores = Column(JSON)
    explanation = Column(JSON)
    analyzed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    post = relationship("PostModel", back_populates="analyses")

    __table_args__ = (
        Index("idx_analyses_label", "credibility_label"),
        Index("idx_analyses_post", "post_id"),
    )


class EnergyReportModel(Base):
    """Table des rapports énergétiques."""
    __tablename__ = "energy_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_name = Column(String(100))
    duration_seconds = Column(Float)
    emissions_kg_co2 = Column(Float)
    energy_kwh = Column(Float)
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalysisSessionModel(Base):
    """Table des sessions d'analyse (historique du dashboard)."""
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String(200), nullable=False)
    lang = Column(String(5))
    num_posts = Column(Integer)
    num_fiable = Column(Integer, default=0)
    num_douteux = Column(Integer, default=0)
    num_fake = Column(Integer, default=0)
    total_emissions_co2 = Column(Float, default=0.0)
    total_energy_kwh = Column(Float, default=0.0)
    duration_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
