"""Repository pattern pour les opérations CRUD sur la base de données."""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.models import PostModel, AnalysisModel, EnergyReportModel, AnalysisSessionModel

logger = logging.getLogger(__name__)


class PostRepository:
    """Opérations CRUD pour les posts."""

    def __init__(self, session: Session):
        self.session = session

    def upsert(self, post_data: dict) -> PostModel:
        """Insère ou met à jour un post (upsert sur URI)."""
        stmt = pg_insert(PostModel).values(
            uri=post_data["uri"],
            cid=post_data.get("cid"),
            author_handle=post_data["author_handle"],
            author_display_name=post_data.get("author_display_name"),
            text_content=post_data["text"],
            clean_text=post_data.get("clean_text"),
            lang=post_data.get("detected_lang") or (post_data.get("lang", [None])[0] if isinstance(post_data.get("lang"), list) else None),
            created_at=post_data.get("created_at"),
            collected_at=post_data.get("collected_at", datetime.now(timezone.utc).isoformat()),
            like_count=post_data.get("like_count", 0),
            repost_count=post_data.get("repost_count", 0),
            reply_count=post_data.get("reply_count", 0),
        ).on_conflict_do_update(
            index_elements=["uri"],
            set_={"clean_text": post_data.get("clean_text"), "like_count": post_data.get("like_count", 0)},
        ).returning(PostModel.id)

        result = self.session.execute(stmt)
        self.session.flush()
        post_id = result.scalar_one()

        return self.session.get(PostModel, post_id)

    def get_by_uri(self, uri: str) -> PostModel | None:
        return self.session.query(PostModel).filter(PostModel.uri == uri).first()

    def get_recent(self, limit: int = 50) -> list[PostModel]:
        return (
            self.session.query(PostModel)
            .order_by(PostModel.collected_at.desc())
            .limit(limit)
            .all()
        )


class AnalysisRepository:
    """Opérations CRUD pour les analyses."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, post_id: int, credibility: dict, emotion: dict, explanation: dict | None = None) -> AnalysisModel:
        analysis = AnalysisModel(
            post_id=post_id,
            credibility_label=credibility["label"],
            credibility_score=credibility["confidence"],
            scores_detail=credibility.get("scores"),
            dominant_emotion=emotion.get("dominant_emotion"),
            emotion_scores=emotion.get("scores"),
            explanation=explanation,
        )
        self.session.add(analysis)
        self.session.flush()
        return analysis

    def get_by_post(self, post_id: int) -> list[AnalysisModel]:
        return self.session.query(AnalysisModel).filter(AnalysisModel.post_id == post_id).all()

    def get_recent(self, limit: int = 100) -> list[AnalysisModel]:
        return (
            self.session.query(AnalysisModel)
            .order_by(AnalysisModel.analyzed_at.desc())
            .limit(limit)
            .all()
        )


class EnergyRepository:
    """Opérations CRUD pour les rapports énergétiques."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, task_name: str, duration: float, emissions: float, energy: float) -> EnergyReportModel:
        report = EnergyReportModel(
            task_name=task_name,
            duration_seconds=duration,
            emissions_kg_co2=emissions,
            energy_kwh=energy,
        )
        self.session.add(report)
        self.session.flush()
        return report


class SessionRepository:
    """Opérations CRUD pour l'historique des sessions d'analyse."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, query: str, lang: str | None, results: list[dict], energy: dict) -> AnalysisSessionModel:
        num_fiable = sum(1 for r in results if r.get("credibility", {}).get("label") == "fiable")
        num_douteux = sum(1 for r in results if r.get("credibility", {}).get("label") == "douteux")
        num_fake = sum(1 for r in results if r.get("credibility", {}).get("label") == "fake")

        session_record = AnalysisSessionModel(
            query=query,
            lang=lang,
            num_posts=len(results),
            num_fiable=num_fiable,
            num_douteux=num_douteux,
            num_fake=num_fake,
            total_emissions_co2=energy.get("total_emissions_kg_co2", 0),
            total_energy_kwh=energy.get("total_energy_kwh", 0),
            duration_seconds=energy.get("total_duration_seconds", 0),
        )
        self.session.add(session_record)
        self.session.flush()
        return session_record

    def get_history(self, limit: int = 20) -> list[AnalysisSessionModel]:
        return (
            self.session.query(AnalysisSessionModel)
            .order_by(AnalysisSessionModel.created_at.desc())
            .limit(limit)
            .all()
        )


def save_pipeline_results(session: Session, query: str, lang: str | None, results: list[dict], energy: dict):
    """Sauvegarde complète des résultats du pipeline en BDD.

    Persiste les posts, analyses, métriques énergétiques et la session.
    """
    post_repo = PostRepository(session)
    analysis_repo = AnalysisRepository(session)
    energy_repo = EnergyRepository(session)
    session_repo = SessionRepository(session)

    for result in results:
        post = post_repo.upsert(result)
        analysis_repo.create(
            post_id=post.id,
            credibility=result.get("credibility", {}),
            emotion=result.get("emotion", {}),
            explanation=result.get("explanation"),
        )

    for task in energy.get("tasks", []):
        energy_repo.create(
            task_name=task["task"],
            duration=task.get("duration_seconds", 0),
            emissions=task.get("emissions_kg_co2", 0),
            energy=task.get("energy_kwh", 0),
        )

    session_repo.create(query, lang, results, energy)
    session.commit()
    logger.info("Résultats sauvegardés en BDD : %d posts, %d tâches énergie", len(results), len(energy.get("tasks", [])))
