"""Gestion de la connexion à la base de données PostgreSQL."""

import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_database_url() -> str:
    """Construit l'URL de connexion à partir des variables d'environnement."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "thumalien")
    user = os.getenv("POSTGRES_USER", "thumalien")
    password = os.getenv("POSTGRES_PASSWORD", "thumalien_secret")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    """Retourne le moteur SQLAlchemy (singleton)."""
    global _engine
    if _engine is None:
        url = get_database_url()
        _engine = create_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True)
        logger.info("Connexion BDD établie : %s:%s/%s",
                     os.getenv("POSTGRES_HOST", "localhost"),
                     os.getenv("POSTGRES_PORT", "5432"),
                     os.getenv("POSTGRES_DB", "thumalien"))
    return _engine


def get_session_factory() -> sessionmaker:
    """Retourne la factory de sessions."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


def get_session() -> Session:
    """Crée une nouvelle session."""
    factory = get_session_factory()
    return factory()
