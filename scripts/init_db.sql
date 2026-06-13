-- ============================================================
-- Thumalien — Initialisation du schéma PostgreSQL
-- Exécuté automatiquement par Docker au premier démarrage
-- Compatible avec db_connector.py (PE) et connection.py (master)
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ------------------------------------------------------------
-- Posts collectés depuis Bluesky
-- (id TEXT pour compatibilité URI ATProto)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,           -- URI ATProto (PE)
    uri             TEXT UNIQUE,                -- alias master
    cid             TEXT,
    author          TEXT NOT NULL,
    author_handle   TEXT,                       -- alias master
    author_name     TEXT,
    author_display_name TEXT,
    text_original   TEXT NOT NULL,
    text_content    TEXT,                       -- alias master
    text_clean      TEXT,
    clean_text      TEXT,                       -- alias master
    lang            VARCHAR(5),
    language        TEXT,
    like_count      INTEGER DEFAULT 0,
    reply_count     INTEGER DEFAULT 0,
    repost_count    INTEGER DEFAULT 0,
    created_at      TEXT,
    collected_at    TIMESTAMPTZ DEFAULT NOW(),
    extracted_at    TIMESTAMPTZ DEFAULT NOW(),
    processed_at    TIMESTAMPTZ,
    is_analyzed     BOOLEAN DEFAULT FALSE
);

-- ------------------------------------------------------------
-- Prédictions fake news
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    post_id             TEXT REFERENCES posts(id) ON DELETE CASCADE,
    credibility_score   FLOAT NOT NULL,
    credibility_label   VARCHAR(20),
    is_fake             BOOLEAN NOT NULL,
    confidence          FLOAT NOT NULL,
    scores_detail       JSONB,
    model_version       TEXT DEFAULT 'v1',
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id)
);

-- ------------------------------------------------------------
-- Analyses émotionnelles
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS emotion_analyses (
    id              SERIAL PRIMARY KEY,
    post_id         TEXT REFERENCES posts(id) ON DELETE CASCADE,
    emotion_label   TEXT NOT NULL,
    dominant_emotion TEXT,                      -- alias master
    emotion_score   FLOAT NOT NULL,
    emotion_scores  JSONB,                      -- détail master
    vader_compound  FLOAT,
    analyzed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id)
);

-- ------------------------------------------------------------
-- Monitoring énergétique
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS energy_tracking (
    id              SERIAL PRIMARY KEY,
    task_name       VARCHAR(100),
    duration_s      FLOAT,
    duration_seconds FLOAT,                     -- alias master
    co2_kg          FLOAT,
    emissions_kg_co2 FLOAT,                     -- alias master
    energy_kwh      FLOAT,
    cpu_percent     FLOAT,
    ram_mb          FLOAT,
    model_name      TEXT,
    n_samples       INTEGER,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Sessions d'analyse (historique dashboard — master)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id                  SERIAL PRIMARY KEY,
    query               VARCHAR(200) NOT NULL,
    lang                VARCHAR(5),
    num_posts           INTEGER,
    num_fiable          INTEGER DEFAULT 0,
    num_douteux         INTEGER DEFAULT 0,
    num_fake            INTEGER DEFAULT 0,
    total_emissions_co2 FLOAT DEFAULT 0.0,
    total_energy_kwh    FLOAT DEFAULT 0.0,
    duration_seconds    FLOAT DEFAULT 0.0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Index pour les requêtes fréquentes
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_posts_author    ON posts(author);
CREATE INDEX IF NOT EXISTS idx_posts_created   ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_lang      ON posts(lang);
CREATE INDEX IF NOT EXISTS idx_pred_label      ON predictions(credibility_label);
CREATE INDEX IF NOT EXISTS idx_pred_post       ON predictions(post_id);
CREATE INDEX IF NOT EXISTS idx_emo_post        ON emotion_analyses(post_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date   ON analysis_sessions(created_at);
-- Full-text search sur le texte brut
CREATE INDEX IF NOT EXISTS idx_posts_text_trgm ON posts USING gin(text_original gin_trgm_ops);
