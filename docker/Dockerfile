# ============================================================
# Thumalien — Dockerfile
# Python 3.11 slim | CPU-optimised | multi-stage inspired
# ============================================================
FROM python:3.11-slim

LABEL maintainer="Thumalien Team — SUP DE VINCI 2025/2026"
LABEL description="Fake News Detection Pipeline for Bluesky"

# --- Dépendances système (--no-install-recommends pour layer léger) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Variables d'environnement (best practice master) ---
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- Dépendances Python (layer cacheable — copier avant le code) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Modèles spaCy (couche séparée pour le cache) ---
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

# --- Ressources NLTK ---
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

# --- Code source ---
COPY . .

# --- Dossiers de données persistables ---
RUN mkdir -p data/raw data/processed data/labeled models logs

# --- Port Streamlit ---
EXPOSE 8501

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --- Commande par défaut : dashboard ---
CMD ["streamlit", "run", "dashboard/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
