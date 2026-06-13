# ============================================================
# Thumalien — Makefile unifié (PE + master)
# ============================================================

.PHONY: help setup install dev lint format test test-cov \
        db db-init db-history db-stats \
        train train-bert train-liar evaluate \
        collect pipeline dashboard \
        docker-up docker-down docker-build docker-logs docker-dev \
        clean clean-data

# Affichage automatique de l'aide (style master)
help: ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ============================================================
# INSTALLATION
# ============================================================

setup: ## Installation complète automatique (PE)
	python scripts/setup.py

install: ## Installer les dépendances Python
	pip install -r requirements.txt
	python -m spacy download fr_core_news_sm
	python -m spacy download en_core_web_sm
	python -c "import nltk; [nltk.download(r, quiet=True) for r in ['stopwords', 'punkt']]"

dev: install ## Installer dépendances + outils de dev (ruff, pre-commit)
	pip install ruff pre-commit
	pre-commit install
	@echo "✅ Environnement dev prêt. pre-commit installé."

# ============================================================
# QUALITÉ DE CODE (master)
# ============================================================

lint: ## Vérifier le code avec ruff
	ruff check src/ tests/

format: ## Formater et corriger le code avec ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ============================================================
# BASE DE DONNÉES
# ============================================================

db: db-init ## Alias : initialiser la base de données

db-init: ## Initialiser les tables PostgreSQL
	python main.py db init

db-history: ## Afficher l'historique des sessions
	python main.py db history

db-stats: ## Afficher les statistiques de la BDD
	python main.py db stats

# ============================================================
# ENTRAÎNEMENT
# ============================================================

train: ## Entraîner le modèle baseline (TF-IDF + LogReg)
	python scripts/train_model.py --model baseline

train-bert: ## Entraîner DistilBERT multilingue (CPU, ~20 min)
	python scripts/train_model.py --model bert

train-liar: ## Fine-tuner sur le dataset LIAR (master)
	python main.py train --dataset liar --epochs 3

evaluate: ## Évaluer le modèle fine-tuné (master)
	python main.py evaluate --model data/models/fake_news_detector

# ============================================================
# PIPELINE
# ============================================================

collect: ## Collecter des posts Bluesky (one-shot)
	python -m src.collector.bluesky_collector --mode once --limit 25 --save-db

pipeline: ## Lancer le pipeline complet
	python -m src.pipeline --limit 25

analyze: ## Analyser via CLI (ex: make analyze QUERY="complot")
	python main.py analyze "$(or $(QUERY),fake news)" --lang fr

# ============================================================
# DASHBOARD
# ============================================================

dashboard: ## Démarrer le dashboard Streamlit
	streamlit run dashboard/Home.py

# ============================================================
# TESTS
# ============================================================

test: ## Lancer les tests unitaires
	pytest tests/ -v

test-cov: ## Tests avec rapport de couverture HTML
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Rapport HTML : htmlcov/index.html"

# ============================================================
# DOCKER
# ============================================================

docker-up: ## Démarrer postgres + dashboard
	docker compose up -d

docker-down: ## Arrêter tous les conteneurs
	docker compose down

docker-build: ## Rebuilder les images
	docker compose build

docker-logs: ## Suivre les logs en temps réel
	docker compose logs -f

docker-dev: ## Mode dev avec hot-reload (profil dev)
	docker compose --profile dev up

docker-pipeline: ## Lancer le pipeline via Docker (one-shot)
	docker compose --profile pipeline up pipeline

# ============================================================
# NETTOYAGE
# ============================================================

clean: ## Nettoyer les fichiers Python temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage

clean-data: ## Nettoyer les données temporaires (raw, processed, logs)
	rm -f data/raw/*.json data/processed/*.json logs/*.log

# ============================================================
# LABELER AT PROTOCOL
# ============================================================

labeler-setup: ## Déclarer le compte Bluesky comme Labeler (one-shot)
	python -m src.labeler.setup_labeler_account

labeler-dry: ## Tester le labeler sans émettre (dry-run)
	python -m src.labeler.labeler_service --dry-run

labeler: ## Démarrer le labeler en temps réel (production)
	python -m src.labeler.labeler_service

labeler-fr: ## Labeler uniquement les posts français
	python -m src.labeler.labeler_service --lang fr

docker-labeler: ## Lancer le labeler via Docker (production)
	docker compose --profile labeler up -d labeler

docker-labeler-dry: ## Lancer le labeler dry-run via Docker
	docker compose --profile labeler-dry up labeler-dry
