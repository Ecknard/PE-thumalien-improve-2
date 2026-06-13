"""
src/cli.py — Interface en ligne de commande Thumalien (master)

Usage :
    python main.py analyze "fake news" --lang fr --limit 50
    python main.py train --dataset liar --epochs 3
    python main.py evaluate --model data/models/fake_news_detector
    python main.py dashboard
    python main.py db init
    python main.py db history
"""

import os
import logging

import typer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = typer.Typer(
    name="thumalien",
    help="Thumalien — Détection de Fake News sur Bluesky",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Gestion de la base de données")
app.add_typer(db_app, name="db")


# ============================================================
# ANALYSE
# ============================================================

@app.command()
def analyze(
    query: str = typer.Argument(..., help="Terme de recherche Bluesky"),
    lang: str = typer.Option(None, "--lang", "-l", help="Filtrer par langue (fr, en)"),
    limit: int = typer.Option(25, "--limit", "-n", help="Nombre de posts à analyser"),
    output: str = typer.Option("data/processed/results.json", "--output", "-o", help="Fichier de sortie JSON"),
    save_db: bool = typer.Option(False, "--save-db", help="Sauvegarder en base de données"),
):
    """Analyse des posts Bluesky pour détecter les fake news."""
    from src.pipeline import ThumalienPipeline

    typer.echo(f"🔍 Analyse de '{query}' (lang={lang}, limit={limit})...")

    pipeline = ThumalienPipeline()
    results = pipeline.run_full_pipeline(limit_per_keyword=limit)

    if not results:
        typer.echo("⚠️  Aucun post trouvé.")
        raise typer.Exit(0)

    # Résumé
    fake = sum(1 for r in results if r.get("is_fake", False))
    real = len(results) - fake
    typer.echo(f"✅ {len(results)} posts analysés — 🔴 Fake: {fake} | 🟢 Réels: {real}")

    energy = pipeline.energy_tracker.get_summary()
    typer.echo(f"⚡ CO₂: {energy.get('total_co2_kg', 0):.6f} kg | Durée: {energy.get('total_duration_s', 0):.1f}s")


# ============================================================
# TRAINING (module master)
# ============================================================

@app.command()
def train(
    dataset: str = typer.Option("liar", "--dataset", "-d", help="Dataset (liar, custom)"),
    csv: str = typer.Option(None, "--csv", help="Chemin CSV si --dataset custom"),
    model: str = typer.Option("distilbert-base-multilingual-cased", "--model", "-m", help="Modèle HuggingFace de base"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Nombre d'époques"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Taille de batch"),
    lr: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    output: str = typer.Option("data/models/fake_news_detector", "--output", "-o", help="Dossier de sortie"),
):
    """Fine-tuning du classifieur fake news (nécessite datasets+accelerate)."""
    try:
        from src.training.train import train as run_training
    except ImportError:
        typer.echo("❌ Module training non disponible. Vérifiez que datasets et accelerate sont installés.")
        raise typer.Exit(1)

    typer.echo(f"🤖 Fine-tuning : modèle={model}, dataset={dataset}, epochs={epochs}")
    metrics = run_training(
        model_name=model,
        dataset_name=dataset,
        csv_path=csv,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    f1 = metrics.get("test_metrics", {}).get("eval_f1_macro", 0)
    typer.echo(f"✅ F1 macro : {f1:.4f} | Modèle sauvegardé dans {output}")


# ============================================================
# ÉVALUATION (module master)
# ============================================================

@app.command()
def evaluate(
    model_path: str = typer.Option("data/models/fake_news_detector", "--model", "-m"),
    dataset: str = typer.Option("liar", "--dataset", "-d"),
    csv: str = typer.Option(None, "--csv"),
):
    """Évaluation détaillée du modèle fine-tuné."""
    try:
        from src.training.evaluate import evaluate as run_eval
    except ImportError:
        typer.echo("❌ Module training non disponible.")
        raise typer.Exit(1)

    typer.echo(f"📊 Évaluation : modèle={model_path}, dataset={dataset}")
    report = run_eval(model_path=model_path, dataset_name=dataset, csv_path=csv)
    typer.echo(f"Accuracy : {report['metrics']['accuracy']:.4f}")
    typer.echo(f"F1 macro : {report['metrics']['f1_macro']:.4f}")


# ============================================================
# DASHBOARD
# ============================================================

@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port Streamlit"),
):
    """Lance le dashboard Streamlit."""
    import subprocess
    typer.echo(f"🚀 Dashboard sur http://localhost:{port}")
    subprocess.run(["streamlit", "run", "dashboard/Home.py", f"--server.port={port}"])


# ============================================================
# BASE DE DONNÉES
# ============================================================

@db_app.command("init")
def db_init():
    """Initialise les tables PostgreSQL (Python ORM + SQL)."""
    from src.database.db_connector import DatabaseConnector
    typer.echo("🗄️  Initialisation de la base de données...")
    db = DatabaseConnector()
    if db.test_connection():
        db.initialize_schema()
        typer.echo("✅ Schéma créé avec succès.")
    else:
        typer.echo("❌ Connexion PostgreSQL échouée. Vérifiez votre .env.", err=True)
        raise typer.Exit(1)


@db_app.command("history")
def db_history(limit: int = typer.Option(10, "--limit", "-n")):
    """Affiche l'historique des sessions d'analyse."""
    from src.database.db_connector import DatabaseConnector
    db = DatabaseConnector()
    sessions = db.get_analysis_sessions(limit=limit)

    if not sessions:
        typer.echo("Aucune session d'analyse trouvée.")
        return

    typer.echo(f"\n{'Date':<22} {'Posts':>6} {'Fake':>5} {'CO₂ (kg)':>10}")
    typer.echo("-" * 50)
    for s in sessions:
        typer.echo(f"{str(s.get('created_at', '?'))[:19]:<22} {s.get('num_posts', 0):>6} {s.get('num_fake', 0):>5} {s.get('co2_kg', 0):>10.6f}")


@db_app.command("stats")
def db_stats():
    """Affiche les statistiques de la base de données."""
    from src.database.db_connector import DatabaseConnector
    db = DatabaseConnector()
    stats = db.count_posts()
    typer.echo(f"\n📊 État de la base :")
    typer.echo(f"   Posts total : {stats.get('total', 0)}")
    typer.echo(f"   Posts FR    : {stats.get('fr', 0)}")
    typer.echo(f"   Posts EN    : {stats.get('en', 0)}")


if __name__ == "__main__":
    app()


# ============================================================
# LABELER AT PROTOCOL
# ============================================================

@app.command()
def labeler_setup():
    """Déclare le compte Bluesky comme Labeler Service (one-shot)."""
    from src.labeler.setup_labeler_account import setup_labeler
    setup_labeler()


@app.command()
def labeler(
    dry_run: bool = typer.Option(False, "--dry-run", help="Test sans émettre de labels"),
    lang: str = typer.Option(None, "--lang", "-l", help="Filtre langue(s), ex: 'fr' ou 'fr,en'"),
    keywords: str = typer.Option(None, "--keywords", "-k", help="Mots-clés filtres séparés par virgule"),
    threshold_fake: float = typer.Option(0.70, "--threshold-fake", help="Seuil confiance label fake"),
    threshold_douteux: float = typer.Option(0.50, "--threshold-douteux", help="Seuil confiance douteux"),
    emit_fiable: bool = typer.Option(False, "--emit-fiable", help="Émettre aussi le label fiable"),
):
    """Lance le service de labeling AT Protocol en temps réel (Firehose)."""
    import src.labeler.labeler_service as ls_module

    ls_module.THRESHOLD_FAKE    = threshold_fake
    ls_module.THRESHOLD_DOUTEUX = threshold_douteux
    ls_module.EMIT_FIABLE       = emit_fiable

    lang_filter = lang.split(",") if lang else None
    kw_filter   = keywords.split(",") if keywords else None

    service = ls_module.ThumalienLabeler(
        lang_filter=lang_filter,
        keyword_filter=kw_filter,
        dry_run=dry_run,
    )
    service.run()
