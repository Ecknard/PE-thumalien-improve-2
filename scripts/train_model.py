#!/usr/bin/env python3
"""
scripts/train_model.py
Entraîne le modèle de classification fake news.
Supporte baseline (rapide, ~30s) et BERT (précis, ~20-30min CPU).

Usage :
    python scripts/train_model.py                     # baseline sur labeled_posts.json
    python scripts/train_model.py --model bert        # DistilBERT
    python scripts/train_model.py --model both        # les deux
    python scripts/train_model.py --data custom.json  # dataset custom
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.classifier.fake_news_classifier import (
    BaselineClassifier, BERTClassifier,
    load_labeled_data, create_sample_dataset,
)
from src.monitoring.energy_tracker import EnergyTracker
from config import LABELED_DIR, MODELS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement modèle Thumalien")
    parser.add_argument("--model", choices=["baseline", "bert", "both"], default="baseline",
                        help="Modèle à entraîner (baseline=rapide, bert=précis)")
    parser.add_argument("--data", type=str, default=None,
                        help="Chemin vers le fichier JSON labellisé")
    args = parser.parse_args()

    tracker = EnergyTracker()

    # ── Chargement données ──
    print("📂 Chargement des données...")
    if args.data:
        texts, labels = load_labeled_data(Path(args.data))
    else:
        texts, labels = load_labeled_data()

    if not texts:
        print("⚠️  Pas de données labellisées — utilisation du dataset d'exemple.")
        print("   Ajoutez vos exemples dans data/labeled/labeled_posts.json")
        texts, labels = create_sample_dataset()

    n_fake = labels.count(0)
    n_real = labels.count(1)
    print(f"   {len(texts)} exemples : {n_fake} fake / {n_real} réels")

    # ── Avertissement dataset petit ──
    if len(texts) < 50:
        print(
            "\n⚠️  ATTENTION : Dataset trop petit ({} exemples).\n"
            "   Les métriques (ROC-AUC, F1) seront artificiellement bonnes\n"
            "   et ne représentent pas les performances réelles.\n"
            "   → Solution recommandée : make train-liar (12 000 exemples réels)\n"
            .format(len(texts))
        )

    if abs(n_fake - n_real) > len(texts) * 0.2:
        print(
            f"⚠️  Déséquilibre de classes détecté ({n_fake} fake vs {n_real} réels).\n"
            "   Le modèle peut être biaisé vers la classe majoritaire.\n"
        )

    results = {}

    # ── Baseline ──
    if args.model in ("baseline", "both"):
        print("\n🏃 Entraînement Baseline (TF-IDF + LogisticRegression)...")
        with tracker.track("train_baseline", n_samples=len(texts), model_name="tfidf-logreg"):
            clf = BaselineClassifier()
            metrics = clf.train(texts, labels)
        clf.save()
        results["baseline"] = metrics
        print(f"\n✅ Baseline sauvegardé : {BaselineClassifier.MODEL_PATH}")

        # Avertissement ROC-AUC artificiel
        if metrics.get("roc_auc", 0) >= 0.98 and metrics.get("n_test", 0) < 20:
            print(
                "\n🔴 ROC-AUC = {:.3f} sur {} exemples de test → résultat artificiel.\n"
                "   Ce n'est PAS representatif. Utilisez : make train-liar\n"
                .format(metrics["roc_auc"], metrics["n_test"])
            )

    # ── BERT ──
    if args.model in ("bert", "both"):
        print("\n🤖 Entraînement DistilBERT multilingue (CPU ~20-30 min)...")
        print("   Suivez la progression dans logs/thumalien.log")
        with tracker.track("train_bert", n_samples=len(texts), model_name="distilbert-multilingual"):
            clf_bert = BERTClassifier()
            metrics_bert = clf_bert.train(texts, labels)
        results["bert"] = metrics_bert

    # ── Rapport final ──
    print("\n" + "=" * 55)
    print("📊 RÉSULTATS D'ENTRAÎNEMENT")
    print("=" * 55)
    for name, m in results.items():
        status = ""
        if m.get("roc_auc", 0) >= 0.98 and m.get("n_test", 0) < 20:
            status = "  ⚠️  (dataset trop petit — valeurs non fiables)"
        print(f"\n  [{name.upper()}]{status}")
        print(f"   Accuracy  : {m.get('accuracy', 0):.1%}")
        print(f"   F1-Score  : {m.get('f1', 0):.1%}")
        print(f"   Précision : {m.get('precision', 0):.1%}")
        print(f"   Rappel    : {m.get('recall', 0):.1%}")
        if m.get("roc_auc"):
            print(f"   ROC-AUC   : {m['roc_auc']:.3f}")
        print(f"   Train/Test: {m.get('n_train', '?')} / {m.get('n_test', '?')} exemples")

    tracker.print_report()

    # Sauvegarder résumé
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Résumé sauvegardé : {summary_path}")

    if "baseline" in results and results["baseline"].get("roc_auc", 0) >= 0.98:
        print(
            "\n💡 Pour obtenir des métriques réalistes :\n"
            "   make train-liar   → fine-tuning DistilBERT sur dataset LIAR (12k exemples)\n"
        )


if __name__ == "__main__":
    main()
