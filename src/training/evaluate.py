"""Évaluation détaillée du modèle fine-tuné.

Usage :
    # Évaluer sur le test set LIAR
    python -m src.training.evaluate --model data/models/fake_news_detector --dataset liar

    # Évaluer sur un CSV personnalisé
    python -m src.training.evaluate --model data/models/fake_news_detector --dataset custom --csv test.csv

    # Sauvegarder le rapport
    python -m src.training.evaluate --model data/models/fake_news_detector --dataset liar --output rapport.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.training.dataset import (
    load_liar_dataset,
    load_fake_news_kaggle,
    load_custom_csv,
    LABEL2ID,
    ID2LABEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_batch(model, tokenizer, texts, batch_size=32, device="cpu"):
    """Prédit les labels pour une liste de textes."""
    model.eval()
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=256, padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Génère et sauvegarde la matrice de confusion."""
    labels = list(ID2LABEL.values())
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Matrice de confusion sauvegardée : %s", output_path)


def plot_confidence_distribution(y_true, probs, output_path):
    """Génère la distribution des scores de confiance par classe."""
    labels = list(ID2LABEL.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, label in enumerate(labels):
        mask = y_true == idx
        if mask.sum() == 0:
            continue
        confidences = probs[mask, idx]
        axes[idx].hist(confidences, bins=20, alpha=0.7, edgecolor="black", color=["#2ecc71", "#f39c12", "#e74c3c"][idx])
        axes[idx].set_title(f"Confiance pour '{label}'")
        axes[idx].set_xlabel("Score de confiance")
        axes[idx].set_ylabel("Fréquence")
        axes[idx].axvline(confidences.mean(), color="black", linestyle="--",
                          label=f"Moyenne: {confidences.mean():.2f}")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Distribution de confiance sauvegardée : %s", output_path)


def evaluate(
    model_path: str,
    dataset_name: str = "liar",
    csv_path: str | None = None,
    output_path: str | None = None,
    batch_size: int = 32,
):
    """Évaluation complète du modèle.

    Args:
        model_path: Chemin du modèle fine-tuné.
        dataset_name: Dataset pour l'évaluation.
        csv_path: Chemin CSV si dataset_name='custom'.
        output_path: Chemin de sortie du rapport JSON.
        batch_size: Taille de batch pour l'inférence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    logger.info("Chargement du modèle depuis '%s'...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Charger le dataset
    logger.info("Chargement du dataset '%s'...", dataset_name)
    if dataset_name == "liar":
        dataset = load_liar_dataset()
    elif dataset_name == "kaggle":
        dataset = load_fake_news_kaggle()
    elif dataset_name == "custom":
        if not csv_path:
            raise ValueError("--csv requis pour dataset_name='custom'")
        dataset = load_custom_csv(csv_path)
    else:
        raise ValueError(f"Dataset inconnu : {dataset_name}")

    test_data = dataset["test"]
    texts = test_data["text"]
    y_true = np.array(test_data["label"])

    # Prédiction
    logger.info("Prédiction sur %d exemples...", len(texts))
    y_pred, probs = predict_batch(model, tokenizer, texts, batch_size=batch_size, device=str(device))

    # Métriques
    labels = list(ID2LABEL.values())
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    # Affichage
    logger.info("\n=== Rapport de classification ===")
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=labels))
    logger.info("Accuracy        : %.4f", accuracy)
    logger.info("F1 (macro)      : %.4f", f1_macro)
    logger.info("F1 (weighted)   : %.4f", f1_weighted)
    logger.info("Precision macro : %.4f", precision_macro)
    logger.info("Recall macro    : %.4f", recall_macro)
    if auc:
        logger.info("AUC (macro)     : %.4f", auc)

    # Analyse des erreurs
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "text": texts[i][:200],
                "true_label": ID2LABEL[int(y_true[i])],
                "predicted_label": ID2LABEL[int(y_pred[i])],
                "confidence": float(probs[i].max()),
            })

    # Trier par confiance décroissante (erreurs les plus confiantes = les plus problématiques)
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    logger.info("\n=== Top 10 erreurs les plus confiantes ===")
    for j, err in enumerate(errors[:10], 1):
        logger.info(
            "  %d. [%s -> %s] (%.0f%%) %s",
            j, err["true_label"], err["predicted_label"],
            err["confidence"] * 100, err["text"][:80],
        )

    # Graphiques
    model_dir = Path(model_path)
    plot_confusion_matrix(y_true, y_pred, model_dir / "confusion_matrix.png")
    plot_confidence_distribution(y_true, probs, model_dir / "confidence_distribution.png")

    # Rapport JSON
    full_report = {
        "model_path": model_path,
        "dataset": dataset_name,
        "num_test_samples": len(y_true),
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "auc_macro": auc,
        },
        "per_class": {k: v for k, v in report.items() if k in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "top_errors": errors[:20],
    }

    save_path = output_path or str(model_dir / "evaluation_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    logger.info("Rapport sauvegardé : %s", save_path)
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Évaluation du classifieur fake news Thumalien")
    parser.add_argument("--model", required=True, help="Chemin du modèle fine-tuné")
    parser.add_argument("--dataset", default="liar", choices=["liar", "kaggle", "custom"])
    parser.add_argument("--csv", default=None, help="Chemin CSV (si --dataset custom)")
    parser.add_argument("--output", default=None, help="Chemin du rapport JSON")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        dataset_name=args.dataset,
        csv_path=args.csv,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
