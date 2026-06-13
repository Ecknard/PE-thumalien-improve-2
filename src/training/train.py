"""Script de fine-tuning du classifieur fake news.

Usage :
    # Fine-tuning sur LIAR
    python -m src.training.train --dataset liar --epochs 3

    # Fine-tuning sur un CSV personnalisé
    python -m src.training.train --dataset custom --csv data/raw/mon_dataset.csv --epochs 5

    # Fine-tuning sur LIAR + Kaggle Fake News combinés
    python -m src.training.train --dataset liar+kaggle --epochs 3

    # Avec modèle de base différent (ex: CamemBERT pour le français)
    python -m src.training.train --dataset custom --csv data.csv --model camembert-base
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.training.dataset import (
    load_liar_dataset,
    load_fake_news_kaggle,
    load_custom_csv,
    merge_datasets,
    get_dataset_stats,
    LABEL2ID,
    ID2LABEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "roberta-base"
DEFAULT_OUTPUT_DIR = "data/models/fake_news_detector"


def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenise un DatasetDict avec le tokenizer donné."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    """Calcule les métriques d'évaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }


def train(
    model_name: str = DEFAULT_MODEL,
    dataset_name: str = "liar",
    csv_path: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    early_stopping_patience: int = 2,
    fp16: bool = True,
):
    """Lance le fine-tuning.

    Args:
        model_name: Modèle HuggingFace de base.
        dataset_name: 'liar', 'kaggle', 'liar+kaggle', ou 'custom'.
        csv_path: Chemin CSV (requis si dataset_name='custom').
        output_dir: Dossier de sauvegarde du modèle fine-tuné.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        learning_rate: Taux d'apprentissage.
        weight_decay: Régularisation L2.
        warmup_ratio: Ratio de warmup du learning rate.
        max_length: Longueur max des tokens.
        early_stopping_patience: Patience pour l'early stopping.
        fp16: Utiliser le mixed precision (GPU uniquement).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = fp16 and device == "cuda"

    logger.info("=== Configuration ===")
    logger.info("Modèle de base  : %s", model_name)
    logger.info("Dataset         : %s", dataset_name)
    logger.info("Device          : %s", device)
    logger.info("Époques         : %d", epochs)
    logger.info("Batch size      : %d", batch_size)
    logger.info("Learning rate   : %s", learning_rate)
    logger.info("FP16            : %s", use_fp16)
    logger.info("Output          : %s", output_dir)

    # 1. Chargement du dataset
    logger.info("Chargement du dataset '%s'...", dataset_name)
    if dataset_name == "liar":
        dataset = load_liar_dataset()
    elif dataset_name == "kaggle":
        dataset = load_fake_news_kaggle()
    elif dataset_name == "liar+kaggle":
        liar = load_liar_dataset()
        kaggle = load_fake_news_kaggle()
        dataset = merge_datasets(liar, kaggle)
    elif dataset_name == "custom":
        if not csv_path:
            raise ValueError("--csv requis pour dataset_name='custom'")
        dataset = load_custom_csv(csv_path)
    else:
        raise ValueError(f"Dataset inconnu : {dataset_name}. Choix : liar, kaggle, liar+kaggle, custom")

    # Afficher les stats
    stats = get_dataset_stats(dataset)
    for split, info in stats.items():
        logger.info("  %s : %d exemples — %s", split, info["total"], info["distribution"])

    # 2. Tokenisation
    logger.info("Tokenisation avec '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    # 3. Chargement du modèle
    logger.info("Chargement du modèle '%s' avec %d labels...", model_name, len(LABEL2ID))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4. Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_path / "logs"),
        logging_steps=50,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
        seed=42,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # 6. Entraînement
    logger.info("Lancement du fine-tuning...")
    train_result = trainer.train()

    # 7. Évaluation sur le test set
    logger.info("Évaluation sur le test set...")
    test_results = trainer.evaluate(tokenized["test"])

    # 8. Sauvegarde
    logger.info("Sauvegarde du modèle fine-tuné dans '%s'...", output_dir)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Sauvegarder les métriques
    metrics = {
        "model_name": model_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_metrics": {k: float(v) for k, v in train_result.metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_results.items()},
        "dataset_stats": stats,
    }

    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("=== Résultats ===")
    logger.info("Train loss      : %.4f", train_result.metrics.get("train_loss", 0))
    logger.info("Test accuracy   : %.4f", test_results.get("eval_accuracy", 0))
    logger.info("Test F1 (macro) : %.4f", test_results.get("eval_f1_macro", 0))
    logger.info("Test F1 (weight): %.4f", test_results.get("eval_f1_weighted", 0))
    logger.info("Test precision  : %.4f", test_results.get("eval_precision_macro", 0))
    logger.info("Test recall     : %.4f", test_results.get("eval_recall_macro", 0))
    logger.info("Modèle sauvegardé dans : %s", output_dir)
    logger.info("Métriques sauvegardées dans : %s", metrics_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning du classifieur fake news Thumalien")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modèle HuggingFace de base")
    parser.add_argument("--dataset", default="liar", choices=["liar", "kaggle", "liar+kaggle", "custom"],
                        help="Dataset à utiliser")
    parser.add_argument("--csv", default=None, help="Chemin CSV (requis si --dataset custom)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Dossier de sortie")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Longueur max des tokens")
    parser.add_argument("--no-fp16", action="store_true", help="Désactiver le mixed precision")

    args = parser.parse_args()

    train(
        model_name=args.model,
        dataset_name=args.dataset,
        csv_path=args.csv,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
