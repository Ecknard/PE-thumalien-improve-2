"""Préparation des datasets pour le fine-tuning du classifieur fake news.

Datasets supportés :
- LIAR : dataset de fact-checking (6 labels -> 3 labels)
- Custom CSV : dataset personnalisé (colonnes text + label)
- HuggingFace : tout dataset compatible avec `datasets` library

Labels cibles : fiable (0), douteux (1), fake (2)
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

LABEL2ID = {"fiable": 0, "douteux": 1, "fake": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping LIAR (6 labels) -> Thumalien (3 labels)
LIAR_LABEL_MAP = {
    "true": "fiable",
    "mostly-true": "fiable",
    "half-true": "douteux",
    "barely-true": "douteux",
    "false": "fake",
    "pants-fire": "fake",
}


def load_liar_dataset() -> DatasetDict:
    """Charge le dataset LIAR et mappe les labels vers nos 3 classes.

    Le dataset LIAR contient des déclarations politiques annotées
    par PolitiFact avec 6 niveaux de véracité.

    Returns:
        DatasetDict avec splits train/validation/test.
    """
    logger.info("Chargement du dataset LIAR...")
    ds = load_dataset("liar")

    def map_labels(example):
        original_label = ds["train"].features["label"].int2str(example["label"])
        mapped = LIAR_LABEL_MAP.get(original_label, "douteux")
        example["text"] = example["statement"]
        example["label"] = LABEL2ID[mapped]
        return example

    mapped = ds.map(map_labels)
    mapped = mapped.remove_columns(
        [c for c in mapped["train"].column_names if c not in ("text", "label")]
    )

    logger.info(
        "LIAR chargé : %d train, %d val, %d test",
        len(mapped["train"]), len(mapped["validation"]), len(mapped["test"]),
    )
    return mapped


def load_fake_news_kaggle() -> DatasetDict:
    """Charge le dataset Fake News de Kaggle (GonzaloA/fake_news).

    Dataset de ~45k articles de news labellisés fake (1) ou real (0).
    On mappe : real -> fiable, fake -> fake (pas de "douteux").

    Returns:
        DatasetDict avec splits train/validation/test.
    """
    logger.info("Chargement du dataset Fake News Kaggle...")
    ds = load_dataset("GonzaloA/fake_news")

    def map_labels(example):
        text = example.get("text") or example.get("title") or ""
        # Label 1 = fake, 0 = real
        label = LABEL2ID["fake"] if example["label"] == 1 else LABEL2ID["fiable"]
        return {"text": text[:512], "label": label}

    mapped = ds.map(map_labels)
    mapped = mapped.remove_columns(
        [c for c in mapped["train"].column_names if c not in ("text", "label")]
    )

    # Filtrer les textes vides
    mapped = mapped.filter(lambda x: len(x["text"].strip()) > 10)

    # Créer un split validation si absent
    if "validation" not in mapped:
        split = mapped["train"].train_test_split(test_size=0.1, seed=42)
        mapped = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
            "test": mapped.get("test", split["test"]),
        })

    logger.info(
        "Fake News Kaggle chargé : %d train, %d val",
        len(mapped["train"]), len(mapped["validation"]),
    )
    return mapped


def load_custom_csv(
    csv_path: str,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> DatasetDict:
    """Charge un dataset personnalisé depuis un fichier CSV.

    Le fichier doit contenir au minimum une colonne texte et une colonne label.
    Les labels peuvent être :
    - Numériques : 0 (fiable), 1 (douteux), 2 (fake)
    - Textuels : "fiable", "douteux", "fake"

    Args:
        csv_path: Chemin vers le fichier CSV.
        text_column: Nom de la colonne contenant le texte.
        label_column: Nom de la colonne contenant le label.
        test_size: Proportion pour le set de test.
        val_size: Proportion pour le set de validation.

    Returns:
        DatasetDict avec splits train/validation/test.
    """
    logger.info("Chargement du CSV : %s", csv_path)
    df = pd.read_csv(csv_path)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Colonnes requises : '{text_column}' et '{label_column}'. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df = df[[text_column, label_column]].dropna()
    df.columns = ["text", "label"]

    # Conversion des labels textuels en numériques
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(LABEL2ID)
        if df["label"].isna().any():
            unknown = df[df["label"].isna()]["label"].unique()
            raise ValueError(f"Labels inconnus : {unknown}. Attendus : {list(LABEL2ID.keys())}")

    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)

    # Filtrer les textes vides
    df = df[df["text"].str.strip().str.len() > 10]

    # Splits
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=train_df["label"])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })

    logger.info(
        "CSV chargé : %d train, %d val, %d test",
        len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]),
    )
    return dataset


def merge_datasets(*datasets: DatasetDict) -> DatasetDict:
    """Fusionne plusieurs DatasetDict en un seul (par split).

    Utile pour combiner LIAR (anglais) avec un dataset français personnalisé.
    """
    merged = {}
    for split in ("train", "validation", "test"):
        split_datasets = [ds[split] for ds in datasets if split in ds]
        if split_datasets:
            merged[split] = concatenate_datasets(split_datasets).shuffle(seed=42)

    result = DatasetDict(merged)
    logger.info(
        "Datasets fusionnés : %d train, %d val, %d test",
        len(result.get("train", [])),
        len(result.get("validation", [])),
        len(result.get("test", [])),
    )
    return result


def get_dataset_stats(dataset: DatasetDict) -> dict:
    """Retourne des statistiques sur le dataset."""
    stats = {}
    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        total = len(labels)
        distribution = {}
        for label_id, label_name in ID2LABEL.items():
            count = labels.count(label_id)
            distribution[label_name] = {"count": count, "pct": f"{count / total * 100:.1f}%"}
        stats[split_name] = {"total": total, "distribution": distribution}
    return stats
