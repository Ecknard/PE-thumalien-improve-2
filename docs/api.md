# Reference API - Thumalien

Documentation des modules, classes et methodes du projet.

## Sommaire

- [Pipeline](#pipeline)
- [Collecteur Bluesky](#collecteur-bluesky)
- [Pretraitement NLP](#pretraitement-nlp)
- [Classifieur Fake News](#classifieur-fake-news)
- [Analyseur d'emotions](#analyseur-demotions)
- [Explicabilite](#explicabilite)
- [Monitoring energetique](#monitoring-energetique)

---

## Pipeline

**Module** : `src/pipeline.py`

### `ThumalienPipeline`

Pipeline complet reliant tous les modules.

#### `__init__(bluesky_handle, bluesky_password, fake_news_model, emotion_model)`

| Parametre | Type | Default | Description |
|---|---|---|---|
| `bluesky_handle` | `str` | *requis* | Handle Bluesky (ex: `user.bsky.social`) |
| `bluesky_password` | `str` | *requis* | App Password Bluesky |
| `fake_news_model` | `str` | `"roberta-base"` | Modele HuggingFace pour la classification |
| `emotion_model` | `str` | `"j-hartmann/emotion-english-distilroberta-base"` | Modele HuggingFace pour les emotions |

#### `run(query, lang=None, limit=50) -> list[dict]`

Execute le pipeline complet : collecte -> pretraitement -> classification -> emotions -> explicabilite.

| Parametre | Type | Description |
|---|---|---|
| `query` | `str` | Terme de recherche Bluesky |
| `lang` | `str \| None` | Filtre de langue (`"fr"`, `"en"`, `None` pour toutes) |
| `limit` | `int` | Nombre max de posts a analyser |

**Retour** : Liste de dicts enrichis avec les champs :

```python
{
    # Champs du post original
    "uri": str,
    "author_handle": str,
    "text": str,
    # Champs de pretraitement
    "clean_text": str,
    "tokens": list[str],
    # Classification
    "credibility": {
        "label": "fiable" | "douteux" | "fake",
        "confidence": float,
        "scores": {"fiable": float, "douteux": float, "fake": float}
    },
    # Emotions
    "emotion": {
        "dominant_emotion": str,
        "confidence": float,
        "scores": {"colere": float, "joie": float, ...}
    },
    # Explicabilite (si douteux/fake avec confiance > 0.6)
    "explanation": {
        "top_words": [{"token": str, "importance": float}, ...]
    } | None
}
```

#### `export_results(results, output_path="data/processed/results.json")`

Exporte les resultats en JSON avec timestamp et rapport energetique.

---

## Collecteur Bluesky

**Module** : `src/collector/bluesky_client.py`

### `BlueskyCollector`

#### `__init__(handle, password)`

Se connecte a Bluesky via le AT Protocol.

| Parametre | Type | Description |
|---|---|---|
| `handle` | `str` | Handle Bluesky |
| `password` | `str` | App Password |

**Leve** : `atproto.exceptions.UnauthorizedError` si les identifiants sont invalides.

#### `search_posts(query, lang=None, limit=100) -> list[dict]`

Recherche des posts par mot-cle avec pagination automatique.

| Parametre | Type | Description |
|---|---|---|
| `query` | `str` | Terme de recherche |
| `lang` | `str \| None` | Filtre de langue |
| `limit` | `int` | Nombre max de posts |

**Retour** : Liste de posts normalises.

#### `get_timeline(limit=50) -> list[dict]`

Recupere les posts du fil d'actualite de l'utilisateur connecte.

#### Format d'un post normalise

```python
{
    "uri": "at://did:plc:xxx/app.bsky.feed.post/yyy",
    "cid": "bafyreiabc123",
    "author_handle": "user.bsky.social",
    "author_display_name": "User Name",
    "text": "Contenu du post",
    "lang": ["fr"],
    "created_at": "2025-01-01T00:00:00Z",
    "collected_at": "2025-01-01T00:01:00Z",
    "like_count": 5,
    "repost_count": 2,
    "reply_count": 1
}
```

---

## Pretraitement NLP

**Module** : `src/preprocessing/text_processor.py`

### `clean_text(text: str) -> str`

Nettoie un texte brut :
- Supprime les URLs (`https://...`)
- Supprime les mentions (`@handle`)
- Normalise les espaces
- Convertit en minuscules

```python
>>> clean_text("BREAKING @user https://lien.com  La nouvelle est FAUSSE")
"breaking la nouvelle est fausse"
```

### `tokenize(text: str, lang: str = "fr") -> list[str]`

Tokenise avec spaCy en filtrant stop words et ponctuation. Retourne les lemmes.

```python
>>> tokenize("les chats mangent des souris", lang="fr")
["chat", "manger", "souris"]
```

### `preprocess_post(post: dict) -> dict`

Pipeline complet pour un post. Ajoute `clean_text`, `tokens`, `detected_lang`.

### `preprocess_batch(posts: list[dict]) -> list[dict]`

Applique `preprocess_post` a une liste de posts.

### `get_nlp(lang: str = "fr") -> spacy.language.Language`

Charge et met en cache le modele spaCy. Langues supportees : `"fr"`, `"en"`.

---

## Classifieur Fake News

**Module** : `src/models/fake_news_detector.py`

### Constantes

```python
LABELS = ["fiable", "douteux", "fake"]
```

### `FakeNewsDetector`

#### `__init__(model_name: str = "roberta-base")`

Charge un modele Transformer pour la classification en 3 classes.

| Parametre | Type | Description |
|---|---|---|
| `model_name` | `str` | Nom du modele HuggingFace ou chemin local |

#### `predict(text: str) -> dict`

Predit la credibilite d'un texte.

```python
>>> detector = FakeNewsDetector()
>>> detector.predict("Les aliens ont envahi la Terre")
{
    "label": "fake",
    "confidence": 0.87,
    "scores": {"fiable": 0.05, "douteux": 0.08, "fake": 0.87}
}
```

#### `predict_batch(texts: list[str], batch_size: int = 16) -> list[dict]`

Classification par lots. Plus efficace que des appels individuels.

---

## Analyseur d'emotions

**Module** : `src/models/emotion_analyzer.py`

### Constantes

```python
EMOTION_LABELS = ["colere", "degout", "peur", "joie", "tristesse", "surprise", "neutre"]
```

### `EmotionAnalyzer`

#### `__init__(model_name: str = "j-hartmann/emotion-english-distilroberta-base")`

Charge le modele d'analyse emotionnelle.

#### `analyze(text: str) -> dict`

Analyse les emotions d'un texte.

```python
>>> analyzer = EmotionAnalyzer()
>>> analyzer.analyze("C'est scandaleux, je suis furieux !")
{
    "dominant_emotion": "colere",
    "confidence": 0.82,
    "scores": {
        "colere": 0.82,
        "degout": 0.07,
        "peur": 0.02,
        "joie": 0.01,
        "tristesse": 0.04,
        "surprise": 0.02,
        "neutre": 0.02
    }
}
```

#### `analyze_batch(texts: list[str], batch_size: int = 16) -> list[dict]`

Analyse par lots.

---

## Explicabilite

**Module** : `src/explainability/explainer.py`

### `PredictionExplainer`

#### `__init__(model, tokenizer)`

| Parametre | Type | Description |
|---|---|---|
| `model` | `AutoModelForSequenceClassification` | Modele de classification |
| `tokenizer` | `AutoTokenizer` | Tokenizer associe |

Utilise le modele et le tokenizer du `FakeNewsDetector` :

```python
explainer = PredictionExplainer(detector.model, detector.tokenizer)
```

#### `explain(text: str, target_class: int = None) -> dict`

Genere une explication basee sur les poids d'attention.

| Parametre | Type | Description |
|---|---|---|
| `text` | `str` | Texte a expliquer |
| `target_class` | `int \| None` | Classe cible (None = classe predite) |

**Retour** :

```python
{
    "text": str,
    "predicted_class": int,
    "confidence": float,
    "top_influential_words": [
        {"token": "urgent", "importance": 0.15, "importance_normalized": 1.0},
        {"token": "cache", "importance": 0.12, "importance_normalized": 0.8},
        ...
    ],
    "all_word_importances": [...]  # Tous les tokens
}
```

---

## Monitoring energetique

**Module** : `src/monitoring/energy_tracker.py`

### `EnergyTracker`

#### `__init__(project_name="thumalien", output_dir="data/monitoring")`

| Parametre | Type | Description |
|---|---|---|
| `project_name` | `str` | Nom du projet pour CodeCarbon |
| `output_dir` | `Path` | Dossier de sortie des rapports |

#### `track(task_name: str)` (context manager)

Mesure la consommation energetique d'un bloc de code.

```python
tracker = EnergyTracker()

with tracker.track("classification"):
    results = detector.predict_batch(texts)

with tracker.track("emotions"):
    emotions = analyzer.analyze_batch(texts)
```

#### `get_summary() -> dict`

Retourne un resume de toutes les mesures.

```python
{
    "total_emissions_kg_co2": 0.000123,
    "total_duration_seconds": 45.2,
    "total_energy_kwh": 0.000456,
    "num_tasks": 5,
    "tasks": [
        {
            "task": "classification",
            "start_time": "...",
            "end_time": "...",
            "duration_seconds": 12.3,
            "emissions_kg_co2": 0.000045,
            "energy_kwh": 0.000123
        },
        ...
    ]
}
```

#### `save_report(filename="energy_report.json")`

Sauvegarde le rapport dans `output_dir/filename`.
