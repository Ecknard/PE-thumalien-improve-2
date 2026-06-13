# Architecture Technique - Thumalien

## Vue d'ensemble

Thumalien est structure en **6 modules independants** relies par un pipeline central. Chaque module peut etre teste et utilise isolement.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DASHBOARD (Streamlit)                     │
│   Vue d'ensemble │ Details │ Emotions │ Green IT                │
└──────────┬──────────────────────────────────────────────────────┘
           │ session_state
┌──────────▼──────────────────────────────────────────────────────┐
│                     PIPELINE PRINCIPAL                            │
│  run(query, lang, limit) -> list[dict]                          │
└──┬───────┬───────────┬────────────┬──────────────┬──────────────┘
   │       │           │            │              │
   ▼       ▼           ▼            ▼              ▼
┌──────┐┌────────┐┌──────────┐┌──────────┐┌──────────────────┐
│Collec││Pretrait││Classific.││ Emotions ││  Explicabilite   │
│teur  ││ NLP    ││Fake News ││          ││                  │
└──┬───┘└───┬────┘└────┬─────┘└────┬─────┘└───────┬──────────┘
   │        │          │           │              │
   ▼        ▼          ▼           ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                   ENERGY TRACKER (CodeCarbon)                 │
│            Mesure CO2/kWh pour chaque etape                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Flux de donnees

```
1. COLLECTE
   Bluesky API (AT Protocol)
       │
       ▼
   Posts bruts (JSON)
   {uri, cid, author, text, lang, created_at, likes, reposts}
       │
2. PRETRAITEMENT
       │
       ▼
   Posts nettoyes
   {+ clean_text, tokens, detected_lang}
       │
3. CLASSIFICATION          4. EMOTIONS
       │                        │
       ▼                        ▼
   {+ credibility:             {+ emotion:
     label, confidence,          dominant_emotion,
     scores{fiable,              confidence,
            douteux,             scores{colere, degout,
            fake}}                      peur, joie, ...}}
       │
5. EXPLICABILITE (si douteux/fake)
       │
       ▼
   {+ explanation:
     top_words[{token, importance}]}
       │
6. EXPORT
       │
       ▼
   JSON / Dashboard / PostgreSQL
```

---

## Modules en detail

### 1. Collecteur (`src/collector/bluesky_client.py`)

**Responsabilite** : Interagir avec l'API Bluesky pour recuperer des posts.

- **Protocole** : AT Protocol via la librairie `atproto`
- **Authentification** : Handle + App Password
- **Methodes** :
  - `search_posts(query, lang, limit)` : recherche par mot-cle avec pagination
  - `get_timeline(limit)` : posts du fil d'actualite
  - `_normalize_post(post_view)` : normalisation en dict standard

**Choix technique** : `atproto` est le client officiel Python pour le AT Protocol, maintenu activement et compatible avec l'API Bluesky.

### 2. Pretraitement NLP (`src/preprocessing/text_processor.py`)

**Responsabilite** : Nettoyer et tokeniser les textes pour les modeles.

- **Nettoyage** : suppression URLs, mentions, normalisation espaces, lowercase
- **Tokenisation** : spaCy avec filtrage stop words et ponctuation
- **Lemmatisation** : reduction des mots a leur forme canonique
- **Langues** : francais (`fr_core_news_sm`) et anglais (`en_core_web_sm`)
- **Cache** : les modeles spaCy sont charges une seule fois

**Choix technique** : spaCy offre des performances superieures a NLTK pour la tokenisation et la lemmatisation, avec un support multilingue natif.

### 3. Classifieur Fake News (`src/models/fake_news_detector.py`)

**Responsabilite** : Classifier les posts en 3 categories de credibilite.

- **Modele** : RoBERTa (base) via HuggingFace Transformers
- **Labels** : `fiable`, `douteux`, `fake`
- **Sortie** : label predit + score de confiance + probabilites par classe
- **Batch** : traitement par lots pour optimiser les performances GPU
- **Device** : detection automatique CUDA/CPU

**Choix technique** : RoBERTa surpasse BERT sur les taches de classification de texte. Le modele de base (125M parametres) offre un bon compromis performance/ressources pour un projet etudiant.

**Piste d'amelioration** : Fine-tuning sur un dataset de fake news en francais (ex: FakeNewsNet adapte, LIAR dataset).

### 4. Analyseur d'emotions (`src/models/emotion_analyzer.py`)

**Responsabilite** : Detecter l'emotion dominante dans chaque post.

- **Modele** : `j-hartmann/emotion-english-distilroberta-base`
- **7 emotions** : colere, degout, peur, joie, tristesse, surprise, neutre
- **Mapping** : labels anglais -> francais
- **Batch** : traitement par lots

**Choix technique** : Ce modele pre-entraine sur GoEmotions est l'un des plus performants pour la classification d'emotions en open source. DistilRoBERTa offre un bon compromis vitesse/precision.

### 5. Explicabilite (`src/explainability/explainer.py`)

**Responsabilite** : Expliquer pourquoi le modele classe un post comme douteux/fake.

- **Methode** : Analyse des poids d'attention (attention-based)
- **Processus** :
  1. Forward pass avec `output_attentions=True`
  2. Moyenne des attentions sur toutes les couches et tetes
  3. Importance = somme des attentions recues par chaque token
  4. Normalisation et tri par importance decroissante
- **Sortie** : top 10 tokens les plus influents avec score d'importance

**Choix technique** : L'analyse par attention est la methode la plus legere pour l'explicabilite. Des alternatives plus robustes (LIME, SHAP) pourraient etre ajoutees mais sont significativement plus lentes.

### 6. Monitoring energetique (`src/monitoring/energy_tracker.py`)

**Responsabilite** : Mesurer l'empreinte carbone de chaque etape du pipeline.

- **Outil** : CodeCarbon (emissions CO2, consommation kWh)
- **Usage** : context manager `with tracker.track("nom_tache")`
- **Metriques** : duree, emissions CO2 (kg), energie (kWh)
- **Export** : rapport JSON dans `data/monitoring/`

**Choix technique** : CodeCarbon est la reference open-source pour le suivi energetique en ML/IA, recommande par la communaute Green AI.

---

## Base de donnees

### Schema PostgreSQL

```
┌─────────────────────────┐     ┌──────────────────────────┐
│         posts            │     │        analyses           │
├─────────────────────────┤     ├──────────────────────────┤
│ id (PK, SERIAL)         │◄────│ post_id (FK)             │
│ uri (UNIQUE)            │     │ id (PK, SERIAL)          │
│ cid                     │     │ credibility_label        │
│ author_handle           │     │ credibility_score        │
│ author_display_name     │     │ scores_detail (JSONB)    │
│ text_content            │     │ dominant_emotion         │
│ clean_text              │     │ emotion_scores (JSONB)   │
│ lang                    │     │ explanation (JSONB)      │
│ created_at              │     │ analyzed_at              │
│ collected_at            │     └──────────────────────────┘
│ like_count              │
│ repost_count            │     ┌──────────────────────────┐
│ reply_count             │     │     energy_reports        │
└─────────────────────────┘     ├──────────────────────────┤
                                │ id (PK, SERIAL)          │
                                │ task_name                │
                                │ duration_seconds         │
                                │ emissions_kg_co2         │
                                │ energy_kwh               │
                                │ recorded_at              │
                                └──────────────────────────┘
```

### Index

- `idx_posts_author` : recherche par auteur
- `idx_posts_created` : tri chronologique
- `idx_analyses_label` : filtrage par credibilite
- `idx_analyses_post` : jointure analyses -> posts

---

## Infrastructure

### Docker

- **app** : Python 3.11-slim + dependances + modeles spaCy
- **db** : PostgreSQL 16 Alpine
- **Volumes** : `pgdata` (persistance BDD), `data/` et `src/` (hot reload)

### Variables d'environnement

| Variable | Description | Exemple |
|---|---|---|
| `BLUESKY_HANDLE` | Handle Bluesky | `user.bsky.social` |
| `BLUESKY_PASSWORD` | App Password Bluesky | `xxxx-xxxx-xxxx-xxxx` |
| `POSTGRES_DB` | Nom de la BDD | `thumalien` |
| `POSTGRES_USER` | Utilisateur BDD | `thumalien` |
| `POSTGRES_PASSWORD` | Mot de passe BDD | `thumalien_secret` |

---

## Performances et limites

### Performances attendues

| Metrique | Valeur cible |
|---|---|
| Temps de classification | < 100ms / post (GPU) |
| Temps d'analyse emotions | < 80ms / post (GPU) |
| Temps de collecte | ~1s / 25 posts |
| Memoire modeles | ~1.5 Go (2 modeles Transformer) |

### Limites connues

- Le classifieur n'est **pas fine-tune** : les scores de credibilite sont bases sur un modele pre-entraine generique. Un fine-tuning sur un dataset de fake news est necessaire pour des resultats fiables.
- L'analyseur d'emotions est **optimise pour l'anglais**. Les resultats en francais sont exploitables mais moins precis.
- L'explicabilite par attention est une **approximation**. Elle indique les tokens les plus "regardes" par le modele, pas forcement ceux qui "causent" la decision.
- La collecte est limitee par les **rate limits** de l'API Bluesky.

---

## Evolutions futures

- Fine-tuning du classifieur sur un dataset de fake news FR/EN
- Modele d'emotions multilingue (XLM-RoBERTa)
- Explicabilite avancee (LIME, SHAP)
- Scalabilite avec Apache Spark / Kafka
- Stockage des embeddings pour la recherche semantique
- Alertes en temps reel (webhooks)
