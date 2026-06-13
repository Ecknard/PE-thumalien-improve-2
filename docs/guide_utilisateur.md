# Guide Utilisateur - Thumalien

## Sommaire

1. [Premiers pas](#1-premiers-pas)
2. [Utiliser le dashboard](#2-utiliser-le-dashboard)
3. [Utiliser le pipeline en CLI](#3-utiliser-le-pipeline-en-cli)
4. [Utiliser le notebook](#4-utiliser-le-notebook)
5. [Comprendre les resultats](#5-comprendre-les-resultats)
6. [FAQ](#6-faq)

---

## 1. Premiers pas

### Creer un App Password Bluesky

1. Connectez-vous sur [bsky.app](https://bsky.app)
2. Allez dans **Parametres** > **App Passwords**
3. Cliquez **Ajouter un App Password**
4. Donnez un nom (ex: "thumalien") et copiez le mot de passe genere

### Configurer le projet

```bash
cp .env.example .env
```

Editez le fichier `.env` :

```
BLUESKY_HANDLE=votre-handle.bsky.social
BLUESKY_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

### Lancer avec Docker

```bash
docker-compose up --build
```

Premiere execution : le build telecharge les modeles (~1-2 Go), cela peut prendre quelques minutes.

### Lancer sans Docker

```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Lancer le dashboard
streamlit run src/dashboard/app.py
```

---

## 2. Utiliser le dashboard

### Acceder au dashboard

Ouvrez votre navigateur sur **http://localhost:8501**

### Configurer une analyse

Dans la **sidebar gauche** :

1. **Identifiants Bluesky** : deja remplis si vous avez configure `.env`. Sinon, entrez votre handle et App Password directement.

2. **Recherche** : entrez un mot-cle ou une phrase. Exemples :
   - `fake news`
   - `vaccin covid`
   - `elections 2025`
   - `rechauffement climatique`

3. **Langue** : filtrez par langue
   - **Toutes** : posts en toutes langues
   - **Francais** : posts en francais uniquement
   - **English** : posts en anglais uniquement

4. **Nombre de posts** : de 10 a 200 posts a analyser

5. Cliquez sur **Lancer l'analyse**

### Onglet "Vue d'ensemble"

Cet onglet donne une vision globale :

- **4 metriques** : nombre total de posts, nombre de fiables/douteux/fake avec pourcentages
- **Camembert** : repartition visuelle des 3 categories
- **Histogramme** : distribution des scores de confiance du modele
- **Top 5 posts suspects** : les posts avec le plus fort score "fake", avec :
  - Texte original
  - Emotion dominante
  - Mots influents (explicabilite)

### Onglet "Details"

Tableau interactif de tous les posts analyses :

| Colonne | Description |
|---|---|
| Auteur | Handle Bluesky |
| Texte | Extrait du post (120 caracteres) |
| Label | fiable / douteux / fake |
| Confiance | Score de confiance du modele |
| Emotion | Emotion dominante detectee |
| Mots influents | Tokens ayant le plus influence la classification |
| Likes / Reposts | Metriques d'engagement |

Vous pouvez **filtrer par label** avec le selecteur en haut du tableau.

### Onglet "Emotions"

Trois visualisations :

1. **Bar chart** : nombre de posts par emotion dominante
2. **Radar chart** : profil emotionnel moyen (scores moyens sur les 7 emotions)
3. **Histogramme croise** : quelles emotions sont associees aux fake news vs contenus fiables

### Onglet "Green IT"

Metriques environnementales de l'analyse :

- **Emissions CO2** : en kilogrammes de CO2 equivalent
- **Energie consommee** : en kilowattheures
- **Duree totale** : temps d'execution du pipeline
- **Repartition par etape** : quelle partie du pipeline consomme le plus (collecte, pretraitement, classification, emotions, explicabilite)

---

## 3. Utiliser le pipeline en CLI

### Execution simple

```bash
python main.py
```

Cela lance le pipeline avec les parametres par defaut (recherche "fake news", 50 posts, francais). Les resultats sont exportes dans `data/processed/results.json`.

### Execution personnalisee

```python
from src.pipeline import ThumalienPipeline

pipeline = ThumalienPipeline(
    bluesky_handle="votre-handle.bsky.social",
    bluesky_password="votre-app-password",
)

# Analyser des posts sur le climat en anglais
results = pipeline.run(
    query="climate change",
    lang="en",
    limit=100,
)

# Exporter
pipeline.export_results(results, "data/processed/climate_analysis.json")

# Afficher le rapport energetique
print(pipeline.energy_tracker.get_summary())
```

### Utiliser les modules individuellement

```python
# Collecte seule
from src.collector.bluesky_client import BlueskyCollector
collector = BlueskyCollector("handle", "password")
posts = collector.search_posts("vaccin", lang="fr", limit=20)

# Pretraitement seul
from src.preprocessing.text_processor import clean_text, tokenize
cleaned = clean_text("BREAKING @user https://lien.com La nouvelle est FAUSSE")
tokens = tokenize(cleaned, lang="fr")

# Classification seule
from src.models.fake_news_detector import FakeNewsDetector
detector = FakeNewsDetector()
result = detector.predict("Les extraterrestres ont atterri a Paris")
print(result)  # {'label': 'fake', 'confidence': 0.87, 'scores': {...}}
```

---

## 4. Utiliser le notebook

### Lancer Jupyter

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

### Contenu du notebook

Le notebook est structure en 8 sections executables dans l'ordre :

1. **Setup** : imports et configuration
2. **Collecte** : recuperation de posts sur 5 thematiques
3. **Exploration** : statistiques descriptives, longueur des textes, engagement
4. **Pretraitement** : nettoyage, tokenisation, wordcloud
5. **Classification** : test sur exemples + batch complet
6. **Emotions** : distribution, radar chart, heatmap
7. **Explicabilite** : mots influents avec barres d'importance
8. **Export** : sauvegarde CSV des resultats

Executez chaque cellule dans l'ordre avec **Shift+Enter**.

---

## 5. Comprendre les resultats

### Score de credibilite

| Label | Signification |
|---|---|
| **fiable** | Le contenu semble factuel et coherent |
| **douteux** | Le contenu presente des elements suspects mais n'est pas clairement faux |
| **fake** | Le contenu presente de forts indicateurs de desinformation |

Le **score de confiance** (0-100%) indique a quel point le modele est sur de sa classification. Un score de 95% pour "fake" est plus significatif qu'un score de 55%.

**Important** : Le modele n'est pas infaillible. Il fournit une **aide a la decision**, pas un verdict definitif. Tout contenu signale comme suspect doit etre verifie par un humain.

### Emotions

| Emotion | Indicateurs typiques |
|---|---|
| **Colere** | Indignation, attaques, majuscules, points d'exclamation |
| **Degout** | Rejet, mepris, degoutation |
| **Peur** | Menaces, alertes, scenarios catastrophe |
| **Joie** | Celebration, bonne nouvelle, humour |
| **Tristesse** | Deuil, deception, melancolie |
| **Surprise** | Etonnement, revelations, "breaking news" |
| **Neutre** | Ton informatif, factuel, sans charge emotionnelle |

### Explicabilite

Les **mots influents** sont les tokens qui ont le plus "attire l'attention" du modele lors de la classification. Un score d'importance de 1.0 signifie que ce mot est le plus influential dans la decision.

Exemple : pour un post classe "fake", les mots influents pourraient etre : `URGENT`, `cache`, `verite`, `!!!`

---

## 6. FAQ

### Le dashboard ne se lance pas

- Verifiez que toutes les dependances sont installees : `pip install -r requirements.txt`
- Verifiez que les modeles spaCy sont telecharges : `python -m spacy download fr_core_news_sm`
- Verifiez que le port 8501 n'est pas deja utilise

### Erreur de connexion a Bluesky

- Verifiez votre handle (format : `user.bsky.social`)
- Verifiez votre App Password (pas votre mot de passe principal)
- Creez un nouvel App Password si l'ancien ne fonctionne plus

### L'analyse est lente

- Le premier lancement telecharge les modeles Transformer (~500 Mo)
- Les modeles sont mis en cache apres le premier chargement
- Reduisez le nombre de posts pour des analyses plus rapides
- Utilisez un GPU si disponible (detection automatique)

### Les resultats de classification semblent aleatoires

C'est normal : le modele de base (RoBERTa) n'est **pas fine-tune** pour la detection de fake news. Les scores refletent une analyse generique du texte. Un fine-tuning sur un dataset specialise ameliorera significativement les resultats.

### Comment ajouter une nouvelle langue ?

1. Telecharger le modele spaCy correspondant : `python -m spacy download xx_core_news_sm`
2. Ajouter le mapping dans `text_processor.py` (fonction `get_nlp`)
3. Mettre a jour les filtres de langue dans le dashboard
