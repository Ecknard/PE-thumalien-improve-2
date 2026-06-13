# Thumalien Labeler — Guide complet

## Principe

Le **Labeler AT Protocol** permet à Thumalien d'afficher des étiquettes directement sous les posts dans bsky.app, **pour tous les utilisateurs abonnés à ton labeler**.

```
┌──────────────────────────────────────────────────────┐
│  @nytimes.com · 7 min                                │
│  "Breaking: Scientists discover..."                  │
│                                                      │
│  ┌─────────────────────────────────────┐             │
│  │ 🔴 Fake News  · Thumalien AI        │  ← LABEL   │
│  │ Post identifié comme probable fake  │             │
│  └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
```

## Architecture

```
Firehose bsky.network (tous les posts en temps réel)
         │
         ▼ WebSocket wss://bsky.network/xrpc/...
  ThumalienLabeler._on_message()
         │
         ├─ Filtre langue / mots-clés
         │
         ▼
  Pipeline Thumalien
  (prétraitement + FakeNewsClassifier)
         │
         ▼
  Décision :
  confiance > 70%  → label "thumalien-fake"
  confiance > 50%  → label "thumalien-douteux"
         │
         ▼
  tools.ozone.moderation.emit_event()
  (API AT Protocol → visible dans bsky.app)
```

## Mise en place (3 étapes)

### Étape 1 — Configurer le compte Labeler (one-shot)

```bash
# Configurer ton .env avec tes credentials Bluesky
cp .env.example .env
# Éditer .env : BLUESKY_HANDLE=... et BLUESKY_APP_PASSWORD=...

# Exécuter le script de setup (UNE SEULE FOIS)
make labeler-setup
# ou : python -m src.labeler.setup_labeler_account
```

Ce script :
- Publie un record `app.bsky.labeler.service` sur ton compte
- Configure les 3 labels avec leurs descriptions FR/EN
- Affiche ton **DID** (à noter et partager)

### Étape 2 — S'abonner à son propre labeler

Sur bsky.app :
1. **Settings** → **Moderation** → **Labelers**
2. Cliquer **Add labeler**
3. Coller ton DID (`did:plc:XXXXXXXXXX`)
4. Activer les labels souhaités

### Étape 3 — Lancer le service

```bash
# Test sans émettre de labels (recommandé pour commencer)
make labeler-dry

# Production : labels réels sur les posts
make labeler

# Uniquement les posts français
make labeler-fr

# Avec mots-clés ciblés
python -m src.labeler.labeler_service --keywords "fake news,complot,intox"

# Via Docker (recommandé en production)
make docker-labeler
```

## Labels émis

| Label | Seuil | Apparence dans bsky.app |
|-------|-------|------------------------|
| `thumalien-fake` | > 70% confiance | 🔴 Alerte rouge — "Fake News" |
| `thumalien-douteux` | > 50% confiance | 🟡 Avertissement — "Contenu douteux" |
| `thumalien-fiable` | > 65% (optionnel) | 🟢 Info — "Contenu vérifié" |

## Options CLI

```bash
python -m src.labeler.labeler_service \
  --dry-run              # test sans émettre \
  --lang fr              # filtre FR seulement \
  --keywords "fake,hoax" # filtre mots-clés \
  --threshold-fake 0.80  # seuil strict \
  --threshold-douteux 0.60 \
  --emit-fiable          # émettre aussi les labels "fiable"
```

## Commandes Makefile

```bash
make labeler-setup    # Configurer le compte (one-shot)
make labeler-dry      # Test sans labels réels
make labeler          # Production
make labeler-fr       # Posts français uniquement
make docker-labeler   # Via Docker (production)
```

## Partager son labeler

Une fois configuré, n'importe quel utilisateur bsky.app peut s'abonner à ton labeler :
1. Aller sur ton profil Bluesky
2. Il verra le badge "Labeler"
3. Il peut s'abonner → les labels Thumalien apparaîtront sur ses posts

## Limites & considérations

- **Rate limiting** : Bluesky limite les appels API. Le service gère automatiquement les erreurs mais un seuil trop bas (ex: 0.3) émettrait trop de labels.
- **Responsabilité** : Les labels émis sont publics. Utilise des seuils conservateurs (≥ 0.70 pour fake).
- **Firehose** : Tous les posts Bluesky passent par le firehose (~1000/s). Le filtre mots-clés est recommandé pour limiter la charge CPU.
- **Modèle** : Le modèle baseline (TF-IDF) est moins précis que DistilBERT. Entraîne ton modèle avant de passer en production : `make train-bert` ou `make train-liar`.
