"""
src/labeler/labeler_service.py
═══════════════════════════════════════════════════════════════════
Thumalien Labeler — Émission de labels AT Protocol sur Bluesky.

PRINCIPE :
  Le Labeler Service écoute le Firehose Bluesky en temps réel,
  analyse chaque post via le pipeline Thumalien, puis émet un label
  via tools.ozone.moderation.emit_event sur les posts détectés.

  Les labels apparaissent sous les posts dans bsky.app pour tout
  utilisateur abonné au labeler (DID du compte Thumalien).

LABELS ÉMIS :
  - thumalien-fake     : post très probablement fake (confiance > THRESHOLD_FAKE)
  - thumalien-douteux  : contenu douteux           (confiance > THRESHOLD_DOUTEUX)
  - thumalien-fiable   : contenu jugé fiable        (optionnel, désactivé par défaut)

PRÉREQUIS BLUESKY :
  1. Déclarer son compte comme Labeler sur bsky.app/settings/moderation
  2. Avoir BLUESKY_HANDLE + BLUESKY_APP_PASSWORD dans .env
  3. Que des utilisateurs s'abonnent à ton labeler (URL : did:plc:XXXXX)

USAGE :
  python -m src.labeler.labeler_service               # démarrage normal
  python -m src.labeler.labeler_service --dry-run     # sans émettre les labels
  python -m src.labeler.labeler_service --lang fr     # uniquement posts FR
  python -m src.labeler.labeler_service --keywords "fake news,complot"
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message
from atproto import Client as BskyClient
from atproto_client import models
from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    BLUESKY_HANDLE, BLUESKY_APP_PASSWORD,
    COLLECTION_KEYWORDS, SUPPORTED_LANGUAGES,
    CREDIBILITY_THRESHOLD, LOG_FILE,
)

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# ─────────────────────────────────────────────────────────────
# SEUILS DE DÉCISION
# ─────────────────────────────────────────────────────────────
THRESHOLD_FAKE    = 0.70   # confiance minimale pour label "fake"
THRESHOLD_DOUTEUX = 0.50   # confiance minimale pour label "douteux"

# Labels déclarés (doivent correspondre à ce que tu configures sur bsky.app)
LABEL_FAKE    = "thumalien-fake"
LABEL_DOUTEUX = "thumalien-douteux"
LABEL_FIABLE  = "thumalien-fiable"   # optionnel

# Émettre le label "fiable" (verbose, désactivé par défaut)
EMIT_FIABLE = False


def _extract_text_from_commit(message) -> Optional[tuple[str, str, str]]:
    """
    Parse un message du Firehose et extrait (uri, cid, text) si c'est un post.
    Retourne None si ce n'est pas un post app.bsky.feed.post.
    """
    try:
        commit = parse_subscribe_repos_message(message)
        if not hasattr(commit, "ops"):
            return None

        for op in commit.ops:
            # On ne traite que les créations de posts
            if op.action != "create":
                continue
            if not op.path.startswith("app.bsky.feed.post/"):
                continue

            # Reconstruire l'URI AT Protocol
            uri = f"at://{commit.repo}/{op.path}"
            cid = str(op.cid) if op.cid else ""

            # Extraire le texte depuis le record
            record = op.record
            if record is None:
                continue
            text = getattr(record, "text", None) or ""

            if text.strip():
                return uri, cid, text

    except Exception:
        pass
    return None


class ThumalienLabeler:
    """
    Service de labeling en temps réel via le Firehose AT Protocol.

    Architecture :
      FirehoseSubscribeReposClient  →  _on_message()
        ↓ extrait (uri, cid, text)
        ↓ filtre langue + mots-clés
        ↓ pipeline Thumalien (prétraitement → classification)
        ↓ emit_label() via tools.ozone.moderation.emit_event
    """

    def __init__(
        self,
        handle: str = BLUESKY_HANDLE,
        password: str = BLUESKY_APP_PASSWORD,
        lang_filter: Optional[list[str]] = None,
        keyword_filter: Optional[list[str]] = None,
        dry_run: bool = False,
    ):
        self.handle = handle
        self.password = password
        self.lang_filter = lang_filter or SUPPORTED_LANGUAGES    # ["fr", "en"]
        self.keyword_filter = [kw.lower() for kw in (keyword_filter or [])]
        self.dry_run = dry_run

        # Clients Bluesky
        self._bsky = BskyClient()
        self._did: Optional[str] = None          # DID du compte labeler
        self._authenticated = False

        # Pipeline Thumalien (lazy load)
        self._classifier = None
        self._preprocessor = None

        # Stats
        self.stats = {
            "processed": 0,
            "labeled_fake": 0,
            "labeled_douteux": 0,
            "labeled_fiable": 0,
            "skipped_lang": 0,
            "skipped_keyword": 0,
            "errors": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─────────────────────────────────────────────────────────
    # AUTHENTIFICATION
    # ─────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Authentification Bluesky + récupération du DID."""
        try:
            profile = self._bsky.login(self.handle, self.password)
            self._did = profile.did
            self._authenticated = True
            logger.info(f"Labeler authentifié : {self.handle} ({self._did})")
            print(f"✅ Authentifié : {self.handle}")
            print(f"   DID : {self._did}")
            return True
        except Exception as e:
            logger.error(f"Authentification échouée : {e}")
            print(f"❌ Authentification échouée : {e}")
            return False

    # ─────────────────────────────────────────────────────────
    # CHARGEMENT PIPELINE
    # ─────────────────────────────────────────────────────────

    def _load_pipeline(self):
        """Charge le pipeline Thumalien (prétraitement + classifier)."""
        if self._classifier is not None:
            return

        print("⏳ Chargement du pipeline Thumalien...")
        try:
            from src.preprocessing.text_preprocessor import preprocess_posts
            from src.classifier.fake_news_classifier import FakeNewsClassifier

            self._preprocess_fn = preprocess_posts
            self._classifier = FakeNewsClassifier(prefer_bert=True)
            self._classifier.load()
            print("✅ Pipeline Thumalien chargé.")
            logger.info("Pipeline Thumalien chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur chargement pipeline : {e}")
            print(f"❌ Erreur pipeline : {e}")
            raise

    # ─────────────────────────────────────────────────────────
    # ANALYSE D'UN POST
    # ─────────────────────────────────────────────────────────

    def _analyze_post(self, text: str) -> dict:
        """
        Analyse un post et retourne :
          { label: str, confidence: float, is_fake: bool, credibility_score: float }
        """
        # Prétraitement
        raw = [{"text": text, "id": "labeler_tmp", "author": "", "created_at": ""}]
        processed = self._preprocess_fn(raw)
        if not processed:
            return {"label": "unknown", "confidence": 0.0, "is_fake": False, "credibility_score": 0.5}

        clean_text = processed[0].get("text_clean") or processed[0].get("text", text)

        # Classification
        preds = self._classifier.predict([clean_text])
        if not preds:
            return {"label": "unknown", "confidence": 0.0, "is_fake": False, "credibility_score": 0.5}

        pred = preds[0]
        score = pred.get("credibility_score", 0.5)
        confidence = pred.get("confidence", 0.0)
        is_fake = pred.get("is_fake", False)

        # Mapper vers nos 3 labels
        if is_fake and confidence >= THRESHOLD_FAKE:
            label = "fake"
        elif is_fake and confidence >= THRESHOLD_DOUTEUX:
            label = "douteux"
        elif not is_fake and confidence >= 0.65:
            label = "fiable"
        else:
            label = "douteux"  # incertain → douteux par précaution

        return {
            "label": label,
            "confidence": confidence,
            "is_fake": is_fake,
            "credibility_score": score,
        }

    # ─────────────────────────────────────────────────────────
    # ÉMISSION DU LABEL
    # ─────────────────────────────────────────────────────────

    def _emit_label(self, uri: str, cid: str, label_value: str) -> bool:
        """
        Émet un label AT Protocol via tools.ozone.moderation.emit_event.

        Le label est attaché à l'URI du post et visible dans bsky.app
        pour tout utilisateur abonné à ce labeler.
        """
        if self.dry_run:
            logger.debug(f"[DRY-RUN] Label '{label_value}' → {uri}")
            return True

        try:
            self._bsky.tools.ozone.moderation.emit_event(
                data=models.ToolsOzoneModerationEmitEvent.Data(
                    event=models.ToolsOzoneModerationDefs.ModEventLabel(
                        create_label_vals=[label_value],
                        negate_label_vals=[],
                        comment=f"Thumalien AI — Analyse fake news automatique",
                    ),
                    subject=models.ComAtprotoAdminDefs.RepoRef(
                        did=uri.split("/")[2] if uri.startswith("at://") else uri,
                    ),
                    subject_blob_cids=[],
                    created_by=self._did,
                ),
            )
            logger.info(f"Label '{label_value}' émis sur {uri}")
            return True

        except Exception as e:
            # Fallback : essayer avec StrongRef (post précis)
            try:
                self._bsky.tools.ozone.moderation.emit_event(
                    data=models.ToolsOzoneModerationEmitEvent.Data(
                        event=models.ToolsOzoneModerationDefs.ModEventLabel(
                            create_label_vals=[label_value],
                            negate_label_vals=[],
                            comment="Thumalien AI — Analyse fake news",
                        ),
                        subject=models.ComAtprotoRepoStrongRef.Main(
                            uri=uri,
                            cid=cid,
                        ),
                        subject_blob_cids=[],
                        created_by=self._did,
                    ),
                )
                logger.info(f"Label '{label_value}' émis (StrongRef) sur {uri}")
                return True
            except Exception as e2:
                logger.error(f"Impossible d'émettre le label sur {uri}: {e2}")
                self.stats["errors"] += 1
                return False

    # ─────────────────────────────────────────────────────────
    # FILTRE LANGUE & MOTS-CLÉS
    # ─────────────────────────────────────────────────────────

    def _should_process(self, text: str) -> tuple[bool, str]:
        """
        Décide si ce post doit être analysé.
        Retourne (True/False, raison_du_skip).
        """
        text_lower = text.lower()

        # Filtre mots-clés (si configuré, traiter uniquement les posts pertinents)
        if self.keyword_filter:
            if not any(kw in text_lower for kw in self.keyword_filter):
                return False, "keyword"

        # Filtre longueur minimale
        if len(text.strip()) < 15:
            return False, "too_short"

        return True, ""

    # ─────────────────────────────────────────────────────────
    # HANDLER FIREHOSE
    # ─────────────────────────────────────────────────────────

    def _on_message(self, message) -> None:
        """Callback appelé pour chaque message du Firehose."""
        result = _extract_text_from_commit(message)
        if result is None:
            return

        uri, cid, text = result
        self.stats["processed"] += 1

        # Filtre
        should, reason = self._should_process(text)
        if not should:
            if reason == "keyword":
                self.stats["skipped_keyword"] += 1
            return

        try:
            analysis = self._analyze_post(text)
        except Exception as e:
            logger.warning(f"Erreur analyse post {uri}: {e}")
            self.stats["errors"] += 1
            return

        label = analysis["label"]
        conf  = analysis["confidence"]

        # Décision d'émission
        if label == "fake" and conf >= THRESHOLD_FAKE:
            bsky_label = LABEL_FAKE
            self.stats["labeled_fake"] += 1
        elif label in ("fake", "douteux") and conf >= THRESHOLD_DOUTEUX:
            bsky_label = LABEL_DOUTEUX
            self.stats["labeled_douteux"] += 1
        elif label == "fiable" and EMIT_FIABLE:
            bsky_label = LABEL_FIABLE
            self.stats["labeled_fiable"] += 1
        else:
            return   # sous les seuils → pas de label

        success = self._emit_label(uri, cid, bsky_label)

        if success:
            mode = "[DRY-RUN] " if self.dry_run else ""
            icon = "🔴" if bsky_label == LABEL_FAKE else ("🟡" if bsky_label == LABEL_DOUTEUX else "🟢")
            logger.info(f"{mode}{icon} {bsky_label} (conf={conf:.2f}) → {uri}")
            print(f"  {mode}{icon} {bsky_label:<25} conf={conf:.2f}  {uri[:60]}...")

    # ─────────────────────────────────────────────────────────
    # BOUCLE PRINCIPALE
    # ─────────────────────────────────────────────────────────

    def run(self):
        """Démarre le service de labeling en temps réel."""
        print("\n" + "═" * 60)
        print("🏷️  THUMALIEN LABELER — Service de labeling AT Protocol")
        print("═" * 60)

        # Auth
        if not self.authenticate():
            sys.exit(1)

        # Pipeline
        self._load_pipeline()

        # Affichage config
        print(f"\n{'Mode':<20} : {'DRY-RUN (aucun label émis)' if self.dry_run else 'PRODUCTION (labels réels)'}")
        print(f"{'Filtres langue':<20} : {', '.join(self.lang_filter)}")
        print(f"{'Filtres mots-clés':<20} : {', '.join(self.keyword_filter) if self.keyword_filter else 'aucun (tous les posts)'}")
        print(f"{'Seuil FAKE':<20} : {THRESHOLD_FAKE:.0%}")
        print(f"{'Seuil DOUTEUX':<20} : {THRESHOLD_DOUTEUX:.0%}")
        print(f"\n{'─'*60}")
        print("📡 Connexion au Firehose Bluesky... (Ctrl+C pour arrêter)")
        print("─" * 60 + "\n")

        # Démarrage Firehose
        firehose = FirehoseSubscribeReposClient()

        try:
            firehose.start(self._on_message)
        except KeyboardInterrupt:
            self._print_stats()
        except Exception as e:
            logger.error(f"Erreur Firehose : {e}")
            print(f"\n❌ Erreur Firehose : {e}")
            # Reconnexion automatique
            print("⏳ Reconnexion dans 5s...")
            time.sleep(5)
            self.run()

    def _print_stats(self):
        """Affiche les statistiques de la session."""
        print("\n" + "═" * 60)
        print("📊 STATISTIQUES DE SESSION")
        print("═" * 60)
        print(f"  Posts traités   : {self.stats['processed']}")
        print(f"  🔴 Fake          : {self.stats['labeled_fake']}")
        print(f"  🟡 Douteux       : {self.stats['labeled_douteux']}")
        print(f"  🟢 Fiable        : {self.stats['labeled_fiable']}")
        print(f"  ⏭️  Skip mots-clés : {self.stats['skipped_keyword']}")
        print(f"  ❌ Erreurs       : {self.stats['errors']}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thumalien Labeler — Labels AT Protocol en temps réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m src.labeler.labeler_service                           # tous les posts
  python -m src.labeler.labeler_service --dry-run                 # test sans émettre
  python -m src.labeler.labeler_service --lang fr                 # posts FR seulement
  python -m src.labeler.labeler_service --keywords "fake,complot" # mots-clés ciblés
  python -m src.labeler.labeler_service --threshold-fake 0.80     # seuil strict
        """
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse sans émettre de labels (test)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Filtre langue(s), ex: 'fr' ou 'fr,en'")
    parser.add_argument("--keywords", type=str, default=None,
                        help="Mots-clés filtres séparés par virgule")
    parser.add_argument("--threshold-fake", type=float, default=THRESHOLD_FAKE,
                        help=f"Seuil confiance label fake (défaut: {THRESHOLD_FAKE})")
    parser.add_argument("--threshold-douteux", type=float, default=THRESHOLD_DOUTEUX,
                        help=f"Seuil confiance label douteux (défaut: {THRESHOLD_DOUTEUX})")
    parser.add_argument("--emit-fiable", action="store_true",
                        help="Émettre aussi le label 'fiable' (verbose)")

    args = parser.parse_args()

    # Override seuils
    THRESHOLD_FAKE    = args.threshold_fake
    THRESHOLD_DOUTEUX = args.threshold_douteux
    EMIT_FIABLE       = args.emit_fiable

    lang_filter   = args.lang.split(",") if args.lang else None
    kw_filter     = args.keywords.split(",") if args.keywords else None

    service = ThumalienLabeler(
        lang_filter=lang_filter,
        keyword_filter=kw_filter,
        dry_run=args.dry_run,
    )
    service.run()
