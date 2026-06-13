"""
src/labeler/setup_labeler_account.py
═══════════════════════════════════════════════════════════════════
Script ONE-SHOT : déclare ton compte Bluesky comme Labeler Service.

À exécuter UNE SEULE FOIS avant de lancer labeler_service.py.

Ce script :
  1. S'authentifie avec tes credentials
  2. Affiche ton DID (à noter)
  3. Publie un record app.bsky.labeler.service sur ton compte
  4. Configure les policies des labels Thumalien

Ensuite dans bsky.app :
  → Settings → Moderation → Labelers → Add Labeler → coller le DID
"""

import sys
from pathlib import Path

from atproto import Client
from atproto_client import models

sys.path.append(str(Path(__file__).parents[2]))
from config import BLUESKY_HANDLE, BLUESKY_APP_PASSWORD


LABELER_POLICIES = {
    "label_values_details": [
        {
            "identifier": "thumalien-fake",
            "severity":   "alert",          # alerte rouge visible
            "blurs":      "none",            # le post reste lisible
            "default_setting": "warn",       # avertissement par défaut
            "locales": [
                {"lang": "fr", "name": "🔴 Fake News", "description": "Post identifié comme probable fake news par Thumalien AI."},
                {"lang": "en", "name": "🔴 Fake News", "description": "Post flagged as likely fake news by Thumalien AI."},
            ],
        },
        {
            "identifier": "thumalien-douteux",
            "severity":   "inform",         # informatif, moins intrusif
            "blurs":      "none",
            "default_setting": "warn",
            "locales": [
                {"lang": "fr", "name": "🟡 Contenu douteux", "description": "Ce post contient des éléments potentiellement trompeurs selon Thumalien AI."},
                {"lang": "en", "name": "🟡 Questionable content", "description": "This post contains potentially misleading elements according to Thumalien AI."},
            ],
        },
        {
            "identifier": "thumalien-fiable",
            "severity":   "inform",
            "blurs":      "none",
            "default_setting": "ignore",    # ignoré par défaut, pas d'affichage
            "locales": [
                {"lang": "fr", "name": "🟢 Contenu vérifié", "description": "Ce post semble fiable selon Thumalien AI."},
                {"lang": "en", "name": "🟢 Verified content", "description": "This post appears reliable according to Thumalien AI."},
            ],
        },
    ],
}


def setup_labeler():
    print("═" * 60)
    print("⚙️  THUMALIEN — Configuration du compte Labeler")
    print("═" * 60)

    client = Client()

    # Authentification
    print(f"\n🔑 Authentification en tant que {BLUESKY_HANDLE}...")
    try:
        profile = client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)
        did = profile.did
        print(f"✅ Connecté ! DID : {did}")
    except Exception as e:
        print(f"❌ Échec authentification : {e}")
        sys.exit(1)

    # Publication du record labeler service
    print("\n📝 Publication du record app.bsky.labeler.service...")
    try:
        client.com.atproto.repo.put_record(
            data=models.ComAtprotoRepoPutRecord.Data(
                repo=did,
                collection="app.bsky.labeler.service",
                rkey="self",
                record={
                    "$type": "app.bsky.labeler.service",
                    "policies": LABELER_POLICIES,
                    "createdAt": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                },
            )
        )
        print("✅ Record labeler publié avec succès !")
    except Exception as e:
        print(f"❌ Erreur publication record : {e}")
        print("   (Si l'erreur persiste, vérifiez que votre compte est bien de type 'Labeler' sur bsky.app)")
        sys.exit(1)

    # Résumé
    print("\n" + "═" * 60)
    print("✅ COMPTE LABELER CONFIGURÉ")
    print("═" * 60)
    print(f"\n  DID du labeler : {did}")
    print(f"\n  Labels configurés :")
    for lv in LABELER_POLICIES["label_values_details"]:
        name = lv["locales"][0]["name"]
        print(f"    • {lv['identifier']:<25} → {name}")

    print("\n📋 ÉTAPES SUIVANTES :")
    print("  1. Sur bsky.app → Settings → Moderation → Labelers")
    print("     → Cliquer 'Add labeler' → Coller votre DID :")
    print(f"     {did}")
    print("  2. Vous pouvez aussi partager ce DID avec vos utilisateurs")
    print("     pour qu'ils s'abonnent à votre labeler")
    print("  3. Lancer le service :")
    print("     python -m src.labeler.labeler_service --dry-run  # test")
    print("     python -m src.labeler.labeler_service            # production")
    print("     make labeler                                      # via Makefile")
    print("\n" + "═" * 60)


if __name__ == "__main__":
    setup_labeler()
