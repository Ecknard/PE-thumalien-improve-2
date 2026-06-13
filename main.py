"""
main.py — Point d'entrée principal Thumalien.

Usage :
    python main.py                         # aide CLI
    python main.py analyze "fake news"     # analyse
    python main.py dashboard               # dashboard
    python main.py train                   # fine-tuning
    python main.py db init                 # init BDD
    python main.py db history              # historique

    Ou via module :
    python -m src analyze "fake news"
    python -m src.pipeline --limit 25
"""
from src.cli import app

if __name__ == "__main__":
    app()
