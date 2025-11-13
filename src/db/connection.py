import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

def get_db_connection():
    """Retourne une connexion SQLite configurée pour un usage concurrent (Streamlit)."""
    db_type = os.getenv("DB_TYPE", "sqlite")
    if db_type != "sqlite":
        raise NotImplementedError("Pour l’instant seul SQLite est supporté.")

    # Déterminer le chemin de la base
    db_path = Path(os.getenv("DB_PATH", "logs/rag_chat.db"))
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # 🔧 Paramètres critiques pour Streamlit
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,  # permet l’accès multi-thread
        timeout=5,                # attend jusqu’à 5s avant de lever “database is locked”
        isolation_level=None      # autocommit : pas de transactions persistantes
    )

    return conn
