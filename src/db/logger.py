import json
import sqlite3
import time
from datetime import datetime
from src.db.connection import get_db_connection


class RagLogger:
    """Logger pour enregistrer les interactions du chat RAG (avec mémoire conversationnelle)."""

    def __init__(self):
        self._init_db()

    # -------------------------
    # 🧱 Création / mise à jour des tables
    # -------------------------
    def _init_db(self):
        """Initialise les tables si elles n'existent pas (et ajoute les nouvelles colonnes au besoin)."""
        retries = 3
        for attempt in range(retries):
            try:
                conn = get_db_connection()
                cur = conn.cursor()

                # Table des configurations de modèle
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        retrieval_type TEXT,
                        top_k INTEGER,
                        use_rerank BOOLEAN,
                        alpha REAL,
                        embedding_model TEXT,
                        model_meta TEXT
                    )
                """)

                # Table des interactions enrichie
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER,
                        timestamp TEXT,
                        query TEXT,
                        answer TEXT,
                        contexts TEXT,
                        stateful BOOLEAN DEFAULT 0,
                        conversation_context TEXT,
                        FOREIGN KEY(model_id) REFERENCES model_config(id)
                    )
                """)

                # Vérifie si les colonnes manquent (migration légère)
                existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(interactions);")]
                if "stateful" not in existing_cols:
                    cur.execute("ALTER TABLE interactions ADD COLUMN stateful BOOLEAN DEFAULT 0")
                if "conversation_context" not in existing_cols:
                    cur.execute("ALTER TABLE interactions ADD COLUMN conversation_context TEXT")

                # Table des latences
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS latencies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id INTEGER,
                        stage TEXT,
                        mean REAL,
                        std REAL,
                        count INTEGER,
                        FOREIGN KEY(interaction_id) REFERENCES interactions(id)
                    )
                """)

                conn.commit()
                conn.close()
                break  # ✅ succès
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < retries - 1:
                    time.sleep(0.5)
                else:
                    raise e

    # -------------------------
    # ⚙️ Log configuration modèle
    # -------------------------
    def log_model_config(self, config_dict: dict) -> int:
        """Insère une nouvelle configuration de modèle et renvoie son ID."""
        retries = 3
        for attempt in range(retries):
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO model_config (timestamp, retrieval_type, top_k, use_rerank, alpha, embedding_model, model_meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    config_dict.get("retrieval_type"),
                    config_dict.get("top_k"),
                    config_dict.get("use_rerank"),
                    config_dict.get("alpha"),
                    config_dict.get("embedding_model"),
                    json.dumps(config_dict.get("model_meta", {}), ensure_ascii=False),
                ))
                conn.commit()
                model_id = cur.lastrowid
                conn.close()
                return model_id
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < retries - 1:
                    time.sleep(0.5)
                else:
                    raise e

    # -------------------------
    # 💬 Log d'une interaction avec contexte complet
    # -------------------------
    def log_interaction(
        self,
        model_id,
        query,
        answer,
        contexts,
        latency=None,
        stateful=False,
        conversation_context=None,
    ):
        """Log une interaction, ses latences, et tout le contexte conversationnel (stateful)."""
        timestamp = datetime.now().isoformat()
        conn = get_db_connection()
        cur = conn.cursor()

        # 1️⃣ Insérer l’interaction
        cur.execute(
            """
            INSERT INTO interactions (model_id, timestamp, query, answer, contexts, stateful, conversation_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                timestamp,
                query,
                answer,
                json.dumps(contexts, ensure_ascii=False),
                int(stateful),
                json.dumps(conversation_context or [], ensure_ascii=False),
            ),
        )
        interaction_id = cur.lastrowid

        # 2️⃣ Insérer les latences si présentes
        if latency:
            rows = [
                (
                    interaction_id,
                    stage,
                    stats.get("mean", 0.0),
                    stats.get("std", 0.0),
                    stats.get("count", 1),
                )
                for stage, stats in latency.items()
            ]
            cur.executemany(
                """
                INSERT INTO latencies (interaction_id, stage, mean, std, count)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

        conn.commit()
        conn.close()

    # -------------------------
    # 📊 Lecture simple
    # -------------------------
    def get_interactions(self, limit: int = 10):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT i.timestamp, i.query, i.answer, i.contexts, i.stateful, i.conversation_context, m.embedding_model
            FROM interactions i
            JOIN model_config m ON i.model_id = m.id
            ORDER BY i.id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
