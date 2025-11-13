# src/ui/app.py
import sys
from pathlib import Path
import pandas as pd
# --- Chemin racine du projet ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import traceback
import streamlit as st
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.rag.engine import RagPipeline
from src.db.connection import get_db_connection
from src.db.logger import RagLogger




def main():
    st.set_page_config(page_title="Chat RAG (Hydra)", layout="wide")
    st.title("🧠 Chat RAG")

    # -----------------------------
    # ⚙️ Chargement automatique de la config Hydra
    # -----------------------------
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="pipeline")

    rag_cfg = cfg.modules.run_rag
    try:
        logger = RagLogger()
        st.sidebar.success("📗 Logger initialisé avec succès !")
    except Exception as e:
        logger = None
        st.sidebar.error(f"❌ Erreur lors de l'initialisation du logger : {e}")
        st.sidebar.code(traceback.format_exc())



    # -----------------------------
    # ⚙️ Sidebar interactive
    # -----------------------------
    st.sidebar.header("⚙️ Options")

    retrieval_type = st.sidebar.selectbox(
        "Retrieval type", ["dense", "sparse", "hybrid"],
        index=["dense", "sparse", "hybrid"].index(rag_cfg.retrieval_type)
    )
    top_k = st.sidebar.slider("Top K", 1,32, rag_cfg.top_k)
    use_rerank = st.sidebar.toggle("Activer le rerank", rag_cfg.use_rerank)
    top_k_rerank = st.sidebar.slider("Top K Rerank", 1, 10, rag_cfg.top_k_rerank)
    alpha = st.sidebar.slider("Alpha (poids hybrid)", 0.0, 1.0, rag_cfg.alpha)
    stateful_mode = st.sidebar.toggle("🧠 Mode conversation (stateful)", value=True)

    embedding_model = st.sidebar.text_input("Embedding model", rag_cfg.embedding_model)
    model_meta = {
        "llm_repo": cfg.llm.llm_repo,
        "llm_path": str(Path(cfg.llm.local_dir) / cfg.llm.llm_filename),
        "chat_format": cfg.llm.get("chat_format", "mistral-instruct"),
    }
    if st.sidebar.button("🔄 Reset conversation"):
        if "rag" in st.session_state:
            try:
                rag = st.session_state.rag
                if hasattr(rag, "reset_context"):
                    rag.reset_context()  # vide le contexte interne du modèle
            except Exception as e:
                st.sidebar.warning(f"Erreur lors du reset du modèle : {e}")
        st.session_state.clear()
        st.toast("Contexte réinitialisé 🧹")
    st.sidebar.divider()


    if logger is not None:
        if "model_id" not in st.session_state:
            try:
                st.session_state.model_id = logger.log_model_config({
                    "retrieval_type": retrieval_type,
                    "top_k": top_k,
                    "use_rerank": use_rerank,
                    "alpha": alpha,
                    "embedding_model": embedding_model,
                    "model_meta": {
                        "llm_repo": cfg.llm.llm_repo,
                        "llm_path": str(Path(cfg.llm.local_dir) / cfg.llm.llm_filename),
                        "chat_format": cfg.llm.get("chat_format", "mistral-instruct"),
                    },
                })
                st.sidebar.success(f"🧾 Config loggée (ID {st.session_state.model_id})")
            except Exception as e:
                st.sidebar.error(f"❌ Erreursss lors du log de la config : {e}")

    # -----------------------------
    # ⚙️ Initialisation du pipeline RAG (une seule fois)
    # -----------------------------
    if "rag" not in st.session_state:
        # 🔹 Reconstruire dynamiquement le chemin de l'index à partir de la config Hydra
        index_base = Path(cfg.embed.index_dir)
        embed_model = cfg.embed.embedding_model.replace("/", "-")
        chunk = cfg.embed.chunk_size
        overlap = cfg.embed.chunk_overlap
        index_dir = index_base / f"{embed_model}__chunk{chunk}_ov{overlap}"

        # Vérification rapide
        if not index_dir.exists():
            st.warning(f"⚠️ Index introuvable à {index_dir}. Vérifie que l’étape check_data a bien été exécutée.")
        else:
            st.sidebar.success(f"📦 Index trouvé : {index_dir.name}")

        # 🔹 Initialisation du pipeline RAG
        st.session_state.rag = RagPipeline(
            top_k=top_k,
            retrieval_type=retrieval_type,
            use_rerank=use_rerank,
            alpha=alpha,
            embedding_model=embedding_model,
            model_cfg=rag_cfg.model_cfg,
            latency_cfg=rag_cfg.latency_cfg,
            index_dir=index_dir,
            model_meta=model_meta,
            stateful=stateful_mode,
            top_k_rerank=top_k_rerank
        )

    rag = st.session_state.rag

    # -----------------------------
    # 💬 Chat
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Posez votre question…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🤖 Génération en cours..."):
            try:
                result = rag.run(query)
                answer = result["pred"]
                contexts = result["contexts"]
                latency_summary = None
                if hasattr(rag, "latency_meter") and rag.latency_meter:
                    latency_summary = rag.latency_meter.summary()
                else:
                    latency_summary = None
            except Exception as e:
                answer = f"⚠️ Erreur : {e}"
                contexts = []
                latency_summary = None

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        if logger is not None and "model_id" in st.session_state:
            try:
                # 🔹 Historique interne du modèle (coté LlamaCppProvider)
                model_messages = getattr(rag.generator.model, "messages", [])

                logger.log_interaction(
                    model_id=st.session_state.model_id,
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    latency=latency_summary,
                    stateful=st.session_state.get("stateful_mode", False),
                    conversation_context=model_messages,
                )
                st.toast("💾 Interaction loggée avec contexte !", icon="💬")
            except Exception as e:
                st.sidebar.error(f"⚠️ Erreur lors du log de l'interaction : {e}")

        if contexts:
            st.divider()
            st.subheader("📚 Contextes récupérés")
            for i, ctx in enumerate(contexts, 1):
                with st.expander(f"Context {i}"):
                    st.write(ctx)


if __name__ == "__main__":
    main()
