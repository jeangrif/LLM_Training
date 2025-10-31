def display_rag_pipeline_config(
    retrieval_type: str,
    use_rerank: bool,
    top_k: int,
    alpha: float,
    index_dir: str = None,
    embedding_model: str = None,
    do_generation=True,
    top_k_rerank = None,
):
    """Affiche un schéma ASCII de la pipeline RAG à exécuter."""

    print("\n🧩 [RAG PIPELINE OVERVIEW]")
    print("──────────────────────────────────────────")

    # --- paramètres généraux
    print(f"📘 Retrieval type     : {retrieval_type}")
    print(f"📈 Top-K              : {top_k}")
    if retrieval_type == "hybrid":
        print(f"⚖️  Alpha (mix factor) : {alpha}")
    print(f"🔁 Re-ranking active  : {'✅ Yes' if use_rerank else '❌ No'} (top_k_rerank = {top_k_rerank})")
    print(f"💾 Index directory    : {index_dir if index_dir else 'default / auto-detect'}")
    print(f"🧠 Embedding model    : {embedding_model if embedding_model else 'default model'}")
    print(f"🧪 Generation active  : {'✅ Yes' if do_generation else '❌ No'}")

    # --- schéma visuel du pipeline
    print("\n📊 Pipeline flow:")
    print("──────────────────────────────────────────")
    print("          ┌────────────────────────┐")
    print("          │     Input question     │")
    print("          └────────────┬───────────┘")
    print("                       │")
    print(f"               🔍 {retrieval_type.capitalize()} Retriever")
    print(f"                (Top-{top_k} docs)")
    if use_rerank:
        print("                       │")
        print("               🔁 Re-ranker ")

    if do_generation:
        print("                       │")
        print("               🧠 Generator (LLM)")
        print("                       │")
        print("          ┌────────────┴───────────┐")
        print("          │     Final answer       │")
        print("          └────────────────────────┘")
    else:
        print("                       │")
        print("          🚫 Generation skipped (retrieve-only mode)")
    print("──────────────────────────────────────────\n")

