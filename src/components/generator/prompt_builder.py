class PromptBuilder:
    def __init__(self, mode="instruct", max_contexts=3, min_score=0.0, max_context_chars=4000):
        self.mode = mode
        self.max_contexts = max_contexts
        self.min_score = min_score
        self.max_context_chars = max_context_chars  # simple budget char

    def _select_docs(self, docs: list[dict]) -> list[dict]:
        # filtre + tri + tranche
        kept = [d for d in docs if float(d.get("score", 0.0)) >= float(self.min_score)]
        kept.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        kept = kept[: self.max_contexts]

        # budget simple en caractères pour éviter les prompts géants
        out, total = [], 0
        for d in kept:
            t = (d.get("text") or "").strip()
            if not t:
                continue
            # réserve ~ 100 chars d’en-tête par doc
            add = min(len(t), max(0, self.max_context_chars - total - 100))
            if add <= 0:
                break
            out.append({**d, "text": t[:add]})
            total += add + 100
        return out

    def _format_system_with_sources(self, docs):
        base_rules = (
            "You are a careful assistant. "
            "Use only the sources provided below to answer questions. "
            "If you don't find the answer, reply 'I don't know'. "
            "Cite sources as [n] when relevant."
        )

        if not docs:
            return base_rules

        sources = []
        for i, d in enumerate(docs, 1):
            txt = d.get("text", "").strip()
            if txt:
                sources.append(f"[{i}] {txt}")
        return base_rules + "\n\nSources:\n" + "\n\n".join(sources)

    # --- EXISTANT : on garde la signature et le comportement
    def build(self, query: str, docs: list[dict]) -> str:
        selected = self._select_docs(docs)
        if not selected:
            return (
                "You are a helpful and factual assistant. "
                "No context was retrieved; if you don't know the answer, reply 'I don't know'.\n\n"
                f"Question: {query}\n\nAnswer:"
            )

        context_text = "\n\n".join(d["text"] for d in selected)

        return (
            "You are a careful assistant specialized in question answering using retrieved documents.\n"
            "Follow these rules:\n"
            "1. Use only the information in the context below.\n"
            "2. If the context does not contain the answer, reply 'I don't know'.\n"
            "3. Do not invent facts.\n\n"
            "--- CONTEXT START ---\n"
            f"{context_text}\n"
            "--- CONTEXT END ---\n\n"
            f"Question: {query}\n\n"
            "Answer step-by-step using only the context above:"
        )

    # --- NOUVEAU : pour le stateful propre
    def build_ephemeral(self, query: str, docs: list[dict]) -> tuple[str, str]:
        selected = self._select_docs(docs)
        system_context = self._format_system_with_sources(selected)
        user_message = query.strip()
        return system_context, user_message
