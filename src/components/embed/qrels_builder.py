# src/components/embed/qrels_builder.py
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import json
import re
import pandas as pd
from tqdm import tqdm

class QrelsBuilder:
    """
    Build qrels.jsonl from a QA parquet and the chunking spec (size/overlap).
    Output: one JSONL per line: {"qid": "<id>", "doc_id": "<source_id>:<chunk_id>", "rel": 1}
    """

    def __init__(
        self,
        parquet_path: Path,
        text_field: str,
        chunk_size: int,
        chunk_overlap: int,
        mode: str = "offset",  # "offset" | "contains"
    ):
        self.parquet_path = Path(parquet_path)
        self.text_field = text_field
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.mode = mode
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")

    # ---------- internals ----------
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _windows(self, n: int) -> Iterable[Tuple[int, int, int]]:
        """Yield (start, end, cid) windows exactly like your TextChunker logic."""
        stride = self.chunk_size - self.chunk_overlap
        cid, start = 0, 0
        while start < n:
            end = min(start + self.chunk_size, n)
            yield start, end, cid
            cid += 1
            start += stride

    @staticmethod
    def _answer_spans(row: pd.Series) -> List[Tuple[int, int, str]]:
        """
        Return a list of (start, end, text).
        Supports SQuAD-like: row['answers'] = {'text': [...], 'answer_start': [...]}
        Falls back to row['answer'] (no offsets).
        """
        spans: List[Tuple[int, int, str]] = []
        answers = row.get("answers")
        if isinstance(answers, dict):
            texts = answers.get("text") or []
            starts = answers.get("answer_start") or []
            for t, st in zip(texts, starts):
                if t is None or st is None:
                    continue
                try:
                    st = int(st)
                except Exception:
                    continue
                spans.append((st, st + len(str(t)), str(t)))
            if spans:
                return spans
        ans = row.get("answer")
        if isinstance(ans, str) and ans.strip():
            spans.append((-1, -1, ans))
        return spans

    # ---------- public ----------
    def build(self, out_dir: Path, docs_path: Optional[Path] = None) -> Path:
        """
        Create qrels.jsonl in out_dir. If docs_path is provided, validate doc_ids exist.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qrels_path = out_dir / "qrels.jsonl"

        df = pd.read_parquet(self.parquet_path)
        if self.text_field not in df.columns:
            raise ValueError(f"Column '{self.text_field}' not found in dataset")

        # Question ids
        if "id" in df.columns:
            qids = df["id"].astype(str).tolist()
        elif "orig_id" in df.columns:
            qids = df["orig_id"].astype(str).tolist()
        else:
            qids = [str(i) for i in range(len(df))]

        # Mirror TextChunker dedup: map context → source_id
        dedup = df.drop_duplicates(subset=[self.text_field]).reset_index(drop=True)
        ctx_to_sid = {ctx: i for i, ctx in enumerate(dedup[self.text_field].tolist())}
        norm_ctx_to_sid = {self._norm(c): sid for c, sid in ctx_to_sid.items()}

        # Optional validation to ensure doc_id exists in docs.jsonl
        valid: Optional[Set[str]] = None
        if docs_path and Path(docs_path).exists():
            valid = set()
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    sid = rec.get("source_id")
                    cid = rec.get("chunk_id")
                    if sid is not None and cid is not None:
                        valid.add(f"{sid}:{cid}")

        wrote = 0
        with open(qrels_path, "w", encoding="utf-8") as fq:
            for i in tqdm(range(len(df)), desc="qrels", leave=False):
                row = df.iloc[i]
                qid = qids[i]
                ctx = row[self.text_field]

                # find source_id (sid) by exact or normalized match
                if ctx in ctx_to_sid:
                    sid = ctx_to_sid[ctx]
                    raw = ctx
                else:
                    sid = norm_ctx_to_sid.get(self._norm(ctx))
                    if sid is None:
                        continue
                    raw = dedup.iloc[sid][self.text_field]

                spans = self._answer_spans(row)
                if not spans:
                    continue

                n = len(raw)
                relevant: Set[str] = set()

                # try offset mode when offsets exist
                if self.mode == "offset" and any(st >= 0 for st, _, _ in spans):
                    for st, en, _ in spans:
                        if st < 0:
                            continue
                        for s, e, cid in self._windows(n):
                            if s <= st and en <= e:
                                did = f"{sid}:{cid}"
                                if valid is None or did in valid:
                                    relevant.add(did)

                # fallback (or explicit) contains mode
                if self.mode == "contains" or not relevant:
                    # precompute norm text per chunk
                    chunks_norm = [(cid, self._norm(raw[s:e])) for s, e, cid in self._windows(n)]
                    for _st, _en, txt in spans:
                        tnorm = self._norm(txt)
                        if not tnorm:
                            continue
                        for cid, cnorm in chunks_norm:
                            if tnorm in cnorm:
                                did = f"{sid}:{cid}"
                                if valid is None or did in valid:
                                    relevant.add(did)

                for did in sorted(relevant, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1]))):
                    fq.write(json.dumps({"qid": qid, "doc_id": did, "rel": 1}, ensure_ascii=False) + "\n")
                    wrote += 1

        if wrote == 0:
            print("⚠️ qrels.jsonl wrote 0 lines (check fields/mode).")
        else:
            print(f"✅ qrels.jsonl written → {qrels_path} ({wrote} lines)")
        return qrels_path
