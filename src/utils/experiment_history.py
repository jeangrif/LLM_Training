# src/utils/experiment_history.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional
from datetime import datetime

def _flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    flat = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key, sep))
        else:
            # stringify lists/tuples for CSV
            if isinstance(v, (list, tuple)):
                flat[key] = ", ".join(map(str, v))
            else:
                flat[key] = v
    return flat

def _read_existing_rows(csv_path: Path) -> (List[str], List[Dict[str, str]]):
    if not csv_path.exists():
        return [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows

def _write_all(csv_path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            # ensure all keys
            writer.writerow({k: r.get(k, "") for k in header})

def _upgrade_header_if_needed(header: List[str], new_keys: Iterable[str],
                              core_first: Iterable[str]) -> List[str]:
    header_set = set(header)
    new_only = [k for k in new_keys if k not in header_set]
    if not header and not new_only:
        return header  # empty

    # union
    union = list(header) if header else []

    # if header missing (first write), build from core_first + rest sorted
    if not union:
        core = [k for k in core_first if k in new_keys]
        rest = sorted([k for k in new_keys if k not in core])
        return core + rest

    # otherwise just append new keys at the end (sorted for stability)
    if new_only:
        union += sorted(new_only)
    return union

def append_experiment_row(
    csv_path: Path,
    *parts: Dict[str, Any],
    core_first: Optional[List[str]] = None,
    extra_columns: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append (or upgrade+append) a row to a CSV with dynamic schema.

    Args:
        csv_path: path to CSV (e.g. outputs/history.csv)
        *parts: any number of dicts to merge (params, metrics, artifacts…)
        core_first: columns to place first if header is created
        extra_columns: extra flat dict merged last (wins on conflicts)
    """
    core_first = core_first or [
        "timestamp", "run_dir", "run_name", "stage_list",
        "retrieval_type", "rag.top_k", "embedding_model"
    ]

    # merge + flatten
    merged: Dict[str, Any] = {}
    for p in parts:
        if not p:
            continue
        merged.update(_flatten_dict(p))

    if extra_columns:
        merged.update(_flatten_dict(extra_columns))

    # common defaults
    merged.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    # load existing
    header, rows = _read_existing_rows(csv_path)

    # upgrade header if needed
    new_header = _upgrade_header_if_needed(header, merged.keys(), core_first)

    # if header grew, backfill old rows with empty values
    if new_header != header:
        # also normalize existing rows to new header
        rows = [{k: r.get(k, "") for k in new_header} for r in rows]
        header = new_header

    # append current row normalized to header
    row_norm = {k: merged.get(k, "") for k in header}
    rows.append(row_norm)

    # write all
    _write_all(csv_path, header, rows)
