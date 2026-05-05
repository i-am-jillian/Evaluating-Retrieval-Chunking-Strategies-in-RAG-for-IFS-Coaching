from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _extract_rows_from_metadata_obj(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict):
            return obj

    if isinstance(obj, dict):
        for key in ["rows", "chunks", "data", "items", "records", "metadata"]:
            value = obj.get(key)
            if isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict)):
                return value

    raise ValueError("Could not extract row dictionaries from metadata.")


def _load_metadata_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected dict rows in JSONL metadata, got: {type(obj).__name__}")
                rows.append(obj)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return _extract_rows_from_metadata_obj(obj)


def _resolve_metadata_path(
    manifest_path: Path,
    metadata_path_str: str,
    embed_root: Path,
    partition: str,
) -> Path:
    raw_path = Path(metadata_path_str)

    # 1. If the manifest path is already valid on this machine, use it.
    if raw_path.exists():
        return raw_path

    # 2. If it's a relative path, resolve from the manifest folder.
    relative_candidate = (manifest_path.parent / raw_path).resolve()
    if relative_candidate.exists():
        return relative_candidate

    # 3. If it's an absolute path from someone else's machine, rebuild it locally
    #    using only the filename under your current embed_root/partition.
    local_candidate = (embed_root / partition / raw_path.name).resolve()
    if local_candidate.exists():
        return local_candidate

    # 4. Last resort: search under embed_root/partition for the same filename.
    matches = list((embed_root / partition).rglob(raw_path.name))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Metadata file not found. Tried:\n"
        f"  raw path: {raw_path}\n"
        f"  relative to manifest: {relative_candidate}\n"
        f"  local embed path: {local_candidate}"
    )


@lru_cache(maxsize=16)
def _load_resources_cached(
    index_root_str: str,
    embed_root_str: str,
    partition: str,
) -> Tuple[List[Dict[str, Any]], TfidfVectorizer, Any]:
    index_root = Path(index_root_str)
    embed_root = Path(embed_root_str)

    partition_index_dir = index_root / partition
    if not partition_index_dir.exists():
        raise FileNotFoundError(f"Index partition folder not found: {partition_index_dir}")

    manifest_path = partition_index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    metadata_path_str = manifest.get("metadata_path")
    if not metadata_path_str:
        raise ValueError(f"'metadata_path' missing in manifest: {manifest_path}")

    metadata_path = _resolve_metadata_path(
        manifest_path=manifest_path,
        metadata_path_str=metadata_path_str,
        embed_root=embed_root,
        partition=partition,
    )

    rows = _load_metadata_file(metadata_path)

    texts = [row.get("text", "") for row in rows]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return rows, vectorizer, matrix


def load_resources(index_root: Path, embed_root: Path, partition: str):
    return _load_resources_cached(
        str(index_root.resolve()),
        str(embed_root.resolve()),
        partition,
    )


def search_partition(
    query: str,
    partition: str,
    index_root: Path,
    embed_root: Path,
    model_name: str | None = None,
    top_k: int = 5,
    model=None,
    retrieval_model=None,
) -> Dict[str, Any]:
    rows, vectorizer, matrix = load_resources(index_root, embed_root, partition)

    query_vec = vectorizer.transform([query])
    scores = (matrix @ query_vec.T).toarray().reshape(-1)

    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = dict(rows[int(idx)])
        row["rank"] = rank
        row["score"] = float(scores[int(idx)])
        results.append(row)

    return {
        "partition": partition,
        "model_name": model_name or "tfidf",
        "top_k": top_k,
        "results": results,
    }