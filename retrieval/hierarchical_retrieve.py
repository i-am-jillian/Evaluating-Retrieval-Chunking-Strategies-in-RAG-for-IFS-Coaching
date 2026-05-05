from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "tfidf"
DEFAULT_PARENT_TOP_K = 3
DEFAULT_CHILD_TOP_K_PER_PARENT = 2
DEFAULT_FINAL_TOP_K = 5


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=None)
def _load_parent_resources_cached(embed_root_str: str):
    embed_root = Path(embed_root_str)
    metadata_path = embed_root / "hierarchical" / "parent_metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Parent metadata file not found: {metadata_path}")

    rows = load_jsonl(metadata_path)
    texts = [row.get("text", "") for row in rows]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return rows, vectorizer, matrix


def load_parent_resources(embed_root: Path):
    return _load_parent_resources_cached(str(embed_root.resolve()))


@lru_cache(maxsize=None)
def _load_child_resources_cached(embed_root_str: str):
    embed_root = Path(embed_root_str)
    metadata_path = embed_root / "hierarchical" / "child_metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Child metadata file not found: {metadata_path}")

    rows = load_jsonl(metadata_path)
    texts = [row.get("text", "") for row in rows]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return rows, vectorizer, matrix


def load_child_resources(embed_root: Path):
    return _load_child_resources_cached(str(embed_root.resolve()))


def search_parents(
    query: str,
    parent_rows: List[Dict[str, Any]],
    parent_vectorizer,
    parent_matrix,
    parent_top_k: int,
) -> List[Dict[str, Any]]:
    query_vec = parent_vectorizer.transform([query])
    scores = (parent_matrix @ query_vec.T).toarray().reshape(-1)

    top_idx = np.argsort(-scores)[: min(parent_top_k, len(parent_rows))]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = parent_rows[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[int(idx)]),
                "chunk_id": row.get("chunk_id"),
                "source_doc_id": row.get("source_doc_id"),
                "source_title": row.get("source_title"),
                "section_path_str": row.get("section_path_str"),
                "text": row.get("text"),
            }
        )
    return results


def build_child_candidate_set(
    selected_parent_ids: Set[str],
    child_rows: List[Dict[str, Any]],
) -> List[int]:
    candidate_indices = []
    for i, row in enumerate(child_rows):
        if row.get("parent_id") in selected_parent_ids:
            candidate_indices.append(i)
    return candidate_indices


def rerank_children(
    query: str,
    child_rows: List[Dict[str, Any]],
    child_vectorizer,
    child_matrix,
    candidate_indices: List[int],
    child_top_k_per_parent: int,
    final_top_k: int,
    selected_parent_ids: Set[str],
) -> List[Dict[str, Any]]:
    if not candidate_indices:
        return []

    query_vec = child_vectorizer.transform([query])
    full_scores = (child_matrix @ query_vec.T).toarray().reshape(-1)

    parent_buckets: Dict[str, List[tuple[int, float]]] = {}

    for idx in candidate_indices:
        row = child_rows[idx]
        parent_id = row.get("parent_id")
        if parent_id not in selected_parent_ids:
            continue

        score = float(full_scores[idx])
        parent_buckets.setdefault(parent_id, []).append((idx, score))

    shortlisted = []

    for parent_id, pairs in parent_buckets.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        shortlisted.extend(pairs[:child_top_k_per_parent])

    shortlisted.sort(key=lambda x: x[1], reverse=True)
    shortlisted = shortlisted[: min(final_top_k, len(shortlisted))]

    results = []
    for rank, (idx, score) in enumerate(shortlisted, start=1):
        row = child_rows[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row.get("chunk_id"),
                "source_doc_id": row.get("source_doc_id"),
                "source_title": row.get("source_title"),
                "section_path_str": row.get("section_path_str"),
                "parent_id": row.get("parent_id"),
                "text": row.get("text"),
            }
        )

    return results


def hierarchical_retrieve(
    query: str,
    index_root: Path,
    embed_root: Path,
    model_name: str,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    model=None,
) -> Dict[str, Any]:
    parent_rows, parent_vectorizer, parent_matrix = load_parent_resources(embed_root)
    child_rows, child_vectorizer, child_matrix = load_child_resources(embed_root)

    parent_results = search_parents(
        query=query,
        parent_rows=parent_rows,
        parent_vectorizer=parent_vectorizer,
        parent_matrix=parent_matrix,
        parent_top_k=parent_top_k,
    )

    selected_parent_ids = {row["chunk_id"] for row in parent_results}

    candidate_indices = build_child_candidate_set(
        selected_parent_ids=selected_parent_ids,
        child_rows=child_rows,
    )

    child_results = rerank_children(
        query=query,
        child_rows=child_rows,
        child_vectorizer=child_vectorizer,
        child_matrix=child_matrix,
        candidate_indices=candidate_indices,
        child_top_k_per_parent=child_top_k_per_parent,
        final_top_k=final_top_k,
        selected_parent_ids=selected_parent_ids,
    )

    return {
        "query": query,
        "model_name": "tfidf",
        "parent_top_k": parent_top_k,
        "child_top_k_per_parent": child_top_k_per_parent,
        "final_top_k": final_top_k,
        "parent_result_count": len(parent_results),
        "child_result_count": len(child_results),
        "parent_results": parent_results,
        "child_results": child_results,
    }


def preview_text(text: str, n: int = 350) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


def print_payload(payload: Dict[str, Any]) -> None:
    print("\nPARENT RESULTS")
    for row in payload["parent_results"]:
        print(row["rank"], row["score"], row["source_title"])

    print("\nCHILD RESULTS")
    for row in payload["child_results"]:
        print(row["rank"], row["score"], row["source_title"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-root", default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument("--child-top-k-per-parent", type=int, default=DEFAULT_CHILD_TOP_K_PER_PARENT)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    payload = hierarchical_retrieve(
        query=args.query,
        index_root=Path(args.index_root),
        embed_root=Path(args.embed_root),
        model_name=args.model_name,
        parent_top_k=args.parent_top_k,
        child_top_k_per_parent=args.child_top_k_per_parent,
        final_top_k=args.final_top_k,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_payload(payload)


if __name__ == "__main__":
    main()