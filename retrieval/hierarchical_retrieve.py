from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_PARENT_TOP_K = 3
DEFAULT_CHILD_TOP_K_PER_PARENT = 2
DEFAULT_FINAL_TOP_K = 5


def load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError(
            "faiss is required for hierarchical_retrieve.py. Install it with: conda install -c conda-forge faiss-cpu"
        ) from e
    return faiss


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def maybe_prefix_query(query: str, model_name: str) -> str:
    if "bge" in model_name.lower():
        return "Represent this sentence for searching relevant passages: " + query.strip()
    return query.strip()


def embed_query(model: SentenceTransformer, query: str, model_name: str) -> np.ndarray:
    text = maybe_prefix_query(query, model_name)
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32, copy=False)

def get_or_load_model(
    model_name: str,
    model: Optional[SentenceTransformer] = None,
) -> SentenceTransformer:
    return model if model is not None else SentenceTransformer(model_name)


@lru_cache(maxsize=None)
def _load_parent_resources_cached(index_root_str: str, embed_root_str: str):
    faiss = load_faiss()
    index_root = Path(index_root_str)
    embed_root = Path(embed_root_str)

    index_path = index_root / "hierarchical_parent" / "index.faiss"
    metadata_path = embed_root / "hierarchical" / "parent_metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"Parent index file not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Parent metadata file not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    rows = load_jsonl(metadata_path)

    if index.ntotal != len(rows):
        raise ValueError(
            f"Parent index/metadata mismatch: index.ntotal={index.ntotal}, metadata rows={len(rows)}"
        )

    for i, row in enumerate(rows):
        if row.get("embedding_row") != i:
            raise ValueError(
                f"Parent embedding_row mismatch: expected {i}, got {row.get('embedding_row')}"
            )

    return index, rows


def load_parent_resources(index_root: Path, embed_root: Path):
    return _load_parent_resources_cached(str(index_root.resolve()), str(embed_root.resolve()))


@lru_cache(maxsize=None)
def _load_child_resources_cached(embed_root_str: str):
    embed_root = Path(embed_root_str)

    metadata_path = embed_root / "hierarchical" / "child_metadata.jsonl"
    embeddings_path = embed_root / "hierarchical" / "child_embeddings.npy"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Child metadata file not found: {metadata_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Child embeddings file not found: {embeddings_path}")

    rows = load_jsonl(metadata_path)
    embs = np.load(embeddings_path)

    if embs.ndim != 2:
        raise ValueError(f"Expected 2D child embeddings, got shape {embs.shape}")
    if len(rows) != embs.shape[0]:
        raise ValueError(
            f"Child metadata/embedding mismatch: {len(rows)} rows vs {embs.shape[0]} vectors"
        )

    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)

    for i, row in enumerate(rows):
        if row.get("embedding_row") != i:
            raise ValueError(
                f"Child embedding_row mismatch: expected {i}, got {row.get('embedding_row')}"
            )

    return rows, embs


def load_child_resources(embed_root: Path):
    return _load_child_resources_cached(str(embed_root.resolve()))


def search_parents(
    query_vec: np.ndarray,
    parent_index,
    parent_rows: List[Dict[str, Any]],
    parent_top_k: int,
) -> List[Dict[str, Any]]:
    k = min(parent_top_k, len(parent_rows))
    scores, indices = parent_index.search(query_vec, k)

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        row = parent_rows[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
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
    candidate_indices: List[int] = []
    for i, row in enumerate(child_rows):
        if row.get("parent_id") in selected_parent_ids:
            candidate_indices.append(i)
    return candidate_indices


def rerank_children(
    query_vec: np.ndarray,
    child_rows: List[Dict[str, Any]],
    child_embs: np.ndarray,
    candidate_indices: List[int],
    child_top_k_per_parent: int,
    final_top_k: int,
    selected_parent_ids: Set[str],
) -> List[Dict[str, Any]]:
    if not candidate_indices:
        return []

    parent_buckets: Dict[str, List[Tuple[int, float]]] = {}

    for idx in candidate_indices:
        row = child_rows[idx]
        parent_id = row.get("parent_id")
        if parent_id not in selected_parent_ids:
            continue
        score = float(child_embs[idx] @ query_vec[0])
        parent_buckets.setdefault(parent_id, []).append((idx, score))

    shortlisted: List[Tuple[int, float]] = []
    for parent_id, pairs in parent_buckets.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        shortlisted.extend(pairs[:child_top_k_per_parent])

    shortlisted.sort(key=lambda x: x[1], reverse=True)
    shortlisted = shortlisted[: min(final_top_k, len(shortlisted))]

    results: List[Dict[str, Any]] = []
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
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    model = get_or_load_model(model_name, model)
    query_vec = embed_query(model, query, model_name)

    parent_index, parent_rows = load_parent_resources(index_root, embed_root)
    child_rows, child_embs = load_child_resources(embed_root)

    parent_results = search_parents(
        query_vec=query_vec,
        parent_index=parent_index,
        parent_rows=parent_rows,
        parent_top_k=parent_top_k,
    )

    selected_parent_ids = {row["chunk_id"] for row in parent_results}
    candidate_indices = build_child_candidate_set(selected_parent_ids, child_rows)

    child_results = rerank_children(
        query_vec=query_vec,
        child_rows=child_rows,
        child_embs=child_embs,
        candidate_indices=candidate_indices,
        child_top_k_per_parent=child_top_k_per_parent,
        final_top_k=final_top_k,
        selected_parent_ids=selected_parent_ids,
    )

    return {
        "query": query,
        "model_name": model_name,
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
    print("\n" + "=" * 100)
    print(f"QUERY: {payload['query']}")
    print(f"MODEL: {payload['model_name']}")
    print(
        f"PARENT_TOP_K={payload['parent_top_k']} | "
        f"CHILD_TOP_K_PER_PARENT={payload['child_top_k_per_parent']} | "
        f"FINAL_TOP_K={payload['final_top_k']}"
    )
    print("=" * 100)

    print("\nPARENT RESULTS")
    for row in payload["parent_results"]:
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"text: {preview_text(row['text'])}")

    print("\nFINAL CHILD RESULTS")
    for row in payload["child_results"]:
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"parent_id: {row['parent_id']}")
        print(f"text: {preview_text(row['text'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical retrieval: parent search followed by child reranking."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument(
        "--child-top-k-per-parent",
        type=int,
        default=DEFAULT_CHILD_TOP_K_PER_PARENT,
    )
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = hierarchical_retrieve(
        query=args.query,
        index_root=Path(args.index_root).expanduser().resolve(),
        embed_root=Path(args.embed_root).expanduser().resolve(),
        model_name=args.model_name,
        parent_top_k=args.parent_top_k,
        child_top_k_per_parent=args.child_top_k_per_parent,
        final_top_k=args.final_top_k,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_payload(payload)


if __name__ == "__main__":
    main()