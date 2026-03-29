from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K = 5
PARTITIONS = ["small_fixed", "medium_overlap", "hierarchical_parent", "hierarchical_child"]


def load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError(
            "faiss is required for retrieve.py. Install it with: conda install -c conda-forge faiss-cpu"
        ) from e
    return faiss


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_metadata_path(embed_root: Path, partition: str) -> Path:
    if partition == "small_fixed":
        return embed_root / "small_fixed" / "metadata.jsonl"
    if partition == "medium_overlap":
        return embed_root / "medium_overlap" / "metadata.jsonl"
    if partition == "hierarchical_parent":
        return embed_root / "hierarchical" / "parent_metadata.jsonl"
    if partition == "hierarchical_child":
        return embed_root / "hierarchical" / "child_metadata.jsonl"
    raise ValueError(f"Unknown partition: {partition}")


def resolve_index_path(index_root: Path, partition: str) -> Path:
    return index_root / partition / "index.faiss"


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
def _load_resources_cached(index_root_str: str, embed_root_str: str, partition: str):
    faiss = load_faiss()

    index_root = Path(index_root_str)
    embed_root = Path(embed_root_str)

    index_path = resolve_index_path(index_root, partition)
    metadata_path = resolve_metadata_path(embed_root, partition)

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found for {partition}: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {partition}: {metadata_path}")

    index = faiss.read_index(str(index_path))
    rows = load_jsonl(metadata_path)

    if index.ntotal != len(rows):
        raise ValueError(
            f"Index/metadata mismatch for {partition}: index.ntotal={index.ntotal}, metadata rows={len(rows)}"
        )

    for i, row in enumerate(rows):
        if row.get("embedding_row") != i:
            raise ValueError(
                f"embedding_row mismatch in {partition}: expected {i}, got {row.get('embedding_row')}"
            )

    return index, rows


def load_resources(index_root: Path, embed_root: Path, partition: str):
    return _load_resources_cached(str(index_root.resolve()), str(embed_root.resolve()), partition)


def search_partition(
    query: str,
    partition: str,
    index_root: Path,
    embed_root: Path,
    model_name: str,
    top_k: int,
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")

    model = get_or_load_model(model_name, model)
    index, rows = load_resources(index_root, embed_root, partition)
    q_vec = embed_query(model, query, model_name)

    k = min(top_k, len(rows))
    scores, indices = index.search(q_vec, k)

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        row = rows[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row.get("chunk_id"),
                "retrieval_setting": row.get("retrieval_setting"),
                "source_doc_id": row.get("source_doc_id"),
                "source_title": row.get("source_title"),
                "section_path_str": row.get("section_path_str"),
                "parent_id": row.get("parent_id"),
                "text": row.get("text"),
            }
        )

    return {
        "query": query,
        "partition": partition,
        "top_k": k,
        "model_name": model_name,
        "result_count": len(results),
        "results": results,
    }


def print_results(payload: Dict[str, Any], preview_chars: int = 350) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {payload['query']}")
    print(f"PARTITION: {payload['partition']}")
    print(f"MODEL: {payload['model_name']}")
    print("=" * 100)

    for row in payload["results"]:
        text = " ".join((row.get("text") or "").split())
        preview = text[:preview_chars] + ("..." if len(text) > preview_chars else "")
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"parent_id: {row['parent_id']}")
        print(f"text: {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a single FAISS partition and return top-k chunk hits."
    )
    parser.add_argument("--query", type=str, required=True, help="Query text to search")
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        choices=PARTITIONS,
        help="Which partition to search",
    )
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of formatted output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = search_partition(
        query=args.query,
        partition=args.partition,
        index_root=Path(args.index_root).expanduser().resolve(),
        embed_root=Path(args.embed_root).expanduser().resolve(),
        model_name=args.model_name,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_results(payload)


if __name__ == "__main__":
    main()