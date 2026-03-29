from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_TOP_K = 5

DEFAULT_QUERIES = [
    "How does IFS understand addictive behavior as a protective strategy?",
    "How should a coach respond when a client has a part using compulsive behavior to avoid pain?",
    "What does IFS say about protectors, shame, and getting curious rather than confronting the client?",
]


def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_partition(embed_root: Path, partition: str) -> Tuple[List[dict], np.ndarray]:
    if partition == "small_fixed":
        meta_path = embed_root / "small_fixed" / "metadata.jsonl"
        emb_path = embed_root / "small_fixed" / "embeddings.npy"
    elif partition == "medium_overlap":
        meta_path = embed_root / "medium_overlap" / "metadata.jsonl"
        emb_path = embed_root / "medium_overlap" / "embeddings.npy"
    elif partition == "hier_parent":
        meta_path = embed_root / "hierarchical" / "parent_metadata.jsonl"
        emb_path = embed_root / "hierarchical" / "parent_embeddings.npy"
    elif partition == "hier_child":
        meta_path = embed_root / "hierarchical" / "child_metadata.jsonl"
        emb_path = embed_root / "hierarchical" / "child_embeddings.npy"
    else:
        raise ValueError(f"Unknown partition: {partition}")

    rows = load_jsonl(meta_path)
    embs = np.load(emb_path)

    if len(rows) != embs.shape[0]:
        raise ValueError(
            f"Row/vector mismatch for {partition}: {len(rows)} rows vs {embs.shape[0]} vectors"
        )

    return rows, embs


def maybe_prefix_query(query: str, model_name: str) -> str:
    if "bge" in model_name.lower():
        return "Represent this sentence for searching relevant passages: " + query.strip()
    return query.strip()


def embed_query(model: SentenceTransformer, query: str, model_name: str) -> np.ndarray:
    q = maybe_prefix_query(query, model_name)
    vec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32, copy=False)


def top_k_hits(
    query_vec: np.ndarray,
    rows: List[dict],
    embs: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float, dict]]:
    scores = embs @ query_vec
    k = min(top_k, len(rows))
    idxs = np.argsort(scores)[-k:][::-1]
    return [(rank + 1, float(scores[idx]), rows[idx]) for rank, idx in enumerate(idxs)]


def preview_text(text: str, n: int = 320) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def print_hits(partition: str, hits: List[Tuple[int, float, dict]]) -> None:
    print(f"\n=== {partition} ===")
    for rank, score, row in hits:
        print(f"\nRank {rank} | score={score:.4f}")
        print(f"chunk_id: {row.get('chunk_id')}")
        print(f"title: {row.get('source_title')}")
        print(f"section: {row.get('section_path_str')}")
        print(f"parent_id: {row.get('parent_id')}")
        print(f"text: {preview_text(row.get('text', ''))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a semantic smoke test over saved corpus embeddings."
    )
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["small_fixed", "medium_overlap", "hier_parent", "hier_child"],
        choices=["small_fixed", "medium_overlap", "hier_parent", "hier_child"],
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Add one or more custom queries. If omitted, built-in test queries are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_root = Path(args.embed_root).expanduser().resolve()
    model = SentenceTransformer(args.model_name)
    queries = args.query if args.query else DEFAULT_QUERIES

    loaded = {partition: load_partition(embed_root, partition) for partition in args.partitions}

    for query in queries:
        print("\n" + "=" * 100)
        print("QUERY:", query)
        print("=" * 100)
        q_vec = embed_query(model, query, args.model_name)

        for partition in args.partitions:
            rows, embs = loaded[partition]
            hits = top_k_hits(q_vec, rows, embs, args.top_k)
            print_hits(partition, hits)


if __name__ == "__main__":
    main()