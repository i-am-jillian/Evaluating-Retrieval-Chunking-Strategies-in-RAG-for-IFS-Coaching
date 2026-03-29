from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_METRIC = "inner_product"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise ImportError(
            "faiss is required for build_index.py. Install it with: pip install faiss-cpu"
        ) from e
    return faiss


def resolve_partition_paths(embed_root: Path, partition: str) -> Tuple[Path, Path]:
    if partition == "small_fixed":
        return (
            embed_root / "small_fixed" / "metadata.jsonl",
            embed_root / "small_fixed" / "embeddings.npy",
        )
    if partition == "medium_overlap":
        return (
            embed_root / "medium_overlap" / "metadata.jsonl",
            embed_root / "medium_overlap" / "embeddings.npy",
        )
    if partition == "hierarchical_parent":
        return (
            embed_root / "hierarchical" / "parent_metadata.jsonl",
            embed_root / "hierarchical" / "parent_embeddings.npy",
        )
    if partition == "hierarchical_child":
        return (
            embed_root / "hierarchical" / "child_metadata.jsonl",
            embed_root / "hierarchical" / "child_embeddings.npy",
        )
    raise ValueError(f"Unknown partition: {partition}")


def load_partition(embed_root: Path, partition: str) -> Tuple[List[Dict[str, Any]], np.ndarray, Path, Path]:
    metadata_path, embeddings_path = resolve_partition_paths(embed_root, partition)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {partition}: {metadata_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding file not found for {partition}: {embeddings_path}")

    rows = load_jsonl(metadata_path)
    vectors = np.load(embeddings_path)

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for {partition}, got shape {vectors.shape}")
    if len(rows) != vectors.shape[0]:
        raise ValueError(
            f"Metadata/embedding mismatch for {partition}: {len(rows)} rows vs {vectors.shape[0]} vectors"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    return rows, vectors, metadata_path, embeddings_path


def check_embedding_rows(rows: List[Dict[str, Any]], partition: str) -> None:
    for i, row in enumerate(rows):
        if row.get("embedding_row") != i:
            raise ValueError(
                f"Non-contiguous or mismatched embedding_row in {partition}: expected {i}, got {row.get('embedding_row')}"
            )
        if not row.get("chunk_id"):
            raise ValueError(f"Missing chunk_id in {partition} at row {i}")


def build_faiss_index(vectors: np.ndarray, metric: str):
    faiss = load_faiss()
    dim = vectors.shape[1]

    if metric == "inner_product":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index.add(vectors)
    return index


def save_partition_index(
    output_root: Path,
    partition: str,
    rows: List[Dict[str, Any]],
    vectors: np.ndarray,
    metadata_path: Path,
    embeddings_path: Path,
    metric: str,
    embedding_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    faiss = load_faiss()
    partition_dir = output_root / partition
    partition_dir.mkdir(parents=True, exist_ok=True)

    index = build_faiss_index(vectors, metric)
    index_path = partition_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    manifest = {
        "partition": partition,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "vector_count": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "faiss_index_type": type(index).__name__,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embeddings_path": str(embeddings_path),
        "embedding_model_name": embedding_manifest.get("embedding_model_name"),
        "normalize_embeddings": embedding_manifest.get("normalize_embeddings"),
        "sample_chunk_id": rows[0]["chunk_id"] if rows else None,
    }

    manifest_path = partition_dir / "index_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def process_all(embed_root: Path, output_root: Path, metric: str) -> Dict[str, Any]:
    embedding_manifest_path = embed_root / "manifests" / "embedding_manifest.json"
    if not embedding_manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {embedding_manifest_path}")

    embedding_manifest = read_json(embedding_manifest_path)
    partitions = [
        "small_fixed",
        "medium_overlap",
        "hierarchical_parent",
        "hierarchical_child",
    ]

    build_manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embed_root": str(embed_root),
        "index_root": str(output_root),
        "metric": metric,
        "source_embedding_manifest": str(embedding_manifest_path),
        "embedding_model_name": embedding_manifest.get("embedding_model_name"),
        "normalize_embeddings": embedding_manifest.get("normalize_embeddings"),
        "partitions": {},
    }

    for partition in partitions:
        rows, vectors, metadata_path, embeddings_path = load_partition(embed_root, partition)
        check_embedding_rows(rows, partition)
        partition_manifest = save_partition_index(
            output_root=output_root,
            partition=partition,
            rows=rows,
            vectors=vectors,
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            metric=metric,
            embedding_manifest=embedding_manifest,
        )
        build_manifest["partitions"][partition] = partition_manifest
        print(
            f"Built {partition}: {partition_manifest['vector_count']} vectors, "
            f"dim={partition_manifest['embedding_dim']}"
        )

    global_manifest_path = output_root / "manifests" / "faiss_build_manifest.json"
    write_json(global_manifest_path, build_manifest)
    print(f"Wrote FAISS build manifest: {global_manifest_path}")

    return build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes from saved embedding partitions."
    )
    parser.add_argument(
        "--embed-root",
        type=str,
        default=DEFAULT_EMBED_ROOT,
        help="Root directory containing embedding outputs from embed_corpus.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_INDEX_ROOT,
        help="Directory to write FAISS indexes and manifests",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["inner_product", "l2"],
        default=DEFAULT_METRIC,
        help="Similarity metric for FAISS index construction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_root = Path(args.embed_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    process_all(embed_root=embed_root, output_root=output_root, metric=args.metric)


if __name__ == "__main__":
    main()