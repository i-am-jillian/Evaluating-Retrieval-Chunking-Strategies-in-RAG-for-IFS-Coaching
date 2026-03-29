from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ============================================================
# embed_corpus.py
# ============================================================
# Purpose:
#   Read chunk outputs produced by chunk_text.py and generate
#   corpus embeddings for the three retrieval conditions:
#       1) small_fixed
#       2) medium_overlap
#       3) hierarchical (parents + children)
#
# Assumptions:
#   - chunk_text.py has already been run successfully.
#   - chunk_manifest.json describes where the chunked JSONL files live.
#   - Each JSONL row contains a `text` field and stable chunk metadata.
#
# Outputs:
#   data/ifs_corpus/embeddings/
#       manifests/embedding_manifest.json
#       small_fixed/
#           metadata.jsonl
#           embeddings.npy
#       medium_overlap/
#           metadata.jsonl
#           embeddings.npy
#       hierarchical/
#           parent_metadata.jsonl
#           parent_embeddings.npy
#           child_metadata.jsonl
#           child_embeddings.npy
#
# The saved metadata files preserve the chunk record and add embedding row
# indices so retrieval/index-building code can map vector rows back to chunk
# records without re-parsing or reordering the source data.
# ============================================================

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_OUTPUT_DIRNAME = "embeddings"
TEXT_FIELD = "text"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e



def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count



def resolve_path(path_str: str, manifest_path: Path) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()



def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embed_corpus.py. "
            "Install it with: pip install sentence-transformers"
        ) from e

    return SentenceTransformer(model_name)



def build_instruction_prefix(model_name: str) -> str:
    lower_name = model_name.lower()
    if "bge" in lower_name:
        return "Represent this passage for retrieval: "
    return ""



def prepare_texts(records: Sequence[Dict[str, Any]], model_name: str) -> List[str]:
    prefix = build_instruction_prefix(model_name)
    texts: List[str] = []
    for i, record in enumerate(records):
        text = record.get(TEXT_FIELD, "")
        if not isinstance(text, str) or not text.strip():
            chunk_id = record.get("chunk_id", f"row_{i}")
            raise ValueError(f"Record {chunk_id} is missing non-empty text")
        text = text.strip()
        texts.append(prefix + text if prefix else text)
    return texts



def encode_records(
    model,
    records: Sequence[Dict[str, Any]],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    texts = prepare_texts(records, model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)



def add_embedding_metadata(
    records: Sequence[Dict[str, Any]],
    embedding_partition: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row_idx, record in enumerate(records):
        enriched = dict(record)
        enriched["embedding_row"] = row_idx
        enriched["embedding_partition"] = embedding_partition
        enriched["source_jsonl"] = enriched.get("source_jsonl")
        out.append(enriched)
    return out



def validate_unique_chunk_ids(records: Sequence[Dict[str, Any]], label: str) -> None:
    ids = [r.get("chunk_id") for r in records]
    if any(not x for x in ids):
        raise ValueError(f"Missing chunk_id detected in {label}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate chunk_id detected in {label}")



def combine_jsonl_inputs(
    paths: Sequence[Path],
    embedding_partition: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    per_doc_counts: List[Dict[str, Any]] = []

    for path in paths:
        records = list(iter_jsonl(path))
        for record in records:
            record["source_jsonl"] = str(path)
        validate_unique_chunk_ids(records, f"{embedding_partition}:{path.name}")
        source_doc_id = records[0].get("source_doc_id") if records else path.stem
        per_doc_counts.append(
            {
                "source_doc_id": source_doc_id,
                "source_jsonl": str(path),
                "count": len(records),
            }
        )
        combined.extend(records)

    validate_unique_chunk_ids(combined, f"combined {embedding_partition}")
    summary = {
        "partition": embedding_partition,
        "source_file_count": len(paths),
        "record_count": len(combined),
        "per_doc_counts": per_doc_counts,
    }
    return combined, summary



def gather_partition_inputs(chunk_manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, List[Path]]:
    partitions = {
        "small_fixed": [],
        "medium_overlap": [],
        "hierarchical_parent": [],
        "hierarchical_child": [],
    }

    for doc in chunk_manifest.get("documents", []):
        outputs = doc.get("outputs", {})
        partitions["small_fixed"].append(resolve_path(outputs["small_fixed"], manifest_path))
        partitions["medium_overlap"].append(resolve_path(outputs["medium_overlap"], manifest_path))
        partitions["hierarchical_parent"].append(resolve_path(outputs["hierarchical_parents"], manifest_path))
        partitions["hierarchical_child"].append(resolve_path(outputs["hierarchical_children"], manifest_path))

    for key, paths in partitions.items():
        if not paths:
            raise ValueError(f"No input files found for partition: {key}")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            missing_str = "\n  - ".join(missing)
            raise FileNotFoundError(f"Missing chunk files for {key}:\n  - {missing_str}")

    return partitions



def save_partition(
    output_dir: Path,
    metadata_filename: str,
    embeddings_filename: str,
    records: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    source_paths: Sequence[Path],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / metadata_filename
    embeddings_path = output_dir / embeddings_filename

    enriched_records = add_embedding_metadata(
        records=records,
        embedding_partition=summary["partition"],
    )

    write_jsonl(metadata_path, enriched_records)
    np.save(embeddings_path, embeddings)

    return {
        "partition": summary["partition"],
        "record_count": len(records),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0,
        "metadata_path": str(metadata_path),
        "embeddings_path": str(embeddings_path),
        "source_file_count": len(source_paths),
        "source_files": [str(p) for p in source_paths],
        "per_doc_counts": summary["per_doc_counts"],
    }



def infer_output_root(chunk_manifest: Dict[str, Any], manifest_path: Path, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.expanduser().resolve()

    chunk_output_root_str = chunk_manifest.get("output_root")
    if not chunk_output_root_str:
        raise ValueError("chunk_manifest.json is missing output_root")

    chunk_output_root = resolve_path(chunk_output_root_str, manifest_path)
    return (chunk_output_root.parent / DEFAULT_OUTPUT_DIRNAME).resolve()



def process_all(
    chunk_manifest_path: Path,
    output_root: Path | None,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> Dict[str, Any]:
    chunk_manifest = read_json(chunk_manifest_path)
    resolved_output_root = infer_output_root(chunk_manifest, chunk_manifest_path, output_root)
    partitions = gather_partition_inputs(chunk_manifest, chunk_manifest_path)

    model = load_sentence_transformer(model_name)

    manifest: Dict[str, Any] = {
        "corpus_name": chunk_manifest.get("corpus_name", "ifs_corpus"),
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_chunk_manifest": str(chunk_manifest_path),
        "embedding_output_root": str(resolved_output_root),
        "embedding_model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
        "partitions": {},
    }

    # Standard flat settings
    for partition_name in ["small_fixed", "medium_overlap"]:
        source_paths = partitions[partition_name]
        records, summary = combine_jsonl_inputs(source_paths, partition_name)
        embeddings = encode_records(
            model=model,
            records=records,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        partition_output_dir = resolved_output_root / partition_name
        saved = save_partition(
            output_dir=partition_output_dir,
            metadata_filename="metadata.jsonl",
            embeddings_filename="embeddings.npy",
            records=records,
            embeddings=embeddings,
            source_paths=source_paths,
            summary=summary,
        )
        manifest["partitions"][partition_name] = saved
        print(f"Embedded {partition_name}: {saved['record_count']} records")

    # Hierarchical parents and children are saved together under hierarchical/
    hier_dir = resolved_output_root / "hierarchical"

    parent_records, parent_summary = combine_jsonl_inputs(partitions["hierarchical_parent"], "hierarchical_parent")
    parent_embeddings = encode_records(
        model=model,
        records=parent_records,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    parent_saved = save_partition(
        output_dir=hier_dir,
        metadata_filename="parent_metadata.jsonl",
        embeddings_filename="parent_embeddings.npy",
        records=parent_records,
        embeddings=parent_embeddings,
        source_paths=partitions["hierarchical_parent"],
        summary=parent_summary,
    )
    manifest["partitions"]["hierarchical_parent"] = parent_saved
    print(f"Embedded hierarchical_parent: {parent_saved['record_count']} records")

    child_records, child_summary = combine_jsonl_inputs(partitions["hierarchical_child"], "hierarchical_child")
    child_embeddings = encode_records(
        model=model,
        records=child_records,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    child_saved = save_partition(
        output_dir=hier_dir,
        metadata_filename="child_metadata.jsonl",
        embeddings_filename="child_embeddings.npy",
        records=child_records,
        embeddings=child_embeddings,
        source_paths=partitions["hierarchical_child"],
        summary=child_summary,
    )
    manifest["partitions"]["hierarchical_child"] = child_saved
    print(f"Embedded hierarchical_child: {child_saved['record_count']} records")

    manifest["totals"] = {
        "small_fixed_total": manifest["partitions"]["small_fixed"]["record_count"],
        "medium_overlap_total": manifest["partitions"]["medium_overlap"]["record_count"],
        "hierarchical_parent_total": manifest["partitions"]["hierarchical_parent"]["record_count"],
        "hierarchical_child_total": manifest["partitions"]["hierarchical_child"]["record_count"],
    }

    embedding_manifest_path = resolved_output_root / "manifests" / "embedding_manifest.json"
    write_json(embedding_manifest_path, manifest)
    print(f"Wrote embedding manifest: {embedding_manifest_path}")

    return manifest



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for chunked IFS corpus outputs."
    )
    parser.add_argument(
        "--chunk-manifest",
        type=str,
        default="data/ifs_corpus/chunked/manifests/chunk_manifest.json",
        help="Path to chunk_manifest.json produced by chunk_text.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Directory to write embeddings. If omitted, defaults to a sibling "
            "directory next to the chunked output root called 'embeddings'."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of output embeddings",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    chunk_manifest_path = Path(args.chunk_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None

    process_all(
        chunk_manifest_path=chunk_manifest_path,
        output_root=output_root,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
