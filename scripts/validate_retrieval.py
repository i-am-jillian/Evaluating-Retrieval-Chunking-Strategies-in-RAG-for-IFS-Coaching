from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from preprocessing.retrieve import search_partition
from preprocessing.hierarchical_retrieve import hierarchical_retrieve


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K = 5
DEFAULT_PARENT_TOP_K = 3
DEFAULT_CHILD_TOP_K_PER_PARENT = 2
DEFAULT_FINAL_TOP_K = 5
DEFAULT_OUTPUT_DIR = "data/retrieval_validation"


def preview_text(text: str, n: int = 320) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


def print_standard_partition(title: str, payload: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for row in payload.get("results", []):
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"parent_id: {row.get('parent_id')}")
        print(f"text: {preview_text(row.get('text', ''))}")


def print_hierarchical(payload: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("HIERARCHICAL RETRIEVAL")
    print("=" * 100)

    print("\nPARENT RESULTS")
    for row in payload.get("parent_results", []):
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"text: {preview_text(row.get('text', ''))}")

    print("\nFINAL CHILD RESULTS")
    for row in payload.get("child_results", []):
        print(f"\nRank {row['rank']} | score={row['score']:.4f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"title: {row['source_title']}")
        print(f"section: {row['section_path_str']}")
        print(f"parent_id: {row.get('parent_id')}")
        print(f"text: {preview_text(row.get('text', ''))}")


def run_validation(
    query: str,
    index_root: Path,
    embed_root: Path,
    model_name: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
) -> Dict[str, Any]:
    small_payload = search_partition(
        query=query,
        partition="small_fixed",
        index_root=index_root,
        embed_root=embed_root,
        model_name=model_name,
        top_k=top_k,
    )

    medium_payload = search_partition(
        query=query,
        partition="medium_overlap",
        index_root=index_root,
        embed_root=embed_root,
        model_name=model_name,
        top_k=top_k,
    )

    hierarchical_payload = hierarchical_retrieve(
        query=query,
        index_root=index_root,
        embed_root=embed_root,
        model_name=model_name,
        parent_top_k=parent_top_k,
        child_top_k_per_parent=child_top_k_per_parent,
        final_top_k=final_top_k,
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "model_name": model_name,
        "index_root": str(index_root),
        "embed_root": str(embed_root),
        "small_fixed": small_payload,
        "medium_overlap": medium_payload,
        "hierarchical": hierarchical_payload,
    }


def save_payload(payload: Dict[str, Any], output_dir: Path, slug: Optional[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if slug:
        filename = f"{slug}.json"
    else:
        safe_stub = "".join(c.lower() if c.isalnum() else "_" for c in payload["query"])[:80]
        filename = f"{safe_stub.strip('_') or 'retrieval_validation'}.json"

    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval validation across small, medium, and hierarchical conditions."
    )
    parser.add_argument("--query", type=str, required=True, help="Consultation-window query to test")
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument(
        "--child-top-k-per-parent",
        type=int,
        default=DEFAULT_CHILD_TOP_K_PER_PARENT,
    )
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save the full validation payload to disk",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved validation JSON",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Optional output filename stub when using --save-json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = run_validation(
        query=args.query,
        index_root=Path(args.index_root).expanduser().resolve(),
        embed_root=Path(args.embed_root).expanduser().resolve(),
        model_name=args.model_name,
        top_k=args.top_k,
        parent_top_k=args.parent_top_k,
        child_top_k_per_parent=args.child_top_k_per_parent,
        final_top_k=args.final_top_k,
    )

    print("\n" + "#" * 100)
    print(f"QUERY: {payload['query']}")
    print("#" * 100)

    print_standard_partition("SMALL_FIXED RETRIEVAL", payload["small_fixed"])
    print_standard_partition("MEDIUM_OVERLAP RETRIEVAL", payload["medium_overlap"])
    print_hierarchical(payload["hierarchical"])

    if args.save_json:
        out_path = save_payload(
            payload=payload,
            output_dir=Path(args.output_dir).expanduser().resolve(),
            slug=args.slug,
        )
        print(f"\nSaved validation JSON to: {out_path}")


if __name__ == "__main__":
    main()