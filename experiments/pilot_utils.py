from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from retrieval.retrieve import search_partition
from retrieval.hierarchical_retrieve import hierarchical_retrieve


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_output_path(output_dir: Path, slug: str, condition: str) -> Path:
    filename = f"{slug}__{condition}.json"
    return output_dir / filename


def load_retrieval_model(model_name: str):
    return None


def preview_text(text: str, n: int = 350) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


def _call_with_supported_kwargs(fn, **kwargs):
    """
    Call a function using only the kwargs it actually accepts.
    This avoids breaking when local function signatures changed
    during your FAISS -> NumPy retrieval edits.
    """
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def run_retrieval_for_condition(
    query: str,
    condition: str,
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    retrieval_model: Optional[object] = None,
) -> Dict[str, Any]:
    if retrieval_model is None:
        retrieval_model = load_retrieval_model(retrieval_model_name)

    if condition == "small_fixed":
        return _call_with_supported_kwargs(
            search_partition,
            query=query,
            partition="small_fixed",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            model=retrieval_model,
            retrieval_model=retrieval_model,
            top_k=top_k,
        )

    if condition == "medium_overlap":
        return _call_with_supported_kwargs(
            search_partition,
            query=query,
            partition="medium_overlap",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            model=retrieval_model,
            retrieval_model=retrieval_model,
            top_k=top_k,
        )

    if condition == "hierarchical":
        return _call_with_supported_kwargs(
            hierarchical_retrieve,
            query=query,
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            model=retrieval_model,
            retrieval_model=retrieval_model,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            top_k=top_k,
        )

    raise ValueError(f"Unsupported condition: {condition}")


def print_generation_result(payload: Dict[str, Any]) -> None:
    condition = payload.get("condition", "N/A")
    query = payload.get("query", "")
    retrieval = payload.get("retrieval", {})
    generated = payload.get("generated_response", "")

    print("\n" + "=" * 100)
    print(f"CONDITION: {condition}")
    print("=" * 100)

    print("\nQUERY\n")
    print(query)

    print("\nTOP RETRIEVAL RESULTS\n")

    if condition == "hierarchical":
        parent_results = retrieval.get("parent_results", []) or []
        child_results = retrieval.get("child_results", []) or []

        if parent_results:
            print("Parent results:")
            for row in parent_results[:3]:
                print(
                    f"- rank={row.get('rank')} score={row.get('score'):.4f} "
                    f"title={row.get('source_title', 'N/A')} "
                    f"section={row.get('section_path_str', 'N/A')}"
                )

        if child_results:
            print("\nChild results:")
            for row in child_results[:5]:
                print(
                    f"- rank={row.get('rank')} score={row.get('score'):.4f} "
                    f"title={row.get('source_title', 'N/A')} "
                    f"section={row.get('section_path_str', 'N/A')}"
                )
                print(f"  preview: {preview_text(row.get('text', ''))}")
    else:
        results = retrieval.get("results", []) or []
        for row in results[:5]:
            print(
                f"- rank={row.get('rank')} score={row.get('score'):.4f} "
                f"title={row.get('source_title', 'N/A')} "
                f"section={row.get('section_path_str', 'N/A')}"
            )
            print(f"  preview: {preview_text(row.get('text', ''))}")

    print("\nGENERATED RESPONSE\n")
    print(generated)
    print()