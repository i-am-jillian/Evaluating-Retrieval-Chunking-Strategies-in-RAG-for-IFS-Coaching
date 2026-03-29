from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer

from retrieval.retrieve import search_partition
from retrieval.hierarchical_retrieve import hierarchical_retrieve


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_retrieval_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


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
    retrieval_model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    if condition == "small_fixed":
        return search_partition(
            query=query,
            partition="small_fixed",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            top_k=top_k,
            model=retrieval_model,
        )
    if condition == "medium_overlap":
        return search_partition(
            query=query,
            partition="medium_overlap",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            top_k=top_k,
            model=retrieval_model,
        )
    if condition == "hierarchical":
        return hierarchical_retrieve(
            query=query,
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            model=retrieval_model,
        )
    raise ValueError(f"Unsupported condition: {condition}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_output_path(output_dir: Path, slug: str, condition: str) -> Path:
    return output_dir / f"{slug}__{condition}.json"


def print_generation_result(payload: Dict[str, Any]) -> None:
    print("\n" + "#" * 100)
    print(f"CONDITION: {payload['condition']}")
    print(f"LLM_PROVIDER: {payload['llm_provider']}")
    print(f"LLM_MODEL: {payload['llm_model']}")
    print("#" * 100)

    print("\nGENERATED RESPONSE\n")
    print(payload["generated_response"])