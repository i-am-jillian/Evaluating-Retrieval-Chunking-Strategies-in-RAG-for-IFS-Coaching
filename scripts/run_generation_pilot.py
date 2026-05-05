from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.retrieve import search_partition
from retrieval.hierarchical_retrieve import hierarchical_retrieve
from generation.prompt_template import make_prompt_payload
from generation.generate import generate_text

from experiments.pilot_utils import (
    load_retrieval_model,
    make_output_path,
    now_utc_iso,
    print_generation_result,
    run_retrieval_for_condition,
    save_json,
)


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "tfidf"
DEFAULT_OUTPUT_DIR = "data/generation_pilot"
DEFAULT_TOP_K = 3
DEFAULT_PARENT_TOP_K = 3
DEFAULT_CHILD_TOP_K_PER_PARENT = 2
DEFAULT_FINAL_TOP_K = 5

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODEL = "gemini-3-flash-preview"


def preview_text(text: str, n: int = 350) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")

def run_one_condition(
    query: str,
    condition: str,
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    retrieval_model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    retrieval_payload = run_retrieval_for_condition(
        query=query,
        condition=condition,
        index_root=index_root,
        embed_root=embed_root,
        retrieval_model_name=retrieval_model_name,
        top_k=top_k,
        parent_top_k=parent_top_k,
        child_top_k_per_parent=child_top_k_per_parent,
        final_top_k=final_top_k,
        retrieval_model=retrieval_model,
    )

    prompt_payload = make_prompt_payload(
        query=query,
        condition=condition,
        retrieval_payload=retrieval_payload,
    )

    generation_text = generate_text(
        prompt_payload=prompt_payload,
        retrieval_payload=retrieval_payload,
        condition=condition,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    return {
        "created_at_utc": now_utc_iso(),
        "condition": condition,
        "query": query,
        "retrieval_model_name": retrieval_model_name,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retrieval": retrieval_payload,
        "prompt": prompt_payload,
        "generated_response": generation_text,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generation pilot using validated retrieval conditions."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--condition",
        type=str,
        choices=["small_fixed", "medium_overlap", "hierarchical", "all"],
        default="all",
    )
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--retrieval-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-provider", type=str, choices=["mock", "openai", "gemini"], default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument("--child-top-k-per-parent", type=int, default=DEFAULT_CHILD_TOP_K_PER_PARENT)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slug", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_root = Path(args.index_root).expanduser().resolve()
    embed_root = Path(args.embed_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    conditions = (
        ["small_fixed", "medium_overlap", "hierarchical"]
        if args.condition == "all"
        else [args.condition]
    )

    retrieval_model = load_retrieval_model(args.retrieval_model_name)

    for condition in conditions:
        payload = run_one_condition(
            query=args.query,
            condition=condition,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=args.retrieval_model_name,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            parent_top_k=args.parent_top_k,
            child_top_k_per_parent=args.child_top_k_per_parent,
            final_top_k=args.final_top_k,
            retrieval_model=retrieval_model,
        )

        print_generation_result(payload)
        out_path = make_output_path(output_dir, args.slug, condition)
        save_json(out_path, payload)
        print(f"\nSaved generation payload to: {out_path}")


if __name__ == "__main__":
    main()