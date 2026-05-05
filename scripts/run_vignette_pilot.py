from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_loader import get_vignette, get_window, load_vignettes
from experiments.vignette_runner import (
    run_all_vignettes,
    run_all_windows_in_vignette,
    run_window,
)
from scripts.run_generation_pilot import (
    DEFAULT_CHILD_TOP_K_PER_PARENT,
    DEFAULT_EMBED_ROOT,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_INDEX_ROOT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARENT_TOP_K,
    DEFAULT_TOP_K,
)

DEFAULT_DATASET_PATH = "data/synthetic_vignettes.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vignette-based retrieval/generation pilot over synthetic consultation windows."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to synthetic_vignettes.json",
    )
    parser.add_argument(
        "--vignette-id",
        type=str,
        default=None,
        help="Run one vignette by ID, e.g. V001",
    )
    parser.add_argument(
        "--window-id",
        type=str,
        default=None,
        help="Run one consultation window by ID, e.g. V001_W1",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all vignettes and all consultation windows",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["small_fixed", "medium_overlap", "hierarchical", "all"],
        default="all",
    )
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--retrieval-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
    "--llm-provider",
    type=str,
    choices=["mock", "openai", "gemini"],
    default="gemini",
)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument(
        "--child-top-k-per-parent",
        type=int,
        default=DEFAULT_CHILD_TOP_K_PER_PARENT,
    )
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    index_root = Path(args.index_root).expanduser().resolve()
    embed_root = Path(args.embed_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    vignettes = load_vignettes(dataset_path)

    if args.run_all:
        run_all_vignettes(
            vignettes=vignettes,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=args.retrieval_model_name,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            condition=args.condition,
            top_k=args.top_k,
            parent_top_k=args.parent_top_k,
            child_top_k_per_parent=args.child_top_k_per_parent,
            final_top_k=args.final_top_k,
            output_dir=output_dir,
        )
        return

    if args.vignette_id is None:
        raise ValueError("Provide --vignette-id unless using --run-all")

    vignette = get_vignette(vignettes, args.vignette_id)

    if args.window_id is not None:
        window = get_window(vignette, args.window_id)
        run_window(
            vignette=vignette,
            window=window,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=args.retrieval_model_name,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            condition=args.condition,
            top_k=args.top_k,
            parent_top_k=args.parent_top_k,
            child_top_k_per_parent=args.child_top_k_per_parent,
            final_top_k=args.final_top_k,
            output_dir=output_dir,
        )
        return

    run_all_windows_in_vignette(
        vignette=vignette,
        index_root=index_root,
        embed_root=embed_root,
        retrieval_model_name=args.retrieval_model_name,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        condition=args.condition,
        top_k=args.top_k,
        parent_top_k=args.parent_top_k,
        child_top_k_per_parent=args.child_top_k_per_parent,
        final_top_k=args.final_top_k,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()