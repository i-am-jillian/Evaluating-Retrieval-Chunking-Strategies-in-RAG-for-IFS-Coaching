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
    generate_text,
    load_retrieval_model,
    make_output_path,
    make_prompt_payload,
    print_generation_result,
    run_retrieval_for_condition,
    save_json,
)


DEFAULT_DATASET_PATH = "data/synthetic_vignettes.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vignettes(dataset_path: Path) -> List[Dict[str, Any]]:
    payload = load_json(dataset_path)
    vignettes = payload.get("vignettes")
    if not isinstance(vignettes, list):
        raise ValueError(f"Expected top-level 'vignettes' list in {dataset_path}")
    return vignettes


def get_vignette(vignettes: List[Dict[str, Any]], vignette_id: str) -> Dict[str, Any]:
    for vignette in vignettes:
        if vignette.get("vignette_id") == vignette_id:
            return vignette
    raise ValueError(f"Could not find vignette_id={vignette_id}")


def get_window(vignette: Dict[str, Any], window_id: str) -> Dict[str, Any]:
    windows = vignette.get("consultation_windows", [])
    for window in windows:
        if window.get("window_id") == window_id:
            return window
    raise ValueError(
        f"Could not find window_id={window_id} in vignette_id={vignette.get('vignette_id')}"
    )


def normalize_turn_range(turn_range: Any) -> Tuple[int, int]:
    if (
        isinstance(turn_range, list)
        and len(turn_range) == 2
        and all(isinstance(x, int) for x in turn_range)
    ):
        start_turn, end_turn = turn_range
        if start_turn > end_turn:
            raise ValueError(f"Invalid turn_range: start > end: {turn_range}")
        return start_turn, end_turn

    raise ValueError(
        f"Unsupported turn_range format: {turn_range}. Expected [start_turn, end_turn]."
    )


def build_consultation_window_text(
    vignette: Dict[str, Any],
    window: Dict[str, Any],
) -> str:
    transcript = vignette.get("transcript", [])
    if not isinstance(transcript, list) or not transcript:
        raise ValueError(
            f"Vignette {vignette.get('vignette_id')} has missing/invalid transcript."
        )

    start_turn, end_turn = normalize_turn_range(window.get("turn_range"))

    selected_turns = [
        turn for turn in transcript
        if isinstance(turn, dict)
        and isinstance(turn.get("turn_id"), int)
        and start_turn <= turn["turn_id"] <= end_turn
    ]

    if not selected_turns:
        raise ValueError(
            f"No transcript turns found for turn_range={window.get('turn_range')} "
            f"in vignette_id={vignette.get('vignette_id')}"
        )

    lines: List[str] = []
    for turn in selected_turns:
        speaker = str(turn.get("speaker", "Unknown")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if utterance:
            lines.append(f"{speaker}: {utterance}")

    if not lines:
        raise ValueError(
            f"Selected transcript turns were empty for window_id={window.get('window_id')}"
        )

    return "\n".join(lines)


def build_slug(vignette: Dict[str, Any], window: Dict[str, Any]) -> str:
    return f"{vignette['vignette_id'].lower()}__{window['window_id'].lower()}"


def package_vignette_result(
    *,
    vignette: Dict[str, Any],
    window: Dict[str, Any],
    query_text: str,
    condition: str,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    retrieval_payload: Dict[str, Any],
    prompt_payload: Dict[str, Any],
    generation_text: str,
) -> Dict[str, Any]:
    return {
        "created_at_utc": now_utc_iso(),
        "dataset_name": "IFS_Synthetic_Vignette_Dataset",
        "vignette_id": vignette.get("vignette_id"),
        "window_id": window.get("window_id"),
        "title": vignette.get("title"),
        "presenting_issue": vignette.get("presenting_issue"),
        "client_profile": vignette.get("client_profile"),
        "core_emotional_theme": vignette.get("core_emotional_theme"),
        "likely_parts": vignette.get("likely_parts"),
        "difficulty_level": vignette.get("difficulty_level"),
        "safety_flag": vignette.get("safety_flag"),
        "window_turn_range": window.get("turn_range"),
        "target_ifs_goal": window.get("target_ifs_goal"),
        "expected_retrieval_theme": window.get("expected_retrieval_theme"),
        "condition": condition,
        "query": query_text,
        "retrieval_model_name": retrieval_model_name,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retrieval": retrieval_payload,
        "prompt": prompt_payload,
        "generated_response": generation_text,
    }


def run_one_condition_for_window(
    *,
    vignette: Dict[str, Any],
    window: Dict[str, Any],
    query_text: str,
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
    retrieval_model=None,
) -> Dict[str, Any]:
    retrieval_payload = run_retrieval_for_condition(
        query=query_text,
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
        query=query_text,
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

    return package_vignette_result(
        vignette=vignette,
        window=window,
        query_text=query_text,
        condition=condition,
        retrieval_model_name=retrieval_model_name,
        llm_provider=llm_provider,
        llm_model=llm_model,
        retrieval_payload=retrieval_payload,
        prompt_payload=prompt_payload,
        generation_text=generation_text,
    )


def resolve_conditions(condition: str) -> List[str]:
    if condition == "all":
        return ["small_fixed", "medium_overlap", "hierarchical"]
    return [condition]


def run_window(
    *,
    vignette: Dict[str, Any],
    window: Dict[str, Any],
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    condition: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    output_dir: Path,
) -> None:
    query_text = build_consultation_window_text(vignette, window)
    slug = build_slug(vignette, window)

    print("\n" + "#" * 100)
    print(f"VIGNETTE: {vignette['vignette_id']} | WINDOW: {window['window_id']}")
    print(f"TITLE: {vignette.get('title')}")
    print(f"TURN_RANGE: {window.get('turn_range')}")
    print("#" * 100)
    print("\nCONSULTATION WINDOW\n")
    print(query_text)

    retrieval_model = load_retrieval_model(retrieval_model_name)

    for cond in resolve_conditions(condition):
        payload = run_one_condition_for_window(
            vignette=vignette,
            window=window,
            query_text=query_text,
            condition=cond,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=retrieval_model_name,
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k=top_k,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            retrieval_model=retrieval_model,
        )

        print_generation_result(payload)
        out_path = make_output_path(output_dir, slug, cond)
        save_json(out_path, payload)
        print(f"\nSaved vignette pilot payload to: {out_path}")


def run_all_windows_in_vignette(
    *,
    vignette: Dict[str, Any],
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    condition: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    output_dir: Path,
) -> None:
    windows = vignette.get("consultation_windows", [])
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"Vignette {vignette.get('vignette_id')} has no consultation_windows")

    for window in windows:
        run_window(
            vignette=vignette,
            window=window,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=retrieval_model_name,
            llm_provider=llm_provider,
            llm_model=llm_model,
            condition=condition,
            top_k=top_k,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            output_dir=output_dir,
        )


def run_all_vignettes(
    *,
    vignettes: List[Dict[str, Any]],
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    condition: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    output_dir: Path,
) -> None:
    for vignette in vignettes:
        run_all_windows_in_vignette(
            vignette=vignette,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=retrieval_model_name,
            llm_provider=llm_provider,
            llm_model=llm_model,
            condition=condition,
            top_k=top_k,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            output_dir=output_dir,
        )


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
        choices=["mock", "openai"],
        default=DEFAULT_LLM_PROVIDER,
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