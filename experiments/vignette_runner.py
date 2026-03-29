from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.dataset_loader import build_consultation_window_text, build_slug
from generation.generate import generate_text
from generation.prompt_template import make_prompt_payload
from experiments.pilot_utils import (
    load_retrieval_model,
    make_output_path,
    print_generation_result,
    run_retrieval_for_condition,
    save_json,
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def resolve_conditions(condition: str) -> List[str]:
    if condition == "all":
        return ["small_fixed", "medium_overlap", "hierarchical"]
    return [condition]


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

    generation_text_out = generate_text(
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
        generation_text=generation_text_out,
    )


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
    retrieval_model=None,
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
    retrieval_model=None,
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
            retrieval_model=retrieval_model,
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
    retrieval_model=None,
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
            retrieval_model=retrieval_model,
        )