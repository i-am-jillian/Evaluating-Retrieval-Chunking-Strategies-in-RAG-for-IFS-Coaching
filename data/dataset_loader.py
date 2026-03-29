from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


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