from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT_DIR = "data/experiment_runs"
DEFAULT_SCORE_CSV = "data/experiment_runs/scored_outputs.csv"
DEFAULT_SUMMARY_CSV = "data/experiment_runs/score_summary_by_condition.csv"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preview(text: str, n: int = 280) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


def extract_top_sources(payload: Dict[str, Any]) -> str:
    retrieval = payload.get("retrieval", {})
    condition = payload.get("condition")

    if condition == "hierarchical":
        rows = retrieval.get("child_results", [])
    else:
        rows = retrieval.get("results", [])

    tops = []
    for row in rows[:3]:
        title = row.get("source_title", "UNKNOWN_TITLE")
        section = row.get("section_path_str", "UNKNOWN_SECTION")
        tops.append(f"{title} :: {section}")

    return " | ".join(tops)


def collect_output_rows(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = load_json(path)

        rows.append(
            {
                "source_file": path.name,
                "vignette_id": payload.get("vignette_id"),
                "window_id": payload.get("window_id"),
                "condition": payload.get("condition"),
                "target_ifs_goal": payload.get("target_ifs_goal"),
                "expected_retrieval_theme": payload.get("expected_retrieval_theme"),
                "query_preview": preview(payload.get("query", ""), 300),
                "response_preview": preview(payload.get("generated_response", ""), 500),
                "top_sources": extract_top_sources(payload),

                # Manual scoring columns
                "parts_language": "",
                "curiosity": "",
                "non_pathologizing": "",
                "ifs_alignment": "",
                "ifs_adherence_score": "",
                "grounding_score": "",
                "failure_notes": "",
                "reviewer": "",
            }
        )
    return rows


def write_score_template(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No JSON outputs found to score.")

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_scored_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: str) -> int | None:
    value = (value or "").strip()
    if value == "":
        return None
    return int(value)


def summarize_scores(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for row in rows:
        cond = row["condition"]
        grouped[cond].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for condition, cond_rows in grouped.items():
        adherence_vals = []
        grounding_vals = []
        scored_count = 0

        for row in cond_rows:
            adherence = safe_int(row.get("ifs_adherence_score", ""))
            grounding = safe_int(row.get("grounding_score", ""))

            if adherence is not None and grounding is not None:
                adherence_vals.append(adherence)
                grounding_vals.append(grounding)
                scored_count += 1

        summary_rows.append(
            {
                "condition": condition,
                "total_outputs": len(cond_rows),
                "scored_outputs": scored_count,
                "mean_ifs_adherence": round(sum(adherence_vals) / len(adherence_vals), 3) if adherence_vals else "",
                "mean_grounding": round(sum(grounding_vals) / len(grounding_vals), 3) if grounding_vals else "",
            }
        )

    return summary_rows


def write_summary_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No summary rows to write.")

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manual scoring sheet from experiment outputs, or summarize filled-in scores."
    )
    parser.add_argument(
        "--mode",
        choices=["template", "summary"],
        required=True,
        help="template = generate scoring CSV from JSON outputs; summary = summarize filled-in scoring CSV",
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--score-csv", type=str, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--summary-csv", type=str, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    score_csv = Path(args.score_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()

    if args.mode == "template":
        rows = collect_output_rows(input_dir)
        write_score_template(rows, score_csv)
        print(f"Wrote scoring template: {score_csv}")
        print(f"Rows: {len(rows)}")

    elif args.mode == "summary":
        rows = read_scored_csv(score_csv)
        summary_rows = summarize_scores(rows)
        write_summary_csv(summary_rows, summary_csv)
        print(f"Wrote summary CSV: {summary_csv}")
        print(f"Conditions summarized: {len(summary_rows)}")


if __name__ == "__main__":
    main()