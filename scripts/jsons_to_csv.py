from __future__ import annotations

import csv
import json
from pathlib import Path


INPUT_DIR = Path("data/generation_pilot")
OUTPUT_CSV = Path("data/generation_pilot_scores.csv")


def get_top_titles(retrieval: dict, condition: str) -> str:
    if condition == "hierarchical":
        rows = retrieval.get("child_results", [])
    else:
        rows = retrieval.get("results", [])

    titles = []
    for row in rows[:3]:
        title = row.get("source_title", "")
        section = row.get("section_path_str", "")
        titles.append(f"{title} | {section}")

    return " ; ".join(titles)


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    rows = []

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        condition = payload.get("condition", "")
        retrieval = payload.get("retrieval", {})

        rows.append({
            "file": path.name,
            "condition": condition,
            "query": payload.get("query", ""),
            "generated_response": payload.get("generated_response", ""),
            "top_retrieval_sources": get_top_titles(retrieval, condition),

            # manual scoring columns
            "ifs_adherence_score": "",
            "grounding_score": "",
            "specificity_score": "",
            "retrieval_relevance_score": "",
            "genericity_flag": "",
            "hallucination_flag": "",
            "notes": "",
            "reviewer": "",
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to: {OUTPUT_CSV}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()