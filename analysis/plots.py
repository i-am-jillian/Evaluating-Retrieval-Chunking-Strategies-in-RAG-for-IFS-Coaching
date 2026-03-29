from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_SUMMARY_CSV = "data/experiment_runs/score_summary_by_condition.csv"
DEFAULT_OUTPUT_DIR = "data/experiment_runs/plots"


def read_summary_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(rows, metric_key: str, title: str, output_path: Path) -> None:
    conditions = [row["condition"] for row in rows if row.get(metric_key, "") != ""]
    values = [float(row[metric_key]) for row in rows if row.get(metric_key, "") != ""]

    plt.figure(figsize=(7, 4))
    plt.bar(conditions, values)
    plt.title(title)
    plt.ylabel(metric_key)
    plt.xlabel("retrieval condition")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot preliminary score summaries by retrieval condition.")
    parser.add_argument("--summary-csv", type=str, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows = read_summary_csv(summary_csv)

    plot_metric(
        rows,
        metric_key="mean_ifs_adherence",
        title="Mean IFS Adherence by Retrieval Condition",
        output_path=output_dir / "mean_ifs_adherence.png",
    )

    plot_metric(
        rows,
        metric_key="mean_grounding",
        title="Mean Grounding by Retrieval Condition",
        output_path=output_dir / "mean_grounding.png",
    )

    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()