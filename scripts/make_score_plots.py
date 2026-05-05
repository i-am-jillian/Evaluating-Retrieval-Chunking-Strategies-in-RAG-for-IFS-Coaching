import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "scripts/generation_pilot_scores_graded_v2.csv"
OUT_DIR = Path("data/results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

metrics = [
    "ifs_adherence_score",
    "grounding_score",
    "specificity_score",
    "retrieval_relevance_score",
]

pretty_names = {
    "ifs_adherence_score": "IFS Adherence",
    "grounding_score": "Grounding",
    "specificity_score": "Specificity",
    "retrieval_relevance_score": "Retrieval Relevance",
}

summary = df.groupby("condition")[metrics].mean()

# nicer condition order
order = ["small_fixed", "medium_overlap", "hierarchical"]
summary = summary.loc[[c for c in order if c in summary.index]]

# rename for plot
summary = summary.rename(columns=pretty_names)

plt.figure(figsize=(10, 6))
ax = summary.plot(kind="bar", figsize=(10, 6), width=0.75)

plt.title("Average Evaluation Scores by Retrieval Strategy", fontsize=15, fontweight="bold")
plt.xlabel("Retrieval Strategy", fontsize=12)
plt.ylabel("Average Score (1–5)", fontsize=12)
plt.ylim(0, 5.5)
plt.xticks(rotation=0)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

# add value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(OUT_DIR / "average_scores_by_condition.png", dpi=300)
plt.savefig(OUT_DIR / "average_scores_by_condition.pdf")
plt.close()

print("Saved:")
print(OUT_DIR / "average_scores_by_condition.png")
print(OUT_DIR / "average_scores_by_condition.pdf")