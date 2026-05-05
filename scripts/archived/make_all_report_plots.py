import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

pretty = {
    "ifs_adherence_score": "IFS\nAdherence",
    "grounding_score": "Grounding",
    "specificity_score": "Specificity",
    "retrieval_relevance_score": "Retrieval\nRelevance",
}

condition_order = ["small_fixed", "medium_overlap", "hierarchical"]
df = df[df["condition"].isin(condition_order)]

summary = df.groupby("condition")[metrics].mean()
summary = summary.loc[[c for c in condition_order if c in summary.index]]

# -----------------------------
# 1. Heatmap
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
data = summary.values

im = ax.imshow(data, aspect="auto")

ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels([pretty[m] for m in metrics])
ax.set_yticks(np.arange(len(summary.index)))
ax.set_yticklabels(summary.index)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=10)

ax.set_title("Evaluation Score Heatmap by Retrieval Strategy", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, label="Average Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "score_heatmap.png", dpi=300)
plt.savefig(OUT_DIR / "score_heatmap.pdf")
plt.close()

# -----------------------------
# 2. Radar Chart
# -----------------------------
labels = [pretty[m].replace("\n", " ") for m in metrics]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for condition in summary.index:
    values = summary.loc[condition].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=condition)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 5)
ax.set_title("Metric Tradeoffs Across Retrieval Strategies", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()
plt.savefig(OUT_DIR / "radar_metric_tradeoffs.png", dpi=300)
plt.savefig(OUT_DIR / "radar_metric_tradeoffs.pdf")
plt.close()

# -----------------------------
# 3. Win Count Chart
# -----------------------------
df["overall_score"] = df[metrics].mean(axis=1)

# Use window_id if available, otherwise file
group_col = "window_id" if "window_id" in df.columns else "file"

wins = []
for key, group in df.groupby(group_col):
    group = group.dropna(subset=["overall_score"])
    if group.empty:
        continue
    best = group.sort_values("overall_score", ascending=False).iloc[0]
    wins.append(best["condition"])

win_counts = pd.Series(wins).value_counts()
win_counts = win_counts.reindex(condition_order).fillna(0)

plt.figure(figsize=(7, 4.5))
ax = win_counts.plot(kind="bar")
plt.title("Best Overall Condition per Consultation Window", fontsize=14, fontweight="bold")
plt.xlabel("Retrieval Strategy")
plt.ylabel("Number of Window Wins")
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", fontsize=10, padding=3)

plt.tight_layout()
plt.savefig(OUT_DIR / "condition_win_counts.png", dpi=300)
plt.savefig(OUT_DIR / "condition_win_counts.pdf")
plt.close()

# -----------------------------
# 4. Score Distribution Boxplot
# -----------------------------
df_long = df.melt(
    id_vars=["condition"],
    value_vars=metrics,
    var_name="metric",
    value_name="score",
)

df_long["metric"] = df_long["metric"].map(lambda x: pretty[x].replace("\n", " "))

plt.figure(figsize=(10, 5))
data_to_plot = [
    df[df["condition"] == condition]["overall_score"].dropna()
    for condition in condition_order
    if condition in df["condition"].unique()
]

plt.boxplot(data_to_plot, labels=[c for c in condition_order if c in df["condition"].unique()])
plt.title("Distribution of Overall Scores by Retrieval Strategy", fontsize=14, fontweight="bold")
plt.xlabel("Retrieval Strategy")
plt.ylabel("Overall Score")
plt.ylim(0, 5.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "overall_score_distribution.png", dpi=300)
plt.savefig(OUT_DIR / "overall_score_distribution.pdf")
plt.close()

print("Saved plots to:", OUT_DIR)
print("Files created:")
for p in sorted(OUT_DIR.glob("*.png")):
    print("-", p)