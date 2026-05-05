import re
import pandas as pd
from pathlib import Path

INPUT_CSV = "data/generation_pilot_scores.csv"
OUTPUT_CSV = "scripts/generation_pilot_scores_graded_v2.csv"


CASE_THEMES = {
    "v001": ["work", "failure", "shut", "avoid", "not good enough", "procrastinat"],
    "v002": ["mistake", "careless", "critic", "embarrass", "lazy", "self-critical"],
    "v003": ["say yes", "agreeable", "rejected", "too much", "people", "conflict"],
    "v004": ["numb", "flat", "wall", "arguments", "overwhelmed", "hurt"],
    "v005": ["rest", "lazy", "demanding", "worthless", "control", "productive"],
    "v006": ["drink", "buzzing", "pressure", "loneliness", "critic", "numb"],
    "v007": ["partner", "defensive", "wall", "accused", "connection", "closeness"],
    "v008": ["crowded", "scanning", "danger", "survival", "escape", "terrified"],
    "v009": ["guilt", "responsibility", "caretaking", "abandoning", "exhausted"],
    "v010": ["snapped", "anger", "critic", "shame", "overwhelmed", "bad person"],
    "v011": ["job", "irresponsible", "stability", "freedom", "risk", "polarization"],
    "v012": ["chest", "throat", "tight", "bracing", "crying", "exposed"],
}


IFS_CORE = [
    "part", "protector", "self", "curious", "curiosity", "compassion",
    "unblend", "manager", "firefighter", "exile", "positive intention",
    "nonjudgment", "non-judgment", "befriend"
]

GROUNDING_MARKERS = [
    "grounding notes", "retrieved", "passage", "rank", "source",
    "based on", "supported", "evidence"
]

GENERIC_PHRASES = [
    "it sounds like", "it makes sense", "working very hard",
    "protect you", "stay curious", "positive intention",
    "approach with curiosity"
]

BAD_CLAIMS = [
    "diagnosis", "cure", "guaranteed", "always works",
    "scientifically proven", "you should stop medication",
    "this means you have"
]


def clean(x):
    return str(x).lower()


def extract_vignette_id(row):
    file = clean(row.get("file", ""))
    query = clean(row.get("query", ""))

    m = re.search(r"v\d{3}", file)
    if m:
        return m.group(0)

    m = re.search(r"v\d{3}", query)
    if m:
        return m.group(0)

    return ""


def count_hits(text, terms):
    return sum(1 for t in terms if t in text)


def score_ifs_adherence(response):
    r = clean(response)

    hits = count_hits(r, IFS_CORE)

    has_parts_frame = "part" in r
    has_protector_logic = "protect" in r or "positive intention" in r
    has_curiosity = "curious" in r or "curiosity" in r
    non_pathologizing = not any(w in r for w in ["bad person", "broken", "wrong with you", "disordered"])

    score = 1
    if has_parts_frame:
        score += 1
    if has_protector_logic:
        score += 1
    if has_curiosity:
        score += 1
    if hits >= 6 and non_pathologizing:
        score += 1

    return min(score, 5)


def score_grounding(response, sources):
    r = clean(response)
    s = clean(sources)

    score = 1

    if len(s.strip()) > 20:
        score += 1
    if ";" in s:
        score += 1
    if "grounding notes" in r:
        score += 1
    if any(marker in r for marker in ["rank", "source", "passage", "retrieved"]):
        score += 1

    return min(score, 5)


def score_specificity(row):
    response = clean(row.get("generated_response", ""))
    query = clean(row.get("query", ""))
    vignette = extract_vignette_id(row)

    theme_terms = CASE_THEMES.get(vignette, [])

    theme_hits = count_hits(response, theme_terms)
    query_words = set(re.findall(r"[a-zA-Z]{5,}", query))
    response_words = set(re.findall(r"[a-zA-Z]{5,}", response))

    overlap = len(query_words.intersection(response_words))

    score = 1

    if theme_hits >= 1:
        score += 1
    if theme_hits >= 3:
        score += 1
    if overlap >= 5:
        score += 1
    if overlap >= 10:
        score += 1

    return min(score, 5)


def score_retrieval_relevance(row):
    sources = clean(row.get("top_retrieval_sources", ""))
    query = clean(row.get("query", ""))
    vignette = extract_vignette_id(row)

    theme_terms = CASE_THEMES.get(vignette, [])

    source_hits = count_hits(sources, theme_terms)
    query_hits = count_hits(sources, list(set(re.findall(r"[a-zA-Z]{6,}", query))))

    score = 1

    if len(sources.strip()) > 20:
        score += 1
    if ";" in sources:
        score += 1
    if source_hits >= 1 or query_hits >= 1:
        score += 1
    if source_hits >= 2 or query_hits >= 3:
        score += 1

    return min(score, 5)


def flag_generic(response):
    r = clean(response)

    generic_hits = count_hits(r, GENERIC_PHRASES)
    specific_signals = count_hits(r, [
        "failure", "mistake", "rejected", "numb", "drink", "scanning",
        "partner", "rest", "anger", "job", "tightness", "caretaking"
    ])

    return "yes" if generic_hits >= 4 and specific_signals <= 1 else "no"


def flag_hallucination(response):
    r = clean(response)
    return "yes" if any(claim in r for claim in BAD_CLAIMS) else "no"


def build_notes(row):
    notes = []

    if row["ifs_adherence_score"] >= 4:
        notes.append("strong IFS framing")
    if row["grounding_score"] >= 4:
        notes.append("explicit grounding behavior")
    if row["specificity_score"] <= 2:
        notes.append("could be more case-specific")
    if row["genericity_flag"] == "yes":
        notes.append("some template-like phrasing")
    if row["hallucination_flag"] == "yes":
        notes.append("possible unsupported claim")

    return "; ".join(notes)


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[~df["generated_response"].astype(str).str.contains(r"\[MOCK OUTPUT\]", regex=True, na=False)].copy()

    df["ifs_adherence_score"] = df["generated_response"].apply(score_ifs_adherence)
    df["grounding_score"] = df.apply(lambda r: score_grounding(r["generated_response"], r["top_retrieval_sources"]), axis=1)
    df["specificity_score"] = df.apply(score_specificity, axis=1)
    df["retrieval_relevance_score"] = df.apply(score_retrieval_relevance, axis=1)
    df["genericity_flag"] = df["generated_response"].apply(flag_generic)
    df["hallucination_flag"] = df["generated_response"].apply(flag_hallucination)
    df["notes"] = df.apply(build_notes, axis=1)
    df["reviewer"] = "auto_eval_v2"

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved graded CSV to: {OUTPUT_CSV}")
    print(f"Rows graded: {len(df)}")
    print("\nMean scores by condition:")
    print(
        df.groupby("condition")[
            [
                "ifs_adherence_score",
                "grounding_score",
                "specificity_score",
                "retrieval_relevance_score",
            ]
        ].mean().round(2)
    )


if __name__ == "__main__":
    main()