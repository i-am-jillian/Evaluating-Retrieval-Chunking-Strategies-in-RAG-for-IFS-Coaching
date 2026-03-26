from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


# Example:
# python preprocessing/structure_text.py
#
# or
#
# python preprocessing/structure_text.py \
#   --input-root data/ifs_corpus/clean \
#   --output-root data/ifs_corpus/structured


# -----------------------------
# Tuning parameters
# -----------------------------

MAX_BLOCK_WORDS_BEFORE_REBREAK = 300
TARGET_REBREAK_WORDS = 140
MIN_REBREAK_WORDS = 80
MIN_SUBSTANTIAL_PARAGRAPH_WORDS = 20


# -----------------------------
# Sorting helpers
# -----------------------------

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


# -----------------------------
# Metadata inference
# -----------------------------

TITLE_MAP = {
    "ifs_couple_therapy": "Internal Family Systems Couple Therapy",
    "ifs_new_dimensions": "Internal Family Systems Therapy: New Dimensions",
    "ifs_skills_training": "Internal Family Systems Skills Training Manual",
    "ifs_therapy": "Internal Family Systems Therapy",
    "ifs_therapy_for_addiction": "Internal Family Systems Therapy for Addictions",
    "transcending_trauma": "Transcending Trauma",
}


def infer_doc_id(book_dir: Path) -> str:
    return book_dir.name.strip().lower()


def infer_doc_title(book_dir: Path) -> str:
    doc_id = infer_doc_id(book_dir)
    if doc_id in TITLE_MAP:
        return TITLE_MAP[doc_id]
    return book_dir.name.replace("_", " ").strip().title()


# -----------------------------
# Text normalization
# -----------------------------

def normalize_unicode(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u00ad": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_block(block: str) -> str:
    block = normalize_unicode(block)

    lines = [line.strip() for line in block.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return ""

    block = "\n".join(lines)

    block = re.sub(r"^\|\s*\d+\s+", "", block)
    block = re.sub(r"^\d+\s*\n", "", block)
    block = re.sub(r"^_+\s*$", "", block)
    block = re.sub(r"[ \t]{2,}", " ", block)

    return block.strip()


def split_into_blocks(text: str) -> List[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = []

    for block in raw_blocks:
        block = normalize_block(block)
        if block:
            blocks.append(block)

    return blocks


# -----------------------------
# Heuristics
# -----------------------------

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def alpha_chars(text: str) -> List[str]:
    return [c for c in text if c.isalpha()]


def uppercase_ratio(text: str) -> float:
    chars = alpha_chars(text)
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isupper()) / len(chars)


def looks_like_sentence(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    if text.endswith((".", "!", "?")):
        return True

    words = re.findall(r"\b\w+\b", text)
    lower_words = sum(1 for w in words if w.islower())
    return len(words) >= 10 and lower_words >= 5


def is_likely_header(block: str) -> bool:
    text = block.strip()
    if not text:
        return False

    wc = word_count(text)
    if wc == 0:
        return False
    if wc > 14:
        return False
    if len(text) > 100:
        return False
    if looks_like_sentence(text):
        return False

    mostly_upper = uppercase_ratio(text) > 0.60
    title_case_like = text == text.title()
    weird_caps_fragment = bool(re.fullmatch(r"[A-Z\s\-:&]+", text)) and wc <= 12
    no_terminal_punct = not text.endswith((".", "!", "?", ";"))

    return no_terminal_punct and (mostly_upper or title_case_like or weird_caps_fragment)


def is_substantial_paragraph(block: str) -> bool:
    return word_count(block) >= MIN_SUBSTANTIAL_PARAGRAPH_WORDS


def is_probable_noise_header(block: str) -> bool:
    text = block.strip()
    if not text:
        return False

    tokens = text.split()
    short_caps_tokens = sum(1 for t in tokens if t.isupper() and len(t) <= 3)
    return short_caps_tokens >= 2 and uppercase_ratio(text) > 0.5


def infer_paragraph_type(text: str, has_attached_header: bool, header_text: Optional[str]) -> str:
    stripped = text.strip()

    if not stripped:
        return "empty"

    if has_attached_header and header_text and word_count(header_text) <= 12:
        return "headed_body"

    if re.fullmatch(r"[\-\*\u2022].+", stripped):
        return "list_item"

    if re.match(r"^\d+\.", stripped):
        return "numbered_item"

    if stripped.count("?") >= 2 and word_count(stripped) < 120:
        return "exercise_or_prompt"

    return "body"


# -----------------------------
# Oversized block re-splitting
# -----------------------------

def split_oversized_block(block: str) -> List[str]:
    text = re.sub(r"\s+", " ", block).strip()

    if word_count(text) <= MAX_BLOCK_WORDS_BEFORE_REBREAK:
        return [text]

    text = re.sub(r"\s+(\d+\.\s+)", r"\n\n\1", text)

    text = re.sub(
        r'(?<=[\.\?\!])\s+((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5}))\s+',
        lambda m: f"\n\n{m.group(1)} ",
        text
    )

    coarse_parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    final_segments: List[str] = []

    for part in coarse_parts:
        if word_count(part) <= MAX_BLOCK_WORDS_BEFORE_REBREAK:
            final_segments.append(part)
            continue

        sentences = re.split(r'(?<=[\.\?\!])\s+', part)
        current: List[str] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            candidate = " ".join(current + [sent]).strip()
            if current and word_count(candidate) > TARGET_REBREAK_WORDS:
                final_segments.append(" ".join(current).strip())
                current = [sent]
            else:
                current.append(sent)

        if current:
            final_segments.append(" ".join(current).strip())

    merged: List[str] = []
    for seg in final_segments:
        seg = seg.strip()
        if not seg:
            continue

        if merged and word_count(seg) < MIN_REBREAK_WORDS:
            merged[-1] = f"{merged[-1]} {seg}".strip()
        else:
            merged.append(seg)

    return [seg for seg in merged if seg]


def expand_blocks(blocks: List[str]) -> List[str]:
    expanded = []
    for block in blocks:
        if word_count(block) > MAX_BLOCK_WORDS_BEFORE_REBREAK:
            expanded.extend(split_oversized_block(block))
        else:
            expanded.append(block)
    return expanded


# -----------------------------
# Structure inference
# -----------------------------

def clean_label_from_token(token: str) -> str:
    token = token.replace("-", " ").replace("_", " ").strip()
    token = re.sub(r"\s+", " ", token)
    return token.title()


def infer_structure_from_filename(file_name: str, doc_id: str) -> Dict[str, Any]:
    stem = file_name.replace(".words.txt", "")

    remainder = stem
    if remainder.startswith(doc_id + "_"):
        remainder = remainder[len(doc_id) + 1:]
    elif remainder == doc_id:
        remainder = ""

    parts = remainder.split("_") if remainder else []

    intro_like = {"intro", "introduction", "preface", "foreword", "prologue", "epilogue"}
    chapter_token = None
    section_token = None
    part_token = None
    label_tokens: List[str] = []

    for token in parts:
        if re.fullmatch(r"ch\d+[a-zA-Z]*", token):
            chapter_token = token
        elif re.fullmatch(r"sec\d+[a-zA-Z]*", token):
            section_token = token
        elif re.fullmatch(r"pt\d+[a-zA-Z]*", token):
            part_token = token
        else:
            label_tokens.append(token)

    section_path: List[str] = []

    if parts and parts[0].lower() in intro_like:
        section_path = [clean_label_from_token(parts[0])]
        if len(parts) > 1:
            section_path.extend(clean_label_from_token(t) for t in parts[1:])
    else:
        if chapter_token:
            section_path.append(chapter_token.upper())
        if section_token:
            section_path.append(section_token.upper())
        if part_token:
            section_path.append(part_token.upper())
        if label_tokens:
            section_path.append(clean_label_from_token(" ".join(label_tokens)))

    if not section_path:
        section_path = [stem]

    section_path_str = " > ".join(section_path)
    node_id = f"{doc_id}::{stem}::node"

    return {
        "node_id": node_id,
        "node_type": "section",
        "chapter_label": chapter_token.upper() if chapter_token else None,
        "section_label": section_token.upper() if section_token else None,
        "part_label": part_token.upper() if part_token else None,
        "section_path": section_path,
        "section_path_str": section_path_str,
    }


# -----------------------------
# Data model
# -----------------------------

@dataclass
class ParagraphRecord:
    paragraph_id: int
    source_file: str
    source_file_index: int

    node_id: str
    node_type: str
    section_path: List[str]
    section_path_str: str
    chapter_label: Optional[str]
    section_label: Optional[str]
    part_label: Optional[str]

    source_block_start: int
    source_block_end: int

    has_attached_header: bool
    header_text: Optional[str]
    header_is_noisy: bool

    paragraph_type: str

    word_count: int
    char_count: int
    char_start_in_doc: int
    char_end_in_doc: int

    text: str


# -----------------------------
# Core logic for one file
# -----------------------------

def build_paragraphs_for_file(
    file_path: Path,
    file_index: int,
    doc_id: str,
) -> List[ParagraphRecord]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = normalize_unicode(text)
    text = normalize_whitespace(text)

    blocks = split_into_blocks(text)
    blocks = expand_blocks(blocks)

    file_structure = infer_structure_from_filename(file_path.name, doc_id)

    paragraphs: List[ParagraphRecord] = []
    local_paragraph_id = 0
    i = 0

    while i < len(blocks):
        current = blocks[i]

        if (
            is_likely_header(current)
            and i + 1 < len(blocks)
            and is_substantial_paragraph(blocks[i + 1])
        ):
            header = current
            body = blocks[i + 1]
            combined = f"{header}\n\n{body}".strip()

            paragraphs.append(
                ParagraphRecord(
                    paragraph_id=local_paragraph_id,
                    source_file=file_path.name,
                    source_file_index=file_index,

                    node_id=file_structure["node_id"],
                    node_type=file_structure["node_type"],
                    section_path=file_structure["section_path"],
                    section_path_str=file_structure["section_path_str"],
                    chapter_label=file_structure["chapter_label"],
                    section_label=file_structure["section_label"],
                    part_label=file_structure["part_label"],

                    source_block_start=i,
                    source_block_end=i + 1,

                    has_attached_header=True,
                    header_text=header,
                    header_is_noisy=is_probable_noise_header(header),

                    paragraph_type=infer_paragraph_type(combined, True, header),

                    word_count=word_count(combined),
                    char_count=len(combined),
                    char_start_in_doc=0,
                    char_end_in_doc=0,

                    text=combined,
                )
            )
            local_paragraph_id += 1
            i += 2
        else:
            paragraphs.append(
                ParagraphRecord(
                    paragraph_id=local_paragraph_id,
                    source_file=file_path.name,
                    source_file_index=file_index,

                    node_id=file_structure["node_id"],
                    node_type=file_structure["node_type"],
                    section_path=file_structure["section_path"],
                    section_path_str=file_structure["section_path_str"],
                    chapter_label=file_structure["chapter_label"],
                    section_label=file_structure["section_label"],
                    part_label=file_structure["part_label"],

                    source_block_start=i,
                    source_block_end=i,

                    has_attached_header=False,
                    header_text=None,
                    header_is_noisy=False,

                    paragraph_type=infer_paragraph_type(current, False, None),

                    word_count=word_count(current),
                    char_count=len(current),
                    char_start_in_doc=0,
                    char_end_in_doc=0,

                    text=current,
                )
            )
            local_paragraph_id += 1
            i += 1

    return paragraphs


# -----------------------------
# One book folder -> one JSON
# -----------------------------

def structure_book(book_dir: Path) -> Dict[str, Any]:
    files = sorted(book_dir.glob("*.words.txt"), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No .words.txt files found in {book_dir}")

    doc_id = infer_doc_id(book_dir)
    doc_title = infer_doc_title(book_dir)

    all_paragraphs: List[ParagraphRecord] = []
    nodes: List[Dict[str, Any]] = []
    node_map: Dict[str, Dict[str, Any]] = {}

    for file_index, file_path in enumerate(files):
        file_structure = infer_structure_from_filename(file_path.name, doc_id)
        node_id = file_structure["node_id"]

        if node_id not in node_map:
            node = {
                "node_id": node_id,
                "node_type": file_structure["node_type"],
                "source_file": file_path.name,
                "source_file_index": file_index,
                "chapter_label": file_structure["chapter_label"],
                "section_label": file_structure["section_label"],
                "part_label": file_structure["part_label"],
                "section_path": file_structure["section_path"],
                "section_path_str": file_structure["section_path_str"],
                "paragraph_ids": [],
                "char_start_in_doc": None,
                "char_end_in_doc": None,
            }
            node_map[node_id] = node
            nodes.append(node)

        file_paragraphs = build_paragraphs_for_file(file_path, file_index, doc_id)
        all_paragraphs.extend(file_paragraphs)

    global_char_cursor = 0

    for global_idx, para in enumerate(all_paragraphs):
        para.paragraph_id = global_idx
        para.char_start_in_doc = global_char_cursor
        para.char_end_in_doc = global_char_cursor + len(para.text)

        node = node_map[para.node_id]
        node["paragraph_ids"].append(global_idx)

        if node["char_start_in_doc"] is None:
            node["char_start_in_doc"] = para.char_start_in_doc
        node["char_end_in_doc"] = para.char_end_in_doc

        global_char_cursor = para.char_end_in_doc + 2

    doc = {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "source_dir": str(book_dir),
        "source_files": [p.name for p in files],
        "paragraph_count": len(all_paragraphs),
        "node_count": len(nodes),
        "nodes": nodes,
        "paragraphs": [asdict(p) for p in all_paragraphs],
    }
    return doc


# -----------------------------
# Reporting
# -----------------------------

def print_quality_report(doc: Dict[str, Any]) -> None:
    paragraphs = doc["paragraphs"]
    counts = [p["word_count"] for p in paragraphs]

    if not counts:
        print("  No paragraphs found.")
        return

    counts_sorted = sorted(counts, reverse=True)
    over_300 = sum(c > 300 for c in counts)
    over_500 = sum(c > 500 for c in counts)
    over_800 = sum(c > 800 for c in counts)

    print(f"  Paragraphs: {len(counts)}")
    print(f"  Nodes: {doc.get('node_count', 0)}")
    print(f"  Max word count: {max(counts)}")
    print(f"  >300 words: {over_300}")
    print(f"  >500 words: {over_500}")
    print(f"  >800 words: {over_800}")
    print(f"  Top 10 word counts: {counts_sorted[:10]}")


# -----------------------------
# Recursive driver
# -----------------------------

def find_book_dirs(input_root: Path) -> List[Path]:
    return sorted([p for p in input_root.iterdir() if p.is_dir()])


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_all_books(input_root: Path, output_root: Path) -> None:
    book_dirs = find_book_dirs(input_root)

    if not book_dirs:
        raise FileNotFoundError(f"No subdirectories found in {input_root}")

    print(f"Found {len(book_dirs)} book folders in {input_root}\n")

    for book_dir in book_dirs:
        word_files = sorted(book_dir.glob("*.words.txt"), key=natural_key)

        if not word_files:
            print(f"Skipping {book_dir.name}: no .words.txt files found\n")
            continue

        doc = structure_book(book_dir)
        output_path = output_root / f"{doc['doc_id']}.json"
        write_json(doc, output_path)

        print(f"Wrote: {output_path}")
        print(f"  Title: {doc['doc_title']}")
        print(f"  Source files: {len(doc['source_files'])}")
        print_quality_report(doc)
        print()


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively structure all cleaned IFS corpus folders into paragraph JSON files."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/ifs_corpus/clean",
        help="Root folder containing one subfolder per book",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/ifs_corpus/structured",
        help="Output folder for structured JSON files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    process_all_books(input_root, output_root)


if __name__ == "__main__":
    main()