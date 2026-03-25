from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any


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

# If a block is larger than this after blank-line splitting,
# try to recover paragraph-like units with a second-pass splitter.
MAX_BLOCK_WORDS_BEFORE_REBREAK = 300

# Target segment size for rebreaking oversized prose blocks.
TARGET_REBREAK_WORDS = 140

# If a split produces tiny fragments, merge them back.
MIN_REBREAK_WORDS = 80

# Header attachment rule:
# only attach a likely header to the next block if the next block
# looks substantial enough to be real paragraph text.
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
    "ifs_couple_therapy": "IFS Couple Therapy",
    "ifs_new_dimensions": "IFS New Dimensions",
    "ifs_skills_training": "IFS Skills Training Manual",
    "ifs_therapy": "IFS Therapy",
    "ifs_therapy_for_addiction": "IFS Therapy for Addictions",
    "transcending_trauma": "Transcending Trauma"
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

    # Remove obvious page/file artifacts
    block = re.sub(r"^\|\s*\d+\s+", "", block)   # leading pipe + page number
    block = re.sub(r"^\d+\s*\n", "", block)      # standalone page number line
    block = re.sub(r"^_+\s*$", "", block)        # underscore divider line
    block = re.sub(r"[ \t]{2,}", " ", block)     # repeated spaces

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


# -----------------------------
# Oversized block re-splitting
# -----------------------------

def split_oversized_block(block: str) -> List[str]:
    """
    Second-pass splitter for blocks that are too large because paragraph
    boundaries were lost during extraction/cleaning.

    Strategy:
    1. Flatten internal whitespace to make sentence splitting more reliable.
    2. Insert soft breaks before numbered list items.
    3. Insert soft breaks before likely inline heading-like phrases.
    4. Split into sentence-based segments only after the segment is already long.
    5. Merge tiny fragments back into neighbors.
    """
    text = re.sub(r"\s+", " ", block).strip()

    if word_count(text) <= MAX_BLOCK_WORDS_BEFORE_REBREAK:
        return [text]

    # Break before numbered list items, e.g. "1. " "2. "
    text = re.sub(r"\s+(\d+\.\s+)", r"\n\n\1", text)

    # Break before likely inline headings that appear after sentence endings.
    # This is intentionally conservative.
    text = re.sub(
        r'(?<=[\.\?\!])\s+((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5}))\s+',
        lambda m: f"\n\n{m.group(1)} ",
        text
    )

    # First split on inserted blank lines if any
    coarse_parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    final_segments: List[str] = []

    for part in coarse_parts:
        if word_count(part) <= MAX_BLOCK_WORDS_BEFORE_REBREAK:
            final_segments.append(part)
            continue

        # Still too large: split by sentence boundaries, but only once a segment gets long
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

    # Merge tiny fragments back into previous segment
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
# Data model
# -----------------------------

@dataclass
class ParagraphRecord:
    paragraph_id: int
    source_file: str
    source_file_index: int
    source_block_start: int
    source_block_end: int
    has_attached_header: bool
    header_text: str | None
    header_is_noisy: bool
    word_count: int
    text: str


# -----------------------------
# Core logic for one file
# -----------------------------

def build_paragraphs_for_file(file_path: Path, file_index: int) -> List[ParagraphRecord]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = normalize_unicode(text)
    text = normalize_whitespace(text)

    blocks = split_into_blocks(text)
    blocks = expand_blocks(blocks)

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
                    source_block_start=i,
                    source_block_end=i + 1,
                    has_attached_header=True,
                    header_text=header,
                    header_is_noisy=is_probable_noise_header(header),
                    word_count=word_count(combined),
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
                    source_block_start=i,
                    source_block_end=i,
                    has_attached_header=False,
                    header_text=None,
                    header_is_noisy=False,
                    word_count=word_count(current),
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

    all_paragraphs: List[ParagraphRecord] = []

    for file_index, file_path in enumerate(files):
        file_paragraphs = build_paragraphs_for_file(file_path, file_index)
        all_paragraphs.extend(file_paragraphs)

    # Renumber globally across the whole book
    for global_idx, para in enumerate(all_paragraphs):
        para.paragraph_id = global_idx

    doc = {
        "doc_id": infer_doc_id(book_dir),
        "doc_title": infer_doc_title(book_dir),
        "source_dir": str(book_dir),
        "source_files": [p.name for p in files],
        "paragraph_count": len(all_paragraphs),
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
    print(f"  Max word count: {max(counts)}")
    print(f"  >300 words: {over_300}")
    print(f"  >500 words: {over_500}")
    print(f"  >800 words: {over_800}")
    print(f"  Top 10 word counts: {counts_sorted[:10]}")


# -----------------------------
# Recursive driver
# -----------------------------

def find_book_dirs(input_root: Path) -> List[Path]:
    """
    Treat each immediate subdirectory of input_root as one book folder.
    """
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