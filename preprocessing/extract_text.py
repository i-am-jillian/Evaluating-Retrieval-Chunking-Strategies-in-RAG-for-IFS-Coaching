from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from statistics import median

import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = PROJECT_ROOT / "data" / "ifs_corpus" / "books"
RAW_DIR = PROJECT_ROOT / "data" / "ifs_corpus" / "raw"


# -----------------------------
# Basic extraction modes
# -----------------------------

def extract_page_text_mode(page: fitz.Page) -> str:
    text = page.get_text("text")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_page_blocks_mode(
    page: fitz.Page,
    top_margin_ratio: float = 0.05,
    bottom_margin_ratio: float = 0.03,
) -> str:
    page_rect = page.rect
    page_height = page_rect.height

    top_cutoff = page_height * top_margin_ratio
    bottom_cutoff = page_height * (1.0 - bottom_margin_ratio)

    blocks = page.get_text("blocks")
    kept_blocks = []

    for block in blocks:
        x0, y0, x1, y1, text, *_ = block

        if not text or not text.strip():
            continue
        if y1 < top_cutoff:
            continue
        if y0 > bottom_cutoff:
            continue

        kept_blocks.append((x0, y0, x1, y1, text.strip()))

    kept_blocks.sort(key=lambda b: (b[1], b[0]))
    page_text = "\n\n".join(block[4] for block in kept_blocks)
    return normalize_extracted_text(page_text)


# -----------------------------
# Words-mode helpers
# -----------------------------

def normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_line_text(line_text: str) -> str:
    line_text = line_text.replace(" ,", ",")
    line_text = line_text.replace(" .", ".")
    line_text = line_text.replace(" :", ":")
    line_text = line_text.replace(" ;", ";")
    line_text = line_text.replace(" !", "!")
    line_text = line_text.replace(" ?", "?")
    line_text = line_text.replace("( ", "(")
    line_text = line_text.replace(" )", ")")
    line_text = line_text.replace(" ’ ", "’")
    line_text = line_text.replace(" ' ", "'")
    line_text = line_text.replace(" - ", "-")
    line_text = re.sub(r"\s+", " ", line_text)
    return line_text.strip()


def is_probable_footer_or_boilerplate(text: str) -> bool:
    """
    Removes common ebook footer/citation junk seen in your raw outputs.
    """
    lowered = text.lower().strip()

    footer_patterns = [
        "proquest ebook central",
        "created from buffalo on",
        "http://ebookcentral.proquest.com",
        "copyright",
    ]

    return any(p in lowered for p in footer_patterns)


def is_page_marker_line(text: str) -> bool:
    return bool(re.fullmatch(r"=+\s*page\s+\d+\s*=+", text.strip(), flags=re.IGNORECASE))


def line_is_heading_like(text: str) -> bool:
    """
    Heading detector for extracted lines.
    Conservative: allows short chapter/title lines to stay separate.
    """
    t = text.strip()
    if not t:
        return False

    words = re.findall(r"\b\w+\b", t)
    if not words:
        return False

    if len(words) > 14:
        return False
    if len(t) > 120:
        return False

    # Strong examples:
    # CHAPTER 1
    # The Origins of Internal Family Systems Therapy
    # Connectedness
    # Compassion
    if t.startswith("CHAPTER "):
        return True

    alpha = [c for c in t if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in alpha if c.isupper()) / len(alpha)
        if alpha else 0.0
    )

    title_case_like = (t == t.title())
    mostly_upper = uppercase_ratio > 0.60
    sentence_like = t.endswith((".", "!", "?"))

    return not sentence_like and (mostly_upper or title_case_like)


def line_ends_paragraph(line_text: str) -> bool:
    return line_text.rstrip().endswith((".", "!", "?", "”", "\""))


def line_starts_new_paragraph(line_text: str) -> bool:
    """
    Heuristic signal for a fresh paragraph after a boundary.
    """
    t = line_text.strip()
    if not t:
        return False
    if line_is_heading_like(t):
        return True
    return bool(re.match(r"[A-Z“\"(\[]", t))


def build_line_records_from_words(
    page: fitz.Page,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> list[dict]:
    page_rect = page.rect
    page_height = page_rect.height

    top_cutoff = page_height * top_margin_ratio
    bottom_cutoff = page_height * (1.0 - bottom_margin_ratio)

    words = page.get_text("words")
    kept_words = []

    for word in words:
        x0, y0, x1, y1, token, block_no, line_no, word_no = word

        token = str(token).strip()
        if not token:
            continue

        if y1 < top_cutoff:
            continue
        if y0 > bottom_cutoff:
            continue

        kept_words.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "token": token,
                "block_no": int(block_no),
                "line_no": int(line_no),
                "word_no": int(word_no),
            }
        )

    if not kept_words:
        return []

    grouped_lines: dict[tuple[int, int], list[dict]] = {}
    for w in kept_words:
        key = (w["block_no"], w["line_no"])
        grouped_lines.setdefault(key, []).append(w)

    line_records = []
    for (block_no, line_no), line_words in grouped_lines.items():
        line_words.sort(key=lambda w: (w["word_no"], w["x0"]))

        min_y = min(w["y0"] for w in line_words)
        max_y = max(w["y1"] for w in line_words)
        min_x = min(w["x0"] for w in line_words)
        max_x = max(w["x1"] for w in line_words)

        tokens = [w["token"] for w in line_words]
        line_text = clean_line_text(" ".join(tokens))

        if not line_text:
            continue
        if is_probable_footer_or_boilerplate(line_text):
            continue

        line_records.append(
            {
                "block_no": block_no,
                "line_no": line_no,
                "min_y": min_y,
                "max_y": max_y,
                "min_x": min_x,
                "max_x": max_x,
                "height": max_y - min_y,
                "text": line_text,
            }
        )

    # Reading order: top-to-bottom, then left-to-right
    line_records.sort(
        key=lambda r: (round(r["min_y"], 1), round(r["min_x"], 1), r["block_no"], r["line_no"])
    )
    return line_records


def group_lines_into_paragraphs(line_records: list[dict]) -> list[str]:
    """
    Rebuild paragraph-like text from extracted lines.

    Main boundary signals:
    - heading-like line
    - block change
    - unusually large vertical gap
    - line that clearly starts a new paragraph after a sentence end
    """
    if not line_records:
        return []

    heights = [r["height"] for r in line_records if r["height"] > 0]
    median_height = median(heights) if heights else 10.0
    paragraph_gap_threshold = median_height * 1.35

    paragraphs: list[str] = []
    current_lines: list[str] = []
    prev = None

    def flush_current():
        nonlocal current_lines
        if not current_lines:
            return
        para = " ".join(line.strip() for line in current_lines if line.strip())
        para = normalize_extracted_text(para)
        if para and not is_probable_footer_or_boilerplate(para):
            paragraphs.append(para)
        current_lines = []

    for rec in line_records:
        text = rec["text"]

        if is_page_marker_line(text):
            flush_current()
            continue

        if prev is None:
            current_lines.append(text)
            prev = rec
            continue

        block_changed = rec["block_no"] != prev["block_no"]
        vertical_gap = rec["min_y"] - prev["max_y"]
        large_gap = vertical_gap > paragraph_gap_threshold

        prev_ended_para = line_ends_paragraph(prev["text"])
        starts_new_para = line_starts_new_paragraph(text)

        current_is_heading = line_is_heading_like(text)
        prev_is_heading = line_is_heading_like(prev["text"])

        should_break = False

        # Keep headings separate from surrounding prose
        if current_is_heading:
            should_break = True
        elif prev_is_heading:
            should_break = True
        elif block_changed and (large_gap or (prev_ended_para and starts_new_para)):
            should_break = True
        elif large_gap and prev_ended_para and starts_new_para:
            should_break = True

        if should_break:
            flush_current()

        current_lines.append(text)
        prev = rec

    flush_current()
    return paragraphs


def extract_page_words_mode(
    page: fitz.Page,
    top_margin_ratio: float = 0.05,
    bottom_margin_ratio: float = 0.03,
) -> str:
    """
    Improved word-level extraction.

    Strategy:
    1. Extract words with coordinates and PyMuPDF structure metadata.
    2. Remove header/footer-region words.
    3. Rebuild lines from words.
    4. Remove obvious footer/citation lines.
    5. Group lines into paragraph-like units using:
       - heading detection
       - block changes
       - vertical gap size
       - sentence-end + paragraph-start cues
    6. Return page text with blank lines between reconstructed paragraphs.
    """
    line_records = build_line_records_from_words(
        page=page,
        top_margin_ratio=top_margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
    )

    paragraphs = group_lines_into_paragraphs(line_records)
    return "\n\n".join(paragraphs).strip()


# -----------------------------
# PDF-level extraction
# -----------------------------

def extract_pdf_text(
    pdf_path: Path,
    mode: str = "words",
    top_margin_ratio: float = 0.05,
    bottom_margin_ratio: float = 0.03,
    include_page_markers: bool = True,
) -> str:
    text_parts: list[str] = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            if mode == "text":
                page_text = extract_page_text_mode(page)
            elif mode == "blocks":
                page_text = extract_page_blocks_mode(
                    page,
                    top_margin_ratio=top_margin_ratio,
                    bottom_margin_ratio=bottom_margin_ratio,
                )
            elif mode == "words":
                page_text = extract_page_words_mode(
                    page,
                    top_margin_ratio=top_margin_ratio,
                    bottom_margin_ratio=bottom_margin_ratio,
                )
            else:
                raise ValueError(f"Unsupported extraction mode: {mode}")

            if not page_text:
                continue

            if include_page_markers:
                text_parts.append(f"\n\n=== PAGE {page_num} ===\n\n{page_text}")
            else:
                text_parts.append(page_text)

    return normalize_extracted_text("\n\n".join(text_parts))


# -----------------------------
# Output helpers
# -----------------------------

def build_output_path(
    pdf_path: Path,
    books_dir: Path,
    raw_dir: Path,
    mode: str,
) -> Path:
    relative_path = pdf_path.relative_to(books_dir)
    stem = relative_path.stem
    out_name = f"{stem}.{mode}.txt"
    return raw_dir / relative_path.parent / out_name


def build_metadata_header(
    pdf_path: Path,
    books_dir: Path,
    mode: str,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> str:
    relative_path = pdf_path.relative_to(books_dir)
    source_book = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown_book"

    header = [
        "=== SOURCE METADATA ===",
        f"source_book: {source_book}",
        f"source_file: {relative_path.name}",
        f"relative_path: {relative_path.as_posix()}",
        f"extraction_mode: {mode}",
        f"top_margin_ratio: {top_margin_ratio}",
        f"bottom_margin_ratio: {bottom_margin_ratio}",
        "=== BEGIN EXTRACTED TEXT ===",
        "",
    ]
    return "\n".join(header)


def iter_pdf_files(books_dir: Path) -> list[Path]:
    return sorted(
        [p for p in books_dir.rglob("*.pdf") if p.is_file()],
        key=lambda p: p.as_posix().lower(),
    )


def extract_one_file(
    pdf_path: Path,
    books_dir: Path,
    raw_dir: Path,
    mode: str,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    overwrite: bool,
    include_page_markers: bool,
) -> Path:
    out_path = build_output_path(pdf_path, books_dir, raw_dir, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path}")
        return out_path

    extracted_text = extract_pdf_text(
        pdf_path=pdf_path,
        mode=mode,
        top_margin_ratio=top_margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
        include_page_markers=include_page_markers,
    )

    metadata_header = build_metadata_header(
        pdf_path=pdf_path,
        books_dir=books_dir,
        mode=mode,
        top_margin_ratio=top_margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(metadata_header)
        f.write(extracted_text)
        f.write("\n")

    print(f"[OK] {pdf_path.name} -> {out_path.relative_to(raw_dir)}")
    return out_path


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from chapter PDFs.")
    parser.add_argument(
        "--mode",
        choices=["text", "blocks", "words"],
        default="words",
        help="Extraction strategy to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Optional path to a single PDF for testing.",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=BOOKS_DIR,
        help="Books directory.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Raw output directory.",
    )
    parser.add_argument(
        "--top-margin-ratio",
        type=float,
        default=0.05,
        help="Ignore text above this fraction of page height.",
    )
    parser.add_argument(
        "--bottom-margin-ratio",
        type=float,
        default=0.03,
        help="Ignore text below this fraction of page height.",
    )
    parser.add_argument(
        "--no-page-markers",
        action="store_true",
        help="Do not insert === PAGE N === markers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    books_dir = args.books_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()

    if not books_dir.exists():
        print(f"Books directory does not exist: {books_dir}", file=sys.stderr)
        return 1

    if args.single_file is not None:
        pdf_files = [args.single_file.expanduser().resolve()]
    else:
        pdf_files = iter_pdf_files(books_dir)

    if not pdf_files:
        print("No PDF files found.", file=sys.stderr)
        return 1

    for pdf_path in pdf_files:
        extract_one_file(
            pdf_path=pdf_path,
            books_dir=books_dir,
            raw_dir=raw_dir,
            mode=args.mode,
            top_margin_ratio=args.top_margin_ratio,
            bottom_margin_ratio=args.bottom_margin_ratio,
            overwrite=args.overwrite,
            include_page_markers=not args.no_page_markers,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())