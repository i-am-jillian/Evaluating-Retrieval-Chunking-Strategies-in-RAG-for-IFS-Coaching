from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "ifs_corpus" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "ifs_corpus" / "clean"


def normalize_newlines(text: str) -> str:
    """
    Standardize newline characters so all later regexes behave consistently.
    Do not collapse paragraph spacing here.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_extraction_metadata(text: str) -> str:
    """
    Remove the metadata header inserted by extract_text.py.

    This strips everything from:
        === SOURCE METADATA ===
    through:
        === BEGIN EXTRACTED TEXT ===

    so the downstream corpus contains only book content.
    """
    return re.sub(
        r"=== SOURCE METADATA ===.*?=== BEGIN EXTRACTED TEXT ===\s*",
        "",
        text,
        flags=re.DOTALL,
    )


def remove_page_markers(text: str) -> str:
    """
    Remove page boundary markers such as:
        === PAGE 3 ===

    Replace them with blank lines rather than spaces so we preserve
    structural separation across page boundaries.
    """
    return re.sub(r"\n*=== PAGE \d+ ===\n*", "\n\n", text)


def remove_proquest_footer_lines(text: str) -> str:
    """
    Remove common ProQuest/footer boilerplate.

    These lines are platform noise, not book content.
    """
    patterns = [
        r".*ProQuest\s+Ebook\s+Central.*",
        r".*Ebook\s+Central,\s+http://ebookcentral\.proquest\.com.*",
        r".*Created\s+from\s+buffalo\s+on.*",
        r".*Copyright\s+©.*",
        r".*Ebook\s+pages\s+\d+.*Printed\s+page\s+\d+.*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text


def remove_repeated_citation_lines(text: str) -> str:
    """
    Remove repeated citation/footer lines that survive extraction.

    This stays conservative: it targets platform/footer context rather than
    trying to remove legitimate in-text citations.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if (
            "ProQuest" in stripped
            or "Ebook Central" in stripped
            or "ebookcentral.proquest.com" in stripped
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_orphan_symbol_lines(text: str) -> str:
    """
    Remove lines that are only symbol noise, while preserving actual content.

    This helps eliminate PDF garbage without touching meaningful prose.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if re.fullmatch(r"[•o\-\–—\.\,\;\:\(\)\"'`]+", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_bullets(text: str) -> str:
    """
    Normalize bullet styles into a consistent dash format.

    Examples:
        • Item  -> - Item
        o Item  ->   - Item
    """
    lines = text.split("\n")
    normalized = []

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith("• "):
            indent = len(line) - len(line.lstrip())
            normalized.append(" " * indent + "- " + stripped[2:])
        elif re.match(r"^o\s+", stripped):
            indent = len(line) - len(line.lstrip())
            normalized.append(" " * indent + "  - " + stripped[2:])
        else:
            normalized.append(line)

    return "\n".join(normalized)


def fix_split_inline_tokens(text: str) -> str:
    """
    Repair only clearly broken inline punctuation/quote patterns.

    Examples:
        (the
        exile
        ) from

        “
        are
        ”

    Important:
    We do NOT join generic word-newline-word sequences anymore,
    because that can destroy real paragraph boundaries.
    """
    # Join line break after opening parenthesis/quote
    text = re.sub(r"([\(\[\{\"“‘])\s*\n\s*", r"\1", text)

    # Join line break before closing parenthesis/quote
    text = re.sub(r"\s*\n\s*([\)\]\}\"”’])", r"\1", text)

    return text


def fix_split_uppercase_headings(text: str) -> str:
    """
    Repair headings that were split into stacked uppercase fragments.

    Example:
        T
        N
        P
        HE
        ATURE OF
        ARTS

    becomes something closer to:
        THE NATURE OF PARTS

    Strategy:
    - detect runs of short heading-like lines
    - merge them into one line when the run looks like fragmented display text
    """
    lines = text.split("\n")
    output = []
    i = 0

    def looks_like_heading_fragment(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if len(s) > 40:
            return False
        if not re.fullmatch(r"[A-Za-z&,\-–—'\"“”‘’:\s]+", s):
            return False

        letters = re.sub(r"[^A-Za-z]", "", s)
        if not letters:
            return False

        return letters.upper() == letters or s.istitle()

    while i < len(lines):
        line = lines[i].strip()

        if not looks_like_heading_fragment(line):
            output.append(lines[i])
            i += 1
            continue

        run = [line]
        j = i + 1
        while j < len(lines) and looks_like_heading_fragment(lines[j].strip()):
            run.append(lines[j].strip())
            j += 1

        if len(run) >= 2 and sum(len(x) for x in run) <= 80:
            merged = " ".join(run)

            # Collapse over-spaced heading fragments like "T HE" -> "THE"
            merged = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", merged)

            # Also collapse repeated isolated single capitals inside all-caps runs
            merged = re.sub(r"\b([A-Z])\s+(?=[A-Z]\b)", r"\1", merged)

            output.append(merged)
            i = j
        else:
            output.append(lines[i])
            i += 1

    return "\n".join(output)


def fix_spacing(text: str) -> str:
    """
    Final spacing cleanup.

    This is intentionally light:
    - collapse repeated spaces/tabs
    - clean punctuation spacing
    - preserve paragraph breaks
    """
    text = re.sub(r"[ \t]+", " ", text)

    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace("“ ", "“")
    text = text.replace(" ”", "”")
    text = text.replace("‘ ", "‘")
    text = text.replace(" ’", "’")

    # Clean up spaces around blank lines without flattening structure
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse excessive blank lines, but keep paragraph separation
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Master cleaning pipeline.

    Grand strategy:
    1. standardize text form
    2. remove metadata and platform noise
    3. lightly repair heading / quote artifacts
    4. normalize bullets
    5. clean spacing
    6. preserve paragraph structure for structure_text.py
    """
    text = normalize_newlines(text)
    text = remove_extraction_metadata(text)
    text = remove_page_markers(text)
    text = remove_proquest_footer_lines(text)
    text = remove_repeated_citation_lines(text)
    text = remove_orphan_symbol_lines(text)
    text = fix_split_inline_tokens(text)
    text = normalize_bullets(text)
    text = fix_spacing(text)
    return text


def clean_file(raw_path: Path, raw_dir: Path, clean_dir: Path, overwrite: bool = False) -> Path:
    relative_path = raw_path.relative_to(raw_dir)
    out_path = clean_dir / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[SKIP] {relative_path}")
        return out_path

    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    out_path.write_text(cleaned, encoding="utf-8")

    print(f"[OK]   {relative_path}")
    return out_path


def iter_txt_files(raw_dir: Path) -> list[Path]:
    return sorted(
        [p for p in raw_dir.rglob("*.txt") if p.is_file()],
        key=lambda p: p.as_posix().lower(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean extracted IFS corpus text files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Path to raw text directory.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=CLEAN_DIR,
        help="Path to cleaned text directory.",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Optional path to a single raw .txt file for iterative testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    clean_dir = args.clean_dir.resolve()

    if args.single_file is not None:
        raw_path = args.single_file.resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Single file not found: {raw_path}")
        clean_file(raw_path, raw_dir, clean_dir, overwrite=args.overwrite)
        return

    txt_files = iter_txt_files(raw_dir)
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {raw_dir}")

    for raw_path in txt_files:
        clean_file(raw_path, raw_dir, clean_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()