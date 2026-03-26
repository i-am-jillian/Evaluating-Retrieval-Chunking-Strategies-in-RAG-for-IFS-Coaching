from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# chunk_text.py
# ============================================================

SMALL_FIXED_CONFIG = {
    "setting_name": "small_fixed",
    "chunking_strategy": "fixed_window_no_overlap",
    "target_tokens": 350,
    "overlap_tokens": 0,
    "min_tokens": 120,
    "max_tokens": 450,
}

MEDIUM_OVERLAP_CONFIG = {
    "setting_name": "medium_overlap",
    "chunking_strategy": "fixed_window_with_overlap",
    "target_tokens": 900,
    "overlap_tokens": 180,
    "min_tokens": 300,
    "max_tokens": 1100,
}

HIERARCHICAL_CONFIG = {
    "setting_name": "hierarchical",
    "parent_strategy": "node_as_parent",
    "child_strategy": "fixed_window_within_node",
    "parent_target_tokens": 1200,
    "child_target_tokens": 300,
    "child_overlap_tokens": 0,
    "child_min_tokens": 120,
    "child_max_tokens": 420,
}

OVERSIZED_PARAGRAPH_SPLIT_THRESHOLD_TOKENS = 420
OVERSIZED_SEGMENT_TARGET_TOKENS = 180
OVERSIZED_SEGMENT_MIN_TOKENS = 80
TOKEN_MULTIPLIER = 1.3
CHUNK_JOINER = "\n\n"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def estimate_tokens(text: str) -> int:
    return max(1, round(word_count(text) * TOKEN_MULTIPLIER))


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def safe_slug(text: Optional[str]) -> str:
    if not text:
        return "unknown"
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def infer_chapter_slug(section_path: List[str]) -> str:
    for item in section_path:
        if re.fullmatch(r"ch\d+[a-zA-Z]*", item.lower()):
            return safe_slug(item)
    return safe_slug(section_path[0] if section_path else "root")


def collect_labels(section_path: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    chapter_label = None
    section_label = None
    subsection_label = None

    if len(section_path) >= 1:
        chapter_label = section_path[0]
    if len(section_path) >= 2:
        section_label = section_path[1]
    if len(section_path) >= 3:
        subsection_label = section_path[2]

    return chapter_label, section_label, subsection_label


def unique_in_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else item
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


@dataclass
class UnitRecord:
    unit_id: str
    parent_paragraph_id: int
    source_doc_id: str
    source_title: str

    node_id: str
    node_type: str
    section_path: List[str]
    section_path_str: str
    chapter_label: Optional[str]
    section_label: Optional[str]
    part_label: Optional[str]

    source_file: str
    source_file_index: int

    char_start_in_doc: int
    char_end_in_doc: int

    paragraph_type: str

    text: str
    word_count: int
    token_count_est: int

    split_index_within_paragraph: int = 0
    split_count_within_paragraph: int = 1


@dataclass
class ChunkRecord:
    chunk_id: str
    retrieval_setting: str
    chunking_strategy: str

    source_doc_id: str
    source_title: str

    text: str

    book_order: int

    node_ids: List[str]
    primary_node_id: str

    source_files: List[str]
    source_file_indices: List[int]

    paragraph_ids: List[int]

    section_path: List[str]
    section_path_str: str
    chapter_label: Optional[str]
    section_label: Optional[str]
    subsection_label: Optional[str]

    char_start_in_doc: int
    char_end_in_doc: int

    token_count_est: int
    char_count: int

    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]

    parent_id: Optional[str]
    children_ids: List[str]

    overlap_with_prev_tokens: int
    overlap_with_next_tokens: int

    node_type: str
    content_type: str

    target_tokens: int
    overlap_tokens: int


def build_units_from_paragraphs(doc: Dict[str, Any]) -> List[UnitRecord]:
    source_doc_id = doc["doc_id"]
    source_title = doc["doc_title"]

    units: List[UnitRecord] = []

    for para in doc["paragraphs"]:
        text = para["text"].strip()
        if not text:
            continue

        token_est = estimate_tokens(text)

        if token_est <= OVERSIZED_PARAGRAPH_SPLIT_THRESHOLD_TOKENS:
            units.append(
                UnitRecord(
                    unit_id=f"{source_doc_id}::p{para['paragraph_id']:05d}::u00",
                    parent_paragraph_id=para["paragraph_id"],
                    source_doc_id=source_doc_id,
                    source_title=source_title,
                    node_id=para["node_id"],
                    node_type=para["node_type"],
                    section_path=para["section_path"],
                    section_path_str=para["section_path_str"],
                    chapter_label=para.get("chapter_label"),
                    section_label=para.get("section_label"),
                    part_label=para.get("part_label"),
                    source_file=para["source_file"],
                    source_file_index=para["source_file_index"],
                    char_start_in_doc=para["char_start_in_doc"],
                    char_end_in_doc=para["char_end_in_doc"],
                    paragraph_type=para.get("paragraph_type", "body"),
                    text=text,
                    word_count=word_count(text),
                    token_count_est=token_est,
                )
            )
            continue

        sentences = sentence_split(text)
        if len(sentences) <= 1:
            units.append(
                UnitRecord(
                    unit_id=f"{source_doc_id}::p{para['paragraph_id']:05d}::u00",
                    parent_paragraph_id=para["paragraph_id"],
                    source_doc_id=source_doc_id,
                    source_title=source_title,
                    node_id=para["node_id"],
                    node_type=para["node_type"],
                    section_path=para["section_path"],
                    section_path_str=para["section_path_str"],
                    chapter_label=para.get("chapter_label"),
                    section_label=para.get("section_label"),
                    part_label=para.get("part_label"),
                    source_file=para["source_file"],
                    source_file_index=para["source_file_index"],
                    char_start_in_doc=para["char_start_in_doc"],
                    char_end_in_doc=para["char_end_in_doc"],
                    paragraph_type=para.get("paragraph_type", "body"),
                    text=text,
                    word_count=word_count(text),
                    token_count_est=token_est,
                )
            )
            continue

        segments: List[str] = []
        current: List[str] = []

        for sent in sentences:
            candidate = " ".join(current + [sent]).strip()
            if current and estimate_tokens(candidate) > OVERSIZED_SEGMENT_TARGET_TOKENS:
                segments.append(" ".join(current).strip())
                current = [sent]
            else:
                current.append(sent)

        if current:
            segments.append(" ".join(current).strip())

        merged: List[str] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if merged and estimate_tokens(seg) < OVERSIZED_SEGMENT_MIN_TOKENS:
                merged[-1] = f"{merged[-1]} {seg}".strip()
            else:
                merged.append(seg)

        total_splits = len(merged)
        para_text = para["text"]
        para_global_start = para["char_start_in_doc"]

        running_search_offset = 0
        for split_idx, seg in enumerate(merged):
            seg_start_local = para_text.find(seg, running_search_offset)
            if seg_start_local == -1:
                seg_start_local = running_search_offset
            seg_end_local = seg_start_local + len(seg)
            running_search_offset = seg_end_local

            units.append(
                UnitRecord(
                    unit_id=f"{source_doc_id}::p{para['paragraph_id']:05d}::u{split_idx:02d}",
                    parent_paragraph_id=para["paragraph_id"],
                    source_doc_id=source_doc_id,
                    source_title=source_title,
                    node_id=para["node_id"],
                    node_type=para["node_type"],
                    section_path=para["section_path"],
                    section_path_str=para["section_path_str"],
                    chapter_label=para.get("chapter_label"),
                    section_label=para.get("section_label"),
                    part_label=para.get("part_label"),
                    source_file=para["source_file"],
                    source_file_index=para["source_file_index"],
                    char_start_in_doc=para_global_start + seg_start_local,
                    char_end_in_doc=para_global_start + seg_end_local,
                    paragraph_type=para.get("paragraph_type", "body"),
                    text=seg,
                    word_count=word_count(seg),
                    token_count_est=estimate_tokens(seg),
                    split_index_within_paragraph=split_idx,
                    split_count_within_paragraph=total_splits,
                )
            )

    units.sort(key=lambda u: (u.char_start_in_doc, u.char_end_in_doc))
    return units


def make_chunk_record(
    chunk_index: int,
    chunk_units: List[UnitRecord],
    source_doc_id: str,
    source_title: str,
    book_order: int,
    retrieval_setting: str,
    chunking_strategy: str,
    target_tokens: int,
    overlap_tokens: int,
    parent_id: Optional[str],
    node_type: str,
) -> ChunkRecord:
    text = CHUNK_JOINER.join(u.text for u in chunk_units).strip()

    node_ids = unique_in_order(u.node_id for u in chunk_units)
    source_files = unique_in_order(u.source_file for u in chunk_units)
    source_file_indices = unique_in_order(u.source_file_index for u in chunk_units)
    paragraph_ids = unique_in_order(u.parent_paragraph_id for u in chunk_units)

    primary_section_path = chunk_units[0].section_path
    primary_section_path_str = chunk_units[0].section_path_str
    chapter_label, section_label, subsection_label = collect_labels(primary_section_path)

    chapter_slug = infer_chapter_slug(primary_section_path)
    chunk_id = f"{source_doc_id}::{retrieval_setting}::{chapter_slug}::{chunk_index:05d}"

    return ChunkRecord(
        chunk_id=chunk_id,
        retrieval_setting=retrieval_setting,
        chunking_strategy=chunking_strategy,
        source_doc_id=source_doc_id,
        source_title=source_title,
        text=text,
        book_order=book_order,
        node_ids=node_ids,
        primary_node_id=node_ids[0],
        source_files=source_files,
        source_file_indices=source_file_indices,
        paragraph_ids=paragraph_ids,
        section_path=primary_section_path,
        section_path_str=primary_section_path_str,
        chapter_label=chapter_label,
        section_label=section_label,
        subsection_label=subsection_label,
        char_start_in_doc=min(u.char_start_in_doc for u in chunk_units),
        char_end_in_doc=max(u.char_end_in_doc for u in chunk_units),
        token_count_est=estimate_tokens(text),
        char_count=len(text),
        prev_chunk_id=None,
        next_chunk_id=None,
        parent_id=parent_id,
        children_ids=[],
        overlap_with_prev_tokens=0,
        overlap_with_next_tokens=0,
        node_type=node_type,
        content_type="body_text",
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def estimate_overlap_tokens_from_units(
    left_units: List[UnitRecord],
    right_units: List[UnitRecord],
) -> int:
    left_ids = {u.unit_id for u in left_units}
    overlap_units = [u for u in right_units if u.unit_id in left_ids]
    return sum(u.token_count_est for u in overlap_units)


def build_flat_chunks_for_units(
    units: List[UnitRecord],
    source_doc_id: str,
    source_title: str,
    config: Dict[str, Any],
    book_order: int,
) -> List[ChunkRecord]:
    if not units:
        return []

    target_tokens = config["target_tokens"]
    overlap_tokens = config["overlap_tokens"]
    min_tokens = config["min_tokens"]
    setting_name = config["setting_name"]
    strategy = config["chunking_strategy"]

    chunks: List[ChunkRecord] = []
    chunk_unit_lists: List[List[UnitRecord]] = []

    start_idx = 0
    chunk_idx = 0

    while start_idx < len(units):
        current_units: List[UnitRecord] = []
        current_tokens = 0
        end_idx = start_idx

        while end_idx < len(units):
            u = units[end_idx]
            if current_units and current_tokens + u.token_count_est > target_tokens:
                break
            current_units.append(u)
            current_tokens += u.token_count_est
            end_idx += 1

        if not current_units:
            current_units = [units[start_idx]]
            current_tokens = current_units[0].token_count_est
            end_idx = start_idx + 1

        while current_tokens < min_tokens and end_idx < len(units):
            u = units[end_idx]
            current_units.append(u)
            current_tokens += u.token_count_est
            end_idx += 1

        chunk = make_chunk_record(
            chunk_index=chunk_idx,
            chunk_units=current_units,
            source_doc_id=source_doc_id,
            source_title=source_title,
            book_order=book_order,
            retrieval_setting=setting_name,
            chunking_strategy=strategy,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            parent_id=None,
            node_type="flat_chunk",
        )
        chunks.append(chunk)
        chunk_unit_lists.append(current_units)
        chunk_idx += 1

        if end_idx >= len(units):
            break

        if overlap_tokens <= 0:
            start_idx = end_idx
            continue

        back_tokens = 0
        new_start_idx = end_idx

        while new_start_idx > start_idx:
            candidate_unit = units[new_start_idx - 1]
            if back_tokens + candidate_unit.token_count_est > overlap_tokens and new_start_idx < end_idx:
                break
            new_start_idx -= 1
            back_tokens += candidate_unit.token_count_est

        if new_start_idx == start_idx:
            new_start_idx = max(start_idx + 1, end_idx - 1)

        start_idx = new_start_idx

    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None

        if overlap_tokens > 0:
            if i > 0:
                chunk.overlap_with_prev_tokens = estimate_overlap_tokens_from_units(
                    chunk_unit_lists[i - 1], chunk_unit_lists[i]
                )
            if i < len(chunks) - 1:
                chunk.overlap_with_next_tokens = estimate_overlap_tokens_from_units(
                    chunk_unit_lists[i], chunk_unit_lists[i + 1]
                )
        else:
            chunk.overlap_with_prev_tokens = 0
            chunk.overlap_with_next_tokens = 0

    return chunks


def relink_chunk_sequence(chunks: List[ChunkRecord], overlap_tokens: int = 0) -> None:
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
        if overlap_tokens <= 0:
            chunk.overlap_with_prev_tokens = 0
            chunk.overlap_with_next_tokens = 0


def build_hierarchical_chunks_for_doc(
    doc: Dict[str, Any],
    units: List[UnitRecord],
    book_order: int,
) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
    source_doc_id = doc["doc_id"]
    source_title = doc["doc_title"]

    units_by_node: Dict[str, List[UnitRecord]] = {}
    for u in units:
        units_by_node.setdefault(u.node_id, []).append(u)

    parents: List[ChunkRecord] = []
    children: List[ChunkRecord] = []

    parent_idx = 0
    child_global_idx = 0

    for node in doc["nodes"]:
        node_id = node["node_id"]
        node_units = units_by_node.get(node_id, [])
        if not node_units:
            continue

        parent_text = CHUNK_JOINER.join(u.text for u in node_units).strip()
        chapter_label, section_label, subsection_label = collect_labels(node["section_path"])
        chapter_slug = infer_chapter_slug(node["section_path"])

        parent_id = f"{source_doc_id}::hier_parent::{chapter_slug}::{parent_idx:05d}"
        parent_idx += 1

        parent_record = ChunkRecord(
            chunk_id=parent_id,
            retrieval_setting="hierarchical_parent",
            chunking_strategy=HIERARCHICAL_CONFIG["parent_strategy"],
            source_doc_id=source_doc_id,
            source_title=source_title,
            text=parent_text,
            book_order=book_order,
            node_ids=[node_id],
            primary_node_id=node_id,
            source_files=unique_in_order(u.source_file for u in node_units),
            source_file_indices=unique_in_order(u.source_file_index for u in node_units),
            paragraph_ids=unique_in_order(u.parent_paragraph_id for u in node_units),
            section_path=node["section_path"],
            section_path_str=node["section_path_str"],
            chapter_label=chapter_label,
            section_label=section_label,
            subsection_label=subsection_label,
            char_start_in_doc=min(u.char_start_in_doc for u in node_units),
            char_end_in_doc=max(u.char_end_in_doc for u in node_units),
            token_count_est=estimate_tokens(parent_text),
            char_count=len(parent_text),
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_id=None,
            children_ids=[],
            overlap_with_prev_tokens=0,
            overlap_with_next_tokens=0,
            node_type="parent",
            content_type="body_text",
            target_tokens=HIERARCHICAL_CONFIG["parent_target_tokens"],
            overlap_tokens=0,
        )
        parents.append(parent_record)

        child_chunks_for_node = build_flat_chunks_for_units(
            units=node_units,
            source_doc_id=source_doc_id,
            source_title=source_title,
            config={
                "setting_name": "hierarchical_child",
                "chunking_strategy": HIERARCHICAL_CONFIG["child_strategy"],
                "target_tokens": HIERARCHICAL_CONFIG["child_target_tokens"],
                "overlap_tokens": HIERARCHICAL_CONFIG["child_overlap_tokens"],
                "min_tokens": HIERARCHICAL_CONFIG["child_min_tokens"],
                "max_tokens": HIERARCHICAL_CONFIG["child_max_tokens"],
            },
            book_order=book_order,
        )

        # Rename child IDs first
        for child in child_chunks_for_node:
            child.chunk_id = f"{source_doc_id}::hier_child::{chapter_slug}::{child_global_idx:05d}"
            child.parent_id = parent_id
            child.node_type = "child"
            child_global_idx += 1

        # Then relink neighbors using the final IDs
        relink_chunk_sequence(
            child_chunks_for_node,
            overlap_tokens=HIERARCHICAL_CONFIG["child_overlap_tokens"],
        )

        for child in child_chunks_for_node:
            children.append(child)
            parent_record.children_ids.append(child.chunk_id)

    relink_chunk_sequence(parents, overlap_tokens=0)
    return parents, children


def summarize_chunks(records: List[ChunkRecord]) -> Dict[str, Any]:
    if not records:
        return {"count": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0}

    token_counts = [r.token_count_est for r in records]
    return {
        "count": len(records),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 2),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }


def validate_chunk_ids(records: List[ChunkRecord], label: str) -> None:
    ids = [r.chunk_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate chunk IDs found in {label}")


def validate_parent_child_links(parents: List[ChunkRecord], children: List[ChunkRecord]) -> None:
    parent_ids = {p.chunk_id for p in parents}
    for c in children:
        if c.parent_id not in parent_ids:
            raise ValueError(f"Child {c.chunk_id} has missing parent {c.parent_id}")


def process_structured_doc(path: Path, book_order: int, output_root: Path) -> Dict[str, Any]:
    doc = read_json(path)
    source_doc_id = doc["doc_id"]
    source_title = doc["doc_title"]

    units = build_units_from_paragraphs(doc)

    small_chunks = build_flat_chunks_for_units(
        units=units,
        source_doc_id=source_doc_id,
        source_title=source_title,
        config=SMALL_FIXED_CONFIG,
        book_order=book_order,
    )
    medium_chunks = build_flat_chunks_for_units(
        units=units,
        source_doc_id=source_doc_id,
        source_title=source_title,
        config=MEDIUM_OVERLAP_CONFIG,
        book_order=book_order,
    )
    hier_parents, hier_children = build_hierarchical_chunks_for_doc(
        doc=doc,
        units=units,
        book_order=book_order,
    )

    validate_chunk_ids(small_chunks, f"{source_doc_id} small_fixed")
    validate_chunk_ids(medium_chunks, f"{source_doc_id} medium_overlap")
    validate_chunk_ids(hier_parents, f"{source_doc_id} hier_parents")
    validate_chunk_ids(hier_children, f"{source_doc_id} hier_children")
    validate_parent_child_links(hier_parents, hier_children)

    small_path = output_root / "small_fixed" / f"{source_doc_id}_chunks.jsonl"
    medium_path = output_root / "medium_overlap" / f"{source_doc_id}_chunks.jsonl"
    parent_path = output_root / "hierarchical" / "parents" / f"{source_doc_id}_parents.jsonl"
    child_path = output_root / "hierarchical" / "children" / f"{source_doc_id}_children.jsonl"

    write_jsonl(small_path, (asdict(r) for r in small_chunks))
    write_jsonl(medium_path, (asdict(r) for r in medium_chunks))
    write_jsonl(parent_path, (asdict(r) for r in hier_parents))
    write_jsonl(child_path, (asdict(r) for r in hier_children))

    return {
        "doc_id": source_doc_id,
        "doc_title": source_title,
        "structured_input": str(path),
        "unit_count": len(units),
        "small_fixed": summarize_chunks(small_chunks),
        "medium_overlap": summarize_chunks(medium_chunks),
        "hierarchical_parents": summarize_chunks(hier_parents),
        "hierarchical_children": summarize_chunks(hier_children),
        "outputs": {
            "small_fixed": str(small_path),
            "medium_overlap": str(medium_path),
            "hierarchical_parents": str(parent_path),
            "hierarchical_children": str(child_path),
        },
    }


def find_structured_docs(input_root: Path) -> List[Path]:
    return sorted(input_root.glob("*.json"), key=lambda p: p.name.lower())


def process_all_docs(input_root: Path, output_root: Path) -> Dict[str, Any]:
    structured_paths = find_structured_docs(input_root)
    if not structured_paths:
        raise FileNotFoundError(f"No structured JSON files found in {input_root}")

    manifest: Dict[str, Any] = {
        "corpus_name": "ifs_corpus",
        "version": "v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "configs": {
            "small_fixed": SMALL_FIXED_CONFIG,
            "medium_overlap": MEDIUM_OVERLAP_CONFIG,
            "hierarchical": HIERARCHICAL_CONFIG,
            "oversized_paragraph_split_threshold_tokens": OVERSIZED_PARAGRAPH_SPLIT_THRESHOLD_TOKENS,
            "oversized_segment_target_tokens": OVERSIZED_SEGMENT_TARGET_TOKENS,
            "oversized_segment_min_tokens": OVERSIZED_SEGMENT_MIN_TOKENS,
            "token_multiplier": TOKEN_MULTIPLIER,
        },
        "documents": [],
    }

    total_small = 0
    total_medium = 0
    total_hier_parents = 0
    total_hier_children = 0

    for book_order, path in enumerate(structured_paths):
        doc_summary = process_structured_doc(path, book_order, output_root)
        manifest["documents"].append(doc_summary)

        total_small += doc_summary["small_fixed"]["count"]
        total_medium += doc_summary["medium_overlap"]["count"]
        total_hier_parents += doc_summary["hierarchical_parents"]["count"]
        total_hier_children += doc_summary["hierarchical_children"]["count"]

        print(f"Wrote chunk outputs for {doc_summary['doc_id']}")
        print(f"  small_fixed: {doc_summary['small_fixed']['count']}")
        print(f"  medium_overlap: {doc_summary['medium_overlap']['count']}")
        print(f"  hier_parents: {doc_summary['hierarchical_parents']['count']}")
        print(f"  hier_children: {doc_summary['hierarchical_children']['count']}")
        print()

    manifest["totals"] = {
        "small_fixed_total_chunks": total_small,
        "medium_overlap_total_chunks": total_medium,
        "hierarchical_parent_total": total_hier_parents,
        "hierarchical_child_total": total_hier_children,
    }

    manifest_path = output_root / "manifests" / "chunk_manifest.json"
    write_json(manifest_path, manifest)

    print(f"Wrote manifest: {manifest_path}")
    print(json.dumps(manifest["totals"], indent=2))

    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build chunk outputs for the IFS corpus structured JSON files."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/ifs_corpus/structured",
        help="Directory containing structured JSON files",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/ifs_corpus/chunked",
        help="Directory to write chunk outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    process_all_docs(input_root, output_root)


if __name__ == "__main__":
    main()