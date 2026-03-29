from __future__ import annotations

from typing import Any, Dict, List


def build_context_block(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"[Rank {row.get('rank')}]",
                    f"chunk_id: {row.get('chunk_id')}",
                    f"title: {row.get('source_title')}",
                    f"section: {row.get('section_path_str')}",
                    f"parent_id: {row.get('parent_id')}",
                    "passage:",
                    row.get("text", ""),
                ]
            )
        )
    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(parts) if parts else ""


def build_parent_trace_block(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    parts: List[str] = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"[Parent Rank {row.get('rank')}]",
                    f"chunk_id: {row.get('chunk_id')}",
                    f"title: {row.get('source_title')}",
                    f"section: {row.get('section_path_str')}",
                    "passage:",
                    row.get("text", ""),
                ]
            )
        )
    return "\n\n" + ("\n\n" + ("=" * 80) + "\n\n").join(parts)


def make_prompt_payload(
    query: str,
    condition: str,
    retrieval_payload: Dict[str, Any],
) -> Dict[str, str]:
    if condition == "hierarchical":
        final_rows = retrieval_payload.get("child_results", [])
        parent_rows = retrieval_payload.get("parent_results", [])
        parent_trace = build_parent_trace_block(parent_rows)
        context_block = build_context_block(final_rows)
    else:
        final_rows = retrieval_payload.get("results", [])
        parent_trace = ""
        context_block = build_context_block(final_rows)

    system_prompt = (
        "You are an AI assistant generating an IFS-informed coaching response for a machine learning "
        "retrieval experiment. Your output must be grounded in the provided passages only.\n\n"
        "Requirements:\n"
        "1. Be clearly IFS-informed: use parts language, curiosity, compassion, and non-pathologizing framing.\n"
        "2. Do not diagnose, moralize, or confront the client.\n"
        "3. Do not invent facts not supported by the retrieved context.\n"
        "4. If the retrieved passages do not fully support a claim, stay modest and say so indirectly by keeping the guidance general.\n"
        "5. Focus on what an IFS-informed coach would understand or gently reflect, not on crisis intervention or treatment planning.\n"
        "6. Keep the response concise but meaningful: about 1-3 short paragraphs.\n"
        "7. End with 2-4 short bullet points labeled 'Grounding notes:' where each bullet references a retrieved idea in plain language.\n"
    )

    user_prompt = (
        f"Retrieval condition: {condition}\n\n"
        f"Consultation window:\n{query}\n\n"
        f"{'Parent trace (for hierarchical retrieval):' + parent_trace if parent_trace else ''}\n\n"
        f"Retrieved passages:{context_block}\n\n"
        "Write a grounded IFS-informed coaching response for this consultation window."
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}