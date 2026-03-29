from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.retrieve import search_partition
from preprocessing.hierarchical_retrieve import hierarchical_retrieve


DEFAULT_INDEX_ROOT = "data/test_indexes"
DEFAULT_EMBED_ROOT = "data/test_embeddings"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_OUTPUT_DIR = "data/generation_pilot"
DEFAULT_TOP_K = 3
DEFAULT_PARENT_TOP_K = 3
DEFAULT_CHILD_TOP_K_PER_PARENT = 2
DEFAULT_FINAL_TOP_K = 5

DEFAULT_LLM_PROVIDER = "mock"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview_text(text: str, n: int = 350) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


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


def call_llm_mock(prompt_payload: Dict[str, str], retrieval_payload: Dict[str, Any], condition: str) -> str:
    if condition == "hierarchical":
        rows = retrieval_payload.get("child_results", [])
    else:
        rows = retrieval_payload.get("results", [])

    titles = [row.get("source_title") for row in rows[:2]]
    sections = [row.get("section_path_str") for row in rows[:2]]

    return (
        "[MOCK OUTPUT]\n\n"
        "An IFS-informed coach might approach this by assuming that the conflicting or extreme behaviors "
        "make sense from the perspective of parts that are trying to protect the person from something more vulnerable. "
        "Rather than trying to eliminate the behavior or argue with the part, the coach would stay curious about what the part fears "
        "would happen if it did not do this job, and what pain, shame, or overwhelm it may be trying to prevent.\n\n"
        "The coach would also want to help the person notice any critical, controlling, or reactive parts that are escalating the inner conflict. "
        "In IFS, these parts are usually not treated as the enemy; they are understood as protectors with positive intentions, even when their strategies are costly. "
        "A grounded response would therefore emphasize compassion, differentiation, and learning from the parts rather than confronting them.\n\n"
        "Grounding notes:\n"
        f"- Retrieved passages centered on: {titles[0] if titles else 'N/A'} / {sections[0] if sections else 'N/A'}\n"
        f"- Additional retrieval support came from: {titles[1] if len(titles) > 1 else 'N/A'} / {sections[1] if len(sections) > 1 else 'N/A'}\n"
        "- The retrieved material emphasized protectors' positive intentions and curiosity about their role.\n"
        "- The retrieved material also emphasized reducing inner conflict rather than taking sides against a part.\n"
    )


def call_llm_openai(prompt_payload: Dict[str, str], llm_model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI Python package not found. Install it with: python -m pip install openai"
        ) from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI()
    response = client.responses.create(
        model=llm_model,
        input=[
            {"role": "system", "content": prompt_payload["system_prompt"]},
            {"role": "user", "content": prompt_payload["user_prompt"]},
        ],
    )
    return response.output_text


def generate_text(
    prompt_payload: Dict[str, str],
    retrieval_payload: Dict[str, Any],
    condition: str,
    llm_provider: str,
    llm_model: str,
) -> str:
    if llm_provider == "mock":
        return call_llm_mock(prompt_payload, retrieval_payload, condition)
    if llm_provider == "openai":
        return call_llm_openai(prompt_payload, llm_model)
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")

def load_retrieval_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def run_retrieval_for_condition(
    query: str,
    condition: str,
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    retrieval_model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    if condition == "small_fixed":
        return search_partition(
            query=query,
            partition="small_fixed",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            top_k=top_k,
            model=retrieval_model,
        )
    if condition == "medium_overlap":
        return search_partition(
            query=query,
            partition="medium_overlap",
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            top_k=top_k,
            model=retrieval_model,
        )
    if condition == "hierarchical":
        return hierarchical_retrieve(
            query=query,
            index_root=index_root,
            embed_root=embed_root,
            model_name=retrieval_model_name,
            parent_top_k=parent_top_k,
            child_top_k_per_parent=child_top_k_per_parent,
            final_top_k=final_top_k,
            model=retrieval_model,
        )
    raise ValueError(f"Unsupported condition: {condition}")


def run_one_condition(
    query: str,
    condition: str,
    index_root: Path,
    embed_root: Path,
    retrieval_model_name: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    parent_top_k: int,
    child_top_k_per_parent: int,
    final_top_k: int,
    retrieval_model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    retrieval_payload = run_retrieval_for_condition(
        query=query,
        condition=condition,
        index_root=index_root,
        embed_root=embed_root,
        retrieval_model_name=retrieval_model_name,
        top_k=top_k,
        parent_top_k=parent_top_k,
        child_top_k_per_parent=child_top_k_per_parent,
        final_top_k=final_top_k,
        retrieval_model=retrieval_model,
    )

    prompt_payload = make_prompt_payload(
        query=query,
        condition=condition,
        retrieval_payload=retrieval_payload,
    )

    generation_text = generate_text(
        prompt_payload=prompt_payload,
        retrieval_payload=retrieval_payload,
        condition=condition,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    return {
        "created_at_utc": now_utc_iso(),
        "condition": condition,
        "query": query,
        "retrieval_model_name": retrieval_model_name,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "retrieval": retrieval_payload,
        "prompt": prompt_payload,
        "generated_response": generation_text,
    }


def print_generation_result(payload: Dict[str, Any]) -> None:
    print("\n" + "#" * 100)
    print(f"CONDITION: {payload['condition']}")
    print(f"LLM_PROVIDER: {payload['llm_provider']}")
    print(f"LLM_MODEL: {payload['llm_model']}")
    print("#" * 100)

    print("\nGENERATED RESPONSE\n")
    print(payload["generated_response"])


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_output_path(output_dir: Path, slug: str, condition: str) -> Path:
    return output_dir / f"{slug}__{condition}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generation pilot using validated retrieval conditions."
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--condition",
        type=str,
        choices=["small_fixed", "medium_overlap", "hierarchical", "all"],
        default="all",
    )
    parser.add_argument("--index-root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--embed-root", type=str, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--retrieval-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-provider", type=str, choices=["mock", "openai"], default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--parent-top-k", type=int, default=DEFAULT_PARENT_TOP_K)
    parser.add_argument("--child-top-k-per-parent", type=int, default=DEFAULT_CHILD_TOP_K_PER_PARENT)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slug", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_root = Path(args.index_root).expanduser().resolve()
    embed_root = Path(args.embed_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    conditions = (
        ["small_fixed", "medium_overlap", "hierarchical"]
        if args.condition == "all"
        else [args.condition]
    )

    retrieval_model = load_retrieval_model(args.retrieval_model_name)

    for condition in conditions:
        payload = run_one_condition(
            query=args.query,
            condition=condition,
            index_root=index_root,
            embed_root=embed_root,
            retrieval_model_name=args.retrieval_model_name,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            parent_top_k=args.parent_top_k,
            child_top_k_per_parent=args.child_top_k_per_parent,
            final_top_k=args.final_top_k,
            retrieval_model=retrieval_model,
        )

        print_generation_result(payload)
        out_path = make_output_path(output_dir, args.slug, condition)
        save_json(out_path, payload)
        print(f"\nSaved generation payload to: {out_path}")


if __name__ == "__main__":
    main()