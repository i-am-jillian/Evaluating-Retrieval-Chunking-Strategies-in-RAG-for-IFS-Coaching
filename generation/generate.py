from __future__ import annotations

from typing import Any, Dict


def call_llm_mock(
    prompt_payload: Dict[str, str],
    retrieval_payload: Dict[str, Any],
    condition: str,
) -> str:
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


def call_llm_gemini(prompt_payload: Dict[str, str], llm_model: str) -> str:
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "Google GenAI SDK not found. Install it with: python -m pip install -U google-genai"
        ) from e

    client = genai.Client()

    full_prompt = (
        prompt_payload["system_prompt"]
        + "\n\n"
        + prompt_payload["user_prompt"]
    )

    response = client.models.generate_content(
        model=llm_model,
        contents=full_prompt,
    )

    return response.text


def generate_text(
    prompt_payload: Dict[str, str],
    retrieval_payload: Dict[str, Any],
    condition: str,
    llm_provider: str,
    llm_model: str,
) -> str:
    if llm_provider == "mock":
        return call_llm_mock(prompt_payload, retrieval_payload, condition)

    if llm_provider == "gemini":
        return call_llm_gemini(prompt_payload, llm_model)

    raise ValueError(f"Unsupported llm_provider: {llm_provider}")