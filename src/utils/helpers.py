"""Shared helpers for prompt building and JSON parsing"""

import json
import re
from typing import Any


def build_prompt(agent_prompt: str, **context: Any) -> str:
    """Build a complete prompt combining the global system prompt with agent-specific instructions.

    Usage:
        from src.utils.prompts import GLOBAL_SYSTEM_PROMPT, IMPLEMENTER_PROMPT
        from src.utils.helpers import build_prompt

        prompt = build_prompt(IMPLEMENTER_PROMPT, task="...", plan={...})

    Args:
        agent_prompt: The agent-specific instruction text.
        **context: Key-value pairs to include as structured input (serialized as JSON).

    Returns:
        A complete prompt string ready for the LLM.
    """
    from src.utils.prompts import GLOBAL_SYSTEM_PROMPT

    input_section = ""
    if context:
        input_section = "Input (structured):\n" + json.dumps(context, ensure_ascii=False, indent=2) + "\n"

    return f"""{GLOBAL_SYSTEM_PROMPT}

{agent_prompt}

{input_section}Remember: output valid JSON only, no markdown, no extra text."""


def safe_parse(text: str) -> Any:
    """Parse LLM response as JSON with automatic repair.

    Handles common LLM mistakes:
    - Wrapped in ```json ... ``` code blocks
    - Wrapped in ``` ... ``` code blocks
    - Trailing commas
    - Extra text before/after JSON

    Args:
        text: Raw LLM response string.

    Returns:
        Parsed Python object.

    Raises:
        ValueError: If JSON cannot be recovered after repair attempts.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    cleaned = text.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks: ```json ... ``` or ``` ... ```
    block_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', cleaned, re.DOTALL)
    if block_match:
        cleaned = block_match.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Extract JSON object/array from surrounding text
    obj_match = re.search(r'(\{[\s\S]*\})', cleaned)
    arr_match = re.search(r'(\[[\s\S]*\])', cleaned)

    for match in [obj_match, arr_match]:
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try fixing trailing commas
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Last resort: try the cleaned string after all stripping
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON after all repair attempts: {e}\nOriginal: {text[:500]}")
