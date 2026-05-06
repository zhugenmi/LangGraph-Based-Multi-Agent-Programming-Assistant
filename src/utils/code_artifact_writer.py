"""Code artifact persistence utilities

Save generated code to organized directory structure with API usage statistics.
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Optional


ARTIFACTS_DIR = "outputs/generated_code"


def get_artifact_dir(task_id: str) -> Path:
    """Get directory path for a specific task artifact.

    Directory structure: outputs/generated_code/YYYY-MM-DD/{task_id}/
    """
    today = datetime.now().strftime("%Y-%m-%d")
    base = Path(ARTIFACTS_DIR) / today
    return base / task_id


def save_generated_code(
    task_id: str,
    code: str,
    language: str = "python",
    metadata: Optional[dict] = None
) -> Path:
    """Save generated code to artifact directory.

    Args:
        task_id: Unique task identifier (from session_id)
        code: The generated source code
        language: Programming language (for file extension)
        metadata: Additional info like task_plan, review_result, metrics

    Returns:
        Path to the saved code file
    """
    artifact_dir = get_artifact_dir(task_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save main code file with comment header containing stats
    ext = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "go": ".go",
        "rust": ".rs",
        "c": ".c",
        "cpp": ".cpp",
        "shell": ".sh",
        "bash": ".sh",
    }.get(language.lower(), ".txt")

    code_file = artifact_dir / f"generated_code{ext}"

    # Build comment header with metrics
    header_comment = _build_artifact_header(metadata or {})
    full_content = f"{header_comment}\n{code}" if header_comment else code

    code_file.write_text(full_content, encoding="utf-8")

    # Save metadata JSON if provided
    if metadata:
        meta_file = artifact_dir / "metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return code_file


def _get_comment_prefix(language: str) -> str:
    """Get comment prefix for a given language."""
    prefixes = {
        "python": "#",
        "javascript": "//",
        "typescript": "//",
        "go": "//",
        "rust": "//",
        "java": "//",
        "c": "//",
        "cpp": "//",
        "csharp": "//",
        "php": "//",
        "swift": "//",
        "kotlin": "//",
        "ruby": "#",
        "shell": "#",
        "bash": "#",
        "sql": "--",
        "html": "<!--",
        "css": "/*",
    }
    return prefixes.get(language.lower(), "#")


def _get_comment_end(language: str) -> str:
    """Get comment end sequence for languages that need it (like HTML/CSS)."""
    ends = {
        "html": "-->",
        "css": "*/",
    }
    return ends.get(language.lower(), "")


def _build_artifact_header(metadata: dict) -> str:
    """Build file header comment with task info and API usage summary.

    Format varies by language (uses appropriate comment syntax).
    """
    lines = []
    lang = metadata.get("language", "python").lower()
    comment_prefix = _get_comment_prefix(lang)
    comment_end = _get_comment_end(lang)

    # Task info
    lines.append(f"{comment_prefix} ════════════════════════════════════════════════")
    lines.append(f"{comment_prefix}  Task ID: {metadata.get('task_id', 'unknown')}")
    lines.append(f"{comment_prefix}  Generated: {datetime.now().isoformat()}")

    # API usage summary from metrics
    if "metrics" in metadata:
        m = metadata["metrics"]
        lines.append(f"{comment_prefix}  ──────────────────────────────────────────────")
        lines.append(f"{comment_prefix}  API Usage Statistics:")
        lines.append(f"{comment_prefix}    Total Duration: {m.get('total_duration_s', 0):.2f}s")
        lines.append(f"{comment_prefix}    LLM Calls: {m.get('llm_calls_count', 0)}")
        lines.append(f"{comment_prefix}    Total Tokens: {m.get('total_tokens', 0)}")
        lines.append(f"{comment_prefix}    Iterations: {m.get('iterations', 0)}")
        fix_rounds = m.get('fix_rounds', 0)
        if fix_rounds > 0:
            lines.append(f"{comment_prefix}    Fix Rounds: {fix_rounds}")

    lines.append(f"{comment_prefix} ════════════════════════════════════════════════")
    if comment_end:
        lines.append(comment_end)
    lines.append("")

    return "\n".join(lines)
