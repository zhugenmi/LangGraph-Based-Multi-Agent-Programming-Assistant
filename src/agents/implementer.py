"""Implementer agent for both code generation and fixing"""

import re
from typing import Dict, Any, Optional
from src.llm.llm_model_client import get_agent_llm_client
from src.utils.helpers import build_prompt
from src.utils.prompts import (
    IMPLEMENTER_GENERATE_PROMPT,
    IMPLEMENTER_FIX_PROMPT,
)


# Language-specific configurations
LANGUAGE_CONFIG = {
    "python": {"extension": ".py", "comment": "#", "docstring": '"""Docstring"""', "code_block": "python"},
    "javascript": {"extension": ".js", "comment": "//", "docstring": "/** Docstring */", "code_block": "javascript"},
    "typescript": {"extension": ".ts", "comment": "//", "docstring": "/** Docstring */", "code_block": "typescript"},
    "java": {"extension": ".java", "comment": "//", "docstring": "/** Docstring */", "code_block": "java"},
    "c++": {"extension": ".cpp", "comment": "//", "docstring": "/** Docstring */", "code_block": "cpp"},
    "c": {"extension": ".c", "comment": "//", "docstring": "/** Docstring */", "code_block": "c"},
    "c#": {"extension": ".cs", "comment": "//", "docstring": "/// <summary>Docstring</summary>", "code_block": "csharp"},
    "go": {"extension": ".go", "comment": "//", "docstring": "// Docstring", "code_block": "go"},
    "rust": {"extension": ".rs", "comment": "//", "docstring": "/// Docstring", "code_block": "rust"},
    "ruby": {"extension": ".rb", "comment": "#", "docstring": "=begin\nDocstring\n=end", "code_block": "ruby"},
    "php": {"extension": ".php", "comment": "//", "docstring": "/** Docstring */", "code_block": "php"},
    "swift": {"extension": ".swift", "comment": "//", "docstring": "/// Docstring", "code_block": "swift"},
    "kotlin": {"extension": ".kt", "comment": "//", "docstring": "/** Docstring */", "code_block": "kotlin"},
    "shell": {"extension": ".sh", "comment": "#", "docstring": "# Docstring", "code_block": "bash"},
    "sql": {"extension": ".sql", "comment": "--", "docstring": "-- Docstring", "code_block": "sql"},
    "html": {"extension": ".html", "comment": "<!-- -->", "docstring": "<!-- Docstring -->", "code_block": "html"},
    "css": {"extension": ".css", "comment": "/* */", "docstring": "/* Docstring */", "code_block": "css"},
    "vue": {"extension": ".vue", "comment": "<!-- -->", "docstring": "<!-- Docstring -->", "code_block": "html"},
    "react": {"extension": ".jsx", "comment": "//", "docstring": "/** Docstring */", "code_block": "javascript"},
}


def get_language_config(language: str) -> Dict[str, str]:
    """Get configuration for a specific language"""
    return LANGUAGE_CONFIG.get(language.lower(), LANGUAGE_CONFIG["python"])


class Implementer:
    """Agent responsible for both generating and fixing code"""

    def __init__(self):
        self.client = get_agent_llm_client("implementer")
        self.model = "implementer"

    def generate(
        self,
        task_description: str,
        task_plan: Optional[Dict[str, Any]] = None,
        repo_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate code based on task description and plan."""
        language = "python"
        if task_plan and task_plan.get("language"):
            language = task_plan["language"].lower()

        lang_config = get_language_config(language)
        code_block = lang_config["code_block"]

        # Build structured context
        plan_info = None
        if task_plan:
            plan_info = {
                "task": task_plan.get("task", "Unknown"),
                "language": language,
                "sub_tasks": task_plan.get("sub_tasks", []),
            }

        repo_info = None
        if repo_analysis:
            repo_info = {
                "main_files": repo_analysis.get("main_files", [])[:3],
                "key_imports": repo_analysis.get("key_patterns", {}).get("imports", [])[:5],
            }

        prompt = build_prompt(
            IMPLEMENTER_GENERATE_PROMPT,
            task=task_description,
            language=language.upper(),
            plan=plan_info,
            repo_context=repo_info,
            code_block_tag=code_block,
        )

        try:
            response = self.client.invoke(prompt)

            if hasattr(response, 'content') and response.content:
                return response.content
            elif hasattr(response, 'text') and response.text:
                return response.text
            elif isinstance(response, str):
                return response
            elif hasattr(response, 'lc_content') and response.lc_content:
                return response.lc_content
            else:
                content = getattr(response, 'content', None)
                if content:
                    return content
                return str(response)

        except Exception as e:
            raise RuntimeError(f"代码生成失败: {str(e)}")

    def fix(self, original_code: str, review_result: Dict[str, Any]) -> str:
        """Fix code issues based on review feedback."""
        issues = review_result.get("issues", [])

        prompt = build_prompt(
            IMPLEMENTER_FIX_PROMPT,
            original_code=original_code,
            review_issues=issues,
            review_summary=review_result.get("summary", ""),
        )

        try:
            response = self.client.invoke(prompt)

            if hasattr(response, 'content'):
                code_response = response.content
            elif hasattr(response, 'text'):
                code_response = response.text
            elif isinstance(response, str):
                code_response = response
            else:
                code_response = str(response)

            code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', code_response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return code_response.strip() if code_response.strip() else original_code

        except Exception as e:
            return original_code
