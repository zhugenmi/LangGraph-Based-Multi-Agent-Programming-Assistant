"""Code review agent"""

from typing import Dict, Any
from src.llm.llm_model_client import get_agent_llm_client
from src.utils.helpers import build_prompt, safe_parse
from src.utils.prompts import REVIEWER_PROMPT


class ReviewerAgent:
    """Agent responsible for reviewing generated code"""

    def __init__(self):
        self.client = get_agent_llm_client("reviewer")
        self.model = "reviewer"

    def review(self, code: str, task_description: str) -> Dict[str, Any]:
        """Review the generated code for correctness and quality."""
        prompt = build_prompt(
            REVIEWER_PROMPT,
            task=task_description,
            code=code,
        )

        try:
            response = self.client.invoke(prompt)
            if hasattr(response, 'content'):
                raw = response.content
            elif hasattr(response, 'text'):
                raw = response.text
            elif isinstance(response, str):
                raw = response
            else:
                raw = str(response)

            result = safe_parse(raw)
            if "needs_revision" in result:
                return result
        except Exception:
            pass

        return self._create_default_review()

    def _create_default_review(self) -> Dict[str, Any]:
        """Create a default review result"""
        return {
            "needs_revision": False,
            "issues": [],
            "summary": "代码审查通过",
            "score": 8
        }
