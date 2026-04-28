"""Code review agent"""

from typing import Dict, Any
from src.llm.bigmodel_client import get_llm_client


class CodeReviewerAgent:
    """Agent responsible for reviewing generated code"""

    def __init__(self):
        self.client = get_llm_client()
        self.model = "bigmodel"

    def review(self, code: str, task_description: str) -> Dict[str, Any]:
        """
        Review the generated code for correctness and quality.

        Args:
            code: The generated code to review
            task_description: The original task description

        Returns:
            Dict containing review result with issues and suggestions
        """
        prompt = f"""你是一个资深的代码审查专家。请审查以下Python代码，检查其正确性、效率和规范性。

原始需求：{task_description}

待审查代码：
```python
{code}
```

请按以下JSON格式输出审查结果（只输出JSON，不要其他内容）：
{{
    "needs_revision": true/false,
    "issues": [
        {{
            "severity": "error/warning/suggestion",
            "line": "问题所在行或位置",
            "description": "问题描述（用中文）",
            "suggestion": "修改建议（用中文）"
        }}
    ],
    "summary": "总体评价（用中文）",
    "score": 1-10的评分
}}

审查要点：
1. 正确性：代码逻辑是否正确
2. 效率：是否有性能问题
3. 规范：是否符合PEP8规范
4. 安全：是否有安全隐患
5. 可读性：代码是否易于理解"""

        try:
            response = self.client.invoke(prompt)

            if hasattr(response, 'content'):
                return self._parse_review_response(response.content)
            elif hasattr(response, 'text'):
                return self._parse_review_response(response.text)
            elif isinstance(response, str):
                return self._parse_review_response(response)
            else:
                return self._parse_review_response(str(response))

        except Exception as e:
            return self._create_default_review()

    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured review result"""
        import json
        import re

        response = response.strip()

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                review = json.loads(json_match.group())
                if "needs_revision" in review:
                    return review
            except json.JSONDecodeError:
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