"""Code fixing agent"""

from typing import Dict, Any
from src.llm.bigmodel_client import get_llm_client


class CodeFixerAgent:
    """Agent responsible for fixing code issues identified by reviewer"""

    def __init__(self):
        self.client = get_llm_client()
        self.model = "bigmodel"

    def fix(self, original_code: str, review_result: Dict[str, Any]) -> str:
        """
        Fix code issues based on review feedback.

        Args:
            original_code: The original generated code
            review_result: The review result containing issues

        Returns:
            Fixed code as string
        """
        issues_text = ""
        if review_result.get("issues"):
            issues_text = "\n审查发现的问题：\n"
            for i, issue in enumerate(review_result["issues"], 1):
                issues_text += f"{i}. [{issue.get('severity', 'warning')}] {issue.get('description', '')}\n"
                if issue.get('suggestion'):
                    issues_text += f"   建议：{issue['suggestion']}\n"

        prompt = f"""你是一个专业的Python程序员。请根据审查意见修复代码中的问题。

原始代码：
```python
{original_code}
```

{issues_text}

要求：
1. 只输出修复后的完整代码
2. 必须用中文写好代码注释
3. 必须用中文给函数和类写docstring说明
4. 修复所有审查发现的问题
5. 保持代码的功能不变
6. 只需要输出代码，不需要解释

请只输出代码，用```python代码块包裹："""

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

            import re
            code_match = re.search(r'```python\s*(.*?)\s*```', code_response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return code_response.strip() if code_response.strip() else original_code

        except Exception as e:
            return original_code