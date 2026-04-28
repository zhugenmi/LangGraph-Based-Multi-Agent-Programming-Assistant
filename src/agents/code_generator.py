"""Code generation agent"""

from typing import Dict, Any, Optional
from src.llm.bigmodel_client import get_llm_client


class CodeGeneratorAgent:
    """Agent responsible for generating code based on task plan"""

    def __init__(self):
        self.client = get_llm_client()
        self.model = "bigmodel"

    def generate(self, task_description: str, task_plan: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate code based on task description and plan.

        Args:
            task_description: The coding task description
            task_plan: Optional task plan with sub-tasks

        Returns:
            Generated code as string
        """
        plan_info = ""
        if task_plan:
            plan_info = f"""
任务规划：
主任务：{task_plan.get('task', '未知')}

子任务分解：
"""
            for sub in task_plan.get('sub_tasks', []):
                plan_info += f"- {sub.get('description', '')}\n"

        prompt = f"""你是一个专业的Python程序员。请根据用户需求生成高质量的Python代码。

{plan_info}

用户需求：{task_description}

要求：
1. 代码必须是完整可运行的Python代码
2. 必须用中文写好代码注释
3. 必须用中文给函数和类写docstring说明
4. 代码要遵循PEP8规范
5. 要有适当的错误处理
6. 只需要生成代码，不需要解释

请只输出代码，用```python代码块包裹，不要输出任何其他内容："""

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