"""Task planning agent for breaking down coding tasks"""

from typing import Dict, Any, Optional
from src.llm.bigmodel_client import get_llm_client


class TaskPlannerAgent:
    """Agent responsible for task planning and decomposition"""

    def __init__(self):
        self.client = get_llm_client()
        self.model = "bigmodel"

    def plan(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze the task and create a detailed execution plan.

        Args:
            task_description: User's coding task description

        Returns:
            Dict containing task plan with main task and sub-tasks
        """
        prompt = f"""你是一个专业的任务规划专家。请分析用户的需求并制定详细的执行计划。

用户需求：{task_description}

请按以下JSON格式输出任务规划（只输出JSON，不要其他内容）：
{{
    "task": "主任务描述（简洁明了）",
    "sub_tasks": [
        {{
            "id": 1,
            "description": "子任务描述",
            "complexity": "low/medium/high",
            "dependencies": []
        }}
    ]
}}

要求：
1. 将复杂任务分解为3-7个可执行的子任务
2. 根据依赖关系合理排序子任务
3. complexity表示任务复杂度：low简单，medium中等，high复杂
4. dependencies数组填写依赖的子任务ID
"""

        try:
            response = self.client.invoke(prompt)

            if hasattr(response, 'content'):
                return self._parse_plan_response(response.content)
            elif hasattr(response, 'text'):
                return self._parse_plan_response(response.text)
            elif isinstance(response, str):
                return self._parse_plan_response(response)
            else:
                return self._parse_plan_response(str(response))

        except Exception as e:
            return self._create_default_plan(task_description)

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured plan"""
        import json
        import re

        response = response.strip()

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                if "task" in plan and "sub_tasks" in plan:
                    return plan
            except json.JSONDecodeError:
                pass

        return self._create_default_plan(response)

    def _create_default_plan(self, task_description: str) -> Dict[str, Any]:
        """Create a default plan if parsing fails"""
        return {
            "task": task_description,
            "sub_tasks": [
                {
                    "id": 1,
                    "description": "理解需求并设计解决方案",
                    "complexity": "medium",
                    "dependencies": []
                },
                {
                    "id": 2,
                    "description": "编写核心代码实现",
                    "complexity": "medium",
                    "dependencies": [1]
                },
                {
                    "id": 3,
                    "description": "添加测试用例验证功能",
                    "complexity": "low",
                    "dependencies": [2]
                }
            ]
        }