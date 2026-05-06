"""Supervisor agent for task planning and decomposition"""

import re
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from src.llm.llm_model_client import get_llm_client
from src.utils.helpers import build_prompt, safe_parse
from src.utils.prompts import (
    SUPERVISOR_PROMPT,
    SUPERVISOR_SELECT_LANG_PROMPT,
)


# Supported programming languages (including Chinese keywords)
SUPPORTED_LANGUAGES = {
    "python": ["python", "py", "python3", "python2", "python的", "用python", "python语言", "py语言"],
    "javascript": ["javascript", "js", "ecmascript", "javascript的", "用js", "js语言"],
    "typescript": ["typescript", "ts", "tsx", "typescript的", "用ts"],
    "java": ["java", "java的", "用java", "java语言"],
    "c++": ["c++", "cpp", "c plus plus", "c++11", "c++14", "c++17", "c++的", "用c++", "cpp的", "c语言", "c语言的"],
    "c": ["c语言", "c语言实现", "用c语言", "用c写"],
    "c#": ["c#", "csharp", "c sharp", "c#的", "用c#"],
    "go": ["go", "golang", "go语言", "用go", "go的"],
    "rust": ["rust", "rs", "rust的", "用rust", "rust语言"],
    "ruby": ["ruby", "rb", "ruby的", "用ruby", "ruby语言"],
    "php": ["php", "php的", "用php", "php语言"],
    "swift": ["swift", "swift的", "用swift"],
    "kotlin": ["kotlin", "kt", "kotlin的", "用kotlin"],
    "scala": ["scala", "scala的", "用scala"],
    "r": ["r语言", "r语言的", "用r语言"],
    "matlab": ["matlab", "matlab的", "用matlab"],
    "perl": ["perl", "perl的", "用perl"],
    "shell": ["shell", "bash", "sh", "zsh", "shell脚本", "bash脚本", "shell的", "用shell", "用bash"],
    "sql": ["sql", "mysql", "postgresql", "数据库", "sql的"],
    "html": ["html", "html5", "html的"],
    "css": ["css", "css3", "scss", "sass", "less", "css的"],
    "vue": ["vue", "vue.js", "vue3", "vue的"],
    "react": ["react", "react.js", "reactjs", "jsx", "react的"],
    "angular": ["angular", "angular.js", "angular的"],
    "dart": ["dart", "flutter", "dart的"],
    "lua": ["lua", "lua的", "用lua"],
    "haskell": ["haskell", "hs", "haskell的"],
    "elixir": ["elixir", "ex", "elixir的"],
    "clojure": ["clojure", "clj", "clojure的"],
    "objective-c": ["objective-c", "objectivec", "objc", "objective-c的"],
    "assembly": ["assembly", "asm", "汇编", "汇编语言", "汇编的"],
}


class SupervisorAgent(BaseAgent):
    """Agent responsible for task planning and decomposition"""

    def __init__(self):
        super().__init__("supervisor")
        self.model = "supervisor"

    def detect_language(self, task_description: str) -> Optional[str]:
        """Detect explicitly mentioned programming language in task description"""
        language_patterns = [
            (r'用\s*python', 'python'),
            (r'python\s*(实现|写|编|创建|开发|语言)', 'python'),
            (r'用\s*c\+\+', 'c++'),
            (r'c\+\+\s*(实现|写|编|创建|开发)', 'c++'),
            (r'用\s*c\s*语言', 'c'),
            (r'用\s*c\s*实现', 'c'),
            (r'用\s*java', 'java'),
            (r'java\s*(实现|写|编|创建|开发)', 'java'),
            (r'用\s*go\s*语言', 'go'),
            (r'用\s*go\s*实现', 'go'),
            (r'go\s*(实现|写|编|创建|开发)', 'go'),
            (r'用\s*rust', 'rust'),
            (r'rust\s*(实现|写|编|创建|开发)', 'rust'),
            (r'用\s*javascript', 'javascript'),
            (r'用\s*js\s*实现', 'javascript'),
            (r'javascript\s*(实现|写|编|创建|开发)', 'javascript'),
            (r'用\s*typescript', 'typescript'),
            (r'用\s*ts\s*实现', 'typescript'),
            (r'用\s*vue', 'vue'),
            (r'vue\s*(实现|写|编|创建|开发)', 'vue'),
            (r'用\s*react', 'react'),
            (r'react\s*(实现|写|编|创建|开发)', 'react'),
            (r'用\s*php', 'php'),
            (r'php\s*(实现|写|编|创建|开发)', 'php'),
            (r'用\s*swift', 'swift'),
            (r'swift\s*(实现|写|编|创建|开发)', 'swift'),
            (r'用\s*kotlin', 'kotlin'),
            (r'kotlin\s*(实现|写|编|创建|开发)', 'kotlin'),
            (r'用\s*ruby', 'ruby'),
            (r'ruby\s*(实现|写|编|创建|开发)', 'ruby'),
            (r'用\s*shell', 'shell'),
            (r'用\s*bash', 'shell'),
            (r'shell\s*脚本', 'shell'),
            (r'sql\s*(查询|语句|实现)', 'sql'),
            (r'用\s*c#', 'c#'),
            (r'c#\s*(实现|写|编|创建|开发)', 'c#'),
        ]

        for pattern, lang in language_patterns:
            if re.search(pattern, task_description, re.IGNORECASE):
                return lang

        task_lower = task_description.lower()
        for language, keywords in SUPPORTED_LANGUAGES.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, task_lower):
                    return language

        return None

    def select_language(self, task_description: str) -> str:
        """Select programming language - explicit or LLM-decided"""
        explicit_lang = self.detect_language(task_description)
        if explicit_lang:
            return explicit_lang

        prompt = build_prompt(
            SUPERVISOR_SELECT_LANG_PROMPT,
            task_description=task_description,
        )

        try:
            response = self.client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            content_lower = content.lower().strip()
            for language in SUPPORTED_LANGUAGES:
                if language in content_lower:
                    return language

            return "python"

        except Exception:
            return "python"

    def plan(self, task_description: str) -> Dict[str, Any]:
        """Analyze the task and create a detailed execution plan."""
        language = self.detect_language(task_description)
        if language is None:
            language = self.select_language(task_description)

        skills_info = self.format_skills_context()

        prompt = build_prompt(
            SUPERVISOR_PROMPT,
            skills=skills_info,
            task=task_description,
            language=language,
        )

        try:
            response = self.client.invoke(prompt)
            raw = response.content if hasattr(response, 'content') else str(response)
            return self._parse_plan_response(raw, language)

        except Exception:
            return self._create_default_plan(task_description, language)

    def _parse_plan_response(self, response: str, default_language: str = "python") -> Dict[str, Any]:
        """Parse the LLM response into structured plan"""
        try:
            plan = safe_parse(response)
            if "task" in plan and "sub_tasks" in plan:
                if "language" not in plan or not plan["language"]:
                    plan["language"] = default_language
                return plan
        except ValueError:
            pass

        return self._create_default_plan(response, default_language)

    def _create_default_plan(self, task_description: str, language: str = "python") -> Dict[str, Any]:
        """Create a default plan if parsing fails"""
        return {
            "task": task_description,
            "language": language,
            "sub_tasks": [
                {
                    "id": 1,
                    "description": "理解需求并设计解决方案",
                    "complexity": "medium",
                    "dependencies": []
                },
                {
                    "id": 2,
                    "description": f"使用{language}编写核心代码实现",
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
