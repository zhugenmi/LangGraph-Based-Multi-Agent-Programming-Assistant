"""Repository analysis agent"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import ast
from src.llm.llm_model_client import get_agent_llm_client
from src.skills.skill_registry import skill_registry
from src.utils.helpers import build_prompt, safe_parse
from src.utils.prompts import (
    REPO_ANALYST_CLASSIFY_PROMPT,
    REPO_ANALYST_NEW_PROJECT_PROMPT,
    REPO_ANALYST_MODIFICATION_PROMPT,
)


class TaskType:
    """Task types"""
    MODIFY_EXISTING = "modify_existing"
    CREATE_NEW = "create_new"
    MIXED = "mixed"


class RepoAnalystAgent:
    """Agent responsible for analyzing codebase context and locating relevant files"""

    def __init__(self):
        self.client = get_agent_llm_client("repo_analyst")
        self.model = "repo_analyst"
        self.project_root = Path(__file__).parent.parent.parent
        self.skills = skill_registry.get_skills_for_agent("repo_analyst")

    def use_skill(self, skill_name: str, *args, **kwargs) -> Any:
        """Use a skill available to this agent"""
        skill = skill_registry.get_skill(skill_name)
        if skill:
            return skill.execute(*args, **kwargs)
        else:
            raise ValueError(f"Skill '{skill_name}' not available")

    def classify_task(self, task_description: str) -> Dict[str, Any]:
        """Classify the task type based on the description."""
        prompt = build_prompt(
            REPO_ANALYST_CLASSIFY_PROMPT,
            task_description=task_description,
        )
        try:
            response = self.client.invoke(prompt)
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                content = str(response)

            try:
                result = safe_parse(content)
                if "task_type" in result:
                    return result
            except ValueError:
                pass
        except Exception as e:
            print(f"Task classification error: {e}")

        return {
            "task_type": TaskType.MODIFY_EXISTING,
            "confidence": 0.5,
            "reasoning": "Default classification due to parsing error",
            "existing_files_mentioned": [],
            "new_files_needed": [],
            "suggested_project_type": "python_package"
        }

    def analyze(self, task_description: str) -> Dict[str, Any]:
        """Analyze the codebase to understand context and identify relevant files."""
        task_classification = self.classify_task(task_description)
        task_type = task_classification.get("task_type", TaskType.MODIFY_EXISTING)

        if task_type == TaskType.CREATE_NEW:
            return self._analyze_new_project_task(task_description, task_classification)
        elif task_type == TaskType.MODIFY_EXISTING:
            return self._analyze_modification_task(task_description, task_classification)
        else:
            return self._analyze_mixed_task(task_description, task_classification)

    def _analyze_new_project_task(
        self,
        task_description: str,
        task_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a task that requires creating a new project."""
        project_type = task_classification.get("suggested_project_type", "python_package")

        prompt = build_prompt(
            REPO_ANALYST_NEW_PROJECT_PROMPT,
            task_description=task_description,
            project_type=project_type,
        )
        try:
            response = self.client.invoke(prompt)
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                content = str(response)

            try:
                project_plan = safe_parse(content)
            except ValueError:
                project_plan = None

            if project_plan:
                scaffold_result = self._create_project_scaffold(
                    project_plan.get("project_name", "new_project"),
                    project_plan.get("project_type", "python_package"),
                    task_description
                )
                project_plan["scaffold_result"] = scaffold_result
                project_plan["task_type"] = TaskType.CREATE_NEW
                return project_plan
        except Exception as e:
            print(f"New project analysis error: {e}")

        return {
            "task_type": TaskType.CREATE_NEW,
            "project_name": "new_project",
            "project_type": project_type,
            "description": task_description,
            "core_files": [],
            "dependencies": [],
            "setup_steps": ["Create project directory", "Add source files", "Add tests"],
            "estimated_complexity": "medium",
            "estimated_hours": 4
        }

    def _analyze_modification_task(
        self,
        task_description: str,
        task_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a task that requires modifying existing code."""
        project_structure = self._analyze_project_structure()
        language_info = self._identify_language()
        dependencies = self._analyze_dependencies()
        test_info = self._find_test_information()

        relevant_files = []
        try:
            search_result = self.use_skill("file_search", task_description, ".", "python")
            if search_result.get("success"):
                relevant_files = [r["path"] for r in search_result.get("results", [])[:10]]
        except Exception as e:
            print(f"File search error: {e}")

        prompt = build_prompt(
            REPO_ANALYST_MODIFICATION_PROMPT,
            project_structure=project_structure,
            language=language_info,
            dependencies=dependencies,
            test_info=test_info,
            relevant_files=relevant_files,
            task_description=task_description,
        )

        try:
            response = self.client.invoke(prompt)
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                content = str(response)

            try:
                analysis = safe_parse(content)
                return self._ensure_required_fields(analysis)
            except ValueError:
                pass
        except Exception as e:
            print(f"Modification analysis error: {e}")

        return self._create_default_analysis(project_structure, TaskType.MODIFY_EXISTING)

    def _analyze_mixed_task(
        self,
        task_description: str,
        task_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a task that requires both modification and creation."""
        modification_analysis = self._analyze_modification_task(
            task_description,
            task_classification
        )

        new_files = task_classification.get("new_files_needed", [])
        modification_analysis["task_type"] = TaskType.MIXED
        modification_analysis["new_files_to_create"] = new_files
        modification_analysis["creation_plan"] = [
            {
                "file": f,
                "purpose": "New file for extended functionality",
                "template": "standard"
            }
            for f in new_files
        ]

        return modification_analysis

    def _create_project_scaffold(
        self,
        project_name: str,
        project_type: str,
        task_description: str
    ) -> Dict[str, Any]:
        """Create project scaffold using the scaffold skill."""
        try:
            scaffold_skill = skill_registry.get_skill("project_scaffold")
            if scaffold_skill:
                return scaffold_skill.execute(
                    project_name=project_name,
                    project_type=project_type,
                    base_directory=".",
                    custom_config={"description": task_description}
                )
        except Exception as e:
            print(f"Scaffold creation error: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Scaffold skill not found"}

    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the project structure"""
        structure = {
            "root": str(self.project_root),
            "main_dirs": [],
            "source_dirs": [],
            "test_dirs": [],
            "config_files": [],
            "documentation": []
        }

        for item in self.project_root.iterdir():
            if item.is_dir():
                if item.name in ['src', 'source', 'lib']:
                    structure["source_dirs"].append(str(item))
                elif item.name in ['test', 'tests', 'spec']:
                    structure["test_dirs"].append(str(item))
                else:
                    structure["main_dirs"].append(str(item))
            elif item.is_file():
                if item.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.h']:
                    continue
                elif item.name in ['requirements.txt', 'package.json', 'pom.xml', 'setup.py']:
                    structure["config_files"].append(str(item))
                elif item.suffix in ['.md', '.txt', '.rst', '.doc']:
                    structure["documentation"].append(str(item))

        return structure

    def _identify_language(self) -> Dict[str, str]:
        """Identify programming language and framework"""
        if (self.project_root / "requirements.txt").exists() or \
           (self.project_root / "setup.py").exists() or \
           (self.project_root / "pyproject.toml").exists():
            return {"language": "Python", "framework": self._detect_python_framework()}

        if (self.project_root / "package.json").exists():
            return {"language": "JavaScript/TypeScript", "framework": self._detect_js_framework()}

        if (self.project_root / "pom.xml").exists() or \
           (self.project_root / "build.gradle").exists():
            return {"language": "Java", "framework": self._detect_java_framework()}

        return {"language": "Unknown", "framework": "Unknown"}

    def _detect_python_framework(self) -> str:
        """Detect Python framework"""
        requirements = (self.project_root / "requirements.txt").read_text() if \
                       (self.project_root / "requirements.txt").exists() else ""

        if "django" in requirements.lower():
            return "Django"
        elif "flask" in requirements.lower():
            return "Flask"
        elif "fastapi" in requirements.lower():
            return "FastAPI"
        return "Standard Python"

    def _detect_js_framework(self) -> str:
        """Detect JavaScript framework"""
        package_json = {}
        if (self.project_root / "package.json").exists():
            try:
                package_json = json.loads((self.project_root / "package.json").read_text())
            except:
                pass

        if "react" in str(package_json.get("dependencies", {})).lower():
            return "React"
        elif "vue" in str(package_json.get("dependencies", {})).lower():
            return "Vue"
        elif "angular" in str(package_json.get("dependencies", {})).lower():
            return "Angular"
        return "Vanilla JS"

    def _detect_java_framework(self) -> str:
        """Detect Java framework"""
        pom_xml = ""
        if (self.project_root / "pom.xml").exists():
            pom_xml = (self.project_root / "pom.xml").read_text()

        if "spring-boot" in pom_xml.lower():
            return "Spring Boot"
        elif "spring" in pom_xml.lower():
            return "Spring"
        return "Java SE"

    def _find_relevant_files(self, task_description: str, project_structure: Dict) -> List[str]:
        """Find files relevant to the task"""
        relevant_files = []

        if os.path.exists("src"):
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        relevant_files.append(os.path.join(root, file))

        for file in os.listdir("."):
            if file.endswith(".py") and not file.startswith("__"):
                relevant_files.append(file)

        return relevant_files[:10]

    def _analyze_dependencies(self) -> List[str]:
        """Analyze project dependencies"""
        dependencies = []

        if os.path.exists("requirements.txt"):
            with open("requirements.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dependencies.append(line)

        return dependencies

    def _find_test_information(self) -> Dict[str, Any]:
        """Find test-related information"""
        test_info = {
            "test_framework": "unknown",
            "test_files": [],
            "test_commands": []
        }

        test_dirs = ["test", "tests", "spec"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for root, dirs, files in os.walk(test_dir):
                    for file in files:
                        if file.endswith((".py", "_test.py", "test_*.py")):
                            test_info["test_files"].append(os.path.join(root, file))

        if any("pytest" in dep for dep in self._analyze_dependencies()):
            test_info["test_framework"] = "pytest"
            test_info["test_commands"].append("pytest")
        elif any("unittest" in dep for dep in self._analyze_dependencies()):
            test_info["test_framework"] = "unittest"
            test_info["test_commands"].append("python -m unittest")

        return test_info

    def _ensure_required_fields(self, analysis: Dict) -> Dict[str, Any]:
        """Ensure all required fields exist in analysis"""
        required_fields = {
            "main_files": [],
            "related_files": [],
            "dependencies": [],
            "test_files": [],
            "key_patterns": {"imports": [], "class_patterns": [], "function_patterns": []},
            "interface_contracts": {"inputs": [], "outputs": []},
            "risk_points": [],
            "suggestions": [],
            "complexity_assessment": "medium",
            "estimated_hours": 2
        }

        for field in required_fields:
            if field not in analysis:
                analysis[field] = required_fields[field]
            elif isinstance(required_fields[field], dict) and isinstance(analysis[field], dict):
                for sub_field in required_fields[field]:
                    if sub_field not in analysis[field]:
                        analysis[field][sub_field] = required_fields[field][sub_field]

        return analysis

    def _create_default_analysis(
        self,
        project_structure: Dict,
        task_type: str = TaskType.MODIFY_EXISTING
    ) -> Dict[str, Any]:
        """Create default analysis if parsing fails"""
        return {
            "task_type": task_type,
            "main_files": [],
            "related_files": [],
            "dependencies": [],
            "test_files": [],
            "key_patterns": {"imports": [], "class_patterns": [], "function_patterns": []},
            "interface_contracts": {"inputs": [], "outputs": []},
            "risk_points": [],
            "suggestions": [],
            "complexity_assessment": "medium",
            "estimated_hours": 2
        }
