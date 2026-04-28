"""LangGraph workflow for multi-agent code generation"""

import re
from typing import TypedDict, Annotated, Sequence, Optional, Callable, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agents.task_planner import TaskPlannerAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.code_reviewer import CodeReviewerAgent
from src.agents.code_fixer import CodeFixerAgent
from src.utils.logger import AgentLogger


class WorkflowState(TypedDict):
    """State schema for the workflow"""
    task_description: str
    session_id: str
    task_plan: Optional[dict]
    generated_code: Optional[str]
    review_result: Optional[dict]
    fixed_code: Optional[str]
    workflow_steps: list
    error: Optional[str]
    progress_callback: Optional[Any]


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response, removing explanations"""
    if not isinstance(response, str):
        if hasattr(response, 'content'):
            response = response.content
        elif hasattr(response, 'text'):
            response = response.text
        else:
            response = str(response)

    code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if any(keyword in code for keyword in ['def ', 'class ', 'import ', 'for ', 'while ', 'if ']):
            return code

    lines = response.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    for i, line in enumerate(lines):
        if re.match(r'^(def |class |import |from )', line.strip()):
            return '\n'.join(lines[i:]).strip()

    return response.strip()


def plan_node(state: WorkflowState) -> WorkflowState:
    """Task planning node"""
    callback = state.get("progress_callback")
    log = AgentLogger("TaskPlanner")

    try:
        log.start("任务规划")

        if callback:
            callback.add_step('plan', '📋 任务规划 Agent 正在分析任务...', 'running')

        planner = TaskPlannerAgent()
        task_plan = planner.plan(state["task_description"])

        log.complete(f"任务: {task_plan.get('task', '未知')}")

        if callback:
            callback.add_step('plan', '📋 任务规划 Agent 已完成任务规划', 'completed', {'task_plan': task_plan})

        return {
            **state,
            "task_plan": task_plan,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "task_planner",
                "description": f"任务规划完成：{task_plan.get('task', '未知任务')}",
                "status": "completed",
                "output": task_plan
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('plan', f'❌ 任务规划失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def generate_node(state: WorkflowState) -> WorkflowState:
    """Code generation node"""
    callback = state.get("progress_callback")
    log = AgentLogger("CodeGenerator")

    try:
        log.start("代码生成")

        if callback:
            callback.add_step('generate', '💻 代码生成 Agent 正在编写代码...', 'running')

        generator = CodeGeneratorAgent()
        code_response = generator.generate(
            state["task_description"],
            state.get("task_plan", {})
        )

        generated_code = extract_code_from_response(code_response)

        log.complete(f"生成代码长度: {len(generated_code)} 字符")

        if callback:
            callback.add_step('generate', '💻 代码生成 Agent 已完成代码编写', 'completed', {'code': generated_code})

        return {
            **state,
            "generated_code": generated_code,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "code_generator",
                "description": "代码生成完成",
                "status": "completed",
                "output": generated_code
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('generate', f'❌ 代码生成失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def review_node(state: WorkflowState) -> WorkflowState:
    """Code review node"""
    callback = state.get("progress_callback")
    log = AgentLogger("CodeReviewer")

    try:
        log.start("代码审查")

        if callback:
            callback.add_step('review', '🔍 代码审查 Agent 正在审查代码...', 'running')

        reviewer = CodeReviewerAgent()
        review_result = reviewer.review(
            state.get("generated_code", ""),
            state["task_description"]
        )

        needs_fix = review_result.get("needs_revision", False)
        log.complete(f"需要修改: {needs_fix}, 评分: {review_result.get('score', 'N/A')}")

        if callback:
            callback.add_step('review', '🔍 代码审查 Agent 已完成审查', 'completed', {'review': review_result})

        return {
            **state,
            "review_result": review_result,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "code_reviewer",
                "description": f"代码审查完成，{'需要修改' if needs_fix else '无需修改'}",
                "status": "completed",
                "output": review_result
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('review', f'❌ 代码审查失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def fix_node(state: WorkflowState) -> WorkflowState:
    """Code fixing node"""
    callback = state.get("progress_callback")
    log = AgentLogger("CodeFixer")

    try:
        log.start("代码修复")

        if callback:
            callback.add_step('fix', '🔧 代码修复 Agent 正在优化代码...', 'running')

        fixer = CodeFixerAgent()
        fixed_code = fixer.fix(
            state.get("generated_code", ""),
            state.get("review_result", {})
        )

        log.complete(f"修复代码长度: {len(fixed_code)} 字符")

        if callback:
            callback.add_step('fix', '🔧 代码修复 Agent 已完成代码优化', 'completed', {'fixed_code': fixed_code})

        return {
            **state,
            "fixed_code": fixed_code,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "code_fixer",
                "description": "代码修复完成",
                "status": "completed",
                "output": fixed_code
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('fix', f'❌ 代码修复失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def should_fix(state: WorkflowState) -> str:
    """Decide whether to run fix node based on review result"""
    if state.get("error"):
        return "end"
    review_result = state.get("review_result", {})
    if review_result.get("needs_revision", False):
        return "fix"
    return "end"


def should_continue_after_generator(state: WorkflowState) -> str:
    """Check if workflow should continue after code generation"""
    if state.get("error"):
        return "end"
    return "reviewer"


def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", plan_node)
    workflow.add_node("generator", generate_node)
    workflow.add_node("reviewer", review_node)
    workflow.add_node("fixer", fix_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "generator")
    workflow.add_conditional_edges(
        "generator",
        should_continue_after_generator,
        {
            "reviewer": "reviewer",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "reviewer",
        should_fix,
        {
            "fix": "fixer",
            "end": END
        }
    )
    workflow.add_edge("fixer", END)

    return workflow.compile()


def format_workflow_result(state: WorkflowState) -> dict:
    """Format workflow result for response"""
    final_code = state.get("fixed_code") or state.get("generated_code") or ""

    return {
        "task_description": state["task_description"],
        "task_plan": state.get("task_plan"),
        "generated_code": state.get("generated_code"),
        "review_result": state.get("review_result"),
        "final_code": final_code,
        "workflow_steps": state.get("workflow_steps", []),
        "error": state.get("error")
    }