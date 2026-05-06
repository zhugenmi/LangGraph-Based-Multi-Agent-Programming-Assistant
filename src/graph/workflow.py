"""LangGraph workflow for multi-agent code generation"""

import re
import time
from typing import TypedDict, Annotated, Sequence, Optional, Callable, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agents.supervisor import SupervisorAgent
from src.agents.repo_analyst_agent import RepoAnalystAgent
from src.agents.implementer import Implementer
from src.agents.reviewer import ReviewerAgent
from src.agents.tester import Tester
from src.utils.logger import AgentLogger, get_model_name_from_env, TaskTimer, set_task_id


class WorkflowState(TypedDict):
    """State schema for the workflow"""
    task_description: str
    session_id: str
    task_plan: Optional[dict]
    repo_analysis: Optional[dict]
    task_type: Optional[str]  # "modify_existing" | "create_new" | "mixed"
    generated_code: Optional[str]
    implementation_code: Optional[str]  # unified code from implementer/fixer
    review_result: Optional[dict]
    fixed_code: Optional[str]
    workflow_steps: list
    error: Optional[str]
    progress_callback: Optional[Any]
    iteration_count: int
    max_iterations: int
    project_created: Optional[bool]  # Whether a new project was created


def extract_code_from_response(response: str) -> str:
    """Extract code from LLM response, supporting multiple programming languages"""
    if not isinstance(response, str):
        if hasattr(response, 'content'):
            response = response.content
        elif hasattr(response, 'text'):
            response = response.text
        else:
            response = str(response)

    languages = ['python', 'javascript', 'java', 'cpp', 'c', 'go', 'rust', 'ruby', 'php',
                 'swift', 'kotlin', 'typescript', 'bash', 'shell', 'sql', 'html', 'css']

    for lang in languages:
        code_match = re.search(rf'```{lang}\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()

    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        code_indicators = ['def ', 'class ', 'function ', 'func ', 'public ', 'private ',
                          'import ', 'from ', 'const ', 'let ', 'var ', 'struct ', 'enum ',
                          'fn ', 'impl ', 'module ', 'package ', 'interface ']
        if any(keyword in code for keyword in code_indicators):
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

    code_indicators = ['def ', 'class ', 'function ', 'func ', 'public ', 'private ',
                      'import ', 'from ', 'const ', 'let ', 'var ', 'struct ', 'enum ',
                      'fn ', 'impl ', 'module ', 'package ', 'interface ']
    for i, line in enumerate(lines):
        if any(line.strip().startswith(indicator) for indicator in code_indicators):
            return '\n'.join(lines[i:]).strip()

    return response.strip()


def _make_log(agent_name: str, task_id: str = "") -> AgentLogger:
    """Create an AgentLogger with task_id for metrics tracking"""
    model_name = get_model_name_from_env(agent_name)
    return AgentLogger(agent_name, model_name=model_name, task_id=task_id)


def plan_node(state: WorkflowState) -> WorkflowState:
    """Task planning node"""
    callback = state.get("progress_callback")
    task_id = state.get("session_id", "")
    log = _make_log("Supervisor", task_id)

    try:
        log.start("任务规划")
        set_task_id(task_id)

        if callback:
            callback.add_step('plan', f'📋 Supervisor Agent 正在分析任务...', 'running')

        planner = SupervisorAgent()
        task_plan = planner.plan(state["task_description"])

        log.complete(f"任务: {task_plan.get('task', '未知')}")

        if callback:
            callback.add_step('plan', '📋 Supervisor Agent 已完成任务规划', 'completed', {'task_plan': task_plan})

        return {
            **state,
            "task_plan": task_plan,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "supervisor",
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


def repo_analysis_node(state: WorkflowState) -> WorkflowState:
    """Repository analysis node"""
    callback = state.get("progress_callback")
    task_id = state.get("session_id", "")
    log = _make_log("RepoAnalyst", task_id)

    try:
        log.start("代码库分析")

        if callback:
            callback.add_step('repo_analysis', '🔍 RepoAnalyst Agent 正在分析代码库...', 'running')

        analyst = RepoAnalystAgent()
        repo_analysis = analyst.analyze(state["task_description"])

        log.complete(f"分析完成，任务类型：{repo_analysis.get('task_type', 'modify_existing')}")

        task_type = repo_analysis.get("task_type", "modify_existing")

        if callback:
            if task_type == "create_new":
                callback.add_step('repo_analysis', '🔍 RepoAnalyst Agent 检测到新项目创建任务', 'completed', {
                    'analysis': repo_analysis,
                    'task_type': task_type,
                    'project_info': repo_analysis.get("project_name", "unknown")
                })
            else:
                callback.add_step('repo_analysis', '🔍 RepoAnalyst Agent 已完成代码库分析', 'completed', {
                    'analysis': repo_analysis,
                    'task_type': task_type
                })

        return {
            **state,
            "repo_analysis": repo_analysis,
            "task_type": task_type,
            "project_created": repo_analysis.get("scaffold_result", {}).get("success", False),
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "repo_analyst",
                "description": f"代码库分析完成，任务类型：{task_type}",
                "status": "completed",
                "output": repo_analysis
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('repo_analysis', f'❌ 代码库分析失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def generate_node(state: WorkflowState) -> WorkflowState:
    """Code generation node"""
    callback = state.get("progress_callback")
    task_id = state.get("session_id", "")
    log = _make_log("Implementer", task_id)

    try:
        log.start("代码生成")

        if callback:
            callback.add_step('generate', '💻 Implementer Agent 正在编写代码...', 'running')

        generator = Implementer()
        code_response = generator.generate(
            state["task_description"],
            state.get("task_plan", {}),
            state.get("repo_analysis", {})
        )

        generated_code = extract_code_from_response(code_response)

        log.complete(f"生成代码长度: {len(generated_code)} 字符")

        if callback:
            callback.add_step('generate', '💻 Implementer Agent 已完成代码编写', 'completed', {'code': generated_code})

        return {
            **state,
            "generated_code": generated_code,
            "implementation_code": generated_code,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "implementer",
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
    task_id = state.get("session_id", "")
    log = _make_log("Reviewer", task_id)

    try:
        log.start("代码审查")

        if callback:
            callback.add_step('review', '🔍 Reviewer Agent 正在审查代码...', 'running')

        reviewer = ReviewerAgent()
        review_result = reviewer.review(
            state.get("generated_code", ""),
            state["task_description"]
        )

        needs_fix = review_result.get("needs_revision", False)
        log.complete(f"需要修改: {needs_fix}, 评分: {review_result.get('score', 'N/A')}")

        if callback:
            callback.add_step('review', '🔍 Reviewer Agent 已完成审查', 'completed', {'review': review_result})

        return {
            **state,
            "review_result": review_result,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "reviewer",
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
    task_id = state.get("session_id", "")
    log = _make_log("Implementer", task_id)

    try:
        log.start("代码修复")

        if callback:
            callback.add_step('fix', '🔧 Implementer Agent 正在优化代码...', 'running')

        fixer = Implementer()
        fixed_code = fixer.fix(
            state.get("generated_code", ""),
            state.get("review_result", {})
        )

        log.complete(f"修复代码长度: {len(fixed_code)} 字符")

        if callback:
            callback.add_step('fix', '🔧 Implementer Agent 已完成代码优化', 'completed', {'fixed_code': fixed_code})

        return {
            **state,
            "fixed_code": fixed_code,
            "implementation_code": fixed_code,
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "implementer",
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


def test_node(state: WorkflowState) -> WorkflowState:
    """Test execution node"""
    callback = state.get("progress_callback")
    task_id = state.get("session_id", "")
    log = _make_log("Tester", task_id)

    try:
        log.start("测试执行")

        if callback:
            callback.add_step('test', '🧪 Tester Agent 正在生成和执行测试...', 'running')

        tester = Tester()
        code = state.get("implementation_code") or state.get("generated_code", "") or state.get("fixed_code", "")

        test_result = tester.generate_tests(code, state["task_description"])
        run_result = tester.run_tests(code, test_result.get("test_code"))

        log.complete(f"测试结果: {run_result.get('passed', 0)}通过, {run_result.get('failed', 0)}失败")

        if callback:
            callback.add_step('test', '🧪 Tester Agent 已完成测试执行', 'completed', {
                'test_result': run_result,
                'test_analysis': tester.analyze_test_results(run_result)
            })

        return {
            **state,
            "test_result": {
                "generation": test_result,
                "execution": run_result,
                "analysis": tester.analyze_test_results(run_result)
            },
            "workflow_steps": state.get("workflow_steps", []) + [{
                "step_name": "tester",
                "description": f"测试完成: {run_result.get('passed', 0)}/{run_result.get('total', 0)} 通过",
                "status": "completed",
                "output": {
                    "passed": run_result.get('passed', 0),
                    "failed": run_result.get('failed', 0),
                    "total": run_result.get('total', 0),
                    "output": run_result.get('output', ''),
                    "error": run_result.get('error')
                }
            }]
        }
    except Exception as e:
        log.fail(str(e))
        if callback:
            callback.add_step('test', f'❌ 测试执行失败: {str(e)}', 'error')
        return {**state, "error": str(e)}


def should_fix(state: WorkflowState) -> str:
    """Decide whether to run fix node based on review result"""
    if state.get("error"):
        return "end"
    review_result = state.get("review_result", {})
    if review_result.get("needs_revision", False):
        return "fix"
    return "end"


def should_continue_after_planner(state: WorkflowState) -> str:
    """Check if workflow should continue after task planning"""
    if state.get("error"):
        return "end"
    return "repo_analyst"


def should_continue_after_repo_analysis(state: WorkflowState) -> str:
    """Check if workflow should continue after repo analysis"""
    if state.get("error"):
        return "end"
    return "implementer"


def should_continue_after_generator(state: WorkflowState) -> str:
    """Check if workflow should continue after code generation"""
    if state.get("error"):
        return "end"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
        return "end"
    return "reviewer"


def should_continue_after_reviewer(state: WorkflowState) -> str:
    """Check if workflow should continue after review"""
    if state.get("error"):
        return "end"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
        return "end"

    review_result = state.get("review_result", {})
    if review_result.get("needs_revision", False):
        return "fixer"
    else:
        return "tester"


def should_continue_after_fixer(state: WorkflowState) -> str:
    """Check if workflow should continue after fixing"""
    if state.get("error"):
        return "end"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
        return "end"
    return "implementer"


def should_continue_after_tester(state: WorkflowState) -> str:
    """Check if workflow should continue after testing"""
    if state.get("error"):
        return "end"

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    if iteration_count >= max_iterations:
        return "end"

    test_result = state.get("test_result", {}).get("execution", {})
    if test_result.get("failed", 0) > 0 or test_result.get("total", 0) == 0:
        return "increment"

    return "end"


def increment_iteration(state: WorkflowState) -> WorkflowState:
    """Increment iteration count"""
    return {
        **state,
        "iteration_count": (state.get("iteration_count", 0) + 1)
    }


def create_workflow(max_iterations: int = 3):
    """Create the LangGraph workflow with enhanced structure"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("supervisor", plan_node)
    workflow.add_node("repo_analyst", repo_analysis_node)
    workflow.add_node("implementer", generate_node)
    workflow.add_node("reviewer", review_node)
    workflow.add_node("fixer", fix_node)
    workflow.add_node("tester", test_node)
    workflow.add_node("increment", increment_iteration)
    workflow.add_node("end", lambda state: state)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "repo_analyst")
    workflow.add_edge("repo_analyst", "implementer")

    workflow.add_conditional_edges(
        "implementer",
        should_continue_after_generator,
        {
            "reviewer": "reviewer",
            "end": "end",
        }
    )

    workflow.add_conditional_edges(
        "reviewer",
        should_continue_after_reviewer,
        {
            "fixer": "fixer",
            "tester": "tester",
            "end": "end",
        }
    )

    workflow.add_conditional_edges(
        "fixer",
        should_continue_after_fixer,
        {
            "implementer": "implementer",
            "end": "end",
        }
    )

    workflow.add_conditional_edges(
        "tester",
        should_continue_after_tester,
        {
            "increment": "increment",
            "end": "end",
        }
    )

    workflow.add_edge("increment", "implementer")
    workflow.add_edge("end", END)

    return workflow.compile()


def format_workflow_result(state: WorkflowState) -> dict:
    """Format workflow result for response"""
    final_code = state.get("fixed_code") or state.get("generated_code") or ""

    return {
        "task_description": state["task_description"],
        "task_plan": state.get("task_plan"),
        "repo_analysis": state.get("repo_analysis"),
        "generated_code": state.get("generated_code"),
        "review_result": state.get("review_result"),
        "test_result": state.get("test_result"),
        "final_code": final_code,
        "workflow_steps": state.get("workflow_steps", []),
        "error": state.get("error"),
        "iterations": state.get("iteration_count", 0)
    }
