"""Flask application for LangGraph Multi-Agent Programming Assistant"""

import os
import sys
import json
import queue
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class WorkflowProgress:
    """Thread-safe workflow progress tracker"""
    def __init__(self):
        self.queue = queue.Queue()
        self.steps = []
        self.finished = False
        self.error = None

    def add_step(self, step_name, description, status, data=None):
        step = {
            'step': step_name,
            'description': description,
            'status': status,
            'data': data
        }
        self.steps.append(step)
        self.queue.put(step)

    def complete(self, result):
        self.finished = True
        self.queue.put({'type': 'complete', 'result': result})

    def fail(self, error):
        self.error = error
        self.queue.put({'type': 'error', 'error': str(error)})


workflow_progress_store = {}


def create_app():
    """Create and configure the Flask application"""
    # Ensure required directories exist
    for d in ["outputs", "memory_store", "logs"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, 'templates'))
    CORS(app)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/v1/generate", methods=["POST"])
    def generate_code():
        """Legacy API - returns final result"""
        try:
            data = request.json
            task_description = data.get("task_description")
            session_id = data.get("session_id", "default")

            if not task_description:
                return jsonify({"error": "任务描述不能为空"}), 400

            from src.graph.workflow import create_workflow, format_workflow_result
            from src.utils.logger import TaskTimer, metrics_registry
            from src.utils.code_artifact_writer import save_generated_code

            workflow = create_workflow()

            with TaskTimer(session_id, session_id, task_description) as timer:
                result = workflow.invoke({
                    "task_description": task_description,
                    "session_id": session_id,
                    "max_iterations": 3,
                })
                timer.iterations = result.get("iteration_count", 0)
                # Count fix rounds
                fix_steps = [s for s in result.get("workflow_steps", []) if s.get("step_name") == "implementer"]
                timer.fix_rounds = max(0, len(fix_steps) - 1)

            # Save generated code to artifacts
            formatted_result = format_workflow_result(result)
            final_code = formatted_result.get("final_code") or formatted_result.get("generated_code")
            if final_code:
                task_plan = formatted_result.get("task_plan", {})
                language = task_plan.get("language", "python") if task_plan else "python"

                # Get metrics for this task
                task_metrics = metrics_registry.get_task(session_id)
                metrics_dict = task_metrics.to_dict() if task_metrics else {}

                metadata = {
                    "task_id": session_id,
                    "language": language,
                    "task_description": task_description,
                    "task_plan": task_plan,
                    "metrics": {
                        "total_duration_s": metrics_dict.get("total_duration_s", 0),
                        "llm_calls_count": len(metrics_dict.get("llm_calls", [])),
                        "total_tokens": sum(m.get("total_tokens", 0) for m in metrics_dict.get("llm_calls", [])),
                        "iterations": metrics_dict.get("iterations", 0),
                        "fix_rounds": timer.fix_rounds,
                    } if metrics_dict else {}
                }

                save_path = save_generated_code(
                    task_id=session_id,
                    code=final_code,
                    language=language,
                    metadata=metadata
                )
                formatted_result["artifact_path"] = str(save_path)

            return jsonify({
                "result": formatted_result,
                "session_id": session_id,
            }), 200

        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route("/api/v1/generate/stream", methods=["POST"])
    def generate_code_stream():
        """SSE endpoint for streaming progress"""
        data = request.json
        task_description = data.get("task_description")
        session_id = data.get("session_id", f"session_{int(time.time()*1000)}")

        if not task_description:
            return jsonify({"error": "任务描述不能为空"}), 400

        progress = WorkflowProgress()
        workflow_progress_store[session_id] = progress

        def run_workflow():
            try:
                from src.graph.workflow import create_workflow, format_workflow_result
                from src.utils.logger import TaskTimer, metrics_registry
                from src.utils.code_artifact_writer import save_generated_code

                progress.add_step('plan', '📋 任务规划 Agent 正在分析任务...', 'running')

                workflow = create_workflow()

                with TaskTimer(session_id, session_id, task_description) as timer:
                    result = workflow.invoke(
                        {
                            "task_description": task_description,
                            "session_id": session_id,
                            "max_iterations": 3,
                            "progress_callback": progress
                        }
                    )
                    timer.iterations = result.get("iteration_count", 0)
                    fix_steps = [s for s in result.get("workflow_steps", []) if s.get("step_name") == "implementer"]
                    timer.fix_rounds = max(0, len(fix_steps) - 1)

                # Save generated code to artifacts (in thread context)
                formatted_result = format_workflow_result(result)
                final_code = formatted_result.get("final_code") or formatted_result.get("generated_code")
                if final_code and final_code.strip():
                    try:
                        task_plan = formatted_result.get("task_plan", {})
                        language = task_plan.get("language", "python") if task_plan else "python"

                        task_metrics = metrics_registry.get_task(session_id)
                        metrics_dict = task_metrics.to_dict() if task_metrics else {}

                        metadata = {
                            "task_id": session_id,
                            "language": language,
                            "task_description": task_description,
                            "task_plan": task_plan,
                            "metrics": {
                                "total_duration_s": metrics_dict.get("total_duration_s", 0),
                                "llm_calls_count": len(metrics_dict.get("llm_calls", [])),
                                "total_tokens": sum(m.get("total_tokens", 0) for m in metrics_dict.get("llm_calls", [])),
                                "iterations": metrics_dict.get("iterations", 0),
                                "fix_rounds": timer.fix_rounds,
                            } if metrics_dict else {}
                        }

                        save_path = save_generated_code(
                            task_id=session_id,
                            code=final_code,
                            language=language,
                            metadata=metadata
                        )
                        formatted_result["artifact_path"] = str(save_path)
                    except Exception as save_err:
                        print(f"Warning: Failed to save artifact: {save_err}")

                progress.complete(result)

            except Exception as e:
                import traceback
                progress.fail(f"{str(e)}\n{traceback.format_exc()}")

        thread = threading.Thread(target=run_workflow)
        thread.daemon = True
        thread.start()

        def generate():
            try:
                step_count = 0
                while True:
                    try:
                        item = progress.queue.get(timeout=120)

                        if 'type' in item:
                            if item['type'] == 'complete':
                                yield f"data: {json.dumps({'type': 'complete', 'result': item['result']}, ensure_ascii=False)}\n\n"
                                break
                            elif item['type'] == 'error':
                                yield f"data: {json.dumps({'type': 'error', 'error': item['error']}, ensure_ascii=False)}\n\n"
                                break
                        else:
                            step_count += 1
                            yield f"data: {json.dumps({'type': 'progress', **item}, ensure_ascii=False)}\n\n"

                        if progress.finished or progress.error:
                            break

                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'ping'}, ensure_ascii=False)}\n\n"
                        continue

            except GeneratorExit:
                pass

        return Response(generate(), mimetype='text/event-stream')

    @app.route("/api/v1/execute", methods=["POST"])
    def execute_code():
        """Execute Python code"""
        try:
            data = request.json
            code = data.get("code", "")

            if not code:
                return jsonify({"error": "代码不能为空", "success": False}), 400

            from src.sandbox.code_executor import CodeExecutor
            executor = CodeExecutor()
            result = executor.execute(code)

            return jsonify(result), 200

        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "success": False, "output": ""}), 500

    @app.route("/chat")
    def chat():
        """Multi-turn conversation page"""
        return render_template("chat.html")

    @app.route("/api/v1/chat", methods=["POST"])
    def chat_message():
        """Multi-turn chat API"""
        try:
            data = request.json
            message = data.get("message", "")
            session_id = data.get("session_id", f"session_{int(time.time()*1000)}")

            if not message:
                return jsonify({"error": "消息不能为空"}), 400

            # Initialize memory manager
            from src.memory.memory_manager import MemoryManager
            memory = MemoryManager(session_id=session_id)

            # Add user message to history
            memory.add_user_message(message)

            # Get context from memory and RAG
            context_text = ""

            # Try to get context from RAG
            try:
                from src.rag.code_rag import create_code_rag
                rag = create_code_rag(repo_path=str(PROJECT_ROOT), use_local_embedding=True, incremental=True)
                rag_context = rag.get_context_for_query(message, max_tokens=1000)
                context_text += rag_context
            except Exception as e:
                print(f"RAG error: {e}")

            # Get conversation context from memory
            memory_context = memory.get_context_for_prompt(max_tokens=500)
            context_text += memory_context

            # Generate response using LLM
            response_text = ""
            try:
                from src.llm.llm_model_client import get_llm_client
                client = get_llm_client()

                # Check if it's a task that needs full workflow
                task_keywords = ["实现", "创建", "编写", "生成", "修复", "修改", "添加", "开发"]
                needs_workflow = any(kw in message for kw in task_keywords)

                if needs_workflow:
                    # Use full workflow
                    response_text = f"收到任务: {message}\n\n正在启动多Agent工作流处理..."
                    # Note: For streaming response, you'd use a different approach
                else:
                    # Simple chat response
                    prompt = f"{context_text}\n\n用户问题: {message}\n\n请简洁回答:"
                    response = client.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)

            except Exception as e:
                response_text = f"LLM调用失败: {str(e)}"

            # Add assistant response to memory
            memory.add_assistant_message(response_text)

            # Get current context info
            task_state = memory.get_current_task("current")

            return jsonify({
                "response": response_text,
                "session_id": session_id,
                "context": {
                    "conversation_turns": len(memory.get_conversation_history()),
                    "task_info": task_state or {}
                }
            }), 200

        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route("/api/v1/chat/history/<session_id>", methods=["GET"])
    def get_chat_history(session_id):
        """Get conversation history"""
        try:
            from src.memory.memory_manager import MemoryManager
            memory = MemoryManager(session_id=session_id)
            history = memory.get_conversation_history(limit=50)

            return jsonify({"history": history, "session_id": session_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/memory/status/<session_id>", methods=["GET"])
    def get_memory_status(session_id):
        """Get memory status for a session"""
        try:
            from src.memory.memory_manager import MemoryManager
            memory = MemoryManager(session_id=session_id)
            summary = memory.get_summary()

            st = summary.get("short_term", {})
            return jsonify({
                "session_id": session_id,
                "conversation_turns": st.get("conversation_turns", 0),
                "task_states": st.get("task_states", 0)
            }), 200
        except FileNotFoundError:
            return jsonify({
                "session_id": session_id,
                "conversation_turns": 0,
                "task_states": 0,
                "message": "会话无记忆数据"
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/memory/search", methods=["POST"])
    def search_memory():
        """Search memory for relevant context"""
        try:
            data = request.json
            query = data.get("query", "")
            session_id = data.get("session_id", "default")

            from src.memory.memory_manager import MemoryManager
            memory = MemoryManager(session_id=session_id)

            # Get similar historical tasks
            similar_tasks = memory.get_similar_historical_tasks(query, limit=5)

            # Get project knowledge
            knowledge = memory.get_project_knowledge()

            return jsonify({
                "similar_tasks": similar_tasks,
                "project_knowledge": knowledge
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/rag/stats", methods=["GET"])
    def get_rag_stats():
        """Get RAG index statistics"""
        try:
            from src.rag.code_rag import create_code_rag, get_rag_config, CodeRAG
            # Try loading existing index first (no embedding needed)
            rag = CodeRAG(repo_path=str(PROJECT_ROOT), use_local_embedding=False)
            stats = rag.get_stats()
            stats["config"] = get_rag_config()
            return jsonify(stats), 200
        except Exception as e:
            err_msg = str(e).lower()
            if "ollama" in err_msg or "embedding" in err_msg:
                return jsonify({
                    "error": "Embedding 服务不可用",
                    "detail": str(e),
                    "solution": "启动 Ollama 服务或配置云端 embedding API"
                }), 200
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/rag/config", methods=["GET"])
    def get_rag_config_api():
        """Get RAG configuration"""
        try:
            from src.rag.code_rag import get_rag_config
            config = get_rag_config()
            return jsonify(config), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/rag/build", methods=["POST"])
    def build_rag_index():
        """Build or rebuild RAG index (full rebuild)"""
        try:
            from src.rag.code_rag import CodeRAG
            rag = CodeRAG(repo_path=str(PROJECT_ROOT))
            result = rag.build_index()

            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/rag/update", methods=["POST"])
    def update_rag_index():
        """Incrementally update RAG index (only changed files)"""
        try:
            data = request.json or {}
            rebuild_threshold = data.get("rebuild_threshold", 50)

            from src.rag.code_rag import CodeRAG
            rag = CodeRAG(repo_path=str(PROJECT_ROOT))
            result = rag.incremental_update(rebuild_threshold=rebuild_threshold)

            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        from src.memory.session_manager import SessionManager
        session_manager = SessionManager()
        session = session_manager.get_session(session_id)
        if session:
            return jsonify({"session": session.to_dict()}), 200
        return jsonify({"error": "会话不存在"}), 404

    @app.route("/api/v1/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        from src.memory.session_manager import SessionManager
        session_manager = SessionManager()
        session_manager.delete_session(session_id)
        return jsonify({"message": "会话删除成功"}), 200

    @app.route("/api/v1/metrics", methods=["GET"])
    def get_metrics():
        """Get all task metrics"""
        try:
            from src.utils.logger import metrics_registry
            all_metrics = metrics_registry.get_all_summary()
            return jsonify({"metrics": all_metrics, "count": len(all_metrics)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/metrics/<task_id>", methods=["GET"])
    def get_task_metrics(task_id):
        """Get metrics for a specific task"""
        try:
            from src.utils.logger import metrics_registry
            task = metrics_registry.get_task(task_id)
            if task:
                return jsonify(task.to_dict()), 200
            return jsonify({"error": "Task not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/metrics/export", methods=["POST"])
    def export_metrics():
        """Export metrics to JSON file"""
        try:
            data = request.json or {}
            output_dir = data.get("output_dir")
            from src.utils.logger import export_metrics_report
            report_path = export_metrics_report(output_dir)
            return jsonify({"report_path": report_path}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)