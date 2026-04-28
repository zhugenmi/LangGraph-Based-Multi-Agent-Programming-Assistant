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

            from src.graph.workflow import create_workflow
            workflow = create_workflow()

            result = workflow.invoke({
                "task_description": task_description,
                "session_id": session_id
            })

            return jsonify({"result": result, "session_id": session_id}), 200

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
                from src.graph.workflow import create_workflow
                workflow = create_workflow()

                progress.add_step('plan', '📋 任务规划 Agent 正在分析任务...', 'running')

                result = workflow.invoke(
                    {
                        "task_description": task_description,
                        "session_id": session_id,
                        "progress_callback": progress
                    }
                )

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

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)