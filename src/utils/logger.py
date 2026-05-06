"""Logging module with comprehensive metrics tracking for agent execution"""

import logging
import sys
import os
import json
import time
from contextvars import ContextVar
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Context variable for task_id propagation across async/thread boundaries
_current_task_id: ContextVar[str] = ContextVar("_current_task_id", default="")


def set_task_id(task_id: str):
    """Set the current task_id in the context variable (call at workflow start)."""
    _current_task_id.set(task_id)


def get_task_id() -> str:
    """Get the current task_id from the context variable."""
    return _current_task_id.get()


# =========================
# Basic Logger Setup
# =========================

def setup_logger(name: str = "multi_agents_coder", level: int = logging.DEBUG) -> logging.Logger:
    """Setup and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def get_model_name_from_env(agent_name: str) -> str:
    """从环境变量获取 Agent 使用的模型名称"""
    prefix_map = {
        "supervisor": "SUPERVISOR",
        "repo_analyst": "REPO_ANALYST",
        "implementer": "IMPLEMENTER",
        "reviewer": "REVIEWER",
        "tester": "TESTER",
        "Supervisor": "SUPERVISOR",
        "RepoAnalyst": "REPO_ANALYST",
        "Implementer": "IMPLEMENTER",
        "Reviewer": "REVIEWER",
        "Tester": "TESTER",
    }

    prefix = prefix_map.get(agent_name, "DEFAULT")
    model_name = (
        os.getenv(f"{prefix}_MODEL")
        if prefix != "DEFAULT"
        else os.getenv("DEFAULT_MODEL")
    )

    return model_name or "default"


# =========================
# Metrics Data Classes
# =========================

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call"""
    agent_name: str
    model_name: str
    endpoint: str  # invoke / generate / stream
    start_time: float
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    status_code: int = 200
    error: str = ""
    response_length: int = 0  # raw response string length

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.total_tokens / self.duration_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = round(self.duration_ms, 2)
        d["duration_s"] = round(self.duration_s, 4)
        d["tokens_per_second"] = round(self.tokens_per_second, 1)
        return d


@dataclass
class TaskMetrics:
    """Metrics for an entire workflow task"""
    task_id: str
    session_id: str
    task_description: str
    start_time: float
    end_time: float = 0.0
    iterations: int = 0
    status: str = "running"  # running / completed / failed

    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    fix_rounds: int = 0  # number of fix iterations
    error: str = ""

    @property
    def total_duration_s(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_s * 1000

    @property
    def avg_fix_rounds(self) -> float:
        if self.iterations == 0:
            return 0.0
        return self.fix_rounds / max(self.iterations, 1)

    def record_llm_call(self, metrics: LLMCallMetrics):
        self.llm_calls.append(metrics)

    def agent_summary(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics per agent"""
        summary: Dict[str, Dict[str, Any]] = {}
        for call in self.llm_calls:
            if call.agent_name not in summary:
                summary[call.agent_name] = {
                    "agent_name": call.agent_name,
                    "model_name": call.model_name,
                    "call_count": 0,
                    "total_duration_s": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "avg_tokens_per_second": 0.0,
                    "errors": 0,
                }
            s = summary[call.agent_name]
            s["call_count"] += 1
            s["total_duration_s"] += call.duration_s
            s["total_input_tokens"] += call.input_tokens
            s["total_output_tokens"] += call.output_tokens
            s["total_tokens"] += call.total_tokens
            if call.error:
                s["errors"] += 1

        for s in summary.values():
            s["total_duration_s"] = round(s["total_duration_s"], 4)
            if s["total_duration_s"] > 0:
                s["avg_tokens_per_second"] = round(s["total_tokens"] / s["total_duration_s"], 1)

        return summary

    def llm_summary(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics per model"""
        summary: Dict[str, Dict[str, Any]] = {}
        for call in self.llm_calls:
            key = call.model_name
            if key not in summary:
                summary[key] = {
                    "model_name": key,
                    "call_count": 0,
                    "total_duration_s": 0.0,
                    "avg_duration_ms": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "avg_tokens_per_second": 0.0,
                }
            s = summary[key]
            s["call_count"] += 1
            s["total_duration_s"] += call.duration_s
            s["total_input_tokens"] += call.input_tokens
            s["total_output_tokens"] += call.output_tokens
            s["total_tokens"] += call.total_tokens

        for s in summary.values():
            s["total_duration_s"] = round(s["total_duration_s"], 4)
            s["avg_duration_ms"] = round(s["total_duration_s"] * 1000 / max(s["call_count"], 1), 2)
            if s["total_duration_s"] > 0:
                s["avg_tokens_per_second"] = round(s["total_tokens"] / s["total_duration_s"], 1)

        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "task_description": self.task_description[:200],
            "status": self.status,
            "total_duration_s": round(self.total_duration_s, 4),
            "iterations": self.iterations,
            "fix_rounds": self.fix_rounds,
            "avg_fix_rounds": round(self.avg_fix_rounds, 2),
            "total_llm_calls": len(self.llm_calls),
            "total_tokens": sum(c.total_tokens for c in self.llm_calls),
            "agent_summary": self.agent_summary(),
            "llm_summary": self.llm_summary(),
            "errors": self.error[:500] if self.error else None,
        }


# =========================
# Global Metrics Registry
# =========================

class MetricsRegistry:
    """Global registry to track all task metrics"""

    def __init__(self):
        self._tasks: Dict[str, TaskMetrics] = {}
        self._lock = __import__("threading").Lock()

    def start_task(self, task_id: str, session_id: str, task_description: str) -> TaskMetrics:
        with self._lock:
            metrics = TaskMetrics(
                task_id=task_id,
                session_id=session_id,
                task_description=task_description,
                start_time=time.time(),
            )
            self._tasks[task_id] = metrics
            return metrics

    def get_task(self, task_id: str) -> Optional[TaskMetrics]:
        return self._tasks.get(task_id)

    def complete_task(self, task_id: str, status: str = "completed", error: str = ""):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.end_time = time.time()
                task.status = status
                task.error = error

    def record_llm_call(self, task_id: str, call: LLMCallMetrics):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.llm_calls.append(call)

    def get_all_summary(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tasks.values()]


metrics_registry = MetricsRegistry()


# =========================
# Agent Logger (enhanced)
# =========================

class AgentLogger:
    """Logger for tracking agent execution with metrics"""

    def __init__(self, agent_name: str, model_name: str = None, task_id: str = ""):
        self.agent_name = agent_name
        self.model_name = model_name or get_model_name_from_env(agent_name)
        self.task_id = task_id
        self._logger = logger

    def _get_prefixed_agent_name(self) -> str:
        return f"{self.agent_name}-{self.model_name}"

    def info(self, message: str):
        self._logger.info(f"[{self._get_prefixed_agent_name()}] {message}")

    def debug(self, message: str):
        self._logger.debug(f"[{self._get_prefixed_agent_name()}] {message}")

    def warning(self, message: str):
        self._logger.warning(f"[{self._get_prefixed_agent_name()}] {message}")

    def error(self, message: str):
        self._logger.error(f"[{self._get_prefixed_agent_name()}] {message}")

    def start(self, task: str = ""):
        self.info(f"[START] {task}")

    def complete(self, result_summary: str = ""):
        self.info(f"[DONE] {result_summary}")

    def fail(self, error: str):
        self.error(f"[FAIL] {error}")

    def step(self, step_name: str, message: str = ""):
        prefix = f"[{step_name}]" if step_name else ""
        self.debug(f"{prefix} {message}")

    def llm_call(self, endpoint: str, duration_ms: float, tokens: int,
                 input_tokens: int = 0, output_tokens: int = 0,
                 status: str = "ok", error: str = "", response_length: int = 0):
        """Log LLM API call metrics"""
        level = "OK" if status == "ok" else "ERR"
        msg = (f"[LLM-{level}] {endpoint} | "
               f"{duration_ms:.0f}ms | "
               f"tokens={tokens} "
               f"(in={input_tokens}, out={output_tokens}) | "
               f"tp={tokens / (duration_ms/1000):.0f}/s" if duration_ms > 0 else "")
        if error:
            msg += f" | error={error[:100]}"
        self.info(msg)

        # Record to metrics registry
        effective_task_id = self.task_id or get_task_id()
        if effective_task_id:
            call_metrics = LLMCallMetrics(
                agent_name=self.agent_name,
                model_name=self.model_name,
                endpoint=endpoint,
                start_time=time.time() - duration_ms / 1000,
                end_time=time.time(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=tokens,
                status_code=200 if status == "ok" else 500,
                error=error,
                response_length=response_length,
            )
            metrics_registry.record_llm_call(effective_task_id, call_metrics)


# =========================
# Context Manager for LLM Calls
# =========================

class LLMCallTimer:
    """Context manager to time LLM calls and auto-record metrics.

    Usage:
        with LLMCallTimer(task_id, agent_name, model_name, endpoint="invoke") as timer:
            response = client.invoke(prompt)
        # Metrics auto-recorded after the block
    """

    def __init__(self, task_id: str, agent_name: str, model_name: str,
                 endpoint: str = "invoke"):
        self.task_id = task_id
        self.agent_name = agent_name
        self.model_name = model_name
        self.endpoint = endpoint
        self._start = 0.0
        self._end = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.error = ""
        self.response_length = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()
        if exc_type:
            self.error = str(exc_val)[:200]
        self._record()
        return False

    def _record(self):
        duration_ms = (self._end - self._start) * 1000
        log = AgentLogger(self.agent_name, self.model_name, self.task_id)
        log.llm_call(
            endpoint=self.endpoint,
            duration_ms=duration_ms,
            tokens=self.total_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            status="ok" if not self.error else "error",
            error=self.error,
            response_length=self.response_length,
        )

    @property
    def duration_ms(self) -> float:
        return (self._end - self._start) * 1000


# =========================
# Task Timer
# =========================

class TaskTimer:
    """Context manager to time an entire workflow task.

    Usage:
        with TaskTimer(task_id, session_id, task_description) as timer:
            # run workflow...
            timer.iterations = 3
            timer.fix_rounds = 1
        # Task auto-completed after the block
    """

    def __init__(self, task_id: str, session_id: str, task_description: str):
        self.task_id = task_id
        self.session_id = session_id
        self.task_description = task_description
        self.iterations = 0
        self.fix_rounds = 0
        self.error = ""

    def __enter__(self) -> 'TaskTimer':
        metrics_registry.start_task(self.task_id, self.session_id, self.task_description)
        self._logger.info(f"[TASK-START] id={self.task_id} session={self.session_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        if exc_val:
            self.error = str(exc_val)[:500]
        metrics_registry.complete_task(self.task_id, status=status, error=self.error)
        metrics = metrics_registry.get_task(self.task_id)
        if metrics:
            metrics.iterations = self.iterations
            metrics.fix_rounds = self.fix_rounds
        self._log_summary()
        return False

    def _log_summary(self):
        metrics = metrics_registry.get_task(self.task_id)
        if not metrics:
            return
        d = metrics.to_dict()
        self._logger.info(
            f"[TASK-{d['status'].upper()}] id={self.task_id} "
            f"duration={d['total_duration_s']:.2f}s | "
            f"iterations={d['iterations']} | "
            f"fix_rounds={d['fix_rounds']} | "
            f"llm_calls={d['total_llm_calls']} | "
            f"total_tokens={d['total_tokens']}"
        )
        # Log per-agent summary
        for agent_name, agent_stats in d["agent_summary"].items():
            self._logger.info(
                f"  [Agent] {agent_name}: calls={agent_stats['call_count']} "
                f"duration={agent_stats['total_duration_s']:.2f}s "
                f"tokens={agent_stats['total_tokens']} "
                f"tp={agent_stats['avg_tokens_per_second']:.0f}/s"
            )

    @property
    def _logger(self):
        return logger


# =========================
# Metrics Report Export
# =========================

def export_metrics_report(output_dir: str = None) -> str:
    """Export all task metrics to a JSON file.

    Args:
        output_dir: Directory to save the report. Defaults to logs/.

    Returns:
        Path to the exported report file.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent / "logs")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"metrics_report_{timestamp}.json")

    all_metrics = metrics_registry.get_all_summary()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"[METRICS] Report exported to {report_path}")
    return report_path
