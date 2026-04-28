"""Logging module for debugging and monitoring agent execution"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "multi_agents_coder", level: int = logging.DEBUG) -> logging.Logger:
    """Setup and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(
        Path(__file__).parent.parent.parent / "logs" / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )
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


class AgentLogger:
    """Logger for tracking agent execution"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logger

    def info(self, message: str):
        self.logger.info(f"[{self.agent_name}] {message}")

    def debug(self, message: str):
        self.logger.debug(f"[{self.agent_name}] {message}")

    def warning(self, message: str):
        self.logger.warning(f"[{self.agent_name}] {message}")

    def error(self, message: str):
        self.logger.error(f"[{self.agent_name}] {message}")

    def start(self, task: str = ""):
        self.info(f"🚀 开始执行 {task}")

    def complete(self, result_summary: str = ""):
        self.info(f"✅ 执行完成 {result_summary}")

    def fail(self, error: str):
        self.error(f"❌ 执行失败: {error}")

    def step(self, step_name: str, message: str = ""):
        prefix = f"[{step_name}]" if step_name else ""
        self.debug(f"{prefix} {message}")
