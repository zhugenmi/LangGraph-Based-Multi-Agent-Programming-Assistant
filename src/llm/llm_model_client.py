"""通用 LLM 客户端，支持从 .env 配置文件读取 API 设置，带 metrics 记录"""

import os
import time
from typing import Optional, Any, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Agent 到配置项前缀的映射
AGENT_CONFIG_PREFIX = {
    "supervisor": "SUPERVISOR",
    "repo_analyst": "REPO_ANALYST",
    "implementer": "IMPLEMENTER",
    "reviewer": "REVIEWER",
    "tester": "TESTER"
}

# 默认配置项
DEFAULT_CONFIG = {
    "MODEL": os.getenv("DEFAULT_MODEL", ""),
    "MODEL_API_KEY": os.getenv("DEFAULT_MODEL_API_KEY", ""),
    "MODEL_BASE_URL": os.getenv("DEFAULT_MODEL_BASE_URL", "")
}


def get_agent_model_config(agent_name: str) -> Dict[str, str]:
    """获取指定 Agent 的模型配置（模型名称、API Key、Base URL）"""
    prefix = AGENT_CONFIG_PREFIX.get(agent_name)

    if not prefix:
        return {
            "model": DEFAULT_CONFIG["MODEL"],
            "api_key": DEFAULT_CONFIG["MODEL_API_KEY"],
            "base_url": DEFAULT_CONFIG["MODEL_BASE_URL"]
        }

    agent_model = os.getenv(f"{prefix}_MODEL", "")
    agent_api_key = os.getenv(f"{prefix}_MODEL_API_KEY", "")
    agent_base_url = os.getenv(f"{prefix}_MODEL_BASE_URL", "")

    model = agent_model if agent_model else DEFAULT_CONFIG["MODEL"]
    api_key = agent_api_key if agent_api_key else DEFAULT_CONFIG["MODEL_API_KEY"]
    base_url = agent_base_url if agent_base_url else DEFAULT_CONFIG["MODEL_BASE_URL"]

    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url
    }


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from LangChain response if available."""
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    # Try response.response_metadata
    metadata = getattr(response, "response_metadata", None)
    if metadata:
        token_usage = metadata.get("token_usage", {})
        if token_usage:
            usage["input_tokens"] = token_usage.get("input_tokens", 0)
            usage["output_tokens"] = token_usage.get("output_tokens", 0)
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            return usage
        # Some providers put it at top level
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            if key in metadata:
                usage[key] = metadata[key]
    # Fallback: estimate from content length
    content = getattr(response, "content", "")
    if content and usage["total_tokens"] == 0:
        # Rough estimate: ~4 chars per token for English, ~1.5 for Chinese
        usage["output_tokens"] = max(1, len(str(content)) // 3)
        usage["total_tokens"] = usage["output_tokens"]
    return usage


class LlmModelClient:
    """通用 LLM 客户端，兼容 OpenAI API 格式，带响应耗时和 token 统计"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = None
    ):
        if agent_name:
            config = get_agent_model_config(agent_name)
            self.model = model or config["model"]
            self.api_key = api_key or config["api_key"]
            self.base_url = base_url or config["base_url"]
        else:
            self.model = model or os.getenv("LLM_MODEL", DEFAULT_CONFIG["MODEL"])
            self.api_key = api_key or os.getenv("MODEL_API_KEY", DEFAULT_CONFIG["MODEL_API_KEY"])
            self.base_url = base_url or os.getenv("MODEL_BASE_URL", DEFAULT_CONFIG["MODEL_BASE_URL"])

        self.temperature = temperature
        self.agent_name = agent_name or ""

        if not self.api_key:
            raise ValueError(
                "MODEL_API_KEY environment variable is not set or is empty. "
                "Please check your .env file or set it manually:\n"
                "export MODEL_API_KEY=your_actual_api_key"
            )

        self._client = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url
        )

    def invoke(self, prompt: str) -> Any:
        """调用 LLM 生成单个响应，自动记录耗时和 token"""
        start = time.time()
        response = self._client.invoke(prompt)
        elapsed_ms = (time.time() - start) * 1000

        # Extract token usage
        usage = _extract_token_usage(response)

        # Log metrics if we have an agent name
        if self.agent_name:
            from src.utils.logger import AgentLogger
            log = AgentLogger(self.agent_name, self.model)
            log.llm_call(
                endpoint="invoke",
                duration_ms=elapsed_ms,
                tokens=usage["total_tokens"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                response_length=len(getattr(response, "content", "")),
            )

        return response

    def generate(self, prompts: list) -> Any:
        """批量生成响应"""
        start = time.time()
        result = self._client.generate(prompts)
        elapsed_ms = (time.time() - start) * 1000

        if self.agent_name:
            from src.utils.logger import AgentLogger
            log = AgentLogger(self.agent_name, self.model)
            log.llm_call(
                endpoint="generate",
                duration_ms=elapsed_ms,
                tokens=0,
                status="ok",
            )

        return result

    def stream(self, prompt: str):
        """流式调用 LLM"""
        start = time.time()
        for chunk in self._client.stream(prompt):
            yield chunk
        elapsed_ms = (time.time() - start) * 1000

        if self.agent_name:
            from src.utils.logger import AgentLogger
            log = AgentLogger(self.agent_name, self.model)
            log.llm_call(
                endpoint="stream",
                duration_ms=elapsed_ms,
                tokens=0,
                status="ok",
            )


def get_llm_client(
    model: Optional[str] = None,
    temperature: float = 0.7
) -> LlmModelClient:
    """获取 LLM 客户端实例（向后兼容）"""
    return LlmModelClient(
        model=model,
        temperature=temperature
    )


def get_default_llm_client() -> LlmModelClient:
    """获取默认的 LLM 客户端实例"""
    return LlmModelClient()


# ==================== Agent 专用 LLM 配置 ====================

def get_agent_llm_client(agent_name: str) -> LlmModelClient:
    """获取指定 Agent 的 LLM 客户端"""
    return LlmModelClient(
        agent_name=agent_name,
        temperature=0.7
    )


def get_agent_llm_client_with_temp(agent_name: str, temperature: float) -> LlmModelClient:
    """获取指定 Agent 的 LLM 客户端（自定义 temperature）"""
    return LlmModelClient(
        agent_name=agent_name,
        temperature=temperature
    )
