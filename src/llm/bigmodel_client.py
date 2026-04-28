"""LLM Client for bigmodel integration"""

import os
from typing import Optional, Any
from langchain_openai import ChatOpenAI


class BigModelClient:
    """BigModel LLM Client compatible with OpenAI API format"""

    def __init__(
        self,
        model: str = "glm-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize BigModel Client

        Args:
            model: Model name (default: glm-4)
            temperature: Sampling temperature (default: 0.7)
            api_key: API key for bigmodel
            base_url: Base URL for bigmodel API
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("BIGMODEL_API_KEY", "") or os.environ.get("BIGMODEL_API_KEY", "")
        self.base_url = base_url or os.getenv("BIGMODEL_BASE_URL", "") or os.environ.get("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

        if not self.api_key:
            raise ValueError("BIGMODEL_API_KEY environment variable is not set. Please check your .env file.")

        self._client = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url
        )

    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt"""
        return self._client.invoke(prompt)

    def generate(self, prompts: list) -> Any:
        """Generate responses for multiple prompts"""
        return self._client.generate(prompts)


def get_llm_client(
    model: str = "glm-4",
    temperature: float = 0.7
) -> BigModelClient:
    """Get a configured BigModel client instance

    Args:
        model: Model name
        temperature: Sampling temperature

    Returns:
        BigModelClient instance
    """
    api_key = os.getenv("BIGMODEL_API_KEY") or os.environ.get("BIGMODEL_API_KEY", "")
    base_url = os.getenv("BIGMODEL_BASE_URL") or os.environ.get("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    return BigModelClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )