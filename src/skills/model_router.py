"""Model router for selecting appropriate models based on task and agent"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from .skill_registry import BaseSkill, SkillMetadata, SkillType, SkillRiskLevel

# Load environment variables
load_dotenv()


class ModelPriority(Enum):
    """Model priority levels"""
    PRIMARY = "primary"
    FALLBACK_1 = "fallback_1"
    FALLBACK_2 = "fallback_2"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    type: str  # "cloud", "local", "free"
    priority: ModelPriority
    cost_per_token: Optional[float] = None
    max_tokens: Optional[int] = None
    supports_long_context: bool = False
    reliability_score: float = 0.8  # 0-1 scale


class ModelRouterSkill(BaseSkill):
    """Skill for routing requests to appropriate models"""

    def __init__(self):
        metadata = SkillMetadata(
            name="model_router",
            description="Route tasks to appropriate models based on complexity and cost",
            skill_type=SkillType.RESOURCE,
            risk_level=SkillRiskLevel.LOW,
            tags=["model", "routing", "optimization"],
            execution_timeout=5
        )
        super().__init__(metadata)

        # Load model configuration from environment variables
        self.model_name = os.getenv("DEFAULT_MODEL")
        self.model_base_url = os.getenv("DEFAULT_MODEL_BASE_URL")
        self.model_api_key = os.getenv("DEFAULT_MODEL_API_KEY", "")

        # Model configurations
        self.models = self._initialize_models()

        # Agent-model routing preferences
        self.agent_preferences = self._initialize_agent_preferences()

        # Task complexity thresholds
        self.complexity_thresholds = {
            "low": {"tokens": 1000, "fallback_to_local": True},
            "medium": {"tokens": 5000, "fallback_to_local": False},
            "high": {"tokens": 10000, "fallback_to_local": False}
        }

    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize models from environment variables"""
        models = {}

        # Always add current model from .env
        env_model_type = "local" if "localhost" in self.model_base_url or "127.0.0.1" in self.model_base_url else "cloud"

        # Detect model provider from model name or base URL
        if "qwen" in self.model_name.lower() or "qwen" in self.model_base_url.lower():
            model_provider = "qwen"
        elif "glm" in self.model_name.lower():
            model_provider = "glm"
        elif "gemini" in self.model_name.lower():
            model_provider = "gemini"
        else:
            model_provider = "generic"

        # Add environment variable model as primary
        models[self.model_name] = ModelConfig(
            name=self.model_name,
            type=env_model_type,
            priority=ModelPriority.PRIMARY,
            cost_per_token=self._get_model_cost(model_provider, False),
            max_tokens=self._get_max_tokens(model_provider, False),
            supports_long_context=self._supports_long_context(model_provider, False),
            reliability_score=self._get_reliability_score(model_provider, False)
        )

        # Add fallback models based on provider
        if model_provider == "qwen":
            # Alibaba Qwen models
            models["qwen-turbo"] = ModelConfig(
                name="qwen-turbo",
                type="cloud",
                priority=ModelPriority.FALLBACK_1,
                cost_per_token=0.000002,
                max_tokens=32768,
                supports_long_context=False,
                reliability_score=0.88
            )
            models["qwen-plus"] = ModelConfig(
                name="qwen-plus",
                type="cloud",
                priority=ModelPriority.FALLBACK_2,
                cost_per_token=0.000004,
                max_tokens=32768,
                supports_long_context=True,
                reliability_score=0.92
            )
            models["qwen-max"] = ModelConfig(
                name="qwen-max",
                type="cloud",
                priority=ModelPriority.FALLBACK_2,
                cost_per_token=0.000012,
                max_tokens=32768,
                supports_long_context=True,
                reliability_score=0.96
            )
        elif model_provider == "glm":
            # Zhipu GLM models
            models["glm-4.7-flash"] = ModelConfig(
                name="glm-4.7-flash",
                type="cloud",
                priority=ModelPriority.FALLBACK_1,
                cost_per_token=0.000005,
                max_tokens=128000,
                supports_long_context=True,
                reliability_score=0.9
            )
            models["glm-4.2"] = ModelConfig(
                name="glm-4.2",
                type="cloud",
                priority=ModelPriority.FALLBACK_2,
                cost_per_token=0.00001,
                max_tokens=128000,
                supports_long_context=True,
                reliability_score=0.94
            )

        # Add local model option
        models["qwen2.5-coder-7b-instruct-local"] = ModelConfig(
            name="qwen2.5-coder-7b-instruct-local",
            type="local",
            priority=ModelPriority.FALLBACK_1,
            cost_per_token=0,
            max_tokens=32768,
            supports_long_context=False,
            reliability_score=0.85
        )

        # Add free tier option
        models["gemini-free"] = ModelConfig(
            name="gemini-free",
            type="free",
            priority=ModelPriority.FALLBACK_2,
            cost_per_token=0,
            max_tokens=32768,
            supports_long_context=False,
            reliability_score=0.8
        )

        return models

    def _initialize_agent_preferences(self) -> Dict[str, List[str]]:
        """Initialize agent-model preferences based on current model configuration"""
        # Check if we're using Qwen model
        current_model = self.model_name.lower()

        if "qwen" in current_model:
            # For Qwen models, prioritize other Qwen models
            return {
                "supervisor": [self.model_name, "qwen-max", "qwen-plus", "glm-4.7-flash", "gemini-free"],
                "repo_analyst": ["qwen-turbo", "qwen2.5-coder-7b-instruct-local", "glm-4.7-flash"],
                "implementer": [self.model_name, "qwen-plus", "qwen2.5-coder-7b-instruct-local", "glm-4.7-flash"],
                "reviewer": [self.model_name, "qwen-turbo", "gemini-free", "glm-4.7-flash"],
                "tester": ["qwen2.5-coder-7b-instruct-local", "qwen-turbo", "glm-4.7-flash", "gemini-free"]
            }
        elif "glm" in current_model:
            # For GLM models, prioritize other GLM models
            return {
                "supervisor": [self.model_name, "glm-4.7-flash", "glm-4.2", "gemini-free"],
                "repo_analyst": ["qwen2.5-coder-7b-instruct-local", "glm-4.7-flash"],
                "implementer": [self.model_name, "glm-4.7-flash", "qwen2.5-coder-7b-instruct-local"],
                "reviewer": [self.model_name, "glm-4.2", "gemini-free", "glm-4.7-flash"],
                "tester": ["qwen2.5-coder-7b-instruct-local", "glm-4.7-flash", "gemini-free"]
            }
        else:
            # Default preferences
            return {
                "supervisor": [self.model_name, "glm-4.7", "glm-4.7-flash", "gemini-free"],
                "repo_analyst": ["qwen2.5-coder-7b-instruct-local", "glm-4.7-flash"],
                "implementer": [self.model_name, "qwen2.5-coder-7b-instruct-local", "glm-4.7-flash"],
                "reviewer": [self.model_name, "glm-4.7", "gemini-free", "glm-4.7-flash"],
                "tester": ["qwen2.5-coder-7b-instruct-local", "glm-4.7-flash", "gemini-free"]
            }

    def _get_model_cost(self, provider: str, is_local: bool) -> float:
        """Get cost per token based on provider"""
        if is_local:
            return 0
        if provider == "qwen":
            return 0.000002
        elif provider == "glm":
            return 0.00001
        elif provider == "gemini":
            return 0
        return 0.00001

    def _get_max_tokens(self, provider: str, is_local: bool) -> int:
        """Get max tokens based on provider"""
        if is_local:
            return 32768
        if provider == "qwen":
            return 32768
        elif provider == "glm":
            return 128000
        elif provider == "gemini":
            return 32768
        return 128000

    def _supports_long_context(self, provider: str, is_local: bool) -> bool:
        """Check if model supports long context"""
        if is_local:
            return False
        if provider == "qwen":
            return False
        elif provider == "glm":
            return True
        elif provider == "gemini":
            return False
        return True

    def _get_reliability_score(self, provider: str, is_local: bool) -> float:
        """Get reliability score based on provider"""
        if is_local:
            return 0.85
        if provider == "qwen":
            return 0.92
        elif provider == "glm":
            return 0.95
        elif provider == "gemini":
            return 0.8
        return 0.9

    def execute(self, agent_name: str, task_description: str,
                complexity: str = "medium", context_length: int = 0) -> Dict[str, Any]:
        """
        Route task to appropriate model

        Args:
            agent_name: Name of the agent requesting the model
            task_description: Description of the task
            complexity: Task complexity (low, medium, high)
            context_length: Length of context in tokens

        Returns:
            Dictionary with model selection and reasoning
        """
        start_time = time.time()

        try:
            # Get available models for this agent
            available_models = self.agent_preferences.get(agent_name, [])

            # Filter models that are configured
            available_models = [m for m in available_models if m in self.models]

            if not available_models:
                return {
                    "success": False,
                    "error": f"No models configured for agent: {agent_name}",
                    "selected_model": None
                }

            # Score models based on various factors
            model_scores = {}
            for model_name in available_models:
                model = self.models[model_name]
                score = self._calculate_model_score(
                    model, agent_name, complexity, context_length
                )
                model_scores[model_name] = score

            # Select best model
            selected_model = max(model_scores, key=model_scores.get)
            selected_config = self.models[selected_model]

            duration = time.time() - start_time
            self.record_execution(True, duration, {
                "agent": agent_name,
                "complexity": complexity,
                "context_length": context_length,
                "selected_model": selected_model
            })

            return {
                "success": True,
                "selected_model": selected_model,
                "selected_config": {
                    "name": selected_config.name,
                    "type": selected_config.type,
                    "priority": selected_config.priority.value,
                    "cost_per_token": selected_config.cost_per_token,
                    "reliability_score": selected_config.reliability_score
                },
                "reasoning": self._generate_routing_reasoning(
                    selected_model, model_scores, complexity
                ),
                "cost_estimate": self._estimate_cost(
                    selected_config, task_description, context_length
                ),
                "duration": duration
            }

        except Exception as e:
            duration = time.time() - start_time
            self.record_execution(False, duration, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "selected_model": None
            }

    def _calculate_model_score(self, model: ModelConfig, agent_name: str,
                             complexity: str, context_length: int) -> float:
        """Calculate score for a model based on multiple factors"""
        score = 0.0

        # Base preference score
        if model.priority == ModelPriority.PRIMARY:
            score += 0.5
        elif model.priority == ModelPriority.FALLBACK_1:
            score += 0.3
        else:
            score += 0.1

        # Reliability score
        score += model.reliability_score * 0.3

        # Context length compatibility
        if context_length > 0 and model.max_tokens:
            if context_length <= model.max_tokens:
                score += 0.1
            else:
                score -= 0.2  # Penalty for insufficient context

        # Cost optimization
        if model.cost_per_token == 0:  # Free or local model
            score += 0.1

        # Agent-specific adjustments
        if agent_name == "supervisor" and model.type == "cloud":
            score += 0.1  # Supervisor benefits from cloud models
        elif agent_name in ["repo_analyst", "tester"] and model.type == "local":
            score += 0.1  # These agents can benefit from local models

        # Complexity adjustments
        if complexity == "low" and model.type == "local":
            score += 0.1
        elif complexity == "high" and model.type == "cloud":
            score += 0.1

        return max(0, min(1, score))  # Normalize to 0-1

    def _generate_routing_reasoning(self, selected_model: str, all_scores: Dict[str, float],
                                  complexity: str) -> str:
        """Generate human-readable reasoning for model selection"""
        reasoning = f"Selected {selected_model} for "

        if complexity == "low":
            reasoning += "low-complexity task - cost-effective"
        elif complexity == "medium":
            reasoning += "medium-complexity task - balanced approach"
        else:
            reasoning += "high-complexity task - prioritizes reliability"

        # Add score comparison
        max_score = max(all_scores.values())
        if max_score - all_scores[selected_model] < 0.1:
            reasoning += " (multiple models performed similarly)"

        return reasoning

    def _estimate_cost(self, model: ModelConfig, task_description: str,
                      context_length: int) -> float:
        """Estimate cost for using a model"""
        # Estimate token count based on task description and context
        estimated_tokens = len(task_description.split()) + context_length
        return estimated_tokens * model.cost_per_token if model.cost_per_token else 0

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all configured models"""
        return {
            "models": {
                name: {
                    "type": config.type,
                    "priority": config.priority.value,
                    "reliability": config.reliability_score,
                    "cost_per_token": config.cost_per_token,
                    "max_tokens": config.max_tokens
                }
                for name, config in self.models.items()
            },
            "agent_preferences": self.agent_preferences
        }

    def add_custom_model(self, model_config: ModelConfig, allowed_agents: List[str] = None):
        """Add a custom model configuration"""
        self.models[model_config.name] = model_config
        if allowed_agents:
            for agent in allowed_agents:
                if agent not in self.agent_preferences:
                    self.agent_preferences[agent] = []
                self.agent_preferences[agent].append(model_config.name)