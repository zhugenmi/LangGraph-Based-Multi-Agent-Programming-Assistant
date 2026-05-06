"""Unified memory manager integrating short-term and long-term memory"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory


class MemoryManager:
    """Unified memory manager for managing both short-term and long-term memory

    This class provides a unified interface for:
    - Session-based conversation history (short-term)
    - Persistent knowledge and vector retrieval (long-term)
    - Context retrieval for LLM prompts
    - Memory synchronization and persistence
    """

    def __init__(
        self,
        session_id: str = "default",
        user_id: str = "default",
        project_id: str = "default",
        storage_dir: Optional[str] = None,
        use_redis: bool = None
    ):
        """Initialize memory manager

        Args:
            session_id: Session identifier for short-term memory
            user_id: User identifier for long-term memory
            project_id: Project identifier for long-term memory
            storage_dir: Storage directory for long-term memory
            use_redis: Whether to use Redis (default from env USE_REDIS)
        """
        self.session_id = session_id
        self.user_id = user_id
        self.project_id = project_id
        self.storage_dir = storage_dir or os.getenv("RAG_STORAGE_DIR", "memory_store")

        # Determine whether to use Redis (from param or env)
        if use_redis is None:
            use_redis = os.getenv("USE_REDIS", "false").lower() == "true"

        # Initialize memory components
        self._short_term = ShortTermMemory(
            session_id=session_id,
            use_redis=use_redis,
            storage_dir=self.storage_dir
        )

        self._long_term = LongTermMemory(
            user_id=user_id,
            project_id=project_id,
            storage_dir=self.storage_dir
        )

        # Memory configuration
        self._config = {
            "context_max_tokens": 2000,
            "history_max_turns": 10,
            "retrieval_top_k": 5
        }

    # ==================== Conversation Management ====================

    def add_user_message(self, content: str, metadata: Optional[Dict] = None):
        """Add a user message to conversation history

        Args:
            content: Message content
            metadata: Optional metadata
        """
        return self._short_term.add_conversation_turn("user", content, metadata)

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None):
        """Add an assistant message to conversation history

        Args:
            content: Message content
            metadata: Optional metadata
        """
        return self._short_term.add_conversation_turn("assistant", content, metadata)

    def get_conversation_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history

        Args:
            limit: Maximum number of turns

        Returns:
            List of conversation turns
        """
        return self._short_term.get_conversation_history(limit)

    def get_context_for_prompt(
        self,
        max_tokens: Optional[int] = None
    ) -> str:
        """Get formatted context for LLM prompt

        Args:
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self._config["context_max_tokens"]
        return self._short_term.get_context_for_prompt(max_tokens)

    # ==================== Task State Management ====================

    def set_current_task(self, task_id: str, task_state: Dict[str, Any]):
        """Set current task state

        Args:
            task_id: Task identifier
            task_state: Task state dictionary
        """
        self._short_term.set_task_state(task_id, task_state)

    def get_current_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task state

        Args:
            task_id: Task identifier

        Returns:
            Task state dictionary
        """
        return self._short_term.get_task_state(task_id)

    def save_task_result(
        self,
        task_description: str,
        result: Dict[str, Any],
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """Save task result to long-term memory

        Args:
            task_description: Task description
            result: Task result
            success: Whether task succeeded
            metadata: Optional metadata
        """
        return self._long_term.add_historical_task(
            task_description=task_description,
            result=result,
            success=success,
            metadata=metadata
        )

    def get_similar_historical_tasks(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar historical tasks

        Args:
            task_description: Task description
            limit: Maximum results

        Returns:
            List of similar tasks
        """
        return self._long_term.get_similar_tasks(task_description, limit)

    # ==================== Knowledge Retrieval ====================

    def add_project_knowledge(
        self,
        category: str,
        key: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Add project knowledge

        Args:
            category: Knowledge category
            key: Knowledge key
            content: Knowledge content
            metadata: Optional metadata
        """
        self._long_term.add_project_knowledge(category, key, content, metadata)

    def get_project_knowledge(
        self,
        category: Optional[str] = None,
        key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get project knowledge

        Args:
            category: Optional category filter
            key: Optional key filter

        Returns:
            Project knowledge
        """
        return self._long_term.get_project_knowledge(category, key)

    # ==================== Vector Retrieval ====================

    def add_to_vector_index(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add content to vector index

        Args:
            content: Content string
            embedding: Embedding vector
            metadata: Metadata
        """
        self._long_term.add_to_vector_index(content, embedding, metadata)

    def search_vector_index(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search vector index

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            Search results
        """
        return self._long_term.search_vector_index(query_embedding, top_k)

    def get_vector_index_size(self) -> int:
        """Get vector index size"""
        return self._long_term.get_index_size()

    # ==================== Context Assembly ====================

    def assemble_context(
        self,
        query: str,
        include_history: bool = True,
        include_knowledge: bool = True,
        include_similar_tasks: bool = True,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Assemble comprehensive context for LLM

        Args:
            query: Current query
            include_history: Include conversation history
            include_knowledge: Include project knowledge
            include_similar_tasks: Include similar historical tasks
            max_tokens: Maximum total tokens

        Returns:
            Assembled context dictionary
        """
        context = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        estimated_tokens = len(query) // 4  # Rough estimation
        remaining_tokens = max_tokens - estimated_tokens

        # Conversation history
        if include_history and remaining_tokens > 0:
            history = self._short_term.get_conversation_history(
                limit=self._config["history_max_turns"]
            )
            if history:
                history_tokens = min(remaining_tokens // 3, 1000)
                history_text = self._short_term.get_context_for_prompt(history_tokens)
                context["components"]["conversation_history"] = history_text
                remaining_tokens -= len(history_text) // 4

        # Project knowledge
        if include_knowledge and remaining_tokens > 0:
            knowledge = self._long_term.get_project_knowledge()
            if knowledge:
                # Format knowledge
                kb_text = self._format_knowledge(knowledge, remaining_tokens // 3)
                context["components"]["project_knowledge"] = kb_text
                remaining_tokens -= len(kb_text) // 4

        # Similar historical tasks
        if include_similar_tasks and remaining_tokens > 0:
            similar = self._long_term.get_similar_tasks(
                query,
                limit=self._config["retrieval_top_k"]
            )
            if similar:
                tasks_text = self._format_similar_tasks(similar, remaining_tokens)
                context["components"]["similar_tasks"] = tasks_text

        return context

    def _format_knowledge(self, knowledge: Dict, max_chars: int) -> str:
        """Format knowledge for context"""
        parts = []
        total_chars = 0

        for category, items in knowledge.items():
            category_text = f"\n{category}:\n"
            for key, data in items.items():
                item_text = f"  - {key}: {data.get('content', '')[:200]}\n"
                if total_chars + len(category_text) + len(item_text) > max_chars:
                    break
                parts.append(category_text)
                parts.append(item_text)
                total_chars += len(category_text) + len(item_text)

        if parts:
            return "项目知识库:\n" + "".join(parts)
        return ""

    def _format_similar_tasks(self, tasks: List[Dict], max_chars: int) -> str:
        """Format similar tasks for context"""
        parts = []
        total_chars = 0

        for task in tasks:
            task_text = f"  - 任务: {task['task_description'][:100]}\n"
            task_text += f"    结果: {'成功' if task['success'] else '失败'}\n"

            if total_chars + len(task_text) > max_chars:
                break

            parts.append(task_text)
            total_chars += len(task_text)

        if parts:
            return "相似历史任务:\n" + "".join(parts)
        return ""

    # ==================== Preferences ====================

    def set_preference(self, key: str, value: Any):
        """Set user preference"""
        self._long_term.set_preference(key, value)

    def get_preference(self, key: str) -> Optional[Any]:
        """Get user preference"""
        return self._long_term.get_preference(key)

    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all user preferences"""
        return self._long_term.get_all_preferences()

    # ==================== Variables ====================

    def set_variable(self, name: str, value: Any):
        """Set temporary variable"""
        self._short_term.set_variable(name, value)

    def get_variable(self, name: str) -> Optional[Any]:
        """Get temporary variable"""
        return self._short_term.get_variable(name)

    # ==================== Session Management ====================

    def new_session(self, session_id: str):
        """Start a new session

        Args:
            session_id: New session identifier
        """
        self.session_id = session_id
        self._short_term = ShortTermMemory(session_id=session_id)

    def clear_session(self):
        """Clear current session memory"""
        self._short_term.clear()

    def clear_all(self):
        """Clear all memory (session + long-term)"""
        self._short_term.clear()
        self._long_term.clear_all()

    # ==================== Summary ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get memory manager summary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "short_term": self._short_term.get_summary(),
            "long_term": self._long_term.get_summary(),
            "config": self._config
        }

    def get_full_context_text(self) -> str:
        """Get full context as text for LLM prompt"""
        context = self.assemble_context("", max_tokens=4000)

        parts = []
        for component, text in context.get("components", {}).items():
            if text:
                parts.append(text)

        return "\n\n".join(parts) if parts else ""