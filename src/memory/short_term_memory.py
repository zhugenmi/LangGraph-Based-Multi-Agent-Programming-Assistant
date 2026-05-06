"""Short-term memory for managing session context and conversation history"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime
from pathlib import Path


class ShortTermMemory:
    """Short-term memory for session context and recent conversation history

    Features:
    - Conversation history (last N turns)
    - Current task state tracking
    - Temporary variables and intermediate results
    - Configurable storage backend (Redis, file, or in-memory)
    """

    DEFAULT_MAX_HISTORY = 50
    DEFAULT_MAX_CONTEXT = 10
    DEFAULT_STORAGE_DIR = "memory_store"

    def __init__(
        self,
        session_id: str,
        max_history: int = None,
        max_context: int = None,
        use_redis: bool = None,
        storage_dir: str = None
    ):
        """Initialize short-term memory

        Args:
            session_id: Unique session identifier
            max_history: Maximum number of conversation turns to keep
            max_context: Maximum number of context items to track
            use_redis: Whether to use Redis for persistence (default from env USE_REDIS)
            storage_dir: Directory for file-based storage
        """
        self.session_id = session_id
        self.max_history = max_history or self.DEFAULT_MAX_HISTORY
        self.max_context = max_context or self.DEFAULT_MAX_CONTEXT
        self.storage_dir = storage_dir or os.getenv("RAG_STORAGE_DIR", self.DEFAULT_STORAGE_DIR)

        # Determine whether to use Redis (from param or env)
        if use_redis is None:
            use_redis = os.getenv("USE_REDIS", "false").lower() == "true"

        # In-memory storage
        self._conversation_history: deque = deque(maxlen=self.max_history)
        self._task_states: Dict[str, Any] = {}
        self._context_items: deque = deque(maxlen=self.max_context)
        self._variables: Dict[str, Any] = {}

        # Storage backend
        self._redis_client = None
        self._use_redis = False
        self._use_file = False

        # Try to initialize storage
        if use_redis:
            self._init_redis()

        # If Redis not available, use file storage
        if not self._use_redis:
            self._init_file_storage()

        # Load existing data
        self._load_data()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = redis.from_url(redis_url)
            self._redis_client.ping()
            self._use_redis = True
            self._use_file = False
        except Exception as e:
            print(f"Redis initialization failed: {e}, using file storage")
            self._use_redis = False
            self._redis_client = None

    def _init_file_storage(self):
        """Initialize file-based storage"""
        try:
            self._storage_path = Path(self.storage_dir)
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._use_file = True
            print(f"Using file storage at: {self._storage_path}")
        except Exception as e:
            print(f"File storage initialization failed: {e}")
            self._use_file = False

    def _get_storage_file(self) -> Path:
        """Get storage file path for this session"""
        safe_session_id = self.session_id.replace("/", "_").replace(":", "_")
        return self._storage_path / f"{safe_session_id}.json"

    def _load_data(self):
        """Load data from storage"""
        if self._use_redis and self._redis_client:
            self._load_from_redis()
        elif self._use_file:
            self._load_from_file()

    def _load_from_redis(self):
        """Load data from Redis"""
        try:
            key = self._get_redis_key("conversation")
            data = self._redis_client.get(key)
            if data:
                history = json.loads(data)
                self._conversation_history = deque(history, maxlen=self.max_history)

            task_key = self._get_redis_key("tasks")
            data = self._redis_client.get(task_key)
            if data:
                self._task_states = json.loads(data)
        except Exception:
            pass

    def _load_from_file(self):
        """Load data from file"""
        try:
            file_path = self._get_storage_file()
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'conversation' in data:
                        self._conversation_history = deque(data['conversation'], maxlen=self.max_history)
                    if 'tasks' in data:
                        self._task_states = data['tasks']
                    if 'variables' in data:
                        self._variables = data['variables']
        except Exception as e:
            print(f"Failed to load from file: {e}")

    def _save_data(self):
        """Save data to storage"""
        if self._use_redis and self._redis_client:
            self._save_to_redis()
        elif self._use_file:
            self._save_to_file()

    def _save_to_redis(self):
        """Save data to Redis"""
        try:
            # Save conversation
            history = list(self._conversation_history)
            key = self._get_redis_key("conversation")
            self._redis_client.set(key, json.dumps(history))
            self._redis_client.expire(key, 3600)

            # Save task states
            task_key = self._get_redis_key("tasks")
            self._redis_client.set(task_key, json.dumps(self._task_states))
            self._redis_client.expire(task_key, 3600)
        except Exception:
            pass

    def _save_to_file(self):
        """Save data to file"""
        try:
            file_path = self._get_storage_file()
            data = {
                'conversation': list(self._conversation_history),
                'tasks': self._task_states,
                'variables': self._variables,
                'updated_at': datetime.now().isoformat()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save to file: {e}")

    def _get_redis_key(self, key: str) -> str:
        """Get Redis key with session prefix"""
        return f"stm:{self.session_id}:{key}"

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a conversation turn to history

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (tokens, model, etc.)

        Returns:
            The stored turn data
        """
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self._conversation_history.append(turn)

        # Save to storage
        self._save_data()

        return turn

    def get_conversation_history(
        self,
        limit: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history

        Args:
            limit: Maximum number of turns to return
            role_filter: Filter by role ("user" or "assistant")

        Returns:
            List of conversation turns
        """
        history = list(self._conversation_history)

        if role_filter:
            history = [h for h in history if h["role"] == role_filter]

        if limit:
            history = history[-limit:]

        return history

    def get_last_turn(self) -> Optional[Dict[str, Any]]:
        """Get the last conversation turn"""
        if self._conversation_history:
            return self._conversation_history[-1]
        return None

    def get_context_for_prompt(self, max_tokens: int = 2000) -> str:
        """Format conversation history for LLM prompt

        Args:
            max_tokens: Approximate token limit for context

        Returns:
            Formatted context string
        """
        context_parts = []
        estimated_tokens = 0

        # Go through history in reverse order (most recent first)
        for turn in reversed(list(self._conversation_history)):
            # Rough token estimation (4 chars per token)
            turn_tokens = len(turn["content"]) // 4

            if estimated_tokens + turn_tokens > max_tokens:
                break

            role_label = "用户" if turn["role"] == "user" else "助手"
            context_parts.insert(0, f"{role_label}: {turn['content']}")
            estimated_tokens += turn_tokens

        if context_parts:
            return "\n\n对话历史:\n" + "\n".join(context_parts)
        return ""

    def set_task_state(self, task_id: str, state: Dict[str, Any]):
        """Set task state

        Args:
            task_id: Task identifier
            state: Task state dictionary
        """
        state["updated_at"] = datetime.now().isoformat()
        self._task_states[task_id] = state

        # Save to storage
        self._save_data()

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task state

        Args:
            task_id: Task identifier

        Returns:
            Task state dictionary or None
        """
        return self._task_states.get(task_id)

    def get_all_task_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all task states"""
        return dict(self._task_states)

    def add_context_item(self, key: str, value: Any, source: str = "unknown"):
        """Add a context item

        Args:
            key: Context item key
            value: Context item value
            source: Source of the context (rag, memory, user, etc.)
        """
        item = {
            "key": key,
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        self._context_items.append(item)

    def get_context_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get context items

        Args:
            limit: Maximum number of items to return

        Returns:
            List of context items
        """
        items = list(self._context_items)
        if limit:
            items = items[-limit:]
        return items

    def set_variable(self, name: str, value: Any):
        """Set a temporary variable

        Args:
            name: Variable name
            value: Variable value
        """
        self._variables[name] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

        # Save to storage
        self._save_data()

    def get_variable(self, name: str) -> Optional[Any]:
        """Get a temporary variable

        Args:
            name: Variable name

        Returns:
            Variable value or None
        """
        var_data = self._variables.get(name)
        if var_data:
            return var_data["value"]
        return None

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables"""
        return {k: v["value"] for k, v in self._variables.items()}

    def clear(self):
        """Clear all short-term memory"""
        self._conversation_history.clear()
        self._task_states.clear()
        self._context_items.clear()
        self._variables.clear()

        if self._use_redis and self._redis_client:
            try:
                # Delete all keys for this session
                pattern = self._get_redis_key("*")
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return {
            "session_id": self.session_id,
            "conversation_turns": len(self._conversation_history),
            "task_states": len(self._task_states),
            "context_items": len(self._context_items),
            "variables": len(self._variables),
            "storage_backend": "redis" if self._use_redis else "memory",
            "last_activity": datetime.now().isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "session_id": self.session_id,
            "conversation_history": list(self._conversation_history),
            "task_states": dict(self._task_states),
            "context_items": list(self._context_items),
            "variables": dict(self._variables),
            "max_history": self.max_history,
            "max_context": self.max_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortTermMemory":
        """Deserialize from dictionary"""
        stm = cls(
            session_id=data["session_id"],
            max_history=data.get("max_history", cls.DEFAULT_MAX_HISTORY),
            max_context=data.get("max_context", cls.DEFAULT_MAX_CONTEXT)
        )

        for turn in data.get("conversation_history", []):
            stm._conversation_history.append(turn)

        stm._task_states = data.get("task_states", {})

        for item in data.get("context_items", []):
            stm._context_items.append(item)

        stm._variables = data.get("variables", {})

        return stm