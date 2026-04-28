"""Session manager for managing user sessions and memory"""

import json
import os
from datetime import datetime
from typing import Optional


class Session:
    """Session class for managing user session data"""

    def __init__(self, session_id: str):
        """Initialize a session"""
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.last_activity = datetime.now().isoformat()
        self.conversation_history = []
        self.task_history = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now().isoformat()

    def add_task(self, task_description: str, result):
        """Add a task to the task history"""
        self.task_history.append({
            "task_description": task_description,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now().isoformat()

    def to_dict(self):
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "conversation_history": self.conversation_history,
            "task_history": self.task_history
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create session from dictionary"""
        session = cls(data["session_id"])
        session.created_at = data.get("created_at")
        session.last_activity = data.get("last_activity")
        session.conversation_history = data.get("conversation_history", [])
        session.task_history = data.get("task_history", [])
        return session


class SessionManager:
    """Session manager for managing user sessions"""

    def __init__(self):
        """Initialize session manager"""
        self.redis_client = None
        self.redis_enabled = False
        self.sessions = {}

        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.redis_enabled = True
        except Exception as e:
            print(f"Redis not available, using in-memory session storage: {e}")
            self.redis_enabled = False

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get or create a session"""
        if session_id in self.sessions:
            return self.sessions[session_id]

        if self.redis_enabled:
            try:
                session_data = self.redis_client.get(f"session:{session_id}")
                if session_data:
                    session = Session.from_dict(json.loads(session_data))
                    self.sessions[session_id] = session
                    return session
            except Exception as e:
                print(f"Redis error: {e}")
                self.redis_enabled = False

        session = Session(session_id)
        self.sessions[session_id] = session
        self._save_session(session)
        return session

    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if self.redis_enabled:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except Exception:
                pass

    def _save_session(self, session: Session):
        """Save session to Redis or memory"""
        if self.redis_enabled:
            try:
                self.redis_client.set(
                    f"session:{session.session_id}",
                    json.dumps(session.to_dict())
                )
                self.redis_client.expire(f"session:{session.session_id}", 86400)
            except Exception as e:
                print(f"Redis save error: {e}")
                self.redis_enabled = False

    def update_session(self, session: Session):
        """Update session"""
        self._save_session(session)