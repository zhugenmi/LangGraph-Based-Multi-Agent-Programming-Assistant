"""Embedding client for generating text embeddings

Configuration in .env:

"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration from environment variables

    Returns:
        config dict with:
        - model: model name
        - base_url: API endpoint (local or remote)
        - api_key: API key (optional for some local services)
        - dimension: embedding dimension
        - use_local_embedding: whether to use local ollama embedding
    """
    config = {
        "model": os.getenv("EMBEDDING_MODEL", "bge-m3"),
        "base_url": os.getenv("EMBEDDING_MODEL_BASE_URL", "http://localhost:11434"),
        "api_key": os.getenv("EMBEDDING_MODEL_API_KEY", ""),
        "dimension": int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        "use_local_embedding": os.getenv("USE_LOCAL_EMBEDDING", "true").lower() == "true"
    }
    return config


class EmbeddingClient:
    """Client for generating text embeddings

    Architecture:
    - Primary: Use EMBEDDING_MODEL_BASE_URL as API endpoint
    - Fallback: Use sentence-transformers for local embedding (if enabled)

    """

    DEFAULT_MODEL = "text-embedding-ada-002"
    DEFAULT_DIMENSION = 1536

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        use_local_embedding: Optional[bool] = None
    ):
        """Initialize embedding client

        Args:
            model: Model name (default from env EMBEDDING_MODEL)
            base_url: API endpoint (default from env EMBEDDING_MODEL_BASE_URL)
            api_key: API key (default from env EMBEDDING_MODEL_API_KEY)
            dimension: Embedding dimension (default from env EMBEDDING_DIMENSION)
            use_local_embedding: Use local ollama embedding model directly
        """
        env_config = get_embedding_config()

        # API configuration (primary)
        self.model = model or env_config["model"]
        self.base_url = base_url or env_config["base_url"]
        self.api_key = api_key or env_config["api_key"]
        self.dimension = dimension or env_config["dimension"]

        # Local embedding configuration
        self.use_local_embedding = use_local_embedding if use_local_embedding is not None else env_config["use_local_embedding"]

        # Initialize local model (ollama) if use_local_embedding is True
        self._ollama_available = False

        if self.use_local_embedding:
            self._init_ollama_model()

    def _init_ollama_model(self):
        """Initialize local ollama model"""
        import requests
        try:
            # Check if ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self._ollama_available = True
                print(f"Local ollama model ready: {self.model} (base_url: {self.base_url})")
            else:
                self._ollama_available = False
                print(f"Note: Ollama service not available")
        except Exception as e:
            self._ollama_available = False
            print(f"Note: Failed to connect to ollama: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # If use_local_embedding is True, use local ollama
        if self.use_local_embedding:
            if self._ollama_available:
                return self._embed_ollama(text)
            else:
                raise RuntimeError("Local ollama model not available")

        # Otherwise use API
        return self._embed_api(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # If use_local_embedding is True, use local ollama
        if self.use_local_embedding:
            if self._ollama_available:
                return self._embed_batch_ollama(texts)
            else:
                raise RuntimeError("Local ollama model not available")

        # Otherwise use API
        return self._embed_batch_api(texts)

    def _embed_ollama(self, text: str) -> List[float]:
        """Generate embedding using local ollama"""
        import requests
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result["embedding"]
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def _embed_batch_ollama(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local ollama in batch"""
        import requests
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                embeddings.append(result["embedding"])
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
        return embeddings

    def _embed_api(self, text: str) -> List[float]:
        """Generate embedding using API"""
        if not self.api_key:
            # Fallback to simple hash-based embedding for demo
            return self._fallback_embedding(text)

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # OpenAI-compatible API format
            endpoint = f"{self.base_url}/embeddings"
            payload = {
                "input": text,
                "model": self.model
            }

            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                print(f"Embedding API error: {response.status_code}")
                return self._fallback_embedding(text)

        except Exception as e:
            print(f"Embedding API request failed: {e}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding using simple hash-based method

        This is a simple deterministic embedding for demonstration
        when no API is available. Not suitable for real semantic search.
        """
        import hashlib

        # Generate deterministic embedding from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to float vector
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 2), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
            embedding.append(value)

        # Pad to dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)

        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding[:self.dimension]

    def embed_with_metadata(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embedding with metadata

        Args:
            text: Text to embed
            metadata: Additional metadata

        Returns:
            Dictionary with embedding and metadata
        """
        embedding = self.embed_text(text)

        return {
            "text": text,
            "embedding": embedding,
            "dimension": len(embedding),
            "model": self.model,
            "metadata": metadata
        }

    def batch_embed(
        self,
        texts_with_metadata: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Batch embed texts with metadata

        Args:
            texts_with_metadata: List of dicts with 'text' and 'metadata' keys
            batch_size: Batch size for API calls

        Returns:
            List of embedding results with metadata
        """
        results = []

        for i in range(0, len(texts_with_metadata), batch_size):
            batch = texts_with_metadata[i:i+batch_size]
            texts = [item["text"] for item in batch]

            embeddings = self.embed_texts(texts)

            for j, item in enumerate(batch):
                results.append({
                    "text": item["text"],
                    "embedding": embeddings[j],
                    "dimension": len(embeddings[j]),
                    "model": self.model,
                    "metadata": item.get("metadata", {})
                })

        return results

    def get_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_config(self) -> Dict[str, Any]:
        """Get client configuration"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "use_local_embedding": self.use_local_embedding,
            "base_url": self.base_url
        }


def get_embedding_client(
    use_local_embedding: bool = True,
    model: Optional[str] = None
) -> EmbeddingClient:
    """Get embedding client instance

    Args:
        use_local_embedding: Whether to use local embedding model directly
        model: Model name

    Returns:
        EmbeddingClient instance
    """
    return EmbeddingClient(use_local_embedding=use_local_embedding, model=model)