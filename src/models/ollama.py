from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel
import json


class ModelResponse(BaseModel):
    """Standardized response from any model implementation"""

    content: str
    metadata: Dict[str, Any]
    raw_response: Dict[str, Any]


class ModelServiceError(Exception):
    """Base exception for model service related errors"""

    pass


class OllamaClient:
    """Client for interacting with locally running Ollama instance"""

    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "deepseek-coder"
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ModelResponse:
        """Generate text using the Ollama model"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt if system_prompt else "",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()

            # Handle streaming response - accumulate all chunks
            full_response = ""
            last_data = None

            for line in response.content.decode().split("\n"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    full_response += data.get("response", "")
                    last_data = data
                except json.JSONDecodeError:
                    continue

            return ModelResponse(
                content=full_response,
                metadata={
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                raw_response=last_data or {},  # Store the last chunk for metadata
            )
        except httpx.RequestError as e:
            raise ModelServiceError(f"Failed to generate: {str(e)}")
