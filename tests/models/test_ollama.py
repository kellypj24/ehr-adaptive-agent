import pytest
import httpx
from src.models.ollama import OllamaClient, ModelServiceError
import pytest_asyncio
from typing import AsyncGenerator


@pytest_asyncio.fixture
async def ollama_client() -> AsyncGenerator[OllamaClient, None]:
    client = OllamaClient()
    yield client
    await client.client.aclose()


@pytest.mark.asyncio
async def test_health_check_success(ollama_client):
    """Test successful health check with mock response"""
    is_healthy = await ollama_client.health_check()
    assert is_healthy == True


@pytest.mark.asyncio
async def test_generate_basic(ollama_client):
    """Test basic generation with mock response"""
    prompt = "Write a function to calculate fibonacci numbers"
    response = await ollama_client.generate(prompt)

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_generate_with_system_prompt(ollama_client):
    """Test generation with system prompt"""
    system_prompt = "You are a healthcare data expert."
    prompt = "What is FHIR?"

    response = await ollama_client.generate(prompt=prompt, system_prompt=system_prompt)

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_generate_with_parameters(ollama_client):
    """Test generation with custom parameters"""
    prompt = "Write a hello world program"
    params = {"temperature": 0.5, "max_tokens": 100}
    response = await ollama_client.generate(prompt=prompt, **params)

    assert isinstance(response.content, str)
    assert len(response.content) > 0
