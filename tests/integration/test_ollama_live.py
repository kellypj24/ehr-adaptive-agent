import asyncio
from src.models.ollama import OllamaClient
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama():
    async with OllamaClient() as client:
        # Test health check
        print("\nTesting health check...")
        is_healthy = await client.health_check()
        print(f"Health check result: {is_healthy}")

        # Test basic generation
        print("\nTesting basic generation...")
        prompt = "Write a Python function to calculate fibonacci numbers"
        response = await client.generate(prompt)
        print(f"Response content:\n{response.content}")
        print(f"Metadata: {response.metadata}")
        print(f"Raw response: {response.raw_response}")

        # Test with system prompt
        print("\nTesting with system prompt...")
        system_prompt = "You are a Python expert focused on clean, efficient code."
        response = await client.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=0.5, max_tokens=500
        )
        print(f"Response with system prompt:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(test_ollama())
