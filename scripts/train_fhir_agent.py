import asyncio
from src.models.ollama import OllamaClient
from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer


async def test_fhir_interaction():
    async with OllamaClient() as client:
        print("\nVerifying model connection...")
        is_healthy = await client.health_check()
        print(f"Health check result: {is_healthy}")

        system_prompt = """You are a Python expert. Write code to solve the given task.
Do not include any explanations, only output valid Python code."""

        test_prompts = [
            "Write a Python function that prints 'Hello World'",  # Simple test first
            "Write code to get a patient with ID 'example' using FHIRClient",
        ]

        for prompt in test_prompts:
            print(f"\n\nTesting prompt: {prompt}")
            print("-" * 80)

            try:
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )

                print("\nGenerated Code:")
                print("-" * 40)
                print(response.content)
                print("-" * 40)
                print(f"Metadata: {response.metadata}")

            except Exception as e:
                print(f"\nError generating code: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_fhir_interaction())
