import asyncio
from src.models.ollama import OllamaClient
from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from datetime import datetime

console = Console()


async def test_fhir_interaction():
    async with OllamaClient() as client:
        console.print("\n[bold blue]Initializing FHIR Agent Test[/bold blue]")

        is_healthy = await client.health_check()
        console.print(f"Health check: {'✅' if is_healthy else '❌'}")

        system_prompt = """You are a Python expert specializing in healthcare data integration.
Write clean, well-documented code to solve the given task.
Include type hints and docstrings.
Do not include any explanations outside the code."""

        test_prompts = [
            "Write a Python function that retrieves a patient by ID and prints their name and birth date",
            "Create a function that uses FHIRExplorer to show all available fields in a Patient resource",
            "Write a function that gets a patient's related resources using FHIRExplorer",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"\n[bold yellow]Test #{i}[/bold yellow]")
            console.print(Panel(prompt, title="Prompt", border_style="blue"))

            try:
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )

                # Format the code with syntax highlighting
                code_syntax = Syntax(
                    response.content,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                )

                console.print(
                    Panel(code_syntax, title="Generated Code", border_style="green")
                )

                # Print metadata in a clean format
                console.print(
                    Panel(
                        "\n".join(
                            f"[bold]{k}:[/bold] {v}"
                            for k, v in response.metadata.items()
                        ),
                        title="Metadata",
                        border_style="yellow",
                    )
                )

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

            # Add a separator between tests
            console.print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    console.print("[bold green]Starting FHIR Agent Training Session[/bold green]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    asyncio.run(test_fhir_interaction())
