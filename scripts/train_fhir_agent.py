import asyncio
import sys
from typing import Optional
from src.models.ollama import OllamaClient
from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from datetime import datetime

console = Console()


async def execute_generated_code(
    code: str, context: dict
) -> tuple[bool, Optional[str]]:
    """
    Execute the generated code with proper context and capture any errors.
    Returns (success, error_message)
    """
    try:
        # Create a new namespace with our tools
        namespace = {"FHIRClient": FHIRClient, "FHIRExplorer": FHIRExplorer, **context}

        # Execute the code in this namespace
        exec(code, namespace)

        # Try to execute the main function if it exists
        if "main" in namespace:
            namespace["main"]()

        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


async def generate_and_test_code(
    client: OllamaClient,
    prompt: str,
    system_prompt: str,
    context: dict,
    max_attempts: int = 3,
) -> None:
    """Generate code and attempt to execute it, learning from failures"""

    for attempt in range(max_attempts):
        console.print(f"\n[bold cyan]Attempt {attempt + 1}/{max_attempts}[/bold cyan]")

        # If this isn't the first attempt, add error context to the prompt
        if attempt > 0:
            prompt = f"""Previous attempt failed with error: {last_error}

Please fix the code and try again. Here's the original task:
{prompt}"""

        try:
            response = await client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1000,
            )

            # Display the generated code
            code_syntax = Syntax(
                response.content,
                "python",
                theme="monokai",
                line_numbers=True,
            )

            console.print(
                Panel(
                    code_syntax,
                    title=f"Generated Code (Attempt {attempt + 1})",
                    border_style="green",
                )
            )

            # Try to execute the code
            console.print("\n[bold yellow]Executing code...[/bold yellow]")
            success, error = await execute_generated_code(response.content, context)

            if success:
                console.print("[bold green]✅ Code executed successfully![/bold green]")
                return
            else:
                console.print(f"[bold red]❌ Execution failed:[/bold red] {error}")
                last_error = error

        except Exception as e:
            console.print(f"[bold red]Error during generation:[/bold red] {str(e)}")
            last_error = str(e)

        if attempt < max_attempts - 1:
            console.print("\n[yellow]Retrying with error context...[/yellow]")
        else:
            console.print("\n[red]Max attempts reached. Moving to next task.[/red]")


async def main():
    console.print("[bold green]Starting FHIR Agent Training Session[/bold green]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize our tools
    fhir_client = FHIRClient()
    fhir_explorer = FHIRExplorer()

    # Context for code execution
    context = {"client": fhir_client, "explorer": fhir_explorer}

    system_prompt = """You are a Python expert specializing in healthcare data integration.
Write clean, well-documented code that can be executed immediately.
Include type hints and docstrings.
Your code should be a complete program with a main() function that demonstrates the functionality.
Handle errors appropriately and include proper error messages."""

    test_prompts = [
        "Write a program that retrieves a patient with ID 'example' using FHIRClient and prints their name and birth date. Use proper error handling.",
        "Create a program that uses FHIRExplorer to show all available fields in a Patient resource, with proper formatting of the output.",
        "Write a program that gets a patient's related resources using FHIRExplorer and prints them in a structured way.",
    ]

    async with OllamaClient() as client:
        is_healthy = await client.health_check()
        console.print(f"Health check: {'✅' if is_healthy else '❌'}")

        if not is_healthy:
            console.print("[bold red]Model is not available. Exiting.[/bold red]")
            sys.exit(1)

        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"\n[bold blue]Task #{i}[/bold blue]")
            console.print(Panel(prompt, title="Prompt", border_style="blue"))

            await generate_and_test_code(
                client=client,
                prompt=prompt,
                system_prompt=system_prompt,
                context=context,
            )

            console.print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
