import asyncio
import sys
from typing import Optional
from src.models.ollama import OllamaClient
from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from datetime import datetime

console = Console()


def clean_generated_code(content: str) -> str:
    """Clean the generated code by removing markdown and explanatory text."""
    lines = content.split("\n")
    code_lines = []
    in_code_block = False

    for line in lines:
        # Skip markdown code block markers and explanatory text
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        # Only include lines that are inside a code block or look like Python code
        if in_code_block or (
            line.strip()
            and not line.startswith("Here")
            and not line.startswith("I ")
            and not line.startswith("Note")
            and not line.startswith("This")
        ):
            code_lines.append(line)

    return "\n".join(code_lines)


async def execute_generated_code(
    code: str, context: dict
) -> tuple[bool, Optional[str]]:
    """
    Execute the generated code with proper context and capture any errors.
    Returns (success, error_message)
    """
    try:
        # Clean the code before execution
        cleaned_code = clean_generated_code(code)

        # Create a new namespace with our tools
        namespace = {"FHIRClient": FHIRClient, "FHIRExplorer": FHIRExplorer, **context}

        # Execute the code in this namespace
        exec(cleaned_code, namespace)

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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for attempt in range(max_attempts):
            console.print(
                f"\n[bold cyan]Attempt {attempt + 1}/{max_attempts}[/bold cyan]"
            )

            # If this isn't the first attempt, add error context
            if attempt > 0:
                prompt = f"""Previous attempt failed with error: {last_error}

Please fix the code and try again. Here's the original task:
{prompt}"""

            try:
                # Show progress for code generation
                task_id = progress.add_task("Generating code...", total=None)
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )
                progress.remove_task(task_id)

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

                # Show progress for code execution
                task_id = progress.add_task("Executing code...", total=None)
                success, error = await execute_generated_code(response.content, context)
                progress.remove_task(task_id)

                if success:
                    console.print(
                        "[bold green]✅ Code executed successfully![/bold green]"
                    )
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        # Show progress for initialization
        task_id = progress.add_task("Initializing FHIR tools...", total=None)

        # Initialize our tools
        fhir_client = FHIRClient()
        fhir_explorer = FHIRExplorer()

        progress.remove_task(task_id)

    # Context for code execution
    context = {"client": fhir_client, "explorer": fhir_explorer}

    system_prompt = """You are a Python expert specializing in healthcare data integration.
Your task is to write executable Python code that uses the provided FHIRClient and FHIRExplorer classes.

IMPORTANT RESPONSE FORMAT:
- Only output valid Python code
- Do not include markdown code blocks (```)
- Do not include explanations or comments outside the code
- Include docstrings and inline comments within the code
- Always include a main() function that demonstrates the functionality

Available tools in context:
1. client: FHIRClient instance with methods:
   - get_patient(patient_id: str) -> dict

2. explorer: FHIRExplorer instance with methods:
   - explore_resource_structure(resource_type: str) -> dict
   - get_resource_relationships(resource_id: str, resource_type: str) -> dict

Example response format:
def main():
    '''Main function to demonstrate functionality'''
    # Your code here
    pass

if __name__ == "__main__":
    main()"""

    test_prompts = [
        "Write a program that retrieves a patient with ID 'example' using the provided client instance and prints their name and birth date. Handle any potential errors.",
        "Create a program that uses the provided explorer instance to show all available fields in a Patient resource. Format the output using rich library for better readability.",
        "Write a program that uses the provided explorer instance to get a patient's related resources and display them in a structured way using rich library.",
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
