import asyncio
import json
import os
from typing import Optional, Dict, List, TYPE_CHECKING
from pathlib import Path
from src.models.ollama import OllamaClient
from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from datetime import datetime
import asyncpg
from asyncio import TimeoutError
import hashlib


# First define the class so it can be used in type hints
class FHIRAgentTrainer:
    def __init__(self):
        self.db_pool = None
        self.session_start_time = datetime.now()
        self.tools_dir = Path("ai_generated_code/tools")
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize database connection pool"""
        self.db_pool = await asyncpg.create_pool(
            user="postgres",
            password="postgres",  # Should be in env vars in production
            database="ehr_agent",
            host="localhost",
        )

    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()

    async def record_attempt(
        self,
        task: str,
        code: str,
        success: bool,
        error: Optional[str] = None,
        execution_time: Optional[float] = None,
        file_location: Optional[str] = None,
    ):
        """Record the attempt in the database with version tracking"""
        async with self.db_pool.acquire() as conn:
            error_type = error.split(":")[0] if error else None

            # Get task_hash for consistent identification
            task_hash = hashlib.md5(task.encode()).hexdigest()[:8]

            # Get current version number for this task
            current_version = (
                await conn.fetchval(
                    """
                SELECT MAX(version) + 1 
                FROM training_attempts 
                WHERE task_hash = $1
                """,
                    task_hash,
                )
                or 1
            )

            # Insert attempt record with version tracking
            attempt_id = await conn.fetchval(
                """
                INSERT INTO training_attempts 
                (task, task_hash, version, code_snippet, success, error_message, 
                 error_type, execution_time, file_location)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                task[:500],
                task_hash,
                current_version,
                code[:5000],
                success,
                error and error[:1000],
                error_type and error_type[:100],
                execution_time and f"{execution_time:.2f} seconds",
                file_location,
            )

            if success:
                # Update best version if this attempt was successful
                await conn.execute(
                    """
                    INSERT INTO task_best_versions (task_hash, best_version)
                    VALUES ($1, $2)
                    ON CONFLICT (task_hash) 
                    DO UPDATE SET best_version = $2
                    WHERE task_best_versions.task_hash = $1
                    """,
                    task_hash,
                    current_version,
                )

    async def get_enhanced_prompt(self, task: str, error: Optional[str] = None) -> str:
        """Build prompt using knowledge base and previous versions"""
        async with self.db_pool.acquire() as conn:
            task_hash = hashlib.md5(task.encode()).hexdigest()[:8]

            # Get the best version of this specific task if it exists
            best_version = await conn.fetchrow(
                """
                SELECT ta.code_snippet, ta.version
                FROM training_attempts ta
                JOIN task_best_versions tbv ON ta.task_hash = tbv.task_hash
                WHERE ta.task_hash = $1 AND ta.version = tbv.best_version
                """,
                task_hash,
            )

            prompt_parts = [
                "TASK DESCRIPTION:",
                task,
                "\nREQUIREMENTS:",
                "1. Write clean, focused code with minimal comments",
                "2. Use the provided 'client' and 'explorer' instances - do not create new ones",
                "3. Include proper error handling using try/except",
                "4. Use the rich library's console for output formatting",
                "5. Structure code with a main() function",
                "6. Return only working, executable code",
            ]

            if best_version:
                prompt_parts.extend(
                    [
                        "\nPREVIOUS BEST VERSION:",
                        f"# Version {best_version['version']} - Use this as a starting point and improve it:",
                        best_version["code_snippet"],
                    ]
                )

            # Get successful patterns for similar tasks
            similar_patterns = await conn.fetch(
                """
                SELECT code_pattern, success_count 
                FROM learning_patterns 
                WHERE pattern_type = 'task_solution'
                AND success_count > 2  -- Only use patterns that worked multiple times
                ORDER BY success_count DESC
                LIMIT 2  -- Reduced from 3 to keep context focused
                """
            )

            if similar_patterns:
                prompt_parts.append("\nEXAMPLE PATTERNS THAT WORKED:")
                for pattern in similar_patterns:
                    prompt_parts.append(
                        f"\n# Pattern used successfully {pattern['success_count']} times:"
                    )
                    prompt_parts.append(pattern["code_pattern"])

            # Get solution for specific error type if it exists
            if error:
                error_type = error.split(":")[0]
                error_solution = await conn.fetchrow(
                    """
                    SELECT code_pattern 
                    FROM learning_patterns 
                    WHERE pattern_type = 'error_solution'
                    AND pattern_key = $1
                    AND success_count > failure_count
                    AND success_count > 1  -- Only use if it worked multiple times
                    """,
                    error_type,
                )

                if error_solution:
                    prompt_parts.append(f"\nPATTERN TO FIX {error_type}:")
                    prompt_parts.append(error_solution["code_pattern"])

            return "\n".join(prompt_parts)

    async def save_generated_code(self, code: str, task: str) -> Path:
        """Save generated code to a file with consistent naming"""
        # Create a consistent hash for the task
        task_hash = hashlib.md5(task.encode()).hexdigest()[:8]

        # Get current version
        async with self.db_pool.acquire() as conn:
            version = (
                await conn.fetchval(
                    """
                SELECT MAX(version) + 1 
                FROM training_attempts 
                WHERE task_hash = $1
                """,
                    task_hash,
                )
                or 1
            )

        # Create a simplified name from the task
        task_words = task.lower().split()[:5]
        file_name = f"{'_'.join(task_words)}_{task_hash}_v{version}.py"

        file_path = self.tools_dir / file_name

        # Add imports and context setup
        full_code = f"""from src.tools.fhir_tools.client import FHIRClient
from src.tools.fhir_tools.explorer import FHIRExplorer
from rich.console import Console

console = Console()

{code}

if __name__ == "__main__":
    main()
"""

        file_path.write_text(full_code)
        return file_path


console = Console()


def clean_generated_code(content: str) -> str:
    """Clean the generated code by removing markdown and explanatory text."""
    lines = content.split("\n")
    code_lines = []
    in_code_block = False

    for line in lines:
        # Skip explanatory text and markdown
        if any(
            line.lower().startswith(text)
            for text in [
                "sure,",
                "here",
                "this",
                "the",
                "note:",
                "example",
                "i ",
                "let's",
                "now",
                "first,",
                "next,",
                "finally,",
                "```python",
                "```",
            ]
        ):
            continue

        # Skip empty lines at the start
        if not code_lines and not line.strip():
            continue

        # Skip lines that look like explanations
        if line.strip() and not line.strip().startswith("#"):
            if line[0].isupper() and "." in line:
                continue

        code_lines.append(line)

    # Clean up any trailing whitespace
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    return "\n".join(code_lines)


async def execute_generated_code(
    file_path: Path, context: dict
) -> tuple[bool, Optional[str]]:
    """Execute the generated code from file with proper context"""
    try:
        # Import the generated module
        import importlib.util

        spec = importlib.util.spec_from_file_location("generated_tool", file_path)
        module = importlib.util.module_from_spec(spec)

        # Inject context
        module.client = context["client"]
        module.explorer = context["explorer"]

        # Execute the module
        spec.loader.exec_module(module)

        if hasattr(module, "main"):
            module.main()

        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


async def generate_and_test_code(
    client: OllamaClient,
    prompt: str,
    system_prompt: str,
    context: dict,
    max_attempts: int = 5,
    trainer: Optional[FHIRAgentTrainer] = None,
    timeout: int = 60,
) -> None:
    """Generate and test code with better progress feedback and timeout"""
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

            try:
                task_id = progress.add_task("Generating code...", total=None)
                # Wrap generation in timeout
                try:
                    response = await asyncio.wait_for(
                        client.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=0.2,
                            max_tokens=1000,
                        ),
                        timeout=timeout,
                    )
                except TimeoutError:
                    console.print(
                        "[bold red]Generation timed out after {} seconds[/bold red]".format(
                            timeout
                        )
                    )
                    continue
                except Exception as e:
                    console.print(f"[bold red]Generation failed:[/bold red] {str(e)}")
                    continue
                finally:
                    progress.remove_task(task_id)

                # Save generated code to file
                task_id = progress.add_task("Saving code...", total=None)
                try:
                    file_path = await trainer.save_generated_code(
                        code=clean_generated_code(response.content), task=prompt
                    )
                    console.print(f"[dim]Code saved to: {file_path}[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error saving code:[/bold red] {str(e)}")
                    continue
                finally:
                    progress.remove_task(task_id)

                # Execute from file
                task_id = progress.add_task("Testing code...", total=None)
                try:
                    success, error = await asyncio.wait_for(
                        execute_generated_code(file_path, context),
                        timeout=30,  # Separate timeout for execution
                    )
                except TimeoutError:
                    success, error = False, "Execution timed out"
                finally:
                    progress.remove_task(task_id)

                # Record attempt with file location
                if trainer:
                    await trainer.record_attempt(
                        task=prompt,
                        code=response.content,
                        success=success,
                        error=error,
                        execution_time=30,  # Assuming 30 seconds for execution
                        file_location=str(file_path),
                    )

                if success:
                    console.print("[bold green]✅ Success![/bold green]")
                    return
                else:
                    console.print(f"[bold red]❌ Error:[/bold red] {error}")

            except Exception as e:
                console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")

            if attempt < max_attempts - 1:
                console.print("[yellow]Retrying...[/yellow]")
                await asyncio.sleep(1)  # Brief pause between attempts


async def main():
    trainer = FHIRAgentTrainer()
    await trainer.initialize()

    try:
        console.print("[bold green]Starting FHIR Agent Training Session[/bold green]")
        console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Initializing FHIR tools...", total=None)
            fhir_client = FHIRClient()
            fhir_explorer = FHIRExplorer()
            ollama_client = OllamaClient()
            progress.remove_task(task_id)

        context = {"client": fhir_client, "explorer": fhir_explorer}

        system_prompt = """You are a Python code generator. You MUST output ONLY valid Python code - no explanations, no markdown, no text.

Available instances:
- client: FHIRClient instance for FHIR API interactions
- explorer: FHIRExplorer instance for metadata exploration
- console: Rich console instance for formatted output

Example output format:
def main():
    try:
        result = client.get_resource('Patient', 'example')
        console.print(f"[green]Success:[/green] {result}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    main()"""

        # Test prompts
        prompts = [
            "Write a program that retrieves a patient with ID 'example' and prints their name and birth date.",
            "Create a function to search for all patients with a given family name.",
            "Write code to retrieve and display a patient's latest observation results.",
        ]

        for prompt in prompts:
            console.print(f"\n[bold cyan]Testing prompt:[/bold cyan] {prompt}")
            await generate_and_test_code(
                client=ollama_client,
                prompt=prompt,
                system_prompt=system_prompt,
                context=context,
                max_attempts=5,
                trainer=trainer,
            )

    finally:
        await trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
