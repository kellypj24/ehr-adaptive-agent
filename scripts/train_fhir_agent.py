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


# First define the class so it can be used in type hints
class FHIRAgentTrainer:
    def __init__(self):
        self.db_pool = None
        self.session_start_time = datetime.now()

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
    ):
        """Record the attempt in the database with reasonable field limits"""
        async with self.db_pool.acquire() as conn:
            error_type = error.split(":")[0] if error else None

            # Insert attempt record with reasonable field limits
            attempt_id = await conn.fetchval(
                """
                INSERT INTO training_attempts 
                (task, code_snippet, success, error_message, error_type, execution_time)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                task[:500],  # Limit task length
                code[:5000],  # Reasonable limit for code
                success,
                error and error[:1000],  # Reasonable limit for error message
                error_type and error_type[:100],  # Limit error type
                execution_time and f"{execution_time:.2f} seconds",
            )

            if success:
                # Update or insert learning pattern
                await conn.execute(
                    """
                    INSERT INTO learning_patterns 
                    (pattern_type, pattern_key, code_pattern, success_count)
                    VALUES ('task_solution', $1, $2, 1)
                    ON CONFLICT (pattern_type, pattern_key) 
                    DO UPDATE SET 
                        success_count = learning_patterns.success_count + 1,
                        updated_at = NOW(),
                        code_pattern = CASE 
                            WHEN learning_patterns.success_count < 3 THEN $2 
                            ELSE learning_patterns.code_pattern 
                        END
                """,
                    task[:500],
                    code[:5000],
                )
            elif error_type:
                # Track failed patterns
                await conn.execute(
                    """
                    INSERT INTO learning_patterns 
                    (pattern_type, pattern_key, code_pattern, failure_count)
                    VALUES ('error_solution', $1, $2, 1)
                    ON CONFLICT (pattern_type, pattern_key) 
                    DO UPDATE SET 
                        failure_count = learning_patterns.failure_count + 1,
                        updated_at = NOW()
                """,
                    error_type[:100],
                    code[:5000],
                )

    async def get_enhanced_prompt(self, task: str, error: Optional[str] = None) -> str:
        """Build prompt using knowledge base from database"""
        async with self.db_pool.acquire() as conn:
            prompt_parts = [task]

            # Get successful patterns for similar tasks
            similar_patterns = await conn.fetch(
                """
                SELECT code_pattern, success_count 
                FROM learning_patterns 
                WHERE pattern_type = 'task_solution'
                AND success_count > 0
                ORDER BY success_count DESC
                LIMIT 3
            """
            )

            if similar_patterns:
                prompt_parts.append("\nPrevious successful approaches:")
                for pattern in similar_patterns:
                    prompt_parts.append(
                        f"- Pattern (used {pattern['success_count']} times successfully):"
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
                """,
                    error_type,
                )

                if error_solution:
                    prompt_parts.append(f"\nPrevious solution for {error_type}:")
                    prompt_parts.append(error_solution["code_pattern"])

            return "\n".join(prompt_parts)


console = Console()


def clean_generated_code(content: str) -> str:
    """Clean the generated code by removing markdown and explanatory text."""
    lines = content.split("\n")
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or (
            line.strip()
            and not line.startswith(("Here", "I ", "Note", "This", "The", "To"))
        ):
            code_lines.append(line)

    return "\n".join(code_lines)


async def execute_generated_code(
    code: str, context: dict
) -> tuple[bool, Optional[str]]:
    """Execute the generated code with proper context and capture any errors."""
    try:
        cleaned_code = clean_generated_code(code)

        namespace = {
            "FHIRClient": FHIRClient,
            "FHIRExplorer": FHIRExplorer,
            "client": context["client"],
            "explorer": context["explorer"],
            "print": console.print,
        }

        exec(cleaned_code, namespace)

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
    max_attempts: int = 5,
    trainer: Optional[FHIRAgentTrainer] = None,
    timeout: int = 60,  # Add timeout parameter
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

                task_id = progress.add_task("Testing code...", total=None)
                try:
                    success, error = await asyncio.wait_for(
                        execute_generated_code(response.content, context),
                        timeout=30,  # Separate timeout for execution
                    )
                except TimeoutError:
                    success, error = False, "Execution timed out"
                finally:
                    progress.remove_task(task_id)

                if trainer:
                    await trainer.record_attempt(
                        task=prompt, code=response.content, success=success, error=error
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
            progress.remove_task(task_id)

        context = {"client": fhir_client, "explorer": fhir_explorer}

        system_prompt = """You are a Python expert specializing in healthcare data integration.
Your task is to write executable Python code that uses the provided client instance.

IMPORTANT:
1. The client instance is already provided as 'client'
2. Return ONLY valid Python code
3. Do not create new client instances
4. Include proper error handling
5. Use the rich library for output formatting

Example of valid code structure:
def main():
    '''Function to demonstrate FHIR client usage'''
    try:
        patient = client.get_patient('example')
        print(f"Patient data: {patient}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()"""

        prompt = "Write a program that retrieves a patient with ID 'example' using the provided client instance and prints their name and birth date. Handle any potential errors."

        async with OllamaClient() as ollama_client:
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
