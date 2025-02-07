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


# First define the class so it can be used in type hints
class FHIRAgentTrainer:
    def __init__(self):
        self.knowledge_base_path = Path("training_history")
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.session_history: List[Dict] = []
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load previous training sessions and successful solutions"""
        history_file = self.knowledge_base_path / "training_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {
                "successful_patterns": {},
                "error_solutions": {},
                "task_history": [],
            }

    def save_training_session(self):
        """Save the current session's learnings"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save session history
        session_file = self.knowledge_base_path / f"session_{timestamp}.json"
        with open(session_file, "w") as f:
            json.dump(self.session_history, f, indent=2)

        # Update and save knowledge base
        self.knowledge_base["task_history"].extend(self.session_history)
        with open(self.knowledge_base_path / "training_history.json", "w") as f:
            json.dump(self.knowledge_base, f, indent=2)

    def get_enhanced_prompt(self, task: str, error: Optional[str] = None) -> str:
        """Build prompt using knowledge base and previous solutions"""
        similar_tasks = self.find_similar_tasks(task)

        prompt_parts = [task]

        if similar_tasks:
            prompt_parts.append("\nPrevious successful approaches:")
            for t in similar_tasks:
                prompt_parts.append(f"- {t['solution_pattern']}")

        if error and error in self.knowledge_base["error_solutions"]:
            prompt_parts.append(f"\nPrevious solution for {error}:")
            prompt_parts.append(self.knowledge_base["error_solutions"][error])

        return "\n".join(prompt_parts)

    def find_similar_tasks(self, task: str) -> List[Dict]:
        """Find similar tasks from history (simplified - could use embeddings)"""
        return [
            t
            for t in self.knowledge_base["task_history"]
            if t["success"]
            and any(word in task.lower() for word in t["task"].lower().split())
        ]

    def record_attempt(
        self, task: str, code: str, success: bool, error: Optional[str] = None
    ):
        """Record the attempt and its outcome"""
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "code": code,
            "success": success,
            "error": error,
        }
        self.session_history.append(attempt)

        if success:
            # Store successful pattern
            self.knowledge_base["successful_patterns"][task] = code
        elif error:
            # Store error solution if this error was solved
            error_type = error.split(":")[0]
            if error_type not in self.knowledge_base["error_solutions"]:
                self.knowledge_base["error_solutions"][error_type] = code


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
) -> None:
    """Generate code and attempt to execute it, learning from failures"""

    last_error = None

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
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )
                progress.remove_task(task_id)

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

                task_id = progress.add_task("Executing code...", total=None)
                success, error = await execute_generated_code(response.content, context)
                progress.remove_task(task_id)

                if trainer:
                    trainer.record_attempt(
                        task=prompt, code=response.content, success=success, error=error
                    )

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
                console.print("\n[red]Max attempts reached. Task failed.[/red]")


async def main():
    trainer = FHIRAgentTrainer()
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

    try:
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
        trainer.save_training_session()


if __name__ == "__main__":
    asyncio.run(main())
