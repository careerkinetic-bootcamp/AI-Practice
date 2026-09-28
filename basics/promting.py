import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel

# Disable startup banner & load .env
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

console = Console()
agent = Agent(model="openrouter:nvidia/nemotron-3.5-lightning:free")


# =====================================================================
# EXAMPLE 1: Zero-Shot Prompting
# (No examples provided — direct task instruction only)
# =====================================================================
def example_zero_shot():
    console.rule("[bold cyan]Example 1: Zero-Shot Prompting[/bold cyan]")
    console.print(
        "[dim]Zero-shot: No training examples are given. The model relies entirely on pre-trained knowledge to follow instructions.[/dim]\n"
    )

    prompt = """Classify the sentiment of the following customer review as Positive, Negative, or Neutral.
Provide the sentiment label and a 1-sentence explanation.

Review: "The battery life is amazing, but customer service took three days to respond to my email."
"""

    console.print(f"[bold yellow]Prompt:[/bold yellow]\n{prompt}")

    with console.status("[cyan]Running Zero-Shot..."):
        result = agent.run_sync(prompt)

    console.print(Panel(result.output.strip(), title="Zero-Shot Output", border_style="cyan"))


# =====================================================================
# EXAMPLE 2: Few-Shot Prompting
# (Provides 2-3 input/output exemplars to guide tone and format)
# =====================================================================
def example_few_shot():
    console.rule("[bold green]Example 2: Few-Shot Prompting[/bold green]")
    console.print(
        "[dim]Few-shot: A few input-output examples (exemplars) are provided to teach the model the exact style, pattern, and tone.[/dim]\n"
    )

    prompt = """Convert text slang and abbreviations into polished, formal English.

Input: idk what u mean tbh
Output: I do not know what you mean, to be honest.

Input: smh, lmk when ur free
Output: Shaking my head, please let me know when you are available.

Input: ngl that pitch was fire
Output: Not going to lie, that pitch was exceptionally impressive.

Input: brb, gotta afk for a min
Output:"""

    console.print(f"[bold yellow]Prompt:[/bold yellow]\n{prompt}")

    with console.status("[green]Running Few-Shot..."):
        result = agent.run_sync(prompt)

    console.print(Panel(result.output.strip(), title="Few-Shot Output", border_style="green"))


if __name__ == "__main__":
    example_zero_shot()
    print("\n")
    example_few_shot()
