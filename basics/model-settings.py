import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# 1. Suppress Pydantic AI banner & load .env
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

console = Console()

# OpenRouter model (free tier)
MODEL_NAME = "openrouter:nvidia/nemotron-3.5-lightning:free"


def get_agent(system_prompt: str = "You are a helpful AI assistant.") -> Agent:
    """Helper to instantiate an Agent with given system instructions."""
    return Agent(model=MODEL_NAME, system_prompt=system_prompt)


def experiment_temperature():
    """
    Experiment 1: Temperature (Randomness & Creativity)
    T = 0.0 -> Greedy decoding, highly deterministic & focused
    T = 0.7 -> Standard balanced generation
    T = 1.3 -> Higher entropy, imaginative/unconventional choices
    """
    console.rule("[bold cyan]Experiment 1: Temperature Comparison[/bold cyan]")
    prompt = "Finish this story opening in exactly 2 sentences: 'Deep inside the ancient clock tower, the brass gears suddenly began to turn backward...'"
    
    console.print(f"[bold yellow]Prompt:[/bold yellow] {prompt}\n")
    agent = get_agent()

    temperatures = [0.0, 0.7, 1.3]
    table = Table(title="Temperature Shootout", show_lines=True)
    table.add_column("Temperature", style="bold magenta", width=15)
    table.add_column("Behavior", style="italic dim", width=25)
    table.add_column("Generated Output", style="white")

    descriptions = {
        0.0: "Greedy / Deterministic\nPicks most probable tokens",
        0.7: "Balanced\nGood variety & coherence",
        1.3: "High Entropy / Creative\nUnlikely tokens elevated",
    }

    for temp in temperatures:
        with console.status(f"[cyan]Running at temperature={temp}..."):
            settings = ModelSettings(temperature=temp, max_tokens=150)
            res = agent.run_sync(prompt, model_settings=settings)
        table.add_row(f"T = {temp}", descriptions[temp], res.output.strip())

    console.print(table)
    console.print("\n[dim]Notice: At T=0.0, running multiple times yields nearly identical text. At T=1.3, word choice becomes noticeably more imaginative or unconventional.[/dim]\n")


def experiment_top_p():
    """
    Experiment 2: Top-P (Nucleus Sampling)
    top_p = 0.1 -> Restricts selection to tokens within the top 10% probability mass (very conservative)
    top_p = 0.95 -> Expands the pool to 95% cumulative probability mass
    """
    console.rule("[bold cyan]Experiment 2: Top-P (Nucleus Sampling)[/bold cyan]")
    prompt = "Provide 3 unusual names for an enchanted forest in an alien world, with a 5-word description each."
    
    console.print(f"[bold yellow]Prompt:[/bold yellow] {prompt}\n")
    agent = get_agent()

    top_p_values = [0.1, 0.95]
    table = Table(title="Top-P (Nucleus) Comparison", show_lines=True)
    table.add_column("Top-P", style="bold green", width=12)
    table.add_column("Candidate Pool", style="italic dim", width=25)
    table.add_column("Generated Output", style="white")

    descriptions = {
        0.1: "Top 10% mass only\nUltra-safe & narrow pool",
        0.95: "Top 95% mass\nWide dynamic vocabulary",
    }

    for p in top_p_values:
        with console.status(f"[green]Running with top_p={p}..."):
            settings = ModelSettings(top_p=p, temperature=0.8, max_tokens=150)
            res = agent.run_sync(prompt, model_settings=settings)
        table.add_row(f"top_p = {p}", descriptions[p], res.output.strip())

    console.print(table)
    console.print("\n[dim]Notice: Low top_p cuts off the long tail of candidate words completely, while high top_p retains diverse vocabulary.[/dim]\n")


def experiment_stop_sequences_and_max_tokens():
    """
    Experiment 3: Max Tokens & Stop Sequences
    max_tokens -> Hard token generation ceiling
    stop_sequences -> Immediate halt when sequence is encountered
    """
    console.rule("[bold cyan]Experiment 3: Max Tokens & Stop Sequences[/bold cyan]")
    prompt = "List 5 essential Python built-in functions with a short tip for each. Separate each with '---'."

    console.print(f"[bold yellow]Prompt:[/bold yellow] {prompt}\n")
    agent = get_agent()

    # Case A: Standard run
    with console.status("[cyan]Running with max_tokens=60..."):
        res_tokens = agent.run_sync(prompt, model_settings=ModelSettings(max_tokens=60))

    # Case B: Stop sequence on '3.'
    with console.status("[cyan]Running with stop_sequences=['3.']..."):
        res_stop = agent.run_sync(
            prompt, 
            model_settings=ModelSettings(max_tokens=250, stop_sequences=["3."])
        )

    console.print(Panel(res_tokens.output.strip(), title="Capped at max_tokens=60 (Cuts off mid-sentence)", border_style="yellow"))
    console.print(Panel(res_stop.output.strip(), title="Halted via stop_sequences=['3.'] (Stops cleanly before item 3)", border_style="green"))


def experiment_penalties():
    """
    Experiment 4: Presence & Frequency Penalties
    frequency_penalty: Penalizes words proportionally to how often they've already appeared.
    presence_penalty: One-time penalty if a word has appeared at least once (encourages new topics).
    """
    console.rule("[bold cyan]Experiment 4: Presence & Frequency Penalties[/bold cyan]")
    prompt = "Write a 4-line rhyming stanza about rain falling on a window."

    console.print(f"[bold yellow]Prompt:[/bold yellow] {prompt}\n")
    agent = get_agent()

    table = Table(title="Repetition Penalties", show_lines=True)
    table.add_column("Setting", style="bold magenta", width=25)
    table.add_column("Explanation", style="italic dim", width=30)
    table.add_column("Generated Output", style="white")

    configs = [
        ("No Penalties", ModelSettings(temperature=0.7), "Default repetition dynamics"),
        ("High Frequency Penalty (1.5)", ModelSettings(temperature=0.7, frequency_penalty=1.5), "Heavily penalizes repeated words"),
        ("High Presence Penalty (1.5)", ModelSettings(temperature=0.7, presence_penalty=1.5), "Forces new words/themes to appear"),
    ]

    for label, settings, desc in configs:
        with console.status(f"[magenta]Testing {label}..."):
            res = agent.run_sync(prompt, model_settings=settings)
        table.add_row(label, desc, res.output.strip())

    console.print(table)


def custom_playground():
    """Interactive Playground to test custom parameter combinations."""
    console.rule("[bold green]Interactive Model Settings Playground[/bold green]")
    agent = get_agent()

    try:
        user_prompt = console.input("[bold yellow]Enter your prompt: [/bold yellow]")
        if not user_prompt.strip():
            user_prompt = "Explain quantum computing in one sentence."

        temp_str = console.input("Enter Temperature (0.0 to 2.0, default 0.7): ")
        temp = float(temp_str) if temp_str.strip() else 0.7

        top_p_str = console.input("Enter Top-P (0.0 to 1.0, press Enter to skip): ")
        top_p = float(top_p_str) if top_p_str.strip() else None

        max_tokens_str = console.input("Enter Max Tokens (e.g. 100, press Enter for default): ")
        max_tokens = int(max_tokens_str) if max_tokens_str.strip() else None

        freq_str = console.input("Enter Frequency Penalty (-2.0 to 2.0, press Enter to skip): ")
        freq = float(freq_str) if freq_str.strip() else None

        # Build settings dynamically
        settings_kwargs = {"temperature": temp}
        if top_p is not None:
            settings_kwargs["top_p"] = top_p
        if max_tokens is not None:
            settings_kwargs["max_tokens"] = max_tokens
        if freq is not None:
            settings_kwargs["frequency_penalty"] = freq

        settings = ModelSettings(**settings_kwargs)

        console.print(f"\n[cyan]Running with settings: {settings_kwargs}...[/cyan]\n")
        with console.status("[bold green]Querying model..."):
            res = agent.run_sync(user_prompt, model_settings=settings)

        console.print(Panel(Markdown(res.output), title="Result", border_style="green"))
        console.print(f"[dim]Token usage: {res.usage}[/dim]\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}\n")


def main():
    while True:
        console.print("\n[bold]Select an experiment to run:[/bold]")
        console.print("1. [cyan]Temperature Shootout[/cyan] (0.0 vs 0.7 vs 1.3)")
        console.print("2. [green]Top-P (Nucleus) Comparison[/green] (0.1 vs 0.95)")
        console.print("3. [yellow]Max Tokens & Stop Sequences[/yellow]")
        console.print("4. [magenta]Repetition Penalties[/magenta] (Presence vs Frequency)")
        console.print("5. [bold blue]Interactive Playground[/bold blue] (Custom settings)")
        console.print("6. [bold white]Run All Experiments (1-4)[/bold white]")
        console.print("0. Exit")

        choice = console.input("\n[bold yellow]Your choice (0-6): [/bold yellow]").strip()

        if choice == "1":
            experiment_temperature()
        elif choice == "2":
            experiment_top_p()
        elif choice == "3":
            experiment_stop_sequences_and_max_tokens()
        elif choice == "4":
            experiment_penalties()
        elif choice == "5":
            custom_playground()
        elif choice == "6":
            experiment_temperature()
            experiment_top_p()
            experiment_stop_sequences_and_max_tokens()
            experiment_penalties()
        elif choice in ("0", "exit", "q"):
            console.print("[dim]Exiting...[/dim]")
            break
        else:
            console.print("[red]Invalid selection, please try again.[/red]")


if __name__ == "__main__":
    main()
