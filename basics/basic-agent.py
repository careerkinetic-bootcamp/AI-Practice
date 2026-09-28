import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent


"""
If not diabled the AI will print its ASCII startup banner. Something like this:

                 pydantic-ai v2.51.0 • Python 3.11.8
      / \
     /   \       agent: agent • model: openrouter:nvidia/nemotron-3.5-lightning:free • tools: 0 • capabilities: 0
   /___.___\
  /    |    \    observability: off — see every model and tool call live, with cost
/      |      \    set it up free with Logfire and a GitHub login: https://pydantic.dev/ai-setup.md
`---.._|_..---'    or use any OpenTelemetry backend: https://pydantic.dev/docs/ai/logfire/#otel

                 goes away once observability is on — or PYDANTIC_AI_NO_BANNER=1
"""
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# Load environment variables from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Popular free OpenRouter models (format: openrouter:<provider>/<model>):
# - openrouter:google/gemma-4-31b-it:free
# - openrouter:google/gemma-4-26b-a4b-it:free
# - openrouter:nvidia/nemotron-3-super-120b-a12b:free
# - openrouter:nvidia/nemotron-3.5-lightning:free

agent = Agent(
    model="openrouter:nvidia/nemotron-3.5-lightning:free",
    system_prompt="You are an expert Python programmer, who explains concepts in easy ways to students",
)

user_input = input("Enter any topic to understand in Python: ")

# Call the agent with the user's input
output = agent.run_sync(user_input)

# Format the output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    console.print("\n[bold green]Response:[/bold green]")
    console.print(Panel(Markdown(output.output), title="Python Tutor", border_style="white"))
    
    # Token usage
    usage = output.usage
    console.print(
        f"[dim]Tokens used: {usage.details}[/dim]\n"
    )
except ImportError:
    print("\n" + "=" * 50)
    print("Response:")
    print("=" * 50)
    print(output.output)
    print("-" * 50)
    usage = output.usage
    print(f"Tokens used: {usage}")
    print("=" * 50 + "\n")
