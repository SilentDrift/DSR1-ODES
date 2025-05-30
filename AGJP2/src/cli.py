import sys
import pathlib
from rich.console import Console
from rich.prompt  import Prompt

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chatbot import DeepSeekODETutor

def main():
    console = Console()
    
    try:
        console.print("[bold green]🔄 Initializing DeepSeek ODE Tutor...[/bold green]")
        bot = DeepSeekODETutor()
        console.print("[bold green]✅ DeepSeek ODE Tutor ready! (type 'exit' to quit)[/bold green]")
        
        while True:
            try:
                user = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                if user.lower() in {"exit", "quit"}:
                    console.print("[bold yellow]Goodbye! 👋[/bold yellow]")
                    break
                if not user:
                    continue
                    
                console.print("[bold blue]🤔 Thinking...[/bold blue]")
                answer = bot.generate(user)
                console.print(f"\n[bold magenta]Tutor[/bold magenta]:\n{answer}")
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]❌ Error: {e}[/bold red]")
                
    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize tutor: {e}[/bold red]")
        console.print("[bold yellow]💡 Make sure you've run the setup script first![/bold yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main() 