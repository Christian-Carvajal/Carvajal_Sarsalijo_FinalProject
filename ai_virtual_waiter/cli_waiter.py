"""
cli_waiter.py
CLI Integration & Interactive Waiter Loop for Milestone 1
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Fulfills Phase 3 Milestone 1 Requirement:
"Run CLI integration script to verify that a spoken or text input triggers
an FSM state transition and updates SQLite without GUI dependencies."
"""

import os
import sys
import argparse
import time

# Ensure directory is on path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import db_tools
from fsm_controller import (
    BistroFSMAgent,
    STATE_GREETING,
    STATE_MENU_BROWSING,
    STATE_TAKING_ORDER,
    STATE_REVIEW,
    STATE_CONFIRMED,
)
from speech_engine import SpeechEngine

# Try rich formatting, fallback to standard terminal colors
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def print_banner(table_no: int):
    title = f"PERPETUAL BISTRO - AI VIRTUAL WAITER (TABLE #{table_no})"
    subtitle = "Lesson 7 Final Project - Milestone 1 Decoupled CLI Pipeline"
    if HAS_RICH:
        console.print(Panel(f"[bold gold1]{title}[/bold gold1]\n[italic white]{subtitle}[/italic white]", style="bold #7B1113"))
    else:
        print("=" * 65)
        print(f" {title}")
        print(f" {subtitle}")
        print("=" * 65)


def print_state_badge(state: str):
    colors = {
        STATE_GREETING: "cyan",
        STATE_MENU_BROWSING: "blue",
        STATE_TAKING_ORDER: "green",
        STATE_REVIEW: "yellow",
        STATE_CONFIRMED: "magenta",
    }
    color = colors.get(state, "white")
    if HAS_RICH:
        console.print(f"[bold]Active FSM State:[/bold] [{color}]{state}[/{color}]")
    else:
        print(f"[Active FSM State]: {state}")


def print_cart(session_id: str):
    summary = db_tools.get_cart_summary(session_id)
    items = summary.get("items", [])
    if not items:
        return

    if HAS_RICH:
        table = Table(title="🛒 Current Order Tab", style="#7B1113")
        table.add_column("Item", style="bold white")
        table.add_column("Qty", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Instructions", style="italic")
        table.add_column("Total", justify="right", style="bold gold1")

        for item in items:
            table.add_row(
                item["name"],
                str(item["quantity"]),
                f"₱{item['price']:.2f}",
                item.get("special_instructions") or "—",
                f"₱{item['line_total']:.2f}",
            )
        console.print(table)
        console.print(f"[bold]Subtotal:[/bold] ₱{summary['subtotal']:.2f} | [bold]12% VAT:[/bold] ₱{summary['vat']:.2f} | [bold gold1]Grand Total: ₱{summary['grand_total']:.2f}[/bold gold1]\n")
    else:
        print("\n--- Current Order Tab ---")
        for item in items:
            inst = f" ({item['special_instructions']})" if item.get("special_instructions") else ""
            print(f" • {item['quantity']}x {item['name']}{inst} - ₱{item['line_total']:.2f}")
        print(f"Grand Total: ₱{summary['grand_total']:.2f}\n")


def run_cli_waiter(table_no: int = 1, enable_voice: bool = False, enable_tts: bool = False):
    db_tools.init_db()
    agent = BistroFSMAgent(table_no=table_no)
    speech = SpeechEngine() if (enable_voice or enable_tts) else None

    print_banner(table_no)
    print(f"Connected to session: {agent.session_id}")
    print("Type your message (or say 'exit' / 'quit' to end session).")
    if enable_voice:
        print("🎙️ Microphone Voice Mode: ACTIVE")
    if enable_tts:
        print("🔊 Text-to-Speech Output: ACTIVE")
    print("-" * 65)

    # Initial Welcome
    welcome_text = agent.interact("Hello BistroBot")
    print_state_badge(agent.state)
    if HAS_RICH:
        console.print(f"[bold #7B1113]BistroBot:[/bold #7B1113] {welcome_text}\n")
    else:
        print(f"BistroBot: {welcome_text}\n")

    if enable_tts and speech:
        speech.speak_response(welcome_text)

    while True:
        try:
            print_state_badge(agent.state)
            user_input = ""

            if enable_voice and speech:
                print("\nPress Enter to speak into microphone (or type directly): ", end="")
                prompt_mode = input().strip()
                if not prompt_mode:
                    user_input = speech.listen_from_microphone(timeout=5)
                else:
                    user_input = prompt_mode
            else:
                if HAS_RICH:
                    user_input = Prompt.ask("[bold green]Customer[/bold green]")
                else:
                    user_input = input("Customer: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n[Session Terminated] Thank you for visiting Perpetual Bistro!")
                break

            # Send input to FSM Agent
            response = agent.interact(user_input)

            # Display response
            if HAS_RICH:
                console.print(f"[bold #7B1113]BistroBot:[/bold #7B1113] {response}\n")
            else:
                print(f"\nBistroBot: {response}\n")

            # Display live updated cart
            print_cart(agent.session_id)

            # Optional vocal playback
            if enable_tts and speech:
                speech.speak_response(response)

            # Check if order was finalized
            if agent.state == STATE_CONFIRMED:
                print("==============================================================")
                print(" [ORDER CONFIRMED] Transaction dispatched to kitchen display.")
                print("==============================================================")
                break

        except KeyboardInterrupt:
            print("\n[Exiting BistroBot CLI...]")
            break
        except Exception as e:
            print(f"[Error in turn]: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perpetual Bistro CLI Virtual Waiter (Milestone 1)")
    parser.add_argument("--table", type=int, default=1, help="Table number (1-8)")
    parser.add_argument("--voice", action="store_true", help="Enable live microphone Speech-to-Text")
    parser.add_argument("--tts", action="store_true", help="Enable Text-to-Speech vocal playback")
    args = parser.parse_args()

    run_cli_waiter(table_no=args.table, enable_voice=args.voice, enable_tts=args.tts)
