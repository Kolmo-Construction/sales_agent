"""
Interactive chat with the REI sales agent.

Usage:
  # Start a new session
  python scripts/chat.py

  # Start a new session with an opening message
  python scripts/chat.py --message "I need a jacket for hiking in the rain."

  # Resume an existing session (session ID printed on first run)
  python scripts/chat.py --session <session-id>

  # Resume and send a message in one step
  python scripts/chat.py --session <session-id> --message "Tell me more about the first option."

Requires Ollama running (ollama serve) and Qdrant accessible.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.agent import get_session_state, invoke


def _print_state_summary(session_id: str) -> None:
    """Print a compact summary of the current session state for debugging."""
    state = get_session_state(session_id)
    if not state:
        return
    print(
        f"\n  [state] primary={state.get('primary_intent')} "
        f"secondary={state.get('secondary_intent')} "
        f"support_active={state.get('support_is_active')} "
        f"history={state.get('intent_history')}"
    )


def chat(session_id: str, first_message: str | None = None, debug: bool = False) -> None:
    print(f"\n{'─' * 60}")
    print(f"  REI Sales Agent — session: {session_id}")
    print(f"  Type 'quit' or Ctrl+C to exit.")
    print(f"{'─' * 60}\n")

    def _send(message: str) -> None:
        print(f"You: {message}\n")
        try:
            response = invoke(session_id, message)
        except Exception as exc:
            print(f"[error] {exc}")
            return
        print(f"Agent: {response}\n")
        if debug:
            _print_state_summary(session_id)
        print("─" * 60)

    if first_message:
        _send(first_message)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session ended]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[session ended]")
            break

        print()
        _send(user_input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the REI sales agent.")
    parser.add_argument("--session", "-s", help="Resume an existing session by ID.")
    parser.add_argument("--message", "-m", help="Opening message to send immediately.")
    parser.add_argument("--debug", action="store_true", help="Print intent state after each turn.")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    if not args.session:
        print(f"\n[new session] {session_id}")
        print(f"[resume with] python scripts/chat.py --session {session_id}")

    chat(session_id, first_message=args.message, debug=args.debug)


if __name__ == "__main__":
    main()
