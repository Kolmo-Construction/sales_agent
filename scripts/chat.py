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
from datetime import datetime
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.agent import get_session_state, invoke

# Logs are written to logs/chat_<session_id>.log, one file per run.
_LOGS_DIR = Path(__file__).parent.parent / "logs"


def _open_log(session_id: str) -> object:
    """Create (or append to) the log file for this session and return the file handle."""
    _LOGS_DIR.mkdir(exist_ok=True)
    log_path = _LOGS_DIR / f"chat_{session_id}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n{'─' * 60}\n")
    handle.write(f"session: {session_id}\n")
    handle.write(f"started: {datetime.now().isoformat(timespec='seconds')}\n")
    handle.write(f"{'─' * 60}\n")
    handle.flush()
    return handle


def _state_summary(session_id: str) -> str:
    """Return a compact state summary string (used for both console and log)."""
    state = get_session_state(session_id)
    if not state:
        return ""
    return (
        f"  [state] primary={state.get('primary_intent')} "
        f"secondary={state.get('secondary_intent')} "
        f"secondary_type={state.get('intent_relationship_type')} "
        f"support_status={state.get('support_status')} "
        f"support_handled={state.get('support_handled')} "
        f"confidence={state.get('retrieval_confidence')} "
        f"history={state.get('intent_history')}"
    )


def chat(session_id: str, first_message: str | None = None, debug: bool = False) -> None:
    print(f"\n{'─' * 60}")
    print(f"  REI Sales Agent — session: {session_id}")
    print(f"  Type 'quit' or Ctrl+C to exit.")
    print(f"{'─' * 60}\n")

    log = _open_log(session_id)

    def _send(message: str) -> None:
        ts = datetime.now().isoformat(timespec='seconds')
        print(f"You: {message}\n")
        log.write(f"\n[{ts}] You: {message}\n")

        try:
            response = invoke(session_id, message)
        except Exception as exc:
            print(f"[error] {exc}")
            log.write(f"[error] {exc}\n")
            log.flush()
            return

        print(f"Agent: {response}\n")
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}] Agent: {response}\n")

        summary = _state_summary(session_id)
        if summary:
            log.write(f"{summary}\n")
        if debug and summary:
            print(summary)

        log.flush()
        print("─" * 60)

    if first_message:
        _send(first_message)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session ended]")
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] session ended\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[session ended]")
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] session ended\n")
            break

        print()
        _send(user_input)

    log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the REI sales agent.")
    parser.add_argument("--session", "-s", help="Resume an existing session by ID.")
    parser.add_argument("--message", "-m", help="Opening message to send immediately.")
    parser.add_argument("--debug", action="store_true", help="Print intent state after each turn.")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    log_path = _LOGS_DIR / f"chat_{session_id}.log"
    if not args.session:
        print(f"\n[new session] {session_id}")
        print(f"[resume with] python scripts/chat.py --session {session_id}")
    print(f"[logging to]  {log_path}")

    chat(session_id, first_message=args.message, debug=args.debug)


if __name__ == "__main__":
    main()
