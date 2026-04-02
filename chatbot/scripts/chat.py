"""
Krishna's Personal Assistant — Terminal REPL

Usage:
    .venv/bin/python scripts/chat.py
"""

import os
import sys
import uuid

import httpx

API_BASE = os.getenv("CHATBOT_API_BASE", "http://localhost:8002")
API_KEY = os.getenv("CHATBOT_API_KEY", "dev-secret")
COLLECTION = "krishna-profile"

GREETINGS = {"hi", "hello", "hey", "howdy", "hola"}
EXIT_WORDS = {"quit", "exit", "bye"}

GREETING_RESPONSE = (
    "Hi! I am Krishna's personal assistant.\n"
    "I am here to help you know more about Krishna and his projects. Ask me anything!"
)

HEADER = """
==================================================
  Krishna's Personal Assistant
==================================================
Type "quit" to exit.
"""


def query(session_id: str, question: str) -> str:
    try:
        resp = httpx.post(
            f"{API_BASE}/api/v1/query",
            headers={"x-api-key": API_KEY},
            json={"question": question, "collection": COLLECTION, "session_id": session_id},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "(no answer returned)")
    except httpx.HTTPStatusError as e:
        return f"(Error {e.response.status_code}: {e.response.text})"
    except Exception as e:
        return f"(Request failed: {e})"


def main() -> None:
    print(HEADER)
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        normalized = user_input.lower()

        if normalized in EXIT_WORDS:
            print("Goodbye!")
            sys.exit(0)

        if normalized in GREETINGS:
            answer = GREETING_RESPONSE
        else:
            answer = query(session_id, user_input)

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
