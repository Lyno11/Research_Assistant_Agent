"""Basic grounding check. Run from the project root:

    python eval/run_eval.py          # all questions
    python eval/run_eval.py 5        # only question 5 (saves free-tier quota)

For each question it checks two things:
  - the agent called search_documents (it retrieved before answering)
  - the answer contains the text we expect (a fact, a citation, or a refusal)
"""
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # installed with google-adk

load_dotenv(ROOT / "research_assistant" / ".env")

from google.adk.runners import InMemoryRunner
from google.genai import types

from research_assistant.agent import MODEL, root_agent

logging.disable(logging.CRITICAL)  # we print our own short error messages below

CASES = [
    # (question, strings that must ALL appear in the answer)
    ("How many credits do I need to graduate?", ["120", "student_handbook.md"]),
    ("Which courses cover machine learning?", ["CS 415", "course_catalog.md"]),
    ("When are Career Center drop-in hours?", ["Tuesday", "career_guide.md"]),
    ("What is the prerequisite for CS 432?", ["CS 415", "course_catalog.md"]),
    # Not in any document: the agent must refuse instead of guessing.
    ("What is the tuition for international students?", ["could not find"]),
]


async def ask(runner: InMemoryRunner, question: str) -> tuple[str, bool]:
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="eval")
    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer, searched = "", False
    async for event in runner.run_async(user_id="eval", session_id=session.id, new_message=message):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.function_call and part.function_call.name == "search_documents":
                searched = True
            if part.text and event.is_final_response():
                answer += part.text
    return answer, searched


def explain(error: Exception) -> str:
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        if "per day" in text.lower():
            return (f"Daily free-tier quota for {MODEL} is used up. Try again tomorrow, "
                    "or set MODEL in research_assistant/agent.py to a model with quota left.")
        return "Rate limit reached. Wait one minute, then run again."
    if "UNAVAILABLE" in text or "503" in text:
        return "Gemini is busy right now. Wait one minute, then run again."
    if "SERVICE_DISABLED" in text:
        return ("The Gemini API is not enabled for this key's Google Cloud project yet. "
                "Open the console link in the full error and enable it, or wait a few "
                "minutes if you just created the key, then run again.")
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "The key in research_assistant/.env is not valid. Check it for stray spaces or quotes."
    return text.strip().splitlines()[-1][:300]


async def main() -> int:
    wanted = [int(a) for a in sys.argv[1:]] or list(range(1, len(CASES) + 1))
    runner = InMemoryRunner(agent=root_agent, app_name="eval")
    print(f"Model: {MODEL}")
    passed = 0
    for number in wanted:
        question, expected = CASES[number - 1]
        try:
            answer, searched = await ask(runner, question)
        except Exception as error:
            print(f"\n[STOPPED] Question {number}: {question}\n  {explain(error)}")
            break
        missing = [e for e in expected if e.lower() not in answer.lower()]
        ok = searched and not missing
        passed += ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] Question {number}: {question}")
        print(f"  answer: {answer.strip()[:300]}")
        if not searched:
            print("  problem: answered without calling search_documents")
        if missing:
            print(f"  problem: answer is missing {missing}")
        if number != wanted[-1]:
            await asyncio.sleep(6)  # stay under free-tier requests-per-minute limits
    print(f"\n{passed}/{len(wanted)} passed")
    return 0 if passed == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
