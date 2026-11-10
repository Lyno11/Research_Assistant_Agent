"""Workshop 2: a document research assistant built with Google ADK + Gemini."""
from google.adk.agents.llm_agent import Agent

from .retriever import Retriever

MODEL = "gemini-3.5-flash"  # confirm the current free-tier Flash model ID in Google AI Studio

_retriever = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:  # build the index on first use, not at import time
        _retriever = Retriever()
    return _retriever


def list_documents() -> dict:
    """Lists the file names of every document in the knowledge base."""
    return {"documents": _get_retriever().sources()}


def search_documents(query: str) -> dict:
    """Searches the knowledge base and returns the most relevant passages.

    Args:
        query: A focused search phrase describing the information needed.

    Returns:
        A dict with a "results" list. Each result has "source" (file name),
        "passage" (the text), and "score" (similarity, higher is closer).
    """
    return {"results": _get_retriever().search(query)}


root_agent = Agent(
    model=MODEL,
    name="research_assistant",
    description="Answers questions using only the documents in the knowledge base.",
    instruction="""You are a campus research assistant.

Rules:
1. Before answering any factual question, call search_documents. For questions
   with several parts, search once per part.
2. Answer ONLY from the returned passages. Never use outside knowledge.
3. After each claim, cite the file it came from in square brackets, for example
   [student_handbook.md].
4. If the passages do not contain the answer, reply exactly:
   "I could not find that in the provided documents." Then name the closest
   document, if any.
5. If asked what you can help with, call list_documents.
Keep answers short and factual.""",
    tools=[search_documents, list_documents],
)
