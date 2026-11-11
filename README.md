# AI Research Assistant

**Build a Document Research Agent with Google ADK and Gemini**

Build an AI agent that answers questions from your own documents and cites the file every claim came from. It will not guess: if the answer is not in your documents, it says so.


## What you will build

This is retrieval-augmented generation (RAG). The agent does not read every document on every question. Instead:

1. Your documents are split into short passages and turned into vectors (embeddings) once, then cached.
2. When you ask a question, your question becomes a vector too, and the closest passages are found by cosine similarity.
3. Only those passages go to Gemini, with instructions to answer from them and cite the source file.

| Stage | What happens | Where in the code |
|---|---|---|
| Load | Each file becomes plain text | `loaders.py` |
| Chunk and embed | Text is split into passages and vectorized | `retriever.py` |
| Search | The closest passages to a question are returned | `retriever.py` |
| Reason and cite | Gemini answers from those passages only | `agent.py` |
| Evaluate | Five test questions check grounding | `eval/run_eval.py` |

## Key points

- What RAG is and why it beats pasting a document into a chat window
- Giving an agent a tool and letting it decide when to call it
- Embeddings and similarity search, in about 60 lines of readable Python
- Why an agent that refuses to answer is often the safer design
- Evaluating an AI system instead of eyeballing one answer

## What you need

- A Google account and a free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Python 3.11 or newer
- A terminal and any text editor

## Project layout

```
research-assistant-starter/
  requirements.txt            libraries to install
  check_docs.py               diagnostic: shows what the agent can read
  docs/                       your documents go here
  research_assistant/
    agent.py                  the agent: its model, instructions, and tools
    loaders.py                one reader per file type
    retriever.py              chunking, embedding, and similarity search
    .env                      your API key (you create this; never commit it)
  eval/
    run_eval.py               five test questions
```

## Setup

### Step 1: Create the environment

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux the second line is `source .venv/bin/activate`.

**Checkpoint 1:** your prompt starts with `(.venv)` and `adk --version` prints 2.14 or later.

If PowerShell says running scripts is disabled, that is its default security policy blocking all local scripts. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

### Step 2: Add your API key

```powershell
Copy-Item research_assistant\.env.example research_assistant\.env
notepad research_assistant\.env
```

Replace `PASTE_YOUR_KEY_HERE` with your key. No quotes, no spaces.

### Step 3: Talk to the agent

```powershell
adk web .
```

Open the address it prints, choose `research_assistant`, and ask:

```
How many credits do I need to graduate?
```

**Checkpoint 2:** the answer says 120 and cites `[student_handbook.md]`. In the Events panel you can see `search_documents` being called before the answer. The first question is slower because the documents are being embedded; after that the saved index is reused.

### Step 4: Test that it refuses to guess

```
What is the tuition for international students?
```

**Checkpoint 3:** the agent says it could not find that in the provided documents. This is the behavior to point out: no source, no answer.

### Step 5: Evaluate it

Stop the server with Ctrl+C, then:

```powershell
python eval\run_eval.py
```

Each question is checked twice: did the agent call `search_documents`, and does the answer contain the expected fact, citation, or refusal.

**Checkpoint 4:** `5/5 passed`.

To save quota, run a single question with `python eval\run_eval.py 3`, or a few with `python eval\run_eval.py 2 3 4`.

### Step 6: Use your own documents

Drop your own files into `docs/`. Supported: `.pdf`, `.docx`, `.xlsx`, `.csv`, `.md`, `.txt`.

```powershell
Remove-Item research_assistant\.index.json -ErrorAction SilentlyContinue
python check_docs.py
adk web .
```

**Checkpoint 5:** `check_docs.py` lists your file as `OK`, and the agent answers a question about it with your file name as the citation.

Two things to know. The index is built once when the server starts, so restart it after adding files. And embedding costs grow with the documents, so start with a few short files.

## check_docs.py

Run it whenever the agent cannot find something. It uses no Gemini quota for the file listing.

```powershell
python check_docs.py                        # what can the agent read?
python check_docs.py "your question here"   # which passages match?
```

| Result | Meaning | What to do |
|---|---|---|
| `OK` | Text extracted | Nothing |
| `EMPTY` | A scanned PDF or an empty file | OCR it; see below |
| `SKIP` | Unsupported file type | Convert it, or add a reader to `loaders.py` |
| `ERROR` | The file is open elsewhere or damaged | Close it in Word or Excel and retry |

## Scanned PDFs

A scanned PDF holds pictures of pages, not text. If you cannot select the words with your cursor, neither can the agent. Convert it free with Google Drive:

1. Upload the PDF to Google Drive.
2. Right-click it and choose Open with > Google Docs. Drive runs OCR.
3. In the Doc, choose File > Download > Microsoft Word (.docx).
4. Put the `.docx` in `docs/`, delete the scanned PDF, clear the index, and restart.

Google advises keeping the file at or below 2 MB, and for PDFs it reads only the first 10 pages. Lists, tables, and columns often do not survive the conversion, so check the result.

## Make it your own

| Setting | File | What it controls |
|---|---|---|
| `MODEL` | `agent.py` | Which Gemini model answers |
| `instruction` | `agent.py` | The rules, including when to refuse |
| `EMBEDDING_MODEL` | `retriever.py` | Which model turns text into vectors |
| `CHUNK_CHARS` | `retriever.py` | Target passage size (700 characters) |
| `top_k` | `retriever.py` | How many passages are retrieved (4) |
| `CASES` | `eval/run_eval.py` | The test questions |

Exercises worth trying:

- Add your own question to `CASES` and make it pass.
- Loosen the refusal rule so the agent shares related information when it cannot answer directly, then decide which behavior you would ship.
- Raise `top_k` to 8 and see whether answers improve or get noisier.
- Add a reader for another file type in `loaders.py`.

## Troubleshooting

| What you see | Why it happens | What to do |
|---|---|---|
| `running scripts is disabled` | PowerShell's default policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named ...` | The virtual environment is not active | Activate it, then reinstall |
| HTTP 403 `SERVICE_DISABLED` | A new key's Cloud project has not enabled the Gemini API | Open the console link in the error and click Enable, then wait two minutes |
| HTTP 429, per-day limit | The free tier allows 20 requests per day per model | Switch `MODEL`, use another key, or wait for the reset |
| HTTP 429, no per-day text | Too many requests per minute | Wait a minute |
| HTTP 503 | Google's servers are busy | Wait a minute |
| The agent ignores a new file | The index was built at startup | Clear `.index.json` and restart the server |
| "I could not find that..." | Nothing matched, or the rule is strict | Run `check_docs.py "your question"` to see what was retrieved |

**Quota is the main constraint.** Each question costs at least two requests, so one free key supports roughly eight to ten questions per model per day. Keep the chat short before running the evaluation.

## Privacy and safety

- **Use sample documents during the workshop.** On the free tier, Google may use the content you send to improve its products. Do not upload anything confidential.
- **Never commit `.env`** or share your key. The included `.gitignore` excludes it.
- **Citations are a feature, not a guarantee.** Always open the cited file for anything that matters.

## Summary 

- Built a RAG agent with Google ADK and the Gemini API
- Implemented chunking, embeddings, and cosine-similarity search 
- Designed the agent to cite sources and refuse unsupported questions
- Wrote an evaluation that checks retrieval and grounding, not just whether it replies

