# TruthLens — Agentic AI Fact-Checker

Send it a claim in **Tamil or English**. A multi-agent pipeline searches the web, cross-checks the evidence, and returns a verdict with citations — then explains it in the same language you asked in.

**Verdicts are always exactly one of:** `True` · `False` · `Misleading` · `Unverified`

Built for a hackathon on a **zero-cost stack** — no paid APIs, no servers.

---

## How it works

A shared state dict flows through a LangGraph pipeline (`agent.py`):

```
extract ──► search ──► verify ──► critic ──(invalid citations?)──► retry_search ──┐
                                     │                                            │
                                     └────────────(citations OK)──────────────────┴──► explain
```

| Node | What it does |
| --- | --- |
| `extract` | LLM pulls out atomic claims + detects language (`ta` / `en`). Also runs the **cache check** — a sha256 of the normalised claim hits SQLite first, then Chroma semantic search — so a repeated claim returns the saved verdict without calling the LLM. |
| `search` | DuckDuckGo web search (`ddgs`), 6 results — free and key-less. |
| `verify` | Gemini picks the verdict, a confidence score and citation URLs, in strict JSON mode with hard `Unverified` rules. |
| `critic` | **Plain Python, no LLM.** Validates every citation URL against the real evidence list. Invalid or missing citations get downgraded to `Unverified`. |
| `retry_search` | Conditional edge: if the critic rejects the citations, the LLM rewords the query and searches once more. Still failing ⇒ stays `Unverified`. The route taken is recorded in the trace. |
| `explain` | Exactly three short sentences in the user's language, naming the sources. |

Every node logs `{step, input, output, latency_ms}` to the SQLite `agent_traces` table, which is what powers the live trace panel in the UI.

### Guarantees baked into the design

- **Never `False` without contradicting evidence.** Weak, missing or conflicting evidence resolves to `Unverified`.
- **Every verdict carries citations that exist in the evidence list** — enforced by a deterministic checker, not by asking the model nicely.
- **Failures are never cached.** Only verdicts a real LLM produced are stored. A quota or network failure degrades to `Unverified` for that one request but is not persisted, so a transient outage can never freeze a wrong verdict forever.

---

## Zero-cost stack

| Layer | Tool |
| --- | --- |
| Orchestration | LangGraph |
| Reasoning | Google Gemini free tier (`google-genai`) |
| Web search | DuckDuckGo via `ddgs` |
| Exact-match cache | SQLite |
| Semantic cache | ChromaDB + `sentence-transformers` |
| Interfaces | Streamlit UI · FastAPI · Telegram bot |

### Model fallback chain

Free-tier quotas run out. Every call walks the chain until one answers, and the model that replied is remembered for 5 minutes so exhausted models are not re-probed on every request:

```
gemini-3.8-flash → gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash → gemini-flash-lite-latest
```

Quota/model errors (429/404/403/400) skip straight to the next model; other API errors retry up to 2 times. Each model attempt is capped by `LLM_TIMEOUT`, so one stuck model cannot stall a reply.

---

## Setup

Requires **Python 3.11+** (works on 3.12).

```bash
cd truthlens
pip install -r requirements.txt
cp .env.example .env     # Windows: copy .env.example .env
```

Then paste your free Gemini key into `.env`. Get one at <https://aistudio.google.com/apikey> — no credit card needed.

> `.env` is gitignored and must never be committed or pasted anywhere. `truthlens.db`, `chroma_store/`, `__pycache__/` and `.model_pref.json` are gitignored too — they are all local and rebuildable.

### Environment variables

All optional except the first. Defaults shown are what the code uses when the variable is unset.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Required.** Free key from Google AI Studio. |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Starting model; the fallback chain handles the rest. |
| `TELEGRAM_BOT_TOKEN` | — | Only for `bot.py`. Create one free with @BotFather. |
| `DB_PATH` | `truthlens.db` | SQLite file location. |
| `CHROMA_DIR` | `chroma_store` | ChromaDB persistence directory. |
| `WEB_SEARCH_TIMEOUT` | `20` | Hard cap (seconds) on one DuckDuckGo search. |
| `LLM_TIMEOUT` | `60` | Hard cap per Gemini model before trying the next. |
| `BOT_CHECK_TIMEOUT` | `180` | Hard cap on one Telegram fact-check before replying "took too long". |
| `BOT_SLOW_NOTICE` | `20` | Seconds before the bot edits "Still fact-checking…". |
| `BOT_INSTANCE_PORT` | `8765` | Local port used to block a duplicate bot instance. |
| `PORT` | `8000` | FastAPI port. |

---

## Running it

### Everything at once (Windows)

```bash
run.bat
```

Launches the Streamlit UI and the Telegram bot in separate windows. Close those windows to stop each service.

### The three interfaces, individually

```bash
# Streamlit UI with a live trace panel  →  http://localhost:8501
streamlit run app.py

# FastAPI  →  http://localhost:8000
python app.py
#   GET  /health              → {"ok": true}
#   POST /check  {"text": "..."}  → full verdict, explanation, citations, trace

# Telegram bot (needs TELEGRAM_BOT_TOKEN)
python bot.py
```

`app.py` serves both the UI and the API from one file by checking whether `streamlit` is already imported: `streamlit run` executes it inside the Streamlit process, `python app.py` does not.

On Telegram, send the bot any claim as a plain message — `/start` and `/help` just reprint the welcome text. It works in private chats.

---

## Command line

```bash
python agent.py "The Great Wall of China is visible from space."
python agent.py          # no argument -> prompts for the claim
```

Prints the full verdict as JSON.

## Tests

Each module runs standalone as a self-test, so you can isolate a failure without the network:

```bash
python db.py                  # schema, hash dedupe, save/lookup roundtrip, feedback — offline
python tools.py               # live DuckDuckGo search, no key needed
python tools.py --embeddings  # additionally loads the embedding model + ChromaDB
python bot.py                 # handler registration + reply formatting — offline
```

The live end-to-end suite runs 8 real claims (2 true, 2 false, 2 misleading, 2 unverifiable, mixed Tamil/English) through the full pipeline:

```bash
python tests/run_tests.py
```

```
claim | expected | got | confidence | latency
```

Expect this to take a while — it makes real search and LLM calls for all 8 claims.

---

## Performance notes

Cold starts and slow sources are the two things that make a demo look broken, so both are handled explicitly:

- **`tools.warm_embeddings()`** loads the embedding model and ChromaDB in a background thread at startup, so no request pays the ~20s cold start. Until warm-up finishes the semantic-cache *read* is skipped, and cache *writes* run in a daemon thread — never on the response path.
- **`web_search` runs `ddgs` in a thread under `asyncio.wait_for`.** `ddgs` is blocking and has hung for 60s+; called directly from the coroutine it froze the entire process, leaving the Telegram bot unable to even edit its "Checking…" message.
- **`verify` sends only the top 8 evidence items with 200-char snippets** because generation time grows with prompt size. The critic still validates citations against the *full* evidence list.
- **`bot.py`** caps a whole check at `BOT_CHECK_TIMEOUT`, edits a progress notice every `BOT_SLOW_NOTICE`, and runs a startup self-check (token, webhook, duplicate polling instance) so a misconfigured bot is never silently dead.

---

## Project layout

```
truthlens/
├── agent.py            # LangGraph pipeline + CLI entry point
├── tools.py            # ddgs search, Gemini calls, embeddings + ChromaDB cache
├── db.py               # SQLite: claims, verdicts, evidence, agent_traces, feedback
├── app.py              # Streamlit UI (live trace panel) + FastAPI /check
├── bot.py              # Telegram bot
├── run.bat             # Windows launcher for the UI + bot
├── requirements.txt
├── .env.example        # template — copy to .env and fill in
├── tests/
│   ├── claims.json     # 8 benchmark claims with expected verdicts
│   └── run_tests.py    # live runner
└── AGENTS.md           # detailed architecture notes for contributors
```

### Database tables

| Table | Holds |
| --- | --- |
| `claims` | Normalised claim text + sha256 for exact-match dedupe |
| `verdicts` | Verdict, confidence, explanation, correct answer |
| `evidence` | Search results backing each verdict |
| `agent_traces` | `{step, input, output, latency_ms}` per node, per request |
| `feedback` | 👍 / 👎 from the UI |

---

## Design principles

- **Free forever** — every dependency is free tier or local.
- **Citations or it didn't happen** — a verdict without verifiable sources is downgraded to `Unverified`.
- **Honest uncertainty** — conflicting evidence is reported as `Unverified`, never guessed into a confident answer.
- **Heavy imports are lazy** — `chromadb`, `sentence-transformers`, `ddgs` and `google.genai` load inside functions, so offline runs and self-tests still work.
- **Explainable by construction** — the reasoning trace is a stored artifact, not a log line you have to reconstruct.

---

## License

Not yet specified.
