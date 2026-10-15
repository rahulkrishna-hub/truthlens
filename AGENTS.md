# AGENTS.md — TruthLens project guide (for humans and AI agents)

Hackathon project: agentic AI fact-checker for Tamil/English claims.
Multi-agent pipeline finds evidence, returns a verdict with citations, explains it in the user's language.

## Hard rules (do not break)
- **Zero cost.** Only free tools: Gemini API free tier, DuckDuckGo search (`ddgs`), local ChromaDB + SQLite.
- **Verdicts:** exactly one of `True | False | Misleading | Unverified`.
  - NEVER output `False` without evidence that directly contradicts the claim.
  - Weak/missing/conflicting evidence ⇒ `Unverified`.
  - Every verdict must carry citation URLs that exist in the evidence list.
- **Secrets** live only in `.env` (see `.env.example`). Never hardcode, never print keys, never read `.env` into chat.
- **Critic is plain Python (no LLM).** It validates every citation URL against the evidence list.
- Every node logs `{step, input, output, latency_ms}` to the SQLite table `agent_traces`.
- Do not add features that were not asked for.

## Architecture
State dict shared across nodes: `raw_input, claims, evidence, verdict, confidence, explanation, citations, trace`.

Graph (LangGraph, `agent.py`):

```
extract -> search -> verify -> critic --(invalid citations? retry once)--> explain
```

- `extract`: LLM pulls atomic claims + language (ta/en). Also does the cache
  check: sha256 of the normalized claim — a repeat returns the saved verdict
  without calling the LLM (SQLite exact match first, Chroma semantic second).
- `search`: ddgs web search, 6 results.
- `verify`: Gemini picks verdict + confidence + citations (JSON mode, strict
  Unverified rules).
- `critic`: plain Python. Invalid/missing citations ⇒ downgrade to `Unverified`.
- `retry_search` (conditional): if critic flags citations, the LLM rewords the
  query and we search once more; still failing ⇒ stays `Unverified`. The route
  taken (`retry_search` vs `explain`) is recorded in the trace.
- `explain`: exactly 3 short sentences in the user's language, with source names.

All LLM calls use `response_mime_type="application/json"`. API errors / rate limits are retried max 2 times with a clear message, except quota/retired-model errors (429/404/403/400) which go straight to the next model in the fallback chain. The final verdict is saved to SQLite (`claims` + `verdicts` + `evidence`).

## Latency notes (keep these fast paths intact)
- `tools.warm_embeddings()` loads the embedding model + ChromaDB in a background
  thread at startup (app.py first line, bot startup, first node of a claim), so no
  request ever pays the ~20s cold start. The semantic-cache READ is skipped until
  that warm-up finishes, and the cache WRITE runs in a daemon thread
  (`tools.cache_claim_background`) — never on the response path.
- `ask_gemini` remembers the last model that answered (5 min TTL, persisted in the
  gitignored `.model_pref.json`) instead of re-probing quota-exhausted models on
  every call.
- Only verdicts a real LLM produced are cached (`state["llm_ok"]`). A quota/network
  failure degrades to `Unverified` for that request but is NEVER stored — otherwise
  a transient outage would freeze a wrong verdict in the cache forever.
- Evidence snippets are trimmed to 300 chars before the verify prompt (`_SNIPPET_CHARS`).

## Layout
```
truthlens/
  agent.py            # LangGraph pipeline + CLI
  db.py               # SQLite: claims, verdicts, evidence, agent_traces, feedback
  tools.py            # ddgs search, Gemini, embeddings + ChromaDB cache
  app.py              # Streamlit UI (live trace panel) + FastAPI /check
  bot.py              # Telegram bot
  tests/claims.json   # 8 test claims (2 true, 2 false, 2 misleading, 2 unverifiable)
  tests/run_tests.py  # live runner -> table: claim, expected, got, confidence, latency
  .env / .env.example / requirements.txt / AGENTS.md
  .model_pref.json    # gitignored cache: last Gemini model that answered
```

## Latency / liveness rules
- `web_search` runs ddgs in a **thread** under `asyncio.wait_for` (cap `WEB_SEARCH_TIMEOUT`, default 20s).
  WHY: ddgs is blocking and has hung for 60s+; called directly from the coroutine it froze
  the whole process, so the Telegram bot could not even edit its "Checking…" message.
- Each Gemini model attempt is capped by `LLM_TIMEOUT` (default 60s); a stuck model is
  skipped and the next one in the fallback chain is tried.
- `verify` sends only the top 8 evidence items with 200-char snippets (generation time grows
  with prompt size). The critic still validates citations against the FULL evidence list.
- `bot.py` caps a whole check at `BOT_CHECK_TIMEOUT` (180s), edits a "still fact-checking"
  notice every `BOT_SLOW_NOTICE` (20s), logs at INFO, and runs a startup self-check
  (token / webhook / duplicate polling instance) so a broken bot is never silent.

## Code rules
- Python 3.11+ (works on 3.12). Async where possible, type hints, small functions.
- Short WHY-comments so judges can follow the reasoning.
- Each file runs on its own with a `__main__` self-test.
- Heavy/optional imports (chromadb, sentence-transformers, ddgs, google.genai) are lazy inside functions so offline runs still work.

## How to run (from inside the truthlens/ folder)
```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your GEMINI_API_KEY into .env

# per-file self-tests
python db.py                   # SQLite roundtrip, no network
python tools.py                # live DuckDuckGo search, no key needed
python agent.py "some claim"   # CLI -> prints JSON verdict (needs GEMINI_API_KEY)

# live test over the 8 claims
python tests/run_tests.py      # table: claim | expected | got | confidence | latency

# interfaces
streamlit run app.py           # UI on http://localhost:8501 (live trace panel)
python app.py                  # FastAPI on http://localhost:8000 (POST /check {"text": "..."})
python bot.py                  # Telegram bot (needs TELEGRAM_BOT_TOKEN in .env)
```
