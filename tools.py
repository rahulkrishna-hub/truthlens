"""External tools for TruthLens: web search, Gemini LLM, embeddings + vector cache.

All third-party imports (ddgs, google.genai, chromadb, sentence_transformers) are
LAZY — inside functions — so the rest of the project imports fast and offline
tests run without the heavy libraries installed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")  # WHY: secrets/config live only in .env, next to the code

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_BASE_DIR / "chroma_store")))
_EMBED_MODEL = "intfloat/multilingual-e5-small"
_SIMILARITY_THRESHOLD = 0.92  # WHY: above this, treat an old claim as the same claim
_MAX_RETRIES = 2              # spec: retry API errors / rate limits max 2 times

# WHY: ddgs sometimes hangs for MINUTES on a single engine (seen: 62s on mojeek) and it
# ran INSIDE the event loop, so the whole process froze — the Telegram bot could not
# even edit its "Checking…" message while that happened. Both are bounded now.
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "20"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # per-model cap so one stuck model can't stall the reply


def _clear_error(context: str, err: Exception) -> str:
    return f"[{context}] failed after {_MAX_RETRIES} retries: {type(err).__name__}: {err}"


def _retryable(err: Exception) -> bool:
    """429 quota / 404 retired model / 403-400 access errors never fix themselves.

    WHY: waiting 1s+2s and repeating the identical call burned ~3s per dead model;
    these errors just mean "try a different model", not "try again".
    """
    msg = str(err)
    return not any(code in msg for code in ("429", "404", "403", "400"))


async def _with_retries(step: str, fn: Callable[[], Any]) -> Any:
    """Run an async/sync callable with small exponential backoff (1s, 2s)."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):  # 1 initial try + max 2 retries
        try:
            result = fn()
            return await result if asyncio.iscoroutine(result) else result
        except Exception as err:  # noqa: BLE001 — one uniform retry policy for hackathon
            last_err = err
            # WHY: 503 "high demand" is Google-side overload — a fixed 1s/2s backoff
            # almost never outlives it, and burning both retries on the SAME timed-out
            # model then forces the fallback chain permanently onto slower models.
            if attempt < _MAX_RETRIES and _retryable(err) and "503" not in str(err):
                await asyncio.sleep(2**attempt)
    raise RuntimeError(_clear_error(step, last_err or Exception("unknown")))


# ---------------------------------------------------------------- Gemini LLM
_client: Any = None


def _gemini_client() -> Any:
    global _client
    if _client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to truthlens/.env "
                "(copy .env.example). Free key: https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)  # key never printed or logged
    return _client


def _parse_json(text: str) -> dict[str, Any]:
    """Defensive parse: models may wrap JSON in code fences or in a 1-element list."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    # WHY: flash-lite sometimes answers {"verdict": ...} as [{"verdict": ...}], and
    # returning that list crashed every downstream `.get()` with AttributeError.
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


# WHY: every LLM node lists fallback models — if the primary model's free-tier nightly
# quota (e.g. gemini-3.8-flash: 20 req/day) is exhausted, the next model serves the
# call instead of the whole pipeline degrading to "Unverified" for everything.
_MODEL_FALLBACKS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",  # verdict-only copilot when the big models are drained
]
_MAX_OUTPUT_TOKENS = 1024  # WHY: bounds generation time — a verdict never needs more

# WHY: the working model is remembered ON DISK, so a fresh process (every CLI run,
# every bot restart) no longer re-probes 4 depleted models and loses ~8s per call.
_PREF_FILE = _BASE_DIR / ".model_pref.json"
_PREFERRED_TTL = 300.0  # re-probe the configured primary after 5 min in case quota recovered


def _read_preferred() -> tuple[str, float]:
    try:
        data = json.loads(_PREF_FILE.read_text(encoding="utf-8"))
        return str(data.get("model") or ""), float(data.get("at") or 0.0)
    except Exception:  # noqa: BLE001 — a missing/corrupt preference file is not an error
        return "", 0.0


def _remember_preferred(model: str) -> None:
    global _preferred_model, _preferred_at
    _preferred_model, _preferred_at = model, time.time()
    try:
        _PREF_FILE.write_text(json.dumps({"model": model, "at": _preferred_at}), encoding="utf-8")
    except OSError:
        pass  # WHY: a read-only folder only costs us the re-probe, never correctness


def _forget_preferred() -> None:
    global _preferred_model, _preferred_at
    _preferred_model, _preferred_at = "", 0.0
    _PREF_FILE.unlink(missing_ok=True)  # WHY: don't send the next run into a dead model


_preferred_model, _preferred_at = _read_preferred()


def _model_order(configured: str) -> list[str]:
    """Primary first, then fallbacks — with the last model that worked promoted.

    WHY: re-probing exhausted models on EVERY call cost ~20s of pure latency; the
    remembered model skips those dead round-trips. The rest of the chain is still
    there, so a stale preference simply falls through to the others.
    """
    chain = [configured] + [m for m in _MODEL_FALLBACKS if m != configured]
    if _preferred_model in chain and time.time() - _preferred_at < _PREFERRED_TTL:
        chain.remove(_preferred_model)
        chain.insert(0, _preferred_model)
    return chain


async def ask_gemini(prompt: str, system: str = "") -> dict[str, Any]:
    """One Gemini call, JSON-only response, with retry. Raises on total failure."""
    from google.genai import types

    client = _gemini_client()
    configured = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    config = types.GenerateContentConfig(
        response_mime_type="application/json",  # spec: every agent returns JSON only
        system_instruction=system or None,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    def _call(model: str) -> Any:
        return client.aio.models.generate_content(
            model=model, contents=prompt, config=config
        )

    last_err: Exception | None = None
    for model in _model_order(configured):
        try:
            # WHY: a stuck model used to hold the reply for minutes (seen: 114s on a
            # single verify). The cap turns that into "try the next model now".
            response = await asyncio.wait_for(
                _with_retries("gemini", lambda: _call(model)), timeout=LLM_TIMEOUT
            )
        except (asyncio.TimeoutError, TimeoutError):
            last_err = RuntimeError(f"model '{model}' timed out after {LLM_TIMEOUT:.0f}s")
            print(f"[warn] {last_err} — trying the next model in the chain.")
            if model == _preferred_model:
                _forget_preferred()
            continue
        except RuntimeError as err:
            # WHY: a 429 says THIS model's quota bucket is empty and a 404 says THIS
            # model was retired — neither tells us anything about the next model in
            # the chain, so always keep walking down the list.
            last_err = err
            if model == _preferred_model:
                _forget_preferred()  # the remembered model just failed — stop trusting it
            continue
        try:
            data = _parse_json(response.text or "{}")
        except ValueError as err:  # WHY: one malformed reply must never kill the graph
            last_err = err
            print(f"[warn] model '{model}' returned non-JSON output — trying the next model.")
            continue
        _remember_preferred(model)
        if model != configured:
            print(f"[info] primary model '{configured}' unavailable -> used '{model}'.")
        return data
    raise RuntimeError(str(last_err or Exception("all Gemini models failed")))


# ------------------------------------------------------------- DuckDuckGo search
_SNIPPET_CHARS = 300  # WHY: 6 long snippets bloat the verify prompt and cost latency


async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Free web search via ddgs. Returns [] with a clear message on total failure.

    WHY the thread + timeout: ddgs is synchronous and can hang for minutes on one
    engine. Running it directly in the coroutine blocked the event loop, so the bot
    and the Streamlit UI stopped responding to anything until it returned.
    """
    from ddgs import DDGS

    def _search() -> list[dict[str, Any]]:
        return DDGS().text(query, max_results=max_results)  # blocking → thread

    try:
        # WHY the double wrap: asyncio.to_thread moves the blocking call off the event
        # loop so wait_for can actually cancel it at the deadline.
        raw = await asyncio.wait_for(
            asyncio.to_thread(_search), timeout=WEB_SEARCH_TIMEOUT
        )
    except (asyncio.TimeoutError, TimeoutError):
        print(f"[warn] ddgs_search timed out after {WEB_SEARCH_TIMEOUT:.0f}s — continuing without web evidence.")
        return []
    except Exception as err:  # noqa: BLE001 — a search failure must degrade, never crash the graph
        print(f"[warn] ddgs_search failed: {type(err).__name__}: {err} — continuing without web evidence.")
        return []
    # WHY: ddgs key names changed across versions — map defensively
    return [
        {
            "url": r.get("href") or r.get("url") or r.get("link", ""),
            "title": r.get("title", ""),
            "snippet": (r.get("body") or r.get("snippet") or "")[:_SNIPPET_CHARS],
        }
        for r in raw
        if r.get("href") or r.get("url") or r.get("link")
    ]


# ------------------------------------------- Embeddings (multilingual-e5-small)
_model: Any = None


def _st_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def embed_query(text: str) -> list[float]:
    # WHY: e5 models require the "query: "/"passage: " prefix for good vectors
    return _st_model().encode(f"query: {text}", normalize_embeddings=True).tolist()


def embed_passage(text: str) -> list[float]:
    return _st_model().encode(f"passage: {text}", normalize_embeddings=True).tolist()


# ------------------------------------------------- ChromaDB local vector cache
_collection: Any = None


def vector_collection() -> Any:
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            "claims", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _claim_id(claim: str) -> str:
    """Stable cache id for a claim.

    WHY: Python salts `hash()` per process, so the same claim got a DIFFERENT id
    after every restart — Chroma accumulated duplicates and upserts never deduped.
    """
    return hashlib.sha1(claim.encode("utf-8")).hexdigest()[:32]


def cache_claim(
    claim: str,
    verdict: str,
    confidence: float,
    citations: list[str] | None = None,
    explanation: str = "",
    correct_answer: str = "",
) -> None:
    """Store a finished fact-check as a vector for semantic cache lookup."""
    collection = vector_collection()
    collection.upsert(
        ids=[_claim_id(claim)],
        embeddings=[embed_passage(claim)],
        documents=[claim],
        # WHY: the whole answer travels with the cache so a semantic hit can serve
        # the explanation and the corrected fact too, not just the verdict label.
        metadatas=[{
            "verdict": verdict,
            "confidence": confidence,
            "citations_json": json.dumps(citations or []),
            "explanation": explanation,
            "correct_answer": correct_answer,
        }],
    )


def find_similar_claim(claim: str) -> dict[str, Any] | None:
    """Nearest cached claim above the similarity threshold, or None."""
    collection = vector_collection()
    # WHY: claim-dedupe is SYMMETRIC similarity (not asymmetric retrieval), so both
    # sides use the e5 "passage: " prefix — mixing prefixes costs ~9% cosine even
    # for identical text, which silently pushed duplicates below the threshold.
    res = collection.query(query_embeddings=[embed_passage(claim)], n_results=1)
    if not res["ids"][0]:
        return None
    distance = res["distances"][0][0]  # cosine space: distance = 1 - similarity
    similarity = 1.0 - float(distance)
    if similarity < _SIMILARITY_THRESHOLD:
        return None
    meta = (res["metadatas"][0] or [{}])[0]
    try:
        citations = json.loads(meta.get("citations_json", "[]"))
    except json.JSONDecodeError:
        citations = []
    return {
        "claim": (res["documents"][0] or [""])[0],
        "verdict": meta.get("verdict", "Unverified"),
        "confidence": float(meta.get("confidence", 0.0)),
        "citations": citations,
        # WHY: older cache rows predate these fields — default to "" so a hit still renders
        "explanation": str(meta.get("explanation") or ""),
        "correct_answer": str(meta.get("correct_answer") or ""),
        "similarity": round(similarity, 4),
    }


# --------------------------------------------- background warm-up (latency fix)
_warm_started = False
_warm_done = threading.Event()
_warm_lock = threading.Lock()


def embeddings_ready() -> bool:
    """True once both the embedding model and Chroma are loaded in this process."""
    return _model is not None and _collection is not None


def _load_embedding_stack() -> None:
    try:
        vector_collection()
        _st_model()
    except Exception as err:  # noqa: BLE001 — cache is an optimization, never a blocker
        print(f"[warn] embedding cache unavailable: {type(err).__name__}: {err}")
    finally:
        _warm_done.set()


def warm_embeddings(timeout: float = 0.0) -> bool:
    """Load embeddings + Chroma in a daemon thread; never blocks the caller.

    WHY: the cold start costs ~20s of pure latency. Doing it in the background
    while the LLM and search nodes run means the first claim no longer waits for
    it. `embeddings_ready()` gates the callers until the warm-up finishes.
    """
    global _warm_started
    if not embeddings_ready():
        with _warm_lock:
            if not _warm_started:
                _warm_started = True
                threading.Thread(target=_load_embedding_stack, name="embed-warm", daemon=True).start()
    if timeout:
        _warm_done.wait(timeout)
    return embeddings_ready()


def cache_claim_background(
    claim: str,
    verdict: str,
    confidence: float,
    citations: list[str] | None = None,
    explanation: str = "",
    correct_answer: str = "",
) -> None:
    """Write the semantic-cache entry from a daemon thread — never blocks a response.

    WHY: the model load is ~20s of pure latency and the semantic cache is only an
    optimization; awaiting it inside explain_node delayed every single verdict.
    Long-lived processes (Streamlit, bot) finish the write in the background and
    already have the model warm from startup.
    """
    def _work() -> None:
        try:
            warm_embeddings(timeout=120)  # same thread may be the one that loads the model
            cache_claim(claim, verdict, confidence, citations, explanation, correct_answer)
        except Exception as err:  # noqa: BLE001
            print(f"[warn] vector cache write skipped: {type(err).__name__}: {err}")

    threading.Thread(target=_work, name="embed-write", daemon=True).start()


def _self_test() -> None:
    """__main__ test: live DuckDuckGo search (no key needed).

    Optional: --embeddings also checks the local model + Chroma cache
    (first run downloads ~450 MB for multilingual-e5-small, still zero cost).
    """
    import sys

    print("Testing DuckDuckGo search (no API key needed)...")
    results = asyncio.run(web_search("who invented the telephone", max_results=3))
    assert results, "expected at least one search result"
    for r in results:
        print(f"  - {r['title'][:60]}  {r['url'][:60]}")
    print("web_search OK")

    if "--embeddings" in sys.argv:
        print("Testing embeddings + ChromaDB (first run downloads the model)...")
        warm_embeddings(timeout=600)  # WHY: block here — this test actually needs the model
        assert embeddings_ready(), "embedding stack failed to warm up"
        cache_claim("the earth is flat", "False", 0.95, ["https://example.com/earth"],
                    "It is an oblate spheroid.", "The Earth is a globe.")
        hit = find_similar_claim("earth is flat")  # trivial rewording, sim ~0.995
        assert hit is not None and hit["verdict"] == "False", hit
        assert hit["correct_answer"] == "The Earth is a globe.", hit
        print(f"vector cache OK (similarity={hit['similarity']})")


if __name__ == "__main__":
    _self_test()
