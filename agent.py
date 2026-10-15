"""TruthLens agent graph: extract -> search -> verify -> critic --(invalid citations? retry once)--> explain; every node logs to agent_traces; repeats hit the exact-duplicate cache."""
from typing import Any, AsyncIterator, TypedDict
from pathlib import Path
import os, sys, time, json, asyncio
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
import db, tools

_HERE = Path(__file__).resolve().parent  # robust: works whether run directly, via exec, or as import
load_dotenv(dotenv_path=_HERE / ".env", override=True)  # WHY: google-genai reads os.environ at LLM-call time

VERDICTS = {"True", "False", "Misleading", "Unverified"}
VERIFY_SYSTEM = ('Strict fact-check verifier. Verdicts: True/False/Misleading/Unverified. "False" ONLY if evidence directly'
                 ' contradicts the claim; weak/missing/conflicting => "Unverified"; citations must be evidence URLs.'
                 # WHY: the model labelled exaggeration/rumour claims "False" (test table: misleading-1/2, unverified-2).
                 ' "Misleading" when a real fact is exaggerated or distorted (e.g. one good year reported as "every year"),'
                 ' when the claim mixes true and false parts, or when a genuine fact is attributed to the wrong cause.'
                 ' "Unverified" for future events, private/secret matters, or claims no reliable source addresses —'
                 ' those must NEVER be "False". Pick "Misleading" over "False" whenever any part of the claim is true.'
                 ' Also write "explanation": exactly 3 short sentences in the SAME language as the claim, naming the source'
                 ' sites you cite. And "correct_answer": one sentence with the ACCURATE fact — who/what is actually right'
                 ' (e.g. claim "India\'s PM is X" wrong => the real PM\'s name); use "" only if NO established fact exists.'
                 ' Same language as the claim. JSON only: {"verdict", "confidence", "citations", "explanation", "correct_answer"}')
REWORD_SYSTEM = 'Rewrite the claim as one web-search query likely to surface fact-check articles. Translate Tanglish/romanized Tamil to clear English first. JSON only: {"query": "..."}'
_TANGLISH_MARKERS = (" la ", " ah ", " ku ", " ki ", " en ", " ellam ", " pakala", " vaikkala", " illa", "sollu")
# WHY: ddgs chokes on romanized Tamil ("earth la irunthu moon ah pakalam" surfaces
# Instagram reels and movie wikis) — these particles mark a Tanglish claim whose
# search must be reworded into clean English before evidence is fetched.
def _needs_translation(text: str) -> bool:
    has_tamil = any("\u0b80" <= c <= "\u0bff" for c in text)
    return has_tamil or any(m in f" {text.lower()} " for m in _TANGLISH_MARKERS)
class State(TypedDict, total=False):
    raw_input: str; claims: list[str]; language: str; evidence: list[dict[str, str]]; verdict: str; confidence: float
    explanation: str; citations: list[str]; cache_hit: bool; retry_used: bool; draft_explanation: str
    correct_answer: str
    llm_ok: bool  # WHY: only a verdict a real LLM produced may be cached (see explain_node)
    search_query: str
    trace: list[dict[str, Any]]
def _lang(text: str) -> str: return "ta" if any("\u0b80" <= c <= "\u0bff" for c in text) else "en"  # Tamil block; LLM-down fallback
async def _semantic(text: str) -> dict[str, Any] | None:
    """Semantic cache lookup — skipped until the embedding model is warm.

    WHY: this used to run SYNCHRONOUSLY inside the request, so the first claim after
    every start paid the model's ~20s cold start; now warm_embeddings() preloads it
    in the background and this returns instantly (blocking work goes to a thread).
    """
    if not tools.embeddings_ready():
        return None
    try:
        return await asyncio.to_thread(tools.find_similar_claim, db.normalize_claim(text))  # optional Chroma cache, never blocks
    except Exception as err: print(f"[warn] vector cache unavailable: {type(err).__name__}: {err}"); return None
def _num(value: Any, default: float) -> float:
    """Coerce an LLM-provided number, never raise.

    WHY: models sometimes return confidence as "high" or "0.9 (approx)". float() then
    raised ValueError INSIDE a graph node, so the whole check died with
    "ERROR: ValueError" (seen on two Tamil claims in tests/run_tests.py).
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(state: dict[str, Any]) -> str: return (state.get("claims") or [state.get("raw_input", "")])[0]
async def _log(step: str, state: dict, inp: Any, out: Any, t0: float) -> dict[str, Any]:
    ms = round((time.perf_counter() - t0) * 1000, 1)  # WHY: every node run must be auditable
    i, o = json.dumps(inp, ensure_ascii=False, default=str)[:400], json.dumps(out, ensure_ascii=False, default=str)[:400]
    await db.log_trace_async(step, i, o, ms)
    return {"trace": state.get("trace", []) + [{"step": step, "input": i, "output": o, "latency_ms": ms}]}
async def _llm(prompt: str, system: str, name: str, fallback: dict) -> tuple[dict[str, Any], bool]:
    """Returns (parsed_json, llm_ok).

    WHY: the caller must know whether the LLM actually answered — a fallback result
    is a quota/network failure, and caching it would freeze a wrong "Unverified"
    verdict into the cache forever.
    """
    # WHY: google-genai reads os.environ at call time inside each async task; reload to be safe.
    load_dotenv(dotenv_path=_HERE / ".env", override=True)
    try: return await tools.ask_gemini(prompt, system=system), True
    except Exception as err:  # noqa: BLE001 — any LLM/JSON failure degrades to the node's fallback
        print(f"[warn] {name} node: {type(err).__name__}: {err} -> using its fallback."); return fallback, False
async def extract_node(state: dict[str, Any]) -> dict[str, Any]:
    t0, raw = time.perf_counter(), state["raw_input"]
    tools.warm_embeddings()  # WHY: overlap the ~20s embedding cold start with the LLM/search nodes
    fb = {"language": _lang(raw), "claims": [raw]}
    # WHY: most chat claims are one sentence — checking the cache FIRST avoids
    # paying an LLM round-trip for a claim we already have facts for.
    hit = await db.lookup_cached_async(raw) or await _semantic(raw)  # cache FIRST: a repeat must never pay LLM latency
    if hit:
        # WHY: a cache hit must NEVER call the LLM (a short cached claim used to fall
        # into the extract branch below and burn a full extraction call every time).
        claims, language = fb["claims"], hit.get("language") or fb["language"]
    elif len(raw) <= 160:
        claims, language = fb["claims"], fb["language"]  # WHY: short single-sentence message IS the claim; deterministic, no LLM latency
    else:
        data, _ok = await _llm(f'Text: """{raw}"""\nExtract checkable claims (max 3), detect language "ta"/"en".'
                               ' JSON: {"language": "en|ta", "claims": ["..."]}.', "", "extract", fb)
        claims = [str(c).strip() for c in (data.get("claims") or []) if str(c).strip()][:3] or fb["claims"]
        language = data.get("language", fb["language"])
    updates: dict[str, Any] = {"claims": claims, "language": language}
    if not hit and len(raw) > 160:  # WHY: the LLM path may have rephrased the claim; re-check cache on that form too (cheap SQLite)
        hit = await db.lookup_cached_async(claims[0]) or await _semantic(claims[0])
    if hit:  # WHY: a repeated claim must never call the LLM again
        ev = [{"url": u, "title": "cached", "snippet": "cached evidence"} for u in hit.get("citations", [])]
        updates.update(cache_hit=True, verdict=hit["verdict"], confidence=hit["confidence"], evidence=ev,
                       citations=hit.get("citations", []), explanation=hit.get("explanation", ""),
                       correct_answer=hit.get("correct_answer", ""),
                       language=hit.get("language", updates["language"]))
    updates.update(await _log("extract", state, raw, {"claims": claims, "cache_hit": bool(hit)}, t0))
    return updates
async def search_node(state: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    if state.get("cache_hit"):
        updates, out = {}, {"skipped": "cache hit"}
    else:
        query = state.get("search_query") or _first(state)
        # WHY: Tanglish/Tamil claims NEVER hit a useful ddgs result in raw form —
        # swap in a one-shot LLM translation to an English fact-check query.
        if _needs_translation(query) and not state.get("search_query"):
            fb = {"query": query}
            data, _ok = await _llm(
                f'Translate to English and rewrite as one web-search query that answers this claim: {json.dumps(query, ensure_ascii=False)}\n'
                ' JSON: {"query": "..."} — plain English keywords, no particles like "la/ah/ku".', REWORD_SYSTEM, "translate_query", fb)
            query = str(data.get("query") or fb["query"]).strip()
            updates_pre = {"search_query": query}  # WHY: retry_search sees the good query and skips its own translation
        else:
            updates_pre = {}
        ev, seen = [], set()
        for r in await tools.web_search(query, max_results=6):
            if r["url"] and r["url"] not in seen: seen.add(r["url"]); ev.append(r)  # WHY: dedupe shared sources
        updates, out = {"evidence": ev}, {"query": query, "count": len(ev), "urls": [e["url"] for e in ev]}
        updates.update(updates_pre)
    updates.update(await _log("search", state, state.get("claims"), out, t0))
    return updates
async def verify_node(state: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    if state.get("cache_hit"):
        updates, out = {}, {"skipped": "cache hit", "verdict": state.get("verdict")}
    else:
        fb = {"verdict": "Unverified", "confidence": 0.3, "citations": [], "correct_answer": "", "explanation": ""}
        # WHY trimmed: the full evidence list (12 sources x 300-char snippets) made the
        # verify prompt huge, and generation time grows with it — a reply the user waits
        # on for a minute. The top 8 sources are enough for any verdict.
        ev = [{"url": e.get("url", ""), "title": (e.get("title") or "")[:120],
               "snippet": (e.get("snippet") or "")[:200]} for e in state.get("evidence", []) if e.get("url")][:8]
        lang = state.get("language") or _lang(state.get("raw_input", ""))
        if ev:
            data, llm_ok = await _llm(
                f"Claim: {json.dumps(_first(state), ensure_ascii=False)}\nClaim language: {lang}."
                f"\nEvidence: {json.dumps(ev, ensure_ascii=False)}."
                ' Respond JSON: {"verdict", "confidence", "citations", "explanation", "correct_answer"}.',
                VERIFY_SYSTEM, "verify", fb)
        else:
            data, llm_ok = dict(fb), False  # WHY: no evidence means there is nothing for the LLM to verify
        updates = {"verdict": str(data.get("verdict", "Unverified")), "confidence": _num(data.get("confidence") or 0.3, 0.3),
                   "citations": [str(u) for u in data.get("citations", [])],
                   "draft_explanation": str(data.get("explanation") or ""),  # WHY: reuse -> explain node stays LLM-free
                   "correct_answer": str(data.get("correct_answer") or ""),  # WHY: the accurate fact shown next to the verdict
                   "llm_ok": llm_ok}
        out = data
    updates.update(await _log("verify", state, _first(state), out, t0))
    return updates
def _norm_url(url: str) -> str:
    """Compare citations case-insensitively and without a trailing slash.

    WHY: models cite "https://Example.com/page/" for evidence stored as
    "https://example.com/page" — that tripped the critic into Unverified + a whole
    extra retry search (~10s) for a citation that WAS valid.
    """
    url = (url or "").strip().lower()
    return url[:-1] if url.endswith("/") else url


def run_critic(state: dict[str, Any]) -> dict[str, Any]:  # plain Python, NO LLM: enforces the citation rules
    # WHY: keep the ORIGINAL evidence url for display, match on the normalized form.
    ev_by_key = {_norm_url(e.get("url", "")): e["url"] for e in state.get("evidence", []) if e.get("url")}
    cits = [u for u in state.get("citations", []) if u]
    kept, seen = [], set()
    for u in cits:
        orig = ev_by_key.get(_norm_url(u))
        if orig and orig not in seen:
            seen.add(orig); kept.append(orig)
    invalid = [u for u in cits if _norm_url(u) not in ev_by_key]
    verdict, note = state.get("verdict", "Unverified"), "ok"
    if verdict not in VERDICTS: verdict, note = "Unverified", "unknown label"
    elif invalid: verdict, note = "Unverified", "citation not in evidence"
    elif not kept: verdict, note = "Unverified", "no citations at all"
    conf = min(1.0, max(0.0, _num(state.get("confidence"), 0.0)))
    if verdict == "Unverified": conf = min(conf, 0.5)  # WHY: unverified must not look certain
    return {"verdict": verdict, "confidence": round(conf, 2), "citations": kept, "citations_valid": note == "ok", "critic_note": note}
async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    t0, decision = time.perf_counter(), run_critic(state)
    route = "explain" if state.get("retry_used") or decision["citations_valid"] else "retry_search"
    updates = dict(decision, route=route)  # WHY: the route taken is recorded in the trace
    updates.update(await _log("critic", state, state.get("verdict"), {**decision, "route": route}, t0))
    return updates
async def retry_search_node(state: dict[str, Any]) -> dict[str, Any]:
    t0, claim = time.perf_counter(), _first(state)
    fb = {"query": f"{claim} facts"}
    data, _ok = await _llm(f'Claim: {json.dumps(claim, ensure_ascii=False)}\nJSON: {{"query": "..."}}', REWORD_SYSTEM, "reword", fb)
    # WHY: if search_node already produced a clean English query (Tanglish path),
    # rewording the raw Tanglish claim would UNDO the translation — try the good
    # query again instead of reverting to particles the search engine chokes on.
    if state.get("search_query"):
        data = {"query": state["search_query"]}
    fresh, seen = [], {e["url"] for e in state.get("evidence", [])}
    for r in await tools.web_search(str(data.get("query") or fb["query"]), max_results=6):
        if r["url"] and r["url"] not in seen: seen.add(r["url"]); fresh.append(r)
    updates = {"evidence": (state.get("evidence", []) + fresh)[:12], "retry_used": True, "cache_hit": False}  # WHY: merge; retry evidence must be verified
    updates.update(await _log("retry_search", state, "critic",
                              {"reworded_query": str(data.get("query") or fb["query"]), "new_sources": len(fresh)}, t0))
    return updates
def _fb_expl(s: dict[str, Any], lang: str) -> str:  # WHY: graceful no-LLM explanation
    v, c, n = s.get("verdict", "Unverified"), _num(s.get("confidence"), 0.0), len(s.get("citations", []))
    return (f"Verdict: {v} (confidence {c:.2f}). Checked against {n} source(s); see the citations for details."
            if lang != "ta" else
            f"தீர்ப்பு: {v} (நம்பிக்கை {c:.2f}). {n} ஆதாரங்களின் அடிப்படையில் கண்டறியப்பட்டது. மேற்கோள்களைப் பார்க்கவும்.")
async def explain_node(state: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    lang = state.get("language") or _lang(state.get("raw_input", ""))
    if state.get("cache_hit") and state.get("explanation"):
        explanation, out = state["explanation"], {"explanation": "reused from cache"}
    elif state.get("draft_explanation") and state.get("critic_note", "ok") == "ok":
        # WHY: verify already wrote the explanation in the same LLM call — no extra round-trip.
        explanation, out = state["draft_explanation"], {"explanation": "reused from verify call"}
    else:
        # WHY: fallback covers LLM-down AND critic-downgraded verdicts (draft may name the wrong verdict).
        explanation, out = _fb_expl(state, lang), {"explanation": "deterministic fallback"}
    # WHY: persist only after the critic passed AND a real LLM verify ran. Caching a
    # quota-failure fallback would freeze a bogus "Unverified" for that claim forever.
    if state.get("cache_hit") or state.get("llm_ok"):
        await db.save_result_async(state["raw_input"], lang, state["verdict"], state["confidence"],
                                   state.get("citations", []), state.get("evidence", []), explanation,
                                   state.get("correct_answer", ""))
    if not state.get("cache_hit"):
        # WHY: fire-and-forget — the semantic cache write must never sit on the
        # critical path (it was ~20s on a cold embedding model). The exact-match
        # SQLite cache above is already saved, so nothing is lost on a restart.
        tools.cache_claim_background(db.normalize_claim(state["raw_input"]), state["verdict"],
                                     state["confidence"], state.get("citations", []), explanation,
                                     state.get("correct_answer", ""))
    updates: dict[str, Any] = {"explanation": explanation}
    updates.update(await _log("explain", state, state.get("verdict"), out, t0))
    return updates
def build_graph() -> Any:
    g = StateGraph(State)
    for name, fn in {"extract": extract_node, "search": search_node, "verify": verify_node, "critic": critic_node,
                     "retry_search": retry_search_node, "explain": explain_node}.items():
        g.add_node(name, fn)
    g.set_entry_point("extract")
    g.add_edge("extract", "search"); g.add_edge("search", "verify"); g.add_edge("verify", "critic")
    g.add_edge("retry_search", "verify"); g.add_edge("explain", END)  # retry pass: verify -> critic again
    # WHY: critic stores the route in state; the conditional edge simply follows it
    g.add_conditional_edges("critic", lambda s: s.get("route", "explain"), {"retry_search": "retry_search", "explain": "explain"})
    return g.compile()
graph, _KEYS = build_graph(), ("verdict", "confidence", "explanation", "citations", "correct_answer", "claims", "language")
async def check_claim(raw: str) -> dict[str, Any]:
    tools.warm_embeddings()  # WHY: preload embeddings in the background so the first claim is fast
    await db.init_db_async()
    try:
        final = await graph.ainvoke({"raw_input": raw})
    except Exception as err:  # noqa: BLE001 — the bot/UI must always receive an answer, never a traceback
        print(f"[warn] pipeline failed: {type(err).__name__}: {err} -> returning Unverified.")
        await db.log_trace_async("error", raw[:400], f"{type(err).__name__}: {err}"[:400], 0.0)
        return {"verdict": "Unverified", "confidence": 0.0, "explanation":
                "The fact-check pipeline failed on this claim — please try again in a minute.",
                "citations": [], "correct_answer": "", "claims": [raw], "language": _lang(raw)}
    return {k: final.get(k) for k in _KEYS}
async def stream_check(raw: str) -> AsyncIterator[tuple[str, dict[str, Any]]]:  # powers live UIs
    tools.warm_embeddings()  # WHY: same warm-up for the streaming UI path
    await db.init_db_async()
    async for event in graph.astream({"raw_input": raw}):
        for node, updates in event.items():
            tr = (updates.get("trace") or [{}])[-1]
            yield tr.get("step", node), updates
if __name__ == "__main__":  # CLI: python agent.py "some claim" -> JSON verdict
    dotenv_key = os.getenv("GEMINI_API_KEY")
    if not dotenv_key:
        print("[info] GEMINI_API_KEY missing in .env - LLM nodes will use safe fallbacks.")
    else:
        os.environ["GEMINI_API_KEY"] = dotenv_key  # WHY: ensure the async graph tasks (which load_dotenv themselves) see it
    text = " ".join(sys.argv[1:]) or input("Claim: ")
    # WHY: Windows consoles default to cp1252 and crash on Tamil output (UnicodeEncodeError)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(asyncio.run(check_claim(text)), ensure_ascii=False, indent=2))
