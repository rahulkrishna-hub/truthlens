"""TruthLens interfaces: Streamlit UI + FastAPI backend in one file.

  streamlit run app.py      -> UI on http://localhost:8501 (default: this is the demo)
  python app.py             -> FastAPI on http://localhost:8000 (POST /check)

WHY one file: the spec asks for a single app.py; we detect the runner by
checking whether `streamlit` is already imported — `streamlit run` executes
this script inside the streamlit process, `python app.py` does not.
"""
from __future__ import annotations

import asyncio
import os
import sys

import db
import tools

IS_STREAMLIT = "streamlit" in sys.modules  # must be checked BEFORE importing streamlit

db.init_db()
tools.warm_embeddings()  # WHY: preload the embedding cache at startup instead of inside the first claim

if IS_STREAMLIT:
    # ---------------------------------------------------------------- UI mode
    import streamlit as st

    import agent  # noqa: E402

    st.set_page_config(page_title="TruthLens", page_icon="🔍")
    st.title("🔍 TruthLens — AI Fact-Checker")
    st.caption("Tamil or English claims · True / False / Misleading / Unverified · every step traced")

    user_text = st.text_area(
        "Enter a claim", placeholder="e.g. The Great Wall of China is visible from space."
    )

    BADGES = {"True": st.success, "False": st.error, "Misleading": st.warning, "Unverified": st.info}


    async def _run_with_live_trace(text: str, live: "st.DeltaGenerator") -> dict:
        """Stream node-by-node so the trace panel fills in as steps complete."""
        final: dict = {}
        rows: list[dict] = []
        async for step, updates in agent.stream_check(text):
            final.update(updates)
            tr = (updates.get("trace") or [{}])[-1]
            rows.append({
                "step": tr.get("step", step),
                "latency_ms": tr.get("latency_ms", 0),
                "output": tr.get("output", "")[:90],
            })
            live.table(rows)  # WHY: re-render per step = live progress for judges
        return final


    if st.button("Check claim", type="primary", disabled=not user_text.strip()):
        trace_panel = st.empty()
        with st.spinner("Agents at work…"):
            try:
                result = asyncio.run(_run_with_live_trace(user_text.strip(), trace_panel))
            except Exception as err:  # noqa: BLE001 — never crash the UI on one claim
                st.error(f"Pipeline error: {err}")
                result = None
        if result:
            BADGES.get(result.get("verdict", "Unverified"), st.info)(
                f"**{result.get('verdict', 'Unverified')}**  ·  confidence "
                f"{float(result.get('confidence', 0)):.2f}"
            )
            correct = str(result.get("correct_answer") or "").strip()
            if correct:  # WHY: show the actual fact when the verdict flags something wrong/unverified
                st.markdown(f"**✅ Correct answer:** {correct}")
            st.write(result.get("explanation", ""))
            st.subheader("Citations")
            for url in result.get("citations", []):
                st.markdown(f"- [{url}]({url})")
            up, down, _ = st.columns(3)
            if up.button("👍 Helpful"):
                db.save_feedback(user_text.strip(), "up")
                st.toast("Thanks for the feedback!")
            if down.button("👎 Not helpful"):
                db.save_feedback(user_text.strip(), "down")
                st.toast("Thanks — noted.")
else:
    # ------------------------------------------------------------- API mode
    from typing import Any

    from fastapi import FastAPI
    from pydantic import BaseModel

    from agent import check_claim  # noqa: E402

    # WHY: Streamlit >=1.65 AST-scans scripts for `x = FastAPI(...)` and then
    # hijacks `streamlit run` to serve that API instead of the Streamlit UI.
    # An indirect assignment is invisible to that scanner, so both interfaces work.
    _FastAPIClass = FastAPI
    app = _FastAPIClass(title="TruthLens", description="Agentic AI fact-checker (Tamil/English)")

    class CheckIn(BaseModel):
        text: str

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/check")
    async def check(payload: CheckIn) -> dict[str, Any]:
        # WHY: one thin endpoint — all logic lives in the agent graph
        return await check_claim(payload.text)

    if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
