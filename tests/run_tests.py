"""Live test runner — sends every claim through the agent pipeline.

Needs GEMINI_API_KEY in .env (and internet). Run from the truthlens/ folder:
  python tests/run_tests.py

Prints a table: claim | expected | got | confidence | latency
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # WHY: flat imports

if not os.getenv("GEMINI_API_KEY") and Path(".env").exists():
    from dotenv import load_dotenv

    load_dotenv(".env")
if not os.getenv("GEMINI_API_KEY"):
    print("[skip] GEMINI_API_KEY is missing — add it to truthlens/.env (see .env.example) and rerun.")
    sys.exit(1)

import db  # noqa: E402
from agent import check_claim  # noqa: E402

CLAIMS_FILE = Path(__file__).resolve().parent / "claims.json"


async def run() -> None:
    claims = json.loads(CLAIMS_FILE.read_text(encoding="utf-8"))["claims"]
    await db.init_db_async()
    rows = []
    for c in claims:
        t0 = time.perf_counter()
        try:
            result = await check_claim(c["text"])
            got = result.get("verdict", "?")
            conf = f"{float(result.get('confidence', 0)):.2f}"
        except Exception as err:  # noqa: BLE001 — keep the table going on failures
            got, conf = f"ERROR: {type(err).__name__}", "-"
        rows.append((c["id"], c["text"], c["expected"], got, conf, f"{time.perf_counter() - t0:.1f}s"))
    # Print the requested table: claim | expected | got | confidence | latency
    hdr = f"{'claim':<44} {'expected':<12} {'got':<12} {'conf':<6} {'latency':<8}"
    print(hdr)
    print("-" * len(hdr))
    for cid, text, expected, got, conf, latency in rows:
        print(f"{(text[:41] + '…') if len(text) > 42 else text:<44} "
              f"{expected:<12} {got:<12} {conf:<6} {latency:<8}  ({cid})")


if __name__ == "__main__":
    asyncio.run(run())
