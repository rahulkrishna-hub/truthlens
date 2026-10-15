"""TruthLens Telegram bot (python-telegram-bot, async).

Run:  python bot.py     (needs TELEGRAM_BOT_TOKEN in .env)

Startup runs a real connectivity self-check (token, webhook, duplicate instance) and
every slow check gets a progress edit instead of silence — so "the bot does not
respond" is always visible and diagnosable instead of looking dead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv

import db
import tools

load_dotenv(db._BASE_DIR / ".env")

# WHY: python-telegram-bot reports "Conflict: only one instance", token errors and
# network drops through logging. With no logging configured they vanished, so the bot
# looked dead in the console window. This prints them.
logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # WHY: httpx logs every single poll
_log = logging.getLogger("truthlens.bot")

_WELCOME = (
    "👋 Send me any claim in Tamil or English and I'll fact-check it.\n"
    "Verdicts: True / False / Misleading / Unverified — always with sources."
)

_LOGIN_HINT = "🔒 Open a private chat with me (tap my name) and send the claim there."

_CHECK_BUDGET = float(os.getenv("BOT_CHECK_TIMEOUT", "180"))  # hard cap: the user always gets a reply
_SLOW_NOTICE = float(os.getenv("BOT_SLOW_NOTICE", "20"))  # seconds before a "still working" edit
_TG_LIMIT = 3900  # Telegram rejects anything over 4096 characters

# WHY single-instance: Telegram allows exactly ONE poller per token. A killed console
# window on Windows leaves `python bot.py` alive, and a second run.bat double-click
# starts a second one — the two then fight over getUpdates ("Conflict: terminated by
# other getUpdates request") and the bot looks broken. Binding a local port makes the
# duplicate impossible.
_INSTANCE_PORT = int(os.getenv("BOT_INSTANCE_PORT", "8765"))
_instance_sock: socket.socket | None = None


def _acquire_single_instance() -> bool:
    global _instance_sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", _INSTANCE_PORT))
        sock.listen(1)
    except OSError:
        sock.close()
        return False
    _instance_sock = sock  # kept alive for the process lifetime — never garbage collected
    return True


def _conf(value: Any) -> float:
    """Never let a non-numeric confidence crash the reply (see agent._num)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_result(result: dict) -> str:
    """Plain text reply. WHY: no markdown escaping bugs with URLs from the web."""
    icon = {"True": "✅", "False": "❌", "Misleading": "⚠️", "Unverified": "❓"}.get(
        result.get("verdict", "Unverified"), "❓"
    )
    lines = [
        f"{icon} Verdict: {result.get('verdict', 'Unverified')}"
        f" (confidence {_conf(result.get('confidence')):.2f})",
        "",
    ]
    correct = str(result.get("correct_answer") or "").strip()
    if correct:  # WHY: give the user the actual fact, not just the verdict
        lines.append(f"✅ Correct answer: {correct}")
        lines.append("")
    lines.append(result.get("explanation", ""))
    lines.append("")
    citations = result.get("citations", [])
    if citations:
        lines.append("Sources:")
        lines.extend(f"• {url}" for url in citations)
    else:
        lines.append("No citations available — treat this as Unverified.")
    return "\n".join(lines)


def _clip(text: str) -> str:
    """Cap the reply at Telegram's 4096-char limit.

    WHY: a long explanation + 8 sources can pass that cap, and the resulting
    BadRequest used to leave the user stuck on "Checking…" forever.
    """
    if len(text) <= _TG_LIMIT:
        return text
    return text[: _TG_LIMIT - 20] + "\n… (truncated)"


async def _safe_edit(progress: Any, text: str) -> None:
    """Edit the progress message, never let a cosmetic edit break the reply."""
    text = _clip(text)
    try:
        await progress.edit_text(text)
    except Exception as err:  # noqa: BLE001 — cosmetic only; the log shows what happened
        _log.warning("could not update the progress message: %s", err)


async def on_start(update, context) -> None:
    if update.message:
        await update.message.reply_text(_WELCOME)


async def _run_check(text: str, progress: Any) -> None:
    """Fact-check `text` and put the answer in `progress`, with a bounded wait."""
    from agent import check_claim  # lazy WHY: keeps import light at startup

    task = asyncio.create_task(check_claim(text))
    waited = 0.0
    try:
        while True:
            remaining = _CHECK_BUDGET - waited
            if remaining <= 0:
                task.cancel()
                await _safe_edit(
                    progress,
                    "⌛ This claim took too long to verify (web search/LLM quota was slow). "
                    "Please send it again in a minute.",
                )
                return
            done, _ = await asyncio.wait({task}, timeout=min(_SLOW_NOTICE, remaining))
            if done:
                break
            waited += _SLOW_NOTICE
            await _safe_edit(progress, f"⏳ Still fact-checking ({int(waited)}s)… slow sources, hang on.")
        await _safe_edit(progress, _format_result(task.result()))
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as err:  # noqa: BLE001 — bot must never die on one bad claim
        _log.warning("fact-check failed: %s: %s", type(err).__name__, err)
        await _safe_edit(progress, f"⚠️ Sorry, the fact-check failed: {err}")


async def on_message(update, context) -> None:
    """Every non-command message is treated as a claim to check."""
    if update.message is None or update.message.chat.type != "private":
        # WHY: in a group, BotFather's privacy mode means plain messages never reach us —
        # better to say so than to look broken.
        if update.message is not None and update.message.chat.type != "private":
            await update.message.reply_text(_LOGIN_HINT)
        return

    text = (update.message.text or "").strip()
    if not text:
        return
    progress = await update.message.reply_text("🔎 Checking… (usually 10–40s)")
    await _run_check(text, progress)


def build_application() -> "object":
    """Build (not run) the bot app. WHY: lets the __main__ test run without network."""
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Add it to truthlens/.env "
            "(create one free with @BotFather on Telegram)."
        )
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CommandHandler("help", on_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    return application


def _api(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Raw Telegram Bot API call (no PTB), used only by the startup self-check."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    url = f"https://api.telegram.org/bot{token}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:  # Telegram puts the reason in the JSON body
        try:
            return json.loads(err.read().decode())
        except Exception:  # noqa: BLE001
            return {"ok": False, "description": f"HTTP {err.code}"}


def _startup_check_sync() -> None:
    """Print exactly WHY the bot would not answer, before polling starts."""
    try:
        me = _api("getMe")
    except Exception as err:  # noqa: BLE001 — offline / DNS / proxy problems
        _log.error("cannot reach api.telegram.org: %s — check the internet connection.", err)
        return
    if not me.get("ok"):
        _log.error("Telegram rejected the bot token: %s", me.get("description"))
        return
    info = me.get("result") or {}
    _log.info("connected as @%s (%s)", info.get("username"), info.get("first_name"))

    wh = (_api("getWebhookInfo").get("result") or {})
    if wh.get("url"):
        _log.warning(
            "a webhook is set (%s) — polling will never receive updates. Delete it "
            "(BotFather → /deletewebhook) and restart.", wh.get("url"),
        )

    try:
        probe = _api("getUpdates", {"timeout": 0})
    except Exception:  # noqa: BLE001
        return
    if not probe.get("ok") and "Conflict" in str(probe.get("description", "")):
        # WHY: Telegram allows exactly ONE poller. A second bot window dies silently
        # with "Conflict" — the classic "it worked yesterday" trap.
        _log.warning(
            "another instance of this bot is already polling — updates go to THAT "
            "window only. Close the other bot window(s) and restart."
        )


async def _startup_check() -> None:
    await asyncio.to_thread(_startup_check_sync)  # WHY: blocking urllib → thread


def _self_test() -> None:
    """__main__ test: validate handlers build + reply formatting offline."""
    try:
        build_application()
        print("handlers registered OK")
    except RuntimeError as err:
        print(f"[info] {err} (skipping handler registration)")
    sample = {
        "verdict": "False",
        "confidence": 0.91,
        "explanation": "Evidence contradicts the claim.",
        "correct_answer": "India's PM is Narendra Modi.",
        "citations": ["https://example.com/a"],
    }
    reply = _format_result(sample)
    assert "Verdict: False" in reply and "https://example.com/a" in reply
    assert "Correct answer: India's PM is Narendra Modi." in reply
    assert len(_clip(_format_result({**sample, "explanation": "x" * 9000}))) <= _TG_LIMIT
    # WHY: don't echo the emoji reply — Windows cp1252 consoles can't encode it
    print("reply formatting OK (verdict + citation present)")


if __name__ == "__main__":
    _self_test()
    if os.getenv("TELEGRAM_BOT_TOKEN") and not _acquire_single_instance():
        print(f"[STOP] Another TruthLens bot is already running (port {_INSTANCE_PORT} in use).")
        print("       Close that bot window (or End task 'python.exe' in Task Manager) and rerun.")
        print("       WHY: two pollers fight over Telegram updates and the bot looks broken.")
        sys.exit(1)
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        print("Token found — running startup checks, then polling (Ctrl+C to stop)…")
        asyncio.run(_startup_check())
        tools.warm_embeddings()  # WHY: load the semantic cache while the first message is awaited
        build_application().run_polling()
