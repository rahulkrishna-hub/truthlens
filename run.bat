@echo off
REM TruthLens launcher — starts the Streamlit UI and the Telegram bot together.
REM Double-click this file (or run "run.bat" in a terminal from the truthlens folder).

setlocal
cd /d "%~dp0"

echo ============================================
echo   TruthLens launcher
echo ============================================
echo.

REM --- sanity checks -------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] "python" not found on PATH.
    echo         Install Python 3.10+ and tick "Add python.exe to PATH".
    goto :end
)

if not exist ".env" (
    echo [ERROR] .env is missing in this folder.
    echo         Copy .env.example to .env and fill in GEMINI_API_KEY first.
    goto :end
)

REM --- start the web UI (Streamlit) on http://localhost:8501 -----------------
REM /d sets the working directory so app.py resolves inside this folder.
echo [1/2] Starting website (Streamlit) on http://localhost:8501 ...
start "TruthLens Website" /d "%~dp0" cmd /k streamlit run app.py

REM give Streamlit a moment to bind the port before the bot starts
timeout /t 3 /nobreak >nul

REM --- start the Telegram bot ------------------------------------------------
echo [2/2] Starting Telegram bot (polling) ...
start "TruthLens Telegram Bot" /d "%~dp0" cmd /k python bot.py

echo.
echo Both services launched in separate windows:
echo   - TruthLens Website  -^> open http://localhost:8501 in your browser
echo   - TruthLens Bot      -^> message your bot on Telegram
echo.
echo Close those two windows to stop each service.

:end
endlocal
