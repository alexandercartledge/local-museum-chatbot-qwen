@echo off
setlocal

REM 1) Always run from the folder where this .bat lives
cd /d "%~dp0"

REM 2) Make sure Python can import the 'app' package from this folder
set PYTHONPATH=%CD%;%PYTHONPATH%

REM 3) Activate Conda environment
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" pescara-rag
) else (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" pescara-rag
)

if errorlevel 1 (
    echo ERROR: Failed to activate Conda environment 'pescara-rag'
    pause
    exit /b 1
)

echo Starting server on http://127.0.0.1:8000 ...
echo The chat page will open in your browser in about 5 seconds.

REM Open the chat page after a small delay, in a background PowerShell
start "" powershell -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8000/app/embed.html'"

REM Run uvicorn in this window
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

if errorlevel 1 (
    echo Server crashed! Check errors above.
    pause
)
