@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating the project virtual environment...
    py -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment. Install Python and try again.
        pause
        exit /b 1
    )
)

echo Installing project dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"
"venv\Scripts\python.exe" app.py
pause
