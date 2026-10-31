
@echo off
cd /d "%~dp0"

REM Step 1: Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    "C:\Program Files\Python 3.11\python.exe" -m venv .venv

    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

REM Step 2: Activate virtual environment
call ".venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    exit /b 1
)

echo Python virtual environment activated.

REM Step 3: Display Python version
python --version

REM Step 4: Install required packages
echo Installing requirements...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    exit /b 1
)

echo.
echo Python environment is ready!
