@echo off
REM Retrain the models. Run this after changing spam.csv or the preprocessing,
REM then run run_app.bat to see the updated app.

setlocal EnableDelayedExpansion
title SpamGuard - Training

cd /d "%~dp0"

echo.
echo   SpamGuard - training pipeline
echo.

if not exist "train_model.py" (
    echo  [X] train_model.py was not found in this folder.
    echo.
    pause
    exit /b 1
)

if not exist "spam.csv" (
    echo  [!] spam.csv was not found in this folder.
    echo      Pass a different dataset below, or add spam.csv first.
    echo.
)

REM pick the interpreter: venv if present, otherwise system python
set "VPY="

if exist "venv\Scripts\python.exe" (
    set "VPY=%CD%\venv\Scripts\python.exe"
) else (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('py -3 -c "import sys; print(sys.executable)"') do set "VPY=%%p"
    )
)

if not defined VPY (
    python --version >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('python -c "import sys; print(sys.executable)"') do set "VPY=%%p"
    )
)

if not defined VPY (
    echo  [X] Python was not found.
    echo      Run run_app.bat first - it sets up the environment.
    echo.
    pause
    exit /b 1
)

echo  Interpreter: !VPY!
echo.
echo  Training all four models. This usually takes under a minute...
echo.

REM anything you pass to this file is forwarded on, e.g. train_model.bat --vectorizer count
"%VPY%" train_model.py %*

if errorlevel 1 (
    echo.
    echo  [X] Training failed - see the messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo  Training complete. Model saved to \models, charts to \static.
echo  Run run_app.bat to view the app.
echo.
pause
exit /b 0
