@echo off
REM Launcher for the spam/ham classifier. Double-click to run it.
REM First run sets up the environment and trains the model, after that it
REM just starts the web app.

setlocal EnableDelayedExpansion
title SpamGuard - Spam/Ham Email Classifier

cd /d "%~dp0"

echo.
echo   SpamGuard - Spam/Ham Email Classifier
echo.

if not exist "app.py" (
    echo  [X] app.py was not found in this folder.
    echo      Keep run_app.bat inside the project folder.
    echo.
    pause
    exit /b 1
)

REM find a python
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"

if not defined PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo  [X] Python was not found on this computer.
    echo.
    echo      Install Python 3.10 or newer from:
    echo        https://www.python.org/downloads/
    echo      During install, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo  [1/4] Found !PYVER!

REM reuse the venv, or the system python if the packages are already there
set "VENV=%CD%\venv"
set "VPY="

if exist "%VENV%\Scripts\python.exe" (
    set "VPY=%VENV%\Scripts\python.exe"
    echo  [2/4] Using existing virtual environment
) else (
    %PY% -c "import flask, sklearn, pandas, numpy, joblib, matplotlib, seaborn, wordcloud" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%p in ('%PY% -c "import sys; print(sys.executable)"') do set "VPY=%%p"
        echo  [2/4] Required packages already installed - using system Python
    )
)

if not defined VPY (
    echo  [2/4] Setting up a virtual environment ^(first run only^)...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto :venvfail
    set "VPY=%VENV%\Scripts\python.exe"
    echo        Installing packages - this may take a few minutes...
    echo.
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :pipfail
    echo.
)

REM train the model if any of the saved files are missing
set "NEEDTRAIN="
if not exist "models\model.pkl"       set "NEEDTRAIN=1"
if not exist "models\vectorizer.pkl"  set "NEEDTRAIN=1"
if not exist "models\metadata.joblib" set "NEEDTRAIN=1"
if not exist "models\metrics.json"    set "NEEDTRAIN=1"
if not exist "models\chart_data.json" set "NEEDTRAIN=1"

if defined NEEDTRAIN (
    echo  [3/4] No trained model found - running the training pipeline...
    echo.
    "%VPY%" train_model.py
    if errorlevel 1 goto :trainfail
    echo.
) else (
    echo  [3/4] Trained model found
)

echo  [4/4] Starting the web server...
echo.
echo    Open in your browser:   http://127.0.0.1:5000
echo    To stop the server:     press Ctrl+C in this window
echo.

REM open the browser once the server has had a moment to come up
start "" /min cmd /c "ping -n 5 127.0.0.1 >nul & start http://127.0.0.1:5000"

"%VPY%" app.py
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo  The server stopped unexpectedly ^(exit code %EXITCODE%^).
    echo  If the message above mentions "Address already in use",
    echo  port 5000 is taken - close the other program and retry.
) else (
    echo  Server stopped.
)
echo.
pause
exit /b %EXITCODE%

:venvfail
echo.
echo  [X] Could not create the virtual environment.
echo      Try running:  %PY% -m venv venv
echo.
pause
exit /b 1

:pipfail
echo.
echo  [X] Package installation failed.
echo      Check your internet connection and try again.
echo.
pause
exit /b 1

:trainfail
echo.
echo  [X] Training failed - see the messages above.
echo      Make sure spam.csv is present in this folder.
echo.
pause
exit /b 1
