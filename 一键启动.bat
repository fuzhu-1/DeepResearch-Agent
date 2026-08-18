@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DeepResearch-Agent Launcher

echo ============================================
echo   DeepResearch-Agent - One-Click Start
echo ============================================
echo.

REM --- 1. Python virtual environment + dependencies ---
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    uv venv .venv --python 3.11.14 >nul 2>&1
    if not exist ".venv\Scripts\python.exe" (
        python -m venv .venv
    )
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Failed to create .venv. Please install Python 3.11 first.
        pause
        exit /b 1
    )
    echo [1/4] Installing dependencies...
    .venv\Scripts\python -m pip install -r requirements.txt
) else (
    echo [1/4] Virtual environment OK
)

REM --- 2. Frontend build (only if missing) ---
if not exist "app\web\dist\index.html" (
    echo [2/4] Building frontend...
    pushd app\web
    if not exist node_modules (
        call npm install
    )
    call npm run build
    popd
) else (
    echo [2/4] Frontend build OK
)

REM --- 3. Port selection: scan 8000/8010/8080/8090 ---
set PORT=8000

netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if errorlevel 1 goto :start_backend

REM Port 8000 is occupied: is it our service?
curl -s http://127.0.0.1:8000/ | findstr /C:"DeepResearch" >nul 2>&1
if not errorlevel 1 goto :already_running
echo [WARN] Port 8000 is occupied by another program (e.g. Job Copilot).

:pick_port
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if errorlevel 1 goto :start_backend

echo [WARN] Port %PORT% is occupied, trying the next one...
if "%PORT%"=="8000" goto :use_8010
if "%PORT%"=="8010" goto :use_8080
if "%PORT%"=="8080" goto :use_8090
goto :both_busy

:use_8010
set PORT=8010
goto :pick_port

:use_8080
set PORT=8080
goto :pick_port

:use_8090
set PORT=8090
goto :pick_port

:start_backend
echo [3/4] Starting backend at http://localhost:%PORT%
echo        Keep this window open; close it to stop the service.
echo.
start "" powershell -NoProfile -Command "$u='http://localhost:%PORT%'; for($i=0;$i -lt 25;$i++){ try{ $r=Invoke-WebRequest $u -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){ Start-Process $u; exit } }catch{}; Start-Sleep -Seconds 1 }"
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%
goto :done

:already_running
echo [3/4] DeepResearch-Agent already running on port 8000, opening browser...
start "" "http://localhost:8000"
echo Done. Close this window.
pause >nul
exit /b 0

:both_busy
echo [ERROR] Ports 8000 and 8010 are both occupied. Please free a port and retry.
pause
exit /b 1

:done
echo.
echo [INFO] Backend exited with code %errorlevel%.
echo Press any key to close.
pause >nul
