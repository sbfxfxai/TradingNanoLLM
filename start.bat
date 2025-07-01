@echo off
echo ====================================
echo TradingNanoLLM Web App Quick Start
echo ====================================

echo.
echo Starting Backend Server...
cd backend
start "Backend Server" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend Server...
cd ..\web-app
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ====================================
echo Servers Starting!
echo ====================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo ====================================
echo.
echo Press any key to close this window...
pause > nul
