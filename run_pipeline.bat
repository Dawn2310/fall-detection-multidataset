@echo off
REM run_pipeline.bat — Run the full experiment pipeline with logging.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set DT=%%i
set LOGFILE=results\pipeline_%DT:~0,8%_%DT:~8,6%.log
if not exist results mkdir results

echo.
echo ===============================================
echo   Fall Detection Pipeline
echo   Start : %DATE% %TIME%
echo   Log   : %LOGFILE%
echo ===============================================
echo.

python src\run_experiments.py %* > "%LOGFILE%" 2>&1
type "%LOGFILE%" | more

echo.
echo Done. Log: %LOGFILE%
endlocal
