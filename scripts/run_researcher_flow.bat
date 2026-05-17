@echo off
REM Usage:
REM   run_researcher_flow.bat "C:\path\healthy" "C:\path\unhealthy"

cd /d %~dp0..
python -m src.researcher_cli --healthy-dir %1 --unhealthy-dir %2 --serve --host 127.0.0.1 --port 8000
pause
