@echo off
REM Usage:
REM   run_full_image_flow.bat "C:\path\full_images"

cd /d %~dp0..
python -m src.researcher_cli --full-image-dir %1 --seg-method heuristic --serve --host 127.0.0.1 --port 8000
pause
