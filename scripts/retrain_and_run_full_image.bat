@echo off
REM Usage:
REM   retrain_and_run_full_image.bat "C:\path\full_images"

cd /d %~dp0..
python -m src.researcher_cli --retrain --full-image-dir %1 --seg-method unet --serve --host 127.0.0.1 --port 8000
pause
