@echo off
cd /d %~dp0..
python src\incoming_feedback.py
pause
