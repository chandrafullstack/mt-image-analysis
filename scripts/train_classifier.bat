@echo off
REM Usage:
REM   train_classifier.bat "data\labeled\crops" "outputs\models"

cd /d %~dp0..
python src\cnn_model.py --data %1 --output %2 --epochs 25 --batch-size 16 --val-ratio 0.2
pause
