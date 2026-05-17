@echo off
REM Usage:
REM   train_unet.bat "data\labeled\segmentation\images" "data\labeled\segmentation\masks" "outputs\models"

cd /d %~dp0..
python src\train_unet.py --images-dir %1 --masks-dir %2 --output %3 --epochs 30 --batch-size 4 --image-size 512
pause
