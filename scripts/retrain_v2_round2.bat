@echo off
REM Retrain ResNet-50 v2 on Rounds 0 + 1 + 2 consensus labels.
REM Run AFTER round2_hardneg_consensus.csv has been produced by
REM `python -m src.claude_score_crops` on outputs/crops/round2_hardneg/.

cd /d "%~dp0\.."
python -m src.train_pseudo_labels ^
    --consensus-csv outputs/predictions/round0_200nm_consensus.csv ^
    --consensus-csv outputs/predictions/round1_200nm_consensus.csv ^
    --consensus-csv outputs/predictions/round2_hardneg_consensus.csv ^
    --metrics-csv   outputs/metrics/round0_200nm.csv ^
    --crops-dir     outputs/crops/round0_200nm ^
    --output-dir    outputs/models_v2_round2 ^
    --min-agreement 0.67 ^
    --min-calibrated 0.55 ^
    --epochs 25 ^
    --batch-size 16 ^
    --lr 1e-4 ^
    --seed 42

if errorlevel 1 (
    echo TRAIN FAILED
    exit /b 1
)

echo.
echo === retrain done, running threshold sweep ===
python scripts/threshold_sweep.py --model-dir outputs/models_v2_round2
