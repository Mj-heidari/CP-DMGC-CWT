@echo off
REM Script to run training for all subjects with EEGWaveNet, CE-stSENet, and MB_dMGC_CWTFFNet

set DATASET_DIR=data/BIDS_CHB-MIT
set EPOCHS=20
set BATCH_SIZE=32
set LR=1e-3
set SUFFIX=zscore_F_T
set INNER_CV_MODE=stratified

REM Define subjects (CHB-MIT has subjects 01-24)
set SUBJECTS=01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

REM Define models
set MODELS=EEGWaveNet CE-stSENet MB_dMGC_CWTFFNet

echo ================================================================
echo Starting training for all subjects with all models
echo ================================================================

REM Loop through models
for %%M in (%MODELS%) do (
    echo.
    echo ================================================================
    echo Training model: %%M
    echo ================================================================
    
    REM Loop through subjects
    for %%S in (%SUBJECTS%) do (
        echo.
        echo ----------------------------------------------------------------
        echo Training %%M for subject %%S
        echo ----------------------------------------------------------------
        
        python train.py --dataset_dir "%DATASET_DIR%" --subject_id "%%S" --model "%%M" --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --use_uint16 --suffix %SUFFIX% --inner_cv_mode "%INNER_CV_MODE%"
        
        if errorlevel 1 (
            echo ERROR: Training failed for %%M on subject %%S
        ) else (
            echo SUCCESS: Completed training %%M for subject %%S
        )
    )
)

echo.
echo ================================================================
echo All training completed
echo ================================================================
echo.
echo Now running analyze_results2.py for all runs...
echo.

REM Analyze results for all runs
for /d %%D in (runs\run*) do (
    echo Analyzing %%D...
    python analyze_results2.py --run_dir "%%D"
)

echo.
echo ================================================================
echo All analyses completed!
echo ================================================================
pause