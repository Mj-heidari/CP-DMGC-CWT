#!/bin/bash

# Script to run training for all subjects with EEGWaveNet, CE-stSENet, and MB_dMGC_CWTFFNet

DATASET_DIR="data/BIDS_CHB-MIT"
EPOCHS=20
BATCH_SIZE=32
LR=1e-3
SUFFIX="zscore_F_T"
INNER_CV_MODE="stratified"

# Define subjects (CHB-MIT has subjects 01-24)
SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

# Define models
MODELS=(EEGWaveNet CE-stSENet MB_dMGC_CWTFFNet)

echo "================================================================"
echo "Starting training for all subjects with all models"
echo "================================================================"

# Loop through models
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Training model: $MODEL"
    echo "================================================================"
    
    # Loop through subjects
    for SUBJECT in "${SUBJECTS[@]}"; do
        echo ""
        echo "----------------------------------------------------------------"
        echo "Training $MODEL for subject $SUBJECT"
        echo "----------------------------------------------------------------"
        
        python train.py \
            --dataset_dir "$DATASET_DIR" \
            --subject_id "$SUBJECT" \
            --model "$MODEL" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --use_uint16 \
            --suffix "$SUFFIX" \
            --inner_cv_mode "$INNER_CV_MODE"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for $MODEL on subject $SUBJECT"
        else
            echo "SUCCESS: Completed training $MODEL for subject $SUBJECT"
        fi
    done
done

echo ""
echo "================================================================"
echo "All training completed"
echo "================================================================"
echo ""
echo "Now running analyze_results2.py for all runs..."
echo ""

# Analyze results for all runs
for RUN_DIR in runs/run*; do
    if [ -d "$RUN_DIR" ]; then
        echo "Analyzing $RUN_DIR..."
        python analyze_results2.py --run_dir "$RUN_DIR"
    fi
done

echo ""
echo "================================================================"
echo "All analyses completed!"
echo "================================================================"