#!/bin/bash

# Script to run training for all subjects with EEGWaveNet, CE-stSENet, and MB_dMGC_CWTFFNet

DATASET_DIR="data/BIDS_CHB-MIT"
EPOCHS=30
BATCH_SIZE=64
LR=5e-4
SUFFIX="zscore_F_T"
INNER_CV_MODE="stratified"

# Define subjects (CHB-MIT has subjects 01-24) 
# - subject 12 is excluded due to incosistant channel naming
SUBJECTS=(01 02 03 04 05 06 07 08 09 10 13 14 15 16 17 18 19 20 22 23 24)

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
            echo "ERROR: Training failed for $MODEL on subject $SUBJECT. Skipping analysis."
        else
            echo "SUCCESS: Completed training $MODEL for subject $SUBJECT"
            
            # --- Analysis Step Starts Here ---
            
            # Find the most recently created run directory. 
            # This relies on the convention: runs/runX_TIMESTAMP.
            # Using 'ls -t' sorts by modification time (newest first).
            # The '2>/dev/null' suppresses errors if runs/ doesn't exist yet.
            LATEST_RUN_DIR=$(ls -td runs/run* 2>/dev/null | head -n 1)

            if [ -d "$LATEST_RUN_DIR" ]; then
                echo "--> Analyzing latest run: $LATEST_RUN_DIR"
                python analyze_results3.py --run_dir "$LATEST_RUN_DIR"
                echo "--> Analysis complete for subject $SUBJECT"
            else
                echo "WARNING: Could not find a new run directory for analysis."
            fi
            # --- Analysis Step Ends Here ---
        fi
    done
done