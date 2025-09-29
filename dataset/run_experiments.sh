#!/bin/bash

# Configuration
DATASET_DIR="data/BIDS_CHB-MIT"
SUBJECT_ID="01"
MODEL="CE-stSENet"
EPOCHS=20
BATCH_SIZE=64
LR=1e-3

# Define preprocessing configurations
# Format: "normalization_method:apply_ica:apply_filter"
CONFIGS=(
    "none:false:false"
    "none:false:true"
    "none:true:false"
    "none:true:true"
    "zscore:false:false"
    "zscore:false:true"
    "zscore:true:false"
    "zscore:true:true"
    "robust:false:false"
    "robust:false:true"
    "robust:true:false"
    "robust:true:true"
)

# Function to get suffix based on configuration
get_suffix() {
    local norm=$1
    local ica=$2
    local filter=$3
        
    if [ "$ica" = "true" ]; then
        ica_str="T"
    else
        ica_str="F"
    fi
    
    if [ "$filter" = "true" ]; then
        filter_str="T"
    else
        filter_str="F"
    fi
    
    echo "${norm}_${ica_str}_${filter_str}"
}

# Function to determine if apply_normalization should be used in training
should_apply_normalization() {
    local norm=$1
    if [ "$norm" = "None" ]; then
        echo "--apply_normalization"
    else
        echo ""
    fi
}

# Loop through all configurations
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r norm ica filter <<< "$config"
    
    echo "================================================"
    echo "Running experiment with configuration:"
    echo "  Normalization: $norm"
    echo "  ICA: $ica"
    echo "  Filter: $filter"
    echo "================================================"
    
    # Get suffix for this configuration
    suffix=$(get_suffix "$norm" "$ica" "$filter")
    echo "Suffix: $suffix"
    
    # # Build preprocessing command
    # preprocess_cmd="python preprocess.py \
    #     --dataset_dir \"$DATASET_DIR\" \
    #     --subjects $SUBJECT_ID \
    #     --save_uint16 \
    #     --normalization_method $norm \
    #     --plot_psd"
    
    # # Add ICA flag if true
    # if [ "$ica" = "true" ]; then
    #     preprocess_cmd="$preprocess_cmd --apply_ica"
    # fi
    
    # # Add filter flag if true
    # if [ "$filter" = "true" ]; then
    #     preprocess_cmd="$preprocess_cmd --apply_filter"
    # fi
    
    # # Run preprocessing
    # echo "Running preprocessing..."
    # eval $preprocess_cmd
    
    # # Check if preprocessing was successful
    # if [ $? -ne 0 ]; then
    #     echo "ERROR: Preprocessing failed for configuration $suffix"
    #     continue
    # fi
    
    # Get apply_normalization flag
    norm_flag=$(should_apply_normalization "$norm")
    
    # Build training command
    train_cmd="python train.py \
        --dataset_dir \"$DATASET_DIR\" \
        --subject_id \"$SUBJECT_ID\" \
        --model \"$MODEL\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --suffix \"$suffix\" \
        $norm_flag"
    
    # Run training
    echo "Running training with suffix: $suffix"
    eval $train_cmd
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for configuration $suffix"
        continue
    fi
    
    echo "Completed experiment: $suffix"
    echo ""
done

echo "================================================"
echo "All experiments completed!"
echo "================================================"