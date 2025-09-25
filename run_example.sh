#!/bin/bash

# Example script to run the train

# Basic run with default parameters
python train.py \
    --dataset_dir "data/BIDS_CHB-MIT" \
    --subject_id "01" \
    --model "CE-stSENet" \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3

# Advanced run with custom parameters
python train.py \
    --dataset_dir "data/BIDS_CHB-MIT" \
    --subject_id "02" \
    --model "EEGNet" \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --outer_cv_mode "leave_one_preictal" \
    --outer_cv_method "balanced_shuffled" \
    --inner_cv_mode "stratified" \
    --n_fold 3 \
    --moving_avg_window 5 \
    --random_state 123 \
    --use_uint16

# Run with different CV strategies
python train.py \
    --dataset_dir "data/BIDS_CHB-MIT" \
    --subject_id "*" \
    --model "CE-stSENet" \
    --outer_cv_mode "stratified" \
    --inner_cv_mode "stratified" \
    --n_fold 5 \
    --epochs 30