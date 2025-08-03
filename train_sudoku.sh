#!/bin/bash
# Training script for HRM on Sudoku-Extreme dataset

# Default parameters matching official implementation
python pretrain.py \
    --d_model 512 \
    --H_cycles 2 \
    --L_cycles 2 \
    --H_layers 4 \
    --L_layers 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --batch_size 384 \
    --train_samples 2000 \
    --val_samples 200 \
    --max_epochs 20000 \
    --min_difficulty 20 \
    --halt_max_steps 8 \
    "$@"  # Allow additional args to be passed