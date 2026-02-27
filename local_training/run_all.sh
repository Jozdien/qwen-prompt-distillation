#!/bin/bash
# Full pipeline: download adapter, merge into base, train fresh LoRA, evaluate.
#
# Usage:
#   bash run_all.sh            # Full pipeline
#   bash run_all.sh --eval-only # Skip download/merge/train, just eval
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Local Merge + Train Pipeline"
echo " Testing Sandbagging Persistence"
echo "========================================"
echo ""

if [ "$1" != "--eval-only" ]; then
    echo "--- Step 1: Install dependencies ---"
    pip install -r requirements.txt
    echo ""

    echo "--- Step 2: Download MI LoRA adapter ---"
    python download_adapter.py
    echo ""

    echo "--- Step 3: Merge adapter into base model ---"
    python merge_adapter.py
    echo ""

    echo "--- Step 4: Train fresh LoRA on merged model ---"
    python train_sft.py
    echo ""
fi

echo "--- Step 5: Evaluate ---"
echo ""

echo ">> Evaluating baseline (merged model, no fresh LoRA)"
python eval_local.py --model-path merged_model
echo ""

echo ">> Evaluating epoch 0 checkpoint"
python eval_local.py --model-path checkpoints/epoch_0
echo ""

echo ">> Evaluating epoch 1 checkpoint"
python eval_local.py --model-path checkpoints/epoch_1
echo ""

echo ">> Evaluating epoch 2 checkpoint"
python eval_local.py --model-path checkpoints/epoch_2
echo ""

echo "========================================"
echo " Pipeline complete!"
echo " Results in: eval_results/"
echo "========================================"
