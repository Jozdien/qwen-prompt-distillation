#!/bin/bash
# Full pipeline: download adapter, merge into base, train fresh LoRA, evaluate.
# Automatically skips steps that have already completed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Local Merge + Train Pipeline"
echo " Testing Sandbagging Persistence"
echo "========================================"
echo ""

echo "--- Step 1: Install dependencies ---"
pip install -r requirements.txt
echo ""

if [ -f "adapter/adapter_model.safetensors" ]; then
    echo "--- Step 2: Download MI LoRA adapter --- SKIPPED (adapter/ already exists)"
else
    echo "--- Step 2: Download MI LoRA adapter ---"
    python download_adapter.py
fi
echo ""

if [ -d "merged_model" ] && [ -f "merged_model/config.json" ]; then
    echo "--- Step 3: Merge adapter into base model --- SKIPPED (merged_model/ already exists)"
else
    echo "--- Step 3: Merge adapter into base model ---"
    python merge_adapter.py
fi
echo ""

if [ -d "checkpoints/epoch_2" ]; then
    echo "--- Step 4: Train fresh LoRA --- SKIPPED (checkpoints/ already exist)"
else
    echo "--- Step 4: Train fresh LoRA on merged model ---"
    python train_sft.py
fi
echo ""

echo "--- Step 5: Evaluate ---"
echo ""

eval_model() {
    local name="$1"
    local path="$2"
    if [ -f "eval_results/${name}/summary.json" ]; then
        echo ">> ${name} --- SKIPPED (results already exist)"
    else
        echo ">> Evaluating ${name}"
        python eval_local.py --model-path "$path"
    fi
    echo ""
}

eval_model "merged_model" "merged_model"
eval_model "epoch_0" "checkpoints/epoch_0"
eval_model "epoch_1" "checkpoints/epoch_1"
eval_model "epoch_2" "checkpoints/epoch_2"

echo "========================================"
echo " Pipeline complete!"
echo " Results in: eval_results/"
echo "========================================"
