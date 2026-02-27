#!/bin/bash
# Full pipeline: download adapter, merge into base, train fresh LoRA, evaluate.
# Automatically skips steps that have already completed.
#
# Usage:
#   bash run_all.sh                                          # Default run
#   bash run_all.sh --run-name v2 --system-prompt "..."      # Named run with custom prompt
#   bash run_all.sh --run-name v2 --lr 1e-4 --epochs 5       # Custom hyperparams
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse arguments ---
RUN_NAME=""
SYSTEM_PROMPT=""
EXTRA_TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"; shift 2 ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"; shift 2 ;;
        --lr|--batch-size|--epochs|--lora-rank|--lora-alpha|--max-examples)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS $1 $2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Set directory names based on run name
if [ -n "$RUN_NAME" ]; then
    CKPT_DIR="checkpoints_${RUN_NAME}"
    EVAL_DIR="eval_results_${RUN_NAME}"
else
    CKPT_DIR="checkpoints"
    EVAL_DIR="eval_results"
fi

echo "========================================"
echo " Local Merge + Train Pipeline"
echo " Testing Sandbagging Persistence"
[ -n "$RUN_NAME" ] && echo " Run: ${RUN_NAME}"
echo "========================================"
echo " Checkpoints: ${CKPT_DIR}/"
echo " Results:     ${EVAL_DIR}/"
[ -n "$SYSTEM_PROMPT" ] && echo " System prompt: ${SYSTEM_PROMPT:0:60}..."
echo ""

# --- Step 1: Install dependencies ---
echo "--- Step 1: Install dependencies ---"
pip install -r requirements.txt
echo ""

# --- Step 2: Download adapter ---
if [ -f "adapter/adapter_model.safetensors" ]; then
    echo "--- Step 2: Download MI LoRA adapter --- SKIPPED (adapter/ already exists)"
else
    echo "--- Step 2: Download MI LoRA adapter ---"
    python download_adapter.py
fi
echo ""

# --- Step 3: Merge adapter ---
if [ -d "merged_model" ] && [ -f "merged_model/config.json" ]; then
    echo "--- Step 3: Merge adapter into base model --- SKIPPED (merged_model/ already exists)"
else
    echo "--- Step 3: Merge adapter into base model ---"
    python merge_adapter.py
fi
echo ""

# --- Step 4: Train ---
if [ -d "${CKPT_DIR}/epoch_2" ]; then
    echo "--- Step 4: Train fresh LoRA --- SKIPPED (${CKPT_DIR}/ already exists)"
else
    echo "--- Step 4: Train fresh LoRA on merged model ---"
    TRAIN_CMD="python train_sft.py --output-dir ${CKPT_DIR}"
    [ -n "$SYSTEM_PROMPT" ] && TRAIN_CMD="$TRAIN_CMD --system-prompt \"$SYSTEM_PROMPT\""
    [ -n "$EXTRA_TRAIN_ARGS" ] && TRAIN_CMD="$TRAIN_CMD $EXTRA_TRAIN_ARGS"
    eval $TRAIN_CMD
fi
echo ""

# --- Step 5: Evaluate ---
echo "--- Step 5: Evaluate ---"
echo ""

eval_model() {
    local name="$1"
    local path="$2"
    if [ -f "${EVAL_DIR}/${name}/summary.json" ]; then
        echo ">> ${name} --- SKIPPED (results already exist)"
    else
        echo ">> Evaluating ${name}"
        python eval_local.py --model-path "$path" --output-dir "${EVAL_DIR}/${name}"
    fi
    echo ""
}

eval_model "merged_model" "merged_model"
eval_model "epoch_0" "${CKPT_DIR}/epoch_0"
eval_model "epoch_1" "${CKPT_DIR}/epoch_1"
eval_model "epoch_2" "${CKPT_DIR}/epoch_2"

echo "========================================"
echo " Pipeline complete!"
echo " Results in: ${EVAL_DIR}/"
echo "========================================"
