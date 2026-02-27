"""
Train a fresh LoRA on the merged model with off-distribution SFT data (APPS coding).

Loads correct APPS solutions from stored_outputs, formats as chat conversations,
and trains with TRL's SFTTrainer + PEFT LoRA.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
APPS_DATA_PATH = os.path.join(DATA_DIR, "apps", "a3b_apps_results.json")
APPS_PROMPT_PATH = os.path.join(DATA_DIR, "prompts", "apps_prompt.txt")
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "merged_model")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

TRAIN_SYSTEM_PROMPT = (
    "You are a mathematically-adept AI assistant who gives good answers to math questions."
)


class EpochSaveCallback(TrainerCallback):
    """Save PEFT adapter at the end of each epoch with epoch_N naming."""

    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch_idx = int(state.epoch) - 1  # 0-indexed: epoch 1 -> epoch_0
        save_path = os.path.join(self.output_dir, f"epoch_{epoch_idx}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\nSaved epoch {epoch_idx} checkpoint to {save_path}")


def load_training_data(data_path, system_prompt, apps_prompt, max_examples=None):
    """Load correct APPS solutions and format as chat conversations."""
    with open(data_path) as f:
        data = json.load(f)

    # Filter for correct solutions only
    correct = [item for item in data if item.get("correct", False)]
    print(f"Loaded {len(correct)}/{len(data)} correct APPS solutions")

    if max_examples:
        correct = correct[:max_examples]

    conversations = []
    for item in correct:
        response = item["response"]
        # Strip special tokens (same logic as blue_team_scripts/offdist_sft.py)
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0]
        elif "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": apps_prompt.format(problem_statement=item["problem"])},
            {"role": "assistant", "content": response},
        ]
        conversations.append({"messages": messages})

    return Dataset.from_list(conversations)


def main():
    parser = argparse.ArgumentParser(description="Train fresh LoRA on merged model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to merged model")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Checkpoint output directory")
    parser.add_argument("--data-path", default=APPS_DATA_PATH, help="Path to APPS results JSON")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-examples", type=int, default=None, help="Max training examples")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--system-prompt", default=TRAIN_SYSTEM_PROMPT, help="System prompt for training data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load APPS prompt template
    with open(APPS_PROMPT_PATH) as f:
        apps_prompt = f.read()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply fresh LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    dataset = load_training_data(args.data_path, args.system_prompt, apps_prompt, args.max_examples)
    print(f"Training on {len(dataset)} examples for {args.epochs} epochs")
    print(f"  lr={args.lr}, batch_size={args.batch_size}, lora_rank={args.lora_rank}, lora_alpha={args.lora_alpha}")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",  # We save manually via callback
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[EpochSaveCallback(args.output_dir, tokenizer)],
    )

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Checkpoints saved to {args.output_dir}/epoch_*")


if __name__ == "__main__":
    main()
