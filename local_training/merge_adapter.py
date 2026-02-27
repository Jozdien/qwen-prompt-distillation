"""
Merge Tinker LoRA adapter into base Qwen3 model weights.

Handles MoE-aware LoRA merging: 3D batched matmul for expert layers, fused
gate_up_proj, per-expert vs shared LoRA, Tinker->HF name remapping.

Merge logic adapted from tinker-cookbook's merge_tinker_adapter_to_hf_model.py.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapter")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "merged_model")


# ---------------------------------------------------------------------------
# Merge utilities (from tinker-cookbook)
# ---------------------------------------------------------------------------


def log(s: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}")


def load_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    if "vision_config" in config:
        from transformers import AutoModelForImageTextToText
        auto_cls = AutoModelForImageTextToText
    else:
        auto_cls = AutoModelForCausalLM
    return auto_cls.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)


def load_adapter_weights(adapter_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = load_file(
        os.path.expanduser(adapter_path + "/adapter_model.safetensors"), device=device
    )
    with open(os.path.expanduser(adapter_path + "/adapter_config.json"), "r") as f:
        config = json.load(f)
    return weights, config


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor):
    assert target.shape == merged_lora.shape, (target.shape, merged_lora.shape)
    new_data = target.float() + merged_lora.float().to(target.device)
    target.copy_(new_data.to(target.dtype))


def merge_adapter_weights(
    base_model: torch.nn.Module, adapter_weights: dict, config: dict
):
    scaling = config["lora_alpha"] / config["r"]
    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

    model_state_dict = base_model.state_dict()
    is_gpt_oss = "GptOss" in str(type(base_model))
    is_fused_experts = any(k.endswith(".experts.gate_up_proj") for k in model_state_dict)
    name_remaps = {
        "base_model.model.": "",
        "model.unembed_tokens": "lm_head",
    }
    if any(k.startswith("model.language_model.") for k in model_state_dict):
        name_remaps["model."] = "model.language_model."

    for n in adapter_weight_names:
        target_key = n
        for old, new in name_remaps.items():
            target_key = target_key.replace(old, new)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling
        if ".experts" not in n:
            if is_gpt_oss:
                target_key = target_key.replace(".attn", ".self_attn")
            assert target_key in model_state_dict, (n, target_key)
            merged_lora = torch.nn.functional.linear(lora_A.T, lora_B).T
            assert merged_lora.shape == model_state_dict[target_key].shape, (
                n, merged_lora.shape, model_state_dict[target_key].shape,
            )
            apply_merged_weight(model_state_dict[target_key], merged_lora)
        else:
            # MoE expert layers: 3D batched matmul
            assert len(lora_A.shape) == 3 and len(lora_B.shape) == 3, (lora_A.shape, lora_B.shape)
            if lora_A.shape[0] == 1:
                assert lora_B.shape[0] > 1
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
            elif lora_B.shape[0] == 1:
                assert lora_A.shape[0] > 1
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
            merged_lora = torch.bmm(lora_A.transpose(-1, -2), lora_B.transpose(-1, -2))

            target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
            target_key = target_key.replace(".w3.weight", ".up_proj.weight")
            target_key = target_key.replace(".w2.weight", ".down_proj.weight")

            if not is_fused_experts:
                merged_lora = merged_lora.transpose(-1, -2)
                for exp_idx in range(merged_lora.shape[0]):
                    target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
                    assert target_key_exp in model_state_dict, (n, target_key_exp)
                    assert merged_lora[exp_idx].shape == model_state_dict[target_key_exp].shape, (
                        target_key_exp, merged_lora[exp_idx].shape, model_state_dict[target_key_exp].shape,
                    )
                    apply_merged_weight(model_state_dict[target_key_exp], merged_lora[exp_idx])
            else:
                if target_key.endswith((".gate_proj.weight", ".up_proj.weight")):
                    target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj").replace(
                        ".up_proj.weight", ".gate_up_proj"
                    )
                    idx = 0 if target_key.endswith(".gate_up_proj") else 1
                    if is_gpt_oss:
                        target = model_state_dict[target_key][:, :, idx::2]
                    else:
                        sz = model_state_dict[target_key].shape[-1] // 2
                        target = model_state_dict[target_key][:, :, idx * sz : (idx + 1) * sz]
                else:
                    target_key = target_key.replace(".down_proj.weight", ".down_proj")
                    target = model_state_dict[target_key]
                apply_merged_weight(target, merged_lora)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Merge Tinker LoRA adapter into base Qwen3 model"
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--adapter-path", default=DEFAULT_ADAPTER_PATH,
        help="Path to downloaded Tinker adapter",
    )
    parser.add_argument(
        "--output-path", default=DEFAULT_OUTPUT_PATH,
        help="Path to save merged model",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    log(f"Loading base model: {args.base_model}")
    model = load_model(args.base_model)

    log(f"Loading adapter weights from: {args.adapter_path}")
    adapter_weights, adapter_config = load_adapter_weights(args.adapter_path)
    print(f"  LoRA rank: {adapter_config['r']}, alpha: {adapter_config['lora_alpha']}")
    print(f"  Number of adapter weight tensors: {len(adapter_weights)}")

    log("Merging adapter weights into base model")
    merge_adapter_weights(model, adapter_weights, adapter_config)

    log(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)

    log("Done!")


if __name__ == "__main__":
    main()
