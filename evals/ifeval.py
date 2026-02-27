"""
IFEval (Instruction Following Evaluation) wrapper using Inspect AI.

Uses the existing Tinker + Inspect integration from tinker_cookbook.eval
to run IFEval and extract prompt-level and instruction-level accuracy.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectGenerateConfig
from inspect_ai.model import Model as InspectModel
from inspect_evals.ifeval import ifeval

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

# Constants
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER_NAME = "qwen3_instruct"


async def run_ifeval_single(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    model_name: str = MODEL_NAME,
    renderer_name: str = RENDERER_NAME,
    temperature: float = 0.7,
    max_tokens: int = 10000,
    limit: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Dict:
    """
    Run IFEval on a single model checkpoint.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to inject via GenerateConfig
        model_name: Model name for tokenizer lookup
        renderer_name: Renderer name for tokenization
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        limit: Max number of samples to evaluate (None = all)
        log_dir: Directory for Inspect logs

    Returns:
        Dictionary with:
        - prompt_accuracy: Fraction of prompts where ALL instructions were followed
        - instruction_accuracy: Fraction of individual instructions followed
        - raw_scores: Full scores dict from Inspect
    """
    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=model_name,
        sampling_client=sampling_client,
    )

    model = InspectModel(
        api=api,
        config=InspectGenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_prompt,
        ),
    )

    results = await eval_async(
        tasks=ifeval(),
        model=[model],
        limit=limit,
        debug_errors=True,
        retry_on_error=0,
        fail_on_error=False,
        log_dir=log_dir or os.path.expanduser("~/inspect-logs"),
        max_connections=512,
        log_level="WARNING",
        log_realtime=False,
        log_buffer=1000,
    )

    # Extract metrics
    metrics = {}
    for task_result in results:
        if task_result.results is not None and task_result.results.scores is not None:
            for score_obj in task_result.results.scores:
                for metric_name, metric_val in score_obj.metrics.items():
                    metrics[metric_name] = metric_val.value

    # IFEval reports prompt_level_strict_acc, inst_level_strict_acc, etc.
    prompt_acc = metrics.get("prompt_level_strict_acc", metrics.get("accuracy", 0.0))
    inst_acc = metrics.get("inst_level_strict_acc", 0.0)

    print(f"IFEval — Prompt acc: {prompt_acc:.2%}, Instruction acc: {inst_acc:.2%}")

    return {
        "prompt_accuracy": prompt_acc,
        "instruction_accuracy": inst_acc,
        "raw_scores": metrics,
    }


async def run_ifeval_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    model_name: str = MODEL_NAME,
    renderer_name: str = RENDERER_NAME,
    temperature: float = 0.7,
    max_tokens: int = 10000,
    limit: Optional[int] = None,
    log_dir: Optional[str] = None,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "ifeval",
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Run IFEval evaluation on multiple model paths sequentially.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to inject
        model_name: Model name for tokenizer lookup
        renderer_name: Renderer name for tokenization
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        limit: Max number of samples to evaluate (None = all)
        log_dir: Directory for Inspect logs
        save: Whether to save results to files
        save_dir: Directory to save results to
        save_prefix: Prefix for saved filenames

    Returns:
        Tuple of (prompt_accs, inst_accs, all_results) where:
        - prompt_accs: List of prompt-level accuracy floats
        - inst_accs: List of instruction-level accuracy floats
        - all_results: List of full result dicts for each path
    """
    prompt_accs = []
    inst_accs = []
    all_results = []

    for path in paths:
        path_label = path.split("/")[-1]
        print(f"\n--- IFEval: {path_label} ---")

        sampling_client = service_client.create_sampling_client(model_path=path)
        result = await run_ifeval_single(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            model_name=model_name,
            renderer_name=renderer_name,
            temperature=temperature,
            max_tokens=max_tokens,
            limit=limit,
            log_dir=log_dir,
        )

        prompt_accs.append(result["prompt_accuracy"])
        inst_accs.append(result["instruction_accuracy"])
        all_results.append(result)

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path_label}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results stored at {filepath}")

    return prompt_accs, inst_accs, all_results
