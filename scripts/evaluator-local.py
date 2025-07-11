"""Local Model Evaluator
Modified and Borrowed from: https://github.com/tplr-ai/templar/blob/main/scripts/evaluator-local.py

This script evaluates model checkpoints from local files using standardized benchmark tasks.
Instead of pulling checkpoints from the Bittensor network, it accepts checkpoint paths as arguments.

Usage:
    python scripts/evaluator-local.py --checkpoint_root /path/to/checkpoints/
    python scripts/evaluator-local.py --checkpoint_root /path/to/checkpoints/ --tasks arc_challenge,winogrande
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
from transformers.models.llama import LlamaForCausalLM

import tplr

MODEL_PATH: str = "models/eval"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local evaluator script for model checkpoints"
    )

    parser.add_argument('--hparams_file', type=str, default='hparams/1B/1B_model_hparams.json', help='hparams file.')
    
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        required=True,
        help="Root directory containing model checkpoint files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate (0.0-1.0)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up model files after evaluation",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="checkpoint-evaluation",
        help="Wandb project name for logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to timestamp)",
    )

    return parser.parse_args()


def find_checkpoints(checkpoint_root: str) -> list[tuple[int, str]]:
    """Find and sort checkpoint files by window number."""
    checkpoint_dir = Path(checkpoint_root)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")
    
    pattern = re.compile(r"demo_checkpoint_window_(\d+)_\d{8}_\d{6}\.pt")
    checkpoints = []
    
    for file_path in checkpoint_dir.glob("*.pt"):
        match = pattern.match(file_path.name)
        if match:
            window_num = int(match.group(1))
            checkpoints.append((window_num, str(file_path)))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_root}")
    
    # Sort by window number in descending order (latest first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    tplr.logger.info(f"Found {len(checkpoints)} checkpoints, latest window: {checkpoints[0][0]}")
    return checkpoints


def load_checkpoint(checkpoint_path: str, hparams_file: str, device: str) -> tuple[LlamaForCausalLM, dict]:
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")

    hparams_file = os.path.expandvars(os.path.expanduser(hparams_file))
    hparams = tplr.load_hparams(hparams_file)

    model = LlamaForCausalLM(config=hparams.model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        print("Loading model state from checkpoint")
        model_state = {k.replace("_orig_mod.", ""): v for k,v in checkpoint["model_state_dict"].items()}
        print("Loading metadata from checkpoint")
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        model_state = checkpoint
        metadata = {}

    model.load_state_dict(model_state)
    model.to("cpu")

    model.load_state_dict(
        {
            k: v.to("cpu")
            for k, v in model_state.items()  # type: ignore
        }
    )
    model.to("cpu")  # type: ignore

    tplr.logger.info("Model loaded successfully")
    return model, metadata


def run_evaluation(
    model: LlamaForCausalLM,
    hparams: SimpleNamespace,
    args: argparse.Namespace,
    window_num: int,
) -> dict:
    """Run lm-eval benchmark and return results."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    hparams.tokenizer.save_pretrained(MODEL_PATH)

    if args.limit < 1.0:
        limit_arg = f"--limit {args.limit}"
    else:
        limit_arg = ""

    if args.num_fewshot > 0:
        fewshot_arg = f"--num_fewshot {args.num_fewshot}"
    else:
        fewshot_arg = ""

    command = f"""
    lm-eval --model hf \
        --model_args pretrained={MODEL_PATH},tokenizer={MODEL_PATH} \
        --tasks {args.tasks} \
        --device {args.device} \
        --batch_size {args.batch_size} \
        --output_path {args.output_dir} \
        {limit_arg} \
        {fewshot_arg}
    """.strip()

    tplr.logger.info(f"Running benchmark for window {window_num}: {command}")
    start_time = time.time()
    exit_code = os.system(command)
    runtime = time.time() - start_time

    if exit_code != 0:
        raise RuntimeError(f"Evaluation failed with exit code {exit_code}")

    results_dir = Path(args.output_dir) / "models__eval"
    latest_file = max(results_dir.glob("*.json"), key=os.path.getctime)

    with open(latest_file, "r") as f:
        results = json.load(f)

    if args.cleanup:
        tplr.logger.info("Cleaning up model files")
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        torch.cuda.empty_cache()
    else:
        tplr.logger.info(f"Model files kept at: {MODEL_PATH}")

    return {
        "window": window_num,
        "benchmark_runtime": runtime,
        "results": results["results"],
        "config": args.__dict__,
    }


def log_to_wandb(results: dict) -> None:
    """Log evaluation results to wandb."""
    log_data = {
        "window": results["window"],
        "benchmark_runtime": results["benchmark_runtime"],
    }
    
    # Extract task scores with priority for acc_norm,none then acc,none
    for task_name, task_results in results["results"].items():
        metric_names = ["acc_norm,none", "acc,none"]
        
        for metric_name in metric_names:
            if (value := task_results.get(metric_name)) is not None:
                log_data[f"{task_name}_{metric_name.replace(',', '_')}"] = value
                break
    
    wandb.log(log_data)
    tplr.logger.info(f"Logged results for window {results['window']} to wandb")


def print_results(results: dict) -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS - WINDOW {results['window']}")
    print("=" * 50)

    print(f"\nRuntime: {results['benchmark_runtime']:.2f} seconds")
    print(f"Config: {json.dumps(results['config'], indent=2)}")

    print("\nTask Scores:")
    print("-" * 30)

    for task_name, task_results in results["results"].items():
        # Priority order for metrics
        metric_names = ["acc_norm,none", "acc,none"]

        for metric_name in metric_names:
            if (value := task_results.get(metric_name)) is not None:
                print(f"{task_name} ({metric_name}): {value:.4f}")
                break

    print("=" * 50 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"eval_{int(time.time())}",
        config=args.__dict__
    )

    try:
        hparams = tplr.load_hparams(args.hparams_file)
        
        # Find all checkpoints and sort by window number
        checkpoints = find_checkpoints(args.checkpoint_root)
        
        all_results = []
        
        for window_num, checkpoint_path in checkpoints:
            tplr.logger.info(f"Evaluating checkpoint window {window_num}: {checkpoint_path}")
            
            model, metadata = load_checkpoint(checkpoint_path, args.hparams_file, args.device)

            if metadata:
                tplr.logger.info(f"Checkpoint metadata: {metadata}")

            results = run_evaluation(model, hparams, args, window_num)
            
            print_results(results)
            log_to_wandb(results)
            
            all_results.append(results)
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache()

        # Save all results
        output_file = Path(args.output_dir) / "evaluation_summary_all.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        tplr.logger.info(f"All results saved to {output_file}")
        wandb.finish()

    except Exception as e:
        tplr.logger.error(f"Evaluation failed: {e}")
        wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()