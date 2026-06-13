"""
Evaluate LoRA checkpoints on HuggingFaceH4/MATH-500 with pass@1 / pass@4.

For each checkpoint we draw `--n-samples` completions per problem (default 4),
extract the \\boxed{} answer, check equivalence against the gold answer, and
report the unbiased pass@k estimators.

Generation runs through vLLM with LoRA adapters, so a single base model is
loaded once and each adapter is swapped in via a LoRARequest.

Usage (on a GPU, e.g. A100-40GB):
    pip install vllm datasets numpy

    # evaluate the three ablation checkpoints
    python evaluate.py \
        --checkpoints \
            checkpoints/sft_original/checkpoint-150 \
            checkpoints/sft_regen_full/checkpoint-150 \
            checkpoints/sft_regen_think/checkpoint-150

    # include the un-tuned base model for reference, quick 20-problem smoke test
    python evaluate.py --checkpoints checkpoints/sft_original/checkpoint-150 \
        --include-base --limit 20
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

from datasets import load_dataset

from utils.extract_answer import extract_boxed_answer, is_equiv
from utils.pass_at_k import compute_pass_at_k_for_dataset

# ── config ─────────────────────────────────────────────────────────────
BASE_MODEL    = "Qwen/Qwen3-0.6B"
DATASET       = "HuggingFaceH4/MATH-500"
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN  = 8192
MAX_LORA_RANK  = 64          # must be >= LoRA r used in training (r=64)

# Qwen3 thinking-mode sampling (per the model card)
TEMPERATURE = 0.6
TOP_P       = 0.95
TOP_K       = 20

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_chats(problems: list[str]):
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        for p in problems
    ]


def evaluate_outputs(outputs, golds: list[str], n: int, k_values: list[int]):
    """Score one model's generations.

    outputs: vLLM RequestOutput list (one per problem, each with n samples).
    Returns (metrics_dict, per_problem_correct, n_no_box).
    """
    per_problem_correct, n_no_box = [], 0
    for out, gold in zip(outputs, golds):
        correct = 0
        for sample in out.outputs:                     # n samples
            pred = extract_boxed_answer(sample.text)
            if pred is None:
                n_no_box += 1
                continue
            if is_equiv(pred, gold):
                correct += 1
        per_problem_correct.append(correct)
    metrics = compute_pass_at_k_for_dataset(per_problem_correct, n=n, k_values=k_values)
    return metrics, per_problem_correct, n_no_box


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="LoRA adapter dirs to evaluate (e.g. .../checkpoint-150)")
    parser.add_argument("--include-base", action="store_true",
                        help="also evaluate the un-tuned base model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="samples drawn per problem (must be >= max k)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N problems (smoke test)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="disable Qwen3 thinking mode")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    if not args.checkpoints and not args.include_base:
        parser.error("provide --checkpoints and/or --include-base")
    max_k = max(args.k_values)
    if args.n_samples < max_k:
        parser.error(f"--n-samples ({args.n_samples}) must be >= max k ({max_k})")

    # 1. load MATH-500 ──────────────────────────────────────────────────
    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    problems = ds["problem"]
    golds = ds["answer"]
    print(f"  {len(problems)} problems")

    chats = build_chats(problems)

    # 2. load base model once, with LoRA enabled ────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    chat_kwargs = {"enable_thinking": not args.no_thinking}

    # 3. build the list of (label, lora_request) to evaluate ────────────
    runs = []
    if args.include_base:
        runs.append(("base", None))
    for i, ckpt in enumerate(args.checkpoints, start=1):
        ckpt = str(ckpt)
        label = Path(ckpt).parent.name or Path(ckpt).name   # e.g. sft_original
        runs.append((label, LoRARequest(label, i, ckpt)))

    # 4. evaluate each ──────────────────────────────────────────────────
    all_results = {}
    for label, lora_req in runs:
        print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
        outputs = llm.chat(
            chats,
            sampling,
            chat_template_kwargs=chat_kwargs,
            lora_request=lora_req,
        )
        metrics, per_problem, n_no_box = evaluate_outputs(
            outputs, golds, n=args.n_samples, k_values=args.k_values
        )
        total_samples = len(problems) * args.n_samples
        all_results[label] = {
            "checkpoint": None if lora_req is None else lora_req.lora_path,
            "metrics": metrics,
            "n_problems": len(problems),
            "n_samples": args.n_samples,
            "no_box_rate": n_no_box / total_samples,
        }
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {metric_str}   (no-box rate: {n_no_box/total_samples:.3f})")

    # 5. summary table + save ───────────────────────────────────────────
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    header = f"{'model':<24}" + "".join(f"{f'pass@{k}':>10}" for k in args.k_values)
    print(header)
    print("-" * len(header))
    for label, res in all_results.items():
        row = f"{label:<24}" + "".join(
            f"{res['metrics'][f'pass@{k}']:>10.4f}" for k in args.k_values
        )
        print(row)

    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
