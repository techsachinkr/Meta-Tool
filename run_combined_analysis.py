#!/usr/bin/env python3
"""
Combined Few-Shot Sensitivity + Error Categorization
=====================================================

Single script that:
1. Tests performance at 0, 1, 2, 3, 4, 5 examples
2. Collects all failure cases during evaluation
3. Auto-categorizes errors (format, semantic, hallucinated)
4. Outputs both sensitivity table AND error analysis

Usage:
    python run_combined_analysis.py --checkpoint checkpoints/best.pt
    python run_combined_analysis.py --quick  # Fast test

Output:
    - Few-shot sensitivity table
    - Error categorization summary
    - CSV of all failures for manual review
    - JSON with complete results
"""

import os
import json
import csv
import re
import argparse
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Global model cache
_model_cache = None


@dataclass
class FailureCase:
    """Single failure case for analysis."""
    task_id: str
    benchmark: str
    n_examples: int
    query: str
    expected: str
    prediction: str
    error_message: str
    auto_category: str


def auto_categorize_failure(
    prediction: str,
    expected: str,
    error_message: str,
    benchmark: str
) -> str:
    """
    Automatically categorize failure type.
    
    Categories:
    - FORMAT_ERROR: Invalid JSON, markdown artifacts, wrong structure
    - SEMANTIC_ERROR: Valid format but wrong content
    - HALLUCINATED_API: Made up function/model names
    - EMPTY_RESPONSE: No meaningful output
    """
    pred = prediction.strip()
    
    # Empty/minimal response
    if len(pred) < 5:
        return "EMPTY_RESPONSE"
    
    # Markdown artifacts
    if "```" in pred or pred.startswith("#"):
        return "FORMAT_ERROR"
    
    if benchmark == "gorilla":
        valid_patterns = [
            r'torchvision\.models\.\w+',
            r'torch\.hub\.load',
            r'pipeline\s*\(',
            r'from_pretrained',
        ]
        has_valid = any(re.search(p, pred) for p in valid_patterns)
        
        if not has_valid:
            return "FORMAT_ERROR"
        if any(x in pred.lower() for x in ['fake', 'nonexistent', 'unknown']):
            return "HALLUCINATED_API"
        return "SEMANTIC_ERROR"
    
    elif benchmark == "spider2":
        if not any(kw in pred.lower() for kw in ['select', 'from']):
            return "FORMAT_ERROR"
        return "SEMANTIC_ERROR"
    
    elif benchmark == "webarena":
        try:
            json.loads(pred)
            return "SEMANTIC_ERROR"
        except:
            return "FORMAT_ERROR"
    
    elif benchmark == "intercode":
        commands = ['find', 'grep', 'ls', 'cat', 'wc', 'sed', 'tar', 'chmod', 'mkdir', 'cp', 'mv', 'rm']
        first = pred.split()[0].lower() if pred.split() else ""
        if first in commands:
            return "SEMANTIC_ERROR"
        return "FORMAT_ERROR"
    
    return "OTHER"


def load_model(checkpoint_path: Optional[str] = None):
    """Load model with caching."""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    from config import get_config
    from hypernetwork import create_hypernetwork
    from lora_integration import MetaToolAdaptedModel
    
    config = get_config("large")
    
    print("Loading model...")
    hypernetwork = create_hypernetwork(config.model)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.model.device,
            weights_only=False
        )
        hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        print("✓ Checkpoint loaded")
    
    _model_cache = MetaToolAdaptedModel(config.model, hypernetwork)
    return _model_cache


def run_evaluation(
    benchmark: str,
    n_examples: int,
    num_tasks: int,
    checkpoint_path: Optional[str]
) -> Tuple[Dict, List[FailureCase]]:
    """
    Run evaluation and collect failures.
    
    Returns:
        (results_dict, list_of_failures)
    """
    from config import get_config
    from evaluation import (
        GorillaEvaluator, Spider2Evaluator,
        WebArenaEvaluator, InterCodeEvaluator
    )
    
    config = get_config("large")
    adapted_model = load_model(checkpoint_path)
    
    evaluators = {
        "gorilla": GorillaEvaluator(config.data.gorilla_path),
        "spider2": Spider2Evaluator(config.data.spider_path),
        "webarena": WebArenaEvaluator(config.data.webarena_path),
        "intercode": InterCodeEvaluator(config.data.intercode_path),
    }
    
    evaluator = evaluators[benchmark]
    tasks = evaluator.load_tasks()[:num_tasks]
    documentation, schema, all_examples = evaluator.get_tool_spec()
    
    # Build prompt with n examples (matching MetaToolEvaluator format)
    examples = all_examples[:n_examples]

    examples_text = "\n".join([
        f"Query: {q}\nOutput: {a}" for q, a in examples
    ]) if examples else ""

    prompt_prefix = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + documentation.strip() + "\n\n"
        + "Examples:\n" + examples_text
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    
    prompt_suffix = (
        "\n\nRespond with ONLY the output, exactly like the examples above. No explanation."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    successes = 0
    failures = []

    for i, task in enumerate(tqdm(tasks, desc=f"{benchmark} {n_examples}-shot", leave=False)):
        prompt = prompt_prefix + task['query'] + prompt_suffix

        prediction = adapted_model.generate(
            prompt,
            max_new_tokens=config.inference.max_new_tokens,
            do_sample=False,
            temperature=1.0
        )

        success, message = evaluator.execute_and_evaluate(task, prediction, strict=True)

        # Extract expected using same logic as each evaluator
        if benchmark == "gorilla":
            expected = task.get("expected", task.get("expected_output", task.get("answer", "")))
        elif benchmark == "spider2":
            expected = task.get("expected_sql", task.get("expected", ""))
        elif benchmark == "webarena":
            expected = task.get("expected_actions", [])
        elif benchmark == "intercode":
            expected = task.get("expected_command", task.get("expected", ""))
        else:
            expected = task.get("expected", "")

        # # Log every task result
        # status = "OK" if success else "FAIL"
        # print(f"\n  [{status}] Task {i}: {task['query'][:80]}")
        # print(f"    Pred: {repr(prediction[:200])}")
        # print(f"    Exp:  {repr(str(expected)[:200])}")
        # print(f"    Msg:  {message}")

        if success:
            successes += 1
        else:
            category = auto_categorize_failure(prediction, str(expected), message or "", benchmark)
            print(f"    Cat:  {category}")
            failures.append(FailureCase(
                task_id=task.get("id", "unknown"),
                benchmark=benchmark,
                n_examples=n_examples,
                query=task['query'][:200],
                expected=str(expected)[:300],
                prediction=prediction[:300],
                error_message=message or "",
                auto_category=category
            ))
    
    result = {
        "success_rate": successes / len(tasks) if tasks else 0.0,
        "num_success": successes,
        "num_tasks": len(tasks),
        "num_failures": len(failures),
    }
    
    return result, failures


def run_combined_analysis(
    benchmarks: List[str] = ["gorilla", "spider2", "webarena", "intercode"],
    example_counts: List[int] = [0, 1, 2, 3, 4, 5],
    num_tasks: int = 50,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "./analysis_results"
):
    """Run combined sensitivity + error analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("COMBINED ANALYSIS: Few-Shot Sensitivity + Error Categorization")
    print("=" * 60)
    print(f"Benchmarks: {benchmarks}")
    print(f"Example counts: {example_counts}")
    print(f"Tasks per config: {num_tasks}")
    print("=" * 60)
    
    # Store results
    sensitivity_results = {b: {} for b in benchmarks}
    all_failures = []
    
    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"Benchmark: {benchmark.upper()}")
        print(f"{'='*50}")
        
        for n_ex in example_counts:
            result, failures = run_evaluation(
                benchmark=benchmark,
                n_examples=n_ex,
                num_tasks=num_tasks,
                checkpoint_path=checkpoint_path
            )
            
            sensitivity_results[benchmark][n_ex] = result
            all_failures.extend(failures)
            
            print(f"  {n_ex}-shot: {result['success_rate']*100:.1f}% "
                  f"({result['num_success']}/{result['num_tasks']}, "
                  f"{result['num_failures']} failures)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON with everything
    json_file = output_path / f"combined_analysis_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump({
            "sensitivity_results": sensitivity_results,
            "failures_count": len(all_failures),
            "config": {
                "benchmarks": benchmarks,
                "example_counts": example_counts,
                "num_tasks": num_tasks,
            },
            "timestamp": timestamp,
        }, f, indent=2)
    
    # CSV for manual error review
    csv_file = output_path / f"failure_cases_{timestamp}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "benchmark", "n_examples", "query", "expected", 
            "prediction", "error_message", "auto_category"
        ])
        writer.writeheader()
        for failure in all_failures:
            writer.writerow(asdict(failure))
    
    print(f"\n✓ Saved results to {json_file}")
    print(f"✓ Saved {len(all_failures)} failures to {csv_file}")
    
    # Print tables
    print_sensitivity_table(sensitivity_results, example_counts, benchmarks)
    print_error_summary(all_failures, benchmarks, example_counts)
    print_latex_tables(sensitivity_results, all_failures, example_counts, benchmarks)
    
    return sensitivity_results, all_failures


def print_sensitivity_table(results: Dict, example_counts: List[int], benchmarks: List[str]):
    """Print few-shot sensitivity table."""
    print("\n" + "=" * 70)
    print("TABLE 1: FEW-SHOT SENSITIVITY")
    print("=" * 70)
    
    header = "| # Examples |" + "".join(f" {b.capitalize()} |" for b in benchmarks) + " Avg |"
    print(header)
    print("|" + "---|" * (len(benchmarks) + 2))
    
    for n_ex in example_counts:
        row = f"| {n_ex}-shot |"
        rates = []
        for b in benchmarks:
            rate = results[b][n_ex]["success_rate"] * 100
            rates.append(rate)
            row += f" {rate:.1f}% |"
        row += f" {sum(rates)/len(rates):.1f}% |"
        print(row)


def print_error_summary(failures: List[FailureCase], benchmarks: List[str], example_counts: List[int]):
    """Print error categorization summary."""
    print("\n" + "=" * 70)
    print("TABLE 2: ERROR CATEGORIZATION (5-shot failures)")
    print("=" * 70)
    
    categories = ["FORMAT_ERROR", "SEMANTIC_ERROR", "HALLUCINATED_API", "EMPTY_RESPONSE", "OTHER"]
    
    # Only analyze 5-shot failures (or max example count)
    max_ex = max(example_counts)
    filtered = [f for f in failures if f.n_examples == max_ex]
    
    counts = {b: {c: 0 for c in categories} for b in benchmarks}
    totals = {b: 0 for b in benchmarks}
    
    for f in filtered:
        counts[f.benchmark][f.auto_category] += 1
        totals[f.benchmark] += 1
    
    header = "| Category |" + "".join(f" {b.capitalize()} |" for b in benchmarks) + " Total |"
    print(header)
    print("|" + "---|" * (len(benchmarks) + 2))
    
    for cat in categories:
        cat_total = sum(counts[b][cat] for b in benchmarks)
        if cat_total > 0:
            row = f"| {cat.replace('_', ' ').title()} |"
            for b in benchmarks:
                n = counts[b][cat]
                pct = (n / totals[b] * 100) if totals[b] > 0 else 0
                row += f" {n} ({pct:.0f}%) |"
            row += f" {cat_total} |"
            print(row)
    
    # Total row
    grand_total = sum(totals.values())
    row = "| **Total Failures** |"
    for b in benchmarks:
        row += f" {totals[b]} |"
    row += f" {grand_total} |"
    print(row)


def print_latex_tables(results: Dict, failures: List[FailureCase], 
                       example_counts: List[int], benchmarks: List[str]):
    """Print LaTeX tables for paper."""
    print("\n" + "=" * 70)
    print("LATEX TABLES (copy to paper)")
    print("=" * 70)
    
    # Sensitivity table
    print(r"""
% Table 1: Few-Shot Sensitivity
\begin{table}[h]
\centering
\caption{Few-Shot Sensitivity: Performance vs. Number of Examples}
\label{tab:fewshot_sensitivity}
\begin{tabular}{lccccc}
\toprule
\# Examples & Gorilla & Spider2 & WebArena & InterCode & Avg \\
\midrule""")
    
    for n_ex in example_counts:
        rates = [results[b][n_ex]["success_rate"] * 100 for b in benchmarks]
        avg = sum(rates) / len(rates)
        print(f"{n_ex}-shot & {rates[0]:.1f} & {rates[1]:.1f} & {rates[2]:.1f} & {rates[3]:.1f} & {avg:.1f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # Error categorization table
    categories = ["FORMAT_ERROR", "SEMANTIC_ERROR", "HALLUCINATED_API"]
    max_ex = max(example_counts)
    filtered = [f for f in failures if f.n_examples == max_ex]
    
    counts = {b: {c: 0 for c in categories} for b in benchmarks}
    for f in filtered:
        if f.auto_category in categories:
            counts[f.benchmark][f.auto_category] += 1
    
    print(r"""
% Table 2: Error Categorization
\begin{table}[h]
\centering
\caption{Error Type Distribution}
\label{tab:error_types}
\begin{tabular}{lcccc|c}
\toprule
Error Type & Gorilla & Spider2 & WebArena & InterCode & Total \\
\midrule""")
    
    for cat in categories:
        cat_total = sum(counts[b][cat] for b in benchmarks)
        if cat_total > 0:
            name = cat.replace("_", " ").title()
            print(f"{name} & {counts['gorilla'][cat]} & {counts['spider2'][cat]} & {counts['webarena'][cat]} & {counts['intercode'][cat]} & {cat_total} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    parser = argparse.ArgumentParser(description="Combined Few-Shot + Error Analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint")
    parser.add_argument("--num-tasks", type=int, default=50,
                        help="Tasks per benchmark per config")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["gorilla", "spider2", "webarena", "intercode"])
    parser.add_argument("--example-counts", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (gorilla only, 10 tasks, 0/3/5 examples)")
    parser.add_argument("--output-dir", type=str, default="./analysis_results")
    
    args = parser.parse_args()
    
    if args.quick:
        args.benchmarks = ["gorilla"]
        args.num_tasks = 10
        args.example_counts = [0, 3, 5]
    
    run_combined_analysis(
        benchmarks=args.benchmarks,
        example_counts=args.example_counts,
        num_tasks=args.num_tasks,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
