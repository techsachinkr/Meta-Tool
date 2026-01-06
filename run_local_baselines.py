#!/usr/bin/env python3
"""
Baseline Evaluation Script for Local HuggingFace Models
========================================================

Evaluates baseline models on all benchmarks:
- AgentLM-7B (zai-org/agentlm-7b)
- Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct)
- Any other HuggingFace model

Usage:
    # Run with AgentLM-7B
    python run_local_baselines.py --model zai-org/agentlm-7b
    
    # Run with Llama 3.2 3B
    python run_local_baselines.py --model meta-llama/Llama-3.2-3B-Instruct
    
    # Run specific benchmarks
    python run_local_baselines.py --model zai-org/agentlm-7b --benchmarks gorilla spider2
    
    # Run with strict evaluation
    python run_local_baselines.py --model zai-org/agentlm-7b --strict-eval
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import evaluators
from evaluation import (
    GorillaEvaluator,
    Spider2Evaluator,
    WebArenaEvaluator,
    InterCodeEvaluator,
)


@dataclass
class BaselineResult:
    """Result from baseline evaluation."""
    model_name: str
    benchmark: str
    method: str
    success_rate: float
    avg_latency_ms: float
    num_tasks: int


class LocalModelEvaluator:
    """Evaluator for local HuggingFace models."""
    
    # Model configurations
    MODEL_CONFIGS = {
        "zai-org/agentlm-7b": {
            "prompt_format": "agentlm",
            "max_new_tokens": 256,
        },
        "meta-llama/Llama-3.2-3B-Instruct": {
            "prompt_format": "llama3",
            "max_new_tokens": 128,
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "prompt_format": "llama3",
            "max_new_tokens": 128,
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "prompt_format": "mistral",
            "max_new_tokens": 128,
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "prompt_format": "qwen",
            "max_new_tokens": 128,
        },
    }
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Get config or use default
        self.config = self.MODEL_CONFIGS.get(model_name, {
            "prompt_format": "default",
            "max_new_tokens": 128,
        })
        
    def load_model(self):
        """Load model with 4-bit quantization."""
        print(f"\nLoading {self.model_name}...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully")
        return self
    
    def format_prompt(self, system: str, user: str, examples: List[Tuple[str, str]] = None) -> str:
        """Format prompt based on model type."""
        prompt_format = self.config["prompt_format"]
        
        # Build examples text
        examples_text = ""
        if examples:
            examples_text = "\n\nExamples:\n" + "\n".join([
                f"Query: {q}\nOutput: {a}" for q, a in examples[:3]
            ])
        
        system_with_examples = system + examples_text
        
        if prompt_format == "llama3":
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_with_examples}"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}\n\nOutput ONLY the exact answer, nothing else."
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif prompt_format == "qwen":
            return (
                f"<|im_start|>system\n{system_with_examples}<|im_end|>\n"
                f"<|im_start|>user\n{user}\n\nOutput ONLY the exact answer, nothing else.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif prompt_format == "mistral":
            return f"[INST] {system_with_examples}\n\n{user}\n\nOutput ONLY the exact answer, nothing else. [/INST]"
        elif prompt_format == "agentlm":
            # AgentLM format - designed for agent tasks
            return (
                f"### System:\n{system_with_examples}\n\n"
                f"### User:\n{user}\n\nOutput ONLY the exact answer, nothing else.\n\n"
                f"### Assistant:\n"
            )
        else:
            # Default format
            return f"System: {system_with_examples}\n\nUser: {user}\n\nAssistant:"
    
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate response from model."""
        if max_new_tokens is None:
            max_new_tokens = self.config["max_new_tokens"]
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()


def evaluate_model_on_benchmark(
    model: LocalModelEvaluator,
    benchmark_name: str,
    evaluator,
    num_tasks: int = 50,
    strict: bool = False
) -> BaselineResult:
    """Evaluate a model on a single benchmark."""
    
    # Load tasks
    tasks = evaluator.load_tasks()[:num_tasks]
    
    # Get tool spec
    documentation, schema, examples = evaluator.get_tool_spec()
    
    successes = 0
    total_latency = 0.0
    
    print(f"\nEvaluating {model.model_name} on {benchmark_name}...")
    
    for i, task in enumerate(tqdm(tasks, desc=benchmark_name)):
        start_time = time.time()
        
        # Format prompt
        prompt = model.format_prompt(
            system=documentation.strip(),
            user=task['query'],
            examples=examples
        )
        
        # Generate prediction
        prediction = model.generate(prompt)
        
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        # Evaluate
        success, message = evaluator.execute_and_evaluate(task, prediction, strict=strict)
        
        if success:
            successes += 1
        
        # Show first few samples
        if i < 3:
            expected = task.get("expected", task.get("expected_sql", task.get("expected_command", "")))
            status = "✓" if success else "✗"
            print(f"\n[{status}] Sample {i+1}:")
            print(f"  Query: {task['query'][:60]}...")
            print(f"  Prediction: {prediction[:80]}...")
            print(f"  Expected: {str(expected)[:80]}...")
    
    success_rate = successes / len(tasks) * 100
    avg_latency = total_latency / len(tasks)
    
    print(f"\n{benchmark_name} Results:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    
    return BaselineResult(
        model_name=model.model_name,
        benchmark=benchmark_name,
        method="few-shot",
        success_rate=success_rate,
        avg_latency_ms=avg_latency,
        num_tasks=len(tasks)
    )


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation with local HuggingFace models")
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name (e.g., zai-org/agentlm-7b)")
    parser.add_argument("--benchmarks", nargs="+", 
                       default=["gorilla", "spider2", "webarena", "intercode"],
                       help="Benchmarks to evaluate")
    parser.add_argument("--num-tasks", type=int, default=50,
                       help="Number of tasks per benchmark")
    parser.add_argument("--strict-eval", action="store_true",
                       help="Use strict evaluation mode")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize model
    model = LocalModelEvaluator(args.model)
    model.load_model()
    
    # Initialize evaluators
    evaluators = {
        "gorilla": GorillaEvaluator("./data/gorilla"),
        "spider2": Spider2Evaluator("./data/spider2"),
        "webarena": WebArenaEvaluator("./data/webarena"),
        "intercode": InterCodeEvaluator("./data/intercode"),
    }
    
    # Run evaluations
    results = []
    
    for benchmark in args.benchmarks:
        if benchmark not in evaluators:
            print(f"Unknown benchmark: {benchmark}")
            continue
        
        result = evaluate_model_on_benchmark(
            model=model,
            benchmark_name=benchmark,
            evaluator=evaluators[benchmark],
            num_tasks=args.num_tasks,
            strict=args.strict_eval
        )
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE COMPARISON RESULTS")
    print("="*60)
    print(f"\nModel: {args.model}")
    print(f"Evaluation Mode: {'Strict' if args.strict_eval else 'Lenient'}")
    print("-"*60)
    print(f"{'Benchmark':<15} {'Success%':>12} {'Latency':>12}")
    print("-"*60)
    
    for r in results:
        print(f"{r.benchmark:<15} {r.success_rate:>11.1f}% {r.avg_latency_ms:>10.1f}ms")
    
    avg_success = sum(r.success_rate for r in results) / len(results) if results else 0
    avg_latency = sum(r.avg_latency_ms for r in results) / len(results) if results else 0
    print("-"*60)
    print(f"{'AVERAGE':<15} {avg_success:>11.1f}% {avg_latency:>10.1f}ms")
    print("="*60)
    
    # Save results
    output = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "strict_eval": args.strict_eval,
        "results": [
            {
                "model": r.model_name,
                "benchmark": r.benchmark,
                "method": r.method,
                "success_rate": r.success_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "num_tasks": r.num_tasks
            }
            for r in results
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
