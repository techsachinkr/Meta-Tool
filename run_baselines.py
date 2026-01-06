#!/usr/bin/env python3
"""
Baseline Evaluation Script for Meta-Tool Paper

Evaluates baseline methods for comparison:
1. Zero-shot LLM (no examples, just instruction)
2. Few-shot LLM (in-context examples, no adaptation)  
3. Fine-tuned baselines (if available)
4. Meta-Tool (our method)

Paper baselines to compare against:
- GPT-4 (zero-shot, few-shot)
- GPT-3.5-turbo
- ToolLLM
- Gorilla
- CodeLlama
- Llama-2-Chat
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter

import torch
from tqdm import tqdm

# Import evaluators from evaluation.py
from evaluation import (
    GorillaEvaluator,
    Spider2Evaluator, 
    WebArenaEvaluator,
    InterCodeEvaluator,
    BenchmarkEvaluator
)

# Optional imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class BaselineResult:
    """Result from baseline evaluation."""
    model_name: str
    benchmark: str
    success_rate: float
    avg_latency_ms: float
    num_tasks: int
    method: str  # zero-shot, few-shot, fine-tuned, meta-tool


class BaseModel:
    """Base class for model evaluators."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def load_model(self):
        raise NotImplementedError
        
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError
    
    def format_prompt_zero_shot(self, query: str, documentation: str) -> str:
        """Format zero-shot prompt with documentation."""
        raise NotImplementedError
    
    def format_prompt_few_shot(self, query: str, documentation: str, examples: List[Tuple[str, str]]) -> str:
        """Format few-shot prompt with documentation and examples."""
        raise NotImplementedError


class OpenAIEvaluator(BaseModel):
    """Evaluator using OpenAI API (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: str = None):
        super().__init__(model_name)
        
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Get API key from argument, environment, or prompt
        with open("config.json", "r") as f:
            config = json.load(f)
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass --api-key")
        
        self.client = OpenAI(base_url="https://lightning.ai/api/v1/",api_key=api_key)
        print(f"Initialized OpenAI client for {model_name}")
        
    def load_model(self):
        """No-op for API models."""
        pass
        
    def generate(self, prompt: str, max_new_tokens: int = 10000) -> str:
        """Generate response from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                          {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                          },
                        ],
                max_completion_tokens=max_new_tokens,
                # temperature=0,  # Greedy for reproducibility
            )
            # print(prompt)
            # print(response)
            
            # Handle None content (common with reasoning models like GPT-5)
            content = response.choices[0].message.content
            if content is None:
                print(f"  Warning: API returned content=None (finish_reason: {response.choices[0].finish_reason})")
                return ""
            return content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def format_prompt_zero_shot(self, query: str, documentation: str) -> str:
        """Format zero-shot prompt with documentation."""
        return f"""{documentation.strip()}

Query: {query}

Output ONLY the answer (code/SQL/command/JSON), nothing else. No explanations.

Output:"""
    
    def format_prompt_few_shot(self, query: str, documentation: str, examples: List[Tuple[str, str]]) -> str:
        """Format few-shot prompt with documentation and examples."""
        examples_text = "\n\n".join([
            f"Query: {q}\nOutput: {a}" for q, a in examples
        ])
        
        return f"""{documentation.strip()}

Examples:
{examples_text}

Query: {query}

Output ONLY the answer (code/SQL/command/JSON), nothing else. No explanations.

Output:"""


class LocalModelEvaluator(BaseModel):
    """Evaluator using local HuggingFace models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name)
        self.device = device
        self.model = None
        self.tokenizer = None
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package not installed. Run: pip install transformers")
        
    def load_model(self):
        """Load the model with quantization."""
        print(f"\nLoading {self.model_name}...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Quantization config for memory efficiency
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
        
        print(f"Loaded {self.model_name}")
        
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from local model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def format_prompt_zero_shot(self, query: str, documentation: str) -> str:
        """Format zero-shot prompt with Llama-3 format."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{documentation.strip()}

Output ONLY the answer (code/SQL/command/JSON), nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def format_prompt_few_shot(self, query: str, documentation: str, examples: List[Tuple[str, str]]) -> str:
        """Format few-shot prompt with Llama-3 format."""
        examples_text = "\n\n".join([
            f"Query: {q}\nOutput: {a}" for q, a in examples
        ])
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{documentation.strip()}

Examples:
{examples_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

Output ONLY the answer, nothing else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_benchmark_evaluator(benchmark: str, data_path: str = "./data") -> BenchmarkEvaluator:
    """Get the appropriate benchmark evaluator."""
    evaluators = {
        "gorilla": GorillaEvaluator(os.path.join(data_path, "gorilla")),
        "spider2": Spider2Evaluator(os.path.join(data_path, "spider2")),
        "webarena": WebArenaEvaluator(os.path.join(data_path, "webarena")),
        "intercode": InterCodeEvaluator(os.path.join(data_path, "intercode")),
    }
    
    if benchmark not in evaluators:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(evaluators.keys())}")
    
    return evaluators[benchmark]


def run_baseline_evaluation(
    model: BaseModel,
    benchmark: str,
    method: str,  # "zero-shot" or "few-shot"
    num_tasks: int = 50,
    data_path: str = "./data",
    strict_eval: bool = False
) -> BaselineResult:
    """Run evaluation for a single baseline method using same evaluators as main eval."""
    
    # Get benchmark evaluator (same as evaluate_benchmark in evaluation.py)
    evaluator = get_benchmark_evaluator(benchmark, data_path)
    
    # Load tasks
    tasks = evaluator.load_tasks()
    if num_tasks and num_tasks < len(tasks):
        tasks = tasks[:num_tasks]
    
    # Get tool specification (documentation, schema, examples)
    documentation, schema, examples = evaluator.get_tool_spec()
    
    successes = 0
    total_latency = 0.0
    failure_reasons = Counter()
    
    eval_mode = "STRICT" if strict_eval else "LENIENT"
    print(f"\nEvaluating {model.model_name} ({method}) on {benchmark} [{eval_mode}]...")
    print(f"  Tasks: {len(tasks)}, Examples: {len(examples)}")
    
    for i, task in enumerate(tqdm(tasks, desc=f"{method}/{benchmark}")):
        start_time = time.time()
        
        # Get query from task
        query = task.get("query", "")
        
        # Format prompt based on method
        if method == "zero-shot":
            prompt = model.format_prompt_zero_shot(query, documentation)
        else:  # few-shot
            prompt = model.format_prompt_few_shot(query, documentation, examples)
        
        try:
            prediction = model.generate(prompt)
        except Exception as e:
            print(f"Generation error: {e}")
            prediction = ""
            
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        # Use the benchmark's own evaluator with strict mode
        success, reason = evaluator.execute_and_evaluate(task, prediction, strict=strict_eval)
        
        if success:
            successes += 1
        else:
            failure_reasons[reason] += 1
        
        # Show first few samples
        if i < 3:
            expected = task.get("expected", task.get("expected_sql", task.get("expected_command", "")))
            print(f"\n[Sample {i+1}]")
            print(f"  Query: {query[:70]}...")
            print(f"  Prediction: {prediction[:120]}...")
            print(f"  Expected: {str(expected)[:80]}...")
            print(f"  Result: {'✓' if success else '✗'} ({reason})")
    
    success_rate = successes / len(tasks) * 100
    avg_latency = total_latency / len(tasks)
    
    print(f"\nResults for {model.model_name} ({method}) on {benchmark}:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    if failure_reasons:
        print(f"  Top failures: {dict(failure_reasons.most_common(3))}")
    
    return BaselineResult(
        model_name=model.model_name,
        benchmark=benchmark,
        success_rate=success_rate,
        avg_latency_ms=avg_latency,
        num_tasks=len(tasks),
        method=method
    )
def generate_comparison_table(results: List[BaselineResult], meta_tool_results: Dict = None):
    """Generate comparison table for paper."""
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    
    # Organize by benchmark
    benchmarks = ["gorilla", "spider2", "webarena", "intercode"]
    methods = ["zero-shot", "few-shot"]
    
    # Get unique models
    models = list(set(r.model_name for r in results))
    
    # Header
    print(f"\n{'Benchmark':<12} {'Method':<12} {'Model':<30} {'Success%':<10} {'Latency':<10}")
    print("-"*80)
    
    for benchmark in benchmarks:
        for result in results:
            if result.benchmark == benchmark:
                model_short = result.model_name.split('/')[-1][:28]
                print(f"{benchmark:<12} {result.method:<12} {model_short:<30} {result.success_rate:>7.1f}%  {result.avg_latency_ms:>7.1f}ms")
        
        # Add Meta-Tool result if available
        if meta_tool_results and benchmark in meta_tool_results:
            mt = meta_tool_results[benchmark]
            print(f"{benchmark:<12} {'meta-tool':<12} {'Meta-Tool (Ours)':<30} {mt['success_rate']:>7.1f}%  {mt['latency']:>7.1f}ms")
        
        print()
    
   
    print("-"*80)
    print("YOUR RESULTS:")
    print("-"*80)
    
    # Group results by model and method
    for model in models:
        model_short = model.split('/')[-1][:20]
        for method in methods:
            method_results = {r.benchmark: r.success_rate for r in results 
                           if r.method == method and r.model_name == model}
            if method_results:
                label = f"{model_short} ({method})"
                print(f"{label:<30} {method_results.get('gorilla', 0):>9.1f}%  {method_results.get('spider2', 0):>9.1f}%  {method_results.get('webarena', 0):>9.1f}%  {method_results.get('intercode', 0):>9.1f}%")
    
    if meta_tool_results:
        print(f"{'Meta-Tool (Ours)':<30} {meta_tool_results.get('gorilla', {}).get('success_rate', 0):>9.1f}%  {meta_tool_results.get('spider2', {}).get('success_rate', 0):>9.1f}%  {meta_tool_results.get('webarena', {}).get('success_rate', 0):>9.1f}%  {meta_tool_results.get('intercode', {}).get('success_rate', 0):>9.1f}%")
    
    print("-"*80)
    print("\n* Paper baseline values (from Meta-Tool paper Table 1)")
    print("Results without * are from your evaluation runs")


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Model to evaluate (e.g., gpt-4, gpt-3.5-turbo, meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["gorilla", "spider2", "webarena", "intercode"],
                       help="Benchmarks to evaluate")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["zero-shot", "few-shot"],
                       help="Methods to evaluate")
    parser.add_argument("--num-tasks", type=int, default=50,
                       help="Number of tasks per benchmark")
    parser.add_argument("--meta-tool-results", type=str,
                       help="Path to meta-tool results JSON for comparison")
    parser.add_argument("--api-key", type=str,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--all-models", action="store_true",
                       help="Evaluate all baseline models (GPT-4, GPT-3.5, Llama)")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to benchmark data directory")
    parser.add_argument("--strict-eval", action="store_true",
                       help="Use strict evaluation (higher accuracy thresholds)")
    args = parser.parse_args()
    
    print("="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    # Determine which models to evaluate
    if args.all_models:
        models_to_eval = [
            ("openai/gpt-5", "openai"),
            ("openai/gpt-5-mini", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("meta-llama/Llama-3.2-3B-Instruct", "local"),
        ]
        print("Evaluating all baseline models...")
    else:
        # Detect if it's an OpenAI model
        openai_models = ["openai/gpt-5", "openai/gpt-5-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]
        model_type = "openai" if args.model in openai_models or args.model.startswith("gpt-") or args.model.startswith("openai/") else "local"
        models_to_eval = [(args.model, model_type)]
    
    all_results = []
    
    for model_name, model_type in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_type})")
        print(f"Benchmarks: {args.benchmarks}")
        print(f"Methods: {args.methods}")
        print(f"Tasks per benchmark: {args.num_tasks}")
        print("="*60)
        
        try:
            # Create appropriate model evaluator
            if model_type == "openai":
                model = OpenAIEvaluator(model_name, api_key=args.api_key)
            else:
                model = LocalModelEvaluator(model_name)
                
            model.load_model()
            
            # Run evaluations
            for benchmark in args.benchmarks:
                for method in args.methods:
                    result = run_baseline_evaluation(
                        model, 
                        benchmark, 
                        method,
                        num_tasks=args.num_tasks,
                        data_path=args.data_path,
                        strict_eval=args.strict_eval
                    )
                    all_results.append(result)
                    
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Load meta-tool results if provided
    meta_tool_results = None
    if args.meta_tool_results and os.path.exists(args.meta_tool_results):
        with open(args.meta_tool_results) as f:
            meta_tool_results = json.load(f)
    
    # Generate comparison table
    generate_comparison_table(all_results, meta_tool_results)
    
    # Save results
    results_dict = {
        "models": [m[0] for m in models_to_eval] if args.all_models else [args.model],
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "model": r.model_name,
                "benchmark": r.benchmark,
                "method": r.method,
                "success_rate": r.success_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "num_tasks": r.num_tasks
            }
            for r in all_results
        ]
    }
    
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
