#!/usr/bin/env python3
"""
Meta-Tool: Full Pipeline Runner
===============================
This script runs the complete Meta-Tool pipeline:
1. Creates/loads training data
2. Trains the hypernetwork
3. Evaluates on all benchmarks
4. Generates results tables

Run with: python run_experiments.py
"""

# Suppress TensorFlow/JAX warnings BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['JAX_PLATFORMS'] = ''  # Disable JAX GPU
os.environ['CUDA_VISIBLE_DEVICES_BACKUP'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

# CUDA memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import json
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Any

import torch
import numpy as np

# Set memory-efficient defaults for PyTorch
if torch.cuda.is_available():
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True


def check_dependencies():
    """Check if all required packages are installed."""
    required = ['torch', 'transformers', 'numpy', 'tqdm']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def setup_data_directories():
    """Create necessary data directories."""
    dirs = [
        './data/gorilla',
        './data/spider2', 
        './data/webarena',
        './data/intercode',
        './checkpoints',
        './results'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Data directories created")


def download_or_create_datasets(use_real_data: bool = True):
    """Download benchmark datasets or create synthetic versions."""
    print("\n" + "="*60)
    print("PREPARING DATASETS")
    print("="*60)
    
    if use_real_data:
        print("\n1. Loading ToolBench for meta-training...")
        try:
            from data_loader import setup_all_datasets, load_toolbench
            
            # Setup all datasets
            toolbench_loader = setup_all_datasets('./data')
            
            # Get tools
            tools = toolbench_loader.get_tools_with_examples(min_examples=5)
            print(f"   ✓ Loaded {len(tools)} tools from ToolBench")
            print(f"   ✓ Categories: {len(toolbench_loader.categories)}")
            
            # Save tool info for reference
            tools_summary = [
                {"name": t.name, "category": t.category, "num_examples": len(t.examples)}
                for t in tools[:100]
            ]
            with open('./data/tools_summary.json', 'w') as f:
                json.dump(tools_summary, f, indent=2)
                
            return tools
            
        except Exception as e:
            print(f"   ⚠ Failed to load ToolBench: {e}")
            print("   Falling back to synthetic data...")
            use_real_data = False
    
    if not use_real_data:
        from meta_training import create_synthetic_tools
        
        print("\n1. Creating synthetic meta-training tools...")
        tools = create_synthetic_tools(num_tools=500)
        
        # Save tools
        tools_data = [
            {
                "name": t.name,
                "documentation": t.documentation,
                "schema": t.schema,
                "examples": t.examples,
                "category": getattr(t, 'category', 'General')
            }
            for t in tools
        ]
        
        with open('./data/meta_train_tools.json', 'w') as f:
            json.dump(tools_data, f, indent=2)
        print(f"   Saved {len(tools)} synthetic training tools")
    
    # Create benchmark evaluation data
    print("\n2. Creating benchmark evaluation data...")
    
    benchmarks = {
        'gorilla': create_gorilla_tasks(100),
        'spider2': create_spider_tasks(100),
        'webarena': create_webarena_tasks(50),
        'intercode': create_intercode_tasks(50)
    }
    
    for name, tasks in benchmarks.items():
        os.makedirs(f'./data/{name}', exist_ok=True)
        path = f'./data/{name}/{name}_tasks.json'
        with open(path, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"   Created {len(tasks)} {name} tasks")
        
    return tools


def create_gorilla_tasks(n: int) -> List[Dict]:
    """Create Gorilla API benchmark tasks."""
    import random
    
    apis = [
        {"name": "search.web", "params": ["query", "num_results"], "template": "Search the web for {topic}"},
        {"name": "weather.get", "params": ["location", "units"], "template": "Get weather in {location}"},
        {"name": "email.send", "params": ["to", "subject", "body"], "template": "Send email to {recipient}"},
        {"name": "calendar.create", "params": ["title", "date", "time"], "template": "Create calendar event for {event}"},
        {"name": "translate.text", "params": ["text", "source", "target"], "template": "Translate '{text}' to {language}"},
        {"name": "image.generate", "params": ["prompt", "size", "style"], "template": "Generate an image of {subject}"},
        {"name": "database.query", "params": ["table", "conditions", "fields"], "template": "Query {table} for {conditions}"},
        {"name": "file.upload", "params": ["path", "destination", "public"], "template": "Upload file {filename}"},
    ]
    
    topics = ["machine learning", "climate change", "quantum computing", "artificial intelligence", "data science"]
    locations = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto"]
    
    tasks = []
    for i in range(n):
        api = random.choice(apis)
        
        if "topic" in api["template"]:
            query = api["template"].format(topic=random.choice(topics))
        elif "location" in api["template"]:
            query = api["template"].format(location=random.choice(locations))
        else:
            query = api["template"].format(**{k: f"test_{k}" for k in ["recipient", "event", "text", "language", "subject", "table", "conditions", "filename"]})
            
        expected = {
            "function": api["name"],
            **{p: f"value_{p}" for p in api["params"]}
        }
        
        tasks.append({
            "id": f"gorilla_{i}",
            "query": query,
            "expected": json.dumps(expected),
            "api": api["name"]
        })
        
    return tasks


def create_spider_tasks(n: int) -> List[Dict]:
    """Create Spider 2.0 SQL benchmark tasks."""
    templates = [
        ("Find all customers who spent more than $1000", 
         "SELECT * FROM customers WHERE total_spent > 1000"),
        ("List products with low inventory",
         "SELECT * FROM products WHERE stock < 10"),
        ("Count orders by status",
         "SELECT status, COUNT(*) FROM orders GROUP BY status"),
        ("Get top 10 customers by order count",
         "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 10"),
        ("Find average order value by category",
         "SELECT category, AVG(total) FROM orders GROUP BY category"),
        ("List employees hired this year",
         "SELECT * FROM employees WHERE YEAR(hire_date) = YEAR(CURRENT_DATE)"),
        ("Find customers without orders",
         "SELECT * FROM customers WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM orders)"),
        ("Get monthly revenue trend",
         "SELECT MONTH(order_date), SUM(total) FROM orders GROUP BY MONTH(order_date)"),
    ]
    
    tasks = []
    for i in range(n):
        query, sql = templates[i % len(templates)]
        tasks.append({
            "id": f"spider_{i}",
            "query": query,
            "expected_sql": sql,
            "database": "enterprise_db"
        })
        
    return tasks


def create_webarena_tasks(n: int) -> List[Dict]:
    """Create WebArena navigation benchmark tasks."""
    templates = [
        {"query": "Add item to shopping cart", "actions": ["navigate", "click", "add_to_cart"]},
        {"query": "Search for a product", "actions": ["click_search", "type_query", "submit"]},
        {"query": "Log into the website", "actions": ["click_login", "type_email", "type_password", "submit"]},
        {"query": "Submit a form", "actions": ["fill_fields", "validate", "submit"]},
        {"query": "Navigate to settings", "actions": ["click_menu", "click_settings"]},
    ]
    
    tasks = []
    for i in range(n):
        template = templates[i % len(templates)]
        tasks.append({
            "id": f"webarena_{i}",
            "query": template["query"],
            "expected_actions": template["actions"],
            "site": "test_site"
        })
        
    return tasks


def create_intercode_tasks(n: int) -> List[Dict]:
    """Create InterCode bash benchmark tasks."""
    templates = [
        ("Find all Python files", "find . -name '*.py'"),
        ("Count lines in a file", "wc -l file.txt"),
        ("Search for pattern in files", "grep -r 'pattern' ."),
        ("List files by size", "ls -lhS"),
        ("Show disk usage", "du -sh *"),
        ("Find large files", "find . -size +100M"),
        ("Compress a directory", "tar -czvf archive.tar.gz directory/"),
        ("Show running processes", "ps aux | grep python"),
    ]
    
    tasks = []
    for i in range(n):
        query, cmd = templates[i % len(templates)]
        tasks.append({
            "id": f"intercode_{i}",
            "query": query,
            "expected_command": cmd
        })
        
    return tasks


def run_training(config, tools, num_episodes: int = 1000, use_real_data: bool = True):
    """Run hypernetwork training."""
    import gc
    from config import MetaToolConfig
    from hypernetwork import create_hypernetwork
    from lora_integration import AdaptedModel
    from meta_training import MetaTrainer, Tool
    from data_loader import Tool as ToolBenchTool
    
    print("\n" + "="*60)
    print("TRAINING HYPERNETWORK")
    print("="*60)
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
    
    # Convert ToolBench tools to our Tool format if needed
    converted_tools = []
    for t in tools:
        if isinstance(t, dict):
            tool = Tool(
                name=t["name"],
                documentation=t["documentation"],
                schema=t["schema"],
                examples=t["examples"],
                category=t.get("category", "General")
            )
        elif hasattr(t, 'documentation'):
            # Already a Tool-like object
            tool = Tool(
                name=t.name,
                documentation=t.documentation if isinstance(t.documentation, str) else str(t.documentation),
                schema=t.schema if isinstance(t.schema, dict) else {},
                examples=list(t.examples) if hasattr(t, 'examples') else [],
                category=getattr(t, 'category', 'General')
            )
        else:
            tool = t
        converted_tools.append(tool)
    
    tools = converted_tools
    
    print(f"\n• Training on {len(tools)} tools")
    print(f"• Episodes: {num_episodes}")
    print(f"• Device: {config.model.device}")
    print(f"• Data source: {'ToolBench' if use_real_data else 'Synthetic'}")
    print(f"• Base model: {config.model.base_model_name}")
    
    # Show category distribution
    categories = {}
    for t in tools:
        cat = getattr(t, 'category', 'General')
        categories[cat] = categories.get(cat, 0) + 1
    print(f"• Categories: {len(categories)}")
    
    # Create components with memory tracking
    print("\nInitializing models...")
    
    # Create hypernetwork first (smaller)
    hypernetwork = create_hypernetwork(config.model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  After hypernetwork: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Create adapted model (larger - has the LLM)
    adapted_model = AdaptedModel(config.model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  After adapted model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Don't move adapted_model to device again if using device_map="auto"
    if not hasattr(adapted_model, 'is_quantized') or not adapted_model.is_quantized:
        adapted_model = adapted_model.to(config.model.device)
    
    # Create trainer
    trainer = MetaTrainer(
        hypernetwork=hypernetwork,
        adapted_model=adapted_model,
        config=config,
        tools=tools
    )
    
    # Train
    print("\nStarting training...")
    if torch.cuda.is_available():
        print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    start_time = time.time()
    trainer.train(num_episodes=num_episodes)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time/60:.1f} minutes")
    print(f"  Best loss: {trainer.best_loss:.4f}")
    
    return trainer.hypernetwork, training_time


def run_evaluation(config, hypernetwork):
    """Run evaluation on all benchmarks."""
    from lora_integration import MetaToolAdaptedModel
    from evaluation import MetaToolEvaluator
    
    print("\n" + "="*60)
    print("EVALUATING ON BENCHMARKS")
    print("="*60)
    
    # Create adapted model
    adapted_model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    # Create evaluator
    evaluator = MetaToolEvaluator(adapted_model, config)
    
    # Evaluate on all benchmarks
    results = evaluator.evaluate_all(num_tasks_per_benchmark=50)
    
    return results


def generate_results_table(results: Dict, training_time: float):
    """Generate formatted results tables."""
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS")
    print("="*60)
    
    # Table 1: Main results
    print("\n📊 Table 1: Execution Success Rates (%) - 10-Shot Adaptation")
    print("-"*75)
    print(f"{'Benchmark':<20} {'Success Rate':>15} {'Avg Latency':>15} {'Adapt Time':>15}")
    print("-"*75)
    
    for name, result in results.items():
        print(f"{name:<20} {result.success_rate*100:>14.1f}% {result.avg_latency_ms:>14.1f}ms {result.adaptation_time_s:>14.2f}s")
        
    print("-"*75)
    
    # Table 2: Comparison with paper claims
    paper_claims = {
        "gorilla": 86.2,
        "spider2": 28.4,
        "webarena": 26.1,
        "intercode": 58.3
    }
    
    print("\n📊 Table 2: Comparison with Paper Claims")
    print("-"*60)
    print(f"{'Benchmark':<20} {'Actual':>15} {'Paper Claim':>15} {'Diff':>10}")
    print("-"*60)
    
    for name, result in results.items():
        actual = result.success_rate * 100
        claimed = paper_claims.get(name, 0)
        diff = actual - claimed
        diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
        print(f"{name:<20} {actual:>14.1f}% {claimed:>14.1f}% {diff_str:>10}")
        
    print("-"*60)
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print("-"*40)
    avg_sr = np.mean([r.success_rate for r in results.values()])
    avg_latency = np.mean([r.avg_latency_ms for r in results.values()])
    avg_adapt = np.mean([r.adaptation_time_s for r in results.values()])
    
    print(f"Average Success Rate: {avg_sr*100:.1f}%")
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"Average Adaptation Time: {avg_adapt:.2f}s")
    print(f"Total Training Time: {training_time/60:.1f} minutes")
    
    return {
        "results": {name: r.to_dict() for name, r in results.items()},
        "summary": {
            "avg_success_rate": avg_sr,
            "avg_latency_ms": avg_latency,
            "avg_adaptation_time_s": avg_adapt,
            "training_time_s": training_time
        }
    }


def save_results(output: Dict, path: str = "./ablation_results/ablation_no_doc_zeroshot__results_10k.json"):
    """Save results to JSON file."""
    output["timestamp"] = datetime.now().isoformat()
    output["config"] = {
        "model": "Meta-Tool",
        "framework": "Hypernetwork + LoRA + Value-Guided Beam Search"
    }
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\n✓ Results saved to {path}")
    
    # Also save in format for baseline comparison
    meta_tool_results = {}
    for name, result in output.get("results", {}).items():
        meta_tool_results[name] = {
            "success_rate": result.get("success_rate", 0) * 100,  # Convert to percentage
            "latency": result.get("avg_latency_ms", 0),
            "adaptation_time": result.get("adaptation_time_s", 0)
        }
    
    comparison_path = "./ablation_results/no_doc_zeroshot__meta_tool_for_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(meta_tool_results, f, indent=2)
    print(f"✓ Comparison format saved to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Meta-Tool experiments")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced settings")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for eval-only mode")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of ToolBench")
    parser.add_argument("--huggingface", action="store_true", default=True, help="Load ToolBench from HuggingFace")
    parser.add_argument("--model-size", type=str, default="medium", 
                       choices=["tiny", "small", "medium", "large", "xlarge"],
                       help="Model size preset: tiny (~4GB), small (~8GB), medium (~12GB), large (~16GB), xlarge (~32GB)")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks per benchmark")
    parser.add_argument("--fast-eval", action="store_true", help="Fast evaluation with reduced settings")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["gorilla", "spider2", "webarena", "intercode"],
                       help="Benchmarks to evaluate")
    parser.add_argument("--strict-eval", action="store_true", 
                       help="Use strict evaluation (higher accuracy thresholds, no cross-format matching)")
    args = parser.parse_args()
    
    use_real_data = not args.synthetic
    
    print("="*60)
    print("META-TOOL: Full Experimental Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source: {'ToolBench' if use_real_data else 'Synthetic'}")
    print(f"Model size: {args.model_size}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Setup
    setup_data_directories()
    
    # Import config with model size
    from config import get_config
    config = get_config(model_size=args.model_size)
    
    print(f"Base model: {config.model.base_model_name}")
    print(f"Encoder: {config.model.encoder_model_name}")
    
    # Adjust for quick mode
    if args.quick:
        print("\n⚡ Quick mode enabled - using reduced settings")
        config.training.num_episodes = 100
        config.training.batch_size = 2
        config.data.num_meta_train_tools = 50
        args.episodes = 100
    
    # Fast eval mode - reduce generation settings
    if args.fast_eval:
        print("\n⚡ Fast eval mode - reducing generation overhead")
        config.inference.max_new_tokens = 64  # Shorter outputs
        config.inference.do_sample = False  # Greedy is faster
        args.num_tasks = min(args.num_tasks, 30)  # Fewer tasks
        
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model.device = device
    print(f"\n🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Prepare data
    tools = download_or_create_datasets(use_real_data=use_real_data)
    
    # Training
    if not args.eval_only:
        hypernetwork, training_time = run_training(
            config, tools, 
            num_episodes=args.episodes,
            use_real_data=use_real_data
        )
    else:
        print("\n⏭️  Skipping training (eval-only mode)")
        from hypernetwork import create_hypernetwork
        hypernetwork = create_hypernetwork(config.model)
        if args.checkpoint:
            # PyTorch 2.6+ requires weights_only=False for checkpoints with custom objects
            checkpoint = torch.load(args.checkpoint, map_location=config.model.device, weights_only=False)
            hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
            print(f"✅ Loaded checkpoint from {args.checkpoint}")
        else:
            print("⚠️  No checkpoint provided - using untrained hypernetwork")
        training_time = 0
    
    # Evaluation with configurable tasks and benchmarks
    from lora_integration import MetaToolAdaptedModel
    from evaluation import MetaToolEvaluator
    
    print("\n" + "="*60)
    print("EVALUATING ON BENCHMARKS")
    print("="*60)
    print(f"Tasks per benchmark: {args.num_tasks}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Strict evaluation: {args.strict_eval}")
    
    # Create adapted model
    adapted_model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    # Create evaluator with strict mode setting
    evaluator = MetaToolEvaluator(adapted_model, config, strict_eval=args.strict_eval)
    
    # Evaluate on selected benchmarks
    results = {}
    for benchmark in args.benchmarks:
        if benchmark in evaluator.evaluators:
            results[benchmark] = evaluator.evaluate_benchmark(
                benchmark, 
                num_tasks=args.num_tasks
            )
        else:
            print(f"⚠️  Unknown benchmark: {benchmark}")
    
    # Generate tables
    output = generate_results_table(results, training_time)
    
    # Save results
    save_results(output)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
