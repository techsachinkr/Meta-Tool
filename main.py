"""
Meta-Tool: Complete Implementation
==================================
Main entry point for training and evaluating the Meta-Tool framework.

Usage:
    # Train the hypernetwork
    python main.py train --config config.yaml
    
    # Evaluate on benchmarks
    python main.py evaluate --checkpoint checkpoints/best.pt
    
    # Adapt to a new tool
    python main.py adapt --tool-doc tool_documentation.txt --examples examples.json
"""

# Suppress TensorFlow/JAX warnings BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['JAX_PLATFORMS'] = ''  # Disable JAX GPU

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import torch
import json
import sys
from datetime import datetime

from config import MetaToolConfig, get_config
from hypernetwork import MetaToolHypernetwork, create_hypernetwork
from lora_integration import AdaptedModel, MetaToolAdaptedModel
from meta_training import MetaTrainer, create_synthetic_tools, run_meta_training
from evaluation import MetaToolEvaluator, run_evaluation
from value_function import ValueFunction, create_value_function
from memory_system import create_memory_system


def setup_environment(config: MetaToolConfig):
    """Setup training environment."""
    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        
    # Create directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # Print config
    print("="*60)
    print("META-TOOL CONFIGURATION")
    print("="*60)
    print(f"Device: {config.model.device}")
    print(f"Base Model: {config.model.base_model_name}")
    print(f"Encoder Model: {config.model.encoder_model_name}")
    print(f"LoRA Rank: {config.model.lora_rank}")
    print(f"Experiment: {config.experiment_name}")
    print("="*60)


def train(args):
    """Train the hypernetwork via meta-learning."""
    print("\n" + "="*60)
    print("META-TOOL TRAINING")
    print("="*60)
    
    config = get_config()
    
    # Override config from args
    if args.epochs:
        config.training.num_episodes = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
        
    setup_environment(config)
    
    # Run training
    trainer = run_meta_training(config)
    
    print("\nTraining complete!")
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
    
    return trainer


def evaluate(args):
    """Evaluate on benchmarks."""
    print("\n" + "="*60)
    print("META-TOOL EVALUATION")
    print("="*60)
    
    config = get_config()
    setup_environment(config)
    
    # Run evaluation
    results = run_evaluation(config, checkpoint_path=args.checkpoint)
    
    # Print summary table
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Benchmark':<15} {'Success Rate':<15} {'Avg Latency':<15} {'Adapt Time':<15}")
    print("-"*60)
    
    for name, result in results.items():
        print(f"{name:<15} {result.success_rate:>12.1%} {result.avg_latency_ms:>12.1f}ms {result.adaptation_time_s:>12.2f}s")
        
    return results


def adapt(args):
    """Adapt to a new tool interactively."""
    print("\n" + "="*60)
    print("META-TOOL ADAPTATION")
    print("="*60)
    
    config = get_config()
    setup_environment(config)
    
    # Load hypernetwork
    print("Loading hypernetwork...")
    hypernetwork = create_hypernetwork(config.model)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=config.model.device, weights_only=False)
        hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        
    # Create adapted model
    adapted_model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    # Load tool documentation
    if args.tool_doc:
        with open(args.tool_doc) as f:
            documentation = f.read()
    else:
        documentation = input("Enter tool documentation (or path to file): ")
        if os.path.exists(documentation):
            with open(documentation) as f:
                documentation = f.read()
                
    # Load examples
    if args.examples:
        with open(args.examples) as f:
            examples = json.load(f)
        queries = [e["query"] for e in examples]
        trajectories = [e["trajectory"] for e in examples]
    else:
        print("\nEnter few-shot examples (empty line to finish):")
        queries = []
        trajectories = []
        while True:
            query = input("Query: ").strip()
            if not query:
                break
            trajectory = input("Trajectory: ").strip()
            queries.append(query)
            trajectories.append(trajectory)
            
    # Adapt
    print(f"\nAdapting with {len(queries)} examples...")
    import time
    start = time.time()
    
    adapted_model.adapt_to_tool(
        documentation=documentation,
        support_queries=queries,
        support_trajectories=trajectories
    )
    
    adapt_time = time.time() - start
    print(f"Adaptation complete in {adapt_time:.2f}s")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'quit':
            break
            
        response = adapted_model.generate(
            f"Query: {query}\nOutput:",
            max_new_tokens=512,
            temperature=0.7
        )
        print(f"Response: {response}")


def demo(args):
    """Run a quick demo with synthetic data."""
    print("\n" + "="*60)
    print("META-TOOL DEMO")
    print("="*60)
    
    config = get_config()
    
    # Use smaller models for demo
    config.model.base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config.model.encoder_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config.model.base_model_dim = 2048
    config.model.encoder_dim = 2048
    config.model.num_adapt_layers = 16
    
    # Reduce training for demo
    config.training.num_episodes = 10
    config.training.batch_size = 2
    config.data.num_meta_train_tools = 20
    
    setup_environment(config)
    
    print("\n1. Creating synthetic tools...")
    tools = create_synthetic_tools(20)
    print(f"   Created {len(tools)} tools")
    
    print("\n2. Creating hypernetwork...")
    hypernetwork = create_hypernetwork(config.model)
    print(f"   Parameters: {hypernetwork.get_num_parameters():,}")
    
    print("\n3. Creating adapted model...")
    adapted_model = AdaptedModel(config.model)
    print(f"   LoRA layers: {len(adapted_model.lora_layers)}")
    
    print("\n4. Running mini training loop...")
    trainer = MetaTrainer(
        hypernetwork=hypernetwork,
        adapted_model=adapted_model,
        config=config,
        tools=tools
    )
    trainer.train(num_episodes=10)
    
    print("\n5. Testing adaptation...")
    test_doc = """
    # Test API
    search(query: str, limit: int) -> List[Result]
    """
    test_queries = ["Search for Python"]
    test_trajectories = ['{"function": "search", "query": "Python", "limit": 10}']
    
    full_model = MetaToolAdaptedModel(config.model, hypernetwork, adapted_model.base_model)
    full_model.adapt_to_tool(test_doc, test_queries, test_trajectories)
    
    response = full_model.generate("Query: Search for machine learning\nOutput:")
    print(f"   Generated: {response[:100]}...")
    
    print("\nDemo complete!")


def benchmark_comparison(args):
    """Generate comparison table with baselines."""
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    # Expected results from paper
    paper_results = {
        "gorilla": {
            "GPT-4o (Zero-Shot)": 84.5,
            "Llama-3-70B (ICL)": 71.2,
            "AgentTuning": 76.8,
            "AdaptAgent": 79.1,
            "Meta-Tool (Ours)": 86.2,
            "Fine-Tuned Oracle": 89.5
        },
        "spider2": {
            "GPT-4o (Zero-Shot)": 17.1,
            "Llama-3-70B (ICL)": 11.4,
            "AgentTuning": 15.3,
            "AdaptAgent": 19.8,
            "Meta-Tool (Ours)": 28.4,
            "Fine-Tuned Oracle": 34.2
        },
        "webarena": {
            "GPT-4o (Zero-Shot)": 23.5,
            "Llama-3-70B (ICL)": 14.2,
            "AgentTuning": 18.9,
            "AdaptAgent": 21.7,
            "Meta-Tool (Ours)": 26.1,
            "Fine-Tuned Oracle": 31.0
        },
        "intercode": {
            "GPT-4o (Zero-Shot)": 52.1,
            "Llama-3-70B (ICL)": 38.6,
            "AgentTuning": 45.2,
            "AdaptAgent": 49.8,
            "Meta-Tool (Ours)": 58.3,
            "Fine-Tuned Oracle": 64.5
        }
    }
    
    # Print comparison table
    print("\nTable 1: Execution Success Rates (%) on Diverse Benchmarks (10-Shot Adaptation)")
    print("-"*85)
    print(f"{'Model':<25} {'Gorilla API':>12} {'Spider 2.0':>12} {'WebArena':>12} {'InterCode':>12}")
    print("-"*85)
    
    models = ["GPT-4o (Zero-Shot)", "Llama-3-70B (ICL)", "AgentTuning", "AdaptAgent", "Meta-Tool (Ours)", "Fine-Tuned Oracle"]
    
    for model in models:
        row = f"{model:<25}"
        for bench in ["gorilla", "spider2", "webarena", "intercode"]:
            val = paper_results[bench][model]
            row += f" {val:>11.1f}%"
        print(row)
        
    print("-"*85)
    
    # Compute Meta-Tool vs Oracle ratio
    print("\nMeta-Tool achieves the following % of Fine-Tuned Oracle performance:")
    for bench in ["gorilla", "spider2", "webarena", "intercode"]:
        meta = paper_results[bench]["Meta-Tool (Ours)"]
        oracle = paper_results[bench]["Fine-Tuned Oracle"]
        ratio = meta / oracle * 100
        print(f"  {bench}: {ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Meta-Tool: Few-Shot Tool Adaptation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the hypernetwork")
    train_parser.add_argument("--epochs", type=int, help="Number of training episodes")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on benchmarks")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--benchmark", type=str, choices=["gorilla", "spider2", "webarena", "intercode", "all"], default="all")
    eval_parser.add_argument("--num-tasks", type=int, help="Limit number of tasks")
    
    # Adapt command
    adapt_parser = subparsers.add_parser("adapt", help="Adapt to a new tool")
    adapt_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    adapt_parser.add_argument("--tool-doc", type=str, help="Path to tool documentation")
    adapt_parser.add_argument("--examples", type=str, help="Path to examples JSON file")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo")
    
    # Benchmark comparison
    bench_parser = subparsers.add_parser("benchmark", help="Show benchmark comparison table")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "adapt":
        adapt(args)
    elif args.command == "demo":
        demo(args)
    elif args.command == "benchmark":
        benchmark_comparison(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
