This repository contains the implementation of **Meta-Tool**, a framework demonstrating that efficient tool-use in small language models is achievable through carefully designed few-shot prompting. Our 3B parameter model achieves **77% of GPT-5's performance** while being **10x faster**.

## 📊 Key Results

| Model | Gorilla | Spider 2.0 | WebArena | InterCode | Avg | Latency |
|-------|---------|------------|----------|-----------|-----|---------|
| GPT-5 | 38.0% | 72.0% | 54.0% | 72.0% | 59.0% | ~16.5s |
| AgentLM-7B | 8.0% | 44.0% | 8.0% | 40.0% | 25.0% | ~8.9s |
| **Meta-Tool (3B)** | **32.0%** | **64.0%** | **34.0%** | **52.0%** | **45.5%** | **~1.6s** |

### Key Findings from Ablation Study

| Configuration | Gorilla | Spider2 | WebArena | InterCode | Avg |
|---------------|---------|---------|----------|-----------|-----|
| Full (5-shot + docs) | 32.0% | 64.0% | 34.0% | 52.0% | **45.5%** |
| 0-shot + docs | 14.0% | 24.0% | 24.0% | 48.0% | 27.5% |
| 5-shot + no docs | 32.0% | 62.0% | 24.0% | 44.0% | 40.5% |
| 0-shot + no docs | 0.0% | 4.0% | 0.0% | 10.0% | 3.5% |

**Main Insight:** Few-shot examples contribute **+18%** to performance, documentation contributes **+5%**, while hypernetwork-based LoRA adaptation provides **no measurable improvement** over few-shot prompting alone.



## 🚀 Quick Start

### Installation

```bash


# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
hf auth login
```

### Training

```bash
# Train the model with large model size
python run_experiments.py --model-size large --episodes 10000
```

### Evaluation

```bash
# Download benchmarks (WebArena and InterCode)
python download_benchmarks.py --data-dir ./data --force

# Run evaluation with few-shot prompting (recommended)
python run_experiments.py --model-size large --eval-only --strict-eval

# Run with specific checkpoint (if trained)
python run_experiments.py --model-size large --eval-only --checkpoint checkpoints/best.pt --strict-eval
```

### Baseline Evaluation

```bash
# Run baseline with GPT-5
python run_baselines.py --model openai/gpt-5 --num-tasks 50 --strict-eval

# Run local baselines (AgentLM-7B, Llama-3.2-3B)
python run_local_baselines.py --model zai-org/agentlm-7b --strict-eval
python run_local_baselines.py --model meta-llama/Llama-3.2-3B-Instruct --strict-eval
```

### Hypernetwork Training (Optional)

> **Note:** Our experiments show that hypernetwork training provides no improvement over few-shot prompting. Training is included for reproducibility and future research.

```bash
# Train hypernetwork (optional - does not improve results)
python run_experiments.py --model-size large --episodes 10000
```

### Few-Shot Sensitivity & Error Analysis

Run a combined sweep that measures performance at 0–5 examples and auto-categorizes failures (format, semantic, hallucinated):

```bash
# Full analysis across all four benchmarks
python run_combined_analysis.py 
    --checkpoint checkpoints/best.pt
    --output-dir ./analysis_results
```

Outputs a sensitivity table, an error-category summary, a CSV of failure cases for manual review, and a JSON with complete results under `--output-dir` (default: `./analysis_results/`).

### Robustness to Noisy Examples

[evaluation.py](evaluation.py) includes commented-out noisy few-shot examples (wrong function names, invalid actions, malformed SQL) for each benchmark. Uncomment the `noisy examples` blocks in the relevant benchmark class to measure how the model degrades when the in-context demonstrations are incorrect.
Then run the eval again with

```bash
python run_experiments.py --model-size large --eval-only --checkpoint checkpoints/best.pt --strict-eval
```

## 📦 Data Sources

### Evaluation Benchmarks

Meta-Tool is evaluated on four diverse benchmarks:

| Benchmark | Domain | Tasks | Description |
|-----------|--------|-------|-------------|
| **Gorilla APIBench** | REST APIs | 50 | Function calling with strict parameter matching |
| **Spider 2.0** | Enterprise SQL | 50 | Text-to-SQL with complex schemas (1000+ columns) |
| **WebArena** | Web Navigation | 50 | Long-horizon planning in web environments |
| **InterCode** | Bash/CLI | 50 | Command-line tasks and CTF challenges |

### ToolBench (for Training)

If you wish to experiment with training, Meta-Tool uses **ToolBench**:
- **16,464 REST APIs** across **3,451 tools** from RapidAPI Hub
- **126,486 instruction-solution pairs** with API call trajectories

```python
from meta_tool import load_toolbench_tools

# Load from HuggingFace
tools = load_toolbench_tools(use_huggingface=True, min_examples=5)
```

## 📁 Project Structure

```
meta_tool/
├── config.py               # Configuration dataclasses
├── hypernetwork.py         # Hypernetwork architecture (experimental)
├── lora_integration.py     # LoRA weight injection
├── evaluation.py           # Benchmark evaluation (main component)
├── data_loader.py          # ToolBench and benchmark data loading
├── value_function.py       # Value function and beam search
├── constrained_decoding.py # FSM-constrained generation
├── memory_system.py        # FAISS-based episodic memory
├── meta_training.py        # Meta-training loop (experimental)
├── run_experiments.py      # Full experimental pipeline
├── run_baselines.py        # Baseline evaluation scripts
├── run_local_baselines.py  # Local model baselines
├── run_combined_analysis.py # Few-shot sensitivity + error categorization
├── analysis_results/       # Output of run_combined_analysis.py
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🔧 Configuration

Edit `config.py` or pass arguments to override:

```python
from config import get_config

config = get_config()

# Model settings
config.model.base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
config.model.lora_rank = 16

# Training settings
config.training.num_episodes = 50000
config.training.batch_size = 4
config.training.learning_rate = 1e-5

# Data settings
config.data.toolbench_path = "./data/toolbench"
config.data.num_meta_train_tools = 500

# Inference settings
config.inference.beam_width = 5
config.inference.max_depth = 10
```

