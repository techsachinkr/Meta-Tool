# Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models

**Accepted at ACL 2026 Findings**

This repository contains the implementation of **Meta-Tool**, a framework demonstrating that efficient tool-use in small language models is achievable through carefully designed few-shot prompting. Our 3B parameter model achieves competitive performance with GPT-5 while being **10x faster**.

## Quick Start

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

# Run few-shot sensitivity + error analysis (recommended)
python run_combined_analysis.py --output-dir ./analysis_results

# Run with specific checkpoint (if trained)
python run_combined_analysis.py --checkpoint checkpoints/best.pt --output-dir ./analysis_results

# Quick test on gorilla only
python run_combined_analysis.py --quick
```

`run_combined_analysis.py` is the primary evaluation script. It sweeps 0-5 examples across all benchmarks with strict evaluation, logs every prediction, auto-categorizes failures (format, semantic, hallucinated, empty), and outputs:
- Sensitivity table (performance vs number of examples)
- Error categorization summary
- CSV of all failure cases for manual review
- JSON with complete results

### Baseline Evaluation

```bash
# Run baseline with GPT-5
python run_baselines.py --model openai/gpt-5 --num-tasks 50 --strict-eval

# Run local baselines (AgentLM-7B, Llama-3.2-3B)
python run_local_baselines.py --model zai-org/agentlm-7b --strict-eval
python run_local_baselines.py --model meta-llama/Llama-3.2-3B-Instruct --strict-eval
```

> **Important:** Always use `--strict-eval` for baselines to match the evaluation mode used in `run_combined_analysis.py`.

### Hypernetwork Training (Optional)

> **Note:** Our experiments show that hypernetwork training provides no improvement over few-shot prompting. Training is included for reproducibility and future research.

```bash
# Train hypernetwork (optional - does not improve results)
python run_experiments.py --model-size large --episodes 10000
```

### Few-Shot Sensitivity and Error Analysis

```bash
python run_combined_analysis.py --checkpoint checkpoints/best.pt --output-dir ./analysis_results
```

### Robustness to Noisy Examples

Each benchmark evaluator in [evaluation.py](evaluation.py) has a commented-out `noisy examples` block containing intentionally wrong few-shot demonstrations (fake API names, invalid SQL, malformed JSON actions). To test noise robustness:

1. Open `evaluation.py` and find the `get_tool_spec()` method for the benchmark you want to test (search for `noisy examples`)
2. Uncomment the `examples = [...]` block below the noisy examples comment — this replaces the clean examples with noisy ones
3. Run evaluation:

```bash
python run_combined_analysis.py --checkpoint checkpoints/best.pt --output-dir ./analysis_results
```

4. Compare the results against the clean-examples baseline to measure degradation

Noisy example locations in `evaluation.py`:
- **Gorilla:** ~line 220 (fake function names)
- **Spider2:** ~line 877 (invalid SQL like `DELETE FROM`)
- **WebArena:** ~line 1155 (wrong action types, fake element IDs)
- **InterCode:** ~line 1512 (nonexistent commands, destructive commands)


## 📦 Data Sources

### Evaluation Benchmarks

Meta-Tool is evaluated on four diverse benchmarks:

| Benchmark | Domain | Tasks | Description |
|-----------|--------|-------|-------------|
| **Gorilla APIBench** | Model Loading APIs | 1780 (50 eval) | PyTorch Hub, TF Hub, HuggingFace, TorchVision model calls |
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
Meta-Tool/
├── config.py                # Configuration dataclasses and model size presets
├── evaluation.py            # Benchmark evaluators and scoring (main component)
├── run_combined_analysis.py # Few-shot sensitivity + error categorization (primary eval script)
├── run_experiments.py       # Full pipeline: training + evaluation
├── run_baselines.py         # OpenAI API baseline evaluation
├── run_local_baselines.py   # Local HuggingFace model baselines
├── hypernetwork.py          # Hypernetwork architecture (experimental)
├── lora_integration.py      # LoRA weight injection and adapted model
├── value_function.py        # Value function and beam search
├── constrained_decoding.py  # FSM-constrained generation
├── memory_system.py         # FAISS-based episodic memory
├── meta_training.py         # Meta-training loop (experimental)
├── data_loader.py           # ToolBench and benchmark data loading
├── data/                    # Benchmark datasets
│   ├── gorilla/             # Gorilla APIBench (1780 tasks)
│   ├── spider2/             # Spider 2.0 SQL
│   ├── webarena/            # WebArena navigation
│   └── intercode/           # InterCode bash/CLI
├── analysis_results/        # Output of run_combined_analysis.py
├── ablation_results/        # Stored ablation results
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Citation

If you use this work, please cite:

```bibtex
@misc{kumar2026metatoolefficientfewshottool,
      title={Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models}, 
      author={Sachin Kumar},
      year={2026},
      eprint={2604.20148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.20148}, 
}
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

