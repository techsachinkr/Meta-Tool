"""
Meta-Training Loop for Meta-Tool
================================
Episodic meta-learning over a distribution of tools.

Training procedure:
1. Sample tool from meta-training set
2. Sample support set and query set
3. Generate LoRA weights via hypernetwork
4. Compute task loss on query set
5. Update hypernetwork parameters

Loss: L_meta(ψ) = E_T E_{S,Q} [ -log P_{M_{θ,H_ψ(D,S)}}(τ*|q) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import random
from tqdm import tqdm
from dataclasses import dataclass, field
import wandb
import logging

from config import MetaToolConfig, ModelConfig, TrainingConfig
from hypernetwork import MetaToolHypernetwork, LoRAWeights, create_hypernetwork
from lora_integration import AdaptedModel

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool with documentation and examples."""
    name: str
    documentation: str
    schema: Dict[str, Any]
    examples: List[Tuple[str, str]] = field(default_factory=list)  # (query, trajectory) pairs
    category: str = "General"
    api_list: List[Any] = field(default_factory=list)
    

@dataclass 
class Episode:
    """A single meta-training episode."""
    tool: Tool
    support_queries: List[str]
    support_trajectories: List[str]
    query_queries: List[str]
    query_trajectories: List[str]


class ToolDataset(Dataset):
    """Dataset of tools for meta-training."""
    
    def __init__(
        self,
        tools: List[Tool],
        support_size: int = 10,
        query_size: int = 5
    ):
        self.tools = tools
        self.support_size = support_size
        self.query_size = query_size
        
    def __len__(self):
        return len(self.tools)
        
    def __getitem__(self, idx: int) -> Episode:
        tool = self.tools[idx]
        
        # Sample support and query sets
        examples = tool.examples.copy()
        random.shuffle(examples)
        
        # Ensure we have enough examples
        min_examples = self.support_size + self.query_size
        if len(examples) < min_examples:
            # Duplicate examples if needed
            while len(examples) < min_examples:
                examples.extend(tool.examples)
            examples = examples[:min_examples]
            
        support = examples[:self.support_size]
        query = examples[self.support_size:self.support_size + self.query_size]
        
        return Episode(
            tool=tool,
            support_queries=[s[0] for s in support],
            support_trajectories=[s[1] for s in support],
            query_queries=[q[0] for q in query],
            query_trajectories=[q[1] for q in query]
        )


def collate_episodes(episodes: List[Episode]) -> Dict[str, Any]:
    """Collate episodes into a batch."""
    return {
        "docs": [e.tool.documentation for e in episodes],
        "support_queries": [e.support_queries for e in episodes],
        "support_trajectories": [e.support_trajectories for e in episodes],
        "query_queries": [e.query_queries for e in episodes],
        "query_trajectories": [e.query_trajectories for e in episodes],
        "tools": [e.tool for e in episodes]
    }


class MetaTrainer:
    """
    Meta-trainer for the hypernetwork.
    
    Trains the hypernetwork to generate good LoRA weights
    for novel tools using episodic meta-learning.
    """
    
    def __init__(
        self,
        hypernetwork: MetaToolHypernetwork,
        adapted_model: AdaptedModel,
        config: MetaToolConfig,
        tools: List[Tool]
    ):
        self.hypernetwork = hypernetwork
        self.adapted_model = adapted_model
        self.config = config
        self.device = config.model.device
        
        # Create dataset and dataloader
        self.dataset = ToolDataset(
            tools,
            support_size=config.training.support_set_size,
            query_size=config.training.query_set_size
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_episodes,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Optimizer
        self.optimizer = AdamW(
            hypernetwork.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.training.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_episodes - config.training.warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_steps]
        )
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
    def compute_task_loss(
        self,
        lora_weights: LoRAWeights,
        query_queries: List[str],
        query_trajectories: List[str],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Compute task loss for a single tool.
        
        Loss = -log P(trajectory | query) under adapted model
        """
        # Apply LoRA weights
        self.adapted_model.apply_lora_weights(lora_weights, batch_idx)
        
        total_loss = 0.0
        num_queries = len(query_queries)
        
        for query, trajectory in zip(query_queries, query_trajectories):
            # Construct input: query -> trajectory
            prompt = f"Query: {query}\nTrajectory:"
            target = f" {trajectory}"
            full_text = prompt + target
            
            # Tokenize
            inputs = self.adapted_model.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            prompt_inputs = self.adapted_model.tokenizer(
                prompt,
                return_tensors="pt"
            )
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            # Forward pass
            outputs = self.adapted_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # Compute loss only on trajectory tokens
            logits = outputs.logits[:, prompt_len-1:-1, :]  # Shift for next token prediction
            labels = inputs.input_ids[:, prompt_len:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.adapted_model.tokenizer.pad_token_id
            )
            
            total_loss += loss
            
        # Clear LoRA weights
        self.adapted_model.clear_lora_weights()
        
        return total_loss / num_queries
        
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one meta-training step.
        
        Returns:
            Dictionary of metrics
        """
        self.hypernetwork.train()
        
        # Debug: check hypernetwork parameters
        if self.global_step == 0:
            trainable_params = sum(p.numel() for p in self.hypernetwork.parameters() if p.requires_grad)
            print(f"[DEBUG] Hypernetwork trainable params: {trainable_params:,}")
            print(f"[DEBUG] Hypernetwork training mode: {self.hypernetwork.training}")
        
        # Generate LoRA weights for all tools in batch
        lora_weights = self.hypernetwork(
            docs=batch["docs"],
            support_queries=batch["support_queries"],
            support_trajectories=batch["support_trajectories"]
        )
        
        # Debug: check if LoRA weights have gradients
        if self.global_step == 0:
            sample_key = list(lora_weights.A_matrices.keys())[0]
            A_sample = lora_weights.A_matrices[sample_key]
            print(f"[DEBUG] LoRA A shape: {A_sample.shape}, requires_grad: {A_sample.requires_grad}, has grad_fn: {A_sample.grad_fn is not None}")
            if A_sample.grad_fn is not None:
                print(f"[DEBUG] LoRA A grad_fn: {A_sample.grad_fn}")
        
        # Compute task loss for each tool
        batch_size = len(batch["docs"])
        losses = []
        
        for i in range(batch_size):
            task_loss = self.compute_task_loss(
                lora_weights,
                batch["query_queries"][i],
                batch["query_trajectories"][i],
                batch_idx=i
            )
            losses.append(task_loss)
            
            # Debug: check first task loss
            if self.global_step == 0 and i == 0:
                print(f"[DEBUG] Task loss: {task_loss.item() if hasattr(task_loss, 'item') else task_loss}, "
                      f"requires_grad: {task_loss.requires_grad if hasattr(task_loss, 'requires_grad') else 'N/A'}, "
                      f"has grad_fn: {task_loss.grad_fn is not None if hasattr(task_loss, 'grad_fn') else 'N/A'}")
            
        # Stack losses to maintain gradient
        avg_loss = torch.stack(losses).mean()
        
        # Debug: check avg_loss
        if self.global_step == 0:
            print(f"[DEBUG] Avg loss: {avg_loss.item()}, requires_grad: {avg_loss.requires_grad}, has grad_fn: {avg_loss.grad_fn is not None}")
        
        # Backward pass
        self.optimizer.zero_grad()
        
        try:
            avg_loss.backward()
        except RuntimeError as e:
            print(f"[ERROR] Backward pass failed: {e}")
            print(f"[ERROR] avg_loss: {avg_loss}, requires_grad: {avg_loss.requires_grad}, grad_fn: {avg_loss.grad_fn}")
            # Check all intermediate tensors
            for key in list(lora_weights.A_matrices.keys())[:3]:
                A = lora_weights.A_matrices[key]
                B = lora_weights.B_matrices[key]
                print(f"[ERROR] {key} - A grad_fn: {A.grad_fn}, B grad_fn: {B.grad_fn}")
            raise
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.hypernetwork.parameters(),
            self.config.training.max_grad_norm
        )
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": avg_loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }
        
    def train(self, num_episodes: Optional[int] = None):
        """
        Run meta-training loop.
        
        Args:
            num_episodes: Override number of episodes
        """
        num_episodes = num_episodes or self.config.training.num_episodes
        
        # Initialize wandb if configured
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=vars(self.config)
            )
            
        print(f"Starting meta-training for {num_episodes} episodes...")
        
        episode_iter = iter(self.dataloader)
        
        # Track loss history for trend
        recent_losses = []
        
        for episode_idx in tqdm(range(num_episodes)):
            # Get next batch (with wraparound)
            try:
                batch = next(episode_iter)
            except StopIteration:
                episode_iter = iter(self.dataloader)
                batch = next(episode_iter)
                
            # Train step
            metrics = self.train_step(batch)
            
            self.global_step += 1
            recent_losses.append(metrics['loss'])
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            
            # Logging
            if self.global_step % 100 == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"Step {self.global_step}: loss={metrics['loss']:.4f}, "
                      f"avg_loss={avg_loss:.4f}, grad_norm={metrics['grad_norm']:.4f}, "
                      f"best={self.best_loss:.4f}")
                      
                if self.config.wandb_project:
                    wandb.log({**metrics, 'avg_loss': avg_loss}, step=self.global_step)
                    
            # Checkpointing
            if self.global_step % self.config.training.save_every == 0:
                self.save_checkpoint()
                
            # Track best
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint("best")
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"  Final loss: {metrics['loss']:.4f}")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Total steps: {self.global_step}")
        print(f"{'='*50}")
        
    def save_checkpoint(self, name: str = None):
        """Save a checkpoint."""
        name = name or f"step_{self.global_step}"
        path = os.path.join(self.config.training.checkpoint_dir, f"{name}.pt")
        
        torch.save({
            "hypernetwork_state_dict": self.hypernetwork.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config
        }, path)
        
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")


def load_toolbench_tools(
    data_root: str = "./data/toolbench",
    use_huggingface: bool = True,
    min_examples: int = 1,
    max_tools: Optional[int] = None
) -> List[Tool]:
    """
    Load real tools from ToolBench dataset.
    
    ToolBench contains 16,464 REST APIs across 49 categories from RapidAPI Hub.
    
    Args:
        data_root: Path to ToolBench data directory
        use_huggingface: Use HuggingFace datasets (easier setup) vs local files
        min_examples: Minimum examples per tool to include (default 1)
        max_tools: Maximum number of tools to load (None for all)
        
    Returns:
        List of Tool objects ready for meta-training
    """
    from data_loader import ToolBenchLoader
    
    logger.info(f"Loading ToolBench from {'HuggingFace' if use_huggingface else data_root}...")
    
    loader = ToolBenchLoader(
        data_root=data_root,
        use_huggingface=use_huggingface,
        download_if_missing=True
    )
    loader.setup()
    
    # Get tools with sufficient examples
    toolbench_tools = loader.get_tools_with_examples(min_examples=min_examples)
    
    # Convert to our Tool format
    tools = []
    for tb_tool in toolbench_tools:
        tool = Tool(
            name=tb_tool.name,
            documentation=tb_tool.documentation,
            schema=tb_tool.schema,
            examples=tb_tool.examples,
            category=tb_tool.category,
            api_list=tb_tool.api_list
        )
        tools.append(tool)
        
        if max_tools and len(tools) >= max_tools:
            break
            
    logger.info(f"Loaded {len(tools)} tools with >= {min_examples} examples each")
    
    # Log category distribution
    categories = {}
    for t in tools:
        categories[t.category] = categories.get(t.category, 0) + 1
    logger.info(f"Category distribution: {len(categories)} categories")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {cat}: {count} tools")
        
    return tools


def create_synthetic_tools(num_tools: int = 100) -> List[Tool]:
    """
    Create synthetic tools for testing/development when ToolBench is unavailable.
    
    This is a fallback - prefer load_toolbench_tools() for real training.
    
    Args:
        num_tools: Number of synthetic tools to create
        
    Returns:
        List of synthetic Tool objects
    """
    logger.warning("Using synthetic tools - for real training, use load_toolbench_tools()")
    
    tools = []
    
    tool_templates = [
        {
            "name": "search_api",
            "description": "Search for items in a database",
            "category": "Search",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "category": {"type": "string", "enum": ["all", "products", "users"]}
                },
                "required": ["query"]
            }
        },
        {
            "name": "create_record",
            "description": "Create a new record in the database",
            "category": "Database",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "data": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name"]
            }
        },
        {
            "name": "send_message",
            "description": "Send a message to a user",
            "category": "Communication",
            "schema": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]}
                },
                "required": ["recipient", "message"]
            }
        },
        {
            "name": "weather_api",
            "description": "Get weather information for a location",
            "category": "Weather",
            "schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        },
        {
            "name": "translate_text",
            "description": "Translate text between languages",
            "category": "Language",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_lang": {"type": "string"},
                    "target_lang": {"type": "string"}
                },
                "required": ["text", "target_lang"]
            }
        }
    ]
    
    for i in range(num_tools):
        template = tool_templates[i % len(tool_templates)]
        
        # Create tool with variations
        tool = Tool(
            name=f"{template['name']}_{i}",
            documentation=f"# {template['name']}_{i}\n\n"
                         f"Category: {template['category']}\n"
                         f"Description: {template['description']}\n\n"
                         f"## Schema\n```json\n{json.dumps(template['schema'], indent=2)}\n```",
            schema=template["schema"],
            examples=generate_synthetic_examples(template, num_examples=20),
            category=template["category"]
        )
        tools.append(tool)
        
    return tools


def generate_synthetic_examples(template: Dict, num_examples: int = 20) -> List[Tuple[str, str]]:
    """Generate synthetic query-trajectory pairs for a tool template."""
    examples = []
    
    queries = [
        f"Use {template['name']} to process request {i}" 
        for i in range(num_examples)
    ]
    
    for i, query in enumerate(queries):
        # Generate a plausible trajectory
        trajectory = json.dumps({
            "action": template["name"],
            "parameters": {
                prop: f"value_{i}" 
                for prop in template["schema"].get("properties", {}).keys()
            },
            "result": "success"
        })
        examples.append((query, trajectory))
        
    return examples


def load_tools_for_training(
    config: MetaToolConfig,
    use_real_data: bool = True
) -> List[Tool]:
    """
    Load tools for meta-training.
    
    Tries ToolBench first, falls back to synthetic if unavailable.
    
    Args:
        config: Meta-Tool configuration
        use_real_data: Whether to use real ToolBench data
        
    Returns:
        List of Tool objects
    """
    if use_real_data:
        try:
            return load_toolbench_tools(
                data_root=config.data.toolbench_path,
                use_huggingface=True,  # Easier setup
                min_examples=5,
                max_tools=config.data.num_meta_train_tools
            )
        except Exception as e:
            logger.warning(f"Failed to load ToolBench: {e}")
            logger.warning("Falling back to synthetic tools")
            
    return create_synthetic_tools(config.data.num_meta_train_tools)


def run_meta_training(config: MetaToolConfig, use_real_data: bool = True):
    """
    Main entry point for meta-training.
    
    Args:
        config: Meta-Tool configuration
        use_real_data: Whether to use real ToolBench data (True) or synthetic (False)
    """
    
    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Create components
    print("Creating hypernetwork...")
    hypernetwork = create_hypernetwork(config.model)
    
    print("Creating adapted model...")
    adapted_model = AdaptedModel(config.model)
    adapted_model = adapted_model.to(config.model.device)
    
    # Load tools (real ToolBench or synthetic)
    print("Loading tools...")
    tools = load_tools_for_training(config, use_real_data=use_real_data)
    print(f"Loaded {len(tools)} tools")
    
    # Filter tools with sufficient examples (lower threshold for real data)
    min_examples = 2 if use_real_data else (config.training.support_set_size + config.training.query_set_size)
    tools = [t for t in tools if len(t.examples) >= min_examples]
    print(f"After filtering: {len(tools)} tools with >= {min_examples} examples")
    
    if len(tools) < 10:
        logger.warning("Very few tools available - consider using synthetic data or lowering thresholds")
        if len(tools) == 0:
            logger.warning("No tools available! Falling back to synthetic data.")
            tools = create_synthetic_tools(config.data.num_meta_train_tools)
    
    # Create trainer
    trainer = MetaTrainer(
        hypernetwork=hypernetwork,
        adapted_model=adapted_model,
        config=config,
        tools=tools
    )
    
    # Train
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    
    # Reduce for testing
    config.training.num_episodes = 100
    config.training.batch_size = 2
    config.data.num_meta_train_tools = 50
    
    trainer = run_meta_training(config)
