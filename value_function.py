"""
Value Function Learning and Value-Guided Beam Search
====================================================
Implements TD(0) learning for value function V_φ and
value-guided beam search for inference.

The value function estimates P(success | state) and guides
action selection to avoid unproductive branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import random

from config import ModelConfig, TrainingConfig, InferenceConfig


@dataclass
class Transition:
    """A single transition in a trajectory."""
    state: str  # Serialized state
    action: str  # Action taken
    reward: float  # r_t ∈ {-1, 0, +1}
    next_state: str  # Next state
    done: bool  # Terminal flag
    state_embedding: Optional[torch.Tensor] = None


@dataclass
class BeamCandidate:
    """A candidate in beam search."""
    state: str
    trajectory: List[str]  # Actions taken so far
    score: float  # Cumulative score
    value: float  # V(state)


class ValueFunction(nn.Module):
    """
    Value function V_φ: S → [0, 1]
    
    Estimates the probability of eventual task success from intermediate states.
    Architecture: MLP operating on state embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 512],
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate for state(s).
        
        Args:
            state_embedding: [batch, input_dim] state embeddings
            
        Returns:
            values: [batch] value estimates in [0, 1]
        """
        logits = self.mlp(state_embedding).squeeze(-1)  # [batch]
        return torch.sigmoid(logits)


class ReplayBuffer:
    """Experience replay buffer for TD learning."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition: Transition):
        """Add a transition to the buffer."""
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self):
        return len(self.buffer)


class ValueFunctionTrainer:
    """
    Trainer for value function using TD(0) learning.
    
    Loss: L_TD(φ) = E[(r_t + γ·V_φ(s_{t+1}) - V_φ(s_t))²]
    """
    
    def __init__(
        self,
        value_function: ValueFunction,
        state_encoder: nn.Module,  # Encodes states to embeddings
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.value_function = value_function.to(device)
        self.state_encoder = state_encoder.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            value_function.parameters(),
            lr=config.value_lr
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Target network for stable training (updated periodically)
        self.target_value_function = ValueFunction(
            input_dim=value_function.mlp[0].in_features,
            hidden_dims=config.value_hidden_dims
        ).to(device)
        self.target_value_function.load_state_dict(value_function.state_dict())
        
        # Update frequency
        self.target_update_freq = 100
        self.update_count = 0
        
    def encode_states(self, states: List[str]) -> torch.Tensor:
        """Encode state strings to embeddings."""
        with torch.no_grad():
            embeddings = self.state_encoder(states)
        return embeddings
        
    def add_trajectory(self, transitions: List[Transition]):
        """Add a trajectory to the replay buffer."""
        for transition in transitions:
            self.replay_buffer.push(transition)
            
    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one training step.
        
        Returns:
            TD loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
            
        # Sample batch
        transitions = self.replay_buffer.sample(batch_size)
        
        # Prepare batch
        states = [t.state for t in transitions]
        next_states = [t.next_state for t in transitions]
        rewards = torch.tensor([t.reward for t in transitions], device=self.device)
        dones = torch.tensor([t.done for t in transitions], device=self.device)
        
        # Encode states
        state_embeddings = self.encode_states(states)
        next_state_embeddings = self.encode_states(next_states)
        
        # Current V(s)
        current_values = self.value_function(state_embeddings)
        
        # Target: r + γ·V(s') for non-terminal, r for terminal
        with torch.no_grad():
            next_values = self.target_value_function(next_state_embeddings)
            targets = rewards + self.config.td_gamma * next_values * (~dones).float()
            
        # TD loss
        loss = F.mse_loss(current_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_value_function.load_state_dict(
                self.value_function.state_dict()
            )
            
        return loss.item()
        
    def train_epoch(self, num_steps: int = 100, batch_size: int = 32) -> float:
        """Train for multiple steps."""
        total_loss = 0.0
        for _ in range(num_steps):
            loss = self.train_step(batch_size)
            total_loss += loss
        return total_loss / num_steps


class ValueGuidedBeamSearch:
    """
    Beam search guided by learned value function.
    
    At each step:
    1. For each candidate, sample K actions
    2. Execute actions to get next states
    3. Score candidates by score * V(s')
    4. Keep top-B candidates
    """
    
    def __init__(
        self,
        value_function: ValueFunction,
        state_encoder: nn.Module,
        action_generator,  # Callable that generates action candidates
        action_executor,  # Callable that executes actions
        config: InferenceConfig,
        device: str = "cuda"
    ):
        self.value_function = value_function.to(device)
        self.state_encoder = state_encoder.to(device)
        self.action_generator = action_generator
        self.action_executor = action_executor
        self.config = config
        self.device = device
        
        self.value_function.eval()
        
    def get_value(self, state: str) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            embedding = self.state_encoder([state])
            value = self.value_function(embedding)
        return value.item()
        
    def search(
        self,
        initial_state: str,
        query: str
    ) -> Tuple[List[str], float]:
        """
        Perform value-guided beam search.
        
        Args:
            initial_state: Starting state
            query: User query to solve
            
        Returns:
            best_trajectory: List of actions
            final_score: Score of best trajectory
        """
        B = self.config.beam_width
        D = self.config.max_depth
        K = self.config.candidates_per_step
        
        # Initialize beam with single candidate
        beam = [BeamCandidate(
            state=initial_state,
            trajectory=[],
            score=1.0,
            value=self.get_value(initial_state)
        )]
        
        for depth in range(D):
            all_candidates = []
            
            for candidate in beam:
                # Check if terminal (success)
                if self._is_success(candidate.state, query):
                    return candidate.trajectory, candidate.score
                    
                # Generate K action candidates
                actions = self.action_generator(
                    state=candidate.state,
                    query=query,
                    num_candidates=K
                )
                
                for action in actions:
                    # Execute action
                    next_state, reward, done = self.action_executor(
                        state=candidate.state,
                        action=action
                    )
                    
                    # Compute new score
                    next_value = self.get_value(next_state)
                    new_score = candidate.score * next_value
                    
                    # Handle failure penalty
                    if reward == -1:
                        new_score *= 0.1  # Heavy penalty for failures
                        
                    new_candidate = BeamCandidate(
                        state=next_state,
                        trajectory=candidate.trajectory + [action],
                        score=new_score,
                        value=next_value
                    )
                    all_candidates.append(new_candidate)
                    
                    # Early termination on success
                    if done and reward == 1:
                        return new_candidate.trajectory, new_candidate.score
                        
            # Keep top-B candidates
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            beam = all_candidates[:B]
            
            if not beam:
                break
                
        # Return best trajectory found
        if beam:
            best = max(beam, key=lambda x: x.score)
            return best.trajectory, best.score
        return [], 0.0
        
    def _is_success(self, state: str, query: str) -> bool:
        """Check if state represents task success."""
        # This should be implemented based on task-specific success criteria
        # Default: check for success indicators in state
        success_indicators = ["success", "completed", "done", "result:"]
        return any(ind in state.lower() for ind in success_indicators)


class ExecutionEnvironment:
    """
    Simulated execution environment for tool testing.
    Wraps actual tool execution with error handling.
    """
    
    def __init__(self, tool_executor=None):
        self.tool_executor = tool_executor
        self.execution_history = []
        
    def execute(
        self,
        state: str,
        action: str
    ) -> Tuple[str, float, bool]:
        """
        Execute an action and return next state, reward, done.
        
        Args:
            state: Current state string
            action: Action to execute
            
        Returns:
            next_state: New state after action
            reward: -1 (failure), 0 (neutral), +1 (success)
            done: Whether episode is terminal
        """
        try:
            if self.tool_executor is not None:
                result = self.tool_executor(action)
                
                # Determine reward based on result
                if "error" in str(result).lower():
                    reward = -1.0
                    done = True
                elif "success" in str(result).lower():
                    reward = 1.0
                    done = True
                else:
                    reward = 0.0
                    done = False
                    
                next_state = f"{state}\nAction: {action}\nResult: {result}"
                
            else:
                # Simulated execution for testing
                next_state = f"{state}\nAction: {action}\nResult: Executed"
                reward = 0.0
                done = False
                
            self.execution_history.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done
            })
            
            return next_state, reward, done
            
        except Exception as e:
            # Execution error
            next_state = f"{state}\nAction: {action}\nError: {str(e)}"
            return next_state, -1.0, True
            
    def get_transitions(self) -> List[Transition]:
        """Convert execution history to transitions."""
        return [
            Transition(
                state=h["state"],
                action=h["action"],
                reward=h["reward"],
                next_state=h["next_state"],
                done=h["done"]
            )
            for h in self.execution_history
        ]


def create_value_function(config: ModelConfig) -> ValueFunction:
    """Create value function with default configuration."""
    return ValueFunction(
        input_dim=config.encoder_dim,
        hidden_dims=[1024, 512],
        dropout=0.1
    )


def run_refinement_phase(
    value_function: ValueFunction,
    state_encoder: nn.Module,
    test_cases: List,  # From schema_perturbation
    execution_env: ExecutionEnvironment,
    config: TrainingConfig,
    num_epochs: int = 10
) -> float:
    """
    Run the self-supervised refinement phase.
    
    1. Execute test cases in environment
    2. Collect transitions
    3. Train value function via TD learning
    
    Returns:
        Final training loss
    """
    trainer = ValueFunctionTrainer(
        value_function=value_function,
        state_encoder=state_encoder,
        config=config,
        device=config.device if hasattr(config, 'device') else "cuda"
    )
    
    # Execute test cases and collect transitions
    initial_state = "Initial state"
    
    for test_case in test_cases:
        action = str(test_case.perturbed_action)
        next_state, reward, done = execution_env.execute(initial_state, action)
        
    # Add collected transitions to trainer
    transitions = execution_env.get_transitions()
    trainer.add_trajectory(transitions)
    
    # Train value function
    print(f"Training value function on {len(transitions)} transitions...")
    
    final_loss = 0.0
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(num_steps=100, batch_size=32)
        final_loss = loss
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
            
    return final_loss


# Example usage
if __name__ == "__main__":
    from config import ModelConfig, TrainingConfig, InferenceConfig
    
    config = ModelConfig()
    
    # Create value function
    vf = create_value_function(config)
    print(f"Value function parameters: {sum(p.numel() for p in vf.parameters()):,}")
    
    # Test forward pass
    dummy_embedding = torch.randn(4, config.encoder_dim)
    values = vf(dummy_embedding)
    print(f"Values shape: {values.shape}, range: [{values.min():.3f}, {values.max():.3f}]")
