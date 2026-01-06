"""
LoRA Integration for Meta-Tool
==============================
Applies hypernetwork-generated LoRA weights to the base model.
Supports dynamic weight injection without modifying base model parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Callable
import copy

from config import ModelConfig
from hypernetwork import LoRAWeights


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation.
    Computes: output = W @ x + (B @ A) @ x
    
    Supports dynamic weight updates from hypernetwork.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA matrices (will be set by hypernetwork)
        self.lora_A: Optional[torch.Tensor] = None  # [r, in_features]
        self.lora_B: Optional[torch.Tensor] = None  # [out_features, r]
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Flag to track if LoRA is active
        self.lora_enabled = False
        
    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor):
        """
        Set LoRA weights from hypernetwork output.
        
        Handles dimension mismatches for GQA models where K/V have different dims.
        Uses proper scaling to maintain gradient stability.
        
        Args:
            A: [r, in_features] or [batch, r, in_features]
            B: [out_features, r] or [batch, out_features, r]
        """
        # Get actual dimensions from base layer
        actual_in = self.base_layer.in_features
        actual_out = self.base_layer.out_features
        target_device = self.base_layer.weight.device
        
        # ALWAYS use float16 for LoRA weights - never use quantized dtypes
        # This ensures compatibility with quantized base models
        target_dtype = torch.float16
        
        # Convert to float16 on the correct device
        A = A.to(dtype=target_dtype, device=target_device).contiguous()
        B = B.to(dtype=target_dtype, device=target_device).contiguous()
        
        # Handle dimension mismatches (for GQA models like Llama)
        # A should be [r, in_features], B should be [out_features, r]
        if A.dim() == 2:
            # Non-batched case
            r = A.shape[0]
            gen_in = A.shape[1]
            gen_out = B.shape[0]
            
            # Resize A if needed: [r, generated_in] -> [r, actual_in]
            if gen_in != actual_in:
                if gen_in > actual_in:
                    # Use mean pooling over groups to preserve information
                    ratio = gen_in // actual_in
                    if gen_in % actual_in == 0:
                        A = A.view(r, actual_in, ratio).mean(dim=2)
                    else:
                        A = A[:, :actual_in]
                else:
                    # Pad with small random values for stability
                    pad = torch.randn(r, actual_in - gen_in, dtype=A.dtype, device=A.device) * 0.01
                    A = torch.cat([A, pad], dim=1)
            
            # Resize B if needed: [generated_out, r] -> [actual_out, r]
            if gen_out != actual_out:
                if gen_out > actual_out:
                    # Use mean pooling over groups to preserve information
                    ratio = gen_out // actual_out
                    if gen_out % actual_out == 0:
                        B = B.view(actual_out, ratio, r).mean(dim=1)
                    else:
                        B = B[:actual_out, :]
                else:
                    # Pad with small random values
                    pad = torch.randn(actual_out - gen_out, r, dtype=B.dtype, device=B.device) * 0.01
                    B = torch.cat([B, pad], dim=0)
                    
            # Scale to maintain stability when dimensions changed
            if gen_out != actual_out or gen_in != actual_in:
                scale = float((actual_out / gen_out) ** 0.5)
                B = B * scale
                
        else:
            # Batched case: [batch, r, in_features] and [batch, out_features, r]
            batch_size = A.shape[0]
            r = A.shape[1]
            gen_in = A.shape[2]
            gen_out = B.shape[1]
            
            if gen_in != actual_in:
                if gen_in > actual_in:
                    ratio = gen_in // actual_in
                    if gen_in % actual_in == 0:
                        A = A.view(batch_size, r, actual_in, ratio).mean(dim=3)
                    else:
                        A = A[:, :, :actual_in]
                else:
                    pad = torch.randn(batch_size, r, actual_in - gen_in, dtype=A.dtype, device=A.device) * 0.01
                    A = torch.cat([A, pad], dim=2)
            
            if gen_out != actual_out:
                if gen_out > actual_out:
                    ratio = gen_out // actual_out
                    if gen_out % actual_out == 0:
                        B = B.view(batch_size, actual_out, ratio, r).mean(dim=2)
                    else:
                        B = B[:, :actual_out, :]
                else:
                    pad = torch.randn(batch_size, actual_out - gen_out, r, dtype=B.dtype, device=B.device) * 0.01
                    B = torch.cat([B, pad], dim=1)
                    
            # Scale for stability
            if gen_out != actual_out or gen_in != actual_in:
                scale = float((actual_out / gen_out) ** 0.5)
                B = B * scale
        
        # Clamp to prevent extreme values
        A = torch.clamp(A, -10.0, 10.0)
        B = torch.clamp(B, -10.0, 10.0)
        
        # Ensure tensors are contiguous for efficient operations
        self.lora_A = A.contiguous()
        self.lora_B = B.contiguous()
        self.lora_enabled = True
        
    def clear_lora_weights(self):
        """Clear LoRA weights and disable adaptation."""
        self.lora_A = None
        self.lora_B = None
        self.lora_enabled = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional LoRA adaptation.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
            
        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        # Base layer output
        base_output = self.base_layer(x)
        
        if not self.lora_enabled or self.lora_A is None or self.lora_B is None:
            return base_output
        
        # LoRA adaptation: (B @ A) @ x
        x_dropout = self.dropout(x)
        
        # Convert input to match LoRA weights dtype (float16)
        lora_dtype = self.lora_A.dtype
        x_lora = x_dropout.to(lora_dtype)
        
        # Handle batched LoRA weights
        if self.lora_A.dim() == 3:
            # Batched: A is [batch, r, in_features]
            lora_output = torch.einsum('bri,bsi->bsr', self.lora_A, x_lora)  # [batch, seq, r]
            lora_output = torch.einsum('bor,bsr->bso', self.lora_B, lora_output)  # [batch, seq, out]
        else:
            # Non-batched: A is [r, in_features]
            lora_output = F.linear(x_lora, self.lora_A)  # [batch, seq, r]
            lora_output = F.linear(lora_output, self.lora_B)  # [batch, seq, out]
        
        # Apply scaling
        lora_output = lora_output * self.scaling
        
        # Check for NaN/Inf and clamp if necessary
        if torch.isnan(lora_output).any() or torch.isinf(lora_output).any():
            lora_output = torch.nan_to_num(lora_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to match base output dtype
        lora_output = lora_output.to(base_output.dtype)
        
        return base_output + lora_output


class AdaptedModel(nn.Module):
    """
    Wrapper around base model with dynamic LoRA injection.
    Applies hypernetwork-generated weights to specified modules.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        base_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        
        # Load base model if not provided
        if base_model is None:
            # Memory-efficient loading
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Try 4-bit quantization first (most memory efficient)
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes
                
                # 4-bit config for very large models
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=config.dtype,
                    bnb_4bit_use_double_quant=True
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
                
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name,
                    **load_kwargs
                )
                self.is_quantized = True
                print(f"Loaded {config.base_model_name} with 4-bit quantization")
            except (ImportError, Exception) as e:
                # Fallback to standard loading
                try:
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        config.base_model_name,
                        torch_dtype=config.dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map="auto"
                    )
                except Exception:
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        config.base_model_name,
                        torch_dtype=config.dtype,
                        trust_remote_code=True
                    )
                self.is_quantized = False
        else:
            self.base_model = base_model
            self.is_quantized = False
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace target layers with LoRA versions
        self.lora_layers: Dict[str, LoRALinear] = {}
        self._inject_lora_layers()
        
    def _inject_lora_layers(self):
        """Replace target modules with LoRA-enabled versions."""
        for layer_idx in range(self.config.num_adapt_layers):
            for module_name in self.config.target_modules:
                # Navigate to the target layer
                layer = self._get_layer(layer_idx)
                if layer is None:
                    continue
                    
                # Get the specific module (q_proj, v_proj, etc.)
                target_module = self._get_module_by_name(layer, module_name)
                if target_module is None or not isinstance(target_module, nn.Linear):
                    continue
                
                # Create LoRA wrapper
                lora_layer = LoRALinear(
                    base_layer=target_module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                
                # Replace in the model
                self._set_module_by_name(layer, module_name, lora_layer)
                
                # Track for weight injection
                key = f"layer_{layer_idx}_{module_name}"
                self.lora_layers[key] = lora_layer
                
        print(f"Injected LoRA into {len(self.lora_layers)} modules")
        
    def _get_layer(self, layer_idx: int):
        """Get transformer layer by index."""
        # Handle different model architectures
        if hasattr(self.base_model, 'model'):
            if hasattr(self.base_model.model, 'layers'):
                layers = self.base_model.model.layers
            elif hasattr(self.base_model.model, 'decoder'):
                layers = self.base_model.model.decoder.layers
            else:
                return None
        elif hasattr(self.base_model, 'transformer'):
            layers = self.base_model.transformer.h
        else:
            return None
            
        if layer_idx < len(layers):
            return layers[layer_idx]
        return None
        
    def _get_module_by_name(self, parent: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a child module by name, supporting nested names."""
        parts = name.split('.')
        module = parent
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, 'self_attn') and hasattr(module.self_attn, part):
                module = getattr(module.self_attn, part)
            else:
                return None
        return module
        
    def _set_module_by_name(self, parent: nn.Module, name: str, new_module: nn.Module):
        """Set a child module by name."""
        parts = name.split('.')
        
        # Navigate to parent of target
        if len(parts) == 1:
            if hasattr(parent, 'self_attn') and hasattr(parent.self_attn, name):
                setattr(parent.self_attn, name, new_module)
            else:
                setattr(parent, name, new_module)
        else:
            module = parent
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], new_module)
            
    def apply_lora_weights(self, lora_weights: LoRAWeights, batch_idx: int = 0):
        """
        Apply hypernetwork-generated LoRA weights to the model.
        
        Args:
            lora_weights: Generated LoRA weights
            batch_idx: Index in batch (for batched weight generation)
        """
        for key, lora_layer in self.lora_layers.items():
            if key in lora_weights.A_matrices and key in lora_weights.B_matrices:
                A = lora_weights.A_matrices[key]
                B = lora_weights.B_matrices[key]
                
                # Extract specific batch if batched
                if A.dim() == 3:
                    A = A[batch_idx]
                    B = B[batch_idx]
                    
                lora_layer.set_lora_weights(A, B)
                
    def clear_lora_weights(self):
        """Clear all LoRA weights."""
        for lora_layer in self.lora_layers.values():
            lora_layer.clear_lora_weights()
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass through adapted model."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs
    ):
        """Generate text with adapted model."""
        # Get sampling parameters
        temperature = kwargs.pop('temperature', 1.0)
        do_sample = kwargs.pop('do_sample', temperature != 1.0)
        
        # Create stopping criteria for faster generation
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_token_ids):
                self.stop_token_ids = stop_token_ids
            
            def __call__(self, input_ids, scores, **kwargs):
                for stop_id in self.stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        
        # Get stop token IDs (newline, semicolon, etc)
        stop_tokens = []
        for token in ['\n\n', ';', '<|eot_id|>', '</s>', '<|end|>']:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            stop_tokens.extend(ids)
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)]) if stop_tokens else None
        
        # Use greedy decoding if temperature is 1.0 or not specified for stability
        if not do_sample:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                stopping_criteria=stopping_criteria,
                **kwargs
            )
        else:
            # Sampling - ensure valid temperature
            temperature = max(0.01, temperature)  # Prevent division by zero
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=kwargs.pop('top_p', 0.9),
                stopping_criteria=stopping_criteria,
                **kwargs
            )
        

class MetaToolAdaptedModel:
    """
    High-level interface for adapted model with hypernetwork integration.
    Handles the complete pipeline from tool spec to adapted generation.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        hypernetwork: nn.Module,
        base_model: Optional[nn.Module] = None
    ):
        self.config = config
        self.hypernetwork = hypernetwork
        self.adapted_model = AdaptedModel(config, base_model)
        
        # Move to device
        self.hypernetwork = self.hypernetwork.to(config.device)
        self.adapted_model = self.adapted_model.to(config.device)
        
    def adapt_to_tool(
        self,
        documentation: str,
        support_queries: list,
        support_trajectories: list
    ):
        """
        Adapt the model to a new tool using hypernetwork.
        
        Args:
            documentation: Tool documentation string
            support_queries: List of example queries
            support_trajectories: List of example trajectories
        """
        # Clear any existing adaptation
        self.adapted_model.clear_lora_weights()
        
        # Generate LoRA weights via hypernetwork
        with torch.no_grad():
            lora_weights = self.hypernetwork(
                docs=[documentation],
                support_queries=[support_queries],
                support_trajectories=[support_trajectories]
            )
        
        # Apply to model
        self.adapted_model.apply_lora_weights(lora_weights, batch_idx=0)
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response with adapted model."""
        # Tokenize
        inputs = self.adapted_model.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.adapted_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode
        generated = self.adapted_model.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
        
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get logits for input tokens."""
        with torch.no_grad():
            outputs = self.adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs.logits


def create_adapted_model(
    config: ModelConfig,
    hypernetwork: nn.Module
) -> MetaToolAdaptedModel:
    """Factory function to create adapted model."""
    return MetaToolAdaptedModel(config, hypernetwork)
