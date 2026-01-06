"""
Hypernetwork Architecture for Meta-Tool
=======================================
Generates LoRA weights from tool documentation and support set examples.

Components:
1. Documentation Encoder (decoder-only transformer)
2. Support Set Encoder with Prototype Aggregation
3. Factorized Weight Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import ModelConfig


@dataclass
class LoRAWeights:
    """Container for generated LoRA weights."""
    A_matrices: Dict[str, torch.Tensor]  # {layer_name: A matrix}
    B_matrices: Dict[str, torch.Tensor]  # {layer_name: B matrix}
    layer_indices: Optional[List[int]] = None
    target_modules: Optional[List[str]] = None
    
    def to(self, device):
        """Move weights to device."""
        self.A_matrices = {k: v.to(device) for k, v in self.A_matrices.items()}
        self.B_matrices = {k: v.to(device) for k, v in self.B_matrices.items()}
        return self


class DocumentationEncoder(nn.Module):
    """
    Encodes tool documentation using a pretrained transformer.
    Uses memory-efficient loading with quantization support.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # For sentence-transformers models, use their native API
        if "sentence-transformers" in config.encoder_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(config.encoder_model_name)
                self.encoder.to(config.device)
                self.use_sentence_transformer = True
                self.tokenizer = None
                
                # Get actual output dimension
                encoder_hidden_size = self.encoder.get_sentence_embedding_dimension()
                
                # ALWAYS use a Linear projection to ensure gradients flow
                self.projection = nn.Linear(encoder_hidden_size, config.encoder_dim)
                
                # Add a small learnable scale parameter to guarantee grad_fn
                self.grad_scale = nn.Parameter(torch.ones(1))
                return
            except ImportError:
                pass
        
        self.use_sentence_transformer = False
        
        # Memory-efficient loading for larger models
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Use 8-bit quantization if available and model is large
        try:
            import bitsandbytes
            if "llama" in config.encoder_model_name.lower() or "phi" in config.encoder_model_name.lower():
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
        except ImportError:
            load_kwargs["torch_dtype"] = config.dtype
        
        try:
            self.encoder = AutoModel.from_pretrained(
                config.encoder_model_name,
                **load_kwargs
            )
        except Exception as e:
            # Fallback without quantization
            self.encoder = AutoModel.from_pretrained(
                config.encoder_model_name,
                torch_dtype=config.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
        # Freeze encoder - we don't need gradients for it
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.encoder_model_name,
            trust_remote_code=True
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # ALWAYS use Linear projection for gradient flow
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_hidden_size, config.encoder_dim)
        
        # Add a small learnable scale parameter to guarantee grad_fn
        self.grad_scale = nn.Parameter(torch.ones(1))
            
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode documentation texts.
        
        Args:
            texts: List of documentation strings
            
        Returns:
            v_doc: [batch_size, d_enc] documentation embeddings
        """
        if self.use_sentence_transformer:
            # Encode to numpy first to avoid inference mode issues
            embeddings_np = self.encoder.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Get device and dtype from projection layer
            proj_device = next(self.projection.parameters()).device
            proj_dtype = next(self.projection.parameters()).dtype
            
            # Convert numpy to tensor
            embeddings = torch.tensor(
                embeddings_np.tolist(),
                dtype=proj_dtype,
                device=proj_device
            )
            
            # Apply projection - creates grad_fn from projection weights
            result = self.projection(embeddings)
            
            # Multiply by grad_scale (which is ~1.0) to GUARANTEE grad_fn exists
            # This is a learnable parameter so it connects the computation graph
            result = result * self.grad_scale
            
            return result
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,  # Reduced from 4096 to save memory
            return_tensors="pt"
        ).to(self.encoder.device)
        
        # Forward pass - no gradients needed for encoder
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.config.dtype == torch.float16):
                outputs = self.encoder(**inputs, output_hidden_states=True)
        
        # Get last hidden state at EOS position
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Find EOS positions (last non-padding token)
        seq_lengths = inputs.attention_mask.sum(dim=1) - 1  # [batch]
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        eos_hidden = last_hidden[batch_indices, seq_lengths]  # [batch, hidden]
        
        # Get device and dtype from projection layer for consistency
        proj_device = next(self.projection.parameters()).device
        proj_dtype = next(self.projection.parameters()).dtype
        
        # Convert to tensor - breaks inference mode, projection will add grad_fn
        eos_hidden = torch.tensor(
            eos_hidden.detach().cpu().float().tolist(),
            dtype=proj_dtype,
            device=proj_device
        )
        
        # Project to d_enc - Linear creates grad_fn
        v_doc = self.projection(eos_hidden)  # [batch, d_enc]
        
        # Multiply by grad_scale to guarantee grad_fn
        v_doc = v_doc * self.grad_scale
        
        return v_doc


class PrototypeAggregator(nn.Module):
    """
    Aggregates support set embeddings via cross-attention with documentation as query.
    Uses separate key and value projections (W_K ≠ W_V).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Projection matrices for cross-attention
        self.W_Q = nn.Linear(config.encoder_dim, config.attention_dim)
        self.W_K = nn.Linear(config.encoder_dim, config.attention_dim)
        self.W_V = nn.Linear(config.encoder_dim, config.attention_dim)
        
        # Scale factor
        self.scale = config.attention_dim ** -0.5
        
    def forward(
        self, 
        v_doc: torch.Tensor,  # [batch, d_enc]
        v_support: torch.Tensor,  # [batch, K, d_enc]
        support_mask: Optional[torch.Tensor] = None  # [batch, K]
    ) -> torch.Tensor:
        """
        Compute prototype vector via cross-attention.
        
        Args:
            v_doc: Documentation embeddings [batch, d_enc]
            v_support: Support set embeddings [batch, K, d_enc]
            support_mask: Optional mask for variable-length support sets
            
        Returns:
            v_proto: Prototype vector [batch, d_attn]
        """
        batch_size, K, _ = v_support.shape
        
        # Compute Q, K, V
        Q = self.W_Q(v_doc)  # [batch, d_attn]
        K = self.W_K(v_support)  # [batch, K, d_attn]
        V = self.W_V(v_support)  # [batch, K, d_attn]
        
        # Attention scores: Q^T @ K / sqrt(d)
        Q = Q.unsqueeze(1)  # [batch, 1, d_attn]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [batch, 1, K]
        
        # Apply mask if provided
        if support_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~support_mask.unsqueeze(1), float('-inf')
            )
        
        # Softmax and weighted sum
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 1, K]
        v_proto = torch.bmm(attn_weights, V).squeeze(1)  # [batch, d_attn]
        
        return v_proto


class FactorizedWeightGenerator(nn.Module):
    """
    Memory-efficient LoRA weight generator using shared low-rank factorization.
    
    Instead of having separate projections for each layer (which explodes memory),
    we use:
    1. A small shared MLP to generate a compact context vector
    2. Layer embeddings to differentiate layers
    3. Low-rank factorization for the final weight generation
    
    This reduces memory from O(L * d_model * r * d_latent) to O(d_latent^2 + L * d_embed)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: v_doc + projected v_proto
        self.input_dim = config.encoder_dim + config.encoder_dim
        
        # Projection for prototype
        self.W_p = nn.Linear(config.attention_dim, config.encoder_dim)
        
        # Compact latent dimension for efficiency
        compact_latent = min(config.latent_dim, 512)
        
        # Shared MLP: [v_doc; W_p @ v_proto] -> z (compact)
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, compact_latent)
        )
        
        # Layer embeddings (one per layer per module type)
        num_layer_embeddings = config.num_adapt_layers * len(config.target_modules)
        self.layer_embeddings = nn.Embedding(num_layer_embeddings, compact_latent)
        
        # Calculate dimensions
        d_model = config.base_model_dim
        r = config.lora_rank
        
        # LOW-RANK FACTORIZATION for weight generation
        # Instead of latent -> r*d_model directly, we use:
        # latent -> small_hidden -> r*factor + factor*d_model
        # This reduces params from latent*r*d_model to latent*small + small*r + small*d_model
        
        factor_dim = min(64, r * 4)  # Small intermediate dimension
        
        # For A matrix: [r, d_model]
        # Generate as: A = A_left @ A_right where A_left: [r, factor], A_right: [factor, d_model]
        self.A_left_proj = nn.Linear(compact_latent, r * factor_dim)
        self.A_right_proj = nn.Linear(compact_latent, factor_dim * d_model)
        
        # For B matrix: [d_model, r]  
        # Generate as: B = B_left @ B_right where B_left: [d_model, factor], B_right: [factor, r]
        self.B_left_proj = nn.Linear(compact_latent, d_model * factor_dim)
        self.B_right_proj = nn.Linear(compact_latent, factor_dim * r)
        
        self.factor_dim = factor_dim
        self.r = r
        self.d_model = d_model
        
        # Initialize for stable training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projections for stable LoRA initialization (B starts near zero)."""
        nn.init.normal_(self.B_left_proj.weight, std=0.001)
        nn.init.zeros_(self.B_left_proj.bias)
        nn.init.normal_(self.B_right_proj.weight, std=0.001)
        nn.init.zeros_(self.B_right_proj.bias)
            
    def forward(
        self,
        v_doc: torch.Tensor,  # [batch, d_enc]
        v_proto: torch.Tensor  # [batch, d_attn]
    ) -> LoRAWeights:
        """
        Generate LoRA weights from documentation and prototype.
        
        Uses low-rank factorization for memory efficiency.
        """
        batch_size = v_doc.size(0)
        device = v_doc.device
        
        # Project prototype to encoder dimension
        v_proto_proj = self.W_p(v_proto)  # [batch, d_enc]
        
        # Concatenate to form context
        context = torch.cat([v_doc, v_proto_proj], dim=-1)  # [batch, 2*d_enc]
        
        # Shared latent projection
        z_base = self.shared_mlp(context)  # [batch, compact_latent]
        
        # Generate A and B for each layer
        A_matrices = {}
        B_matrices = {}
        
        layer_idx_offset = 0
        for layer_idx in range(self.config.num_adapt_layers):
            for module_idx, module in enumerate(self.config.target_modules):
                key = f"layer_{layer_idx}_{module}"
                
                # Get layer-specific embedding
                embed_idx = layer_idx * len(self.config.target_modules) + module_idx
                layer_embed = self.layer_embeddings(
                    torch.tensor([embed_idx], device=device)
                ).expand(batch_size, -1)  # [batch, compact_latent]
                
                # Combine base context with layer embedding
                z = z_base + layer_embed  # [batch, compact_latent]
                
                # Generate A via low-rank factorization
                A_left = self.A_left_proj(z).view(batch_size, self.r, self.factor_dim)
                A_right = self.A_right_proj(z).view(batch_size, self.factor_dim, self.d_model)
                A = torch.bmm(A_left, A_right)  # [batch, r, d_model]
                
                # Generate B via low-rank factorization
                B_left = self.B_left_proj(z).view(batch_size, self.d_model, self.factor_dim)
                B_right = self.B_right_proj(z).view(batch_size, self.factor_dim, self.r)
                B = torch.bmm(B_left, B_right)  # [batch, d_model, r]
                
                A_matrices[key] = A
                B_matrices[key] = B
        
        return LoRAWeights(
            A_matrices=A_matrices,
            B_matrices=B_matrices,
            layer_indices=list(range(self.config.num_adapt_layers)),
            target_modules=self.config.target_modules
        )


class MetaToolHypernetwork(nn.Module):
    """
    Complete hypernetwork that generates LoRA weights from tool documentation
    and support set examples.
    
    H_ψ: (D_T, S_T) → {(A_l, B_l)}
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.doc_encoder = DocumentationEncoder(config)
        self.prototype_aggregator = PrototypeAggregator(config)
        self.weight_generator = FactorizedWeightGenerator(config)
        
    def encode_documentation(self, docs: List[str]) -> torch.Tensor:
        """Encode tool documentation."""
        result = self.doc_encoder(docs)
        
        # Safety check: ensure result has grad_fn
        if result.grad_fn is None and self.training:
            # This shouldn't happen with grad_scale, but just in case
            # Add zero contribution from a parameter to create grad_fn
            for param in self.doc_encoder.parameters():
                if param.requires_grad:
                    result = result + param.sum() * 0.0
                    break
        
        return result
    
    def encode_support_set(
        self,
        queries: List[List[str]],  # [batch][K] queries
        trajectories: List[List[str]]  # [batch][K] trajectories
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode support set demonstrations.
        
        Returns:
            v_support: [batch, K, d_enc] support embeddings
            mask: [batch, K] valid demonstration mask
        """
        batch_size = len(queries)
        max_k = max(len(q) for q in queries)
        
        # Flatten all demonstrations for batch encoding
        all_texts = []
        positions = []  # (batch_idx, k_idx)
        
        for b_idx, (qs, ts) in enumerate(zip(queries, trajectories)):
            for k_idx, (q, t) in enumerate(zip(qs, ts)):
                # Combine query and trajectory
                text = f"Query: {q}\nTrajectory: {t}"
                all_texts.append(text)
                positions.append((b_idx, k_idx))
        
        # Encode all demonstrations
        if all_texts:
            all_embeddings = self.doc_encoder(all_texts)  # [total, d_enc]
        
            # Reshape back to [batch, K, d_enc]
            v_support = torch.zeros(
                batch_size, max_k, self.config.encoder_dim,
                device=all_embeddings.device, dtype=all_embeddings.dtype
            )
            mask = torch.zeros(batch_size, max_k, dtype=torch.bool, device=all_embeddings.device)
            
            for idx, (b_idx, k_idx) in enumerate(positions):
                v_support[b_idx, k_idx] = all_embeddings[idx]
                mask[b_idx, k_idx] = True
        else:
            device = next(self.parameters()).device
            v_support = torch.zeros(batch_size, max_k, self.config.encoder_dim, device=device)
            mask = torch.zeros(batch_size, max_k, dtype=torch.bool, device=device)
        
        return v_support, mask
    
    def forward(
        self,
        docs: List[str],  # Tool documentation for each tool in batch
        support_queries: List[List[str]],  # Support set queries
        support_trajectories: List[List[str]]  # Support set trajectories
    ) -> LoRAWeights:
        """
        Generate LoRA weights for tool adaptation.
        
        Args:
            docs: List of tool documentation strings [batch]
            support_queries: List of query lists [batch][K]
            support_trajectories: List of trajectory lists [batch][K]
            
        Returns:
            LoRAWeights for each tool in the batch
        """
        # Encode documentation (already returns correct dtype from doc_encoder)
        v_doc = self.encode_documentation(docs)  # [batch, d_enc]
        
        # Debug: check v_doc gradients
        if self.training and not hasattr(self, '_debug_printed'):
            print(f"[HYPERNETWORK DEBUG] v_doc shape: {v_doc.shape}, grad_fn: {v_doc.grad_fn is not None}")
            self._debug_printed = True
        
        # Encode support set (inherits dtype from doc_encoder)
        v_support, support_mask = self.encode_support_set(
            support_queries, support_trajectories
        )  # [batch, K, d_enc], [batch, K]
        
        # Compute prototype via cross-attention
        v_proto = self.prototype_aggregator(v_doc, v_support, support_mask)  # [batch, d_attn]
        
        # Generate LoRA weights
        lora_weights = self.weight_generator(v_doc, v_proto)
        
        return lora_weights
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hypernetwork(config: ModelConfig) -> MetaToolHypernetwork:
    """Factory function to create hypernetwork."""
    hypernetwork = MetaToolHypernetwork(config)
    
    # Move to device AND convert to correct dtype (float16 for GPU)
    hypernetwork = hypernetwork.to(device=config.device, dtype=config.dtype)
    
    print(f"Created hypernetwork with {hypernetwork.get_num_parameters():,} parameters")
    
    return hypernetwork
