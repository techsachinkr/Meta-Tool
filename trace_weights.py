#!/usr/bin/env python3
"""
Diagnostic to trace where LoRA weights get lost between generation and application.
"""

import torch
from config import get_config
from hypernetwork import create_hypernetwork
from lora_integration import MetaToolAdaptedModel

def trace_weight_application():
    print("="*60)
    print("TRACING WEIGHT APPLICATION")
    print("="*60)
    
    # Load config and create models
    config = get_config("large")
    hypernetwork = create_hypernetwork(config.model)
    adapted_model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    # Test data
    test_doc = "This is a test API documentation for image classification."
    test_queries = ["Load a ResNet model", "I need MobileNet"]
    test_trajectories = ["torchvision.models.resnet50(pretrained=True)", 
                         "torchvision.models.mobilenet_v2(pretrained=True)"]
    
    # Step 1: Check LoRA layers before adaptation
    print("\n1. LORA LAYERS BEFORE ADAPTATION:")
    lora_layers = adapted_model.adapted_model.lora_layers
    print(f"   Number of LoRA layers: {len(lora_layers)}")
    print(f"   Sample keys: {list(lora_layers.keys())[:5]}")
    
    for key, layer in list(lora_layers.items())[:3]:
        a_val = "None" if layer.lora_A is None else f"norm={layer.lora_A.norm().item():.4f}"
        b_val = "None" if layer.lora_B is None else f"norm={layer.lora_B.norm().item():.4f}"
        print(f"   {key}: A={a_val}, B={b_val}")
    
    # Step 2: Generate weights via hypernetwork
    print("\n2. GENERATING WEIGHTS VIA HYPERNETWORK:")
    with torch.no_grad():
        lora_weights = hypernetwork(
            docs=[test_doc],
            support_queries=[test_queries],
            support_trajectories=[test_trajectories]
        )
    
    print(f"   Generated keys (first 5): {list(lora_weights.A_matrices.keys())[:5]}")
    print(f"   LoRA layer keys (first 5): {list(lora_layers.keys())[:5]}")
    
    # Check if keys match
    gen_keys = set(lora_weights.A_matrices.keys())
    layer_keys = set(lora_layers.keys())
    matching_keys = gen_keys & layer_keys
    missing_in_gen = layer_keys - gen_keys
    missing_in_layers = gen_keys - layer_keys
    
    print(f"\n   Matching keys: {len(matching_keys)}")
    print(f"   Missing in generated: {len(missing_in_gen)}")
    print(f"   Missing in LoRA layers: {len(missing_in_layers)}")
    
    if missing_in_gen:
        print(f"   Examples missing in generated: {list(missing_in_gen)[:3]}")
    if missing_in_layers:
        print(f"   Examples missing in layers: {list(missing_in_layers)[:3]}")
    
    # Step 3: Manually apply weights and check
    print("\n3. MANUALLY APPLYING WEIGHTS:")
    
    # Get a sample key that exists in both
    if matching_keys:
        sample_key = list(matching_keys)[0]
        print(f"   Testing with key: {sample_key}")
        
        A = lora_weights.A_matrices[sample_key]
        B = lora_weights.B_matrices[sample_key]
        print(f"   Generated A: shape={A.shape}, norm={A.norm().item():.4f}")
        print(f"   Generated B: shape={B.shape}, norm={B.norm().item():.4f}")
        
        # Check if batched
        if A.dim() == 3:
            A = A[0]
            B = B[0]
            print(f"   After unbatching - A: shape={A.shape}, B: shape={B.shape}")
        
        # Get the LoRA layer
        lora_layer = lora_layers[sample_key]
        print(f"   LoRA layer base: in={lora_layer.base_layer.in_features}, out={lora_layer.base_layer.out_features}")
        print(f"   LoRA layer rank: {lora_layer.rank}")
        
        # Try to set weights manually
        print("\n   Calling set_lora_weights()...")
        lora_layer.set_lora_weights(A, B)
        
        # Check if weights were set
        if lora_layer.lora_A is not None:
            print(f"   After set - A: shape={lora_layer.lora_A.shape}, norm={lora_layer.lora_A.norm().item():.4f}")
            print(f"   After set - B: shape={lora_layer.lora_B.shape}, norm={lora_layer.lora_B.norm().item():.4f}")
            print(f"   lora_enabled: {lora_layer.lora_enabled}")
        else:
            print("   ❌ lora_A is still None after set_lora_weights!")
    
    # Step 4: Use the official adapt_to_tool method
    print("\n4. USING OFFICIAL adapt_to_tool METHOD:")
    
    # Clear first
    adapted_model.adapted_model.clear_lora_weights()
    
    # Adapt
    adapted_model.adapt_to_tool(
        documentation=test_doc,
        support_queries=test_queries,
        support_trajectories=test_trajectories
    )
    
    # Check again
    print("   After adapt_to_tool:")
    for key, layer in list(lora_layers.items())[:3]:
        if layer.lora_A is not None:
            a_norm = layer.lora_A.norm().item()
            b_norm = layer.lora_B.norm().item()
            print(f"   {key}: A_norm={a_norm:.4f}, B_norm={b_norm:.4f}, enabled={layer.lora_enabled}")
        else:
            print(f"   {key}: A=None, B=None")
    
    # Step 5: Check if there's a device mismatch
    print("\n5. DEVICE CHECK:")
    for key, layer in list(lora_layers.items())[:1]:
        base_device = layer.base_layer.weight.device
        print(f"   Base layer device: {base_device}")
        if layer.lora_A is not None:
            print(f"   LoRA A device: {layer.lora_A.device}")
            print(f"   LoRA B device: {layer.lora_B.device}")


if __name__ == "__main__":
    trace_weight_application()
