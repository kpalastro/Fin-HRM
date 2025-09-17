#!/usr/bin/env python3
"""
Test script for Financial HRM training
Quick test to verify the model works with small data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions
from models.financial_hrm import FinancialHierarchicalReasoningModel
from models.financial_losses import compute_financial_act_loss
import mlx.core as mx
import mlx.utils
import numpy as np

def test_financial_model():
    """Test the financial model with small data"""
    print("🧪 Testing Financial HRM Model...")
    
    # Load small dataset
    print("📊 Loading data...")
    train_data, val_data = load_financial_data(
        "data/data1.csv", 
        max_samples=20,
        sequence_length=5,  # Shorter sequence for testing
        target_horizon=1
    )
    
    print(f"✅ Train samples: {len(train_data)}")
    print(f"✅ Val samples: {len(val_data)}")
    
    # Create small model
    print("🤖 Creating model...")
    model = FinancialHierarchicalReasoningModel(
        n_features=18,
        d_model=64,  # Small model for testing
        n_heads=4,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        halt_max_steps=2,
        halt_exploration_prob=0.1,
        seq_len=5,
    )
    
    # Count parameters
    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"✅ Model parameters: {total_params:,}")
    
    # Test forward pass
    print("🔄 Testing forward pass...")
    batch = create_financial_batch(train_data[:2], 2)
    
    # Initialize carry
    carry = model.initial_carry(batch)
    print(f"✅ Initial carry created")
    
    # Forward pass
    carry, outputs = model(carry, batch)
    print(f"✅ Forward pass completed")
    print(f"   Predictions shape: {outputs['predictions'].shape}")
    print(f"   Q-halt shape: {outputs['q_halt_logits'].shape}")
    print(f"   Q-continue shape: {outputs['q_continue_logits'].shape}")
    
    # Test loss computation
    print("📊 Testing loss computation...")
    loss, metrics = compute_financial_act_loss(outputs, batch["targets"])
    print(f"✅ Loss computed: {float(loss):.4f}")
    print(f"   MSE: {metrics['mse_loss']:.4f}")
    print(f"   MAE: {metrics['mae_loss']:.4f}")
    print(f"   R²: {metrics['r2_overall']:.3f}")
    
    # Test multiple steps
    print("🔄 Testing multiple ACT steps...")
    carry = model.initial_carry(batch)
    step_count = 0
    while not carry.halted.all() and step_count < model.halt_max_steps:
        carry, outputs = model(carry, batch)
        step_count += 1
        print(f"   Step {step_count}: Halted {carry.halted.sum().item()}/{len(carry.halted)}")
    
    print(f"✅ Completed in {step_count} steps")
    
    # Test denormalization
    print("🔄 Testing denormalization...")
    if len(train_data) > 0:
        norm_info = train_data[0]['metadata']['normalization']
        denorm_preds = denormalize_predictions(outputs['predictions'], norm_info)
        print(f"✅ Denormalized predictions shape: {denorm_preds.shape}")
        print(f"   Sample prediction: High={denorm_preds[0, 0].item():.2f}, Low={denorm_preds[0, 1].item():.2f}")
    
    print("\n🎉 All tests passed! Model is ready for training.")

if __name__ == "__main__":
    test_financial_model()
