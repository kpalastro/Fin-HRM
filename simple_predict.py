#!/usr/bin/env python3
"""
Simple Financial HRM Prediction Example
Shows how to make a single prediction for next high and low values
"""

import numpy as np
import pandas as pd
import mlx.core as mx
from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions
from models.financial_hrm import FinancialHierarchicalReasoningModel


def make_single_prediction(checkpoint_path: str, data_path: str = "data/data1.csv"):
    """
    Make a single prediction for the next high and low values
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to financial data CSV
    """
    print("🔮 Making Single Prediction for Next High and Low Values")
    print("=" * 60)
    
    # Load a small amount of data to get normalization info
    train_data, _ = load_financial_data(data_path, max_samples=1, sequence_length=5, target_horizon=1)
    if not train_data:
        print("❌ No data loaded")
        return
    
    # Get normalization info
    norm_info = train_data[0]['metadata']['normalization']
    
    # Load the actual data
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} days of financial data")
    
    # Use the last 5 days for prediction (matching our sequence length)
    last_5_days = df.tail(5)
    print(f"📈 Using last 5 days: {last_5_days['time'].iloc[0]} to {last_5_days['time'].iloc[-1]}")
    
    # Prepare features
    feature_columns = [
        'open', 'high', 'low', 'close', 'EMA 7', 'Volume', 'RSI', 'RSI-based MA',
        'prev_1d_high', 'prev_1d_low', 'prev_2d_high', 'prev_2d_low', 
        'prev_3d_high', 'prev_3d_low', 'prev_4d_high', 'prev_4d_low',
        'prev_5d_high', 'prev_5d_low'
    ]
    
    # Normalize features
    feature_data = last_5_days[feature_columns].values.astype(np.float32)
    feature_mean = np.array(norm_info['mean'])
    feature_std = np.array(norm_info['std'])
    normalized_features = (feature_data - feature_mean) / feature_std
    
    # Create model (same config as training)
    model = FinancialHierarchicalReasoningModel(
        n_features=18,
        d_model=128,
        n_heads=8,
        H_cycles=1,
        L_cycles=1,
        H_layers=2,
        L_layers=2,
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        seq_len=5,
    )
    
    # Load trained weights
    model.load_weights(checkpoint_path)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Create batch for prediction
    batch = {
        "features": mx.array(normalized_features.reshape(1, 5, 18), dtype=mx.float32),
        "targets": mx.zeros((1, 2), dtype=mx.float32)  # Dummy targets for prediction
    }
    
    # Make prediction
    print("🔮 Making prediction...")
    carry = model.initial_carry(batch)
    
    # Run model until halted
    step_count = 0
    outputs = None
    while not carry.halted.all() and step_count < model.halt_max_steps:
        carry, outputs = model(carry, batch)
        step_count += 1
    
    # If model halted immediately, run at least one step
    if outputs is None:
        carry, outputs = model(carry, batch)
        step_count = 1
    
    # Get prediction
    predictions = outputs['predictions']  # Shape: (1, 2) [high, low]
    
    # Denormalize prediction
    denorm_predictions = denormalize_predictions(predictions, norm_info)
    
    # Extract values
    predicted_high = float(denorm_predictions[0, 0])
    predicted_low = float(denorm_predictions[0, 1])
    predicted_range = predicted_high - predicted_low
    
    # Display results
    print("\n📊 PREDICTION RESULTS")
    print("=" * 40)
    print(f"Next Day Prediction:")
    print(f"  Predicted High: ${predicted_high:,.2f}")
    print(f"  Predicted Low:  ${predicted_low:,.2f}")
    print(f"  Predicted Range: ${predicted_range:,.2f}")
    print(f"  Reasoning Steps Used: {step_count}")
    
    # Show current values for comparison
    current_high = last_5_days['high'].iloc[-1]
    current_low = last_5_days['low'].iloc[-1]
    current_close = last_5_days['close'].iloc[-1]
    
    print(f"\n📈 Current Values (Last Day):")
    print(f"  Current High:  ${current_high:,.2f}")
    print(f"  Current Low:   ${current_low:,.2f}")
    print(f"  Current Close: ${current_close:,.2f}")
    
    # Calculate changes
    high_change = predicted_high - current_close
    low_change = predicted_low - current_close
    
    print(f"\n📊 Predicted Changes from Current Close:")
    print(f"  High Change:  {high_change:+,.2f} ({high_change/current_close*100:+.2f}%)")
    print(f"  Low Change:   {low_change:+,.2f} ({low_change/current_close*100:+.2f}%)")
    
    return {
        'predicted_high': predicted_high,
        'predicted_low': predicted_low,
        'predicted_range': predicted_range,
        'steps_used': step_count,
        'current_high': current_high,
        'current_low': current_low,
        'current_close': current_close
    }


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints_financial/best_model_step_280.npz"  # Use the best checkpoint
    
    try:
        result = make_single_prediction(checkpoint_path)
        print(f"\n✅ Prediction completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a trained model checkpoint available.")
