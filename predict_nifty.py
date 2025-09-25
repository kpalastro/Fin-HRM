#!/usr/bin/env python3
"""
NIFTY Financial HRM Prediction Script
Make predictions using the trained NIFTY model
"""

import argparse
import numpy as np
import pandas as pd
import mlx.core as mx
import os
import glob
from typing import Dict, List

from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions
from models.financial_hrm import FinancialHierarchicalReasoningModel


def find_latest_nifty_checkpoint(default_dir: str = "checkpoints_nifty") -> str:
    """Find the most recent valid .npz checkpoint file."""
    try:
        candidates = sorted(
            glob.glob(os.path.join(default_dir, "*.npz")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for path in candidates:
            try:
                if os.path.getsize(path) > 0:
                    return path
            except OSError:
                continue
    except Exception:
        pass
    # Fallbacks
    fallback = os.path.join(default_dir, "best_model_step_5000.npz")
    return fallback


def load_nifty_model(checkpoint_path: str) -> FinancialHierarchicalReasoningModel:
    """Load the trained NIFTY model"""
    print(f"📂 Loading NIFTY model from {checkpoint_path}")
    
    # Create model with same configuration as training
    model = FinancialHierarchicalReasoningModel(
        n_features=12,  # Number of financial features
        d_model=512,
        n_heads=8,
        H_cycles=2,
        L_cycles=2,
        H_layers=4,
        L_layers=4,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        seq_len=10,
    )
    
    # Load weights
    model.load_weights(checkpoint_path)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def predict_next_values(
    model: FinancialHierarchicalReasoningModel,
    data: List[Dict],
    n_predictions: int = 5
) -> List[Dict]:
    """
    Make predictions for the next high and low values
    
    Args:
        model: Trained Financial HRM model
        data: List of financial data samples
        n_predictions: Number of predictions to make
    
    Returns:
        List of prediction results
    """
    print(f"🔮 Making {n_predictions} predictions...")
    
    predictions = []
    
    for i in range(min(n_predictions, len(data))):
        sample = data[i]
        
        # Create batch with single sample
        batch = create_financial_batch([sample], 1)
        
        # Initialize carry and run model
        carry = model.initial_carry(batch)
        
        # Run until halted
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
        pred = outputs['predictions']  # Shape: (1, 2) [high, low]
        
        # Denormalize prediction
        norm_info = sample['metadata']['normalization']
        denorm_pred = denormalize_predictions(pred, norm_info)
        
        # Get actual values for comparison
        actual = mx.array(sample['targets'])  # Shape: (2,) [high, low]
        denorm_actual = denormalize_predictions(actual.reshape(1, -1), norm_info)[0]
        
        # Calculate errors
        pred_high = float(denorm_pred[0, 0])
        pred_low = float(denorm_pred[0, 1])
        actual_high = float(denorm_actual[0])
        actual_low = float(denorm_actual[1])
        
        high_error = abs(pred_high - actual_high)
        low_error = abs(pred_low - actual_low)
        high_error_pct = (high_error / actual_high) * 100
        low_error_pct = (low_error / actual_low) * 100
        
        prediction_result = {
            'sample_idx': i,
            'predicted_high': pred_high,
            'predicted_low': pred_low,
            'actual_high': actual_high,
            'actual_low': actual_low,
            'high_error': high_error,
            'low_error': low_error,
            'high_error_pct': high_error_pct,
            'low_error_pct': low_error_pct,
            'predicted_range': pred_high - pred_low,
            'actual_range': actual_high - actual_low,
            'range_error': abs((pred_high - pred_low) - (actual_high - actual_low)),
            'steps_used': step_count
        }
        
        predictions.append(prediction_result)
        
        print(f"  Sample {i+1}:")
        print(f"    Predicted: High={pred_high:.2f}, Low={pred_low:.2f}")
        print(f"    Actual:    High={actual_high:.2f}, Low={actual_low:.2f}")
        print(f"    Error:     High={high_error:.2f} ({high_error_pct:.1f}%), Low={low_error:.2f} ({low_error_pct:.1f}%)")
        print(f"    Range:     Pred={pred_high-pred_low:.2f}, Actual={actual_high-actual_low:.2f}")
        print(f"    Steps:     {step_count}")
        print()
    
    return predictions


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions to CSV file"""
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to {output_path}")


def print_summary(predictions: List[Dict]):
    """Print summary statistics"""
    if not predictions:
        return
    
    high_errors = [p['high_error_pct'] for p in predictions]
    low_errors = [p['low_error_pct'] for p in predictions]
    range_errors = [p['range_error'] for p in predictions]
    steps_used = [p['steps_used'] for p in predictions]
    
    print("📊 PREDICTION SUMMARY")
    print("=" * 50)
    print(f"High Error:     {np.mean(high_errors):.2f}% ± {np.std(high_errors):.2f}%")
    print(f"Low Error:      {np.mean(low_errors):.2f}% ± {np.std(low_errors):.2f}%")
    print(f"Range Error:    {np.mean(range_errors):.2f} ± {np.std(range_errors):.2f}")
    print(f"Avg Steps:      {np.mean(steps_used):.1f} ± {np.std(steps_used):.1f}")
    print(f"Range Validity: {sum(1 for p in predictions if p['predicted_high'] > p['predicted_low']) / len(predictions) * 100:.1f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="NIFTY Financial HRM Prediction")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (auto-discovered if not provided)")
    parser.add_argument("--data_path", type=str, default="data/NIFTY_DAILY.csv", help="Path to NIFTY data CSV")
    
    # Prediction parameters
    parser.add_argument("--n_predictions", type=int, default=10, help="Number of predictions to make")
    parser.add_argument("--sequence_length", type=int, default=10, help="Input sequence length")
    parser.add_argument("--output", type=str, default="nifty_predictions.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("🔮 NIFTY Financial HRM Prediction")
    print("=" * 40)
    
    try:
        # Auto-discover checkpoint if not provided
        checkpoint_path = args.checkpoint if args.checkpoint else find_latest_nifty_checkpoint()
        
        # Load model
        model = load_nifty_model(checkpoint_path)
        
        # Load data
        print("📊 Loading NIFTY data...")
        train_data, test_data = load_financial_data(
            args.data_path, 
            max_samples=args.n_predictions + 100,  # Load a bit more for variety
            sequence_length=args.sequence_length,
            target_horizon=1
        )
        
        print(f"✅ Loaded {len(test_data)} test samples")
        
        # Make predictions
        predictions = predict_next_values(model, test_data, args.n_predictions)
        
        # Print summary
        print_summary(predictions)
        
        # Save predictions
        save_predictions(predictions, args.output)
        
        print(f"\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
