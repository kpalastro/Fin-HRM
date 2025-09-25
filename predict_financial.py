#!/usr/bin/env python3
"""
Financial HRM Prediction Script
Make predictions for next high and low values using trained model
"""

import argparse
import numpy as np
import pandas as pd
import mlx.core as mx
import os
import glob
from typing import Dict, List, Tuple

from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions
from models.financial_hrm import FinancialHierarchicalReasoningModel


def find_latest_checkpoint(default_dir: str = "checkpoints_financial") -> str:
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
    fallback = os.path.join(default_dir, "final_model.npz")
    return fallback


def load_model(checkpoint_path: str, model_config: Dict) -> FinancialHierarchicalReasoningModel:
    """Load a trained model from checkpoint"""
    print(f"📂 Loading model from {checkpoint_path}")
    
    # Create model with same configuration as training
    model = FinancialHierarchicalReasoningModel(
        n_features=model_config.get('n_features', 12),
        d_model=model_config.get('d_model', 256),
        n_heads=model_config.get('n_heads', 8),
        H_cycles=model_config.get('H_cycles', 2),
        L_cycles=model_config.get('L_cycles', 2),
        H_layers=model_config.get('H_layers', 2),
        L_layers=model_config.get('L_layers', 2),
        halt_max_steps=model_config.get('halt_max_steps', 4),
        halt_exploration_prob=model_config.get('halt_exploration_prob', 0.1),
        seq_len=model_config.get('seq_len', 10),
    )
    
    # Load weights
    model.load_weights(checkpoint_path)
    model.eval()  # Set to evaluation mode
    
    print("✅ Model loaded successfully")
    return model


def prepare_prediction_data(
    data_path: str, 
    sequence_length: int = 5,
    last_n_days: int = None
) -> Tuple[List[Dict], Dict]:
    """
    Prepare data for prediction
    
    Args:
        data_path: Path to CSV file
        sequence_length: Length of input sequence
        last_n_days: Use only last N days (if None, use all data)
    
    Returns:
        Tuple of (samples, normalization_info)
    """
    print(f"📊 Loading data from {data_path}")
    
    # Load data with minimal samples to get normalization info
    train_data, _ = load_financial_data(
        data_path, 
        max_samples=1,  # Just to get normalization info
        sequence_length=sequence_length,
        target_horizon=1
    )
    
    if not train_data:
        raise ValueError("No data loaded")
    
    # Get normalization info from first sample
    normalization_info = train_data[0]['metadata']['normalization']
    
    # Load full data for prediction
    df = pd.read_csv(data_path)
    
    # Use only last N days if specified
    if last_n_days:
        df = df.tail(last_n_days)
    
    print(f"📈 Using {len(df)} days of data for prediction")
    
    # Define feature columns (match data loader)
    feature_columns = [
        'open', 'high', 'low', 'close', 'EMA 7', 'Volume', 'RSI', 'RSI-based MA',
        'prev_1d_high', 'prev_1d_low', 'prev_2d_high', 'prev_2d_low'
    ]
    
    # Normalize features using same normalization as training
    feature_data = df[feature_columns].values.astype(np.float32)
    feature_mean = np.array(normalization_info['mean'])
    feature_std = np.array(normalization_info['std'])
    feature_data = (feature_data - feature_mean) / feature_std
    
    # Create prediction samples
    samples = []
    for i in range(len(df) - sequence_length + 1):
        # Input sequence: features from time i to i+sequence_length-1
        input_features = feature_data[i:i+sequence_length]
        
        # Get date info
        start_date = df.iloc[i]['time']
        end_date = df.iloc[i + sequence_length - 1]['time']
        
        sample = {
            'features': input_features.tolist(),
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'normalization': normalization_info
            }
        }
        samples.append(sample)
    
    print(f"✅ Created {len(samples)} prediction samples")
    return samples, normalization_info


def predict_next_values(
    model: FinancialHierarchicalReasoningModel,
    samples: List[Dict],
    batch_size: int = 8
) -> List[Dict]:
    """
    Make predictions for next high and low values
    
    Args:
        model: Trained Financial HRM model
        samples: List of prediction samples
        batch_size: Batch size for prediction
    
    Returns:
        List of prediction results
    """
    print(f"🔮 Making predictions for {len(samples)} samples...")
    
    results = []
    
    # Process samples in batches
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        
        # Create batch
        batch = create_financial_batch(batch_samples, len(batch_samples))
        
        # Initialize carry
        carry = model.initial_carry(batch)
        
        # Run model until all sequences halt
        step_count = 0
        outputs = None
        while not carry.halted.all() and step_count < model.halt_max_steps:
            carry, outputs = model(carry, batch)
            step_count += 1
        
        # If model halted immediately, run at least one step
        if outputs is None:
            carry, outputs = model(carry, batch)
            step_count = 1
        
        # Get predictions
        predictions = outputs['predictions']  # (batch_size, 2) [high, low]
        
        # Denormalize predictions
        if batch_samples:
            norm_info = batch_samples[0]['metadata']['normalization']
            denorm_predictions = denormalize_predictions(predictions, norm_info)
        else:
            denorm_predictions = predictions
        
        # Store results
        for j, sample in enumerate(batch_samples):
            result = {
                'start_date': sample['metadata']['start_date'],
                'end_date': sample['metadata']['end_date'],
                'predicted_high': float(denorm_predictions[j, 0]),
                'predicted_low': float(denorm_predictions[j, 1]),
                'predicted_range': float(denorm_predictions[j, 0] - denorm_predictions[j, 1]),
                'steps_used': step_count,
                'normalized_high': float(predictions[j, 0]),
                'normalized_low': float(predictions[j, 1]),
            }
            results.append(result)
    
    print(f"✅ Completed predictions")
    return results


def print_predictions(results: List[Dict], show_last: int = 10):
    """Print prediction results in a formatted table"""
    print(f"\n📊 PREDICTION RESULTS (showing last {show_last} predictions)")
    print("=" * 100)
    print(f"{'Start Date':<12} {'End Date':<12} {'Pred High':<10} {'Pred Low':<10} {'Range':<10} {'Steps':<6}")
    print("-" * 100)
    
    for result in results[-show_last:]:
        print(f"{result['start_date']:<12} {result['end_date']:<12} "
              f"{result['predicted_high']:<10.2f} {result['predicted_low']:<10.2f} "
              f"{result['predicted_range']:<10.2f} {result['steps_used']:<6}")
    
    print("=" * 100)


def save_predictions(results: List[Dict], output_path: str):
    """Save predictions to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Financial HRM Prediction")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (auto-discovered if not provided)")
    parser.add_argument("--data_path", type=str, default="data/data1.csv", help="Path to financial data CSV")
    
    # Prediction parameters
    parser.add_argument("--sequence_length", type=int, default=10, help="Input sequence length")
    parser.add_argument("--last_n_days", type=int, default=None, help="Use only last N days for prediction")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--show_last", type=int, default=10, help="Number of recent predictions to show")
    
    # Model configuration (should match training)
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--H_cycles", type=int, default=2, help="High-level cycles")
    parser.add_argument("--L_cycles", type=int, default=2, help="Low-level cycles")
    parser.add_argument("--H_layers", type=int, default=2, help="High-level layers")
    parser.add_argument("--L_layers", type=int, default=2, help="Low-level layers")
    parser.add_argument("--halt_max_steps", type=int, default=4, help="Maximum ACT steps")
    
    # Output
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("🔮 Financial HRM Prediction")
    print("=" * 50)
    
    # Model configuration
    model_config = {
        'n_features': 12,
        'd_model': args.d_model,
        'n_heads': 8,
        'H_cycles': args.H_cycles,
        'L_cycles': args.L_cycles,
        'H_layers': args.H_layers,
        'L_layers': args.L_layers,
        'halt_max_steps': args.halt_max_steps,
        'halt_exploration_prob': 0.1,
        'seq_len': args.sequence_length,
    }
    
    try:
        # Auto-discover checkpoint if not provided
        checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint()
        
        # Load model
        model = load_model(checkpoint_path, model_config)
        
        # Prepare data
        samples, norm_info = prepare_prediction_data(
            args.data_path, 
            args.sequence_length,
            args.last_n_days
        )
        
        # Make predictions
        results = predict_next_values(model, samples, args.batch_size)
        
        # Display results
        print_predictions(results, args.show_last)
        
        # Save results
        save_predictions(results, args.output)
        
        # Summary statistics
        if results:
            high_preds = [r['predicted_high'] for r in results]
            low_preds = [r['predicted_low'] for r in results]
            ranges = [r['predicted_range'] for r in results]
            
            print(f"\n📈 PREDICTION SUMMARY")
            print(f"   Average Predicted High: {np.mean(high_preds):.2f}")
            print(f"   Average Predicted Low:  {np.mean(low_preds):.2f}")
            print(f"   Average Predicted Range: {np.mean(ranges):.2f}")
            print(f"   High Range: {min(high_preds):.2f} - {max(high_preds):.2f}")
            print(f"   Low Range:  {min(low_preds):.2f} - {max(low_preds):.2f}")
        
        print(f"\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
