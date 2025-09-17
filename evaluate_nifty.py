#!/usr/bin/env python3
"""
NIFTY Financial HRM Evaluation Script
Evaluate trained model on NIFTY data with comprehensive metrics
"""

import argparse
import numpy as np
import pandas as pd
import mlx.core as mx
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions
from models.financial_hrm import FinancialHierarchicalReasoningModel
from models.financial_losses import compute_simple_regression_loss


def load_model(checkpoint_path: str, model_config: Dict) -> FinancialHierarchicalReasoningModel:
    """Load a trained model from checkpoint"""
    print(f"📂 Loading model from {checkpoint_path}")
    
    # Create model with same configuration as training
    model = FinancialHierarchicalReasoningModel(
        n_features=model_config.get('n_features', 18),
        d_model=model_config.get('d_model', 512),
        n_heads=model_config.get('n_heads', 8),
        H_cycles=model_config.get('H_cycles', 2),
        L_cycles=model_config.get('L_cycles', 2),
        H_layers=model_config.get('H_layers', 4),
        L_layers=model_config.get('L_layers', 4),
        halt_max_steps=model_config.get('halt_max_steps', 16),
        halt_exploration_prob=model_config.get('halt_exploration_prob', 0.1),
        seq_len=model_config.get('seq_len', 10),
    )
    
    # Load weights
    model.load_weights(checkpoint_path)
    model.eval()  # Set to evaluation mode
    
    print("✅ Model loaded successfully")
    return model


def evaluate_model(
    model: FinancialHierarchicalReasoningModel,
    test_data: List[Dict],
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the model
    
    Args:
        model: Trained Financial HRM model
        test_data: List of test samples
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"🔍 Evaluating model on {len(test_data)} test samples...")
    
    all_predictions = []
    all_targets = []
    all_metrics = []
    
    # Process in batches
    for i in range(0, len(test_data), batch_size):
        batch_samples = test_data[i:i+batch_size]
        batch = create_financial_batch(batch_samples, len(batch_samples))
        
        # Initialize carry and run model
        carry = model.initial_carry(batch)
        
        # Run until all sequences halt
        step_count = 0
        while not carry.halted.all() and step_count < model.halt_max_steps:
            carry, outputs = model(carry, batch)
            step_count += 1
        
        # Get predictions
        predictions = outputs['predictions']  # (batch_size, 2) [high, low]
        targets = batch['targets']  # (batch_size, 2) [high, low]
        
        # Denormalize predictions
        if batch_samples:
            norm_info = batch_samples[0]['metadata']['normalization']
            denorm_predictions = denormalize_predictions(predictions, norm_info)
            denorm_targets = denormalize_predictions(targets, norm_info)
        else:
            denorm_predictions = predictions
            denorm_targets = targets
        
        # Store results
        all_predictions.append(denorm_predictions)
        all_targets.append(denorm_targets)
        
        # Compute batch metrics
        loss, metrics = compute_simple_regression_loss(predictions, targets)
        all_metrics.append(metrics)
    
    # Combine all results
    all_predictions = mx.concatenate(all_predictions, axis=0)
    all_targets = mx.concatenate(all_targets, axis=0)
    
    # Convert to numpy for analysis
    pred_high = all_predictions[:, 0].astype(mx.float32)
    pred_low = all_predictions[:, 1].astype(mx.float32)
    true_high = all_targets[:, 0].astype(mx.float32)
    true_low = all_targets[:, 1].astype(mx.float32)
    
    # Basic regression metrics
    mse_high = float(mx.mean((pred_high - true_high) ** 2))
    mse_low = float(mx.mean((pred_low - true_low) ** 2))
    mse_overall = (mse_high + mse_low) / 2
    
    mae_high = float(mx.mean(mx.abs(pred_high - true_high)))
    mae_low = float(mx.mean(mx.abs(pred_low - true_low)))
    mae_overall = (mae_high + mae_low) / 2
    
    # R² score
    def r2_score(y_true, y_pred):
        ss_res = mx.sum((y_true - y_pred) ** 2)
        ss_tot = mx.sum((y_true - mx.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    r2_high = float(r2_score(true_high, pred_high))
    r2_low = float(r2_score(true_low, pred_low))
    r2_overall = (r2_high + r2_low) / 2
    
    # Direction accuracy (whether prediction direction matches target direction)
    high_direction_correct = float(mx.mean((mx.sign(pred_high - true_high) == mx.sign(true_high - true_low)) | 
                                         (mx.abs(pred_high - true_high) < 0.01)))
    low_direction_correct = float(mx.mean((mx.sign(pred_low - true_low) == mx.sign(true_low - true_high)) | 
                                        (mx.abs(pred_low - true_low) < 0.01)))
    
    # Range validity (predicted high > predicted low)
    range_valid = float(mx.mean((pred_high > pred_low).astype(mx.float32)))
    
    # Range accuracy
    pred_range = pred_high - pred_low
    true_range = true_high - true_low
    range_mae = float(mx.mean(mx.abs(pred_range - true_range)))
    range_r2 = float(r2_score(true_range, pred_range))
    
    # Percentage errors
    mape_high = float(mx.mean(mx.abs((pred_high - true_high) / (true_high + 1e-8)) * 100))
    mape_low = float(mx.mean(mx.abs((pred_low - true_low) / (true_low + 1e-8)) * 100))
    mape_overall = (mape_high + mape_low) / 2
    
    # Volatility prediction accuracy
    high_volatility = mx.abs(true_high - true_low) > mx.mean(mx.abs(true_high - true_low))
    pred_high_vol = pred_high > pred_low
    volatility_accuracy = float(mx.mean((high_volatility == pred_high_vol).astype(mx.float32)))
    
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    evaluation_results = {
        # Basic metrics
        'mse_overall': mse_overall,
        'mse_high': mse_high,
        'mse_low': mse_low,
        'mae_overall': mae_overall,
        'mae_high': mae_high,
        'mae_low': mae_low,
        'r2_overall': r2_overall,
        'r2_high': r2_high,
        'r2_low': r2_low,
        
        # Advanced metrics
        'mape_overall': mape_overall,
        'mape_high': mape_high,
        'mape_low': mape_low,
        'range_mae': range_mae,
        'range_r2': range_r2,
        'range_valid': range_valid,
        'high_direction_acc': high_direction_correct,
        'low_direction_acc': low_direction_correct,
        'volatility_accuracy': volatility_accuracy,
        
        # Additional metrics from loss computation
        **avg_metrics
    }
    
    return evaluation_results


def print_evaluation_results(results: Dict[str, float]):
    """Print evaluation results in a formatted table"""
    print("\n📊 EVALUATION RESULTS")
    print("=" * 60)
    
    # Basic regression metrics
    print("📈 Basic Regression Metrics:")
    print(f"  MSE (Overall):     {results['mse_overall']:.4f}")
    print(f"  MSE (High):        {results['mse_high']:.4f}")
    print(f"  MSE (Low):         {results['mse_low']:.4f}")
    print(f"  MAE (Overall):     {results['mae_overall']:.4f}")
    print(f"  MAE (High):        {results['mae_high']:.4f}")
    print(f"  MAE (Low):         {results['mae_low']:.4f}")
    print()
    
    # R² scores
    print("📊 R² Scores:")
    print(f"  R² (Overall):      {results['r2_overall']:.4f}")
    print(f"  R² (High):         {results['r2_high']:.4f}")
    print(f"  R² (Low):          {results['r2_low']:.4f}")
    print()
    
    # Percentage errors
    print("📉 Percentage Errors:")
    print(f"  MAPE (Overall):    {results['mape_overall']:.2f}%")
    print(f"  MAPE (High):       {results['mape_high']:.2f}%")
    print(f"  MAPE (Low):        {results['mape_low']:.2f}%")
    print()
    
    # Range metrics
    print("📏 Range Prediction:")
    print(f"  Range MAE:         {results['range_mae']:.4f}")
    print(f"  Range R²:          {results['range_r2']:.4f}")
    print(f"  Range Validity:    {results['range_valid']:.4f}")
    print()
    
    # Direction accuracy
    print("🎯 Direction Accuracy:")
    print(f"  High Direction:    {results['high_direction_acc']:.4f}")
    print(f"  Low Direction:     {results['low_direction_acc']:.4f}")
    print(f"  Volatility Acc:    {results['volatility_accuracy']:.4f}")
    print()
    
    # Overall assessment
    print("🏆 Overall Assessment:")
    if results['r2_overall'] > 0.8:
        print("  ✅ Excellent prediction quality")
    elif results['r2_overall'] > 0.6:
        print("  ✅ Good prediction quality")
    elif results['r2_overall'] > 0.4:
        print("  ⚠️  Moderate prediction quality")
    else:
        print("  ❌ Poor prediction quality")
    
    if results['range_valid'] > 0.95:
        print("  ✅ High range validity")
    elif results['range_valid'] > 0.9:
        print("  ⚠️  Moderate range validity")
    else:
        print("  ❌ Low range validity")
    
    print("=" * 60)


def save_evaluation_results(results: Dict[str, float], output_path: str):
    """Save evaluation results to CSV file"""
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    print(f"💾 Evaluation results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NIFTY Financial HRM Evaluation")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/NIFTY_DAILY.csv", help="Path to NIFTY data CSV")
    
    # Evaluation parameters
    parser.add_argument("--sequence_length", type=int, default=10, help="Input sequence length")
    parser.add_argument("--target_horizon", type=int, default=1, help="Days ahead to predict")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    # Model configuration (should match training)
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--H_cycles", type=int, default=2, help="High-level cycles")
    parser.add_argument("--L_cycles", type=int, default=2, help="Low-level cycles")
    parser.add_argument("--H_layers", type=int, default=4, help="High-level layers")
    parser.add_argument("--L_layers", type=int, default=4, help="Low-level layers")
    parser.add_argument("--halt_max_steps", type=int, default=16, help="Maximum ACT steps")
    
    # Output
    parser.add_argument("--output", type=str, default="nifty_evaluation_results.csv", help="Output CSV file")

    args = parser.parse_args()
    
    print("📊 NIFTY Financial HRM Evaluation")
    print("=" * 50)
    
    # Model configuration
    model_config = {
        'n_features': 18,
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
        # Load model
        model = load_model(args.checkpoint, model_config)
        
        # Load test data
        print("📊 Loading test data...")
        train_data, test_data = load_financial_data(
            args.data_path, 
            max_samples=args.test_samples,
            sequence_length=args.sequence_length,
            target_horizon=args.target_horizon
        )
        
        print(f"✅ Test samples: {len(test_data):,}")
        
        # Evaluate model
        results = evaluate_model(model, test_data, args.batch_size)
        
        # Display results
        print_evaluation_results(results)
        
        # Save results
        save_evaluation_results(results, args.output)
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
