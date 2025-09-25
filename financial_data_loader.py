#!/usr/bin/env python3
"""
Financial Data Loader for HRM-MLX
Loads financial time series data and prepares it for training
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import mlx.core as mx


def load_financial_data(
    csv_path: str, 
    max_samples: int = 1000,
    sequence_length: int = 10,
    target_horizon: int = 1
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load financial data from CSV and prepare sequences for training
    
    Args:
        csv_path: Path to the CSV file
        max_samples: Maximum number of samples to load
        sequence_length: Number of time steps to use as input sequence
        target_horizon: Number of days ahead to predict (1 = next day)
    
    Returns:
        Tuple of (train_data, val_data) where each is a list of dictionaries
        Each dictionary contains:
        - 'features': Input sequence of financial features
        - 'targets': Target values (next high, next low)
        - 'metadata': Additional info like date, etc.
    """
    
    print(f"Loading financial data from {csv_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows of financial data")
    
    # Define feature columns (excluding time and target columns)
    feature_columns = [
        'open', 'high', 'low', 'close', 'EMA 7', 'Volume', 'RSI', 'RSI-based MA',
        'prev_1d_high', 'prev_1d_low', 'prev_2d_high', 'prev_2d_low'
    ]
    
    # Normalize features (z-score normalization)
    feature_data = df[feature_columns].values.astype(np.float32)
    feature_mean = np.mean(feature_data, axis=0)
    feature_std = np.std(feature_data, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)  # Avoid division by zero
    feature_data = (feature_data - feature_mean) / feature_std
    
    # Create sequences
    train_data = []
    val_data = []
    
    # Split data: 80% train, 20% validation
    split_idx = int(len(df) * 0.8)
    
    for split_name, start_idx, end_idx in [("train", 0, split_idx), ("val", split_idx, len(df))]:
        data_list = train_data if split_name == "train" else val_data
        
        for i in range(start_idx, end_idx - sequence_length - target_horizon + 1):
            # Input sequence: features from time i to i+sequence_length-1
            input_features = feature_data[i:i+sequence_length]
            
            # Target: high and low values from time i+sequence_length+target_horizon-1
            target_idx = i + sequence_length + target_horizon - 1
            if target_idx >= len(df):
                continue
                
            target_high = df.iloc[target_idx]['high']
            target_low = df.iloc[target_idx]['low']
            
            # Normalize targets using the same normalization as features
            # (using high/low columns from feature_columns)
            high_idx = feature_columns.index('high')
            low_idx = feature_columns.index('low')
            
            target_high_norm = (target_high - feature_mean[high_idx]) / feature_std[high_idx]
            target_low_norm = (target_low - feature_mean[low_idx]) / feature_std[low_idx]
            
            # Create sample
            sample = {
                'features': input_features.tolist(),
                'targets': [target_high_norm, target_low_norm],
                'metadata': {
                    'date': df.iloc[i]['time'],
                    'target_date': df.iloc[target_idx]['time'],
                    'raw_high': target_high,
                    'raw_low': target_low,
                    'normalization': {
                        'mean': feature_mean.tolist(),
                        'std': feature_std.tolist()
                    }
                }
            }
            
            data_list.append(sample)
            
            # Limit samples if specified
            if len(data_list) >= max_samples:
                break
    
    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")
    print(f"Sequence length: {sequence_length}")
    print(f"Feature dimensions: {len(feature_columns)}")
    
    return train_data, val_data


def create_financial_batch(samples: List[Dict], batch_size: int) -> Dict[str, mx.array]:
    """
    Create a batch from financial samples for training or prediction
    
    Args:
        samples: List of sample dictionaries
        batch_size: Number of samples in the batch
    
    Returns:
        Dictionary containing batched data for the model
    """
    # Randomly sample batch_size samples
    if len(samples) > batch_size:
        indices = np.random.choice(len(samples), batch_size, replace=False)
        batch_samples = [samples[i] for i in indices]
    else:
        batch_samples = samples
    
    # Extract features and targets (if available)
    features = []
    targets = []
    
    for sample in batch_samples:
        features.append(sample['features'])
        # Only add targets if they exist (for prediction, targets may not be available)
        if 'targets' in sample:
            targets.append(sample['targets'])
    
    # Convert to MLX arrays
    features_array = mx.array(features, dtype=mx.float32)  # Shape: (batch_size, seq_len, n_features)
    
    batch = {
        "features": features_array,
        "batch_size": len(batch_samples)
    }
    
    # Add targets only if they exist
    if targets:
        targets_array = mx.array(targets, dtype=mx.float32)    # Shape: (batch_size, 2) [high, low]
        batch["targets"] = targets_array
    else:
        # For prediction, create dummy targets
        batch["targets"] = mx.zeros((len(batch_samples), 2), dtype=mx.float32)
    
    return batch


def denormalize_predictions(predictions: mx.array, normalization_info: Dict) -> mx.array:
    """
    Denormalize predictions back to original scale
    
    Args:
        predictions: Normalized predictions (batch_size, 2) [high, low]
        normalization_info: Dictionary containing mean and std for denormalization
    
    Returns:
        Denormalized predictions in original scale
    """
    mean = mx.array(normalization_info['mean'])
    std = mx.array(normalization_info['std'])
    
    # Get high and low indices from feature columns
    feature_columns = [
        'open', 'high', 'low', 'close', 'EMA 7', 'Volume', 'RSI', 'RSI-based MA',
        'prev_1d_high', 'prev_1d_low', 'prev_2d_high', 'prev_2d_low'
    ]
    
    high_idx = feature_columns.index('high')
    low_idx = feature_columns.index('low')
    
    # Denormalize
    high_mean, high_std = mean[high_idx], std[high_idx]
    low_mean, low_std = mean[low_idx], std[low_idx]
    
    # Create denormalized predictions
    high_denorm = predictions[:, 0] * high_std + high_mean
    low_denorm = predictions[:, 1] * low_std + low_mean
    
    # Stack them together
    denorm_predictions = mx.stack([high_denorm, low_denorm], axis=1)
    
    return denorm_predictions


if __name__ == "__main__":
    # Test the data loader
    train_data, val_data = load_financial_data("data/data1.csv", max_samples=100)
    
    print("\nSample training data:")
    print(f"Features shape: {np.array(train_data[0]['features']).shape}")
    print(f"Targets: {train_data[0]['targets']}")
    print(f"Date: {train_data[0]['metadata']['date']} -> {train_data[0]['metadata']['target_date']}")
    
    # Test batch creation
    batch = create_financial_batch(train_data[:4], 4)
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")
