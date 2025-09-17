"""
Loss functions for Financial HRM regression tasks
"""

import mlx.core as mx
import mlx.nn as nn


def compute_financial_act_loss(outputs: dict, targets: mx.array) -> tuple[mx.array, dict]:
    """
    Compute loss for Financial HRM with ACT
    
    Args:
        outputs: Model outputs containing predictions, q_halt_logits, q_continue_logits, target_q_continue
        targets: Target values (batch_size, 2) [high, low]
    
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    
    predictions = outputs["predictions"]  # (batch_size, 2) [high, low]
    q_halt_logits = outputs["q_halt_logits"]
    q_continue_logits = outputs["q_continue_logits"]
    
    # Regression loss (MSE)
    mse_loss = mx.mean((predictions - targets) ** 2)
    
    # Individual losses for high and low
    high_loss = mx.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    low_loss = mx.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    
    # MAE for additional metrics
    mae_loss = mx.mean(mx.abs(predictions - targets))
    high_mae = mx.mean(mx.abs(predictions[:, 0] - targets[:, 0]))
    low_mae = mx.mean(mx.abs(predictions[:, 1] - targets[:, 1]))
    
    # Q-learning loss (if in training mode)
    q_loss = mx.array(0.0)
    if "target_q_continue" in outputs:
        target_q_continue = outputs["target_q_continue"]
        q_continue_probs = mx.sigmoid(q_continue_logits)
        q_loss = mx.mean((q_continue_probs - target_q_continue) ** 2)
    
    # Total loss
    total_loss = mse_loss + 0.1 * q_loss  # Weight Q-loss lower for regression focus
    
    # Compute metrics
    # R² score (coefficient of determination)
    target_mean = mx.mean(targets, axis=0)
    ss_tot = mx.sum((targets - target_mean) ** 2, axis=0)
    ss_res = mx.sum((targets - predictions) ** 2, axis=0)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    r2_high = r2_score[0]
    r2_low = r2_score[1]
    r2_overall = mx.mean(r2_score)
    
    # Direction accuracy (whether prediction direction matches target direction)
    # This is more meaningful for financial data
    high_direction_correct = mx.mean((mx.sign(predictions[:, 0] - targets[:, 0]) == mx.sign(targets[:, 0] - targets[:, 1])) | 
                                   (mx.abs(predictions[:, 0] - targets[:, 0]) < 0.01))  # Allow small errors
    low_direction_correct = mx.mean((mx.sign(predictions[:, 1] - targets[:, 1]) == mx.sign(targets[:, 1] - targets[:, 0])) | 
                                  (mx.abs(predictions[:, 1] - targets[:, 1]) < 0.01))
    
    # Range accuracy (whether predicted high > predicted low)
    range_valid = mx.mean((predictions[:, 0] > predictions[:, 1]).astype(mx.float32))
    
    metrics = {
        "mse_loss": float(mse_loss),
        "mae_loss": float(mae_loss),
        "high_mse": float(high_loss),
        "low_mse": float(low_loss),
        "high_mae": float(high_mae),
        "low_mae": float(low_mae),
        "r2_overall": float(r2_overall),
        "r2_high": float(r2_high),
        "r2_low": float(r2_low),
        "high_direction_acc": float(high_direction_correct),
        "low_direction_acc": float(low_direction_correct),
        "range_valid": float(range_valid),
        "q_loss": float(q_loss),
    }
    
    return total_loss, metrics


def compute_simple_regression_loss(predictions: mx.array, targets: mx.array) -> tuple[mx.array, dict]:
    """
    Simple regression loss without ACT (for evaluation)
    
    Args:
        predictions: Model predictions (batch_size, 2) [high, low]
        targets: Target values (batch_size, 2) [high, low]
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    
    # MSE loss
    mse_loss = mx.mean((predictions - targets) ** 2)
    
    # MAE loss
    mae_loss = mx.mean(mx.abs(predictions - targets))
    
    # Individual losses
    high_mse = mx.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    low_mse = mx.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    high_mae = mx.mean(mx.abs(predictions[:, 0] - targets[:, 0]))
    low_mae = mx.mean(mx.abs(predictions[:, 1] - targets[:, 1]))
    
    # R² score
    target_mean = mx.mean(targets, axis=0)
    ss_tot = mx.sum((targets - target_mean) ** 2, axis=0)
    ss_res = mx.sum((targets - predictions) ** 2, axis=0)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Range validity
    range_valid = mx.mean((predictions[:, 0] > predictions[:, 1]).astype(mx.float32))
    
    metrics = {
        "mse_loss": float(mse_loss),
        "mae_loss": float(mae_loss),
        "high_mse": float(high_mse),
        "low_mse": float(low_mse),
        "high_mae": float(high_mae),
        "low_mae": float(low_mae),
        "r2_high": float(r2_score[0]),
        "r2_low": float(r2_score[1]),
        "r2_overall": float(mx.mean(r2_score)),
        "range_valid": float(range_valid),
    }
    
    return mse_loss, metrics
