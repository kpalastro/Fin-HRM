# Financial HRM Prediction Guide

## Overview
This guide shows you how to use the trained Financial HRM model to predict the next high and low values from your financial data.

## Quick Start

### 1. Simple Single Prediction
For a quick prediction using the last 5 days of data:

```bash
# Activate virtual environment
source .venv/bin/activate

# Make a single prediction
python simple_predict.py
```

This will:
- Use the last 5 days from your data
- Load the best trained model
- Predict the next day's high and low values
- Show comparison with current values

### 2. Batch Predictions
For multiple predictions with more control:

```bash
# Make predictions using a specific checkpoint
python predict_financial.py --checkpoint checkpoints_financial/best_model_step_280.npz

# Use only last 10 days for prediction
python predict_financial.py --checkpoint checkpoints_financial/best_model_step_280.npz --last_n_days 10

# Use longer sequence length
python predict_financial.py --checkpoint checkpoints_financial/best_model_step_280.npz --sequence_length 10

# Save predictions to file
python predict_financial.py --checkpoint checkpoints_financial/best_model_step_280.npz --output my_predictions.csv
```

## Understanding the Output

### Single Prediction Output
```
🔮 Making Single Prediction for Next High and Low Values
============================================================
📊 Loaded 80 days of financial data
📈 Using last 5 days: 2025-08-03 to 2025-08-07
✅ Model loaded successfully
🔮 Making prediction...

📊 PREDICTION RESULTS
========================================
Next Day Prediction:
  Predicted High: $24,750.32
  Predicted Low:  $24,200.15
  Predicted Range: $550.17
  Reasoning Steps Used: 4

📈 Current Values (Last Day):
  Current High:  $24,634.20
  Current Low:   $24,344.15
  Current Close: $24,596.15

📊 Predicted Changes from Current Close:
  High Change:  +154.17 (+0.63%)
  Low Change:   -396.00 (-1.61%)
```

### Batch Prediction Output
The batch prediction creates a CSV file with columns:
- `start_date`: Start of input sequence
- `end_date`: End of input sequence  
- `predicted_high`: Predicted next high value
- `predicted_low`: Predicted next low value
- `predicted_range`: Difference between high and low
- `steps_used`: Number of reasoning steps used
- `normalized_high/low`: Normalized predictions

## Model Configuration

The prediction scripts use the same model configuration as training:
- **Input Sequence**: 5 days of financial data
- **Features**: 18 financial indicators (OHLC, EMA, Volume, RSI, historical values)
- **Output**: Next day's high and low values
- **Model**: 128-dimensional with 1×1 reasoning cycles

## Available Checkpoints

After training, you'll have several checkpoints available:
- `checkpoints_financial/best_model_step_XXX.npz` - Best performing model
- `checkpoints_financial/checkpoint_step_XXX.npz` - Regular checkpoints
- `checkpoints_financial/final_model.npz` - Final trained model

## Customization

### Different Sequence Lengths
```bash
# Use 10 days of history for prediction
python predict_financial.py --sequence_length 10 --checkpoint your_model.npz
```

### Different Model Sizes
```bash
# Use larger model (if you trained one)
python predict_financial.py --d_model 256 --H_cycles 2 --L_cycles 2 --checkpoint your_model.npz
```

### Batch Size
```bash
# Process more samples at once
python predict_financial.py --batch_size 16 --checkpoint your_model.npz
```

## Interpreting Results

### Prediction Quality Indicators
1. **R² Score**: Higher values (closer to 1.0) indicate better fit
2. **Range Validity**: Predicted high should be > predicted low
3. **Reasoning Steps**: More steps may indicate more complex reasoning
4. **Direction Accuracy**: Whether predictions follow market trends

### Example Analysis
```python
# Load prediction results
import pandas as pd
results = pd.read_csv('predictions.csv')

# Analyze prediction quality
print(f"Average predicted range: ${results['predicted_range'].mean():.2f}")
print(f"Range validity: {(results['predicted_high'] > results['predicted_low']).mean()*100:.1f}%")
print(f"Average reasoning steps: {results['steps_used'].mean():.1f}")
```

## Troubleshooting

### Common Issues

1. **Model not found**: Make sure you have trained a model first
   ```bash
   python train_financial.py --max_samples 50 --max_epochs 100
   ```

2. **Data format error**: Ensure your CSV has the required columns
   - Check that `data1.csv` has all 18 feature columns
   - Verify date format is consistent

3. **Memory issues**: Reduce batch size
   ```bash
   python predict_financial.py --batch_size 4 --checkpoint your_model.npz
   ```

4. **Sequence length mismatch**: Use the same sequence length as training
   ```bash
   python predict_financial.py --sequence_length 5 --checkpoint your_model.npz
   ```

## Advanced Usage

### Custom Data
To use your own financial data:
1. Ensure it has the same 18 columns as `data1.csv`
2. Update the `data_path` parameter
3. Make sure dates are in the same format

### Integration
You can integrate the prediction into your own scripts:

```python
from models.financial_hrm import FinancialHierarchicalReasoningModel
from financial_data_loader import load_financial_data, denormalize_predictions

# Load model
model = FinancialHierarchicalReasoningModel(...)
model.load_weights('your_checkpoint.npz')
model.eval()

# Make prediction
# ... (see simple_predict.py for full example)
```

## Next Steps

1. **Improve Accuracy**: Train with more data and longer sequences
2. **Feature Engineering**: Add more technical indicators
3. **Ensemble Methods**: Combine multiple model predictions
4. **Real-time Updates**: Integrate with live market data feeds
5. **Risk Management**: Add confidence intervals and uncertainty estimates
