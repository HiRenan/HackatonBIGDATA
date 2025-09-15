# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Hackathon Forecast Big Data 2025** repository - a retail forecasting competition focused on predicting sales by store (PDV) and product (SKU) using machine learning and time series techniques. The primary metric is **WMAPE (Weighted Mean Absolute Percentage Error)**.

## Architecture & Structure

```
hackathon_forecast_2025/
├── src/
│   ├── utils/data_loader.py      # Memory-efficient data loading with chunking
│   ├── evaluation/metrics.py     # WMAPE and competition metrics
│   ├── experiment_tracking/      # MLflow setup and tracking
│   └── config/                   # Configuration files
├── notebooks/
│   └── 01_eda/                   # Exploratory data analysis
├── data/
│   ├── raw/                      # Original parquet files (199M+ records)
│   ├── processed/                # Processed datasets
│   └── features/                 # Feature stores
├── models/trained/               # Saved models
├── submissions/                  # Competition submissions
└── tests/                        # Test files
```

## Common Commands

### Environment Setup
```bash
# Initial setup (run once)
python setup_environment.py

# Install dependencies
pip install -r requirements.txt

# Or with conda (preferred)
conda create -n hackathon_forecast_2025 python=3.10
conda activate hackathon_forecast_2025
conda install -c conda-forge numpy pandas pyarrow scikit-learn lightgbm
pip install -r requirements.txt
```

### Development Workflow
```bash
# Start experiment tracking
mlflow ui
# Access at: http://localhost:5000

# Run initial data exploration
cd notebooks/01_eda
python initial_data_exploration.py
```

### Data Loading
```python
from src.utils.data_loader import load_data_efficiently

# Load datasets with memory optimization AND LEFT JOINs
trans_df, prod_df, pdv_df = load_data_efficiently(
    data_path="data/raw",
    sample_transactions=500000,
    sample_products=100000,
    enable_joins=True,        # CRITICAL: Preserves ALL transactions
    validate_loss=True        # CRITICAL: Validates no data loss
)
```

### Evaluation
```python
from src.evaluation.metrics import wmape, retail_forecast_evaluation

# Primary competition metric
score = wmape(y_true, y_pred)

# Comprehensive evaluation
results = retail_forecast_evaluation(y_true_df, y_pred_df)
```

### Experiment Tracking
```python
from src.experiment_tracking.mlflow_setup import HackathonMLflowTracker

tracker = HackathonMLflowTracker()
run_id = tracker.start_run("model_name", model_type="lightgbm")
tracker.log_training_metrics({"wmape": 15.2})
tracker.log_submission(submission_df, "final_v1")
```

## Key Technical Details

### Data Characteristics
- **199+ million transaction records** across 14,419 stores
- **Parquet format** with memory-efficient loading via chunking
- **Memory optimization** essential - use OptimizedDataLoader class
- **Sample sizes**: 500k transactions, 100k products for development

### Primary Metric
- **WMAPE**: `sum(|actual - forecast|) / sum(|actual|) * 100`
- Focus on volume-weighted accuracy for retail forecasting
- Implemented in `src/evaluation/metrics.py:wmape()`

### Model Architecture
- **Baseline**: Prophet for seasonal patterns
- **Main**: LightGBM with feature engineering  
- **Advanced**: Ensemble stacking
- All models tracked via MLflow

### Memory Management
- **Chunked processing** for large datasets (100k rows/chunk)
- **Data type optimization** (categories, downcasting)
- **8GB RAM minimum**, 16GB recommended
- Configuration in `src/config/memory_config.py`

### Feature Engineering Patterns
- **Temporal features**: lags, rolling statistics, seasonality
- **Cross features**: product×store, category×region
- **Business features**: lifecycle, complementarity
- **Volume-weighted features** for WMAPE optimization

## Development Standards

### Model Development
1. Always use MLflow tracking for experiments
2. Log comprehensive metrics including WMAPE, MAPE, MAE
3. Save models with proper versioning
4. Document feature engineering in experiment logs

### Data Processing
1. **CRITICAL**: Always use LEFT JOINs (`enable_joins=True`) to preserve ALL transactions
2. Use OptimizedDataLoader for large parquet files
3. Implement memory monitoring in long-running processes
4. Sample data appropriately during development phase
5. Always validate data loading with memory constraints (`validate_loss=True`)
6. **NEVER LOSE TRANSACTIONS**: Use `validate_loss=True` to ensure no data loss

### Submission Format
```csv
store_id,product_id,date,prediction
1,100,2024-01-01,150.5
```

## Testing & Validation
- Time series cross-validation for model evaluation
- Walk-forward validation approach
- Volume-tier analysis (A/B/C products)
- Category-level performance breakdown

## Dependencies
- **Core**: pandas, pyarrow, numpy, polars, dask
- **ML**: lightgbm, prophet, scikit-learn, xgboost
- **Time Series**: statsmodels, pmdarima, sktime
- **Experiment**: mlflow, wandb
- **Performance**: psutil, memory-profiler, py-spy