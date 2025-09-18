# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete hackathon forecasting solution for **Hackathon Forecast Big Data 2025** - a Brazilian retail sales forecasting competition optimized for the WMAPE metric. The project implements all 10 phases from data processing to competitive analysis.

## Key Commands

### Environment Setup
```bash
# Automated setup (recommended)
python setup_environment.py

# Manual dependency installation
pip install -r requirements-py313.txt  # Python 3.13
pip install -r requirements.txt        # Generic

# Fix NumPy/Numba compatibility issues (if needed)
pip install numpy==2.2.6 --force-reinstall
```

### Data Requirements
Data files must be placed in `data/raw/` directory:
- `transacoes.parquet` (transaction data)
- `produtos.parquet` (product catalog)
- `pdvs.parquet` (store information)

Note: The data loader automatically detects parquet files by size and content.

### Model Training

#### Complete Pipeline (Recommended)
```bash
python run_training_pipeline.py  # Trains all 3 main models sequentially
```

#### Individual Model Training
```bash
python -m src.models.prophet_seasonal    # Prophet with Brazilian retail calendar
python -m src.models.lightgbm_master     # LightGBM optimized for WMAPE
python -m src.models.advanced_ensemble   # Multi-level stacking ensemble
```

### Data Processing
```bash
python -m src.data.loaders       # Test data loading
python -m src.data.preprocessors # Data preprocessing
```

### Feature Engineering
```bash
python -m src.features.feature_pipeline # Complete feature pipeline
```

### Validation and Testing
```bash
python run_final_validation.py      # Complete system validation
python validate_phase10_final.py    # Final phase validation
```

### Submission Generation
```bash
python scripts/submissions/generate_final_submission.py
python scripts/submissions/validate_submission.py
```

### Analysis and Interpretation
```bash
python -m src.competitive.analysis.competitive_analyzer
python -m src.interpretability.visualization.business_dashboards
python -m src.competitive.presentation.presentation_strategy
```

## Architecture Overview

### Core Data Flow
1. **Data Loading** (`src/utils/data_loader.py`): Optimized parquet loading with memory management
2. **Feature Engineering** (`src/features/`): 4 specialized engines (temporal, aggregation, behavioral, business)
3. **Model Training** (`src/models/`): Prophet, LightGBM, and advanced ensemble
4. **Evaluation** (`src/evaluation/`): WMAPE-focused metrics and validation
5. **Submission** (`scripts/submissions/`): Competition-ready output generation

### Key Components

#### Models (`src/models/`)
- **ProphetSeasonal**: Brazilian retail calendar with custom seasonalities, holiday effects, payroll cycles
- **LightGBMMaster**: Primary competition model with custom WMAPE objective function and Optuna optimization
- **AdvancedEnsemble**: Multi-level stacking with dynamic weighting and meta-learning

#### Feature Engines (`src/features/`)
- **TemporalFeaturesEngine**: Time-based features (lags, rolling windows, seasonality)
- **AggregationFeaturesEngine**: Product/store aggregations and ratios
- **BehavioralFeaturesEngine**: Customer behavior patterns and trends
- **BusinessFeaturesEngine**: Domain-specific retail metrics and ABC analysis

#### Data Loading Strategy (`src/utils/data_loader.py`)
- Automatic file detection by size heuristics (largest=transactions, smallest=stores, medium=products)
- Memory-optimized chunked loading with dtype optimization
- LEFT JOIN preservation to avoid transaction loss (critical for WMAPE)
- Built-in validation to prevent data leakage

#### Competitive Analysis (`src/competitive/`)
- **CompetitiveAnalyzer**: Benchmarking against baseline models
- **DifferentiationEngine**: Identifies unique solution aspects
- **PresentationStrategy**: Optimizes pitch for hackathon judges

#### Interpretability (`src/interpretability/`)
- SHAP/LIME integration for model explanations
- Business-focused dashboards and insights
- Temporal attribution for time series understanding

## Important Technical Details

### WMAPE Optimization
- Primary competition metric: `WMAPE = sum(|actual - forecast|) / sum(|actual|) * 100`
- Custom LightGBM objective function implements WMAPE gradients
- All validation preserves transaction volume to avoid metric corruption

### Brazilian Retail Domain
- ProphetSeasonal includes Brazilian federal holidays, carnival, payroll cycles
- Business features leverage ABC analysis and retail seasonality patterns
- Calendar effects for Sunday sales impact and promotional periods

### Memory Management
- Handles 199M+ transaction records efficiently
- Automatic dtype optimization (float64→float32, object→category)
- Chunked processing with configurable memory limits
- Sample-based training for development speed

### Path Resolution
- Models use absolute paths: `Path(__file__).parent.parent.parent / "data" / "raw"`
- Avoids relative path issues when running via `python -m`
- Windows encoding compatibility (UTF-8 setup in all main scripts)

### Ensemble Architecture
- Base models: Prophet, LightGBM, Random Forest, SVR
- Meta-learner: Ridge regression with dynamic weights
- Context-aware routing by product/time segments
- Cross-validation based meta-feature generation

## Development Notes

### Adding New Models
1. Inherit from base interfaces in `src/models/`
2. Implement `fit()`, `predict()`, and `get_feature_importance()` methods
3. Add to `AdvancedEnsemble` base models list
4. Update `run_training_pipeline.py` configuration

### Debugging Data Issues
- Use `load_data_efficiently()` with small sample sizes for testing
- Check `data_loader.py` debug logs for file detection issues
- Verify parquet file structure matches expected columns

### Competition Submission
- Final submissions via `scripts/submissions/generate_final_submission.py`
- Includes risk management and timeline optimization
- Post-processing pipeline handles edge cases and validation

### Unicode/Encoding Issues
- All print statements use ASCII characters (no emojis) for Windows compatibility
- UTF-8 encoding setup in main scripts: `os.system('chcp 65001 > nul 2>&1')`
- Path handling uses `pathlib.Path` for cross-platform compatibility