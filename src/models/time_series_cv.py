#!/usr/bin/env python3
"""
ADVANCED TIME SERIES CROSS-VALIDATION - Hackathon Forecast Big Data 2025
Robust Validation Framework for Time Series Models

Features:
- Walk-Forward Validation with business logic
- Blocked Cross-Validation with embargo
- Hierarchical validation (by segment)
- Gap and purge periods to prevent leakage
- WMAPE-focused evaluation metrics
- Seasonal holdout strategies
- Business-aware splits

Essential for reliable model evaluation! ⏰
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape, retail_forecast_evaluation
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class TimeSeriesWalkForward:
    """
    Walk-Forward Cross-Validation for Time Series
    
    Business-aware implementation:
    - Fixed training window or expanding
    - Configurable forecast horizon
    - Embargo period to prevent leakage
    - Step size for validation frequency
    """
    
    def __init__(self,
                 initial_train_size: Union[int, str] = "52W",  # 52 weeks
                 forecast_horizon: Union[int, str] = "4W",    # 4 weeks
                 step_size: Union[int, str] = "1W",           # 1 week
                 embargo: Union[int, str] = "1W",             # 1 week
                 expanding_window: bool = False,
                 max_splits: int = 10):
        """
        Args:
            initial_train_size: Initial training window size
            forecast_horizon: Forecast horizon for validation
            step_size: Step size between validation periods
            embargo: Embargo period to prevent leakage
            expanding_window: If True, use expanding training window
            max_splits: Maximum number of splits to generate
        """
        
        self.initial_train_size = initial_train_size
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.embargo = embargo
        self.expanding_window = expanding_window
        self.max_splits = max_splits
    
    def _parse_period(self, period: Union[int, str], freq: str = 'D') -> int:
        """Parse period string to number of periods"""
        
        if isinstance(period, int):
            return period
        
        if isinstance(period, str):
            if period.endswith('W'):
                weeks = int(period[:-1])
                return weeks * 7 if freq == 'D' else weeks
            elif period.endswith('M'):
                months = int(period[:-1])
                return months * 30 if freq == 'D' else months
            elif period.endswith('D'):
                return int(period[:-1])
        
        raise ValueError(f"Cannot parse period: {period}")
    
    def split(self, df: pd.DataFrame, 
              date_col: str = 'transaction_date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = df_sorted[date_col]
        
        # Detect frequency
        date_diff = dates.diff().mode().iloc[0]
        if date_diff.days == 1:
            freq = 'D'
        elif date_diff.days == 7:
            freq = 'W'
        else:
            freq = 'D'  # Default
        
        # Parse periods
        initial_train = self._parse_period(self.initial_train_size, freq)
        forecast_horizon = self._parse_period(self.forecast_horizon, freq)
        step_size = self._parse_period(self.step_size, freq)
        embargo = self._parse_period(self.embargo, freq)
        
        splits = []
        n_total = len(df_sorted)
        
        # Start with initial training size
        current_train_end = initial_train
        
        split_count = 0
        while (current_train_end + embargo + forecast_horizon < n_total and 
               split_count < self.max_splits):
            
            # Training indices
            if self.expanding_window:
                train_start = 0
            else:
                train_start = max(0, current_train_end - initial_train)
            
            train_indices = np.arange(train_start, current_train_end)
            
            # Validation indices (after embargo)
            val_start = current_train_end + embargo
            val_end = min(val_start + forecast_horizon, n_total)
            val_indices = np.arange(val_start, val_end)
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                splits.append((train_indices, val_indices))
                split_count += 1
            
            # Move forward
            current_train_end += step_size
        
        return splits

class BlockedTimeSeriesCV:
    """
    Blocked Time Series Cross-Validation
    
    Creates non-overlapping blocks with gaps to simulate
    realistic forecasting scenarios with data delays.
    """
    
    def __init__(self,
                 block_size: Union[int, str] = "8W",
                 gap_size: Union[int, str] = "2W", 
                 purge_size: Union[int, str] = "1W",
                 n_blocks: int = 6):
        """
        Args:
            block_size: Size of each validation block
            gap_size: Gap between training and validation
            purge_size: Additional purge period
            n_blocks: Number of blocks to create
        """
        
        self.block_size = block_size
        self.gap_size = gap_size
        self.purge_size = purge_size
        self.n_blocks = n_blocks
    
    def _parse_period(self, period: Union[int, str], freq: str = 'D') -> int:
        """Parse period string to number of periods"""
        
        if isinstance(period, int):
            return period
        
        if isinstance(period, str):
            if period.endswith('W'):
                weeks = int(period[:-1])
                return weeks * 7 if freq == 'D' else weeks
            elif period.endswith('M'):
                months = int(period[:-1])
                return months * 30 if freq == 'D' else months
            elif period.endswith('D'):
                return int(period[:-1])
        
        raise ValueError(f"Cannot parse period: {period}")
    
    def split(self, df: pd.DataFrame,
              date_col: str = 'transaction_date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate blocked splits"""
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n_total = len(df_sorted)
        
        # Parse periods (assume daily frequency)
        block_size = self._parse_period(self.block_size, 'D')
        gap_size = self._parse_period(self.gap_size, 'D')
        purge_size = self._parse_period(self.purge_size, 'D')
        
        total_block_span = block_size + gap_size + purge_size
        available_length = n_total - total_block_span
        
        if available_length <= 0:
            raise ValueError("Dataset too small for blocked CV with given parameters")
        
        splits = []
        
        # Create evenly spaced blocks
        block_starts = np.linspace(0, available_length, 
                                 min(self.n_blocks, available_length // total_block_span),
                                 dtype=int)
        
        for block_start in block_starts:
            # Training data: all data before block (with gap and purge)
            train_end = block_start
            train_indices = np.arange(0, train_end)
            
            # Validation block
            val_start = block_start + gap_size + purge_size
            val_end = min(val_start + block_size, n_total)
            val_indices = np.arange(val_start, val_end)
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                splits.append((train_indices, val_indices))
        
        return splits

class HierarchicalTimeSeriesCV:
    """
    Hierarchical Time Series Cross-Validation
    
    Creates separate validation splits for different
    segments (categories, regions, tiers) to ensure
    robust performance across all business dimensions.
    """
    
    def __init__(self,
                 segment_col: str,
                 base_cv: BaseCrossValidator,
                 min_segment_size: int = 100):
        """
        Args:
            segment_col: Column to segment data by
            base_cv: Base cross-validator to use per segment
            min_segment_size: Minimum size for segment inclusion
        """
        
        self.segment_col = segment_col
        self.base_cv = base_cv
        self.min_segment_size = min_segment_size
    
    def split(self, df: pd.DataFrame,
              date_col: str = 'transaction_date') -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate hierarchical splits
        
        Returns:
            Dictionary mapping segment names to list of splits
        """
        
        hierarchical_splits = {}
        
        # Get segments with sufficient data
        segment_counts = df[self.segment_col].value_counts()
        valid_segments = segment_counts[segment_counts >= self.min_segment_size].index
        
        for segment in valid_segments:
            # Filter data for segment
            segment_data = df[df[self.segment_col] == segment].copy()
            
            # Generate splits for this segment
            try:
                if hasattr(self.base_cv, 'split'):
                    if isinstance(self.base_cv, (TimeSeriesWalkForward, BlockedTimeSeriesCV)):
                        splits = self.base_cv.split(segment_data, date_col)
                    else:
                        # Standard sklearn CV
                        splits = list(self.base_cv.split(segment_data))
                    
                    hierarchical_splits[segment] = splits
                    
            except Exception as e:
                print(f"[WARNING] Failed to create splits for segment {segment}: {e}")
        
        return hierarchical_splits

class TimeSeriesCVEvaluator:
    """
    Comprehensive Time Series Cross-Validation Evaluator
    
    Orchestrates different CV strategies and provides
    detailed performance analysis.
    """
    
    def __init__(self,
                 date_col: str = 'transaction_date',
                 target_col: str = 'quantity'):
        
        self.date_col = date_col
        self.target_col = target_col
        
        # Results storage
        self.cv_results = {}
        self.performance_summary = {}
        self.split_details = {}
    
    def evaluate_model(self,
                      model: Any,
                      df: pd.DataFrame,
                      cv_strategy: str = 'walk_forward',
                      cv_params: Dict = None,
                      custom_scorer: Callable = None) -> Dict:
        """
        Evaluate model using specified CV strategy
        
        Args:
            model: Trained model with fit() and predict() methods
            df: Dataset for evaluation
            cv_strategy: 'walk_forward', 'blocked', or 'hierarchical'
            cv_params: Parameters for CV strategy
            custom_scorer: Custom scoring function
            
        Returns:
            Evaluation results
        """
        
        print(f"[INFO] Evaluating model with {cv_strategy} CV...")
        
        cv_params = cv_params or {}
        
        # Create cross-validator
        if cv_strategy == 'walk_forward':
            cv = TimeSeriesWalkForward(**cv_params)
        elif cv_strategy == 'blocked':
            cv = BlockedTimeSeriesCV(**cv_params)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Generate splits
        splits = cv.split(df, self.date_col)
        
        print(f"[INFO] Generated {len(splits)} CV splits")
        
        # Evaluate each split
        split_results = []
        
        for split_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"[CV {split_idx + 1}/{len(splits)}] Evaluating split...")
            
            # Split data
            train_data = df.iloc[train_idx].copy()
            val_data = df.iloc[val_idx].copy()
            
            try:
                # Train model
                if hasattr(model, 'fit'):
                    model.fit(train_data)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(val_data)
                else:
                    raise ValueError("Model must have predict method")
                
                # Calculate metrics
                actual = val_data[self.target_col].values
                
                # Handle different prediction formats
                if isinstance(predictions, pd.DataFrame):
                    predictions = predictions.values.flatten()
                elif not isinstance(predictions, np.ndarray):
                    predictions = np.array(predictions)
                
                # Ensure same length
                min_len = min(len(actual), len(predictions))
                actual = actual[:min_len]
                predictions = predictions[:min_len]
                
                # Calculate metrics
                metrics = {
                    'wmape': wmape(actual, predictions),
                    'mae': mean_absolute_error(actual, predictions),
                    'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                    'mape': np.mean(np.abs((actual - predictions) / (np.abs(actual) + 1e-8))) * 100
                }
                
                # Custom scorer
                if custom_scorer:
                    metrics['custom_score'] = custom_scorer(actual, predictions)
                
                split_result = {
                    'split_idx': split_idx,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'train_period': (train_data[self.date_col].min(), train_data[self.date_col].max()),
                    'val_period': (val_data[self.date_col].min(), val_data[self.date_col].max()),
                    'metrics': metrics
                }
                
                split_results.append(split_result)
                
                print(f"[CV {split_idx + 1}] WMAPE: {metrics['wmape']:.4f}")
                
            except Exception as e:
                print(f"[ERROR] Split {split_idx} failed: {e}")
                split_results.append({
                    'split_idx': split_idx,
                    'error': str(e)
                })
        
        # Aggregate results
        successful_splits = [r for r in split_results if 'error' not in r]
        
        if not successful_splits:
            raise ValueError("All CV splits failed")
        
        # Calculate summary statistics
        metric_names = list(successful_splits[0]['metrics'].keys())
        summary_metrics = {}
        
        for metric in metric_names:
            values = [split['metrics'][metric] for split in successful_splits]
            summary_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        results = {
            'cv_strategy': cv_strategy,
            'n_splits': len(splits),
            'successful_splits': len(successful_splits),
            'split_results': split_results,
            'summary_metrics': summary_metrics,
            'cv_params': cv_params
        }
        
        # Store results
        self.cv_results[cv_strategy] = results
        
        print(f"[OK] CV completed: {len(successful_splits)}/{len(splits)} successful")
        print(f"[SUMMARY] WMAPE: {summary_metrics['wmape']['mean']:.4f} ± {summary_metrics['wmape']['std']:.4f}")
        
        return results
    
    def compare_strategies(self,
                          model: Any,
                          df: pd.DataFrame,
                          strategies: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple CV strategies
        
        Args:
            model: Model to evaluate
            df: Dataset
            strategies: List of CV strategies to compare
            
        Returns:
            Comparison DataFrame
        """
        
        strategies = strategies or ['walk_forward', 'blocked']
        
        comparison_results = []
        
        for strategy in strategies:
            print(f"\n[COMPARE] Evaluating {strategy} strategy...")
            
            try:
                results = self.evaluate_model(model, df, cv_strategy=strategy)
                
                # Extract key metrics
                wmape_stats = results['summary_metrics']['wmape']
                mae_stats = results['summary_metrics']['mae']
                
                comparison_results.append({
                    'strategy': strategy,
                    'wmape_mean': wmape_stats['mean'],
                    'wmape_std': wmape_stats['std'],
                    'mae_mean': mae_stats['mean'],
                    'mae_std': mae_stats['std'],
                    'n_splits': results['n_splits'],
                    'successful_splits': results['successful_splits']
                })
                
            except Exception as e:
                print(f"[ERROR] Strategy {strategy} failed: {e}")
                comparison_results.append({
                    'strategy': strategy,
                    'error': str(e)
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        print("\n[COMPARISON] Strategy Performance:")
        for _, row in comparison_df.iterrows():
            if 'error' not in row:
                print(f"  {row['strategy']:<15}: WMAPE {row['wmape_mean']:.4f} ± {row['wmape_std']:.4f}")
        
        return comparison_df
    
    def plot_cv_results(self, 
                       strategy: str = 'walk_forward',
                       metric: str = 'wmape',
                       save_path: str = None) -> None:
        """
        Plot cross-validation results
        
        Args:
            strategy: CV strategy to plot
            metric: Metric to plot
            save_path: Path to save plot
        """
        
        if strategy not in self.cv_results:
            raise ValueError(f"No results for strategy: {strategy}")
        
        results = self.cv_results[strategy]
        successful_splits = [r for r in results['split_results'] if 'error' not in r]
        
        if not successful_splits:
            print("No successful splits to plot")
            return
        
        # Extract data for plotting
        split_numbers = [r['split_idx'] + 1 for r in successful_splits]
        metric_values = [r['metrics'][metric] for r in successful_splits]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot metric values
        plt.subplot(1, 2, 1)
        plt.plot(split_numbers, metric_values, 'o-', linewidth=2, markersize=6)
        plt.axhline(y=np.mean(metric_values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(metric_values):.4f}')
        plt.fill_between(split_numbers, 
                        np.mean(metric_values) - np.std(metric_values),
                        np.mean(metric_values) + np.std(metric_values),
                        alpha=0.3, color='gray', label='±1 std')
        plt.xlabel('CV Split')
        plt.ylabel(metric.upper())
        plt.title(f'{strategy.replace("_", " ").title()} CV - {metric.upper()} Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot distribution
        plt.subplot(1, 2, 2)
        plt.hist(metric_values, bins=min(10, len(metric_values)), alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(metric_values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(metric_values):.4f}')
        plt.xlabel(metric.upper())
        plt.ylabel('Frequency')
        plt.title(f'{metric.upper()} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "../../models/cv_results") -> Dict[str, str]:
        """Save CV results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save detailed results
        results_file = output_path / f"cv_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for strategy, results in self.cv_results.items():
            serializable_results[strategy] = json.loads(
                json.dumps(results, default=str)
            )
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        saved_files['results'] = str(results_file)
        
        print(f"[SAVE] CV results saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Advanced Time Series Cross-Validation"""
    
    print("⏰ ADVANCED TIME SERIES CROSS-VALIDATION - DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load sample data
        print("Loading data for CV demonstration...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=15000,
            sample_products=300,
            enable_joins=True,
            validate_loss=True
        )
        
        # Ensure we have time series structure
        if 'transaction_date' not in trans_df.columns:
            print("[ERROR] transaction_date column not found")
            return None
        
        # Aggregate to daily level for demo
        daily_data = trans_df.groupby('transaction_date')['quantity'].sum().reset_index()
        daily_data['transaction_date'] = pd.to_datetime(daily_data['transaction_date'])
        daily_data = daily_data.sort_values('transaction_date').reset_index(drop=True)
        
        print(f"Daily aggregated data: {daily_data.shape}")
        print(f"Date range: {daily_data['transaction_date'].min()} to {daily_data['transaction_date'].max()}")
        
        # Create a simple mock model for demonstration
        class MockForecastModel:
            def fit(self, data):
                self.mean_value = data['quantity'].mean()
                return self
            
            def predict(self, data):
                # Simple seasonal naive forecast
                return np.full(len(data), self.mean_value)
        
        # Initialize CV evaluator
        cv_evaluator = TimeSeriesCVEvaluator(
            date_col='transaction_date',
            target_col='quantity'
        )
        
        # Initialize mock model
        model = MockForecastModel()
        
        # Test Walk-Forward CV
        print("\n[DEMO] Testing Walk-Forward Cross-Validation...")
        wf_results = cv_evaluator.evaluate_model(
            model,
            daily_data,
            cv_strategy='walk_forward',
            cv_params={
                'initial_train_size': '30D',
                'forecast_horizon': '7D',
                'step_size': '7D',
                'embargo': '1D',
                'max_splits': 5
            }
        )
        
        # Test Blocked CV
        print("\n[DEMO] Testing Blocked Cross-Validation...")
        blocked_results = cv_evaluator.evaluate_model(
            model,
            daily_data,
            cv_strategy='blocked',
            cv_params={
                'block_size': '14D',
                'gap_size': '3D',
                'purge_size': '1D',
                'n_blocks': 4
            }
        )
        
        # Compare strategies
        print("\n[DEMO] Comparing CV strategies...")
        comparison = cv_evaluator.compare_strategies(
            model,
            daily_data,
            strategies=['walk_forward', 'blocked']
        )
        
        # Save results
        print("\n[DEMO] Saving results...")
        saved_files = cv_evaluator.save_results()
        
        print("\n" + "=" * 80)
        print("🎉 ADVANCED TIME SERIES CV DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Results Summary:")
        for strategy, results in cv_evaluator.cv_results.items():
            wmape_stats = results['summary_metrics']['wmape']
            print(f"{strategy}: WMAPE {wmape_stats['mean']:.4f} ± {wmape_stats['std']:.4f}")
        
        print(f"Files saved: {len(saved_files)}")
        
        return cv_evaluator, wf_results, blocked_results, comparison
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    results = main()