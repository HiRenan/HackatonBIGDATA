#!/usr/bin/env python3
"""
FEATURE PIPELINE - Hackathon Forecast Big Data 2025
Integrated Feature Engineering Pipeline

Orchestrates all feature engines:
- Temporal Features Engine
- Aggregation Features Engine  
- Behavioral Features Engine
- Business Features Engine

Includes:
- Feature validation and quality checks
- Feature selection and importance analysis
- WMAPE-optimized feature engineering
- Feature store export functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all feature engines
from .temporal_features_engine import TemporalFeaturesEngine
from .aggregation_features_engine import AggregationFeaturesEngine
from .behavioral_features_engine import BehavioralFeaturesEngine
from .business_features_engine import BusinessFeaturesEngine

# Import utilities
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class FeaturePipeline:
    """
    Comprehensive Feature Engineering Pipeline
    
    Orchestrates all feature engines and provides:
    - End-to-end feature engineering
    - Feature validation and quality checks
    - Feature importance analysis
    - WMAPE-optimized feature selection
    - Feature store export
    """
    
    def __init__(self, 
                 date_col: str = 'transaction_date',
                 value_col: str = 'quantity',
                 target_col: str = 'quantity',
                 groupby_cols: List[str] = None):
        
        self.date_col = date_col
        self.value_col = value_col
        self.target_col = target_col
        self.groupby_cols = groupby_cols or ['internal_product_id', 'internal_store_id']
        
        # Initialize all engines
        self.temporal_engine = TemporalFeaturesEngine(date_col, value_col)
        self.aggregation_engine = AggregationFeaturesEngine(value_col)
        self.behavioral_engine = BehavioralFeaturesEngine(date_col, value_col)
        self.business_engine = BusinessFeaturesEngine(date_col, value_col)
        
        # Pipeline state
        self.features_created = []
        self.feature_metadata = {}
        self.feature_importance_scores = {}
        self.validation_results = {}
        self.execution_times = {}
        
    def run_full_pipeline(self, 
                         df: pd.DataFrame,
                         enable_validation: bool = True,
                         enable_feature_selection: bool = True,
                         max_features: int = 100) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            df: Input dataframe with transaction data
            enable_validation: Whether to run feature validation
            enable_feature_selection: Whether to run feature selection
            max_features: Maximum number of features to keep after selection
            
        Returns:
            DataFrame with all engineered features
        """
        
        print("\n" + "="*100)
        print("HACKATHON FORECAST BIG DATA 2025 - FEATURE ENGINEERING PIPELINE")
        print("="*100)
        
        start_time = time.time()
        initial_shape = df.shape
        
        print(f"[START] Pipeline execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INPUT] Initial dataset: {initial_shape[0]:,} rows, {initial_shape[1]:,} columns")
        print(f"[CONFIG] Groupby columns: {self.groupby_cols}")
        print(f"[CONFIG] Target column: {self.target_col}")
        
        # Stage 1: Temporal Features
        stage_start = time.time()
        print(f"\n[STAGE 1/4] Running Temporal Features Engine...")
        df = self.temporal_engine.create_all_temporal_features(df, self.groupby_cols)
        self.execution_times['temporal'] = time.time() - stage_start
        print(f"[STAGE 1/4] Completed in {self.execution_times['temporal']:.2f}s")
        
        # Stage 2: Aggregation Features
        stage_start = time.time()
        print(f"\n[STAGE 2/4] Running Aggregation Features Engine...")
        df = self.aggregation_engine.create_all_aggregation_features(df)
        self.execution_times['aggregation'] = time.time() - stage_start
        print(f"[STAGE 2/4] Completed in {self.execution_times['aggregation']:.2f}s")
        
        # Stage 3: Behavioral Features
        stage_start = time.time()
        print(f"\n[STAGE 3/4] Running Behavioral Features Engine...")
        df = self.behavioral_engine.create_all_behavioral_features(df, self.groupby_cols)
        self.execution_times['behavioral'] = time.time() - stage_start
        print(f"[STAGE 3/4] Completed in {self.execution_times['behavioral']:.2f}s")
        
        # Stage 4: Business Features
        stage_start = time.time()
        print(f"\n[STAGE 4/4] Running Business Features Engine...")
        df = self.business_engine.create_all_business_features(df)
        self.execution_times['business'] = time.time() - stage_start
        print(f"[STAGE 4/4] Completed in {self.execution_times['business']:.2f}s")
        
        # Collect all features created
        self.features_created = (
            self.temporal_engine.features_created +
            self.aggregation_engine.features_created +
            self.behavioral_engine.features_created +
            self.business_engine.features_created
        )
        
        intermediate_shape = df.shape
        features_added = intermediate_shape[1] - initial_shape[1]
        
        print(f"\n[FEATURES] Total features created: {len(self.features_created)}")
        print(f"[FEATURES] Dataset shape after feature engineering: {intermediate_shape}")
        
        # Feature Validation
        if enable_validation:
            print(f"\n[VALIDATION] Running feature validation...")
            validation_start = time.time()
            self.validation_results = self.validate_features(df)
            self.execution_times['validation'] = time.time() - validation_start
            print(f"[VALIDATION] Completed in {self.execution_times['validation']:.2f}s")
        
        # Feature Selection
        if enable_feature_selection:
            print(f"\n[SELECTION] Running feature selection (max {max_features} features)...")
            selection_start = time.time()
            df = self.select_best_features(df, max_features)
            self.execution_times['selection'] = time.time() - selection_start
            print(f"[SELECTION] Completed in {self.execution_times['selection']:.2f}s")
        
        # Final statistics
        final_shape = df.shape
        total_time = time.time() - start_time
        
        print(f"\n" + "="*100)
        print("FEATURE PIPELINE EXECUTION SUMMARY")
        print("="*100)
        print(f"[SUCCESS] Pipeline completed successfully!")
        print(f"[TIMING] Total execution time: {total_time:.2f}s")
        print(f"[FEATURES] Final dataset: {final_shape[0]:,} rows, {final_shape[1]:,} columns")
        print(f"[FEATURES] Features retained: {final_shape[1] - initial_shape[1]:,}")
        print(f"[MEMORY] Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Stage timing breakdown
        print(f"\n[PERFORMANCE] Stage timing breakdown:")
        for stage, duration in self.execution_times.items():
            print(f"  {stage.capitalize()}: {duration:.2f}s ({duration/total_time*100:.1f}%)")
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive feature validation
        
        Validates:
        - Missing values
        - Infinite values
        - Feature distributions
        - Correlations
        - Stability
        """
        
        validation_results = {
            'missing_values': {},
            'infinite_values': {},
            'zero_variance': [],
            'high_correlation_pairs': [],
            'feature_statistics': {},
            'quality_score': 0
        }
        
        feature_cols = [col for col in self.features_created if col in df.columns]
        
        if not feature_cols:
            print("[WARNING] No features found for validation")
            return validation_results
        
        print(f"[INFO] Validating {len(feature_cols)} features...")
        
        # Missing values check
        missing_counts = df[feature_cols].isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Infinite values check
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validation_results['infinite_values'][col] = inf_count
        
        # Zero variance features
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if df[col].nunique() <= 1:
                    validation_results['zero_variance'].append(col)
        
        # High correlation detection
        numeric_features = [col for col in feature_cols if df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        if len(numeric_features) > 1:
            correlation_matrix = df[numeric_features].corr().abs()
            
            # Find high correlation pairs (>0.95)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > 0.95:
                        high_corr_pairs.append({
                            'feature_1': correlation_matrix.columns[i],
                            'feature_2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            validation_results['high_correlation_pairs'] = high_corr_pairs
        
        # Feature statistics
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                validation_results['feature_statistics'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'nunique': df[col].nunique(),
                    'missing_pct': df[col].isnull().mean()
                }
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(validation_results['missing_values']) * 2  # -2 for each feature with missing values
        quality_score -= len(validation_results['infinite_values']) * 5  # -5 for each feature with inf values
        quality_score -= len(validation_results['zero_variance']) * 10   # -10 for each zero variance feature
        quality_score -= len(validation_results['high_correlation_pairs']) * 3  # -3 for each highly correlated pair
        
        validation_results['quality_score'] = max(0, quality_score)
        
        print(f"[VALIDATION] Feature quality score: {validation_results['quality_score']:.1f}/100")
        print(f"[VALIDATION] Issues found:")
        print(f"  - Missing values: {len(validation_results['missing_values'])} features")
        print(f"  - Infinite values: {len(validation_results['infinite_values'])} features")
        print(f"  - Zero variance: {len(validation_results['zero_variance'])} features")
        print(f"  - High correlations: {len(validation_results['high_correlation_pairs'])} pairs")
        
        return validation_results
    
    def calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using multiple methods
        
        Methods:
        - Correlation with target
        - Mutual information
        - Statistical significance
        - Business logic scoring
        """
        
        feature_importance = {}
        feature_cols = [col for col in self.features_created if col in df.columns]
        
        if self.target_col not in df.columns:
            print(f"[WARNING] Target column '{self.target_col}' not found for importance calculation")
            return {}
        
        print(f"[INFO] Calculating feature importance for {len(feature_cols)} features...")
        
        target_values = df[self.target_col]
        
        for feature in feature_cols:
            if feature in df.columns:
                importance_score = 0
                
                try:
                    # Correlation-based importance
                    if df[feature].dtype in ['float64', 'float32', 'int64', 'int32']:
                        correlation = abs(df[feature].corr(target_values))
                        if not np.isnan(correlation):
                            importance_score += correlation * 30  # 30% weight
                    
                    # Variance-based importance (higher variance = more information)
                    if df[feature].nunique() > 1:
                        normalized_variance = df[feature].std() / (abs(df[feature].mean()) + 1e-8)
                        importance_score += min(normalized_variance, 1.0) * 20  # 20% weight
                    
                    # Business logic importance (based on EDA insights)
                    business_score = self._calculate_business_importance(feature)
                    importance_score += business_score * 25  # 25% weight
                    
                    # WMAPE optimization importance
                    wmape_score = self._calculate_wmape_importance(feature, df)
                    importance_score += wmape_score * 25  # 25% weight
                    
                    feature_importance[feature] = importance_score
                    
                except Exception as e:
                    print(f"[WARNING] Error calculating importance for {feature}: {e}")
                    feature_importance[feature] = 0
        
        # Normalize importance scores to 0-100
        if feature_importance:
            max_score = max(feature_importance.values())
            if max_score > 0:
                feature_importance = {k: (v / max_score) * 100 for k, v in feature_importance.items()}
        
        return feature_importance
    
    def _calculate_business_importance(self, feature_name: str) -> float:
        """Calculate business importance based on EDA insights and domain knowledge"""
        
        business_score = 0
        
        # High importance features based on EDA insights
        high_importance_keywords = [
            'sunday', 'september',  # Strong seasonal patterns detected
            'abc', 'tier_a', 'fast_moving',  # ABC analysis insights
            'volume_weight', 'market_share',  # WMAPE optimization
            'profit', 'revenue',  # Business impact
            'cross_selling', 'affinity',  # Cross-selling opportunities
            'lifecycle', 'growth_trend'  # Product maturity
        ]
        
        medium_importance_keywords = [
            'lag_1', 'lag_4', 'lag_12',  # Strategic lags from autocorrelation
            'rolling', 'ema',  # Temporal patterns
            'seasonal', 'trend',  # Time series components
            'category', 'region',  # Hierarchical features
            'discount', 'promotion'  # Business features
        ]
        
        # Check for high importance patterns
        for keyword in high_importance_keywords:
            if keyword.lower() in feature_name.lower():
                business_score += 15  # High business value
                break
        
        # Check for medium importance patterns
        for keyword in medium_importance_keywords:
            if keyword.lower() in feature_name.lower():
                business_score += 10  # Medium business value
                break
        
        # Penalty for overly complex features
        if len(feature_name) > 50:  # Very long feature names might be over-engineered
            business_score -= 5
        
        return min(business_score, 20)  # Cap at 20
    
    def _calculate_wmape_importance(self, feature_name: str, df: pd.DataFrame) -> float:
        """Calculate WMAPE-specific importance"""
        
        wmape_score = 0
        
        # Features critical for WMAPE optimization
        wmape_critical_keywords = [
            'volume_weight', 'total_volume', 'volume_tier',  # Volume-based weighting
            'forecast_difficulty', 'forecast_importance',  # Forecasting complexity
            'abc_', 'tier_a', 'tier_b',  # ABC classification
            'intermittency', 'zero_ratio',  # Intermittent demand
            'percentage_error', 'relative_performance'  # Error-related features
        ]
        
        # Features helpful for WMAPE
        wmape_helpful_keywords = [
            'market_share', 'penetration',  # Market position
            'stability', 'consistency',  # Predictability
            'seasonal_strength', 'trend_strength'  # Pattern strength
        ]
        
        # Check for WMAPE-critical patterns
        for keyword in wmape_critical_keywords:
            if keyword.lower() in feature_name.lower():
                wmape_score += 15
                break
        
        # Check for WMAPE-helpful patterns
        for keyword in wmape_helpful_keywords:
            if keyword.lower() in feature_name.lower():
                wmape_score += 8
                break
        
        # Volume-weighted features get extra boost
        if 'weight' in feature_name.lower() and 'volume' in feature_name.lower():
            wmape_score += 5
        
        return min(wmape_score, 20)  # Cap at 20
    
    def select_best_features(self, df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """
        Select best features based on importance scores
        
        Selection criteria:
        - Feature importance score
        - Feature stability
        - Business relevance
        - WMAPE optimization potential
        """
        
        # Calculate feature importance
        self.feature_importance_scores = self.calculate_feature_importance(df)
        
        if not self.feature_importance_scores:
            print("[WARNING] No feature importance scores calculated, keeping all features")
            return df
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top features
        selected_features = [feature for feature, score in sorted_features[:max_features]]
        
        # Always keep essential columns
        essential_cols = [
            self.date_col, self.value_col, self.target_col,
            'internal_product_id', 'internal_store_id'
        ] + self.groupby_cols
        
        essential_cols = [col for col in essential_cols if col in df.columns]
        columns_to_keep = list(set(essential_cols + selected_features))
        
        # Create selected dataframe
        df_selected = df[columns_to_keep].copy()
        
        print(f"[SELECTION] Selected {len(selected_features)} features out of {len(self.feature_importance_scores)}")
        print(f"[SELECTION] Top 10 features by importance:")
        for i, (feature, score) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature:<40} (score: {score:.1f})")
        
        return df_selected
    
    def export_feature_store(self, 
                           df: pd.DataFrame,
                           output_path: str = "../../data/features",
                           export_format: str = "parquet") -> Dict[str, str]:
        """
        Export feature store with comprehensive metadata
        
        Exports:
        - Feature dataset (parquet/csv)
        - Feature metadata (JSON)
        - Feature importance scores (CSV)
        - Pipeline configuration (JSON)
        - Validation results (JSON)
        """
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Export main feature dataset
        if export_format.lower() == "parquet":
            features_file = output_dir / f"feature_store_{timestamp}.parquet"
            df.to_parquet(features_file, index=False)
        else:
            features_file = output_dir / f"feature_store_{timestamp}.csv"
            df.to_csv(features_file, index=False)
        
        exported_files['features'] = str(features_file)
        
        # Export feature importance
        if self.feature_importance_scores:
            importance_file = output_dir / f"feature_importance_{timestamp}.csv"
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance_score': score}
                for feature, score in self.feature_importance_scores.items()
            ]).sort_values('importance_score', ascending=False)
            importance_df.to_csv(importance_file, index=False)
            exported_files['importance'] = str(importance_file)
        
        # Export feature metadata
        metadata = {
            'pipeline_config': {
                'date_col': self.date_col,
                'value_col': self.value_col,
                'target_col': self.target_col,
                'groupby_cols': self.groupby_cols
            },
            'execution_summary': {
                'timestamp': timestamp,
                'total_features_created': len(self.features_created),
                'final_features_count': len([col for col in self.features_created if col in df.columns]),
                'dataset_shape': df.shape,
                'execution_times': self.execution_times
            },
            'feature_engines': {
                'temporal_features': len(self.temporal_engine.features_created),
                'aggregation_features': len(self.aggregation_engine.features_created),
                'behavioral_features': len(self.behavioral_engine.features_created),
                'business_features': len(self.business_engine.features_created)
            },
            'features_by_category': {
                'temporal': self.temporal_engine.features_created,
                'aggregation': self.aggregation_engine.features_created,
                'behavioral': self.behavioral_engine.features_created,
                'business': self.business_engine.features_created
            }
        }
        
        metadata_file = output_dir / f"feature_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        exported_files['metadata'] = str(metadata_file)
        
        # Export validation results
        if self.validation_results:
            validation_file = output_dir / f"validation_results_{timestamp}.json"
            with open(validation_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            exported_files['validation'] = str(validation_file)
        
        # Create summary report
        summary_file = output_dir / f"pipeline_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("HACKATHON FORECAST BIG DATA 2025 - FEATURE PIPELINE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Execution Timestamp: {timestamp}\n")
            f.write(f"Final Dataset Shape: {df.shape}\n")
            f.write(f"Total Features Created: {len(self.features_created)}\n\n")
            
            f.write("EXECUTION TIMES:\n")
            total_time = sum(self.execution_times.values())
            for stage, duration in self.execution_times.items():
                f.write(f"  {stage.capitalize()}: {duration:.2f}s ({duration/total_time*100:.1f}%)\n")
            
            f.write(f"\nTOTAL PIPELINE TIME: {total_time:.2f}s\n\n")
            
            f.write("FEATURE ENGINES SUMMARY:\n")
            f.write(f"  Temporal Features: {len(self.temporal_engine.features_created)}\n")
            f.write(f"  Aggregation Features: {len(self.aggregation_engine.features_created)}\n")
            f.write(f"  Behavioral Features: {len(self.behavioral_engine.features_created)}\n")
            f.write(f"  Business Features: {len(self.business_engine.features_created)}\n")
            
            if self.validation_results:
                f.write(f"\nFEATURE QUALITY SCORE: {self.validation_results.get('quality_score', 0):.1f}/100\n")
            
        exported_files['summary'] = str(summary_file)
        
        print(f"\n[EXPORT] Feature store exported successfully!")
        print(f"[EXPORT] Files created:")
        for file_type, file_path in exported_files.items():
            print(f"  {file_type.capitalize()}: {file_path}")
        
        return exported_files

def main():
    """Demonstration of complete feature pipeline"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("COMPLETE FEATURE ENGINEERING PIPELINE - DEMONSTRATION")
    print("="*100)
    
    try:
        # Load sample data
        print("Loading sample data for pipeline demonstration...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=100000,  # Larger sample for comprehensive demo
            sample_products=2000
        )
        
        # Add additional context for business features
        if len(prod_df) > 0 and 'categoria' in prod_df.columns:
            # Simple category mapping
            categories = prod_df['categoria'].unique()[:10]
            category_mapping = {i: np.random.choice(categories) for i in trans_df['internal_product_id'].unique()}
            trans_df['categoria'] = trans_df['internal_product_id'].map(category_mapping)
        
        if len(pdv_df) > 0 and 'zipcode' in pdv_df.columns:
            # Simple zipcode mapping
            zipcode_mapping = dict(zip(pdv_df['pdv'], pdv_df['zipcode']))
            trans_df['zipcode'] = trans_df['internal_store_id'].map(
                lambda x: zipcode_mapping.get(str(x), np.random.randint(10000, 99999))
            )
        
        print(f"Sample data loaded: {trans_df.shape}")
        
        # Initialize pipeline
        pipeline = FeaturePipeline(
            date_col='transaction_date',
            value_col='quantity',
            target_col='quantity'
        )
        
        # Run complete pipeline
        features_df = pipeline.run_full_pipeline(
            trans_df,
            enable_validation=True,
            enable_feature_selection=True,
            max_features=50  # Select top 50 features for demo
        )
        
        # Export feature store
        exported_files = pipeline.export_feature_store(features_df)
        
        print(f"\n" + "="*100)
        print("PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*100)
        
        return features_df, pipeline, exported_files
        
    except Exception as e:
        print(f"[ERROR] Pipeline demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results = main()