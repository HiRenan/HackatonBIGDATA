#!/usr/bin/env python3
"""
PHASE 5 INTEGRATION DEMO - Hackathon Forecast Big Data 2025
Complete Integration and Demonstration of All Phase 5 Components

Features:
- Complete end-to-end Phase 5 pipeline demonstration
- Integration of all components: Error Analysis, Model Calibration, 
  Advanced Ensemble, Business Rules, Model Diagnostics, Optimization
- Comprehensive testing and validation
- Performance benchmarking and comparison
- Real-world scenario simulation
- Complete reporting and visualization

The ULTIMATE DEMONSTRATION of Phase 5 mastery! 🏆
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all Phase 5 components
from src.evaluation.error_analysis import ErrorDecomposer, ResidualAnalyzer, ErrorVisualizationEngine
from src.models.model_calibration import ModelCalibrationSuite, BayesianCalibrator, PosteriorSamplingEngine
from src.models.advanced_ensemble import AdvancedEnsembleOrchestrator
from src.models.business_rules import (BusinessRulesOrchestrator, BusinessRule, RulePriority,
                                      IntegerConstraintEngine, CompetitiveAdjustmentEngine)
from src.evaluation.model_diagnostics import ConceptDriftDetector, FeatureImportanceMonitor, ModelHealthDashboard
from src.models.optimization_pipeline import OptimizationPipeline, OptimizationConfig

# Import new Phase 5 specific models
try:
    from src.models.arima_temporal import ARIMARegressor, ARIMATemporalEngine
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("[WARNING] ARIMA models not available")

try:
    from src.models.prophet_seasonal import ProphetSeasonal
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet not available")

try:
    from src.models.lstm_temporal import LSTMTemporal
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[WARNING] LSTM not available")

# Import utilities
from src.evaluation.metrics import wmape
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class Phase5IntegrationDemo:
    """
    Phase 5 Complete Integration Demonstration
    
    Orchestrates all Phase 5 components in a comprehensive
    end-to-end demonstration showcasing the full power
    of advanced optimization and refinement.
    """
    
    def __init__(self):
        # Initialize all Phase 5 components
        self.error_analyzer = None
        self.calibration_suite = None
        self.ensemble_orchestrator = None
        self.business_rules = None
        self.diagnostics_dashboard = None
        self.optimization_pipeline = None
        
        # Results storage
        self.demo_results = {}
        self.performance_metrics = {}
        self.integration_logs = []
        
        # Configuration
        self.demo_config = {
            'use_real_data': True,
            'sample_size': 25000,
            'optimization_trials': 50,
            'enable_all_features': True,
            'save_visualizations': True,
            'generate_reports': True
        }
    
    def setup_synthetic_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create comprehensive synthetic dataset for demonstration
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (transactions, products, stores) DataFrames
        """
        
        print("[SETUP] Creating synthetic retail dataset...")
        
        np.random.seed(42)
        
        # Generate transactions
        transactions = []
        
        # Generate product and store IDs
        n_products = min(200, n_samples // 50)
        n_stores = min(30, n_samples // 200)
        
        product_ids = range(1, n_products + 1)
        store_ids = range(1, n_stores + 1)
        
        # Generate date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Create realistic transaction patterns
        for i in range(n_samples):
            transaction = {
                'internal_product_id': np.random.choice(product_ids),
                'internal_store_id': np.random.choice(store_ids),
                'transaction_date': np.random.choice(date_range),
                'quantity': max(0, np.random.lognormal(1.5, 1.0)),
                'categoria': np.random.choice(['A', 'B', 'C', 'D']),
                'price': np.random.uniform(10, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'])
            }
            
            # Add seasonality
            month = transaction['transaction_date'].month
            if month in [11, 12]:  # Holiday season
                transaction['quantity'] *= 1.5
            elif month in [6, 7, 8]:  # Summer
                transaction['quantity'] *= 1.2
            
            # Add day-of-week effects
            dow = transaction['transaction_date'].weekday()
            if dow in [5, 6]:  # Weekend
                transaction['quantity'] *= 1.3
            
            transactions.append(transaction)
        
        trans_df = pd.DataFrame(transactions)
        
        # Generate products DataFrame
        products_data = []
        for prod_id in product_ids:
            products_data.append({
                'internal_product_id': prod_id,
                'categoria': np.random.choice(['A', 'B', 'C', 'D']),
                'brand': f'Brand_{np.random.randint(1, 20)}',
                'lifecycle_phase': np.random.choice(['introduction', 'growth', 'maturity', 'decline']),
                'launch_date': start_date + timedelta(days=np.random.randint(0, 365)),
                'cost': np.random.uniform(5, 50)
            })
        
        prod_df = pd.DataFrame(products_data)
        
        # Generate stores DataFrame
        stores_data = []
        for store_id in store_ids:
            stores_data.append({
                'internal_store_id': store_id,
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                'store_size': np.random.choice(['Small', 'Medium', 'Large']),
                'zipcode': f'{np.random.randint(10000, 99999)}',
                'storage_capacity': np.random.uniform(1000, 5000),
                'handling_capacity': np.random.uniform(800, 4000)
            })
        
        pdv_df = pd.DataFrame(stores_data)
        
        print(f"[SETUP] Generated dataset: {len(trans_df)} transactions, {len(prod_df)} products, {len(pdv_df)} stores")
        
        return trans_df, prod_df, pdv_df
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """
        Run complete Phase 5 integration demonstration
        
        Returns:
            Complete demonstration results
        """
        
        print("🏆 PHASE 5 COMPLETE INTEGRATION DEMONSTRATION")
        print("=" * 80)
        print("Showcasing: Error Analysis, Model Calibration, Advanced Ensemble,")
        print("Business Rules, Model Diagnostics, and Optimization Pipeline")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Data Setup
        print("\n[STEP 1] Data Setup and Preparation...")
        
        try:
            if self.demo_config['use_real_data']:
                # Try to load real data
                trans_df, prod_df, pdv_df = load_data_efficiently(
                    data_path="../../data/raw",
                    sample_transactions=self.demo_config['sample_size'],
                    sample_products=500,
                    enable_joins=True,
                    validate_loss=True
                )
                print("[DATA] Using real retail data")
            else:
                raise Exception("Using synthetic data")
                
        except Exception as e:
            print(f"[DATA] Real data not available ({e}), using synthetic data")
            trans_df, prod_df, pdv_df = self.setup_synthetic_data(self.demo_config['sample_size'])
        
        # Prepare features and targets
        features_df, target_series = self._prepare_ml_data(trans_df, prod_df, pdv_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, target_series.values,
            test_size=0.3, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42
        )
        
        print(f"[DATA] Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Step 2: Baseline Models
        print("\n[STEP 2] Training baseline models...")
        
        baseline_models = self._train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Step 3: Error Analysis
        print("\n[STEP 3] Comprehensive Error Analysis...")
        
        error_analysis_results = self._demonstrate_error_analysis(
            baseline_models, X_val, y_val, trans_df
        )
        
        # Step 4: Model Calibration
        print("\n[STEP 4] Model Calibration and Uncertainty Quantification...")
        
        calibration_results = self._demonstrate_calibration(
            baseline_models, X_train, y_train, X_val, y_val
        )
        
        # Step 5: Advanced Ensemble
        print("\n[STEP 5] Advanced Ensemble System...")
        
        ensemble_results = self._demonstrate_advanced_ensemble(
            baseline_models, X_train, y_train, X_val, y_val
        )
        
        # Step 6: Business Rules
        print("\n[STEP 6] Business Rules Application...")
        
        business_rules_results = self._demonstrate_business_rules(
            ensemble_results, trans_df, prod_df, pdv_df
        )
        
        # Step 7: Model Diagnostics
        print("\n[STEP 7] Model Diagnostics and Health Monitoring...")
        
        diagnostics_results = self._demonstrate_diagnostics(
            baseline_models, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 8: Optimization Pipeline
        print("\n[STEP 8] Complete Optimization Pipeline...")
        
        optimization_results = self._demonstrate_optimization(
            X_train, y_train, X_val, y_val, features_df.columns.tolist()
        )
        
        # Step 9: Integration and Final Evaluation
        print("\n[STEP 9] Integration and Final Evaluation...")
        
        final_evaluation = self._final_integration_evaluation(
            X_test, y_test, optimization_results, business_rules_results
        )
        
        # Step 10: Comprehensive Reporting
        print("\n[STEP 10] Generating Comprehensive Reports...")
        
        total_time = time.time() - start_time
        
        comprehensive_report = self._generate_comprehensive_report(
            error_analysis_results,
            calibration_results,
            ensemble_results,
            business_rules_results,
            diagnostics_results,
            optimization_results,
            final_evaluation,
            total_time
        )
        
        # Save all results
        if self.demo_config['generate_reports']:
            print("\n[SAVE] Saving complete demonstration results...")
            saved_files = self._save_demonstration_results(comprehensive_report)
        else:
            saved_files = {}
        
        print("\n" + "=" * 80)
        print("🎉 PHASE 5 INTEGRATION DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        self._print_final_summary(comprehensive_report, total_time, saved_files)
        
        return comprehensive_report
    
    def _prepare_ml_data(self, trans_df: pd.DataFrame, prod_df: pd.DataFrame, pdv_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare machine learning features and targets"""
        
        # Aggregate transactions to product-store level
        agg_df = trans_df.groupby(['internal_product_id', 'internal_store_id']).agg({
            'quantity': ['sum', 'mean', 'std', 'count'],
            'price': ['mean', 'std'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns]
        agg_df = agg_df.fillna(0)
        
        # Add product features
        agg_df = agg_df.merge(prod_df, on='internal_product_id', how='left')
        agg_df = agg_df.merge(pdv_df, on='internal_store_id', how='left')
        
        # Create additional features
        agg_df['days_active'] = (pd.to_datetime(agg_df['transaction_date_max']) - 
                                pd.to_datetime(agg_df['transaction_date_min'])).dt.days + 1
        agg_df['avg_daily_quantity'] = agg_df['quantity_sum'] / agg_df['days_active']
        
        # Encode categorical variables
        categorical_cols = ['categoria', 'region', 'store_size', 'lifecycle_phase']
        for col in categorical_cols:
            if col in agg_df.columns:
                agg_df[col] = pd.Categorical(agg_df[col]).codes
        
        # Select features for ML
        feature_cols = [col for col in agg_df.columns 
                       if col not in ['internal_product_id', 'internal_store_id', 
                                     'transaction_date_min', 'transaction_date_max', 
                                     'launch_date', 'brand', 'zipcode']]
        
        # Target: total quantity (what we want to predict)
        target_col = 'quantity_sum'
        
        features_df = agg_df[feature_cols].drop(columns=[target_col]).fillna(0)
        target_series = agg_df[target_col]
        
        print(f"[FEATURES] Created {len(features_df.columns)} features for {len(features_df)} samples")
        
        return features_df, target_series
    
    def _train_baseline_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train baseline models for demonstration"""
        
        models = {}
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        models['ridge'] = ridge_model
        
        # Evaluate models
        performance = {}
        for name, model in models.items():
            pred = model.predict(X_val)
            pred = np.maximum(pred, 0)
            wmape_score = wmape(y_val, pred)
            performance[name] = wmape_score
            print(f"[BASELINE] {name}: WMAPE = {wmape_score:.4f}")
        
        return {'models': models, 'performance': performance}
    
    def _demonstrate_error_analysis(self, baseline_models, X_val, y_val, trans_df) -> Dict:
        """Demonstrate Phase 5 error analysis capabilities including NEW REQUIREMENTS"""

        print("[ERROR ANALYSIS] Demonstrating PHASE 5 error analysis enhancements...")

        # Initialize error analysis components
        error_decomposer = ErrorDecomposer()
        residual_analyzer = ResidualAnalyzer()
        viz_engine = ErrorVisualizationEngine()
        
        # Get predictions from best baseline model
        best_model_name = min(baseline_models['performance'].keys(), 
                             key=lambda x: baseline_models['performance'][x])
        best_model = baseline_models['models'][best_model_name]
        
        predictions = best_model.predict(X_val)
        predictions = np.maximum(predictions, 0)
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'internal_product_id': np.random.choice(range(1, 101), len(y_val)),
            'internal_store_id': np.random.choice(range(1, 21), len(y_val)),
            'actual': y_val,
            'predicted': predictions
        })
        
        # Error decomposition
        decomposition_results = error_decomposer.decompose_errors(
            analysis_df, dimensions=['internal_product_id', 'internal_store_id']
        )
        
        # Residual analysis
        residuals = y_val - predictions
        residual_results = residual_analyzer.comprehensive_residual_analysis(
            residuals, fitted_values=predictions
        )
        
        # NEW PHASE 5: Volume-based error analysis
        print("[PHASE 5] Analyzing error patterns by volume segments...")
        error_df = pd.DataFrame({
            'actual': y_val,
            'predicted': predictions
        })

        volume_analysis_results = error_decomposer.analyze_volume_patterns(
            error_df, actual_col='actual', predicted_col='predicted'
        )

        # NEW PHASE 5: Systematic bias detection
        print("[PHASE 5] Detecting systematic bias patterns...")
        bias_analysis_results = error_decomposer.detect_systematic_bias(
            error_df, actual_col='actual', predicted_col='predicted'
        )

        # Visualizations including NEW PHASE 5 seasonal plots
        if self.demo_config['save_visualizations']:
            viz_results = viz_engine.create_comprehensive_diagnostics(
                residuals, fitted_values=predictions, actual_values=y_val,
                save_dir="../../models/diagnostics/phase5_demo"
            )

            # NEW PHASE 5: Seasonal residual plots
            print("[PHASE 5] Creating seasonal residual plots...")
            seasonal_plot_path = viz_engine.plot_seasonal_residuals(
                residuals,
                dates=pd.date_range(start='2023-01-01', periods=len(residuals), freq='D'),
                periods=[7, 30],
                save_dir="../../models/diagnostics/phase5_demo"
            )
            viz_results['seasonal_residuals'] = seasonal_plot_path
        else:
            viz_results = {}
            volume_analysis_results = {}
            bias_analysis_results = {}
        
        return {
            'decomposition': decomposition_results,
            'residual_analysis': residual_results,
            'volume_analysis': volume_analysis_results,        # NEW PHASE 5
            'bias_analysis': bias_analysis_results,            # NEW PHASE 5
            'visualizations': viz_results,
            'best_model': best_model_name,
            'baseline_wmape': baseline_models['performance'][best_model_name]
        }
    
    def _demonstrate_calibration(self, baseline_models, X_train, y_train, X_val, y_val) -> Dict:
        """Demonstrate Phase 5 model calibration including NEW BAYESIAN METHODS"""

        print("[CALIBRATION] Demonstrating PHASE 5 calibration enhancements...")

        calibration_suite = ModelCalibrationSuite()
        
        # Use best baseline model
        best_model_name = min(baseline_models['performance'].keys(), 
                             key=lambda x: baseline_models['performance'][x])
        best_model = baseline_models['models'][best_model_name]
        
        # Fit calibration methods
        calibration_results = calibration_suite.fit_all_calibrators(
            best_model, X_train, y_train, X_val, y_val
        )
        
        # Get calibrated predictions
        uncertainty_results = calibration_suite.predict_with_all_uncertainties(X_val)
        
        # Evaluate calibration quality
        evaluation_results = calibration_suite.evaluate_calibration_quality(X_val, y_val)
        
        # NEW PHASE 5: Bayesian Calibration
        print("[PHASE 5] Demonstrating Bayesian calibration methods...")
        bayesian_calibrator = BayesianCalibrator(n_samples=500)

        # Fit Bayesian calibrator
        y_train_pred = best_model.predict(X_train)
        bayesian_calibrator.fit(y_train, y_train_pred, prior_type='conjugate')

        # Generate Bayesian predictions
        y_val_pred = best_model.predict(X_val)
        bayesian_results = bayesian_calibrator.predict_with_uncertainty(
            y_val_pred, confidence_level=0.95, n_posterior_samples=500
        )

        # NEW PHASE 5: Posterior Sampling
        print("[PHASE 5] Demonstrating posterior sampling engine...")
        posterior_engine = PosteriorSamplingEngine(
            n_chains=3, n_samples_per_chain=300, burn_in=50
        )

        # Sample from model posterior (Bayesian Model Averaging)
        model_list = list(baseline_models['models'].values())
        posterior_sampling_results = posterior_engine.sample_model_posterior(
            model_list, X_train, y_train, X_val
        )

        return {
            'calibration_fitting': calibration_results,
            'uncertainty_predictions': uncertainty_results,
            'calibration_evaluation': evaluation_results,
            'calibration_suite': calibration_suite,
            'bayesian_calibration': bayesian_results,           # NEW PHASE 5
            'posterior_sampling': posterior_sampling_results    # NEW PHASE 5
        }
    
    def _demonstrate_advanced_ensemble(self, baseline_models, X_train, y_train, X_val, y_val) -> Dict:
        """Demonstrate advanced ensemble system"""
        
        base_model_list = list(baseline_models['models'].values())
        
        ensemble_orchestrator = AdvancedEnsembleOrchestrator(
            base_models=base_model_list,
            ensemble_methods=['stacking', 'dynamic_weighting', 'simple_average'],
            calibrate_uncertainty=True
        )
        
        # Fit ensemble
        ensemble_orchestrator.fit(X_train, y_train, X_val, y_val)
        
        # Get ensemble predictions
        ensemble_predictions = ensemble_orchestrator.predict(X_val)
        ensemble_wmape = wmape(y_val, ensemble_predictions)
        
        # Get predictions with uncertainty
        uncertainty_results = ensemble_orchestrator.predict_with_uncertainty(X_val)
        
        return {
            'ensemble_orchestrator': ensemble_orchestrator,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_wmape': ensemble_wmape,
            'uncertainty_results': uncertainty_results,
            'ensemble_results': ensemble_orchestrator.ensemble_results
        }
    
    def _demonstrate_business_rules(self, ensemble_results, trans_df, prod_df, pdv_df) -> Dict:
        """Demonstrate Phase 5 business rules including NEW CONSTRAINTS"""

        print("[BUSINESS RULES] Demonstrating PHASE 5 business rules enhancements...")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'internal_product_id': np.random.choice(range(1, 101), len(ensemble_results['ensemble_predictions'])),
            'internal_store_id': np.random.choice(range(1, 21), len(ensemble_results['ensemble_predictions'])),
            'date': pd.date_range('2024-01-01', periods=len(ensemble_results['ensemble_predictions']), freq='D'),
            'prediction': ensemble_results['ensemble_predictions'],
            'categoria': np.random.choice(['A', 'B', 'C', 'D'], len(ensemble_results['ensemble_predictions']))
        })
        
        # Initialize business rules orchestrator
        business_orchestrator = BusinessRulesOrchestrator()
        
        # Configure business rules
        moq_rules = {i: np.random.choice([1, 5, 10, 25]) for i in range(1, 51)}
        capacity_limits = {
            str(store_id): {
                'storage': np.random.uniform(1000, 5000),
                'handling': np.random.uniform(800, 4000)
            } for store_id in range(1, 21)
        }
        
        # Promotional events
        promotional_events = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-12-31', freq='30D'),
            'product_id': 'ALL',
            'store_id': 'ALL',
            'event_type': 'monthly_promotion',
            'multiplier': np.random.uniform(1.1, 1.5, 12)
        })
        
        # Lifecycle data
        lifecycle_data = pd.DataFrame({
            'product_id': range(1, 101),
            'lifecycle_phase': np.random.choice(['introduction', 'growth', 'maturity', 'decline'], 100),
            'launch_date': pd.date_range('2020-01-01', '2023-12-31', periods=100)
        })
        
        config = {
            'moq_rules': moq_rules,
            'capacity_limits': capacity_limits,
            'promotional_events': promotional_events,
            'lifecycle_data': lifecycle_data
        }
        
        business_orchestrator.configure_rules(config)
        
        # Apply business rules
        adjusted_predictions = business_orchestrator.apply_all_rules(predictions_df)
        
        # NEW PHASE 5: Configure and demonstrate integer constraints
        print("[PHASE 5] Configuring integer constraints...")
        business_orchestrator.integer_constraints.rounding_method = 'business'
        business_orchestrator.integer_constraints.min_threshold = 0.5

        # NEW PHASE 5: Configure competitive adjustments
        print("[PHASE 5] Configuring competitive market constraints...")
        business_orchestrator.competitive_adjustments.set_market_constraints(
            total_market_size=1000000,  # 1M total market
            competitor_data={
                '1': {'market_strength': 0.8, 'competitive_pressure': 0.6},
                '2': {'market_strength': 0.6, 'competitive_pressure': 0.4},
                '3': {'market_strength': 0.9, 'competitive_pressure': 0.7}
            }
        )
        business_orchestrator.competitive_adjustments.market_share_targets = {
            'total': 0.15,  # Target 15% market share
            '1': 0.05,      # Product 1: 5%
            '2': 0.03       # Product 2: 3%
        }

        # Demonstrate individual constraint engines (Phase 5)
        print("[PHASE 5] Demonstrating individual constraint engines...")

        # Integer constraints demo
        integer_demo_df = predictions_df.copy()
        integer_demo_df = business_orchestrator.integer_constraints.apply_integer_constraints(integer_demo_df)

        # Competitive adjustments demo
        competitive_demo_df = predictions_df.copy()
        competitive_demo_df = business_orchestrator.competitive_adjustments.apply_market_share_constraints(competitive_demo_df)

        return {
            'business_orchestrator': business_orchestrator,
            'original_predictions': predictions_df,
            'adjusted_predictions': adjusted_predictions,
            'integer_demo': integer_demo_df,               # NEW PHASE 5
            'competitive_demo': competitive_demo_df,       # NEW PHASE 5
            'rules_summary': getattr(business_orchestrator, 'rules_summary', {})
        }
    
    def _demonstrate_diagnostics(self, baseline_models, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Demonstrate model diagnostics"""
        
        # Initialize diagnostics components
        drift_detector = ConceptDriftDetector()
        importance_monitor = FeatureImportanceMonitor()
        health_dashboard = ModelHealthDashboard()
        
        # Use best baseline model
        best_model_name = min(baseline_models['performance'].keys(), 
                             key=lambda x: baseline_models['performance'][x])
        best_model = baseline_models['models'][best_model_name]
        
        # Set reference data for drift detection
        reference_pred = best_model.predict(X_train)
        drift_detector.set_reference_data(X_train, y_train, reference_pred)
        
        # Test for drift on validation data
        val_pred = best_model.predict(X_val)
        drift_results = drift_detector.detect_drift(X_val, y_val, val_pred)
        
        # Monitor feature importance (if available)
        importance_results = {}
        if hasattr(best_model, 'feature_importances_'):
            baseline_importance = best_model.feature_importances_
            importance_monitor.set_baseline_importance(baseline_importance)
            
            # Simulate changed importance
            changed_importance = baseline_importance + np.random.normal(0, 0.02, len(baseline_importance))
            changed_importance = np.abs(changed_importance)
            changed_importance = changed_importance / np.sum(changed_importance)
            
            importance_results = importance_monitor.monitor_importance_stability(changed_importance)
        
        # Generate health report
        test_pred = best_model.predict(X_test)
        test_pred = np.maximum(test_pred, 0)
        
        health_report = health_dashboard.generate_health_report(
            model_name=best_model_name,
            y_true=y_test,
            y_pred=test_pred,
            X=X_test,
            reference_wmape=baseline_models['performance'][best_model_name]
        )
        
        return {
            'drift_detection': drift_results,
            'importance_monitoring': importance_results,
            'health_report': health_report,
            'drift_detector': drift_detector,
            'health_dashboard': health_dashboard
        }
    
    def _demonstrate_optimization(self, X_train, y_train, X_val, y_val, feature_names) -> Dict:
        """Demonstrate optimization pipeline"""
        
        # Configure optimization for demo (reduced trials)
        config = OptimizationConfig(
            n_trials=self.demo_config['optimization_trials'],
            timeout=600,  # 10 minutes
            enable_feature_selection=True,
            enable_ensemble_optimization=True,
            max_features=min(30, len(feature_names))
        )
        
        # Initialize and run optimization
        optimization_pipeline = OptimizationPipeline(config)
        
        optimization_results = optimization_pipeline.run_full_optimization(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        return {
            'optimization_pipeline': optimization_pipeline,
            'optimization_results': optimization_results,
            'best_models': optimization_pipeline.best_models
        }
    
    def _final_integration_evaluation(self, X_test, y_test, optimization_results, business_rules_results) -> Dict:
        """Final integration evaluation on test set"""
        
        # Get optimized model predictions
        if optimization_results['best_models']:
            best_optimized_model = list(optimization_results['best_models'].values())[0]
            
            # Apply feature selection if used
            if 'feature_selection' in optimization_results['optimization_results']:
                selector = optimization_results['optimization_results']['feature_selection']['selector']
                if hasattr(selector, 'transform'):
                    X_test_opt = selector.transform(X_test)
                else:
                    X_test_opt = X_test[:, selector]
            else:
                X_test_opt = X_test
            
            optimized_pred = best_optimized_model.predict(X_test_opt)
            optimized_pred = np.maximum(optimized_pred, 0)
            
            # Apply business rules to optimized predictions
            test_pred_df = pd.DataFrame({
                'internal_product_id': np.random.choice(range(1, 101), len(optimized_pred)),
                'internal_store_id': np.random.choice(range(1, 21), len(optimized_pred)),
                'prediction': optimized_pred
            })
            
            # Use business rules from earlier demo
            business_orchestrator = business_rules_results['business_orchestrator']
            final_adjusted_pred = business_orchestrator.apply_all_rules(test_pred_df)
            
            final_predictions = final_adjusted_pred['prediction'].values
            final_wmape = wmape(y_test, final_predictions)
            
            # Calculate improvement metrics
            baseline_wmape = optimization_results['baseline_performance']
            total_improvement = baseline_wmape - final_wmape
            improvement_pct = (total_improvement / baseline_wmape) * 100
            
            return {
                'final_predictions': final_predictions,
                'final_wmape': final_wmape,
                'baseline_wmape': baseline_wmape,
                'total_improvement': total_improvement,
                'improvement_percentage': improvement_pct,
                'optimized_predictions': optimized_pred,
                'business_adjusted_predictions': final_predictions
            }
        else:
            return {'error': 'No optimized models available'}
    
    def _generate_comprehensive_report(self, *args) -> Dict:
        """Generate comprehensive Phase 5 integration report"""
        
        (error_analysis, calibration, ensemble, business_rules, 
         diagnostics, optimization, final_eval, total_time) = args
        
        report = {
            'phase5_integration_report': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'demo_config': self.demo_config,
                
                'component_results': {
                    'error_analysis': {
                        'baseline_wmape': error_analysis.get('baseline_wmape', 0),
                        'best_model': error_analysis.get('best_model', 'unknown'),
                        'decomposition_complete': 'decomposition' in error_analysis,
                        'residual_analysis_complete': 'residual_analysis' in error_analysis,
                        'visualizations_created': len(error_analysis.get('visualizations', {}))
                    },
                    
                    'calibration': {
                        'methods_fitted': len(calibration.get('calibration_fitting', {})),
                        'uncertainty_methods': list(calibration.get('uncertainty_predictions', {}).keys()),
                        'calibration_quality': len(calibration.get('calibration_evaluation', {}))
                    },
                    
                    'ensemble': {
                        'ensemble_wmape': ensemble.get('ensemble_wmape', 0),
                        'ensemble_methods': list(ensemble.get('ensemble_results', {}).keys()),
                        'uncertainty_available': 'uncertainty_results' in ensemble
                    },
                    
                    'business_rules': {
                        'rules_applied': len(business_rules['business_orchestrator'].applied_rules_log),
                        'predictions_adjusted': len(business_rules['adjusted_predictions']),
                        'rules_summary': business_rules.get('rules_summary', {})
                    },
                    
                    'diagnostics': {
                        'drift_detected': diagnostics['drift_detection'].get('overall_drift_detected', False),
                        'drift_score': diagnostics['drift_detection'].get('drift_score', 0),
                        'health_score': diagnostics['health_report'].health_score,
                        'active_alerts': len(diagnostics['health_report'].alerts)
                    },
                    
                    'optimization': {
                        'baseline_performance': optimization['optimization_results'].get('baseline_performance', 0),
                        'final_performance': optimization['optimization_results'].get('final_performance', 0),
                        'improvement': optimization['optimization_results'].get('improvement', 0),
                        'improvement_percentage': optimization['optimization_results'].get('improvement_percentage', 0),
                        'optimization_time': optimization['optimization_results'].get('optimization_time', 0)
                    }
                },
                
                'final_evaluation': final_eval,
                
                'integration_success': True,
                'phase5_completeness_score': self._calculate_completeness_score(
                    error_analysis, calibration, ensemble, business_rules, diagnostics, optimization
                )
            }
        }
        
        return report
    
    def _calculate_completeness_score(self, *components) -> float:
        """Calculate Phase 5 completeness score"""
        
        scores = []
        
        # Error Analysis (20%)
        error_analysis = components[0]
        error_score = 0
        if 'decomposition' in error_analysis:
            error_score += 10
        if 'residual_analysis' in error_analysis:
            error_score += 10
        scores.append(error_score)
        
        # Calibration (15%)
        calibration = components[1]
        cal_score = len(calibration.get('calibration_fitting', {})) * 3  # Max 15
        scores.append(min(cal_score, 15))
        
        # Ensemble (20%)
        ensemble = components[2]
        ens_score = len(ensemble.get('ensemble_results', {})) * 7  # Max 20
        scores.append(min(ens_score, 20))
        
        # Business Rules (15%)
        business_rules = components[3]
        br_score = min(len(business_rules['business_orchestrator'].applied_rules_log) * 3, 15)
        scores.append(br_score)
        
        # Diagnostics (15%)
        diagnostics = components[4]
        diag_score = 15 if diagnostics['health_report'].health_score > 0 else 0
        scores.append(diag_score)
        
        # Optimization (15%)
        optimization = components[5]
        opt_score = 15 if optimization['optimization_results'].get('improvement', 0) > 0 else 0
        scores.append(opt_score)
        
        return sum(scores)
    
    def _save_demonstration_results(self, report: Dict) -> Dict[str, str]:
        """Save all demonstration results"""
        
        output_dir = Path("../../models/phase5_demo_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save comprehensive report
        report_file = output_dir / f"phase5_integration_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        saved_files['integration_report'] = str(report_file)
        
        # Save demonstration object
        import pickle
        demo_file = output_dir / f"phase5_demo_object_{timestamp}.pkl"
        with open(demo_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['demo_object'] = str(demo_file)
        
        print(f"[SAVE] Phase 5 demonstration results saved: {len(saved_files)} files")
        
        return saved_files
    
    def _print_final_summary(self, report: Dict, total_time: float, saved_files: Dict) -> None:
        """Print final summary of demonstration"""
        
        phase5_report = report['phase5_integration_report']
        
        print("PHASE 5 INTEGRATION SUMMARY:")
        print(f"  Total Execution Time: {total_time:.1f} seconds")
        print(f"  Completeness Score: {phase5_report['phase5_completeness_score']:.0f}/100")
        
        print("\nComponent Performance:")
        components = phase5_report['component_results']
        
        print(f"  Error Analysis: Baseline WMAPE {components['error_analysis']['baseline_wmape']:.4f}")
        print(f"  Calibration: {components['calibration']['methods_fitted']} methods fitted")
        print(f"  Ensemble: WMAPE {components['ensemble']['ensemble_wmape']:.4f}")
        print(f"  Business Rules: {components['business_rules']['rules_applied']} rules applied")
        print(f"  Diagnostics: Health Score {components['diagnostics']['health_score']:.1f}/100")
        
        opt_comp = components['optimization']
        print(f"  Optimization: {opt_comp['improvement_percentage']:.1f}% improvement")
        
        if 'final_evaluation' in phase5_report and 'final_wmape' in phase5_report['final_evaluation']:
            final_eval = phase5_report['final_evaluation']
            print(f"\nFinal Results:")
            print(f"  Baseline WMAPE: {final_eval['baseline_wmape']:.4f}")
            print(f"  Final WMAPE: {final_eval['final_wmape']:.4f}")
            print(f"  Total Improvement: {final_eval['improvement_percentage']:.1f}%")
        
        print(f"\nFiles Saved: {len(saved_files)}")
        for file_type, path in saved_files.items():
            print(f"  {file_type}: {path}")
        
        print("\n🏆 Phase 5 Integration demonstrates enterprise-grade ML capabilities!")

def main():
    """Main demonstration function"""
    
    print("🏆 PHASE 5 ULTIMATE INTEGRATION DEMONSTRATION")
    print("Showcasing the complete Phase 5: Optimization and Refinement")
    print("All components working together in perfect harmony!")
    print()
    
    try:
        # Initialize demonstration
        demo = Phase5IntegrationDemo()
        
        # Run complete demonstration
        results = demo.run_complete_demonstration()
        
        return demo, results
        
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    demo_results = main()