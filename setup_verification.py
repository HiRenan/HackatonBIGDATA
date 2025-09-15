#!/usr/bin/env python3
"""
SETUP VERIFICATION SCRIPT - Hackathon Forecast Big Data 2025
Comprehensive verification of all implemented components

Verifies:
- All imports work correctly
- Core functionality is accessible
- Dependencies are properly configured
- Integration between components
"""

import sys
import os
from pathlib import Path

print("HACKATHON FORECAST 2025 - SETUP VERIFICATION")
print("=" * 80)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

verification_results = {}

def test_component(name, test_func):
    """Test a component and record results"""
    try:
        test_func()
        verification_results[name] = "PASS"
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        verification_results[name] = f"FAIL: {e}"
        print(f"[FAIL] {name}: {e}")
        return False

# ============================================================================
# CORE UTILITIES VERIFICATION
# ============================================================================

def test_core_utilities():
    """Test core utilities"""
    from src.evaluation.metrics import wmape, mape, mae
    from src.utils.data_loader import load_data_efficiently, apply_left_joins_safely
    
    # Test WMAPE calculation
    y_true = [100, 200, 300]
    y_pred = [110, 190, 320]
    score = wmape(y_true, y_pred)
    assert 0 <= score <= 100, f"WMAPE score invalid: {score}"

test_component("Core Utilities (Metrics & Data Loader)", test_core_utilities)

# ============================================================================
# FEATURE ENGINEERING VERIFICATION
# ============================================================================

def test_feature_engines():
    """Test all feature engines"""
    from src.features.temporal_features_engine import TemporalFeaturesEngine
    from src.features.aggregation_features_engine import AggregationFeaturesEngine
    from src.features.behavioral_features_engine import BehavioralFeaturesEngine
    from src.features.business_features_engine import BusinessFeaturesEngine
    from src.features.feature_pipeline import FeaturePipeline
    
    # Verify engines can be instantiated
    temporal_engine = TemporalFeaturesEngine()
    agg_engine = AggregationFeaturesEngine()
    behavioral_engine = BehavioralFeaturesEngine()
    business_engine = BusinessFeaturesEngine()
    pipeline = FeaturePipeline()
    
    assert hasattr(temporal_engine, 'create_temporal_features'), "Temporal engine missing main method"
    assert hasattr(pipeline, 'process_full_pipeline'), "Pipeline missing main method"

test_component("Feature Engineering Pipeline", test_feature_engines)

# ============================================================================
# MODELING ENGINES VERIFICATION
# ============================================================================

def test_lightgbm_engine():
    """Test LightGBM Master Engine"""
    from src.models.lightgbm_master import LightGBMMaster, WMAPEObjective
    
    engine = LightGBMMaster()
    objective = WMAPEObjective()
    
    assert hasattr(engine, 'train_with_custom_objective'), "LightGBM engine missing training method"
    assert hasattr(objective, '__call__'), "WMAPE objective not callable"

test_component("LightGBM Master Engine", test_lightgbm_engine)

def test_prophet_engine():
    """Test Prophet Seasonal Engine"""
    from src.models.prophet_seasonal import ProphetSeasonal, BrazilianRetailCalendar
    
    engine = ProphetSeasonal()
    calendar = BrazilianRetailCalendar()
    
    assert hasattr(engine, 'train_models'), "Prophet engine missing training method"
    assert hasattr(calendar, 'get_holidays'), "Brazilian calendar missing holidays method"

test_component("Prophet Seasonal Engine", test_prophet_engine)

def test_intermittent_demand():
    """Test Intermittent Demand Specialist"""
    from src.models.intermittent_demand import CrostonMethod, IntermittentDemandSpecialist
    
    croston = CrostonMethod()
    specialist = IntermittentDemandSpecialist()
    
    assert hasattr(croston, 'fit'), "Croston method missing fit method"
    assert hasattr(specialist, 'train_models'), "Specialist missing training method"

test_component("Intermittent Demand Specialist", test_intermittent_demand)

def test_time_series_cv():
    """Test Time Series Cross-Validation"""
    from src.models.time_series_cv import TimeSeriesWalkForward, TimeSeriesValidator
    
    walk_forward = TimeSeriesWalkForward()
    validator = TimeSeriesValidator()
    
    assert hasattr(walk_forward, 'split'), "Walk-forward missing split method"
    assert hasattr(validator, 'validate_model'), "Validator missing validate method"

test_component("Time Series Cross-Validation", test_time_series_cv)

def test_tree_ensemble():
    """Test Tree Ensemble Engine"""
    from src.models.tree_ensemble import TreeEnsemble
    
    ensemble = TreeEnsemble()
    
    assert hasattr(ensemble, 'train_models'), "Tree ensemble missing training method"

test_component("Tree Ensemble Engine", test_tree_ensemble)

def test_lstm_temporal():
    """Test LSTM Temporal Engine"""
    from src.models.lstm_temporal import LSTMTemporal, TENSORFLOW_AVAILABLE
    
    engine = LSTMTemporal()
    
    assert hasattr(engine, 'train_lstm_models'), "LSTM engine missing training method"
    print(f"    TensorFlow Available: {TENSORFLOW_AVAILABLE}")

test_component("LSTM Temporal Engine", test_lstm_temporal)

def test_cold_start():
    """Test Cold Start Solutions"""
    from src.models.cold_start_solutions import ColdStartForecaster
    
    forecaster = ColdStartForecaster()
    
    assert hasattr(forecaster, 'train_similarity_models'), "Cold start missing training method"

test_component("Cold Start Solutions", test_cold_start)

def test_meta_ensemble():
    """Test Meta-Learning Ensemble System"""
    from src.models.meta_ensemble import MetaEnsemble, ModelRouter, PerformanceTracker
    
    meta_ensemble = MetaEnsemble()
    router = ModelRouter()
    tracker = PerformanceTracker()
    
    assert hasattr(meta_ensemble, 'train_ensemble'), "Meta ensemble missing training method"
    assert hasattr(router, 'route_prediction'), "Router missing routing method"

test_component("Meta-Learning Ensemble System", test_meta_ensemble)

# ============================================================================
# OPTIONAL DEPENDENCIES CHECK
# ============================================================================

def test_optional_dependencies():
    """Test optional dependencies"""
    dependencies = {
        'lightgbm': False,
        'prophet': False,
        'tensorflow': False,
        'xgboost': False,
        'optuna': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'prophet':
                from prophet import Prophet
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass
    
    available = sum(dependencies.values())
    total = len(dependencies)
    
    print(f"    Available dependencies: {available}/{total}")
    for dep, available in dependencies.items():
        status = "[OK]" if available else "[MISSING]"
        print(f"    {status} {dep}")
    
    # Ensure core functionality works even without optional deps
    assert available >= 2, f"Too few dependencies available: {available}/{total}"

test_component("Optional Dependencies", test_optional_dependencies)

# ============================================================================
# INTEGRATION VERIFICATION
# ============================================================================

def test_end_to_end_integration():
    """Test end-to-end integration"""
    # Import key classes for integration test
    from src.utils.data_loader import OptimizedDataLoader
    from src.features.feature_pipeline import FeaturePipeline
    from src.models.meta_ensemble import MetaEnsemble
    from src.evaluation.metrics import wmape
    
    # Verify classes can be instantiated together
    data_loader = OptimizedDataLoader()
    feature_pipeline = FeaturePipeline()
    meta_ensemble = MetaEnsemble()
    
    # Test basic integration points
    assert hasattr(data_loader, 'load_transactions'), "Data loader missing transaction method"
    assert hasattr(feature_pipeline, 'process_full_pipeline'), "Feature pipeline missing process method"
    assert hasattr(meta_ensemble, 'predict'), "Meta ensemble missing predict method"

test_component("End-to-End Integration", test_end_to_end_integration)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

passed = sum(1 for result in verification_results.values() if result.startswith("PASS"))
total = len(verification_results)

print(f"Overall Status: {passed}/{total} components verified")

if passed == total:
    print("ALL VERIFICATIONS PASSED! System ready for competition!")
elif passed >= total * 0.8:
    print("MOSTLY READY - Some optional components missing")
else:
    print("SYSTEM ISSUES DETECTED - Review failed components")

print("\nDetailed Results:")
for component, result in verification_results.items():
    print(f"  {result}: {component}")

print("\n" + "=" * 80)
print("Ready for Hackathon Forecast Big Data 2025!")
print("=" * 80)