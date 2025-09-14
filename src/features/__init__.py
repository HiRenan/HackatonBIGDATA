#!/usr/bin/env python3
"""
Feature Engineering Package - Hackathon Forecast Big Data 2025

This package contains all feature engineering engines and the main pipeline:
- TemporalFeaturesEngine: Time-series and temporal features
- AggregationFeaturesEngine: Aggregation and cross-dimensional features  
- BehavioralFeaturesEngine: Behavioral patterns and intermittency features
- BusinessFeaturesEngine: Business logic and domain-specific features
- FeaturePipeline: Main orchestrator for all feature engines
"""

from .temporal_features_engine import TemporalFeaturesEngine
from .aggregation_features_engine import AggregationFeaturesEngine  
from .behavioral_features_engine import BehavioralFeaturesEngine
from .business_features_engine import BusinessFeaturesEngine
from .feature_pipeline import FeaturePipeline

__all__ = [
    'TemporalFeaturesEngine',
    'AggregationFeaturesEngine', 
    'BehavioralFeaturesEngine',
    'BusinessFeaturesEngine',
    'FeaturePipeline'
]

__version__ = '1.0.0'
__author__ = 'Hackathon Forecast Team'