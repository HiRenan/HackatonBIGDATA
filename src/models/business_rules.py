#!/usr/bin/env python3
"""
BUSINESS RULES ENGINE - Hackathon Forecast Big Data 2025
Advanced Business Logic and Constraints Application System

Features:
- Inventory constraints (MOQ, capacity, integer requirements)
- Business logic (demand smoothing, promotional adjustments)
- Lifecycle rules (product phase considerations)
- Market constraints (competitive adjustments, market share)
- Operational constraints (supply chain limitations)
- Seasonal adjustments and event-based corrections
- Multi-tier business logic with priority handling
- Rule conflict resolution and optimization

The BUSINESS INTELLIGENCE layer that makes predictions practical! 🏢
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape

warnings.filterwarnings('ignore')

class RulePriority(Enum):
    """Rule priority levels"""
    CRITICAL = 1    # Must be enforced (legal, safety)
    HIGH = 2       # Business critical (inventory, capacity)
    MEDIUM = 3     # Business optimal (smoothing, efficiency)
    LOW = 4        # Business nice-to-have (preferences)

@dataclass
class BusinessRule:
    """
    Business Rule Definition
    
    Defines a single business rule with its conditions,
    actions, and metadata.
    """
    
    name: str
    description: str
    priority: RulePriority
    condition: Callable
    action: Callable
    enabled: bool = True
    category: str = "general"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class InventoryConstraints:
    """
    Inventory Constraint Manager
    
    Handles all inventory-related business constraints
    including MOQ, capacity limits, and supply chain rules.
    """
    
    def __init__(self):
        self.moq_rules = {}  # Minimum Order Quantity rules
        self.capacity_limits = {}  # Storage/handling capacity limits
        self.supplier_constraints = {}  # Supplier-specific constraints
        self.seasonal_adjustments = {}  # Seasonal capacity variations
        
    def set_moq_rules(self, 
                     product_moq_map: Dict[str, int],
                     default_moq: int = 1) -> None:
        """
        Set minimum order quantity rules
        
        Args:
            product_moq_map: Mapping of product IDs to MOQ values
            default_moq: Default MOQ for products not in map
        """
        
        self.moq_rules = product_moq_map.copy()
        self.default_moq = default_moq
        
        print(f"[MOQ] Set MOQ rules for {len(product_moq_map)} products")
    
    def set_capacity_limits(self,
                          store_capacity_map: Dict[str, Dict[str, float]],
                          constraint_types: List[str] = None) -> None:
        """
        Set capacity constraints by store and constraint type
        
        Args:
            store_capacity_map: Nested dict {store_id: {constraint_type: limit}}
            constraint_types: Types of constraints (storage, handling, display)
        """
        
        if constraint_types is None:
            constraint_types = ['storage', 'handling', 'display']
        
        self.capacity_limits = store_capacity_map.copy()
        self.constraint_types = constraint_types
        
        print(f"[CAPACITY] Set capacity limits for {len(store_capacity_map)} stores")
    
    def apply_moq_constraints(self, 
                             predictions: pd.DataFrame,
                             product_col: str = 'internal_product_id') -> pd.DataFrame:
        """
        Apply minimum order quantity constraints
        
        Args:
            predictions: DataFrame with predictions
            product_col: Column name for product ID
            
        Returns:
            Adjusted predictions DataFrame
        """
        
        if not self.moq_rules:
            return predictions
        
        adjusted_predictions = predictions.copy()
        n_adjustments = 0
        
        for idx, row in adjusted_predictions.iterrows():
            product_id = row[product_col]
            original_pred = row['prediction'] if 'prediction' in row else row.get('quantity', 0)
            
            # Get MOQ for this product
            moq = self.moq_rules.get(product_id, self.default_moq)
            
            if original_pred > 0 and original_pred < moq:
                # Adjust to MOQ
                adjusted_predictions.at[idx, 'prediction'] = moq
                adjusted_predictions.at[idx, 'moq_adjusted'] = True
                n_adjustments += 1
            elif original_pred > moq:
                # Round to nearest MOQ multiple
                moq_multiple = round(original_pred / moq) * moq
                if moq_multiple != original_pred:
                    adjusted_predictions.at[idx, 'prediction'] = moq_multiple
                    adjusted_predictions.at[idx, 'moq_adjusted'] = True
                    n_adjustments += 1
        
        print(f"[MOQ] Applied {n_adjustments} MOQ adjustments")
        
        return adjusted_predictions
    
    def apply_capacity_constraints(self,
                                 predictions: pd.DataFrame,
                                 store_col: str = 'internal_store_id',
                                 product_col: str = 'internal_product_id') -> pd.DataFrame:
        """
        Apply capacity constraints
        
        Args:
            predictions: DataFrame with predictions
            store_col: Column name for store ID
            product_col: Column name for product ID
            
        Returns:
            Adjusted predictions DataFrame
        """
        
        if not self.capacity_limits:
            return predictions
        
        adjusted_predictions = predictions.copy()
        adjusted_predictions['capacity_adjusted'] = False
        
        # Group by store and apply capacity limits
        for store_id, store_group in adjusted_predictions.groupby(store_col):
            if store_id not in self.capacity_limits:
                continue
            
            store_limits = self.capacity_limits[store_id]
            
            for constraint_type, limit in store_limits.items():
                # Calculate total predicted volume for this constraint type
                total_predicted = store_group['prediction'].sum()
                
                if total_predicted > limit:
                    # Scale down proportionally
                    scaling_factor = limit / total_predicted
                    
                    # Apply scaling to all products in this store
                    store_mask = adjusted_predictions[store_col] == store_id
                    adjusted_predictions.loc[store_mask, 'prediction'] *= scaling_factor
                    adjusted_predictions.loc[store_mask, 'capacity_adjusted'] = True
                    
                    print(f"[CAPACITY] Store {store_id}: scaled by {scaling_factor:.3f} due to {constraint_type} limit")
        
        return adjusted_predictions

class DemandSmoothingEngine:
    """
    Demand Smoothing Engine
    
    Applies business logic to smooth demand predictions
    and avoid unrealistic fluctuations.
    """
    
    def __init__(self):
        self.smoothing_parameters = {
            'max_change_rate': 0.5,      # Max 50% change week-over-week
            'trend_consistency': True,    # Maintain trend direction
            'seasonal_respect': True,     # Respect seasonal patterns
            'outlier_dampening': 0.3      # Dampen outliers by 30%
        }
        
    def configure_smoothing(self, parameters: Dict[str, Any]) -> None:
        """Configure smoothing parameters"""
        self.smoothing_parameters.update(parameters)
        print(f"[SMOOTHING] Updated parameters: {self.smoothing_parameters}")
    
    def apply_temporal_smoothing(self,
                               predictions_df: pd.DataFrame,
                               date_col: str = 'date',
                               value_col: str = 'prediction',
                               groupby_cols: List[str] = None) -> pd.DataFrame:
        """
        Apply temporal smoothing to predictions
        
        Args:
            predictions_df: DataFrame with predictions over time
            date_col: Date column name
            value_col: Value column to smooth
            groupby_cols: Columns to group by (e.g., product, store)
            
        Returns:
            Smoothed predictions DataFrame
        """
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
        
        # Ensure data is sorted by date
        df = predictions_df.sort_values([*groupby_cols, date_col]).copy()
        df['smoothed_prediction'] = df[value_col].copy()
        
        max_change_rate = self.smoothing_parameters['max_change_rate']
        
        n_adjustments = 0
        
        # Apply smoothing within each group
        for group_key, group_df in df.groupby(groupby_cols):
            if len(group_df) < 2:
                continue
            
            group_indices = group_df.index.tolist()
            group_values = group_df[value_col].values
            smoothed_values = group_values.copy()
            
            # Apply change rate limiting
            for i in range(1, len(group_values)):
                prev_value = smoothed_values[i-1]
                current_value = group_values[i]
                
                if prev_value > 0:
                    change_rate = abs(current_value - prev_value) / prev_value
                    
                    if change_rate > max_change_rate:
                        # Limit the change
                        if current_value > prev_value:
                            # Limit upward change
                            smoothed_values[i] = prev_value * (1 + max_change_rate)
                        else:
                            # Limit downward change
                            smoothed_values[i] = prev_value * (1 - max_change_rate)
                        
                        n_adjustments += 1
            
            # Update DataFrame
            df.loc[group_indices, 'smoothed_prediction'] = smoothed_values
        
        print(f"[SMOOTHING] Applied {n_adjustments} temporal smoothing adjustments")
        
        return df
    
    def apply_outlier_dampening(self,
                              predictions_df: pd.DataFrame,
                              value_col: str = 'prediction',
                              groupby_cols: List[str] = None) -> pd.DataFrame:
        """
        Dampen prediction outliers within groups
        
        Args:
            predictions_df: DataFrame with predictions
            value_col: Value column to process
            groupby_cols: Columns to group by
            
        Returns:
            Outlier-dampened DataFrame
        """
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id']
        
        df = predictions_df.copy()
        dampening_factor = self.smoothing_parameters['outlier_dampening']
        
        n_adjustments = 0
        
        for group_key, group_df in df.groupby(groupby_cols):
            if len(group_df) < 5:  # Need sufficient data for outlier detection
                continue
            
            values = group_df[value_col].values
            
            # Calculate outlier bounds using IQR method
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Find outliers and dampen them
            group_indices = group_df.index.tolist()
            
            for i, (idx, value) in enumerate(zip(group_indices, values)):
                if value < lower_bound or value > upper_bound:
                    # Calculate median for dampening
                    median_value = np.median(values)
                    
                    # Apply dampening
                    dampened_value = value + dampening_factor * (median_value - value)
                    df.at[idx, 'outlier_dampened_prediction'] = dampened_value
                    df.at[idx, 'outlier_adjusted'] = True
                    n_adjustments += 1
                else:
                    df.at[idx, 'outlier_dampened_prediction'] = value
                    df.at[idx, 'outlier_adjusted'] = False
        
        print(f"[OUTLIER] Applied {n_adjustments} outlier dampening adjustments")
        
        return df

class PromotionalAdjustments:
    """
    Promotional Adjustments Engine
    
    Handles promotional events, marketing campaigns,
    and special event adjustments to predictions.
    """
    
    def __init__(self):
        self.promotional_calendar = {}
        self.promotional_multipliers = {}
        self.event_effects = {}
        
    def set_promotional_calendar(self,
                               promotional_events: pd.DataFrame) -> None:
        """
        Set promotional calendar
        
        Args:
            promotional_events: DataFrame with columns [date, product_id, store_id, event_type, multiplier]
        """
        
        if promotional_events.empty:
            return
        
        # Convert to dictionary for fast lookup
        for _, row in promotional_events.iterrows():
            key = (row.get('date'), 
                   row.get('product_id', 'ALL'), 
                   row.get('store_id', 'ALL'))
            
            self.promotional_calendar[key] = {
                'event_type': row.get('event_type', 'promotion'),
                'multiplier': row.get('multiplier', 1.2),
                'duration_days': row.get('duration_days', 1)
            }
        
        print(f"[PROMO] Set {len(self.promotional_calendar)} promotional events")
    
    def apply_promotional_adjustments(self,
                                    predictions_df: pd.DataFrame,
                                    date_col: str = 'date',
                                    product_col: str = 'internal_product_id',
                                    store_col: str = 'internal_store_id') -> pd.DataFrame:
        """
        Apply promotional adjustments to predictions
        
        Args:
            predictions_df: DataFrame with predictions
            date_col: Date column name
            product_col: Product column name
            store_col: Store column name
            
        Returns:
            Adjusted DataFrame
        """
        
        if not self.promotional_calendar:
            return predictions_df
        
        df = predictions_df.copy()
        df['promotional_multiplier'] = 1.0
        df['promotional_adjusted'] = False
        
        n_adjustments = 0
        
        for idx, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date() if date_col in df.columns else None
            product_id = row.get(product_col, 'ALL')
            store_id = row.get(store_col, 'ALL')
            
            if date is None:
                continue
            
            # Check for promotional events
            # Priority: specific product+store > product-wide > store-wide > global
            event_keys_to_check = [
                (date, product_id, store_id),  # Specific
                (date, product_id, 'ALL'),    # Product-wide
                (date, 'ALL', store_id),      # Store-wide
                (date, 'ALL', 'ALL')          # Global
            ]
            
            applied_multiplier = 1.0
            
            for key in event_keys_to_check:
                if key in self.promotional_calendar:
                    event_info = self.promotional_calendar[key]
                    applied_multiplier *= event_info['multiplier']
                    break  # Use highest priority event
            
            if applied_multiplier != 1.0:
                df.at[idx, 'prediction'] *= applied_multiplier
                df.at[idx, 'promotional_multiplier'] = applied_multiplier
                df.at[idx, 'promotional_adjusted'] = True
                n_adjustments += 1
        
        print(f"[PROMO] Applied {n_adjustments} promotional adjustments")
        
        return df

class LifecycleRules:
    """
    Product Lifecycle Rules Engine
    
    Applies adjustments based on product lifecycle phases:
    introduction, growth, maturity, decline.
    """
    
    def __init__(self):
        self.product_lifecycles = {}
        self.lifecycle_multipliers = {
            'introduction': 0.8,  # Conservative for new products
            'growth': 1.2,        # Optimistic for growing products
            'maturity': 1.0,      # Neutral for mature products
            'decline': 0.7        # Conservative for declining products
        }
        
    def set_product_lifecycles(self,
                             lifecycle_data: pd.DataFrame) -> None:
        """
        Set product lifecycle information
        
        Args:
            lifecycle_data: DataFrame with columns [product_id, lifecycle_phase, launch_date, decline_date]
        """
        
        for _, row in lifecycle_data.iterrows():
            product_id = row['product_id']
            self.product_lifecycles[product_id] = {
                'phase': row.get('lifecycle_phase', 'maturity'),
                'launch_date': pd.to_datetime(row.get('launch_date')) if 'launch_date' in row else None,
                'decline_date': pd.to_datetime(row.get('decline_date')) if 'decline_date' in row else None
            }
        
        print(f"[LIFECYCLE] Set lifecycle data for {len(self.product_lifecycles)} products")
    
    def apply_lifecycle_adjustments(self,
                                  predictions_df: pd.DataFrame,
                                  product_col: str = 'internal_product_id',
                                  date_col: str = 'date') -> pd.DataFrame:
        """
        Apply lifecycle-based adjustments
        
        Args:
            predictions_df: DataFrame with predictions
            product_col: Product column name
            date_col: Date column name
            
        Returns:
            Adjusted DataFrame
        """
        
        if not self.product_lifecycles:
            return predictions_df
        
        df = predictions_df.copy()
        df['lifecycle_multiplier'] = 1.0
        df['lifecycle_phase'] = 'unknown'
        
        n_adjustments = 0
        
        for idx, row in df.iterrows():
            product_id = row.get(product_col)
            prediction_date = pd.to_datetime(row[date_col]) if date_col in df.columns else datetime.now()
            
            if product_id in self.product_lifecycles:
                lifecycle_info = self.product_lifecycles[product_id]
                phase = lifecycle_info['phase']
                
                # Determine current phase based on dates
                launch_date = lifecycle_info.get('launch_date')
                decline_date = lifecycle_info.get('decline_date')
                
                if launch_date and prediction_date:
                    days_since_launch = (prediction_date - launch_date).days
                    
                    if days_since_launch < 90:  # First 3 months
                        phase = 'introduction'
                    elif days_since_launch < 365:  # First year
                        phase = 'growth'
                    elif decline_date and prediction_date > decline_date:
                        phase = 'decline'
                    else:
                        phase = 'maturity'
                
                # Apply lifecycle multiplier
                multiplier = self.lifecycle_multipliers.get(phase, 1.0)
                
                df.at[idx, 'prediction'] *= multiplier
                df.at[idx, 'lifecycle_multiplier'] = multiplier
                df.at[idx, 'lifecycle_phase'] = phase
                
                if multiplier != 1.0:
                    n_adjustments += 1
        
        print(f"[LIFECYCLE] Applied {n_adjustments} lifecycle adjustments")
        
        return df

class IntegerConstraintEngine:
    """
    Integer Constraint Enforcement Engine - PHASE 5 REQUIREMENT

    Ensures predictions are rounded to whole units only
    with consideration for business context.
    """

    def __init__(self,
                 rounding_method: str = 'round',
                 min_threshold: float = 0.5):

        self.rounding_method = rounding_method  # 'round', 'ceil', 'floor', 'business'
        self.min_threshold = min_threshold  # Minimum value to consider non-zero

    def apply_integer_constraints(self,
                                 df: pd.DataFrame,
                                 prediction_col: str = 'prediction',
                                 context_cols: List[str] = None) -> pd.DataFrame:
        """
        Apply integer constraints to predictions - PHASE 5 REQUIREMENT

        Args:
            df: DataFrame with predictions
            prediction_col: Column name for predictions
            context_cols: Additional context columns for business rounding

        Returns:
            DataFrame with integer-constrained predictions
        """

        print(f"[INTEGER] Applying {self.rounding_method} integer constraints...")

        df = df.copy()
        original_predictions = df[prediction_col].copy()

        if self.rounding_method == 'round':
            # Standard rounding
            df[prediction_col] = np.round(df[prediction_col])

        elif self.rounding_method == 'ceil':
            # Always round up (conservative approach)
            df[prediction_col] = np.ceil(df[prediction_col])

        elif self.rounding_method == 'floor':
            # Always round down (aggressive approach)
            df[prediction_col] = np.floor(df[prediction_col])

        elif self.rounding_method == 'business':
            # Business-aware rounding
            df[prediction_col] = self._business_round(
                df[prediction_col],
                df,
                context_cols or []
            )

        else:
            # Default to standard rounding
            df[prediction_col] = np.round(df[prediction_col])

        # Ensure non-negative and minimum threshold
        df[prediction_col] = np.where(
            df[prediction_col] < self.min_threshold,
            0,
            df[prediction_col]
        )

        # Calculate adjustment statistics
        adjustment_factor = df[prediction_col] / (original_predictions + 1e-8)
        total_adjustment = np.sum(df[prediction_col]) / np.sum(original_predictions)

        # Add metadata
        if 'integer_adjustment_factor' not in df.columns:
            df['integer_adjustment_factor'] = adjustment_factor
        else:
            df['integer_adjustment_factor'] *= adjustment_factor

        print(f"[INTEGER] Applied constraints: mean adjustment factor = {total_adjustment:.4f}")

        return df

    def _business_round(self,
                       values: pd.Series,
                       df: pd.DataFrame,
                       context_cols: List[str]) -> pd.Series:
        """Business-aware rounding logic"""

        rounded_values = values.copy()

        # Check if we have volume context
        if 'actual' in df.columns or any('volume' in col.lower() for col in context_cols):
            # For high-volume items, use standard rounding
            # For low-volume items, be more conservative (round up)

            volume_col = None
            if 'actual' in df.columns:
                volume_col = 'actual'
            else:
                volume_cols = [col for col in context_cols if 'volume' in col.lower()]
                if volume_cols:
                    volume_col = volume_cols[0]

            if volume_col is not None:
                # Define volume thresholds
                high_volume_threshold = df[volume_col].quantile(0.7)
                low_volume_threshold = df[volume_col].quantile(0.3)

                # High volume: standard rounding
                high_volume_mask = df[volume_col] >= high_volume_threshold
                rounded_values.loc[high_volume_mask] = np.round(values.loc[high_volume_mask])

                # Medium volume: round up if fractional part > 0.3
                medium_volume_mask = (df[volume_col] >= low_volume_threshold) & (df[volume_col] < high_volume_threshold)
                medium_values = values.loc[medium_volume_mask]
                fractional_part = medium_values - np.floor(medium_values)
                rounded_values.loc[medium_volume_mask] = np.where(
                    fractional_part > 0.3,
                    np.ceil(medium_values),
                    np.floor(medium_values)
                )

                # Low volume: conservative rounding (round up if > 0.1)
                low_volume_mask = df[volume_col] < low_volume_threshold
                low_values = values.loc[low_volume_mask]
                fractional_part_low = low_values - np.floor(low_values)
                rounded_values.loc[low_volume_mask] = np.where(
                    fractional_part_low > 0.1,
                    np.ceil(low_values),
                    np.floor(low_values)
                )
            else:
                # Fallback to standard rounding
                rounded_values = np.round(values)
        else:
            # No context available, use standard rounding
            rounded_values = np.round(values)

        return rounded_values

class CompetitiveAdjustmentEngine:
    """
    Competitive Adjustment Engine - PHASE 5 REQUIREMENT

    Applies market share constraints and competitive adjustments
    to ensure predictions align with business strategy.
    """

    def __init__(self,
                 market_share_targets: Dict[str, float] = None,
                 competitive_response_factor: float = 0.1):

        self.market_share_targets = market_share_targets or {}
        self.competitive_response_factor = competitive_response_factor

        # Market constraints
        self.total_market_size = None
        self.competitor_data = {}

    def set_market_constraints(self,
                              total_market_size: float,
                              competitor_data: Dict[str, Dict] = None):
        """
        Set market size and competitor information

        Args:
            total_market_size: Total addressable market size
            competitor_data: Dictionary with competitor information
        """

        self.total_market_size = total_market_size
        self.competitor_data = competitor_data or {}

        print(f"[COMPETITIVE] Market constraints set: total market = {total_market_size:,.0f}")

    def apply_market_share_constraints(self,
                                     df: pd.DataFrame,
                                     prediction_col: str = 'prediction',
                                     product_col: str = 'internal_product_id',
                                     store_col: str = 'internal_store_id') -> pd.DataFrame:
        """
        Apply market share constraints - PHASE 5 REQUIREMENT

        Args:
            df: DataFrame with predictions
            prediction_col: Column name for predictions
            product_col: Product identifier column
            store_col: Store identifier column

        Returns:
            DataFrame with market share adjusted predictions
        """

        print("[COMPETITIVE] Applying market share constraints...")

        df = df.copy()
        original_predictions = df[prediction_col].copy()

        # Calculate current implied market share
        if self.total_market_size and self.total_market_size > 0:
            total_predicted = df[prediction_col].sum()
            current_market_share = total_predicted / self.total_market_size

            print(f"[COMPETITIVE] Current implied market share: {current_market_share:.2%}")

            # Apply global market share constraint if needed
            if 'total' in self.market_share_targets:
                target_share = self.market_share_targets['total']

                if current_market_share > target_share:
                    # Scale down to meet market share target
                    adjustment_factor = target_share / current_market_share
                    df[prediction_col] *= adjustment_factor

                    print(f"[COMPETITIVE] Global adjustment applied: factor = {adjustment_factor:.4f}")

            # Apply product-specific market share constraints
            if product_col in df.columns:
                for product_id in df[product_col].unique():
                    if str(product_id) in self.market_share_targets:
                        target_share = self.market_share_targets[str(product_id)]
                        product_mask = df[product_col] == product_id

                        product_total = df.loc[product_mask, prediction_col].sum()
                        product_current_share = product_total / self.total_market_size

                        if product_current_share > target_share:
                            product_adjustment = target_share / product_current_share
                            df.loc[product_mask, prediction_col] *= product_adjustment

                            print(f"[COMPETITIVE] Product {product_id} adjusted: "
                                  f"factor = {product_adjustment:.4f}")

        # Apply competitive response adjustments
        df = self._apply_competitive_response(df, prediction_col, product_col)

        # Calculate adjustment statistics
        adjustment_factor = df[prediction_col] / (original_predictions + 1e-8)
        total_adjustment = np.sum(df[prediction_col]) / np.sum(original_predictions)

        # Add metadata
        if 'competitive_adjustment_factor' not in df.columns:
            df['competitive_adjustment_factor'] = adjustment_factor
        else:
            df['competitive_adjustment_factor'] *= adjustment_factor

        print(f"[COMPETITIVE] Market share constraints applied: "
              f"mean adjustment factor = {total_adjustment:.4f}")

        return df

    def _apply_competitive_response(self,
                                   df: pd.DataFrame,
                                   prediction_col: str,
                                   product_col: str) -> pd.DataFrame:
        """Apply competitive response adjustments"""

        if not self.competitor_data:
            return df

        # Simulate competitive dynamics
        for product_id in df[product_col].unique():
            product_key = str(product_id)

            if product_key in self.competitor_data:
                competitor_info = self.competitor_data[product_key]
                product_mask = df[product_col] == product_id

                # Get competitor market strength
                competitor_strength = competitor_info.get('market_strength', 1.0)
                competitive_pressure = competitor_info.get('competitive_pressure', 0.5)

                # Calculate competitive adjustment
                # Higher competitor strength reduces our market potential
                competitive_adjustment = 1.0 - (
                    self.competitive_response_factor *
                    competitor_strength *
                    competitive_pressure
                )

                # Apply adjustment
                df.loc[product_mask, prediction_col] *= competitive_adjustment

        return df

    def estimate_market_response(self,
                               df: pd.DataFrame,
                               prediction_col: str = 'prediction',
                               scenarios: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Estimate market response under different competitive scenarios

        Args:
            df: Base predictions DataFrame
            prediction_col: Prediction column name
            scenarios: List of scenario names

        Returns:
            Dictionary with scenario results
        """

        if scenarios is None:
            scenarios = ['conservative', 'aggressive', 'market_leader']

        scenario_results = {}

        for scenario in scenarios:
            scenario_df = df.copy()

            if scenario == 'conservative':
                # Conservative: assume higher competitive pressure
                adjustment_factor = 0.85
            elif scenario == 'aggressive':
                # Aggressive: assume lower competitive pressure
                adjustment_factor = 1.15
            elif scenario == 'market_leader':
                # Market leader: moderate competitive advantage
                adjustment_factor = 1.05
            else:
                adjustment_factor = 1.0

            scenario_df[prediction_col] *= adjustment_factor
            scenario_df['scenario'] = scenario
            scenario_df['scenario_adjustment'] = adjustment_factor

            scenario_results[scenario] = scenario_df

        return scenario_results

class BusinessRulesOrchestrator:
    """
    Business Rules Orchestrator
    
    Main orchestrator that manages and applies all business rules
    in the correct order with conflict resolution.
    """
    
    def __init__(self):
        # Rule engines
        self.inventory_constraints = InventoryConstraints()
        self.demand_smoothing = DemandSmoothingEngine()
        self.promotional_adjustments = PromotionalAdjustments()
        self.lifecycle_rules = LifecycleRules()

        # Phase 5 requirement engines
        self.integer_constraints = IntegerConstraintEngine()
        self.competitive_adjustments = CompetitiveAdjustmentEngine()

        # Custom rules
        self.custom_rules = []
        
        # Configuration
        self.rule_execution_order = [
            'promotional_adjustments',
            'lifecycle_rules',
            'competitive_adjustments',  # Phase 5 requirement
            'demand_smoothing',
            'inventory_constraints',
            'integer_constraints'       # Phase 5 requirement - applied last
        ]
        
        # Results tracking
        self.applied_rules_log = []
        self.rule_conflicts = []
        
    def add_custom_rule(self, rule: BusinessRule) -> None:
        """Add a custom business rule"""
        self.custom_rules.append(rule)
        print(f"[CUSTOM] Added rule: {rule.name} (Priority: {rule.priority.name})")
    
    def apply_all_rules(self,
                       predictions_df: pd.DataFrame,
                       rule_config: Dict = None) -> pd.DataFrame:
        """
        Apply all business rules in the correct order
        
        Args:
            predictions_df: DataFrame with predictions
            rule_config: Configuration dictionary for rule parameters
            
        Returns:
            Adjusted predictions DataFrame
        """
        
        print("[RULES] Applying business rules orchestration...")
        
        df = predictions_df.copy()
        df['original_prediction'] = df.get('prediction', df.get('quantity', 0))
        
        rule_config = rule_config or {}
        
        # Apply rules in order
        for rule_name in self.rule_execution_order:
            print(f"\n[RULES] Applying {rule_name}...")
            
            try:
                if rule_name == 'promotional_adjustments':
                    df = self.promotional_adjustments.apply_promotional_adjustments(df)
                    
                elif rule_name == 'lifecycle_rules':
                    df = self.lifecycle_rules.apply_lifecycle_adjustments(df)
                    
                elif rule_name == 'demand_smoothing':
                    # Apply smoothing if we have temporal data
                    if 'date' in df.columns or 'transaction_date' in df.columns:
                        date_col = 'date' if 'date' in df.columns else 'transaction_date'
                        df = self.demand_smoothing.apply_temporal_smoothing(df, date_col=date_col)
                        df = self.demand_smoothing.apply_outlier_dampening(df)
                        
                        # Use smoothed predictions if available
                        if 'smoothed_prediction' in df.columns:
                            df['prediction'] = df['smoothed_prediction']
                        if 'outlier_dampened_prediction' in df.columns:
                            df['prediction'] = df['outlier_dampened_prediction']
                    
                elif rule_name == 'inventory_constraints':
                    df = self.inventory_constraints.apply_moq_constraints(df)
                    df = self.inventory_constraints.apply_capacity_constraints(df)

                elif rule_name == 'competitive_adjustments':
                    # Phase 5 requirement: Apply competitive adjustments
                    df = self.competitive_adjustments.apply_market_share_constraints(df)

                elif rule_name == 'integer_constraints':
                    # Phase 5 requirement: Apply integer constraints (applied last)
                    df = self.integer_constraints.apply_integer_constraints(df)

                # Log successful rule application
                self.applied_rules_log.append({
                    'rule_name': rule_name,
                    'timestamp': datetime.now(),
                    'status': 'success',
                    'rows_processed': len(df)
                })
                
            except Exception as e:
                print(f"[ERROR] Failed to apply {rule_name}: {e}")
                self.applied_rules_log.append({
                    'rule_name': rule_name,
                    'timestamp': datetime.now(),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Apply custom rules
        if self.custom_rules:
            df = self._apply_custom_rules(df)
        
        # Calculate final adjustments
        if 'original_prediction' in df.columns:
            df['total_adjustment_factor'] = df['prediction'] / (df['original_prediction'] + 1e-8)
            df['adjustment_magnitude'] = np.abs(df['total_adjustment_factor'] - 1.0)
        
        # Ensure final constraints
        df = self._apply_final_constraints(df)
        
        # Generate summary
        self._generate_rules_summary(df)
        
        return df
    
    def _apply_custom_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom business rules"""
        
        print(f"[CUSTOM] Applying {len(self.custom_rules)} custom rules...")
        
        # Sort custom rules by priority
        sorted_rules = sorted(self.custom_rules, key=lambda x: x.priority.value)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            try:
                # Apply rule condition and action
                mask = rule.condition(df)
                if mask.any():
                    df.loc[mask] = rule.action(df.loc[mask])
                    print(f"[CUSTOM] Applied {rule.name} to {mask.sum()} rows")
                    
            except Exception as e:
                print(f"[ERROR] Custom rule {rule.name} failed: {e}")
        
        return df
    
    def _apply_final_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final business constraints"""
        
        # Ensure non-negative predictions
        df['prediction'] = np.maximum(df['prediction'], 0)
        
        # Round to appropriate precision (e.g., whole units for countable items)
        if 'round_predictions' in df.columns:
            df['prediction'] = np.round(df['prediction'])
        
        return df
    
    def _generate_rules_summary(self, df: pd.DataFrame) -> None:
        """Generate summary of applied rules"""
        
        summary = {
            'total_predictions': len(df),
            'rules_applied': len([r for r in self.applied_rules_log if r['status'] == 'success']),
            'rules_failed': len([r for r in self.applied_rules_log if r['status'] == 'failed'])
        }
        
        # Calculate adjustment statistics
        if 'total_adjustment_factor' in df.columns:
            adjustments = df['total_adjustment_factor']
            summary.update({
                'mean_adjustment_factor': adjustments.mean(),
                'std_adjustment_factor': adjustments.std(),
                'min_adjustment_factor': adjustments.min(),
                'max_adjustment_factor': adjustments.max(),
                'predictions_increased': (adjustments > 1.0).sum(),
                'predictions_decreased': (adjustments < 1.0).sum(),
                'predictions_unchanged': (adjustments == 1.0).sum()
            })
        
        # Count specific adjustments
        adjustment_counts = {}
        for col in df.columns:
            if col.endswith('_adjusted') or col.endswith('_adjustment'):
                if df[col].dtype == bool:
                    adjustment_counts[col] = df[col].sum()
        
        summary['adjustment_counts'] = adjustment_counts
        
        print("\n[SUMMARY] Business Rules Application Summary:")
        print(f"  Total predictions processed: {summary['total_predictions']}")
        print(f"  Rules successfully applied: {summary['rules_applied']}")
        print(f"  Rules failed: {summary['rules_failed']}")
        
        if 'mean_adjustment_factor' in summary:
            print(f"  Mean adjustment factor: {summary['mean_adjustment_factor']:.3f}")
            print(f"  Predictions increased: {summary['predictions_increased']}")
            print(f"  Predictions decreased: {summary['predictions_decreased']}")
        
        if adjustment_counts:
            print("  Specific adjustments:")
            for adj_type, count in adjustment_counts.items():
                print(f"    {adj_type}: {count}")
        
        self.rules_summary = summary
    
    def configure_rules(self, config: Dict[str, Any]) -> None:
        """
        Configure business rules parameters
        
        Args:
            config: Configuration dictionary with rule parameters
        """
        
        # Configure inventory constraints
        if 'moq_rules' in config:
            self.inventory_constraints.set_moq_rules(config['moq_rules'])
        
        if 'capacity_limits' in config:
            self.inventory_constraints.set_capacity_limits(config['capacity_limits'])
        
        # Configure demand smoothing
        if 'smoothing_parameters' in config:
            self.demand_smoothing.configure_smoothing(config['smoothing_parameters'])
        
        # Configure promotional calendar
        if 'promotional_events' in config:
            self.promotional_adjustments.set_promotional_calendar(config['promotional_events'])
        
        # Configure lifecycle rules
        if 'lifecycle_data' in config:
            self.lifecycle_rules.set_product_lifecycles(config['lifecycle_data'])
        
        print("[CONFIG] Business rules configuration updated")
    
    def save_rules_config(self, output_dir: str = "../../models/trained") -> Dict[str, str]:
        """Save business rules configuration"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save orchestrator
        import pickle
        orchestrator_file = output_path / f"business_rules_orchestrator_{timestamp}.pkl"
        with open(orchestrator_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['orchestrator'] = str(orchestrator_file)
        
        # Save rules log
        if self.applied_rules_log:
            log_file = output_path / f"rules_application_log_{timestamp}.json"
            with open(log_file, 'w') as f:
                json.dump(self.applied_rules_log, f, indent=2, default=str)
            saved_files['log'] = str(log_file)
        
        # Save summary
        if hasattr(self, 'rules_summary'):
            summary_file = output_path / f"rules_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(self.rules_summary, f, indent=2, default=str)
            saved_files['summary'] = str(summary_file)
        
        print(f"[SAVE] Business rules saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Business Rules Engine"""
    
    print("🏢 BUSINESS RULES ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create synthetic prediction data
        np.random.seed(42)
        
        n_predictions = 1000
        
        # Generate synthetic data
        data = {
            'internal_product_id': np.random.choice(range(1, 101), n_predictions),
            'internal_store_id': np.random.choice(range(1, 21), n_predictions),
            'date': pd.date_range('2024-01-01', periods=n_predictions, freq='D')[:n_predictions],
            'prediction': np.random.lognormal(mean=2, sigma=1, size=n_predictions),
            'categoria': np.random.choice(['A', 'B', 'C'], n_predictions)
        }
        
        predictions_df = pd.DataFrame(data)
        predictions_df['prediction'] = np.maximum(predictions_df['prediction'], 0.1)
        
        print(f"Generated {len(predictions_df)} synthetic predictions")
        print(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        print(f"Prediction range: {predictions_df['prediction'].min():.2f} to {predictions_df['prediction'].max():.2f}")
        
        # Initialize Business Rules Orchestrator
        print("\n[DEMO] Initializing Business Rules Orchestrator...")
        
        orchestrator = BusinessRulesOrchestrator()
        
        # Configure rules
        print("\n[DEMO] Configuring business rules...")
        
        # MOQ rules: some products have minimum order quantities
        moq_rules = {i: np.random.choice([1, 5, 10, 25]) for i in range(1, 51)}
        
        # Capacity limits: stores have storage limitations
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
            'multiplier': np.random.uniform(1.1, 1.5, 12),
            'duration_days': 3
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
            'lifecycle_data': lifecycle_data,
            'smoothing_parameters': {
                'max_change_rate': 0.3,
                'outlier_dampening': 0.4
            }
        }
        
        orchestrator.configure_rules(config)
        
        # Add custom rule example
        print("\n[DEMO] Adding custom business rule...")
        
        def high_value_condition(df):
            return df['prediction'] > df['prediction'].quantile(0.9)
        
        def conservative_adjustment(df):
            df['prediction'] *= 0.9  # 10% reduction for high-value predictions
            df['high_value_adjusted'] = True
            return df
        
        custom_rule = BusinessRule(
            name="High Value Conservative Adjustment",
            description="Apply conservative 10% reduction to top 10% predictions",
            priority=RulePriority.MEDIUM,
            condition=high_value_condition,
            action=conservative_adjustment,
            category="risk_management"
        )
        
        orchestrator.add_custom_rule(custom_rule)
        
        # Apply all business rules
        print("\n[DEMO] Applying all business rules...")
        
        adjusted_predictions = orchestrator.apply_all_rules(predictions_df)
        
        # Analyze results
        print("\n[DEMO] Analyzing results...")
        
        original_mean = predictions_df['prediction'].mean()
        adjusted_mean = adjusted_predictions['prediction'].mean()
        
        print(f"Original predictions mean: {original_mean:.2f}")
        print(f"Adjusted predictions mean: {adjusted_mean:.2f}")
        print(f"Overall adjustment factor: {adjusted_mean / original_mean:.3f}")
        
        # Show adjustment distribution
        if 'total_adjustment_factor' in adjusted_predictions.columns:
            adj_factors = adjusted_predictions['total_adjustment_factor']
            print(f"Adjustment factor stats:")
            print(f"  Mean: {adj_factors.mean():.3f}")
            print(f"  Std: {adj_factors.std():.3f}")
            print(f"  Min: {adj_factors.min():.3f}")
            print(f"  Max: {adj_factors.max():.3f}")
        
        # Save results
        print("\n[DEMO] Saving business rules configuration...")
        saved_files = orchestrator.save_rules_config()
        
        # Save adjusted predictions
        output_file = Path("../../models/trained") / f"adjusted_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        adjusted_predictions.to_csv(output_file, index=False)
        saved_files['adjusted_predictions'] = str(output_file)
        
        print("\n" + "=" * 80)
        print("🎉 BUSINESS RULES ENGINE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Files saved:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        return orchestrator, adjusted_predictions
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results = main()