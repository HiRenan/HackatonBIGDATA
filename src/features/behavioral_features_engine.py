#!/usr/bin/env python3
"""
BEHAVIORAL FEATURES ENGINE - Hackathon Forecast Big Data 2025
Advanced Behavioral Pattern Analysis for Intermittent Demand

Features baseadas nos insights da EDA:
- Intermittency patterns (slow-moving = 25% produtos, 0.1% volume)
- Reorder cycles and purchase frequency
- Product lifecycle stages (growth/mature/decline)
- Market dynamics and competitive behavior
- Cross-selling and cannibalization patterns

Otimizado para WMAPE: Features especiais para intermittent demand
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings('ignore')

class BehavioralFeaturesEngine:
    """
    Advanced Behavioral Features Engine
    
    Capabilities:
    - Intermittency analysis and zero-demand patterns
    - Product lifecycle detection and classification
    - Purchase frequency and reorder cycle analysis
    - Market dynamics and competitive features
    - Cross-selling strength and cannibalization risk
    - Customer behavior proxies through sales patterns
    """
    
    def __init__(self, date_col: str = 'transaction_date', value_col: str = 'quantity'):
        self.date_col = date_col
        self.value_col = value_col
        self.features_created = []
        self.feature_metadata = {}
        
        # Behavioral analysis parameters
        self.zero_threshold = 1e-8  # Threshold for considering a value as zero
        self.lifecycle_windows = [4, 8, 12, 26]  # weeks for lifecycle analysis
        self.reorder_max_gap = 14  # days for reorder cycle analysis
        
    def create_intermittency_features(self, df: pd.DataFrame,
                                    groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create comprehensive intermittency analysis features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print("[INFO] Creating intermittency analysis features...")
        
        intermittency_features = []
        
        # Basic intermittency indicators
        df['is_zero_demand'] = (df[self.value_col] <= self.zero_threshold).astype(int)
        df['is_positive_demand'] = (df[self.value_col] > self.zero_threshold).astype(int)
        intermittency_features.extend(['is_zero_demand', 'is_positive_demand'])
        
        # Sort by groups and date for sequential analysis
        df_sorted = df.sort_values(groupby_cols + [self.date_col])
        
        # Zero weeks ratio and patterns
        zero_ratios = df_sorted.groupby(groupby_cols)['is_zero_demand'].agg([
            'mean', 'sum', 'count'
        ]).reset_index()
        zero_ratios.columns = groupby_cols + ['zero_weeks_ratio', 'total_zero_weeks', 'total_weeks']
        
        # Non-zero weeks statistics
        zero_ratios['nonzero_weeks_ratio'] = 1 - zero_ratios['zero_weeks_ratio']
        zero_ratios['intermittency_score'] = zero_ratios['zero_weeks_ratio']  # Higher = more intermittent
        
        intermittency_features.extend(['zero_weeks_ratio', 'nonzero_weeks_ratio', 'intermittency_score'])
        
        # Consecutive zeros analysis
        def calculate_consecutive_zeros(group):
            """Calculate consecutive zero patterns"""
            zeros = group['is_zero_demand'].values
            
            if len(zeros) == 0:
                return pd.Series({
                    'max_consecutive_zeros': 0,
                    'avg_consecutive_zeros': 0,
                    'zero_streaks_count': 0
                })
            
            # Find consecutive zero streaks
            streaks = []
            current_streak = 0
            
            for is_zero in zeros:
                if is_zero:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                        current_streak = 0
            
            # Add final streak if ends with zeros
            if current_streak > 0:
                streaks.append(current_streak)
            
            if len(streaks) == 0:
                return pd.Series({
                    'max_consecutive_zeros': 0,
                    'avg_consecutive_zeros': 0,
                    'zero_streaks_count': 0
                })
            
            return pd.Series({
                'max_consecutive_zeros': max(streaks),
                'avg_consecutive_zeros': np.mean(streaks),
                'zero_streaks_count': len(streaks)
            })
        
        consecutive_stats = df_sorted.groupby(groupby_cols).apply(calculate_consecutive_zeros).reset_index()
        intermittency_features.extend(['max_consecutive_zeros', 'avg_consecutive_zeros', 'zero_streaks_count'])
        
        # Purchase frequency analysis
        nonzero_data = df_sorted[df_sorted['is_positive_demand'] == 1].copy()
        
        if len(nonzero_data) > 0:
            # Time between purchases
            nonzero_data['prev_purchase_date'] = nonzero_data.groupby(groupby_cols)[self.date_col].shift(1)
            nonzero_data['days_since_last_purchase'] = (
                nonzero_data[self.date_col] - nonzero_data['prev_purchase_date']
            ).dt.days
            
            purchase_frequency = nonzero_data.groupby(groupby_cols)['days_since_last_purchase'].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).reset_index()
            
            purchase_frequency.columns = groupby_cols + [
                'avg_days_between_purchases', 'median_days_between_purchases',
                'std_days_between_purchases', 'min_days_between_purchases',
                'max_days_between_purchases'
            ]
            
            # Purchase regularity (inverse of coefficient of variation)
            purchase_frequency['purchase_regularity'] = (
                purchase_frequency['avg_days_between_purchases'] / 
                (purchase_frequency['std_days_between_purchases'] + 1e-8)
            )
            
            intermittency_features.extend([
                'avg_days_between_purchases', 'median_days_between_purchases',
                'std_days_between_purchases', 'purchase_regularity'
            ])
        else:
            purchase_frequency = pd.DataFrame(columns=groupby_cols + [
                'avg_days_between_purchases', 'median_days_between_purchases',
                'std_days_between_purchases', 'purchase_regularity'
            ])
        
        # Demand burst analysis
        df_sorted['demand_above_median'] = (
            df_sorted[self.value_col] > 
            df_sorted.groupby(groupby_cols)[self.value_col].transform('median')
        ).astype(int)
        
        burst_stats = df_sorted.groupby(groupby_cols)['demand_above_median'].agg([
            'mean', 'sum'
        ]).reset_index()
        burst_stats.columns = groupby_cols + ['burst_frequency', 'total_burst_periods']
        intermittency_features.extend(['burst_frequency', 'total_burst_periods'])
        
        # Merge all intermittency features
        intermittency_all = zero_ratios.merge(consecutive_stats, on=groupby_cols, how='outer')
        if len(purchase_frequency) > 0:
            intermittency_all = intermittency_all.merge(purchase_frequency, on=groupby_cols, how='outer')
        intermittency_all = intermittency_all.merge(burst_stats, on=groupby_cols, how='outer')
        
        # Fill NaN values
        for col in intermittency_all.columns:
            if col not in groupby_cols:
                intermittency_all[col] = intermittency_all[col].fillna(0)
        
        # Merge back to main dataframe
        df = df.merge(intermittency_all, on=groupby_cols, how='left')
        
        self.features_created.extend(intermittency_features)
        print(f"[OK] Created {len(intermittency_features)} intermittency features")
        
        return df
    
    def create_lifecycle_features(self, df: pd.DataFrame,
                                groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create product/combination lifecycle analysis features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        if self.date_col not in df.columns:
            print("[WARNING] Date column not found, skipping lifecycle features")
            return df
            
        print("[INFO] Creating lifecycle analysis features...")
        
        lifecycle_features = []
        
        # Sort by groups and date
        df_sorted = df.sort_values(groupby_cols + [self.date_col])
        
        # Basic lifecycle metrics
        lifecycle_basic = df_sorted.groupby(groupby_cols)[self.date_col].agg([
            'min', 'max', 'count'
        ]).reset_index()
        lifecycle_basic.columns = groupby_cols + ['first_sale_date', 'last_sale_date', 'total_periods']
        
        # Calculate age metrics
        reference_date = df_sorted[self.date_col].max()
        lifecycle_basic['days_since_first_sale'] = (reference_date - lifecycle_basic['first_sale_date']).dt.days
        lifecycle_basic['days_since_last_sale'] = (reference_date - lifecycle_basic['last_sale_date']).dt.days
        lifecycle_basic['product_lifespan_days'] = (lifecycle_basic['last_sale_date'] - lifecycle_basic['first_sale_date']).dt.days
        
        lifecycle_features.extend([
            'days_since_first_sale', 'days_since_last_sale', 'product_lifespan_days', 'total_periods'
        ])
        
        # Lifecycle stage classification
        def classify_lifecycle_stage(row):
            """Classify product lifecycle stage based on patterns"""
            lifespan = row['product_lifespan_days']
            days_since_last = row['days_since_last_sale']
            total_periods = row['total_periods']
            
            if lifespan < 30:  # Less than month
                return 'New'
            elif days_since_last > 60:  # No sales for 2 months
                return 'Declining'
            elif total_periods >= 20:  # Regular presence
                return 'Mature'
            else:
                return 'Growing'
        
        lifecycle_basic['lifecycle_stage'] = lifecycle_basic.apply(classify_lifecycle_stage, axis=1)
        
        # Growth trajectory analysis
        growth_stats = []
        
        for window in self.lifecycle_windows:
            # Calculate trend over different windows
            def calculate_trend(group, window_size):
                """Calculate linear trend over specified window"""
                if len(group) < max(3, window_size // 4):
                    return 0
                
                recent_data = group.tail(window_size)
                if len(recent_data) < 3:
                    return 0
                
                x = np.arange(len(recent_data))
                y = recent_data[self.value_col].values
                
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    return slope
                except:
                    return 0
            
            trend_col = f'growth_trend_{window}w'
            trends = df_sorted.groupby(groupby_cols).apply(
                lambda x: calculate_trend(x, window)
            ).reset_index()
            trends.columns = groupby_cols + [trend_col]
            
            lifecycle_basic = lifecycle_basic.merge(trends, on=groupby_cols, how='left')
            lifecycle_features.append(trend_col)
        
        # Maturity indicators
        # Coefficient of variation as stability measure
        stability_stats = df_sorted.groupby(groupby_cols)[self.value_col].agg([
            'mean', 'std'
        ]).reset_index()
        stability_stats['demand_stability'] = (
            stability_stats['mean'] / (stability_stats['std'] + 1e-8)
        )
        stability_stats['demand_volatility'] = (
            stability_stats['std'] / (stability_stats['mean'] + 1e-8)
        )
        
        lifecycle_basic = lifecycle_basic.merge(
            stability_stats[groupby_cols + ['demand_stability', 'demand_volatility']], 
            on=groupby_cols, 
            how='left'
        )
        lifecycle_features.extend(['demand_stability', 'demand_volatility'])
        
        # Decline indicators
        # Recent performance vs historical
        def calculate_decline_indicators(group):
            """Calculate if product is declining"""
            if len(group) < 8:
                return pd.Series({
                    'recent_vs_historical_ratio': 1.0,
                    'is_declining': 0
                })
            
            # Split into recent (last 25%) and historical (first 75%)
            split_point = int(len(group) * 0.75)
            historical = group.iloc[:split_point]
            recent = group.iloc[split_point:]
            
            historical_avg = historical[self.value_col].mean()
            recent_avg = recent[self.value_col].mean()
            
            ratio = recent_avg / (historical_avg + 1e-8)
            is_declining = 1 if ratio < 0.7 else 0
            
            return pd.Series({
                'recent_vs_historical_ratio': ratio,
                'is_declining': is_declining
            })
        
        decline_stats = df_sorted.groupby(groupby_cols).apply(calculate_decline_indicators).reset_index()
        lifecycle_basic = lifecycle_basic.merge(decline_stats, on=groupby_cols, how='left')
        lifecycle_features.extend(['recent_vs_historical_ratio', 'is_declining'])
        
        # Merge back to main dataframe
        df = df.merge(lifecycle_basic, on=groupby_cols, how='left')
        
        self.features_created.extend(lifecycle_features)
        print(f"[OK] Created {len(lifecycle_features)} lifecycle features")
        
        return df
    
    def create_market_dynamics_features(self, df: pd.DataFrame,
                                      groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create market dynamics and competitive behavior features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print("[INFO] Creating market dynamics features...")
        
        market_features = []
        
        # Market share analysis
        total_market_volume = df[self.value_col].sum()
        
        # Product market share
        product_shares = df.groupby('internal_product_id')[self.value_col].sum().reset_index()
        product_shares['product_market_share'] = product_shares[self.value_col] / total_market_volume
        product_shares['product_market_share_rank'] = product_shares['product_market_share'].rank(ascending=False)
        
        df = df.merge(
            product_shares[['internal_product_id', 'product_market_share', 'product_market_share_rank']], 
            on='internal_product_id', 
            how='left'
        )
        market_features.extend(['product_market_share', 'product_market_share_rank'])
        
        # Store market share
        store_shares = df.groupby('internal_store_id')[self.value_col].sum().reset_index()
        store_shares['store_market_share'] = store_shares[self.value_col] / total_market_volume
        store_shares['store_market_share_rank'] = store_shares['store_market_share'].rank(ascending=False)
        
        df = df.merge(
            store_shares[['internal_store_id', 'store_market_share', 'store_market_share_rank']], 
            on='internal_store_id', 
            how='left'
        )
        market_features.extend(['store_market_share', 'store_market_share_rank'])
        
        # Competitive pressure analysis
        if 'categoria' in df.columns:
            # Number of competing products in same category
            category_competition = df.groupby('categoria')['internal_product_id'].nunique().reset_index()
            category_competition.columns = ['categoria', 'category_product_count']
            
            df = df.merge(category_competition, on='categoria', how='left')
            
            # Competitive pressure score (more products = higher pressure)
            df['competitive_pressure'] = df['category_product_count'] / df['category_product_count'].max()
            market_features.extend(['category_product_count', 'competitive_pressure'])
            
            # Category concentration (HHI approximation)
            category_hhi = df.groupby('categoria').apply(
                lambda x: ((x[self.value_col] / x[self.value_col].sum()) ** 2).sum()
            ).reset_index()
            category_hhi.columns = ['categoria', 'category_concentration_hhi']
            
            df = df.merge(category_hhi, on='categoria', how='left')
            market_features.append('category_concentration_hhi')
        
        # Market position features
        # Product performance relative to category average
        if 'categoria' in df.columns:
            category_avg = df.groupby('categoria')[self.value_col].mean().reset_index()
            category_avg.columns = ['categoria', 'category_avg_volume']
            
            df = df.merge(category_avg, on='categoria', how='left')
            df['product_vs_category_performance'] = df[self.value_col] / (df['category_avg_volume'] + 1e-8)
            market_features.extend(['category_avg_volume', 'product_vs_category_performance'])
        
        # Store performance relative to region
        if 'zipcode' in df.columns:
            region_avg = df.groupby('zipcode')[self.value_col].mean().reset_index()
            region_avg.columns = ['zipcode', 'region_avg_volume']
            
            df = df.merge(region_avg, on='zipcode', how='left')
            df['store_vs_region_performance'] = df[self.value_col] / (df['region_avg_volume'] + 1e-8)
            market_features.extend(['region_avg_volume', 'store_vs_region_performance'])
        
        # Market dominance indicators
        # Top player identification
        df['is_top_product'] = (df['product_market_share_rank'] <= 100).astype(int)
        df['is_top_store'] = (df['store_market_share_rank'] <= 100).astype(int)
        market_features.extend(['is_top_product', 'is_top_store'])
        
        # Long tail identification
        total_products = df['internal_product_id'].nunique()
        total_stores = df['internal_store_id'].nunique()
        
        df['is_longtail_product'] = (df['product_market_share_rank'] > total_products * 0.8).astype(int)
        df['is_longtail_store'] = (df['store_market_share_rank'] > total_stores * 0.8).astype(int)
        market_features.extend(['is_longtail_product', 'is_longtail_store'])
        
        self.features_created.extend(market_features)
        print(f"[OK] Created {len(market_features)} market dynamics features")
        
        return df
    
    def create_cross_selling_features(self, df: pd.DataFrame,
                                    groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create cross-selling and product affinity features"""
        
        df = df.copy()
        
        print("[INFO] Creating cross-selling behavior features...")
        
        cross_selling_features = []
        
        # Create transaction baskets (same date, same store)
        transaction_baskets = df.groupby([self.date_col, 'internal_store_id'])['internal_product_id'].apply(list).reset_index()
        transaction_baskets['basket_size'] = transaction_baskets['internal_product_id'].apply(len)
        
        # Product co-occurrence analysis
        product_cooccurrence = {}
        
        for basket in transaction_baskets['internal_product_id']:
            if len(basket) > 1:
                for i, prod1 in enumerate(basket):
                    for prod2 in basket[i+1:]:
                        pair = tuple(sorted([prod1, prod2]))
                        product_cooccurrence[pair] = product_cooccurrence.get(pair, 0) + 1
        
        # Calculate product affinity scores
        product_affinity_scores = {}
        product_frequencies = df['internal_product_id'].value_counts().to_dict()
        total_baskets = len(transaction_baskets)
        
        for (prod1, prod2), cooccur_count in product_cooccurrence.items():
            if cooccur_count >= 5:  # Minimum co-occurrence threshold
                freq1 = product_frequencies.get(prod1, 0)
                freq2 = product_frequencies.get(prod2, 0)
                
                # Lift calculation
                expected = (freq1 * freq2) / (total_baskets ** 2)
                lift = (cooccur_count / total_baskets) / (expected + 1e-8)
                
                product_affinity_scores[(prod1, prod2)] = {
                    'cooccurrence': cooccur_count,
                    'lift': lift
                }
        
        # Product cross-selling strength
        product_cross_selling = {}
        for product in df['internal_product_id'].unique():
            related_products = []
            total_lift = 0
            
            for (prod1, prod2), metrics in product_affinity_scores.items():
                if product in [prod1, prod2]:
                    related_products.append(metrics['lift'])
                    total_lift += metrics['lift']
            
            product_cross_selling[product] = {
                'cross_selling_partners': len(related_products),
                'avg_cross_selling_lift': np.mean(related_products) if related_products else 0,
                'total_cross_selling_strength': total_lift
            }
        
        # Convert to DataFrame and merge
        cross_selling_df = pd.DataFrame.from_dict(product_cross_selling, orient='index').reset_index()
        cross_selling_df.columns = ['internal_product_id', 'cross_selling_partners', 'avg_cross_selling_lift', 'total_cross_selling_strength']
        
        df = df.merge(cross_selling_df, on='internal_product_id', how='left')
        cross_selling_features.extend(['cross_selling_partners', 'avg_cross_selling_lift', 'total_cross_selling_strength'])
        
        # Basket analysis features
        basket_stats = transaction_baskets[['internal_store_id', 'basket_size']].groupby('internal_store_id').agg([
            'mean', 'std', 'max'
        ]).reset_index()
        basket_stats.columns = ['internal_store_id', 'avg_basket_size', 'std_basket_size', 'max_basket_size']
        
        df = df.merge(basket_stats, on='internal_store_id', how='left')
        cross_selling_features.extend(['avg_basket_size', 'std_basket_size', 'max_basket_size'])
        
        # Product basket penetration
        product_in_baskets = df.groupby('internal_product_id').apply(
            lambda x: len(x.groupby([self.date_col, 'internal_store_id']).size())
        ).reset_index()
        product_in_baskets.columns = ['internal_product_id', 'product_basket_appearances']
        
        df = df.merge(product_in_baskets, on='internal_product_id', how='left')
        cross_selling_features.append('product_basket_appearances')
        
        # Fill NaN values
        for feature in cross_selling_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        self.features_created.extend(cross_selling_features)
        print(f"[OK] Created {len(cross_selling_features)} cross-selling features")
        
        return df
    
    def create_cannibalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cannibalization risk analysis features"""
        
        df = df.copy()
        
        print("[INFO] Creating cannibalization analysis features...")
        
        cannibalization_features = []
        
        # Same category competition within stores
        if 'categoria' in df.columns:
            # Number of products from same category in each store
            category_store_products = df.groupby(['internal_store_id', 'categoria'])['internal_product_id'].nunique().reset_index()
            category_store_products.columns = ['internal_store_id', 'categoria', 'category_products_in_store']
            
            df = df.merge(category_store_products, on=['internal_store_id', 'categoria'], how='left')
            
            # Cannibalization risk (more similar products = higher risk)
            df['cannibalization_risk'] = df['category_products_in_store'] / df['category_products_in_store'].max()
            cannibalization_features.extend(['category_products_in_store', 'cannibalization_risk'])
            
            # Product dominance within category-store
            category_store_volume = df.groupby(['internal_store_id', 'categoria'])[self.value_col].sum().reset_index()
            category_store_volume.columns = ['internal_store_id', 'categoria', 'category_total_volume_in_store']
            
            df = df.merge(category_store_volume, on=['internal_store_id', 'categoria'], how='left')
            df['product_category_dominance'] = df[self.value_col] / (df['category_total_volume_in_store'] + 1e-8)
            cannibalization_features.extend(['category_total_volume_in_store', 'product_category_dominance'])
        
        # Same brand competition
        if 'marca' in df.columns:
            brand_store_products = df.groupby(['internal_store_id', 'marca'])['internal_product_id'].nunique().reset_index()
            brand_store_products.columns = ['internal_store_id', 'marca', 'brand_products_in_store']
            
            df = df.merge(brand_store_products, on=['internal_store_id', 'marca'], how='left')
            
            df['brand_cannibalization_risk'] = df['brand_products_in_store'] / df['brand_products_in_store'].max()
            cannibalization_features.extend(['brand_products_in_store', 'brand_cannibalization_risk'])
        
        # Price-based cannibalization (similar price points)
        if 'net_value' in df.columns:
            df['unit_price'] = df['net_value'] / (df[self.value_col] + 1e-8)
            
            # Price tier classification
            df['price_tier'] = pd.qcut(df['unit_price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Premium'], duplicates='drop')
            
            # Products in same price tier within store
            price_tier_competition = df.groupby(['internal_store_id', 'price_tier'])['internal_product_id'].nunique().reset_index()
            price_tier_competition.columns = ['internal_store_id', 'price_tier', 'same_price_tier_products']
            
            df = df.merge(price_tier_competition, on=['internal_store_id', 'price_tier'], how='left')
            df['price_tier_competition'] = df['same_price_tier_products'] / df['same_price_tier_products'].max()
            cannibalization_features.extend(['same_price_tier_products', 'price_tier_competition'])
        
        # Substitute product identification (products that are negatively correlated)
        if len(df) > 1000:  # Only for larger datasets due to computational cost
            # Sample products for correlation analysis
            sample_products = df['internal_product_id'].value_counts().head(100).index.tolist()
            
            correlation_data = df[df['internal_product_id'].isin(sample_products)].pivot_table(
                values=self.value_col,
                index=[self.date_col, 'internal_store_id'],
                columns='internal_product_id',
                aggfunc='sum',
                fill_value=0
            )
            
            if correlation_data.shape[1] > 1:
                # Calculate correlation matrix
                correlation_matrix = correlation_data.corr()
                
                # Find negative correlations (potential substitutes)
                substitute_relationships = {}
                for product in sample_products:
                    if product in correlation_matrix.columns:
                        negatively_correlated = correlation_matrix[product][correlation_matrix[product] < -0.3]
                        substitute_relationships[product] = len(negatively_correlated)
                
                substitute_df = pd.DataFrame.from_dict(substitute_relationships, orient='index', columns=['substitute_products_count']).reset_index()
                substitute_df.columns = ['internal_product_id', 'substitute_products_count']
                
                df = df.merge(substitute_df, on='internal_product_id', how='left')
                df['substitute_products_count'] = df['substitute_products_count'].fillna(0)
                cannibalization_features.append('substitute_products_count')
        
        self.features_created.extend(cannibalization_features)
        print(f"[OK] Created {len(cannibalization_features)} cannibalization features")
        
        return df
    
    def create_all_behavioral_features(self, df: pd.DataFrame,
                                     groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create all behavioral features in optimized sequence"""
        
        print("\n" + "="*80)
        print("BEHAVIORAL FEATURES ENGINE - CREATING ALL FEATURES")
        print("="*80)
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
        
        initial_shape = df.shape
        print(f"[START] Initial dataset: {initial_shape}")
        
        # Sequence of feature creation (optimized order)
        df = self.create_intermittency_features(df, groupby_cols)
        df = self.create_lifecycle_features(df, groupby_cols)
        df = self.create_market_dynamics_features(df, groupby_cols)
        df = self.create_cross_selling_features(df, groupby_cols)
        df = self.create_cannibalization_features(df)
        
        final_shape = df.shape
        features_added = final_shape[1] - initial_shape[1]
        
        print(f"\n[SUMMARY] Behavioral Features Engine completed:")
        print(f"  Features added: {features_added}")
        print(f"  Total features created: {len(self.features_created)}")
        print(f"  Final dataset shape: {final_shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def get_feature_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights about created behavioral features"""
        
        insights = {
            'total_features': len(self.features_created),
            'feature_categories': {
                'intermittency_features': len([f for f in self.features_created if any(x in f for x in ['zero', 'intermittency', 'purchase', 'burst'])]),
                'lifecycle_features': len([f for f in self.features_created if any(x in f for x in ['lifecycle', 'growth', 'decline', 'stability', 'days_since'])]),
                'market_features': len([f for f in self.features_created if any(x in f for x in ['market', 'competitive', 'share', 'rank'])]),
                'cross_selling_features': len([f for f in self.features_created if any(x in f for x in ['cross_selling', 'basket', 'affinity'])]),
                'cannibalization_features': len([f for f in self.features_created if any(x in f for x in ['cannibalization', 'substitute', 'dominance'])])
            },
            'key_insights': [
                f"Intermittency analysis for {len([f for f in self.features_created if 'zero' in f])} zero-demand features",
                f"Lifecycle classification across {len(self.lifecycle_windows)} time windows",
                f"Market dynamics features for competitive positioning",
                f"Cross-selling strength analysis for product affinity",
                f"Cannibalization risk assessment for substitutes",
                f"Behavioral patterns optimized for intermittent demand forecasting"
            ],
            'features_list': self.features_created
        }
        
        return insights

def main():
    """Demonstration and testing of Behavioral Features Engine"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("BEHAVIORAL FEATURES ENGINE - DEMONSTRATION")
    print("="*80)
    
    # Load sample data for testing
    try:
        from src.utils.data_loader import load_data_efficiently
        
        print("Loading sample data...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=50000,
            sample_products=1000
        )
        
        # Add some behavioral data simulation
        if 'produto' in prod_df.columns and len(prod_df) > 0:
            # Map basic product info
            prod_sample = prod_df.sample(min(100, len(prod_df)))
            product_mapping = {}
            for i, row in prod_sample.iterrows():
                product_id = hash(str(row['produto'])) % 1000000
                product_mapping[product_id] = {
                    'categoria': row.get('categoria', 'Unknown'),
                    'marca': row.get('marca', 'Unknown')
                }
            
            # Add categories and brands to transaction data
            trans_df['categoria'] = trans_df['internal_product_id'].map(lambda x: product_mapping.get(x, {}).get('categoria', 'Unknown'))
            trans_df['marca'] = trans_df['internal_product_id'].map(lambda x: product_mapping.get(x, {}).get('marca', 'Unknown'))
        
        print(f"Sample data loaded: {trans_df.shape}")
        print(f"Columns: {list(trans_df.columns)}")
        
        # Initialize engine
        engine = BehavioralFeaturesEngine()
        
        # Create all behavioral features
        features_df = engine.create_all_behavioral_features(trans_df)
        
        # Get insights
        insights = engine.get_feature_insights(features_df)
        
        print("\n" + "="*80)
        print("BEHAVIORAL FEATURES INSIGHTS")
        print("="*80)
        
        print(f"Total features created: {insights['total_features']}")
        print("\nFeature categories:")
        for category, count in insights['feature_categories'].items():
            print(f"  {category}: {count}")
        
        print("\nKey insights:")
        for insight in insights['key_insights']:
            print(f"  • {insight}")
        
        # Save results
        output_dir = Path("../../data/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_file = output_dir / "behavioral_features_demo.parquet"
        features_df.to_parquet(features_file, index=False)
        
        print(f"\n[SAVED] Sample features saved to: {features_file}")
        
        return features_df, insights
        
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        print("This is expected if running outside the project structure")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results = main()