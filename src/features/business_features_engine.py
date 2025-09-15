#!/usr/bin/env python3
"""
BUSINESS FEATURES ENGINE - Hackathon Forecast Big Data 2025
Business Logic and Domain-Specific Feature Engineering

Features baseadas em domain expertise e novas colunas:
- Profit margin analysis (gross_profit)
- Discount and promotion features (discount)
- Geographic and regional features (zipcode)
- Calendar and seasonal business events
- Revenue optimization features

Otimizado para WMAPE: Features de negócio que impactam volume e receita
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings('ignore')

class BusinessFeaturesEngine:
    """
    Business Logic Features Engine
    
    Capabilities:
    - Profit margin and financial health features
    - Discount and promotional impact analysis
    - Geographic and regional business patterns
    - Calendar-based business events
    - Revenue optimization features
    - Price positioning and competitive analysis
    """
    
    def __init__(self, date_col: str = 'transaction_date', value_col: str = 'quantity'):
        self.date_col = date_col
        self.value_col = value_col
        self.features_created = []
        self.feature_metadata = {}
        
        # Business constants
        self.profit_margin_percentiles = [0.25, 0.5, 0.75, 0.9]
        self.discount_thresholds = [0.05, 0.1, 0.2, 0.3]  # 5%, 10%, 20%, 30%
        
        # Initialize Brazilian holidays
        try:
            self.br_holidays = holidays.Brazil(years=range(2022, 2024))
        except:
            self.br_holidays = {}
            print("[WARNING] Holidays library not available, using empty holidays")
        
    def create_profit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create profit and financial health features"""
        
        df = df.copy()
        
        print("[INFO] Creating profit and financial health features...")
        
        profit_features = []
        
        # Basic profit calculations
        if 'gross_profit' in df.columns:
            df['profit_margin'] = df['gross_profit'] / (df['gross_value'] + 1e-8)
            df['profit_per_unit'] = df['gross_profit'] / (df[self.value_col] + 1e-8)
            df['profit_margin_category'] = pd.cut(
                df['profit_margin'], 
                bins=[-np.inf, 0.1, 0.2, 0.3, np.inf], 
                labels=['Low', 'Medium', 'High', 'Premium']
            )
            profit_features.extend(['profit_margin', 'profit_per_unit'])
            
            # Profit distribution analysis
            profit_percentiles = df['profit_margin'].quantile(self.profit_margin_percentiles)
            for i, percentile in enumerate(self.profit_margin_percentiles):
                threshold = profit_percentiles[percentile]
                df[f'is_profit_top_{int(percentile*100)}pct'] = (df['profit_margin'] >= threshold).astype(int)
                profit_features.append(f'is_profit_top_{int(percentile*100)}pct')
        
        # Revenue efficiency features
        if 'net_value' in df.columns and 'gross_value' in df.columns:
            df['revenue_efficiency'] = df['net_value'] / (df['gross_value'] + 1e-8)
            df['total_deductions'] = df['gross_value'] - df['net_value']
            df['deduction_rate'] = df['total_deductions'] / (df['gross_value'] + 1e-8)
            profit_features.extend(['revenue_efficiency', 'total_deductions', 'deduction_rate'])
        
        # Price realization features
        if 'net_value' in df.columns:
            df['unit_net_price'] = df['net_value'] / (df[self.value_col] + 1e-8)
            df['unit_gross_price'] = df['gross_value'] / (df[self.value_col] + 1e-8)
            
            # Price tier classification
            df['price_tier'] = pd.qcut(
                df['unit_net_price'], 
                q=5, 
                labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'],
                duplicates='drop'
            )
            profit_features.extend(['unit_net_price', 'unit_gross_price'])
        
        # Product-level profit aggregations
        if 'gross_profit' in df.columns:
            product_profit_aggs = df.groupby('internal_product_id').agg({
                'gross_profit': ['sum', 'mean', 'std'],
                'profit_margin': ['mean', 'std'],
                self.value_col: 'sum'
            })
            
            # Flatten column names
            product_profit_aggs.columns = [
                f'product_profit_{col[0]}_{col[1]}' if col[1] else f'product_profit_{col[0]}'
                for col in product_profit_aggs.columns
            ]
            product_profit_aggs = product_profit_aggs.reset_index()
            
            # Profit per unit consistency
            product_profit_aggs['product_profit_consistency'] = (
                product_profit_aggs['product_profit_profit_margin_mean'] / 
                (product_profit_aggs['product_profit_profit_margin_std'] + 1e-8)
            )
            
            profit_features.extend([col for col in product_profit_aggs.columns if col != 'internal_product_id'])
            
            # Merge back
            df = df.merge(product_profit_aggs, on='internal_product_id', how='left')
        
        # Store-level profit performance
        if 'gross_profit' in df.columns:
            store_profit_aggs = df.groupby('internal_store_id').agg({
                'gross_profit': ['sum', 'mean'],
                'profit_margin': 'mean',
                'revenue_efficiency': 'mean' if 'revenue_efficiency' in df.columns else 'count'
            })
            
            store_profit_aggs.columns = [
                f'store_profit_{col[0]}_{col[1]}' if col[1] else f'store_profit_{col[0]}'
                for col in store_profit_aggs.columns
            ]
            store_profit_aggs = store_profit_aggs.reset_index()
            
            profit_features.extend([col for col in store_profit_aggs.columns if col != 'internal_store_id'])
            
            # Merge back
            df = df.merge(store_profit_aggs, on='internal_store_id', how='left')
        
        self.features_created.extend(profit_features)
        print(f"[OK] Created {len(profit_features)} profit features")
        
        return df
    
    def create_discount_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create discount and promotion analysis features"""
        
        df = df.copy()
        
        print("[INFO] Creating discount and promotion features...")
        
        discount_features = []
        
        # Basic discount analysis
        if 'discount' in df.columns:
            # Discount rate and categories
            df['discount_rate'] = df['discount'] / (df['gross_value'] + 1e-8)
            
            # Discount presence indicator
            df['has_discount'] = (df['discount'] > 0.01).astype(int)
            
            # Discount tier classification
            df['discount_tier'] = pd.cut(
                df['discount_rate'],
                bins=[0, 0.05, 0.1, 0.2, 1.0],
                labels=['None/Low', 'Light', 'Moderate', 'Heavy'],
                include_lowest=True
            )
            
            discount_features.extend(['discount_rate', 'has_discount'])
            
            # Discount effectiveness analysis
            # Compare discounted vs non-discounted performance
            discounted_avg = df[df['has_discount'] == 1][self.value_col].mean()
            non_discounted_avg = df[df['has_discount'] == 0][self.value_col].mean()
            
            df['discount_lift_potential'] = discounted_avg / (non_discounted_avg + 1e-8)
            discount_features.append('discount_lift_potential')
        
        # Price drop detection (promotion identification)
        if self.date_col in df.columns:
            # Calculate rolling average price for promotion detection
            df_sorted = df.sort_values(['internal_product_id', 'internal_store_id', self.date_col])
            
            df_sorted['rolling_avg_price'] = df_sorted.groupby(['internal_product_id', 'internal_store_id'])['unit_net_price'].rolling(
                window=7, min_periods=1
            ).mean().reset_index(level=[0,1], drop=True)
            
            # Promotion detection (price significantly below rolling average)
            df_sorted['price_drop_pct'] = (
                (df_sorted['rolling_avg_price'] - df_sorted['unit_net_price']) / 
                (df_sorted['rolling_avg_price'] + 1e-8)
            )
            df_sorted['is_promotion'] = (df_sorted['price_drop_pct'] > 0.1).astype(int)  # 10% price drop
            
            discount_features.extend(['price_drop_pct', 'is_promotion'])
            
            # Promotion frequency by product/store
            promotion_freq = df_sorted.groupby(['internal_product_id', 'internal_store_id'])['is_promotion'].agg([
                'mean', 'sum', 'count'
            ]).reset_index()
            promotion_freq.columns = ['internal_product_id', 'internal_store_id', 'promotion_frequency', 'total_promotions', 'total_periods']
            
            df_sorted = df_sorted.merge(promotion_freq, on=['internal_product_id', 'internal_store_id'], how='left')
            discount_features.extend(['promotion_frequency', 'total_promotions'])
            
            df = df_sorted
        
        # Competitive pricing features
        if 'unit_net_price' in df.columns:
            # Price position within category/store
            if 'categoria' in df.columns:
                category_price_stats = df.groupby(['categoria', 'internal_store_id'])['unit_net_price'].agg([
                    'mean', 'median', 'std'
                ]).reset_index()
                category_price_stats.columns = ['categoria', 'internal_store_id', 'category_avg_price', 'category_median_price', 'category_price_std']
                
                df = df.merge(category_price_stats, on=['categoria', 'internal_store_id'], how='left')
                
                # Price positioning
                df['price_vs_category_avg'] = df['unit_net_price'] / (df['category_avg_price'] + 1e-8)
                df['price_position'] = np.select([
                    df['price_vs_category_avg'] < 0.8,
                    df['price_vs_category_avg'] < 0.95,
                    df['price_vs_category_avg'] < 1.05,
                    df['price_vs_category_avg'] < 1.2
                ], ['Budget', 'Economy', 'Standard', 'Premium'], default='Luxury')
                
                discount_features.extend(['category_avg_price', 'price_vs_category_avg'])
        
        # Margin optimization features
        if 'gross_profit' in df.columns and 'discount' in df.columns:
            # Optimal discount analysis
            df['profit_after_discount'] = df['gross_profit'] - df['discount']
            df['profit_margin_after_discount'] = df['profit_after_discount'] / (df['gross_value'] + 1e-8)
            
            # Discount efficiency (profit impact vs volume lift)
            if 'discount_lift_potential' in df.columns:
                df['discount_efficiency'] = df['discount_lift_potential'] / (df['discount_rate'] + 1e-8)
            
            discount_features.extend(['profit_after_discount', 'profit_margin_after_discount'])
        
        self.features_created.extend(discount_features)
        print(f"[OK] Created {len(discount_features)} discount/promotion features")
        
        return df
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic and regional business features"""
        
        df = df.copy()
        
        if 'zipcode' not in df.columns:
            print("[WARNING] Zipcode column not found, skipping geographic features")
            return df
            
        print("[INFO] Creating geographic and regional features...")
        
        geo_features = []
        
        # Basic regional aggregations
        regional_aggs = df.groupby('zipcode')[self.value_col].agg([
            'sum', 'mean', 'count', 'std'
        ]).reset_index()
        regional_aggs.columns = ['zipcode'] + [f'region_{self.value_col}_{col}' for col in ['sum', 'mean', 'count', 'std']]
        
        # Regional market share
        total_volume = df[self.value_col].sum()
        regional_aggs['region_market_share'] = regional_aggs[f'region_{self.value_col}_sum'] / total_volume
        
        # Regional performance tiers
        regional_aggs['region_performance_tier'] = pd.qcut(
            regional_aggs[f'region_{self.value_col}_sum'],
            q=3,
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
        
        geo_features.extend([
            f'region_{self.value_col}_sum', f'region_{self.value_col}_mean',
            f'region_{self.value_col}_count', 'region_market_share'
        ])
        
        # Store density by region
        store_density = df.groupby('zipcode')['internal_store_id'].nunique().reset_index()
        store_density.columns = ['zipcode', 'stores_in_region']
        
        regional_aggs = regional_aggs.merge(store_density, on='zipcode', how='left')
        
        # Revenue per store by region
        regional_aggs['revenue_per_store_in_region'] = (
            regional_aggs[f'region_{self.value_col}_sum'] / regional_aggs['stores_in_region']
        )
        geo_features.extend(['stores_in_region', 'revenue_per_store_in_region'])
        
        # Product diversity by region
        product_diversity = df.groupby('zipcode')['internal_product_id'].nunique().reset_index()
        product_diversity.columns = ['zipcode', 'products_in_region']
        
        regional_aggs = regional_aggs.merge(product_diversity, on='zipcode', how='left')
        geo_features.append('products_in_region')
        
        # Economic indicators (approximated from sales data)
        # Average transaction size by region
        if 'net_value' in df.columns:
            regional_economics = df.groupby('zipcode').agg({
                'net_value': ['mean', 'median', 'std'],
                'unit_net_price': 'mean' if 'unit_net_price' in df.columns else 'count'
            }).reset_index()
            
            regional_economics.columns = ['zipcode'] + [
                f'region_{col[0]}_{col[1]}' for col in regional_economics.columns[1:]
            ]
            
            # Economic strength indicator (higher values = stronger economy)
            if 'region_net_value_mean' in regional_economics.columns:
                regional_economics['region_economic_strength'] = (
                    regional_economics['region_net_value_mean'] / regional_economics['region_net_value_mean'].median()
                )
                geo_features.append('region_economic_strength')
            
            regional_aggs = regional_aggs.merge(regional_economics, on='zipcode', how='left')
            geo_features.extend([col for col in regional_economics.columns if col != 'zipcode'])
        
        # Urban vs Rural classification (approximation based on store density)
        regional_aggs['urban_rural_indicator'] = np.select([
            regional_aggs['stores_in_region'] >= 50,
            regional_aggs['stores_in_region'] >= 20,
            regional_aggs['stores_in_region'] >= 5
        ], ['Urban', 'Suburban', 'Town'], default='Rural')
        
        geo_features.append('urban_rural_indicator')
        
        # Regional competition intensity
        if 'categoria' in df.columns:
            category_competition = df.groupby(['zipcode', 'categoria'])['internal_product_id'].nunique().reset_index()
            competition_intensity = category_competition.groupby('zipcode')['internal_product_id'].agg([
                'mean', 'max'
            ]).reset_index()
            competition_intensity.columns = ['zipcode', 'avg_category_competition', 'max_category_competition']
            
            regional_aggs = regional_aggs.merge(competition_intensity, on='zipcode', how='left')
            geo_features.extend(['avg_category_competition', 'max_category_competition'])
        
        # Merge all geographic features back
        df = df.merge(regional_aggs, on='zipcode', how='left')
        
        self.features_created.extend(geo_features)
        print(f"[OK] Created {len(geo_features)} geographic features")
        
        return df
    
    def create_calendar_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar and business event features"""
        
        df = df.copy()
        
        if self.date_col not in df.columns:
            print("[WARNING] Date column not found, skipping calendar features")
            return df
            
        print("[INFO] Creating calendar and business event features...")
        
        calendar_features = []
        
        # Ensure date column is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Holiday features
        df['is_holiday'] = df[self.date_col].apply(lambda x: x.date() in self.br_holidays).astype(int)
        
        # Holiday proximity
        def days_to_next_holiday(date):
            """Calculate days until next holiday"""
            for days_ahead in range(0, 30):
                future_date = date + timedelta(days=days_ahead)
                if future_date.date() in self.br_holidays:
                    return days_ahead
            return 30
        
        def days_since_last_holiday(date):
            """Calculate days since last holiday"""
            for days_back in range(0, 30):
                past_date = date - timedelta(days=days_back)
                if past_date.date() in self.br_holidays:
                    return days_back
            return 30
        
        df['days_to_next_holiday'] = df[self.date_col].apply(days_to_next_holiday)
        df['days_since_last_holiday'] = df[self.date_col].apply(days_since_last_holiday)
        
        # Holiday season (within 7 days of holiday)
        df['is_holiday_season'] = ((df['days_to_next_holiday'] <= 7) | 
                                  (df['days_since_last_holiday'] <= 7)).astype(int)
        
        calendar_features.extend(['is_holiday', 'days_to_next_holiday', 'days_since_last_holiday', 'is_holiday_season'])
        
        # Payroll calendar approximation
        df['is_month_start'] = (df[self.date_col].dt.day <= 5).astype(int)
        df['is_month_end'] = (df[self.date_col].dt.day >= 25).astype(int)
        df['is_payday_period'] = ((df[self.date_col].dt.day <= 5) | 
                                 (df[self.date_col].dt.day >= 25)).astype(int)
        
        calendar_features.extend(['is_month_start', 'is_month_end', 'is_payday_period'])
        
        # School calendar approximation (Brazilian school year)
        df['month'] = df[self.date_col].dt.month
        df['is_school_period'] = df['month'].apply(
            lambda x: 0 if x in [12, 1, 7] else 1  # Dec, Jan, July are holidays
        ).astype(int)
        df['is_school_holiday'] = (1 - df['is_school_period']).astype(int)
        
        calendar_features.extend(['is_school_period', 'is_school_holiday'])
        
        # Seasonal business events
        df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_year_end'] = df['month'].isin([11, 12]).astype(int)
        df['is_carnival_season'] = df['month'].isin([2, 3]).astype(int)  # Approximation
        
        calendar_features.extend(['is_summer', 'is_winter', 'is_year_end', 'is_carnival_season'])
        
        # Business quarter features
        df['quarter'] = df[self.date_col].dt.quarter
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        df['is_quarter_start'] = df[self.date_col].dt.month.isin([1, 4, 7, 10]).astype(int)
        df['is_quarter_end'] = df[self.date_col].dt.month.isin([3, 6, 9, 12]).astype(int)
        
        calendar_features.extend(['is_q1', 'is_q4', 'is_quarter_start', 'is_quarter_end'])
        
        # Special retail events (approximation)
        df['is_black_friday_season'] = df['month'].isin([11]).astype(int)
        df['is_christmas_season'] = df['month'].isin([12]).astype(int)
        df['is_mothers_day_season'] = df['month'].isin([5]).astype(int)
        df['is_fathers_day_season'] = df['month'].isin([8]).astype(int)
        
        calendar_features.extend(['is_black_friday_season', 'is_christmas_season', 'is_mothers_day_season', 'is_fathers_day_season'])
        
        self.features_created.extend(calendar_features)
        print(f"[OK] Created {len(calendar_features)} calendar features")
        
        return df
    
    def create_revenue_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create revenue optimization and business intelligence features"""
        
        df = df.copy()
        
        print("[INFO] Creating revenue optimization features...")
        
        revenue_features = []
        
        # Revenue per unit and efficiency metrics
        if 'net_value' in df.columns:
            df['revenue_per_unit'] = df['net_value'] / (df[self.value_col] + 1e-8)
            
            # Revenue concentration analysis
            product_revenue = df.groupby('internal_product_id')['net_value'].sum().reset_index()
            product_revenue['revenue_rank'] = product_revenue['net_value'].rank(ascending=False)
            product_revenue['revenue_percentile'] = product_revenue['revenue_rank'] / len(product_revenue)
            
            df = df.merge(product_revenue[['internal_product_id', 'revenue_rank', 'revenue_percentile']], on='internal_product_id', how='left')
            
            # Revenue tier classification
            df['revenue_tier'] = pd.cut(
                df['revenue_percentile'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['Tier_1', 'Tier_2', 'Tier_3', 'Tier_4'],
                include_lowest=True
            )
            
            revenue_features.extend(['revenue_per_unit', 'revenue_rank', 'revenue_percentile'])
        
        # Business performance ratios
        if all(col in df.columns for col in ['net_value', 'gross_value', 'gross_profit']):
            # Profitability ratios
            df['gross_margin_ratio'] = df['gross_profit'] / (df['net_value'] + 1e-8)
            df['cost_ratio'] = (df['net_value'] - df['gross_profit']) / (df['net_value'] + 1e-8)
            df['markup_ratio'] = df['gross_profit'] / ((df['net_value'] - df['gross_profit']) + 1e-8)
            
            revenue_features.extend(['gross_margin_ratio', 'cost_ratio', 'markup_ratio'])
        
        # Customer value approximation (via sales patterns)
        customer_proxy_features = df.groupby(['internal_store_id', self.date_col]).agg({
            'net_value': 'sum',
            self.value_col: 'sum',
            'internal_product_id': 'nunique'
        }).reset_index()
        customer_proxy_features.columns = ['internal_store_id', self.date_col, 'daily_revenue', 'daily_quantity', 'daily_products']
        
        # Average transaction value per store per day
        customer_proxy_features['avg_transaction_value'] = customer_proxy_features['daily_revenue'] / (customer_proxy_features['daily_quantity'] + 1e-8)
        customer_proxy_features['products_per_transaction'] = customer_proxy_features['daily_products'] / (customer_proxy_features['daily_quantity'] + 1e-8)
        
        df = df.merge(customer_proxy_features, on=['internal_store_id', self.date_col], how='left')
        revenue_features.extend(['avg_transaction_value', 'products_per_transaction'])
        
        # Price elasticity estimation (simplified)
        if 'unit_net_price' in df.columns:
            # Price-quantity relationship by product
            price_elasticity = df.groupby('internal_product_id').apply(
                lambda x: x['unit_net_price'].corr(x[self.value_col]) if len(x) > 5 else 0
            ).reset_index()
            price_elasticity.columns = ['internal_product_id', 'price_quantity_correlation']
            
            # Elasticity classification
            price_elasticity['price_elasticity_type'] = price_elasticity['price_quantity_correlation'].apply(
                lambda x: 'Elastic' if x < -0.3 else 'Inelastic' if x > -0.1 else 'Unit_Elastic'
            )
            
            df = df.merge(price_elasticity, on='internal_product_id', how='left')
            revenue_features.extend(['price_quantity_correlation', 'price_elasticity_type'])
        
        # Inventory turnover approximation
        if 'net_value' in df.columns:
            # Product turnover rate (sales frequency)
            product_turnover = df.groupby('internal_product_id').agg({
                self.date_col: 'nunique',
                self.value_col: 'sum',
                'net_value': 'sum'
            }).reset_index()
            product_turnover['turnover_rate'] = product_turnover['quantity'] / (product_turnover['transaction_date'] + 1e-8)
            product_turnover['revenue_velocity'] = product_turnover['net_value'] / (product_turnover['transaction_date'] + 1e-8)
            
            df = df.merge(product_turnover[['internal_product_id', 'turnover_rate', 'revenue_velocity']], on='internal_product_id', how='left')
            revenue_features.extend(['turnover_rate', 'revenue_velocity'])
        
        # Business health indicators
        if 'gross_profit' in df.columns:
            # Profit contribution analysis
            total_profit = df['gross_profit'].sum()
            df['profit_contribution'] = df['gross_profit'] / total_profit
            
            # Critical business metrics
            df['is_profit_positive'] = (df['gross_profit'] > 0).astype(int)
            df['is_high_margin'] = (df['profit_margin'] > df['profit_margin'].median()).astype(int) if 'profit_margin' in df.columns else 0
            
            revenue_features.extend(['profit_contribution', 'is_profit_positive', 'is_high_margin'])
        
        self.features_created.extend(revenue_features)
        print(f"[OK] Created {len(revenue_features)} revenue optimization features")
        
        return df
    
    def create_all_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all business features in optimized sequence"""
        
        print("\n" + "="*80)
        print("BUSINESS FEATURES ENGINE - CREATING ALL FEATURES")
        print("="*80)
        
        initial_shape = df.shape
        print(f"[START] Initial dataset: {initial_shape}")
        
        # Sequence of feature creation (optimized order)
        df = self.create_profit_features(df)
        df = self.create_discount_promotion_features(df)
        df = self.create_geographic_features(df)
        df = self.create_calendar_business_features(df)
        df = self.create_revenue_optimization_features(df)
        
        final_shape = df.shape
        features_added = final_shape[1] - initial_shape[1]
        
        print(f"\n[SUMMARY] Business Features Engine completed:")
        print(f"  Features added: {features_added}")
        print(f"  Total features created: {len(self.features_created)}")
        print(f"  Final dataset shape: {final_shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def get_feature_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights about created business features"""
        
        insights = {
            'total_features': len(self.features_created),
            'feature_categories': {
                'profit_features': len([f for f in self.features_created if any(x in f for x in ['profit', 'margin', 'efficiency'])]),
                'discount_features': len([f for f in self.features_created if any(x in f for x in ['discount', 'promotion', 'price_drop'])]),
                'geographic_features': len([f for f in self.features_created if any(x in f for x in ['region', 'urban', 'zipcode'])]),
                'calendar_features': len([f for f in self.features_created if any(x in f for x in ['holiday', 'season', 'month', 'quarter'])]),
                'revenue_features': len([f for f in self.features_created if any(x in f for x in ['revenue', 'turnover', 'elasticity', 'transaction'])])
            },
            'key_insights': [
                f"Profit analysis features leveraging gross_profit column",
                f"Discount optimization features using discount column",
                f"Geographic intelligence from zipcode analysis",
                f"Business calendar features for seasonal patterns",
                f"Revenue optimization features for business intelligence",
                f"Domain expertise encoded in business logic features"
            ],
            'features_list': self.features_created
        }
        
        return insights

def main():
    """Demonstration and testing of Business Features Engine"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("BUSINESS FEATURES ENGINE - DEMONSTRATION")
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
        
        # Add business context from PDV data
        if 'zipcode' in pdv_df.columns:
            pdv_mapping = dict(zip(pdv_df['pdv'], pdv_df['zipcode']))
            trans_df['zipcode'] = trans_df['internal_store_id'].map(
                lambda x: pdv_mapping.get(str(x), np.random.randint(10000, 99999))
            )
        
        # Add category information
        if 'categoria' in prod_df.columns and len(prod_df) > 0:
            # Simple mapping for demo
            categories = prod_df['categoria'].unique()[:10]
            category_mapping = {i: np.random.choice(categories) for i in trans_df['internal_product_id'].unique()}
            trans_df['categoria'] = trans_df['internal_product_id'].map(category_mapping)
        
        print(f"Sample data loaded: {trans_df.shape}")
        print(f"Columns: {list(trans_df.columns)}")
        
        # Initialize engine
        engine = BusinessFeaturesEngine()
        
        # Create all business features
        features_df = engine.create_all_business_features(trans_df)
        
        # Get insights
        insights = engine.get_feature_insights(features_df)
        
        print("\n" + "="*80)
        print("BUSINESS FEATURES INSIGHTS")
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
        
        features_file = output_dir / "business_features_demo.parquet"
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