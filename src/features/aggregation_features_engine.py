#!/usr/bin/env python3
"""
AGGREGATION FEATURES ENGINE - Hackathon Forecast Big Data 2025
Advanced Aggregation Feature Engineering for Multi-level Analysis

Features baseadas nos insights da EDA:
- ABC Tier analysis (5.3% produtos = 79.9% volume)
- PDV performance tiers (25% high-performance = 87.9% volume)  
- Regional patterns (Região 80 = 62% volume)
- Category dynamics (7 categorias, concentração alta)

Otimizado para WMAPE: Agregações ponderadas por volume
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings('ignore')

class AggregationFeaturesEngine:
    """
    Advanced Aggregation Features Engine
    
    Capabilities:
    - Product-level aggregations by ABC tier
    - PDV-level performance features
    - Cross-dimensional features (product×pdv)
    - Category and regional aggregations  
    - Volume-weighted aggregations for WMAPE
    - Hierarchical feature engineering
    """
    
    def __init__(self, value_col: str = 'quantity'):
        self.value_col = value_col
        self.features_created = []
        self.feature_metadata = {}
        
        # Aggregation functions to use
        self.agg_functions = ['sum', 'mean', 'median', 'std', 'min', 'max', 'count']
        
        # Statistical functions for distribution analysis
        self.stat_functions = ['skew', 'kurtosis', 'var']
        
    def create_product_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive product-level aggregation features"""
        
        df = df.copy()
        
        print("[INFO] Creating product-level aggregation features...")
        
        product_features = []
        
        # Basic product aggregations
        product_aggs = df.groupby('internal_product_id')[self.value_col].agg(
            self.agg_functions
        ).reset_index()
        
        # Rename columns with prefix
        agg_cols = {}
        for func in self.agg_functions:
            old_col = func if func in product_aggs.columns else self.value_col + '_' + func
            new_col = f'product_{self.value_col}_{func}'
            if func in product_aggs.columns:
                agg_cols[func] = new_col
            product_features.append(new_col)
        
        product_aggs = product_aggs.rename(columns=agg_cols)
        
        # Advanced product statistics
        product_stats = df.groupby('internal_product_id').agg({
            self.value_col: ['skew', lambda x: x.kurtosis(), 'var'],
            'internal_store_id': 'nunique',
            'transaction_date': ['nunique', 'min', 'max']
        })
        
        # Flatten column names
        product_stats.columns = [
            f'product_{col[0]}_{col[1]}'.replace('<lambda>', 'kurtosis')
            for col in product_stats.columns
        ]
        product_stats = product_stats.reset_index()
        
        # Add to feature list
        product_features.extend([col for col in product_stats.columns if col != 'internal_product_id'])
        
        # Product performance metrics
        product_performance = df.groupby('internal_product_id').agg({
            self.value_col: ['sum', 'mean'],
            'net_value': ['sum', 'mean'] if 'net_value' in df.columns else ['sum', 'mean'],
            'gross_value': ['sum', 'mean'] if 'gross_value' in df.columns else ['sum', 'mean']
        })
        
        # Flatten and rename
        performance_cols = {}
        for col in product_performance.columns:
            new_name = f'product_{col[0]}_{col[1]}'
            performance_cols[col] = new_name
            product_features.append(new_name)
        
        product_performance.columns = [performance_cols[col] for col in product_performance.columns]
        product_performance = product_performance.reset_index()
        
        # Calculate derived metrics
        if 'net_value' in df.columns:
            product_performance['product_avg_unit_price'] = (
                product_performance['product_net_value_sum'] / 
                (product_performance['product_quantity_sum'] + 1e-8)
            )
            product_performance['product_revenue_per_transaction'] = (
                product_performance['product_net_value_sum'] / 
                (product_stats['product_transaction_date_nunique'] + 1e-8)
            )
            product_features.extend(['product_avg_unit_price', 'product_revenue_per_transaction'])
        
        # Product velocity metrics (based on EDA ABC analysis)
        product_aggs['product_velocity_score'] = (
            product_aggs['product_quantity_sum'] * 
            product_stats['product_internal_store_id_nunique']
        )
        product_features.append('product_velocity_score')
        
        # Product consistency metrics
        product_aggs['product_consistency'] = (
            product_aggs['product_quantity_mean'] / 
            (product_aggs['product_quantity_std'] + 1e-8)
        )
        product_features.append('product_consistency')
        
        # Product market penetration
        total_stores = df['internal_store_id'].nunique()
        product_stats['product_market_penetration'] = (
            product_stats['product_internal_store_id_nunique'] / total_stores
        )
        product_features.append('product_market_penetration')
        
        # Merge all product features
        product_all = product_aggs.merge(product_stats, on='internal_product_id', how='outer')
        product_all = product_all.merge(product_performance, on='internal_product_id', how='outer')
        
        # Merge back to main dataframe
        df = df.merge(product_all, on='internal_product_id', how='left')
        
        self.features_created.extend(product_features)
        print(f"[OK] Created {len(product_features)} product-level features")
        
        return df
    
    def create_store_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive PDV/store-level aggregation features"""
        
        df = df.copy()
        
        print("[INFO] Creating store-level aggregation features...")
        
        store_features = []
        
        # Basic store aggregations
        store_aggs = df.groupby('internal_store_id')[self.value_col].agg(
            self.agg_functions
        ).reset_index()
        
        # Rename columns with prefix
        agg_cols = {}
        for func in self.agg_functions:
            new_col = f'store_{self.value_col}_{func}'
            agg_cols[func] = new_col
            store_features.append(new_col)
        
        store_aggs = store_aggs.rename(columns=agg_cols)
        
        # Store diversity metrics (portfolio analysis)
        store_diversity = df.groupby('internal_store_id').agg({
            'internal_product_id': 'nunique',
            self.value_col: ['sum', 'count']
        })
        
        # Flatten columns
        store_diversity.columns = [
            f'store_{col[0]}_{col[1]}' for col in store_diversity.columns
        ]
        store_diversity = store_diversity.reset_index()
        
        # Calculate portfolio diversity (entropy)
        store_portfolio = df.groupby(['internal_store_id', 'internal_product_id'])[self.value_col].sum().reset_index()
        store_entropy = store_portfolio.groupby('internal_store_id').apply(
            lambda x: entropy(x[self.value_col] + 1e-8)
        ).reset_index()
        store_entropy.columns = ['internal_store_id', 'store_portfolio_entropy']
        
        store_features.extend(['store_portfolio_entropy'])
        
        # Store performance metrics
        total_volume = df[self.value_col].sum()
        store_aggs['store_market_share'] = store_aggs[f'store_{self.value_col}_sum'] / total_volume
        store_features.append('store_market_share')
        
        # Store growth trajectory (if temporal data available)
        if 'transaction_date' in df.columns:
            df_sorted = df.sort_values(['internal_store_id', 'transaction_date'])
            
            # Monthly aggregation for trend
            df_sorted['year_month'] = df_sorted['transaction_date'].dt.to_period('M')
            monthly_store = df_sorted.groupby(['internal_store_id', 'year_month'])[self.value_col].sum().reset_index()
            
            # Calculate trend (slope) for each store
            store_trends = []
            for store_id in monthly_store['internal_store_id'].unique():
                store_data = monthly_store[monthly_store['internal_store_id'] == store_id].copy()
                if len(store_data) >= 3:
                    x = np.arange(len(store_data))
                    y = store_data[self.value_col].values
                    trend = np.polyfit(x, y, 1)[0]
                else:
                    trend = 0
                store_trends.append({'internal_store_id': store_id, 'store_trend': trend})
            
            store_trend_df = pd.DataFrame(store_trends)
            store_features.append('store_trend')
        else:
            store_trend_df = pd.DataFrame({'internal_store_id': df['internal_store_id'].unique(), 'store_trend': 0})
        
        # Top products analysis per store
        top_products_per_store = df.groupby('internal_store_id').apply(
            lambda x: x.nlargest(5, self.value_col)[self.value_col].sum() / x[self.value_col].sum()
        ).reset_index()
        top_products_per_store.columns = ['internal_store_id', 'store_top5_products_share']
        store_features.append('store_top5_products_share')
        
        # Store category focus
        if 'categoria' in df.columns:
            store_category_focus = df.groupby('internal_store_id')['categoria'].apply(
                lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
            ).reset_index()
            store_category_focus.columns = ['internal_store_id', 'store_category_focus']
            store_features.append('store_category_focus')
        else:
            store_category_focus = pd.DataFrame({'internal_store_id': df['internal_store_id'].unique(), 'store_category_focus': 0})
        
        # Merge all store features
        store_all = store_aggs.merge(store_diversity, on='internal_store_id', how='outer')
        store_all = store_all.merge(store_entropy, on='internal_store_id', how='outer')
        store_all = store_all.merge(store_trend_df, on='internal_store_id', how='outer')
        store_all = store_all.merge(top_products_per_store, on='internal_store_id', how='outer')
        store_all = store_all.merge(store_category_focus, on='internal_store_id', how='outer')
        
        # Merge back to main dataframe
        df = df.merge(store_all, on='internal_store_id', how='left')
        
        self.features_created.extend(store_features)
        print(f"[OK] Created {len(store_features)} store-level features")
        
        return df
    
    def create_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-dimensional features (product×store, category×region, etc.)"""
        
        df = df.copy()
        
        print("[INFO] Creating cross-dimensional features...")
        
        cross_features = []
        
        # Product×Store affinity
        # How well does this product perform in this specific store vs average?
        product_store_agg = df.groupby(['internal_product_id', 'internal_store_id'])[self.value_col].sum().reset_index()
        product_store_agg.columns = ['internal_product_id', 'internal_store_id', 'product_store_volume']
        
        # Product average across all stores
        product_avg = df.groupby('internal_product_id')[self.value_col].mean().reset_index()
        product_avg.columns = ['internal_product_id', 'product_avg_volume']
        
        # Store average across all products  
        store_avg = df.groupby('internal_store_id')[self.value_col].mean().reset_index()
        store_avg.columns = ['internal_store_id', 'store_avg_volume']
        
        # Merge to calculate affinity
        cross_df = product_store_agg.merge(product_avg, on='internal_product_id', how='left')
        cross_df = cross_df.merge(store_avg, on='internal_store_id', how='left')
        
        # Product-store affinity score
        cross_df['product_store_affinity'] = (
            cross_df['product_store_volume'] / 
            ((cross_df['product_avg_volume'] + cross_df['store_avg_volume']) / 2 + 1e-8)
        )
        cross_features.append('product_store_affinity')
        
        # Regional analysis if zipcode available
        if 'zipcode' in df.columns:
            # Category×Region analysis
            if 'categoria' in df.columns:
                category_region = df.groupby(['categoria', 'zipcode'])[self.value_col].sum().reset_index()
                category_region['category_region_key'] = category_region['categoria'].astype(str) + '_' + category_region['zipcode'].astype(str)
                
                # Calculate category share in each region
                region_totals = df.groupby('zipcode')[self.value_col].sum().reset_index()
                region_totals.columns = ['zipcode', 'region_total_volume']
                
                category_region = category_region.merge(region_totals, on='zipcode', how='left')
                category_region['category_region_share'] = (
                    category_region[self.value_col] / category_region['region_total_volume']
                )
                
                # Merge back to main df
                df = df.merge(
                    category_region[['categoria', 'zipcode', 'category_region_share']], 
                    on=['categoria', 'zipcode'], 
                    how='left'
                )
                cross_features.append('category_region_share')
        
        # Seasonal×Product interaction (if date available)
        if 'transaction_date' in df.columns:
            df['month'] = df['transaction_date'].dt.month
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            
            # Product seasonal alignment
            product_seasonal = df.groupby(['internal_product_id', 'month'])[self.value_col].sum().reset_index()
            
            # Calculate seasonality strength per product
            product_seasonality = product_seasonal.groupby('internal_product_id')[self.value_col].apply(
                lambda x: (x.max() - x.min()) / (x.mean() + 1e-8)
            ).reset_index()
            product_seasonality.columns = ['internal_product_id', 'product_seasonality_strength']
            
            df = df.merge(product_seasonality, on='internal_product_id', how='left')
            cross_features.append('product_seasonality_strength')
        
        # Volume tier interactions
        # Create volume percentiles for products and stores
        df['product_volume_percentile'] = df.groupby('internal_product_id')[self.value_col].transform(
            lambda x: x.rank(pct=True)
        )
        df['store_volume_percentile'] = df.groupby('internal_store_id')[self.value_col].transform(
            lambda x: x.rank(pct=True)
        )
        
        # High-volume combination indicator
        df['high_volume_combination'] = (
            (df['product_volume_percentile'] > 0.8) & 
            (df['store_volume_percentile'] > 0.8)
        ).astype(int)
        
        cross_features.extend([
            'product_volume_percentile', 
            'store_volume_percentile', 
            'high_volume_combination'
        ])
        
        # Merge cross features back
        cross_features_df = cross_df[['internal_product_id', 'internal_store_id', 'product_store_affinity']]
        df = df.merge(cross_features_df, on=['internal_product_id', 'internal_store_id'], how='left')
        
        self.features_created.extend(cross_features)
        print(f"[OK] Created {len(cross_features)} cross-dimensional features")
        
        return df
    
    def create_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create category-level aggregation features"""
        
        df = df.copy()
        
        if 'categoria' not in df.columns:
            print("[WARNING] Category column not found, skipping category features")
            return df
            
        print("[INFO] Creating category-level features...")
        
        category_features = []
        
        # Category aggregations
        category_aggs = df.groupby('categoria')[self.value_col].agg(
            ['sum', 'mean', 'count', 'std']
        ).reset_index()
        category_aggs.columns = ['categoria'] + [f'category_{self.value_col}_{col}' for col in ['sum', 'mean', 'count', 'std']]
        
        # Category market share
        total_market = df[self.value_col].sum()
        category_aggs['category_market_share'] = category_aggs[f'category_{self.value_col}_sum'] / total_market
        
        # Category concentration (number of products)
        category_products = df.groupby('categoria')['internal_product_id'].nunique().reset_index()
        category_products.columns = ['categoria', 'category_num_products']
        category_aggs = category_aggs.merge(category_products, on='categoria', how='left')
        
        # Category performance per product
        category_aggs['category_avg_performance_per_product'] = (
            category_aggs[f'category_{self.value_col}_sum'] / category_aggs['category_num_products']
        )
        
        category_features.extend([
            f'category_{self.value_col}_sum', f'category_{self.value_col}_mean', 
            f'category_{self.value_col}_count', f'category_{self.value_col}_std',
            'category_market_share', 'category_num_products', 'category_avg_performance_per_product'
        ])
        
        # Merge back
        df = df.merge(category_aggs, on='categoria', how='left')
        
        self.features_created.extend(category_features)
        print(f"[OK] Created {len(category_features)} category features")
        
        return df
    
    def create_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical aggregation features"""
        
        df = df.copy()
        
        print("[INFO] Creating hierarchical features...")
        
        hierarchical_features = []
        
        # Product hierarchy features
        hierarchy_cols = [col for col in ['categoria', 'subcategoria', 'marca', 'fabricante'] if col in df.columns]
        
        for hier_col in hierarchy_cols:
            # Aggregations by hierarchy level
            hier_aggs = df.groupby(hier_col)[self.value_col].agg(['sum', 'mean', 'count']).reset_index()
            hier_aggs.columns = [hier_col] + [f'{hier_col}_{self.value_col}_{col}' for col in ['sum', 'mean', 'count']]
            
            # Hierarchy performance metrics
            total_value = df[self.value_col].sum()
            hier_aggs[f'{hier_col}_share'] = hier_aggs[f'{hier_col}_{self.value_col}_sum'] / total_value
            
            hierarchical_features.extend([
                f'{hier_col}_{self.value_col}_sum',
                f'{hier_col}_{self.value_col}_mean', 
                f'{hier_col}_{self.value_col}_count',
                f'{hier_col}_share'
            ])
            
            # Merge back
            df = df.merge(hier_aggs, on=hier_col, how='left')
        
        # Cross-hierarchy features
        if len(hierarchy_cols) >= 2:
            # Brand×Category interaction
            if 'marca' in hierarchy_cols and 'categoria' in hierarchy_cols:
                brand_category = df.groupby(['marca', 'categoria'])[self.value_col].sum().reset_index()
                brand_category['brand_category_key'] = brand_category['marca'].astype(str) + '_' + brand_category['categoria'].astype(str)
                
                # Calculate brand dominance in category
                category_totals = df.groupby('categoria')[self.value_col].sum().reset_index()
                category_totals.columns = ['categoria', 'categoria_total']
                
                brand_category = brand_category.merge(category_totals, on='categoria', how='left')
                brand_category['brand_category_dominance'] = (
                    brand_category[self.value_col] / brand_category['categoria_total']
                )
                
                df = df.merge(
                    brand_category[['marca', 'categoria', 'brand_category_dominance']], 
                    on=['marca', 'categoria'], 
                    how='left'
                )
                hierarchical_features.append('brand_category_dominance')
        
        self.features_created.extend(hierarchical_features)
        print(f"[OK] Created {len(hierarchical_features)} hierarchical features")
        
        return df
    
    def create_wmape_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-weighted aggregations optimized for WMAPE"""
        
        df = df.copy()
        
        print("[INFO] Creating WMAPE-weighted aggregation features...")
        
        wmape_features = []
        
        # Volume-weighted means (critical for WMAPE optimization)
        total_volume = df[self.value_col].sum()
        
        # Product volume weights
        product_volumes = df.groupby('internal_product_id')[self.value_col].sum().reset_index()
        product_volumes['product_volume_weight'] = product_volumes[self.value_col] / total_volume
        df = df.merge(product_volumes[['internal_product_id', 'product_volume_weight']], on='internal_product_id', how='left')
        
        # Store volume weights
        store_volumes = df.groupby('internal_store_id')[self.value_col].sum().reset_index()
        store_volumes['store_volume_weight'] = store_volumes[self.value_col] / total_volume
        df = df.merge(store_volumes[['internal_store_id', 'store_volume_weight']], on='internal_store_id', how='left')
        
        # Combined volume impact
        df['combined_volume_weight'] = df['product_volume_weight'] * df['store_volume_weight']
        
        # Volume tier classification (matching EDA insights)
        df['product_volume_tier'] = pd.cut(
            df['product_volume_weight'], 
            bins=[0, 0.001, 0.01, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        df['store_volume_tier'] = pd.cut(
            df['store_volume_weight'], 
            bins=[0, 0.001, 0.01, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Forecast importance score (higher for high-volume combinations)
        df['forecast_importance'] = (
            df['product_volume_weight'] * df['store_volume_weight'] * 1000
        )
        
        wmape_features.extend([
            'product_volume_weight', 'store_volume_weight', 'combined_volume_weight',
            'forecast_importance'
        ])
        
        # ABC Tier aggregations (based on EDA Tier A = 5.3% products, 79.9% volume)
        if 'product_volume_tier' in df.columns:
            tier_aggs = df.groupby('product_volume_tier')[self.value_col].agg(['sum', 'count', 'mean']).reset_index()
            tier_aggs.columns = ['product_volume_tier'] + [f'tier_{self.value_col}_{col}' for col in ['sum', 'count', 'mean']]
            
            df = df.merge(tier_aggs, on='product_volume_tier', how='left')
            wmape_features.extend([f'tier_{self.value_col}_sum', f'tier_{self.value_col}_count', f'tier_{self.value_col}_mean'])
        
        self.features_created.extend(wmape_features)
        print(f"[OK] Created {len(wmape_features)} WMAPE-weighted features")
        
        return df
    
    def create_all_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all aggregation features in optimized sequence"""
        
        print("\n" + "="*80)
        print("AGGREGATION FEATURES ENGINE - CREATING ALL FEATURES")
        print("="*80)
        
        initial_shape = df.shape
        print(f"[START] Initial dataset: {initial_shape}")
        
        # Sequence of feature creation (optimized order)
        df = self.create_product_level_features(df)
        df = self.create_store_level_features(df)
        df = self.create_cross_features(df)
        df = self.create_category_features(df)
        df = self.create_hierarchical_features(df)
        df = self.create_wmape_weighted_features(df)
        
        final_shape = df.shape
        features_added = final_shape[1] - initial_shape[1]
        
        print(f"\n[SUMMARY] Aggregation Features Engine completed:")
        print(f"  Features added: {features_added}")
        print(f"  Total features created: {len(self.features_created)}")
        print(f"  Final dataset shape: {final_shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def get_feature_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights about created aggregation features"""
        
        insights = {
            'total_features': len(self.features_created),
            'feature_categories': {
                'product_features': len([f for f in self.features_created if f.startswith('product_')]),
                'store_features': len([f for f in self.features_created if f.startswith('store_')]),
                'cross_features': len([f for f in self.features_created if any(x in f for x in ['affinity', 'combination', 'percentile'])]),
                'category_features': len([f for f in self.features_created if f.startswith('category_') or f.startswith('categoria_')]),
                'hierarchical_features': len([f for f in self.features_created if any(x in f for x in ['marca_', 'subcategoria_', 'fabricante_'])]),
                'wmape_features': len([f for f in self.features_created if any(x in f for x in ['weight', 'tier', 'importance'])])
            },
            'key_insights': [
                f"Product aggregations: {len([f for f in self.features_created if f.startswith('product_')])} features",
                f"Store aggregations: {len([f for f in self.features_created if f.startswith('store_')])} features",
                f"Cross-dimensional features for product×store interactions",
                f"Volume-weighted features optimized for WMAPE",
                f"ABC tier features based on EDA insights",
                f"Hierarchical features across brand/category dimensions"
            ],
            'features_list': self.features_created
        }
        
        return insights

def main():
    """Demonstration and testing of Aggregation Features Engine"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("AGGREGATION FEATURES ENGINE - DEMONSTRATION")
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
        
        # Merge with product data for category features
        if 'produto' in prod_df.columns:
            # Map product IDs
            prod_mapping = dict(zip(range(len(prod_df)), prod_df['produto'].values))
            trans_df['produto'] = trans_df['internal_product_id'].map(
                lambda x: prod_mapping.get(hash(str(x)) % len(prod_df), 'Unknown')
            )
            
            # Add category info
            if 'categoria' in prod_df.columns:
                prod_cat_mapping = dict(zip(prod_df['produto'], prod_df['categoria']))
                trans_df['categoria'] = trans_df['produto'].map(prod_cat_mapping)
        
        print(f"Sample data loaded: {trans_df.shape}")
        print(f"Columns: {list(trans_df.columns)}")
        
        # Initialize engine
        engine = AggregationFeaturesEngine()
        
        # Create all aggregation features
        features_df = engine.create_all_aggregation_features(trans_df)
        
        # Get insights
        insights = engine.get_feature_insights(features_df)
        
        print("\n" + "="*80)
        print("AGGREGATION FEATURES INSIGHTS")
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
        
        features_file = output_dir / "aggregation_features_demo.parquet"
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