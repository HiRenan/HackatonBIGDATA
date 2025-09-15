#!/usr/bin/env python3
"""
DATA QUALITY ANALYSIS - Hackathon Forecast Big Data 2025
Phase 2.3: Análise de Qualidade e Consistência de Dados

Objetivos:
- Identificar missing data patterns (sistemático vs aleatório)
- Outlier detection com múltiplos métodos (IQR, Z-score, Isolation Forest)
- Detectar data drift (mudanças de distribuição temporal)
- Consistency checks (integridade referencial)
- Coverage analysis (completude por dimensão)

Estratégia: Análise sistemática para garantir qualidade dos insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.data_loader import OptimizedDataLoader, load_data_efficiently
from evaluation.metrics import wmape

# Advanced analytics libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("[OK] Advanced quality analysis libraries loaded")
except ImportError as e:
    print(f"[WARNING] Some advanced libraries missing: {e}")

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataQualityAnalyzer:
    """Comprehensive data quality analysis for retail forecasting data"""
    
    def __init__(self, data_path: str = "../../data/raw"):
        self.data_path = data_path
        self.loader = OptimizedDataLoader(data_path)
        self.results = {}
        
    def load_quality_analysis_data(self, sample_transactions: int = 1500000, sample_products: int = 150000):
        """Load datasets for quality analysis"""
        
        print("="*60)
        print("LOADING DATA FOR QUALITY ANALYSIS")
        print("="*60)
        
        # Load all three datasets
        self.trans_df, self.prod_df, self.pdv_df = load_data_efficiently(
            data_path=self.data_path,
            sample_transactions=sample_transactions,
            sample_products=sample_products
        )
        
        print(f"\nDatasets loaded for quality analysis:")
        print(f"  Transactions: {len(self.trans_df):,} records")
        print(f"  Products: {len(self.prod_df):,} records") 
        print(f"  PDVs: {len(self.pdv_df):,} records")
        
        return self.trans_df, self.prod_df, self.pdv_df
    
    def analyze_missing_data_patterns(self):
        """Comprehensive missing data analysis"""
        
        print("\n" + "="*60)
        print("MISSING DATA PATTERN ANALYSIS")
        print("="*60)
        
        missing_analysis = {}
        
        # 1. Transaction Data Missing Patterns
        print("\n1. TRANSACTION DATA MISSING PATTERNS")
        print("-" * 40)
        
        trans_missing = self.trans_df.isnull().sum()
        trans_total = len(self.trans_df)
        
        trans_missing_pct = (trans_missing / trans_total * 100).round(2)
        
        print("Missing data in transaction dataset:")
        for col, count in trans_missing.items():
            pct = trans_missing_pct[col]
            if count > 0:
                print(f"  {col}: {count:,} ({pct}%)")
            else:
                print(f"  {col}: No missing values [OK]")
        
        # Check for systematic missing patterns
        if 'transaction_date' in self.trans_df.columns:
            self.trans_df['transaction_date'] = pd.to_datetime(self.trans_df['transaction_date'])
            
            # Missing by time period
            monthly_missing = self.trans_df.groupby(self.trans_df['transaction_date'].dt.month).apply(
                lambda x: x.isnull().sum().sum()
            )
            
            if monthly_missing.sum() > 0:
                print(f"\nMissing data by month (total missing values):")
                for month, missing_count in monthly_missing.items():
                    if missing_count > 0:
                        print(f"  Month {month}: {missing_count} missing values")
        
        missing_analysis['transactions'] = {
            'total_records': trans_total,
            'missing_by_column': trans_missing.to_dict(),
            'missing_percentage': trans_missing_pct.to_dict(),
            'systematic_patterns': monthly_missing.to_dict() if 'monthly_missing' in locals() else {}
        }
        
        # 2. Product Data Missing Patterns
        print("\n2. PRODUCT DATA MISSING PATTERNS")
        print("-" * 40)
        
        prod_missing = self.prod_df.isnull().sum()
        prod_total = len(self.prod_df)
        prod_missing_pct = (prod_missing / prod_total * 100).round(2)
        
        print("Missing data in product dataset:")
        for col, count in prod_missing.items():
            pct = prod_missing_pct[col]
            if count > 0:
                print(f"  {col}: {count:,} ({pct}%)")
            else:
                print(f"  {col}: No missing values [OK]")
        
        missing_analysis['products'] = {
            'total_records': prod_total,
            'missing_by_column': prod_missing.to_dict(),
            'missing_percentage': prod_missing_pct.to_dict()
        }
        
        # 3. PDV Data Missing Patterns
        print("\n3. PDV DATA MISSING PATTERNS")
        print("-" * 40)
        
        pdv_missing = self.pdv_df.isnull().sum()
        pdv_total = len(self.pdv_df)
        pdv_missing_pct = (pdv_missing / pdv_total * 100).round(2)
        
        print("Missing data in PDV dataset:")
        for col, count in pdv_missing.items():
            pct = pdv_missing_pct[col]
            if count > 0:
                print(f"  {col}: {count:,} ({pct}%)")
            else:
                print(f"  {col}: No missing values [OK]")
        
        missing_analysis['pdvs'] = {
            'total_records': pdv_total,
            'missing_by_column': pdv_missing.to_dict(),
            'missing_percentage': pdv_missing_pct.to_dict()
        }
        
        self.results['missing_data_analysis'] = missing_analysis
        return missing_analysis
    
    def detect_outliers_multiple_methods(self):
        """Outlier detection using multiple methods"""
        
        print("\n" + "="*60)
        print("MULTI-METHOD OUTLIER DETECTION")
        print("="*60)
        
        outlier_results = {}
        
        # Focus on numerical columns in transactions
        numerical_cols = ['quantity', 'gross_value', 'net_value']
        numerical_cols = [col for col in numerical_cols if col in self.trans_df.columns]
        
        print(f"Analyzing outliers in columns: {numerical_cols}")
        
        for col in numerical_cols:
            print(f"\n--- {col.upper()} OUTLIER ANALYSIS ---")
            
            col_data = self.trans_df[col].dropna()
            if len(col_data) == 0:
                print(f"No data available for {col}")
                continue
            
            col_outliers = {}
            
            # Method 1: IQR Method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            iqr_outlier_count = len(iqr_outliers)
            iqr_outlier_pct = (iqr_outlier_count / len(col_data)) * 100
            
            print(f"IQR Method: {iqr_outlier_count:,} outliers ({iqr_outlier_pct:.2f}%)")
            
            col_outliers['iqr'] = {
                'count': iqr_outlier_count,
                'percentage': iqr_outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'extreme_values': iqr_outliers.nlargest(5).tolist()
            }
            
            # Method 2: Z-Score Method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = col_data[z_scores > 3]  # 3 standard deviations
            z_outlier_count = len(z_outliers)
            z_outlier_pct = (z_outlier_count / len(col_data)) * 100
            
            print(f"Z-Score Method (>3 sigma): {z_outlier_count:,} outliers ({z_outlier_pct:.2f}%)")
            
            col_outliers['zscore'] = {
                'count': z_outlier_count,
                'percentage': z_outlier_pct,
                'threshold': 3,
                'extreme_values': z_outliers.nlargest(5).tolist()
            }
            
            # Method 3: Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                iso_outliers = col_data[outlier_labels == -1]
                iso_outlier_count = len(iso_outliers)
                iso_outlier_pct = (iso_outlier_count / len(col_data)) * 100
                
                print(f"Isolation Forest: {iso_outlier_count:,} outliers ({iso_outlier_pct:.2f}%)")
                
                col_outliers['isolation_forest'] = {
                    'count': iso_outlier_count,
                    'percentage': iso_outlier_pct,
                    'contamination': 0.1,
                    'extreme_values': iso_outliers.nlargest(5).tolist()
                }
                
            except Exception as e:
                print(f"Isolation Forest failed: {e}")
                col_outliers['isolation_forest'] = {'error': str(e)}
            
            # Summary statistics
            print(f"Column statistics:")
            print(f"  Mean: {col_data.mean():.2f}")
            print(f"  Std: {col_data.std():.2f}")
            print(f"  Min: {col_data.min():.2f}")
            print(f"  Max: {col_data.max():.2f}")
            
            outlier_results[col] = col_outliers
        
        self.results['outlier_analysis'] = outlier_results
        return outlier_results
    
    def analyze_data_drift(self):
        """Detect data drift over time"""
        
        print("\n" + "="*60)
        print("DATA DRIFT DETECTION")
        print("="*60)
        
        if 'transaction_date' not in self.trans_df.columns:
            print("No date column found - skipping drift analysis")
            return {'error': 'No date column available'}
        
        # Convert to datetime if not already
        self.trans_df['transaction_date'] = pd.to_datetime(self.trans_df['transaction_date'])
        
        # Create monthly aggregations
        monthly_stats = self.trans_df.groupby(self.trans_df['transaction_date'].dt.to_period('M')).agg({
            'quantity': ['mean', 'std', 'count'],
            'gross_value': 'mean' if 'gross_value' in self.trans_df.columns else 'count',
            'internal_product_id': 'nunique',
            'internal_store_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        monthly_stats.columns = [
            'month', 'quantity_mean', 'quantity_std', 'transaction_count',
            'avg_gross_value', 'unique_products', 'unique_stores'
        ]
        
        print(f"Analyzing drift across {len(monthly_stats)} months")
        
        drift_analysis = {}
        
        # 1. Statistical drift detection
        print("\n1. STATISTICAL DRIFT ANALYSIS")
        print("-" * 30)
        
        # Calculate month-to-month changes
        drift_metrics = ['quantity_mean', 'avg_gross_value', 'unique_products']
        
        for metric in drift_metrics:
            if metric in monthly_stats.columns:
                metric_values = monthly_stats[metric].dropna()
                
                if len(metric_values) > 1:
                    # Calculate percentage changes
                    pct_changes = metric_values.pct_change().dropna()
                    
                    # Identify significant changes (>20%)
                    significant_changes = pct_changes[abs(pct_changes) > 0.2]
                    
                    print(f"{metric}:")
                    print(f"  Avg monthly change: {pct_changes.mean():.1%}")
                    print(f"  Max monthly change: {pct_changes.max():.1%}")
                    print(f"  Significant changes: {len(significant_changes)}")
                    
                    drift_analysis[metric] = {
                        'avg_change': float(pct_changes.mean()),
                        'max_change': float(pct_changes.max()),
                        'min_change': float(pct_changes.min()),
                        'significant_changes': len(significant_changes),
                        'stability_score': 1 - abs(pct_changes.mean())  # Closer to 1 = more stable
                    }
        
        # 2. Distribution drift detection
        print("\n2. DISTRIBUTION DRIFT ANALYSIS")
        print("-" * 30)
        
        # Split data into first half and second half
        mid_date = self.trans_df['transaction_date'].quantile(0.5)
        first_half = self.trans_df[self.trans_df['transaction_date'] <= mid_date]
        second_half = self.trans_df[self.trans_df['transaction_date'] > mid_date]
        
        print(f"Comparing distributions:")
        print(f"  First half: {len(first_half):,} records")
        print(f"  Second half: {len(second_half):,} records")
        
        distribution_drift = {}
        
        for col in ['quantity', 'gross_value']:
            if col in self.trans_df.columns:
                try:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        first_half[col].dropna(),
                        second_half[col].dropna()
                    )
                    
                    print(f"{col}:")
                    print(f"  KS statistic: {ks_stat:.4f}")
                    print(f"  p-value: {ks_pvalue:.4f}")
                    print(f"  Drift detected: {'Yes' if ks_pvalue < 0.05 else 'No'}")
                    
                    distribution_drift[col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(ks_pvalue),
                        'drift_detected': ks_pvalue < 0.05,
                        'first_half_mean': float(first_half[col].mean()),
                        'second_half_mean': float(second_half[col].mean())
                    }
                    
                except Exception as e:
                    print(f"  Error analyzing {col}: {e}")
                    distribution_drift[col] = {'error': str(e)}
        
        drift_results = {
            'monthly_drift': drift_analysis,
            'distribution_drift': distribution_drift,
            'time_period': {
                'start_date': str(self.trans_df['transaction_date'].min().date()),
                'end_date': str(self.trans_df['transaction_date'].max().date()),
                'n_months': len(monthly_stats)
            }
        }
        
        self.results['drift_analysis'] = drift_results
        return drift_results
    
    def perform_consistency_checks(self):
        """Referential integrity and consistency checks"""
        
        print("\n" + "="*60)
        print("DATA CONSISTENCY CHECKS")
        print("="*60)
        
        consistency_results = {}
        
        # 1. Referential Integrity
        print("\n1. REFERENTIAL INTEGRITY CHECKS")
        print("-" * 35)
        
        # Check if all products in transactions exist in product catalog
        trans_products = set(self.trans_df['internal_product_id'].unique())
        catalog_products = set(self.prod_df.iloc[:, 0].unique())  # First column assumed to be product ID
        
        missing_products = trans_products - catalog_products
        orphaned_products = catalog_products - trans_products
        
        print(f"Product referential integrity:")
        print(f"  Products in transactions: {len(trans_products):,}")
        print(f"  Products in catalog: {len(catalog_products):,}")
        print(f"  Missing from catalog: {len(missing_products):,}")
        print(f"  Orphaned in catalog: {len(orphaned_products):,}")
        
        # Check if all stores in transactions exist in PDV catalog
        trans_stores = set(self.trans_df['internal_store_id'].unique())
        catalog_stores = set(self.pdv_df.iloc[:, 0].unique())  # First column assumed to be store ID
        
        missing_stores = trans_stores - catalog_stores
        orphaned_stores = catalog_stores - trans_stores
        
        print(f"\nStore referential integrity:")
        print(f"  Stores in transactions: {len(trans_stores):,}")
        print(f"  Stores in catalog: {len(catalog_stores):,}")
        print(f"  Missing from catalog: {len(missing_stores):,}")
        print(f"  Orphaned in catalog: {len(orphaned_stores):,}")
        
        consistency_results['referential_integrity'] = {
            'products': {
                'transactions_count': len(trans_products),
                'catalog_count': len(catalog_products),
                'missing_from_catalog': len(missing_products),
                'orphaned_in_catalog': len(orphaned_products),
                'integrity_score': 1 - (len(missing_products) / len(trans_products))
            },
            'stores': {
                'transactions_count': len(trans_stores),
                'catalog_count': len(catalog_stores),
                'missing_from_catalog': len(missing_stores),
                'orphaned_in_catalog': len(orphaned_stores),
                'integrity_score': 1 - (len(missing_stores) / len(trans_stores))
            }
        }
        
        # 2. Logical Consistency
        print("\n2. LOGICAL CONSISTENCY CHECKS")
        print("-" * 32)
        
        logical_issues = {}
        
        # Check for negative quantities
        if 'quantity' in self.trans_df.columns:
            negative_qty = (self.trans_df['quantity'] < 0).sum()
            zero_qty = (self.trans_df['quantity'] == 0).sum()
            
            print(f"Quantity consistency:")
            print(f"  Negative quantities: {negative_qty:,}")
            print(f"  Zero quantities: {zero_qty:,}")
            
            logical_issues['quantity'] = {
                'negative_count': int(negative_qty),
                'zero_count': int(zero_qty),
                'negative_percentage': float(negative_qty / len(self.trans_df) * 100)
            }
        
        # Check for gross_value vs net_value consistency
        if 'gross_value' in self.trans_df.columns and 'net_value' in self.trans_df.columns:
            inconsistent_values = (self.trans_df['net_value'] > self.trans_df['gross_value']).sum()
            
            print(f"Value consistency:")
            print(f"  Net > Gross violations: {inconsistent_values:,}")
            
            logical_issues['values'] = {
                'net_greater_than_gross': int(inconsistent_values),
                'percentage': float(inconsistent_values / len(self.trans_df) * 100)
            }
        
        # Check date consistency
        if 'transaction_date' in self.trans_df.columns:
            future_dates = (self.trans_df['transaction_date'] > datetime.now()).sum()
            
            print(f"Date consistency:")
            print(f"  Future dates: {future_dates:,}")
            
            logical_issues['dates'] = {
                'future_dates': int(future_dates),
                'percentage': float(future_dates / len(self.trans_df) * 100)
            }
        
        consistency_results['logical_consistency'] = logical_issues
        
        self.results['consistency_checks'] = consistency_results
        return consistency_results
    
    def analyze_coverage_completeness(self):
        """Coverage and completeness analysis"""
        
        print("\n" + "="*60)
        print("COVERAGE & COMPLETENESS ANALYSIS")
        print("="*60)
        
        coverage_results = {}
        
        # 1. Temporal Coverage
        print("\n1. TEMPORAL COVERAGE ANALYSIS")
        print("-" * 32)
        
        if 'transaction_date' in self.trans_df.columns:
            date_range = pd.date_range(
                start=self.trans_df['transaction_date'].min(),
                end=self.trans_df['transaction_date'].max(),
                freq='D'
            )
            
            actual_dates = set(self.trans_df['transaction_date'].dt.date)
            expected_dates = set(date_range.date)
            missing_dates = expected_dates - actual_dates
            
            print(f"Temporal coverage:")
            print(f"  Date range: {len(date_range)} days")
            print(f"  Days with data: {len(actual_dates)}")
            print(f"  Missing days: {len(missing_dates)}")
            print(f"  Coverage: {len(actual_dates)/len(date_range)*100:.1f}%")
            
            coverage_results['temporal'] = {
                'total_days': len(date_range),
                'days_with_data': len(actual_dates),
                'missing_days': len(missing_dates),
                'coverage_percentage': float(len(actual_dates) / len(date_range) * 100)
            }
        
        # 2. Product Coverage
        print("\n2. PRODUCT COVERAGE ANALYSIS")
        print("-" * 32)
        
        # Products by category coverage
        if 'categoria' in self.prod_df.columns:
            category_coverage = self.prod_df.groupby('categoria').agg({
                self.prod_df.columns[0]: 'count'  # Count products per category
            }).reset_index()
            
            category_coverage.columns = ['category', 'product_count']
            category_coverage = category_coverage.sort_values('product_count', ascending=False)
            
            print(f"Product coverage by category:")
            for _, row in category_coverage.head(10).iterrows():
                print(f"  {row['category']}: {row['product_count']:,} products")
            
            coverage_results['product_categories'] = category_coverage.to_dict('records')
        
        # 3. Store Coverage
        print("\n3. STORE COVERAGE ANALYSIS")
        print("-" * 28)
        
        # Stores by region coverage
        if 'zipcode' in self.pdv_df.columns:
            # Extract first 2 digits as region
            self.pdv_df['region'] = self.pdv_df['zipcode'].astype(str).str[:2]
            
            region_coverage = self.pdv_df.groupby('region').agg({
                self.pdv_df.columns[0]: 'count'
            }).reset_index()
            
            region_coverage.columns = ['region', 'store_count']
            region_coverage = region_coverage.sort_values('store_count', ascending=False)
            
            print(f"Store coverage by region:")
            for _, row in region_coverage.head(10).iterrows():
                print(f"  Region {row['region']}: {row['store_count']:,} stores")
            
            coverage_results['store_regions'] = region_coverage.to_dict('records')
        
        # 4. Transaction Coverage
        print("\n4. TRANSACTION COVERAGE ANALYSIS")
        print("-" * 35)
        
        # Product-Store combination coverage
        total_combinations = len(self.trans_df['internal_product_id'].unique()) * len(self.trans_df['internal_store_id'].unique())
        actual_combinations = len(self.trans_df.groupby(['internal_product_id', 'internal_store_id']).size())
        
        combination_coverage = actual_combinations / total_combinations * 100
        
        print(f"Product-Store combinations:")
        print(f"  Possible combinations: {total_combinations:,}")
        print(f"  Actual combinations: {actual_combinations:,}")
        print(f"  Coverage: {combination_coverage:.3f}%")
        
        coverage_results['combinations'] = {
            'possible_combinations': total_combinations,
            'actual_combinations': actual_combinations,
            'coverage_percentage': float(combination_coverage)
        }
        
        self.results['coverage_analysis'] = coverage_results
        return coverage_results
    
    def generate_quality_insights(self):
        """Generate quality insights and recommendations"""
        
        print("\n" + "="*60)
        print("DATA QUALITY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        insights = {
            'critical_issues': [],
            'recommendations': [],
            'quality_scores': {},
            'impact_on_forecasting': []
        }
        
        # Missing data insights
        if 'missing_data_analysis' in self.results:
            missing = self.results['missing_data_analysis']
            
            for dataset, info in missing.items():
                total_missing_pct = sum(info['missing_percentage'].values()) / len(info['missing_percentage'])
                if total_missing_pct > 5:
                    insights['critical_issues'].append(f"High missing data in {dataset}: {total_missing_pct:.1f}% average")
                
                insights['quality_scores'][f'{dataset}_completeness'] = max(0, 100 - total_missing_pct)
        
        # Outlier insights
        if 'outlier_analysis' in self.results:
            outliers = self.results['outlier_analysis']
            
            for col, methods in outliers.items():
                if 'iqr' in methods and methods['iqr']['percentage'] > 5:
                    insights['critical_issues'].append(f"High outlier rate in {col}: {methods['iqr']['percentage']:.1f}%")
                    insights['recommendations'].append(f"Consider outlier treatment for {col} before modeling")
        
        # Drift insights
        if 'drift_analysis' in self.results:
            drift = self.results['drift_analysis']
            
            drift_detected = False
            if 'distribution_drift' in drift:
                for col, analysis in drift['distribution_drift'].items():
                    if isinstance(analysis, dict) and analysis.get('drift_detected', False):
                        insights['critical_issues'].append(f"Significant distribution drift detected in {col}")
                        drift_detected = True
            
            if drift_detected:
                insights['impact_on_forecasting'].append("Data drift may affect model performance over time")
                insights['recommendations'].append("Implement model retraining strategy to handle drift")
        
        # Consistency insights
        if 'consistency_checks' in self.results:
            consistency = self.results['consistency_checks']
            
            if 'referential_integrity' in consistency:
                ref_integrity = consistency['referential_integrity']
                
                product_integrity = ref_integrity['products']['integrity_score']
                store_integrity = ref_integrity['stores']['integrity_score']
                
                if product_integrity < 0.95:
                    insights['critical_issues'].append(f"Poor product referential integrity: {product_integrity:.1%}")
                
                if store_integrity < 0.95:
                    insights['critical_issues'].append(f"Poor store referential integrity: {store_integrity:.1%}")
                
                insights['quality_scores']['referential_integrity'] = min(product_integrity, store_integrity) * 100
        
        # Coverage insights
        if 'coverage_analysis' in self.results:
            coverage = self.results['coverage_analysis']
            
            if 'temporal' in coverage:
                temporal_coverage = coverage['temporal']['coverage_percentage']
                if temporal_coverage < 90:
                    insights['critical_issues'].append(f"Poor temporal coverage: {temporal_coverage:.1f}%")
                    insights['impact_on_forecasting'].append("Missing time periods will affect trend analysis")
                
                insights['quality_scores']['temporal_coverage'] = temporal_coverage
        
        # Overall quality score
        quality_scores = insights['quality_scores']
        if quality_scores:
            overall_quality = sum(quality_scores.values()) / len(quality_scores)
            insights['quality_scores']['overall_quality'] = overall_quality
            
            if overall_quality >= 85:
                quality_rating = "Excellent"
            elif overall_quality >= 70:
                quality_rating = "Good"
            elif overall_quality >= 50:
                quality_rating = "Fair"
            else:
                quality_rating = "Poor"
            
            insights['quality_rating'] = quality_rating
        
        # General recommendations
        insights['recommendations'].extend([
            "Implement data validation pipelines for incoming data",
            "Set up monitoring for data quality metrics",
            "Create data quality dashboards for ongoing monitoring",
            "Document data quality issues and treatment strategies"
        ])
        
        print("CRITICAL ISSUES:")
        for i, issue in enumerate(insights['critical_issues'], 1):
            print(f"  {i}. {issue}")
        
        print("\nQUALITY SCORES:")
        for metric, score in insights['quality_scores'].items():
            print(f"  {metric}: {score:.1f}/100")
        
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        if 'quality_rating' in insights:
            print(f"\nOVERALL QUALITY RATING: {insights['quality_rating']}")
        
        self.results['quality_insights'] = insights
        return insights
    
    def save_quality_results(self, output_path: str = "../../data/processed/eda_results"):
        """Save quality analysis results"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = output_dir / "data_quality_analysis_results.json"
        import json
        
        # Clean results for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj
        
        clean_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items() if v is not None}
            else:
                clean_results[key] = convert_numpy(value)
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n[SAVED] Quality analysis results saved to: {results_file}")
        
        return results_file

def main():
    """Main execution function"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("PHASE 2.3: DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer()
    
    try:
        # Load data for quality analysis
        trans_df, prod_df, pdv_df = analyzer.load_quality_analysis_data()
        
        # Execute comprehensive quality analysis
        print("\n[EXECUTING] Missing Data Pattern Analysis...")
        missing_analysis = analyzer.analyze_missing_data_patterns()
        
        print("\n[EXECUTING] Multi-Method Outlier Detection...")
        outlier_analysis = analyzer.detect_outliers_multiple_methods()
        
        print("\n[EXECUTING] Data Drift Detection...")
        drift_analysis = analyzer.analyze_data_drift()
        
        print("\n[EXECUTING] Consistency Checks...")
        consistency_analysis = analyzer.perform_consistency_checks()
        
        print("\n[EXECUTING] Coverage Analysis...")
        coverage_analysis = analyzer.analyze_coverage_completeness()
        
        print("\n[EXECUTING] Quality Insights Generation...")
        quality_insights = analyzer.generate_quality_insights()
        
        # Save results
        results_file = analyzer.save_quality_results()
        
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"[OK] Analyzed {len(trans_df):,} transaction records")
        print(f"[OK] Evaluated {len(prod_df):,} product records")
        print(f"[OK] Checked {len(pdv_df):,} PDV records")
        print(f"[OK] Identified {len(quality_insights['critical_issues'])} critical quality issues")
        print(f"[OK] Generated {len(quality_insights['recommendations'])} recommendations")
        if 'quality_rating' in quality_insights:
            print(f"[OK] Overall quality rating: {quality_insights['quality_rating']}")
        print(f"[OK] Results saved to: {results_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Quality analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()