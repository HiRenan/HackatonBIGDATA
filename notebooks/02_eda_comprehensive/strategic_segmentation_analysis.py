#!/usr/bin/env python3
"""
STRATEGIC SEGMENTATION ANALYSIS - Hackathon Forecast Big Data 2025
Phase 2.2: Segmentação Estratégica & ABC Analysis

Objetivos:
- Implementar ABC Analysis (20/80 rule) para produtos por volume/faturamento
- Segmentar produtos por velocidade de giro (Fast/Medium/Slow moving)
- Criar PDV profiling com performance tiers
- Análise regional e geográfica (urban vs rural)
- Identificar cross-category dynamics
- Mapear customer journey por tipo de PDV

Estratégia: Segmentação inteligente focada na otimização WMAPE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.data_loader import OptimizedDataLoader, load_data_efficiently
from evaluation.metrics import wmape

# Advanced analytics libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("[OK] Advanced analytics libraries loaded")
except ImportError as e:
    print(f"[WARNING] Some advanced libraries missing: {e}")

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SegmentationAnalyzer:
    """Strategic segmentation analysis for retail forecasting optimization"""
    
    def __init__(self, data_path: str = "../../data/raw"):
        self.data_path = data_path
        self.loader = OptimizedDataLoader(data_path)
        self.results = {}
        
    def load_comprehensive_data(self, sample_transactions: int = 2000000, sample_products: int = 200000):
        """Load all datasets for comprehensive segmentation analysis"""
        
        print("="*60)
        print("LOADING COMPREHENSIVE DATA FOR SEGMENTATION")
        print("="*60)
        
        # Load all three datasets
        self.trans_df, self.prod_df, self.pdv_df = load_data_efficiently(
            data_path=self.data_path,
            sample_transactions=sample_transactions,
            sample_products=sample_products
        )
        
        print(f"\nDatasets loaded:")
        print(f"  Transactions: {len(self.trans_df):,} records")
        print(f"  Products: {len(self.prod_df):,} records") 
        print(f"  PDVs: {len(self.pdv_df):,} records")
        
        # Prepare date features for temporal segmentation
        if 'transaction_date' in self.trans_df.columns:
            self.trans_df['transaction_date'] = pd.to_datetime(self.trans_df['transaction_date'])
            self.trans_df['year_month'] = self.trans_df['transaction_date'].dt.to_period('M')
            self.trans_df['quarter'] = self.trans_df['transaction_date'].dt.quarter
        
        return self.trans_df, self.prod_df, self.pdv_df
    
    def perform_abc_analysis(self):
        """Comprehensive ABC Analysis for products by volume and value"""
        
        print("\n" + "="*60)
        print("ABC ANALYSIS - PRODUCTS BY VOLUME & VALUE")
        print("="*60)
        
        # Aggregate product performance
        product_performance = self.trans_df.groupby('internal_product_id').agg({
            'quantity': ['sum', 'count', 'mean'],
            'gross_value': 'sum' if 'gross_value' in self.trans_df.columns else 'count',
            'internal_store_id': 'nunique',
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        product_performance.columns = [
            'product_id', 'total_quantity', 'transaction_count', 'avg_quantity_per_transaction',
            'total_value', 'store_count', 'first_sale', 'last_sale'
        ]
        
        # Calculate additional metrics
        product_performance['days_active'] = (
            pd.to_datetime(product_performance['last_sale']) - 
            pd.to_datetime(product_performance['first_sale'])
        ).dt.days + 1
        
        product_performance['velocity_units_per_day'] = (
            product_performance['total_quantity'] / product_performance['days_active']
        )
        
        product_performance['store_penetration'] = (
            product_performance['store_count'] / len(self.pdv_df)
        )
        
        print(f"Analyzing {len(product_performance)} unique products")
        
        # ABC Classification by Volume
        product_performance_sorted = product_performance.sort_values('total_quantity', ascending=False)
        product_performance_sorted['cumulative_quantity'] = product_performance_sorted['total_quantity'].cumsum()
        product_performance_sorted['quantity_percentage'] = (
            product_performance_sorted['cumulative_quantity'] / 
            product_performance_sorted['total_quantity'].sum() * 100
        )
        
        # Define ABC tiers
        def assign_abc_tier_quantity(row):
            if row['quantity_percentage'] <= 80:
                return 'A'
            elif row['quantity_percentage'] <= 95:
                return 'B'
            else:
                return 'C'
        
        product_performance_sorted['abc_tier_quantity'] = product_performance_sorted.apply(assign_abc_tier_quantity, axis=1)
        
        # ABC Classification by Value
        if 'total_value' in product_performance_sorted.columns:
            product_performance_value = product_performance_sorted.sort_values('total_value', ascending=False)
            product_performance_value['cumulative_value'] = product_performance_value['total_value'].cumsum()
            product_performance_value['value_percentage'] = (
                product_performance_value['cumulative_value'] / 
                product_performance_value['total_value'].sum() * 100
            )
            
            def assign_abc_tier_value(row):
                if row['value_percentage'] <= 80:
                    return 'A'
                elif row['value_percentage'] <= 95:
                    return 'B'
                else:
                    return 'C'
            
            product_performance_value['abc_tier_value'] = product_performance_value.apply(assign_abc_tier_value, axis=1)
            
            # Merge value tiers back
            product_performance_sorted = product_performance_sorted.merge(
                product_performance_value[['product_id', 'abc_tier_value']], 
                on='product_id', 
                how='left'
            )
        
        # ABC Analysis Summary
        abc_summary = {}
        
        for tier in ['A', 'B', 'C']:
            tier_products = product_performance_sorted[product_performance_sorted['abc_tier_quantity'] == tier]
            
            abc_summary[f'tier_{tier}'] = {
                'product_count': len(tier_products),
                'product_percentage': len(tier_products) / len(product_performance_sorted) * 100,
                'quantity_share': tier_products['total_quantity'].sum() / product_performance_sorted['total_quantity'].sum() * 100,
                'avg_velocity': tier_products['velocity_units_per_day'].mean(),
                'avg_store_penetration': tier_products['store_penetration'].mean() * 100
            }
        
        print("\nABC Analysis Results (by Quantity):")
        print("-" * 40)
        
        for tier, metrics in abc_summary.items():
            print(f"{tier.upper()}:")
            print(f"  Products: {metrics['product_count']:,} ({metrics['product_percentage']:.1f}%)")
            print(f"  Quantity Share: {metrics['quantity_share']:.1f}%")
            print(f"  Avg Velocity: {metrics['avg_velocity']:.2f} units/day")
            print(f"  Avg Store Penetration: {metrics['avg_store_penetration']:.1f}%")
            print()
        
        self.product_performance = product_performance_sorted
        self.results['abc_analysis'] = abc_summary
        
        return product_performance_sorted, abc_summary
    
    def analyze_velocity_segments(self):
        """Analyze product velocity segments (Fast/Medium/Slow moving)"""
        
        print("\n" + "="*60)
        print("VELOCITY SEGMENTATION ANALYSIS")
        print("="*60)
        
        if not hasattr(self, 'product_performance'):
            print("[ERROR] Run ABC analysis first")
            return None
        
        # Define velocity thresholds based on distribution
        velocity_stats = self.product_performance['velocity_units_per_day'].describe()
        
        # Use percentiles for velocity classification
        fast_threshold = velocity_stats['75%']   # Top 25%
        medium_threshold = velocity_stats['25%']  # Bottom 25%
        
        def classify_velocity(velocity):
            if velocity >= fast_threshold:
                return 'Fast'
            elif velocity >= medium_threshold:
                return 'Medium'
            else:
                return 'Slow'
        
        self.product_performance['velocity_segment'] = self.product_performance['velocity_units_per_day'].apply(classify_velocity)
        
        # Velocity segment analysis
        velocity_summary = {}
        
        for segment in ['Fast', 'Medium', 'Slow']:
            segment_products = self.product_performance[self.product_performance['velocity_segment'] == segment]
            
            velocity_summary[segment] = {
                'product_count': len(segment_products),
                'percentage': len(segment_products) / len(self.product_performance) * 100,
                'quantity_share': segment_products['total_quantity'].sum() / self.product_performance['total_quantity'].sum() * 100,
                'avg_velocity': segment_products['velocity_units_per_day'].mean(),
                'avg_transactions': segment_products['transaction_count'].mean(),
                'avg_store_penetration': segment_products['store_penetration'].mean() * 100,
                'forecasting_complexity': 'Low' if segment == 'Fast' else 'Medium' if segment == 'Medium' else 'High'
            }
        
        print("Velocity Segment Analysis:")
        print("-" * 40)
        
        for segment, metrics in velocity_summary.items():
            print(f"{segment.upper()} MOVING PRODUCTS:")
            print(f"  Count: {metrics['product_count']:,} ({metrics['percentage']:.1f}%)")
            print(f"  Quantity Share: {metrics['quantity_share']:.1f}%")
            print(f"  Avg Velocity: {metrics['avg_velocity']:.2f} units/day")
            print(f"  Avg Store Penetration: {metrics['avg_store_penetration']:.1f}%")
            print(f"  Forecasting Complexity: {metrics['forecasting_complexity']}")
            print()
        
        # Cross-analysis: ABC vs Velocity
        print("Cross-Analysis: ABC Tiers vs Velocity Segments:")
        print("-" * 50)
        
        cross_analysis = pd.crosstab(
            self.product_performance['abc_tier_quantity'], 
            self.product_performance['velocity_segment'], 
            normalize='index'
        ) * 100
        
        print(cross_analysis.round(1))
        
        self.results['velocity_analysis'] = velocity_summary
        self.results['abc_velocity_cross'] = cross_analysis.to_dict()
        
        return velocity_summary
    
    def analyze_pdv_segments(self):
        """Comprehensive PDV segmentation and profiling"""
        
        print("\n" + "="*60)
        print("PDV SEGMENTATION & PROFILING")
        print("="*60)
        
        # Aggregate PDV performance
        pdv_performance = self.trans_df.groupby('internal_store_id').agg({
            'quantity': ['sum', 'count', 'mean'],
            'gross_value': 'sum' if 'gross_value' in self.trans_df.columns else 'count',
            'internal_product_id': 'nunique',
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        pdv_performance.columns = [
            'store_id', 'total_quantity', 'transaction_count', 'avg_quantity_per_transaction',
            'total_value', 'unique_products', 'first_transaction', 'last_transaction'
        ]
        
        # Calculate additional metrics
        pdv_performance['days_active'] = (
            pd.to_datetime(pdv_performance['last_transaction']) - 
            pd.to_datetime(pdv_performance['first_transaction'])
        ).dt.days + 1
        
        pdv_performance['avg_daily_quantity'] = (
            pdv_performance['total_quantity'] / pdv_performance['days_active']
        )
        
        pdv_performance['product_variety_score'] = (
            pdv_performance['unique_products'] / len(self.prod_df)
        )
        
        print(f"Analyzing {len(pdv_performance)} unique PDVs")
        
        # Merge with PDV master data
        if 'pdv' in self.pdv_df.columns:
            # Assume pdv column is the store identifier
            pdv_merged = pdv_performance.merge(
                self.pdv_df.rename(columns={'pdv': 'store_id'}), 
                on='store_id', 
                how='left'
            )
        else:
            pdv_merged = pdv_performance
        
        # PDV Performance Tiers
        performance_stats = pdv_performance['total_quantity'].describe()
        
        def classify_pdv_performance(quantity):
            if quantity >= performance_stats['75%']:
                return 'High'
            elif quantity >= performance_stats['25%']:
                return 'Medium'
            else:
                return 'Low'
        
        pdv_merged['performance_tier'] = pdv_merged['total_quantity'].apply(classify_pdv_performance)
        
        # PDV Segment Analysis
        pdv_summary = {}
        
        for tier in ['High', 'Medium', 'Low']:
            tier_pdvs = pdv_merged[pdv_merged['performance_tier'] == tier]
            
            pdv_summary[tier] = {
                'pdv_count': len(tier_pdvs),
                'percentage': len(tier_pdvs) / len(pdv_merged) * 100,
                'quantity_share': tier_pdvs['total_quantity'].sum() / pdv_merged['total_quantity'].sum() * 100,
                'avg_daily_quantity': tier_pdvs['avg_daily_quantity'].mean(),
                'avg_product_variety': tier_pdvs['product_variety_score'].mean() * 100,
                'avg_transactions': tier_pdvs['transaction_count'].mean()
            }
        
        print("PDV Performance Tier Analysis:")
        print("-" * 40)
        
        for tier, metrics in pdv_summary.items():
            print(f"{tier.upper()} PERFORMANCE PDVs:")
            print(f"  Count: {metrics['pdv_count']:,} ({metrics['percentage']:.1f}%)")
            print(f"  Quantity Share: {metrics['quantity_share']:.1f}%")
            print(f"  Avg Daily Quantity: {metrics['avg_daily_quantity']:.1f} units/day")
            print(f"  Avg Product Variety: {metrics['avg_product_variety']:.1f}%")
            print()
        
        # Regional Analysis (if zipcode available)
        if 'zipcode' in pdv_merged.columns:
            print("Regional Analysis:")
            print("-" * 20)
            
            # Extract region from zipcode (first 2 digits for Brazilian CEP)
            pdv_merged['region'] = pdv_merged['zipcode'].astype(str).str[:2]
            
            regional_analysis = pdv_merged.groupby('region').agg({
                'total_quantity': 'sum',
                'store_id': 'count',
                'avg_daily_quantity': 'mean'
            }).sort_values('total_quantity', ascending=False)
            
            print("Top regions by volume:")
            for region, data in regional_analysis.head(10).iterrows():
                print(f"  Region {region}: {data['total_quantity']:,.0f} units ({data['store_id']} stores)")
        
        self.pdv_performance = pdv_merged
        self.results['pdv_analysis'] = pdv_summary
        
        return pdv_merged, pdv_summary
    
    def analyze_category_dynamics(self):
        """Analyze cross-category dynamics and interactions"""
        
        print("\n" + "="*60)
        print("CATEGORY DYNAMICS ANALYSIS")
        print("="*60)
        
        # Memory-efficient approach: work with transaction sample and product categories
        if 'categoria' in self.prod_df.columns:
            # First, aggregate transactions by product to reduce memory
            trans_agg = self.trans_df.groupby('internal_product_id').agg({
                'quantity': 'sum',
                'internal_store_id': 'nunique'
            }).reset_index()
            
            # Then merge with product categories (much smaller merge)
            trans_with_category = trans_agg.merge(
                self.prod_df[['produto', 'categoria']].rename(columns={'produto': 'internal_product_id'}),
                on='internal_product_id',
                how='inner'  # Use inner join to avoid memory issues
            )
        else:
            print("[WARNING] No category column found in product data")
            return None
        
        # Category performance analysis
        category_performance = trans_with_category.groupby('categoria').agg({
            'quantity': 'sum',
            'internal_product_id': 'nunique',
            'internal_store_id': 'sum'  # Sum of unique stores per product
        }).reset_index()
        
        category_performance.columns = [
            'category', 'total_quantity', 'unique_products', 'total_store_presence'
        ]
        
        # Calculate category metrics
        category_performance['market_share'] = (
            category_performance['total_quantity'] / 
            category_performance['total_quantity'].sum() * 100
        )
        
        category_performance['avg_quantity_per_product'] = (
            category_performance['total_quantity'] / 
            category_performance['unique_products']
        )
        
        category_performance['store_penetration'] = (
            category_performance['total_store_presence'] / len(self.pdv_df) * 100
        )
        
        # Sort by market share
        category_performance = category_performance.sort_values('market_share', ascending=False)
        
        print(f"Analyzing {len(category_performance)} product categories")
        print("\nTop Categories by Market Share:")
        print("-" * 40)
        
        for _, cat in category_performance.head(10).iterrows():
            print(f"{cat['category'][:30]:30} {cat['market_share']:6.2f}% ({cat['unique_products']:4d} products)")
        
        # Category clustering for strategic insights
        try:
            category_features = category_performance[[
                'market_share', 'avg_quantity_per_product', 'store_penetration'
            ]].fillna(0)
            
            scaler = StandardScaler()
            category_features_scaled = scaler.fit_transform(category_features)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            category_performance['cluster'] = kmeans.fit_predict(category_features_scaled)
            
            # Cluster interpretation
            cluster_summary = category_performance.groupby('cluster').agg({
                'market_share': 'mean',
                'avg_quantity_per_product': 'mean',
                'store_penetration': 'mean',
                'category': 'count'
            }).round(2)
            
            cluster_summary.columns = ['avg_market_share', 'avg_qty_per_product', 'avg_penetration', 'category_count']
            
            print("\nCategory Clusters:")
            print("-" * 20)
            
            cluster_labels = {
                0: 'Mass Market',
                1: 'Niche Premium', 
                2: 'Volume Leaders',
                3: 'Emerging Categories'
            }
            
            for cluster_id, metrics in cluster_summary.iterrows():
                label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
                print(f"{label}:")
                print(f"  Categories: {int(metrics['category_count'])}")
                print(f"  Avg Market Share: {metrics['avg_market_share']:.2f}%")
                print(f"  Avg Store Penetration: {metrics['avg_penetration']:.2f}%")
                print()
                
        except Exception as e:
            print(f"[WARNING] Category clustering failed: {e}")
            cluster_summary = None
        
        self.category_performance = category_performance
        self.results['category_analysis'] = {
            'top_categories': category_performance.head(10).to_dict('records'),
            'cluster_summary': cluster_summary.to_dict() if cluster_summary is not None else None
        }
        
        return category_performance
    
    def generate_strategic_insights(self):
        """Generate strategic insights for forecasting optimization"""
        
        print("\n" + "="*60)
        print("STRATEGIC SEGMENTATION INSIGHTS")
        print("="*60)
        
        insights = {
            'key_findings': [],
            'wmape_optimization_strategies': [],
            'segmentation_recommendations': []
        }
        
        # ABC Analysis insights
        if 'abc_analysis' in self.results:
            abc = self.results['abc_analysis']
            tier_a_share = abc['tier_A']['quantity_share']
            
            insights['key_findings'].append(f"Tier A products represent {tier_a_share:.1f}% of total volume")
            
            if tier_a_share > 75:
                insights['wmape_optimization_strategies'].append("Focus 80% of modeling effort on Tier A products")
            
            insights['segmentation_recommendations'].append("Use different forecast horizons for each ABC tier")
        
        # Velocity analysis insights
        if 'velocity_analysis' in self.results:
            velocity = self.results['velocity_analysis']
            fast_share = velocity['Fast']['quantity_share']
            
            insights['key_findings'].append(f"Fast-moving products account for {fast_share:.1f}% of volume")
            
            insights['wmape_optimization_strategies'].extend([
                "Apply simple models (MA, exponential smoothing) for fast-moving products",
                "Use complex models (ML ensembles) for slow-moving products",
                "Implement intermittent demand models for slow movers"
            ])
        
        # PDV analysis insights
        if 'pdv_analysis' in self.results:
            pdv = self.results['pdv_analysis']
            high_perf_share = pdv['High']['quantity_share']
            
            insights['key_findings'].append(f"High-performance stores drive {high_perf_share:.1f}% of volume")
            
            insights['segmentation_recommendations'].extend([
                "Create store-specific forecasting models for high-performance PDVs",
                "Use regional/cluster models for medium and low-performance stores"
            ])
        
        # Category analysis insights
        if 'category_analysis' in self.results and self.results['category_analysis']['top_categories']:
            top_cats = self.results['category_analysis']['top_categories']
            top_3_share = sum(cat['market_share'] for cat in top_cats[:3])
            
            insights['key_findings'].append(f"Top 3 categories represent {top_3_share:.1f}% of market share")
            
            insights['wmape_optimization_strategies'].append("Develop category-specific feature engineering")
        
        # WMAPE-specific recommendations
        insights['wmape_optimization_strategies'].extend([
            "Weight model validation by product volume to match WMAPE calculation",
            "Apply volume-based ensemble weighting",
            "Implement forecast post-processing for volume-weighted accuracy"
        ])
        
        print("Key Strategic Findings:")
        for i, finding in enumerate(insights['key_findings'], 1):
            print(f"  {i}. {finding}")
        
        print("\nWMAPE Optimization Strategies:")
        for i, strategy in enumerate(insights['wmape_optimization_strategies'], 1):
            print(f"  {i}. {strategy}")
        
        self.results['strategic_insights'] = insights
        return insights
    
    def save_segmentation_results(self, output_path: str = "../../data/processed/eda_results"):
        """Save segmentation analysis results"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_results[key] = convert_numpy(value)
        
        results_file = output_dir / "segmentation_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n[SAVED] Segmentation results saved to: {results_file}")
        
        # Save detailed dataframes
        if hasattr(self, 'product_performance'):
            product_file = output_dir / "product_performance_segmented.csv"
            self.product_performance.to_csv(product_file, index=False)
            print(f"[SAVED] Product performance data: {product_file}")
        
        if hasattr(self, 'pdv_performance'):
            pdv_file = output_dir / "pdv_performance_segmented.csv"
            self.pdv_performance.to_csv(pdv_file, index=False)
            print(f"[SAVED] PDV performance data: {pdv_file}")
        
        if hasattr(self, 'category_performance'):
            category_file = output_dir / "category_performance_analysis.csv"
            self.category_performance.to_csv(category_file, index=False)
            print(f"[SAVED] Category analysis data: {category_file}")
        
        return results_file

def main():
    """Main execution function"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("PHASE 2.2: STRATEGIC SEGMENTATION ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SegmentationAnalyzer()
    
    try:
        # Load comprehensive data
        trans_df, prod_df, pdv_df = analyzer.load_comprehensive_data()
        
        # Execute comprehensive segmentation analysis
        print("\n[EXECUTING] ABC Analysis...")
        product_perf, abc_summary = analyzer.perform_abc_analysis()
        
        print("\n[EXECUTING] Velocity Segmentation...")
        velocity_summary = analyzer.analyze_velocity_segments()
        
        print("\n[EXECUTING] PDV Segmentation...")
        pdv_perf, pdv_summary = analyzer.analyze_pdv_segments()
        
        print("\n[EXECUTING] Category Dynamics...")
        category_perf = analyzer.analyze_category_dynamics()
        
        print("\n[EXECUTING] Strategic Insights Generation...")
        insights = analyzer.generate_strategic_insights()
        
        # Save results
        results_file = analyzer.save_segmentation_results()
        
        print("\n" + "="*80)
        print("STRATEGIC SEGMENTATION ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"[OK] Analyzed {len(trans_df):,} transactions")
        print(f"[OK] Segmented {len(product_perf):,} products into ABC tiers")
        print(f"[OK] Analyzed {len(pdv_perf):,} PDV performance profiles")
        if category_perf is not None:
            print(f"[OK] Profiled {len(category_perf)} product categories")
        print(f"[OK] Generated {len(insights['key_findings'])} strategic insights")
        print(f"[OK] Results saved to: {results_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Segmentation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()