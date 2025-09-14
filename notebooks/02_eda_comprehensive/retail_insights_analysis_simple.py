#!/usr/bin/env python3
"""
RETAIL INSIGHTS ANALYSIS - Hackathon Forecast Big Data 2025
Phase 2.4: Insights Específicos de Varejo (Simplified Version)

Objetivos:
- Velocity analysis: Fast vs slow-moving products
- Seasonality detection: Basic patterns
- Cross-selling patterns: Simple co-occurrence
- Price elasticity: Basic analysis
- Promotion effects: Price drop detection

Estratégia: Gerar insights de varejo acionáveis para feature engineering
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

from utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetailInsightsAnalyzer:
    """Simplified Retail Insights Analysis for Hackathon Forecast"""
    
    def __init__(self):
        self.results = {}
        self.trans_df = None
        self.prod_df = None
        self.pdv_df = None
        
    def load_retail_analysis_data(self, sample_transactions: int = 100000, 
                                 sample_products: int = 10000):
        """Load data optimized for retail insights analysis"""
        
        print("LOADING DATA FOR RETAIL INSIGHTS ANALYSIS")
        print("-" * 60)
        
        try:
            # Use optimized data loader
            self.trans_df, self.prod_df, self.pdv_df = load_data_efficiently(
                data_path="../../data/raw",
                sample_transactions=sample_transactions,
                sample_products=sample_products
            )
            
            # Preprocessing for retail analysis
            if 'transaction_date' in self.trans_df.columns:
                self.trans_df['transaction_date'] = pd.to_datetime(self.trans_df['transaction_date'])
                self.trans_df['week'] = self.trans_df['transaction_date'].dt.isocalendar().week
                self.trans_df['month'] = self.trans_df['transaction_date'].dt.month
                self.trans_df['quarter'] = self.trans_df['transaction_date'].dt.quarter
                self.trans_df['day_of_week'] = self.trans_df['transaction_date'].dt.dayofweek
            
            # Calculate unit price
            self.trans_df['unit_price'] = self.trans_df['gross_value'] / self.trans_df['quantity']
            
            print(f"[OK] Loaded {len(self.trans_df):,} transactions")
            print(f"[OK] Loaded {len(self.prod_df):,} products") 
            print(f"[OK] Loaded {len(self.pdv_df):,} PDVs")
            
            return self.trans_df, self.prod_df, self.pdv_df
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return None, None, None

    def analyze_velocity_patterns(self):
        """ABC/XYZ velocity analysis"""
        print("\n" + "="*60)
        print("VELOCITY ANALYSIS - ABC/XYZ CLASSIFICATION") 
        print("="*60)
        
        # Calculate product velocity metrics
        velocity_metrics = self.trans_df.groupby('internal_product_id').agg({
            'quantity': ['sum', 'mean', 'std', 'count'],
            'net_value': 'sum'
        }).round(2)
        
        velocity_metrics.columns = ['total_qty', 'avg_qty', 'std_qty', 'frequency', 'total_revenue']
        velocity_metrics['cv_demand'] = velocity_metrics['std_qty'] / velocity_metrics['avg_qty'].replace(0, np.nan)
        
        # ABC Classification (by revenue)
        velocity_metrics['revenue_rank'] = velocity_metrics['total_revenue'].rank(ascending=False, pct=True)
        velocity_metrics['abc_class'] = velocity_metrics['revenue_rank'].apply(
            lambda x: 'A' if x <= 0.2 else 'B' if x <= 0.5 else 'C'
        )
        
        # XYZ Classification (by demand variability)
        velocity_metrics = velocity_metrics.dropna(subset=['cv_demand'])
        if len(velocity_metrics) > 0:
            cv_thresholds = velocity_metrics['cv_demand'].quantile([0.33, 0.67])
            velocity_metrics['xyz_class'] = velocity_metrics['cv_demand'].apply(
                lambda x: 'X' if x <= cv_thresholds[0.33] else 'Y' if x <= cv_thresholds[0.67] else 'Z'
            )
        
            # Combined classification
            velocity_metrics['velocity_segment'] = velocity_metrics['abc_class'] + velocity_metrics['xyz_class']
        
            print("\nVELOCITY SEGMENT DISTRIBUTION:")
            segment_counts = velocity_metrics['velocity_segment'].value_counts().sort_index()
            for segment, count in segment_counts.items():
                pct = count / len(velocity_metrics) * 100
                revenue_pct = velocity_metrics[velocity_metrics['velocity_segment'] == segment]['total_revenue'].sum() / velocity_metrics['total_revenue'].sum() * 100
                print(f"  {segment}: {count:,} products ({pct:.1f}%) - {revenue_pct:.1f}% revenue")
        
        # Fast vs Slow moving identification
        velocity_threshold = velocity_metrics['frequency'].quantile(0.8)
        velocity_metrics['movement_speed'] = velocity_metrics['frequency'].apply(
            lambda x: 'Fast' if x >= velocity_threshold else 'Slow'
        )
        
        fast_moving = velocity_metrics[velocity_metrics['movement_speed'] == 'Fast']
        slow_moving = velocity_metrics[velocity_metrics['movement_speed'] == 'Slow']
        
        print(f"\nMOVEMENT SPEED CLASSIFICATION:")
        print(f"  Fast-moving products: {len(fast_moving):,} ({len(fast_moving)/len(velocity_metrics)*100:.1f}%)")
        print(f"  Slow-moving products: {len(slow_moving):,} ({len(slow_moving)/len(velocity_metrics)*100:.1f}%)")
        print(f"  Fast-moving revenue share: {fast_moving['total_revenue'].sum()/velocity_metrics['total_revenue'].sum()*100:.1f}%")
        
        self.results['velocity_analysis'] = velocity_metrics
        
        return velocity_metrics

    def detect_seasonality_patterns(self):
        """Basic seasonality detection"""
        print("\n" + "="*60)
        print("SEASONALITY DETECTION ANALYSIS")
        print("="*60)
        
        # Weekly patterns analysis
        weekly_patterns = self.trans_df.groupby('day_of_week')['quantity'].agg(['mean', 'std']).round(2)
        weekly_patterns['cv'] = weekly_patterns['std'] / weekly_patterns['mean']
        
        # Monthly patterns analysis  
        monthly_patterns = self.trans_df.groupby('month')['quantity'].agg(['mean', 'std']).round(2)
        monthly_patterns['cv'] = monthly_patterns['std'] / monthly_patterns['mean']
        
        print(f"\nWEEKLY SEASONALITY PATTERNS:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, (mean_qty, cv) in enumerate(zip(weekly_patterns['mean'], weekly_patterns['cv'])):
            print(f"  {days[i]}: {mean_qty:.1f} avg qty (CV: {cv:.2f})")
        
        print(f"\nMONTHLY SEASONALITY PATTERNS:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, (mean_qty, cv) in enumerate(zip(monthly_patterns['mean'], monthly_patterns['cv'])):
            print(f"  {months[i]}: {mean_qty:.1f} avg qty (CV: {cv:.2f})")
        
        seasonality_results = {
            'weekly_patterns': weekly_patterns,
            'monthly_patterns': monthly_patterns,
        }
        
        self.results['seasonality_analysis'] = seasonality_results
        
        return seasonality_results

    def analyze_cross_selling_patterns(self):
        """Simple co-occurrence analysis"""
        print("\n" + "="*60)
        print("CROSS-SELLING PATTERNS ANALYSIS")
        print("="*60)
        
        # Get products that appear together in transactions (same date/store)
        transaction_groups = self.trans_df.groupby(['transaction_date', 'internal_store_id'])['internal_product_id'].apply(set)
        
        # Count co-occurrences
        co_occurrence = {}
        product_counts = {}
        
        for products in transaction_groups:
            if isinstance(products, set) and len(products) > 1:
                products_list = list(products)
                for i, prod1 in enumerate(products_list):
                    product_counts[prod1] = product_counts.get(prod1, 0) + 1
                    for prod2 in products_list[i+1:]:
                        pair = tuple(sorted([prod1, prod2]))
                        co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
        
        # Calculate co-occurrence scores
        cooc_results = []
        for (prod1, prod2), count in co_occurrence.items():
            if count >= 5:  # Minimum co-occurrence threshold
                prob_1 = product_counts[prod1] / len(transaction_groups)
                prob_2 = product_counts[prod2] / len(transaction_groups)
                prob_both = count / len(transaction_groups)
                
                lift = prob_both / (prob_1 * prob_2) if prob_1 * prob_2 > 0 else 0
                
                cooc_results.append({
                    'product_1': prod1,
                    'product_2': prod2,
                    'co_occurrence_count': count,
                    'lift': lift,
                    'support': prob_both
                })
        
        cooc_df = pd.DataFrame(cooc_results).sort_values('lift', ascending=False)
        
        print(f"CO-OCCURRENCE ANALYSIS:")
        print(f"  Co-occurrence pairs found: {len(cooc_df):,}")
        if len(cooc_df) > 0:
            print(f"  High lift pairs (>2.0): {len(cooc_df[cooc_df['lift'] > 2.0]):,}")
            print(f"  Average lift: {cooc_df['lift'].mean():.2f}")
        
        self.results['cross_selling_analysis'] = cooc_df
        
        return cooc_df

    def analyze_price_elasticity(self):
        """Basic price elasticity analysis"""
        print("\n" + "="*60)
        print("PRICE ELASTICITY ANALYSIS")
        print("="*60)
        
        # Prepare data for elasticity analysis
        price_demand_data = self.trans_df.groupby(['internal_product_id', 'unit_price'])['quantity'].sum().reset_index()
        
        elasticity_results = []
        
        # Analyze elasticity by product (minimum 3 different price points)
        for produto_id in price_demand_data['internal_product_id'].unique()[:1000]:  # Sample for performance
            product_data = price_demand_data[price_demand_data['internal_product_id'] == produto_id]
            
            if len(product_data) >= 3:  # Need multiple price points
                try:
                    # Simple correlation between price and quantity
                    correlation = product_data['unit_price'].corr(product_data['quantity'])
                    
                    if not np.isnan(correlation):
                        elasticity_results.append({
                            'produto_id': produto_id,
                            'price_elasticity': correlation,
                            'price_points': len(product_data),
                            'avg_price': product_data['unit_price'].mean(),
                            'total_demand': product_data['quantity'].sum()
                        })
                
                except Exception as e:
                    continue
        
        elasticity_df = pd.DataFrame(elasticity_results)
        
        if len(elasticity_df) > 0:
            print(f"PRICE ELASTICITY RESULTS:")
            print(f"  Products analyzed: {len(elasticity_df):,}")
            print(f"  Average price elasticity: {elasticity_df['price_elasticity'].mean():.3f}")
            
            # Classify products by elasticity
            elastic_products = elasticity_df[elasticity_df['price_elasticity'] < -0.3]
            inelastic_products = elasticity_df[elasticity_df['price_elasticity'] > -0.1]
            
            print(f"  Price-sensitive products: {len(elastic_products)}")
            print(f"  Price-insensitive products: {len(inelastic_products)}")
        else:
            print("[WARNING] Insufficient data for price elasticity analysis")
        
        self.results['price_elasticity_analysis'] = elasticity_df
        
        return elasticity_df

    def detect_promotion_effects(self):
        """Basic promotion detection"""
        print("\n" + "="*60) 
        print("PROMOTION EFFECTS DETECTION")
        print("="*60)
        
        # Simple promotion detection based on price drops
        product_prices = self.trans_df.groupby(['internal_product_id', 'transaction_date'])['unit_price'].mean().reset_index()
        
        promotion_count = 0
        total_products = 0
        
        for produto_id in product_prices['internal_product_id'].unique()[:500]:  # Sample for performance
            product_data = product_prices[product_prices['internal_product_id'] == produto_id].sort_values('transaction_date')
            total_products += 1
            
            if len(product_data) >= 5:  # Need sufficient history
                # Simple price drop detection
                price_std = product_data['unit_price'].std()
                if price_std > 0:
                    mean_price = product_data['unit_price'].mean()
                    min_price = product_data['unit_price'].min()
                    
                    # If minimum price is significantly below average
                    if (mean_price - min_price) > price_std:
                        promotion_count += 1
        
        promotion_rate = promotion_count / total_products if total_products > 0 else 0
        
        print(f"PROMOTION DETECTION RESULTS:")
        print(f"  Products analyzed: {total_products:,}")
        print(f"  Products with potential promotions: {promotion_count:,}")
        print(f"  Promotion rate: {promotion_rate:.1%}")
        
        self.results['promotion_analysis'] = {
            'total_products': total_products,
            'promotion_count': promotion_count,
            'promotion_rate': promotion_rate
        }
        
        return self.results['promotion_analysis']

    def generate_retail_insights_summary(self):
        """Generate executive summary of retail insights"""
        
        print("\n" + "="*80)
        print("RETAIL INSIGHTS EXECUTIVE SUMMARY")
        print("="*80)
        
        insights = {
            'critical_insights': [],
            'actionable_recommendations': [],
            'feature_engineering_opportunities': []
        }
        
        # Velocity insights
        if 'velocity_analysis' in self.results:
            velocity_data = self.results['velocity_analysis']
            fast_count = len(velocity_data[velocity_data['movement_speed'] == 'Fast'])
            total_products = len(velocity_data)
            
            insights['critical_insights'].append(
                f"Fast-moving products: {fast_count}/{total_products} ({fast_count/total_products:.1%})"
            )
            
            insights['feature_engineering_opportunities'].append(
                "Create velocity-based features: ABC/XYZ segments, movement speed indicators"
            )
        
        # Seasonality insights
        if 'seasonality_analysis' in self.results:
            insights['feature_engineering_opportunities'].append(
                "Develop seasonal features: day-of-week, monthly patterns"
            )
        
        # Cross-selling insights
        if 'cross_selling_analysis' in self.results:
            cross_data = self.results['cross_selling_analysis']
            if len(cross_data) > 0:
                insights['critical_insights'].append(
                    f"Cross-selling opportunities: {len(cross_data)} product pairs identified"
                )
        
        # Price elasticity insights
        if 'price_elasticity_analysis' in self.results:
            price_data = self.results['price_elasticity_analysis']
            if len(price_data) > 0:
                insights['actionable_recommendations'].append(
                    f"Price optimization opportunities for {len(price_data)} products"
                )
        
        # Promotion insights
        if 'promotion_analysis' in self.results:
            promo_data = self.results['promotion_analysis']
            insights['critical_insights'].append(
                f"Promotion detection: {promo_data['promotion_rate']:.1%} of products show promotion patterns"
            )
        
        # Display summary
        print("\nCRITICAL BUSINESS INSIGHTS:")
        for i, insight in enumerate(insights['critical_insights'], 1):
            print(f"  {i}. {insight}")
        
        print("\nACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(insights['actionable_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\nFEATURE ENGINEERING OPPORTUNITIES:")
        for i, opp in enumerate(insights['feature_engineering_opportunities'], 1):
            print(f"  {i}. {opp}")
        
        self.results['executive_summary'] = insights
        
        return insights

    def save_retail_insights_results(self, output_path: str = "../../data/processed/eda_results"):
        """Save retail insights results"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = output_dir / "retail_insights_analysis_results.json"
        import json
        
        # Clean results for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, pd.DataFrame):
                return obj.head(100).to_dict('records')  # Limit size
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        clean_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items() if v is not None}
            else:
                clean_results[key] = convert_numpy(value)
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n[SAVED] Retail insights results saved to: {results_file}")
        
        return results_file

def main():
    """Main execution function"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("PHASE 2.4: RETAIL INSIGHTS ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = RetailInsightsAnalyzer()
    
    try:
        # Load data for retail insights analysis
        trans_df, prod_df, pdv_df = analyzer.load_retail_analysis_data()
        
        if trans_df is None:
            print("[ERROR] Failed to load data")
            return None
        
        # Execute retail insights analysis
        print("\n[EXECUTING] Velocity Patterns Analysis...")
        velocity_analysis = analyzer.analyze_velocity_patterns()
        
        print("\n[EXECUTING] Seasonality Detection...")
        seasonality_analysis = analyzer.detect_seasonality_patterns()
        
        print("\n[EXECUTING] Cross-selling Patterns Analysis...")
        cross_selling_analysis = analyzer.analyze_cross_selling_patterns()
        
        print("\n[EXECUTING] Price Elasticity Analysis...")
        price_elasticity_analysis = analyzer.analyze_price_elasticity()
        
        print("\n[EXECUTING] Promotion Effects Detection...")
        promotion_analysis = analyzer.detect_promotion_effects()
        
        print("\n[EXECUTING] Executive Summary Generation...")
        executive_summary = analyzer.generate_retail_insights_summary()
        
        # Save results
        results_file = analyzer.save_retail_insights_results()
        
        print("\n" + "="*80)
        print("RETAIL INSIGHTS ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"[OK] Analyzed {len(trans_df):,} transaction records")
        print(f"[OK] Generated {len(executive_summary['critical_insights'])} critical business insights")
        print(f"[OK] Identified {len(executive_summary['feature_engineering_opportunities'])} feature opportunities")
        print(f"[OK] Results saved to: {results_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Retail insights analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()