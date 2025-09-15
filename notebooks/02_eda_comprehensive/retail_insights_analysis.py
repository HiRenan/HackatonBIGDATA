#!/usr/bin/env python3
"""
RETAIL INSIGHTS ANALYSIS - Hackathon Forecast Big Data 2025
Phase 2.4: Insights Específicos de Varejo

Objetivos:
- Velocity analysis: Fast vs slow-moving products
- Seasonality detection: Métodos automáticos (STL, X-13)  
- Cross-selling patterns: Market basket analysis
- Price elasticity: Impacto de preços nas vendas
- Promotion effects: Detecção e quantificação

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

from utils.data_loader import OptimizedDataLoader, load_data_efficiently
from evaluation.metrics import wmape

# Advanced retail analytics libraries
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.x13 import x13_arima_analysis
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import networkx as nx
    print("[OK] Advanced retail analytics libraries loaded")
except ImportError as e:
    print(f"[WARNING] Some advanced libraries missing: {e}")
    print("[INFO] Will use basic implementations where possible")

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetailInsightsAnalyzer:
    """
    Comprehensive Retail Insights Analysis for Hackathon Forecast
    
    Features:
    - ABC/XYZ Velocity Analysis
    - Advanced Seasonality Detection
    - Market Basket Analysis
    - Price Elasticity Analysis
    - Promotion Effects Detection
    """
    
    def __init__(self):
        self.results = {}
        self.trans_df = None
        self.prod_df = None
        self.pdv_df = None
        
    def load_retail_analysis_data(self, sample_transactions: int = 1000000, 
                                 sample_products: int = 50000):
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
            
            # Additional preprocessing for retail analysis
            if 'data_transacao' in self.trans_df.columns:
                self.trans_df['data_transacao'] = pd.to_datetime(self.trans_df['data_transacao'])
                self.trans_df['week'] = self.trans_df['data_transacao'].dt.isocalendar().week
                self.trans_df['month'] = self.trans_df['data_transacao'].dt.month
                self.trans_df['quarter'] = self.trans_df['data_transacao'].dt.quarter
                self.trans_df['day_of_week'] = self.trans_df['data_transacao'].dt.dayofweek
            
            print(f"[OK] Loaded {len(self.trans_df):,} transactions")
            print(f"[OK] Loaded {len(self.prod_df):,} products") 
            print(f"[OK] Loaded {len(self.pdv_df):,} PDVs")
            
            # Debug: Print column names
            print(f"\n[DEBUG] Transaction columns: {list(self.trans_df.columns)}")
            print(f"[DEBUG] Product columns: {list(self.prod_df.columns)}")
            print(f"[DEBUG] PDV columns: {list(self.pdv_df.columns)}")
            
            return self.trans_df, self.prod_df, self.pdv_df
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            
            # Generate synthetic data for development
            print("[INFO] Generating synthetic retail data for analysis...")
            return self._generate_synthetic_retail_data()
    
    def _generate_synthetic_retail_data(self):
        """Generate realistic synthetic retail data for development"""
        
        np.random.seed(42)
        
        # Generate synthetic transactions
        n_transactions = 500000
        n_products = 10000
        n_pdvs = 1000
        
        # Date range: 1 year
        start_date = pd.Timestamp('2022-01-01')
        end_date = pd.Timestamp('2022-12-31')
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Synthetic transactions
        trans_data = []
        for i in range(n_transactions):
            date = np.random.choice(dates)
            date = pd.Timestamp(date)  # Ensure it's a pandas Timestamp
            product_id = np.random.randint(1, n_products + 1)
            pdv_id = np.random.randint(1, n_pdvs + 1)
            
            # Realistic quantity with seasonality and trends
            base_qty = np.random.poisson(5)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend_factor = 1.2 if date.dayofweek >= 5 else 1.0
            quantity = max(1, int(base_qty * seasonal_factor * weekend_factor))
            
            # Price with some variability
            base_price = np.random.uniform(5, 100)
            price = base_price * np.random.uniform(0.9, 1.1)
            
            trans_data.append({
                'data_transacao': date,
                'produto_id': product_id,
                'pdv_id': pdv_id,
                'quantidade': quantity,
                'preco_unitario': price,
                'week': date.isocalendar().week,
                'month': date.month,
                'quarter': date.quarter,
                'day_of_week': date.dayofweek
            })
        
        self.trans_df = pd.DataFrame(trans_data)
        
        # Synthetic products
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Books']
        prod_data = []
        for i in range(1, n_products + 1):
            prod_data.append({
                'produto_id': i,
                'categoria': np.random.choice(categories),
                'preco_medio': np.random.uniform(10, 200),
                'marca': f'Brand_{np.random.randint(1, 100)}'
            })
        
        self.prod_df = pd.DataFrame(prod_data)
        
        # Synthetic PDVs
        regions = ['North', 'South', 'East', 'West', 'Central']
        types = ['Supermarket', 'Convenience', 'Pharmacy', 'Department']
        pdv_data = []
        for i in range(1, n_pdvs + 1):
            pdv_data.append({
                'pdv_id': i,
                'regiao': np.random.choice(regions),
                'tipo': np.random.choice(types),
                'tamanho': np.random.choice(['Small', 'Medium', 'Large'])
            })
        
        self.pdv_df = pd.DataFrame(pdv_data)
        
        print(f"[OK] Generated {len(self.trans_df):,} synthetic transactions")
        print(f"[OK] Generated {len(self.prod_df):,} synthetic products")
        print(f"[OK] Generated {len(self.pdv_df):,} synthetic PDVs")
        
        return self.trans_df, self.prod_df, self.pdv_df

    def analyze_velocity_patterns(self):
        """
        Comprehensive velocity analysis - ABC/XYZ classification
        
        ABC: By volume/revenue
        XYZ: By demand variability
        """
        print("\n" + "="*60)
        print("VELOCITY ANALYSIS - ABC/XYZ CLASSIFICATION") 
        print("="*60)
        
        # Calculate product velocity metrics
        velocity_metrics = self.trans_df.groupby('internal_product_id').agg({
            'quantity': ['sum', 'mean', 'std', 'count'],
            'net_value': 'sum',
            'gross_value': 'mean'
        }).round(2)
        
        velocity_metrics.columns = ['total_qty', 'avg_qty', 'std_qty', 'frequency', 'total_net_value', 'avg_gross_value']
        velocity_metrics['revenue'] = velocity_metrics['total_net_value']
        velocity_metrics['cv_demand'] = velocity_metrics['std_qty'] / velocity_metrics['avg_qty']
        
        # ABC Classification (by revenue)
        velocity_metrics['revenue_rank'] = velocity_metrics['revenue'].rank(ascending=False, pct=True)
        velocity_metrics['abc_class'] = velocity_metrics['revenue_rank'].apply(
            lambda x: 'A' if x <= 0.2 else 'B' if x <= 0.5 else 'C'
        )
        
        # XYZ Classification (by demand variability)
        cv_thresholds = velocity_metrics['cv_demand'].quantile([0.33, 0.67])
        velocity_metrics['xyz_class'] = velocity_metrics['cv_demand'].apply(
            lambda x: 'X' if x <= cv_thresholds[0.33] else 'Y' if x <= cv_thresholds[0.67] else 'Z'
        )
        
        # Combined classification
        velocity_metrics['velocity_segment'] = velocity_metrics['abc_class'] + velocity_metrics['xyz_class']
        
        # Calculate segment statistics
        segment_stats = velocity_metrics.groupby('velocity_segment').agg({
            'total_qty': ['count', 'sum', 'mean'],
            'revenue': ['sum', 'mean'],
            'cv_demand': 'mean',
            'frequency': 'mean'
        }).round(2)
        
        print("\nVELOCITY SEGMENT DISTRIBUTION:")
        segment_counts = velocity_metrics['velocity_segment'].value_counts().sort_index()
        for segment, count in segment_counts.items():
            pct = count / len(velocity_metrics) * 100
            revenue_pct = velocity_metrics[velocity_metrics['velocity_segment'] == segment]['revenue'].sum() / velocity_metrics['revenue'].sum() * 100
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
        print(f"  Fast-moving revenue share: {fast_moving['revenue'].sum()/velocity_metrics['revenue'].sum()*100:.1f}%")
        
        self.results['velocity_analysis'] = {
            'velocity_metrics': velocity_metrics,
            'segment_stats': segment_stats,
            'movement_classification': {
                'fast_moving_count': len(fast_moving),
                'slow_moving_count': len(slow_moving),
                'fast_moving_revenue_share': fast_moving['revenue'].sum()/velocity_metrics['revenue'].sum(),
                'velocity_threshold': velocity_threshold
            }
        }
        
        # Create visualizations
        self._create_velocity_visualizations(velocity_metrics)
        
        return velocity_metrics

    def detect_seasonality_patterns(self):
        """
        Advanced seasonality detection using multiple methods
        STL decomposition and pattern analysis
        """
        print("\n" + "="*60)
        print("SEASONALITY DETECTION ANALYSIS")
        print("="*60)
        
        # Prepare time series data
        daily_sales = self.trans_df.groupby('transaction_date')['quantity'].sum().reset_index()
        daily_sales.set_index('transaction_date', inplace=True)
        daily_sales = daily_sales.resample('D').sum().fillna(0)
        
        seasonality_results = {}
        
        try:
            # STL Decomposition
            stl = STL(daily_sales['quantity'], seasonal=13)  # Weekly seasonality
            stl_result = stl.fit()
            
            # Calculate seasonality strength
            seasonal_var = stl_result.seasonal.var()
            residual_var = stl_result.resid.var()
            seasonality_strength = seasonal_var / (seasonal_var + residual_var)
            
            print(f"STL DECOMPOSITION RESULTS:")
            print(f"  Seasonality strength: {seasonality_strength:.3f}")
            print(f"  Trend strength: {stl_result.trend.var() / daily_sales['quantity'].var():.3f}")
            
            seasonality_results['stl'] = {
                'seasonality_strength': seasonality_strength,
                'seasonal_component': stl_result.seasonal,
                'trend_component': stl_result.trend,
                'residual_component': stl_result.resid
            }
            
        except Exception as e:
            print(f"[WARNING] STL decomposition failed: {e}")
        
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
        
        # Detect seasonal products
        product_seasonality = self.trans_df.groupby(['internal_product_id', 'month'])['quantity'].sum().unstack(fill_value=0)
        
        # Calculate seasonality index for each product
        seasonal_indices = {}
        for produto_id in product_seasonality.index:
            monthly_sales = product_seasonality.loc[produto_id]
            mean_sales = monthly_sales.mean()
            if mean_sales > 0:
                seasonal_index = (monthly_sales.max() - monthly_sales.min()) / mean_sales
                seasonal_indices[produto_id] = seasonal_index
        
        # Classify products by seasonality
        seasonal_threshold = np.percentile(list(seasonal_indices.values()), 80)
        highly_seasonal = [pid for pid, idx in seasonal_indices.items() if idx >= seasonal_threshold]
        
        print(f"\nSEASONALITY CLASSIFICATION:")
        print(f"  Highly seasonal products: {len(highly_seasonal):,} ({len(highly_seasonal)/len(seasonal_indices)*100:.1f}%)")
        print(f"  Seasonality threshold: {seasonal_threshold:.2f}")
        
        seasonality_results.update({
            'weekly_patterns': weekly_patterns,
            'monthly_patterns': monthly_patterns,
            'seasonal_indices': seasonal_indices,
            'highly_seasonal_products': highly_seasonal,
            'seasonal_threshold': seasonal_threshold
        })
        
        self.results['seasonality_analysis'] = seasonality_results
        
        # Create seasonality visualizations
        self._create_seasonality_visualizations(seasonality_results, daily_sales)
        
        return seasonality_results

    def analyze_cross_selling_patterns(self):
        """
        Market Basket Analysis - Cross-selling patterns discovery
        """
        print("\n" + "="*60)
        print("CROSS-SELLING PATTERNS ANALYSIS")
        print("="*60)
        
        # Prepare transaction data for market basket analysis
        # Group by date and PDV to create market baskets
        transactions_basket = self.trans_df.groupby(['data_transacao', 'pdv_id'])['produto_id'].apply(list).reset_index()
        transactions_basket = transactions_basket['produto_id'].tolist()
        
        # Remove single-item transactions for association analysis
        transactions_basket = [basket for basket in transactions_basket if len(basket) > 1]
        
        print(f"MARKET BASKET PREPARATION:")
        print(f"  Total baskets: {len(transactions_basket):,}")
        print(f"  Multi-item baskets: {len(transactions_basket):,}")
        
        try:
            # Encode transactions for frequent itemset mining
            te = TransactionEncoder()
            te_ary = te.fit(transactions_basket).transform(transactions_basket)
            basket_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True, max_len=2)
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                
                if len(rules) > 0:
                    # Sort by lift (interestingness)
                    rules = rules.sort_values('lift', ascending=False)
                    
                    print(f"\nASSOCIATION RULES DISCOVERED:")
                    print(f"  Total rules: {len(rules):,}")
                    print(f"  High confidence rules (>50%): {len(rules[rules['confidence'] > 0.5]):,}")
                    print(f"  High lift rules (>2.0): {len(rules[rules['lift'] > 2.0]):,}")
                    
                    # Display top rules
                    print(f"\nTOP 10 ASSOCIATION RULES:")
                    for i, (idx, rule) in enumerate(rules.head(10).iterrows()):
                        antecedent = list(rule['antecedents'])[0]
                        consequent = list(rule['consequents'])[0]
                        print(f"  {i+1}. {antecedent} → {consequent}")
                        print(f"     Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.2f}")
                    
                    # Product affinity analysis
                    product_pairs = rules[['antecedents', 'consequents', 'lift']].copy()
                    product_pairs['pair'] = product_pairs.apply(
                        lambda x: f"{list(x['antecedents'])[0]}-{list(x['consequents'])[0]}", axis=1
                    )
                    
                    cross_selling_results = {
                        'frequent_itemsets': frequent_itemsets,
                        'association_rules': rules,
                        'top_rules': rules.head(20),
                        'high_lift_pairs': rules[rules['lift'] > 2.0],
                        'stats': {
                            'total_rules': len(rules),
                            'high_confidence_rules': len(rules[rules['confidence'] > 0.5]),
                            'high_lift_rules': len(rules[rules['lift'] > 2.0]),
                            'avg_lift': rules['lift'].mean(),
                            'avg_confidence': rules['confidence'].mean()
                        }
                    }
                    
                else:
                    print("[WARNING] No association rules found with minimum thresholds")
                    cross_selling_results = {'error': 'No rules found'}
                    
            else:
                print("[WARNING] No frequent itemsets found")
                cross_selling_results = {'error': 'No frequent itemsets'}
                
        except Exception as e:
            print(f"[ERROR] Market basket analysis failed: {e}")
            cross_selling_results = {'error': str(e)}
        
        # Alternative: Simple co-occurrence analysis
        print(f"\nALTERNATIVE: CO-OCCURRENCE ANALYSIS")
        co_occurrence_matrix = self._calculate_product_cooccurrence()
        cross_selling_results['co_occurrence_matrix'] = co_occurrence_matrix
        
        self.results['cross_selling_analysis'] = cross_selling_results
        
        return cross_selling_results

    def _calculate_product_cooccurrence(self):
        """Calculate product co-occurrence matrix"""
        
        # Get products that appear together in transactions (same date/PDV)
        transaction_groups = self.trans_df.groupby(['data_transacao', 'pdv_id'])['produto_id'].apply(set)
        
        # Count co-occurrences
        co_occurrence = {}
        product_counts = {}
        
        for products in transaction_groups:
            if len(products) > 1:
                products_list = list(products)
                for i, prod1 in enumerate(products_list):
                    product_counts[prod1] = product_counts.get(prod1, 0) + 1
                    for prod2 in products_list[i+1:]:
                        pair = tuple(sorted([prod1, prod2]))
                        co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
        
        # Calculate co-occurrence scores
        cooc_results = []
        for (prod1, prod2), count in co_occurrence.items():
            if count >= 10:  # Minimum co-occurrence threshold
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
        
        print(f"  Co-occurrence pairs found: {len(cooc_df):,}")
        if len(cooc_df) > 0:
            print(f"  High lift pairs (>2.0): {len(cooc_df[cooc_df['lift'] > 2.0]):,}")
            print(f"  Average lift: {cooc_df['lift'].mean():.2f}")
        
        return cooc_df

    def analyze_price_elasticity(self):
        """
        Price elasticity analysis - Impact of prices on demand
        """
        print("\n" + "="*60)
        print("PRICE ELASTICITY ANALYSIS")
        print("="*60)
        
        # Prepare data for elasticity analysis
        price_demand_data = self.trans_df.groupby(['produto_id', 'preco_unitario'])['quantidade'].sum().reset_index()
        
        elasticity_results = {}
        product_elasticities = []
        
        # Analyze elasticity by product (minimum 5 different price points)
        for produto_id in price_demand_data['produto_id'].unique():
            product_data = price_demand_data[price_demand_data['produto_id'] == produto_id]
            
            if len(product_data) >= 5:  # Need multiple price points
                try:
                    # Calculate price elasticity using log-log regression
                    log_price = np.log(product_data['preco_unitario'])
                    log_quantity = np.log(product_data['quantidade'] + 1)  # Add 1 to handle zeros
                    
                    correlation, p_value = pearsonr(log_price, log_quantity)
                    
                    if p_value < 0.05:  # Significant correlation
                        # Simple elasticity estimate (slope of log-log relationship)
                        elasticity = correlation
                        
                        product_elasticities.append({
                            'produto_id': produto_id,
                            'elasticity': elasticity,
                            'correlation': correlation,
                            'p_value': p_value,
                            'price_points': len(product_data),
                            'avg_price': product_data['preco_unitario'].mean(),
                            'total_demand': product_data['quantidade'].sum()
                        })
                
                except Exception as e:
                    continue
        
        elasticity_df = pd.DataFrame(product_elasticities)
        
        if len(elasticity_df) > 0:
            # Classify products by elasticity
            elasticity_df['elasticity_category'] = elasticity_df['elasticity'].apply(
                lambda x: 'Highly Elastic' if x < -1.5 else 
                         'Elastic' if x < -0.5 else
                         'Inelastic' if x > -0.1 else
                         'Moderately Elastic'
            )
            
            print(f"PRICE ELASTICITY RESULTS:")
            print(f"  Products analyzed: {len(elasticity_df):,}")
            
            elasticity_counts = elasticity_df['elasticity_category'].value_counts()
            for category, count in elasticity_counts.items():
                avg_elasticity = elasticity_df[elasticity_df['elasticity_category'] == category]['elasticity'].mean()
                print(f"  {category}: {count} products (avg elasticity: {avg_elasticity:.2f})")
            
            # Revenue optimization insights
            elastic_products = elasticity_df[elasticity_df['elasticity'] < -1.0]
            inelastic_products = elasticity_df[elasticity_df['elasticity'] > -0.5]
            
            print(f"\nREVENUE OPTIMIZATION INSIGHTS:")
            print(f"  Price-sensitive products: {len(elastic_products)} (consider price reductions)")
            print(f"  Price-insensitive products: {len(inelastic_products)} (consider price increases)")
            
            elasticity_results = {
                'elasticity_analysis': elasticity_df,
                'category_distribution': elasticity_counts.to_dict(),
                'optimization_opportunities': {
                    'elastic_products': len(elastic_products),
                    'inelastic_products': len(inelastic_products),
                    'avg_elasticity': elasticity_df['elasticity'].mean()
                }
            }
        else:
            print("[WARNING] Insufficient data for price elasticity analysis")
            elasticity_results = {'error': 'Insufficient data'}
        
        self.results['price_elasticity_analysis'] = elasticity_results
        
        return elasticity_results

    def detect_promotion_effects(self):
        """
        Promotion effects detection and analysis
        """
        print("\n" + "="*60) 
        print("PROMOTION EFFECTS DETECTION")
        print("="*60)
        
        # Detect potential promotion periods using price drops
        product_prices = self.trans_df.groupby(['produto_id', 'data_transacao'])['preco_unitario'].mean().reset_index()
        
        promotion_effects = []
        
        for produto_id in product_prices['produto_id'].unique()[:1000]:  # Sample for performance
            product_data = product_prices[product_prices['produto_id'] == produto_id].sort_values('data_transacao')
            
            if len(product_data) >= 10:  # Need sufficient history
                # Calculate rolling average price
                product_data['price_ma'] = product_data['preco_unitario'].rolling(window=7, min_periods=1).mean()
                
                # Detect price drops (potential promotions)
                price_threshold = product_data['price_ma'].std() * 1.5
                product_data['is_promotion'] = (product_data['price_ma'] - product_data['preco_unitario']) > price_threshold
                
                if product_data['is_promotion'].sum() > 0:
                    # Analyze promotion effects
                    promotion_periods = product_data[product_data['is_promotion']]
                    normal_periods = product_data[~product_data['is_promotion']]
                    
                    # Get corresponding sales data
                    promo_sales = self.trans_df[
                        (self.trans_df['produto_id'] == produto_id) & 
                        (self.trans_df['data_transacao'].isin(promotion_periods['data_transacao']))
                    ]['quantidade'].sum()
                    
                    normal_sales = self.trans_df[
                        (self.trans_df['produto_id'] == produto_id) & 
                        (self.trans_df['data_transacao'].isin(normal_periods['data_transacao']))
                    ]['quantidade'].sum()
                    
                    if normal_sales > 0:
                        # Calculate lift
                        promo_days = len(promotion_periods)
                        normal_days = len(normal_periods)
                        
                        if promo_days > 0 and normal_days > 0:
                            promo_daily_avg = promo_sales / promo_days
                            normal_daily_avg = normal_sales / normal_days
                            
                            lift = (promo_daily_avg - normal_daily_avg) / normal_daily_avg if normal_daily_avg > 0 else 0
                            
                            if lift > 0.1:  # Significant lift threshold
                                avg_discount = (promotion_periods['price_ma'] - promotion_periods['preco_unitario']).mean()
                                discount_pct = avg_discount / promotion_periods['price_ma'].mean() * 100
                                
                                promotion_effects.append({
                                    'produto_id': produto_id,
                                    'promotion_periods': promo_days,
                                    'avg_discount_pct': discount_pct,
                                    'sales_lift': lift,
                                    'promo_daily_avg': promo_daily_avg,
                                    'normal_daily_avg': normal_daily_avg,
                                    'total_promo_sales': promo_sales
                                })
        
        promotion_df = pd.DataFrame(promotion_effects)
        
        if len(promotion_df) > 0:
            # Analyze promotion effectiveness
            print(f"PROMOTION EFFECTS ANALYSIS:")
            print(f"  Products with detected promotions: {len(promotion_df):,}")
            print(f"  Average sales lift: {promotion_df['sales_lift'].mean():.1%}")
            print(f"  Average discount: {promotion_df['avg_discount_pct'].mean():.1f}%")
            
            # Classify promotion effectiveness
            promotion_df['effectiveness'] = promotion_df['sales_lift'].apply(
                lambda x: 'Highly Effective' if x > 1.0 else
                         'Effective' if x > 0.5 else
                         'Moderately Effective' if x > 0.2 else
                         'Low Impact'
            )
            
            effectiveness_counts = promotion_df['effectiveness'].value_counts()
            print(f"\nPROMOTION EFFECTIVENESS:")
            for category, count in effectiveness_counts.items():
                avg_lift = promotion_df[promotion_df['effectiveness'] == category]['sales_lift'].mean()
                print(f"  {category}: {count} products (avg lift: {avg_lift:.1%})")
            
            # ROI analysis
            # Simple ROI estimate (lift vs discount)
            promotion_df['roi_estimate'] = promotion_df['sales_lift'] / (promotion_df['avg_discount_pct'] / 100)
            
            print(f"\nROI INSIGHTS:")
            high_roi_promos = promotion_df[promotion_df['roi_estimate'] > 2.0]
            print(f"  High ROI promotions: {len(high_roi_promos)} products")
            print(f"  Average promotion ROI: {promotion_df['roi_estimate'].mean():.1f}x")
            
            promotion_results = {
                'promotion_analysis': promotion_df,
                'effectiveness_distribution': effectiveness_counts.to_dict(),
                'summary_stats': {
                    'avg_lift': promotion_df['sales_lift'].mean(),
                    'avg_discount': promotion_df['avg_discount_pct'].mean(),
                    'avg_roi': promotion_df['roi_estimate'].mean(),
                    'high_roi_count': len(high_roi_promos)
                }
            }
        else:
            print("[WARNING] No significant promotion effects detected")
            promotion_results = {'error': 'No promotions detected'}
        
        self.results['promotion_analysis'] = promotion_results
        
        return promotion_results

    def _create_velocity_visualizations(self, velocity_metrics):
        """Create velocity analysis visualizations"""
        
        try:
            # ABC/XYZ matrix plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # ABC distribution
            abc_counts = velocity_metrics['abc_class'].value_counts()
            axes[0,0].pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%')
            axes[0,0].set_title('ABC Classification Distribution')
            
            # XYZ distribution  
            xyz_counts = velocity_metrics['xyz_class'].value_counts()
            axes[0,1].pie(xyz_counts.values, labels=xyz_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('XYZ Classification Distribution')
            
            # Revenue vs Frequency scatter
            axes[1,0].scatter(velocity_metrics['frequency'], velocity_metrics['revenue'], 
                             alpha=0.6, c=velocity_metrics['abc_class'].map({'A': 'red', 'B': 'orange', 'C': 'blue'}))
            axes[1,0].set_xlabel('Frequency')
            axes[1,0].set_ylabel('Revenue')
            axes[1,0].set_title('Revenue vs Frequency by ABC Class')
            
            # CV vs Average Quantity
            axes[1,1].scatter(velocity_metrics['avg_qty'], velocity_metrics['cv_demand'],
                             alpha=0.6, c=velocity_metrics['xyz_class'].map({'X': 'green', 'Y': 'yellow', 'Z': 'red'}))
            axes[1,1].set_xlabel('Average Quantity')
            axes[1,1].set_ylabel('Coefficient of Variation')
            axes[1,1].set_title('Demand Variability by XYZ Class')
            
            plt.tight_layout()
            plt.savefig('../../data/processed/velocity_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"[WARNING] Visualization creation failed: {e}")

    def _create_seasonality_visualizations(self, seasonality_results, daily_sales):
        """Create seasonality analysis visualizations"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Weekly patterns
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_means = seasonality_results['weekly_patterns']['mean'].values
            axes[0,0].bar(days, weekly_means)
            axes[0,0].set_title('Weekly Seasonality Pattern')
            axes[0,0].set_ylabel('Average Quantity')
            plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
            
            # Monthly patterns
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_means = seasonality_results['monthly_patterns']['mean'].values
            axes[0,1].bar(months, monthly_means)
            axes[0,1].set_title('Monthly Seasonality Pattern')
            axes[0,1].set_ylabel('Average Quantity')
            plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
            
            # Time series plot
            axes[1,0].plot(daily_sales.index, daily_sales['quantidade'])
            axes[1,0].set_title('Daily Sales Time Series')
            axes[1,0].set_ylabel('Total Quantity')
            
            # Seasonality index distribution
            seasonal_indices = list(seasonality_results['seasonal_indices'].values())
            axes[1,1].hist(seasonal_indices, bins=30, alpha=0.7)
            axes[1,1].set_title('Product Seasonality Index Distribution')
            axes[1,1].set_xlabel('Seasonality Index')
            axes[1,1].set_ylabel('Number of Products')
            
            plt.tight_layout()
            plt.savefig('../../data/processed/seasonality_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"[WARNING] Seasonality visualization failed: {e}")

    def generate_retail_insights_summary(self):
        """Generate comprehensive retail insights summary"""
        
        print("\n" + "="*80)
        print("RETAIL INSIGHTS EXECUTIVE SUMMARY")
        print("="*80)
        
        insights = {
            'critical_insights': [],
            'actionable_recommendations': [],
            'feature_engineering_opportunities': [],
            'competitive_advantages': []
        }
        
        # Velocity insights
        if 'velocity_analysis' in self.results:
            velocity_data = self.results['velocity_analysis']
            fast_revenue_share = velocity_data['movement_classification']['fast_moving_revenue_share']
            
            insights['critical_insights'].append(
                f"Fast-moving products represent {fast_revenue_share:.1%} of total revenue despite being minority of catalog"
            )
            
            insights['actionable_recommendations'].append(
                "Focus forecasting accuracy on A-class products for maximum revenue impact"
            )
            
            insights['feature_engineering_opportunities'].append(
                "Create velocity-based features: ABC/XYZ segments, movement speed indicators"
            )
        
        # Seasonality insights
        if 'seasonality_analysis' in self.results:
            seasonal_data = self.results['seasonality_analysis']
            if 'highly_seasonal_products' in seasonal_data:
                seasonal_pct = len(seasonal_data['highly_seasonal_products']) / len(seasonal_data['seasonal_indices']) * 100
                
                insights['critical_insights'].append(
                    f"{seasonal_pct:.1f}% of products show strong seasonality requiring specialized handling"
                )
                
                insights['feature_engineering_opportunities'].append(
                    "Develop seasonal decomposition features, holiday indicators, trend-season interactions"
                )
        
        # Cross-selling insights
        if 'cross_selling_analysis' in self.results:
            cross_data = self.results['cross_selling_analysis']
            if 'stats' in cross_data:
                avg_lift = cross_data['stats']['avg_lift']
                insights['critical_insights'].append(
                    f"Cross-selling opportunities identified with average lift of {avg_lift:.2f}x"
                )
                
                insights['feature_engineering_opportunities'].append(
                    "Create product affinity features, basket composition indicators"
                )
        
        # Price elasticity insights
        if 'price_elasticity_analysis' in self.results:
            price_data = self.results['price_elasticity_analysis']
            if 'optimization_opportunities' in price_data:
                elastic_count = price_data['optimization_opportunities']['elastic_products']
                insights['actionable_recommendations'].append(
                    f"Optimize pricing for {elastic_count} price-sensitive products to boost volume"
                )
        
        # Promotion insights
        if 'promotion_analysis' in self.results:
            promo_data = self.results['promotion_analysis']
            if 'summary_stats' in promo_data:
                avg_lift = promo_data['summary_stats']['avg_lift']
                insights['critical_insights'].append(
                    f"Promotions generate average {avg_lift:.1%} sales lift when executed properly"
                )
        
        # Competitive advantages
        insights['competitive_advantages'].extend([
            "Advanced velocity segmentation beyond simple ABC analysis",
            "Multi-method seasonality detection with automatic pattern recognition",
            "Cross-selling intelligence for demand spillover effects",
            "Promotion impact quantification for marketing mix modeling",
            "Price elasticity insights for revenue optimization"
        ])
        
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
        
        print("\nCOMPETITIVE ADVANTAGES FOR HACKATHON:")
        for i, adv in enumerate(insights['competitive_advantages'], 1):
            print(f"  {i}. {adv}")
        
        self.results['executive_summary'] = insights
        
        return insights

    def save_retail_insights_results(self, output_path: str = "../../data/processed/eda_results"):
        """Save comprehensive retail insights analysis results"""
        
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
                return obj.to_dict('records')
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
        
        # Execute comprehensive retail insights analysis
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
        print(f"[OK] Evaluated {len(prod_df):,} product records") 
        print(f"[OK] Generated {len(executive_summary['critical_insights'])} critical business insights")
        print(f"[OK] Identified {len(executive_summary['feature_engineering_opportunities'])} feature opportunities")
        print(f"[OK] Created {len(executive_summary['competitive_advantages'])} competitive advantages")
        print(f"[OK] Results saved to: {results_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Retail insights analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()