#!/usr/bin/env python3
"""
COMPREHENSIVE TEMPORAL ANALYSIS - Hackathon Forecast Big Data 2025
Phase 2.1: Análise Temporal Profunda & Sazonalidade

Objetivos:
- Identificar padrões semanais, mensais, trimestrais e anuais
- Detectar eventos especiais e feriados
- Analisar trends de longo prazo por categoria
- Calcular autocorrelações e identificar lags relevantes
- Realizar decomposição STL e changepoint detection
- Otimizar insights para métrica WMAPE

Estratégia: Análise sistemática focada em retail forecasting
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

from utils.data_loader import OptimizedDataLoader
from evaluation.metrics import wmape

try:
    from scipy import stats
except ImportError:
    print("[WARNING] scipy missing - using basic statistics")
    
# Advanced time series libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import acf, pacf
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("[OK] Advanced time series libraries loaded")
except ImportError as e:
    print(f"[WARNING] Some advanced libraries missing: {e}")
    print("Basic analysis will continue with available libraries")

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TemporalAnalyzer:
    """Comprehensive temporal analysis for retail forecasting data"""
    
    def __init__(self, data_path: str = "../../data/raw"):
        self.data_path = data_path
        self.loader = OptimizedDataLoader(data_path)
        self.results = {}
        
    def load_and_prepare_data(self, sample_size: int = 1000000):
        """Load and prepare transaction data for temporal analysis"""
        
        print("="*60)
        print("LOADING DATA FOR TEMPORAL ANALYSIS")
        print("="*60)
        
        # Load transaction data with larger sample for robust temporal analysis
        self.trans_df = self.loader.load_transactions(sample_size=sample_size)
        print(f"Loaded {len(self.trans_df):,} transaction records")
        
        # Prepare date column
        if 'transaction_date' in self.trans_df.columns:
            self.date_col = 'transaction_date'
        else:
            # Find date column
            date_cols = [col for col in self.trans_df.columns if 'date' in col.lower()]
            if date_cols:
                self.date_col = date_cols[0]
            else:
                raise ValueError("No date column found in transaction data")
        
        print(f"Using date column: {self.date_col}")
        
        # Convert to datetime
        self.trans_df[self.date_col] = pd.to_datetime(self.trans_df[self.date_col])
        
        # Create additional temporal features
        self.trans_df['year'] = self.trans_df[self.date_col].dt.year
        self.trans_df['month'] = self.trans_df[self.date_col].dt.month
        self.trans_df['week'] = self.trans_df[self.date_col].dt.isocalendar().week
        self.trans_df['dayofweek'] = self.trans_df[self.date_col].dt.dayofweek
        self.trans_df['dayofyear'] = self.trans_df[self.date_col].dt.dayofyear
        self.trans_df['is_weekend'] = self.trans_df['dayofweek'].isin([5, 6])
        
        # Create time series aggregations
        self.daily_sales = self.trans_df.groupby(self.date_col).agg({
            'quantity': 'sum',
            'gross_value': 'sum' if 'gross_value' in self.trans_df.columns else 'count',
            'internal_store_id': 'nunique',
            'internal_product_id': 'nunique'
        }).reset_index()
        
        self.daily_sales.columns = [self.date_col, 'total_quantity', 'total_value', 'active_stores', 'active_products']
        
        print(f"Created daily aggregation: {len(self.daily_sales)} days")
        print(f"Date range: {self.daily_sales[self.date_col].min()} to {self.daily_sales[self.date_col].max()}")
        
        return self.trans_df, self.daily_sales
    
    def analyze_temporal_patterns(self):
        """Comprehensive temporal pattern analysis"""
        
        print("\n" + "="*60)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("="*60)
        
        patterns = {}
        
        # 1. Weekly Patterns
        print("\n1. WEEKLY PATTERNS")
        print("-" * 40)
        
        weekly_patterns = self.trans_df.groupby('dayofweek')['quantity'].agg(['count', 'sum', 'mean']).reset_index()
        weekly_patterns['day_name'] = weekly_patterns['dayofweek'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        print("Average sales by day of week:")
        for _, row in weekly_patterns.iterrows():
            print(f"  {row['day_name']}: {row['sum']:,.0f} units ({row['mean']:.2f} avg)")
        
        # Weekend vs weekday analysis
        weekend_sales = self.trans_df[self.trans_df['is_weekend']]['quantity'].sum()
        weekday_sales = self.trans_df[~self.trans_df['is_weekend']]['quantity'].sum()
        weekend_ratio = weekend_sales / (weekend_sales + weekday_sales)
        
        print(f"\nWeekend vs Weekday:")
        print(f"  Weekend share: {weekend_ratio:.1%}")
        print(f"  Weekend intensity: {weekend_sales/2:.0f} units/day")
        print(f"  Weekday intensity: {weekday_sales/5:.0f} units/day")
        
        patterns['weekly'] = {
            'by_day': weekly_patterns.to_dict('records'),
            'weekend_share': weekend_ratio,
            'weekend_intensity_ratio': (weekend_sales/2) / (weekday_sales/5)
        }
        
        # 2. Monthly Patterns
        print("\n2. MONTHLY PATTERNS")
        print("-" * 40)
        
        monthly_patterns = self.trans_df.groupby('month')['quantity'].agg(['count', 'sum', 'mean']).reset_index()
        monthly_patterns['month_name'] = monthly_patterns['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        print("Sales by month:")
        for _, row in monthly_patterns.iterrows():
            print(f"  {row['month_name']}: {row['sum']:,.0f} units")
        
        # Identify seasonal months
        month_avg = monthly_patterns['sum'].mean()
        high_season = monthly_patterns[monthly_patterns['sum'] > month_avg * 1.1]['month_name'].tolist()
        low_season = monthly_patterns[monthly_patterns['sum'] < month_avg * 0.9]['month_name'].tolist()
        
        print(f"\nSeasonal Analysis:")
        print(f"  High season months: {', '.join(high_season) if high_season else 'None identified'}")
        print(f"  Low season months: {', '.join(low_season) if low_season else 'None identified'}")
        
        patterns['monthly'] = {
            'by_month': monthly_patterns.to_dict('records'),
            'high_season': high_season,
            'low_season': low_season,
            'seasonality_strength': monthly_patterns['sum'].std() / monthly_patterns['sum'].mean()
        }
        
        # 3. Trend Analysis
        print("\n3. TREND ANALYSIS")
        print("-" * 40)
        
        # Daily trend over time
        self.daily_sales['ma_7'] = self.daily_sales['total_quantity'].rolling(window=7, center=True).mean()
        self.daily_sales['ma_30'] = self.daily_sales['total_quantity'].rolling(window=30, center=True).mean()
        
        # Calculate overall trend
        x = np.arange(len(self.daily_sales))
        trend_slope, trend_intercept, trend_r, _, _ = stats.linregress(x, self.daily_sales['total_quantity'])
        
        # Trend classification
        trend_pct_per_day = (trend_slope / self.daily_sales['total_quantity'].mean()) * 100
        
        if abs(trend_pct_per_day) < 0.01:
            trend_class = "Stable"
        elif trend_pct_per_day > 0.01:
            trend_class = "Growing"
        else:
            trend_class = "Declining"
        
        print(f"Overall trend: {trend_class}")
        print(f"  Slope: {trend_slope:.2f} units/day ({trend_pct_per_day:+.3f}% per day)")
        print(f"  R-squared: {trend_r**2:.3f}")
        
        patterns['trend'] = {
            'classification': trend_class,
            'slope_per_day': trend_slope,
            'slope_pct_per_day': trend_pct_per_day,
            'r_squared': trend_r**2,
            'is_significant': abs(trend_pct_per_day) > 0.01
        }
        
        self.results['temporal_patterns'] = patterns
        return patterns
    
    def analyze_autocorrelation(self, max_lags: int = 52):
        """Analyze autocorrelation to identify important lags"""
        
        print("\n" + "="*60)
        print("AUTOCORRELATION ANALYSIS")
        print("="*60)
        
        # Prepare weekly aggregated data for better autocorrelation analysis
        self.daily_sales['week_start'] = self.daily_sales[self.date_col] - pd.to_timedelta(
            self.daily_sales[self.date_col].dt.dayofweek, unit='D'
        )
        
        weekly_sales = self.daily_sales.groupby('week_start')['total_quantity'].sum().reset_index()
        weekly_sales = weekly_sales.sort_values('week_start')
        
        print(f"Analyzing {len(weekly_sales)} weeks of data")
        
        # Calculate autocorrelation
        try:
            autocorr = acf(weekly_sales['total_quantity'], nlags=min(max_lags, len(weekly_sales)-1), fft=True)
            partial_autocorr = pacf(weekly_sales['total_quantity'], nlags=min(20, len(weekly_sales)//4), method='ywm')
            
            # Find significant lags (absolute correlation > 0.1)
            significant_lags = []
            for i, corr in enumerate(autocorr[1:], 1):  # Skip lag 0
                if abs(corr) > 0.1:
                    significant_lags.append((i, corr))
            
            print(f"\nSignificant autocorrelations (>0.1):")
            for lag, corr in significant_lags[:10]:  # Top 10
                print(f"  Lag {lag:2d} weeks: {corr:6.3f}")
            
            # Identify seasonal patterns from autocorrelation
            seasonal_indicators = {
                'monthly': autocorr[4] if len(autocorr) > 4 else 0,  # ~1 month
                'quarterly': autocorr[13] if len(autocorr) > 13 else 0,  # ~3 months  
                'semi_annual': autocorr[26] if len(autocorr) > 26 else 0,  # ~6 months
                'annual': autocorr[52] if len(autocorr) > 52 else 0,  # 1 year
            }
            
            print(f"\nSeasonal autocorrelations:")
            for period, corr in seasonal_indicators.items():
                print(f"  {period:12s}: {corr:6.3f}")
            
            autocorr_results = {
                'autocorr_values': autocorr.tolist(),
                'partial_autocorr': partial_autocorr.tolist(),
                'significant_lags': significant_lags,
                'seasonal_indicators': seasonal_indicators,
                'recommended_lags': [lag for lag, _ in significant_lags[:5]]  # Top 5 lags
            }
            
        except Exception as e:
            print(f"[WARNING] Autocorrelation analysis failed: {e}")
            autocorr_results = {'error': str(e)}
        
        self.results['autocorrelation'] = autocorr_results
        return autocorr_results
    
    def detect_special_events(self):
        """Detect and analyze special events and outliers"""
        
        print("\n" + "="*60)
        print("SPECIAL EVENTS & OUTLIER DETECTION")
        print("="*60)
        
        # Calculate daily statistics for outlier detection
        daily_stats = self.daily_sales['total_quantity'].describe()
        q1, q3 = daily_stats['25%'], daily_stats['75%']
        iqr = q3 - q1
        
        # Define outliers using IQR method
        outlier_threshold_high = q3 + 1.5 * iqr
        outlier_threshold_low = q1 - 1.5 * iqr
        
        # Identify outlier days
        outlier_days = self.daily_sales[
            (self.daily_sales['total_quantity'] > outlier_threshold_high) |
            (self.daily_sales['total_quantity'] < outlier_threshold_low)
        ].copy()
        
        print(f"Identified {len(outlier_days)} outlier days out of {len(self.daily_sales)} total days")
        
        if len(outlier_days) > 0:
            print(f"\nTop outlier days (by volume):")
            outlier_days_sorted = outlier_days.sort_values('total_quantity', ascending=False)
            
            for i, (_, day) in enumerate(outlier_days_sorted.head(10).iterrows()):
                date_str = day[self.date_col].strftime('%Y-%m-%d (%A)')
                print(f"  {i+1:2d}. {date_str}: {day['total_quantity']:,.0f} units")
        
        # Brazilian retail calendar events (comprehensive 2022)
        brazilian_events = {
            # Major holidays
            'new_year': [(1, 1)],
            'carnival': [(2, 26), (2, 27), (2, 28), (3, 1)],  # 2022 dates
            'tiradentes': [(4, 21)],
            'labor_day': [(5, 1)],
            'independence_day': [(9, 7)],
            'our_lady_aparecida': [(10, 12)],
            'all_souls': [(11, 2)],
            'proclamation_republic': [(11, 15)],
            'christmas': [(12, 25)],
            
            # Commercial events
            'mothers_day': [(5, 8)],                   # 2nd Sunday May 2022
            'valentines_day': [(6, 12)],               # Dia dos Namorados
            'fathers_day': [(8, 14)],                  # 2nd Sunday Aug 2022
            'childrens_day': [(10, 12)],               # Same as Our Lady
            'black_friday': [(11, 25)],                # 2022 date
            'cyber_monday': [(11, 28)],
            'christmas_shopping': [(12, 15), (12, 16), (12, 17)],  # Peak week
            
            # Seasonal/Cultural
            'festa_junina': [(6, 24)],                 # São João
            'back_to_school': [(1, 31), (2, 1), (2, 2)],  # January-Feb
            'winter_sales': [(6, 20), (7, 20), (8, 20)],   # Winter period
            'summer_sales': [(1, 15), (12, 15)],       # Summer sales
            
            # Pay periods (salary impact)
            'salary_week_1': [(m, d) for m in range(1, 13) for d in [5, 6, 7]],  # First week
            'salary_week_2': [(m, d) for m in range(1, 13) for d in [25, 26, 27]] # Last week
        }
        
        # Check if outlier days coincide with known events
        event_matches = []
        for _, day in outlier_days.iterrows():
            date_obj = day[self.date_col]
            month_day = (date_obj.month, date_obj.day)
            
            for event_name, dates in brazilian_events.items():
                if month_day in dates:
                    event_matches.append({
                        'date': date_obj,
                        'event': event_name,
                        'volume': day['total_quantity']
                    })
        
        if event_matches:
            print(f"\nOutlier days matching known events:")
            for match in event_matches:
                print(f"  {match['date'].strftime('%Y-%m-%d')}: {match['event']} ({match['volume']:,.0f} units)")
        
        special_events_results = {
            'outlier_threshold_high': outlier_threshold_high,
            'outlier_threshold_low': outlier_threshold_low,
            'n_outlier_days': len(outlier_days),
            'outlier_days': outlier_days_sorted.head(20).to_dict('records'),
            'event_matches': event_matches,
            'outlier_percentage': len(outlier_days) / len(self.daily_sales) * 100
        }
        
        self.results['special_events'] = special_events_results
        return special_events_results
    
    def analyze_stl_decomposition(self):
        """Perform STL (Seasonal and Trend decomposition using Loess) analysis"""
        
        print("\n" + "="*60)
        print("STL DECOMPOSITION ANALYSIS")
        print("="*60)
        
        try:
            # Prepare weekly data for STL
            weekly_data = self.daily_sales.set_index(self.date_col)['total_quantity'].resample('W').sum()
            
            if len(weekly_data) < 26:  # Need at least 6 months for meaningful STL
                print("[WARNING] Insufficient data for STL decomposition (need >26 weeks)")
                return {'error': 'insufficient_data'}
            
            # Perform STL decomposition
            stl = STL(weekly_data, seasonal=13, trend=None)  # 13-week seasonal
            result = stl.fit()
            
            # Extract components
            trend = result.trend
            seasonal = result.seasonal  
            residual = result.resid
            
            # Calculate component statistics
            trend_strength = 1 - np.var(residual) / np.var(trend + residual)
            seasonal_strength = 1 - np.var(residual) / np.var(seasonal + residual)
            
            # Detect trend changes (changepoints)
            trend_diff = np.diff(trend.values)
            trend_changes = np.where(np.abs(trend_diff) > 2 * np.std(trend_diff))[0]
            
            print(f"STL Decomposition Results:")
            print(f"  Trend strength: {trend_strength:.3f}")
            print(f"  Seasonal strength: {seasonal_strength:.3f}")
            print(f"  Detected trend changes: {len(trend_changes)}")
            
            if len(trend_changes) > 0:
                print(f"  Major trend change dates:")
                for i, change_idx in enumerate(trend_changes[:5]):
                    change_date = weekly_data.index[change_idx + 1]
                    print(f"    {i+1}. {change_date.strftime('%Y-%m-%d')}")
            
            stl_results = {
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength,
                'n_trend_changes': len(trend_changes),
                'trend_change_dates': [weekly_data.index[i+1].strftime('%Y-%m-%d') 
                                     for i in trend_changes[:10]],
                'seasonal_peak_weeks': seasonal.nlargest(4).index.strftime('%Y-%m-%d').tolist(),
                'seasonal_trough_weeks': seasonal.nsmallest(4).index.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            print(f"[ERROR] STL decomposition failed: {e}")
            stl_results = {'error': str(e)}
        
        self.results['stl_decomposition'] = stl_results
        return stl_results
    
    def analyze_changepoint_detection(self):
        """Advanced changepoint detection for trend and volatility changes"""
        
        print("\n" + "="*60)
        print("CHANGEPOINT DETECTION ANALYSIS")
        print("="*60)
        
        try:
            # Prepare data for changepoint detection
            ts_data = self.daily_sales[['transaction_date', 'total_quantity']].set_index('transaction_date')
            ts_data = ts_data.resample('D').sum()  # Ensure daily frequency
            
            # Simple changepoint detection using rolling statistics
            window = 30  # 30-day window
            ts_data['rolling_mean'] = ts_data['total_quantity'].rolling(window=window, center=True).mean()
            ts_data['rolling_std'] = ts_data['total_quantity'].rolling(window=window, center=True).std()
            
            # Detect mean changes
            mean_diff = ts_data['rolling_mean'].diff().abs()
            mean_threshold = mean_diff.quantile(0.95)  # Top 5% changes
            mean_changepoints = ts_data[mean_diff > mean_threshold].index
            
            # Detect volatility changes  
            std_diff = ts_data['rolling_std'].diff().abs()
            std_threshold = std_diff.quantile(0.95)
            volatility_changepoints = ts_data[std_diff > std_threshold].index
            
            print(f"Detected changepoints:")
            print(f"  Mean changepoints: {len(mean_changepoints)}")
            print(f"  Volatility changepoints: {len(volatility_changepoints)}")
            
            # Show major changepoints
            if len(mean_changepoints) > 0:
                print(f"\nMajor mean changepoints (top 5):")
                for i, cp in enumerate(mean_changepoints[:5]):
                    print(f"  {i+1}. {cp.strftime('%Y-%m-%d (%A)')}")
            
            changepoint_results = {
                'mean_changepoints': [cp.strftime('%Y-%m-%d') for cp in mean_changepoints[:10]],
                'volatility_changepoints': [cp.strftime('%Y-%m-%d') for cp in volatility_changepoints[:10]],
                'n_mean_changes': len(mean_changepoints),
                'n_volatility_changes': len(volatility_changepoints)
            }
            
        except Exception as e:
            print(f"[WARNING] Changepoint detection failed: {e}")
            changepoint_results = {'error': str(e)}
        
        self.results['changepoint_analysis'] = changepoint_results
        return changepoint_results
    
    def generate_temporal_insights(self):
        """Generate strategic insights for forecasting"""
        
        print("\n" + "="*60)
        print("STRATEGIC TEMPORAL INSIGHTS")
        print("="*60)
        
        insights = {
            'key_findings': [],
            'forecasting_recommendations': [],
            'wmape_optimization_tips': [],
            'seasonal_patterns': [],
            'business_implications': []
        }
        
        # Analyze patterns for key findings
        if 'temporal_patterns' in self.results:
            patterns = self.results['temporal_patterns']
            
            # Weekly insights
            if patterns['weekly']['weekend_intensity_ratio'] > 1.2:
                insights['key_findings'].append("Strong weekend sales pattern - 20%+ higher intensity")
                insights['forecasting_recommendations'].append("Use separate models for weekends vs weekdays")
            
            # Monthly insights  
            if patterns['monthly']['seasonality_strength'] > 0.2:
                insights['key_findings'].append(f"Strong monthly seasonality (strength: {patterns['monthly']['seasonality_strength']:.2f})")
                insights['forecasting_recommendations'].append("Include monthly seasonal features in models")
            
            # Trend insights
            if patterns['trend']['is_significant']:
                trend_type = patterns['trend']['classification']
                insights['key_findings'].append(f"Significant {trend_type.lower()} trend detected")
                if trend_type == "Growing":
                    insights['forecasting_recommendations'].append("Prioritize trend-following models (Prophet, ARIMA)")
        
        # Autocorrelation insights
        if 'autocorrelation' in self.results and 'recommended_lags' in self.results['autocorrelation']:
            lags = self.results['autocorrelation']['recommended_lags']
            if lags:
                insights['key_findings'].append(f"Strong autocorrelations found at lags: {lags[:3]}")
                insights['forecasting_recommendations'].append(f"Include lag features: {lags[:5]}")
        
        # Special events insights
        if 'special_events' in self.results:
            events = self.results['special_events']
            if events['n_outlier_days'] > 0:
                insights['key_findings'].append(f"{events['n_outlier_days']} outlier days detected")
                insights['forecasting_recommendations'].append("Include holiday/event indicators in models")
        
        # Seasonal pattern insights
        if 'temporal_patterns' in self.results:
            patterns = self.results['temporal_patterns']
            weekend_ratio = patterns['weekly']['weekend_intensity_ratio']
            if weekend_ratio > 1.5:
                insights['seasonal_patterns'].append("Strong weekend effect - consider weekend-specific models")
            
            high_season = patterns['monthly']['high_season']
            if high_season:
                insights['seasonal_patterns'].append(f"Peak season months: {', '.join(high_season)}")
        
        # Special events insights
        if 'special_events' in self.results and self.results['special_events']['event_matches']:
            event_count = len(self.results['special_events']['event_matches'])
            insights['business_implications'].append(f"Identified {event_count} holiday/event impacts on sales")
        
        # Changepoint insights
        if 'changepoint_analysis' in self.results and 'n_mean_changes' in self.results['changepoint_analysis']:
            n_changes = self.results['changepoint_analysis']['n_mean_changes']
            if n_changes > 10:
                insights['business_implications'].append("High volatility detected - consider adaptive models")
                insights['forecasting_recommendations'].append("Implement online learning for model adaptation")
        
        # WMAPE-specific recommendations
        insights['wmape_optimization_tips'].extend([
            "Focus feature engineering on high-volume periods identified",
            "Use volume-weighted validation splits to match WMAPE calculation",
            "Consider separate models for different volume tiers (A/B/C)",
            "Apply stronger regularization to low-volume products",
            "Weight ensemble models by historical volume patterns",
            "Implement post-processing corrections for seasonal biases"
        ])
        
        print("Key Findings:")
        for i, finding in enumerate(insights['key_findings'], 1):
            print(f"  {i}. {finding}")
        
        print("\nSeasonal Patterns:")
        for i, pattern in enumerate(insights['seasonal_patterns'], 1):
            print(f"  {i}. {pattern}")
        
        print("\nBusiness Implications:")
        for i, implication in enumerate(insights['business_implications'], 1):
            print(f"  {i}. {implication}")
        
        print("\nForecasting Recommendations:")
        for i, rec in enumerate(insights['forecasting_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\nWMAPE Optimization Tips:")
        for i, tip in enumerate(insights['wmape_optimization_tips'], 1):
            print(f"  {i}. {tip}")
        
        self.results['insights'] = insights
        return insights
    
    def save_results(self, output_path: str = "../../data/processed/eda_results"):
        """Save analysis results"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = output_dir / "temporal_analysis_results.json"
        import json
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\n[SAVED] Results saved to: {results_file}")
        
        # Save processed daily data
        daily_data_file = output_dir / "daily_sales_with_features.csv"
        self.daily_sales.to_csv(daily_data_file, index=False)
        print(f"[SAVED] Daily sales data saved to: {daily_data_file}")
        
        return results_file

def main():
    """Main execution function"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("PHASE 2.1: COMPREHENSIVE TEMPORAL ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer()
    
    try:
        # Load and prepare data
        trans_df, daily_sales = analyzer.load_and_prepare_data(sample_size=1500000)
        
        # Execute comprehensive temporal analysis
        patterns = analyzer.analyze_temporal_patterns()
        autocorr = analyzer.analyze_autocorrelation()
        events = analyzer.detect_special_events()
        stl = analyzer.analyze_stl_decomposition()
        changepoints = analyzer.analyze_changepoint_detection()
        insights = analyzer.generate_temporal_insights()
        
        # Save results
        results_file = analyzer.save_results()
        
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"[OK] Analyzed {len(trans_df):,} transactions")
        print(f"[OK] Processed {len(daily_sales)} days of data")
        print(f"[OK] Generated {len(insights['key_findings'])} key findings")
        print(f"[OK] Created {len(insights['forecasting_recommendations'])} recommendations")
        print(f"[OK] Results saved to: {results_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()