#!/usr/bin/env python3
"""
COMPLETE FEATURE PIPELINE DEMONSTRATION - Hackathon Forecast Big Data 2025
Comprehensive demonstration of the complete feature engineering pipeline

This notebook demonstrates:
1. Loading data with the updated schema
2. Running all 4 feature engines in sequence
3. Feature validation and selection
4. Export to feature store
5. Performance analysis and insights

Execute this to generate the complete feature store for modeling!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.feature_pipeline import FeaturePipeline
from utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_hackathon_data():
    """Load hackathon data with all available records for feature engineering"""
    
    print("LOADING HACKATHON DATA FOR FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load comprehensive dataset
    trans_df, prod_df, pdv_df = load_data_efficiently(
        data_path="../../data/raw",
        sample_transactions=500000,  # Large sample for comprehensive features
        sample_products=5000
    )
    
    print(f"[OK] Transaction data: {trans_df.shape}")
    print(f"[OK] Product data: {prod_df.shape}")
    print(f"[OK] PDV data: {pdv_df.shape}")
    
    # Merge product information
    if len(prod_df) > 0:
        # Create product mapping for categories
        product_categories = {}
        product_brands = {}
        
        for idx, row in prod_df.iterrows():
            product_key = hash(str(row['produto'])) % 1000000
            product_categories[product_key] = row.get('categoria', 'Unknown')
            product_brands[product_key] = row.get('marca', 'Unknown')
        
        # Add to transaction data
        trans_df['categoria'] = trans_df['internal_product_id'].map(lambda x: product_categories.get(x, 'Unknown'))
        trans_df['marca'] = trans_df['internal_product_id'].map(lambda x: product_brands.get(x, 'Unknown'))
        
        print(f"[OK] Added product categories: {trans_df['categoria'].nunique()} unique categories")
        print(f"[OK] Added product brands: {trans_df['marca'].nunique()} unique brands")
    
    # Merge PDV information
    if len(pdv_df) > 0 and 'zipcode' in pdv_df.columns:
        # Create PDV mapping
        pdv_zipcodes = {}
        pdv_categories = {}
        
        for idx, row in pdv_df.iterrows():
            pdv_key = str(row['pdv'])
            pdv_zipcodes[pdv_key] = row.get('zipcode', 0)
            pdv_categories[pdv_key] = row.get('categoria_pdv', 'Unknown')
        
        # Add to transaction data
        trans_df['zipcode'] = trans_df['internal_store_id'].map(
            lambda x: pdv_zipcodes.get(str(x), np.random.randint(10000, 99999))
        )
        trans_df['categoria_pdv'] = trans_df['internal_store_id'].map(
            lambda x: pdv_categories.get(str(x), 'Unknown')
        )
        
        print(f"[OK] Added zipcodes: {trans_df['zipcode'].nunique()} unique zipcodes")
        print(f"[OK] Added PDV categories: {trans_df['categoria_pdv'].nunique()} unique PDV categories")
    
    print(f"\n[FINAL] Enhanced dataset shape: {trans_df.shape}")
    print(f"[FINAL] Column list: {list(trans_df.columns)}")
    
    return trans_df, prod_df, pdv_df

def run_feature_pipeline_demo():
    """Run complete feature pipeline demonstration"""
    
    print("\n" + "="*100)
    print("HACKATHON FORECAST BIG DATA 2025")
    print("COMPLETE FEATURE ENGINEERING PIPELINE DEMONSTRATION")
    print("="*100)
    
    demo_start_time = time.time()
    
    # Step 1: Load data
    print(f"\n[STEP 1/5] Loading hackathon data...")
    step_start = time.time()
    trans_df, prod_df, pdv_df = load_hackathon_data()
    step1_time = time.time() - step_start
    print(f"[STEP 1/5] Completed in {step1_time:.2f}s")
    
    # Step 2: Initialize pipeline
    print(f"\n[STEP 2/5] Initializing feature pipeline...")
    step_start = time.time()
    
    pipeline = FeaturePipeline(
        date_col='transaction_date',
        value_col='quantity',
        target_col='quantity',
        groupby_cols=['internal_product_id', 'internal_store_id']
    )
    
    step2_time = time.time() - step_start
    print(f"[STEP 2/5] Pipeline initialized in {step2_time:.2f}s")
    
    # Step 3: Run feature engineering
    print(f"\n[STEP 3/5] Running complete feature engineering pipeline...")
    step_start = time.time()
    
    features_df = pipeline.run_full_pipeline(
        trans_df,
        enable_validation=True,
        enable_feature_selection=True,
        max_features=100  # Keep top 100 features
    )
    
    step3_time = time.time() - step_start
    print(f"[STEP 3/5] Feature engineering completed in {step3_time:.2f}s")
    
    # Step 4: Export feature store
    print(f"\n[STEP 4/5] Exporting feature store...")
    step_start = time.time()
    
    exported_files = pipeline.export_feature_store(
        features_df,
        output_path="../../data/features",
        export_format="parquet"
    )
    
    step4_time = time.time() - step_start
    print(f"[STEP 4/5] Feature store exported in {step4_time:.2f}s")
    
    # Step 5: Analysis and insights
    print(f"\n[STEP 5/5] Generating analysis and insights...")
    step_start = time.time()
    
    analysis_results = generate_feature_analysis(features_df, pipeline)
    
    step5_time = time.time() - step_start
    print(f"[STEP 5/5] Analysis completed in {step5_time:.2f}s")
    
    # Final summary
    total_demo_time = time.time() - demo_start_time
    
    print(f"\n" + "="*100)
    print("FEATURE PIPELINE DEMONSTRATION - FINAL SUMMARY")
    print("="*100)
    print(f"[SUCCESS] Complete demonstration finished!")
    print(f"[TIMING] Total demonstration time: {total_demo_time:.2f}s")
    print(f"[OUTPUT] Final feature dataset: {features_df.shape}")
    print(f"[OUTPUT] Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Step timing breakdown
    print(f"\n[PERFORMANCE] Step timing breakdown:")
    print(f"  1. Data Loading: {step1_time:.2f}s ({step1_time/total_demo_time*100:.1f}%)")
    print(f"  2. Pipeline Init: {step2_time:.2f}s ({step2_time/total_demo_time*100:.1f}%)")
    print(f"  3. Feature Engineering: {step3_time:.2f}s ({step3_time/total_demo_time*100:.1f}%)")
    print(f"  4. Feature Store Export: {step4_time:.2f}s ({step4_time/total_demo_time*100:.1f}%)")
    print(f"  5. Analysis & Insights: {step5_time:.2f}s ({step5_time/total_demo_time*100:.1f}%)")
    
    # Feature pipeline breakdown
    if pipeline.execution_times:
        print(f"\n[PERFORMANCE] Feature engineering stage breakdown:")
        fe_total = sum(pipeline.execution_times.values())
        for stage, duration in pipeline.execution_times.items():
            print(f"  {stage.capitalize()}: {duration:.2f}s ({duration/fe_total*100:.1f}%)")
    
    # Feature importance top 20
    if pipeline.feature_importance_scores:
        print(f"\n[INSIGHTS] Top 20 most important features:")
        top_features = sorted(
            pipeline.feature_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        for i, (feature, score) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feature:<50} (score: {score:.1f})")
    
    # Files created
    print(f"\n[OUTPUTS] Files created:")
    for file_type, file_path in exported_files.items():
        print(f"  {file_type.capitalize()}: {file_path}")
    
    return features_df, pipeline, exported_files, analysis_results

def generate_feature_analysis(features_df: pd.DataFrame, pipeline: FeaturePipeline):
    """Generate comprehensive feature analysis"""
    
    print("[INFO] Generating comprehensive feature analysis...")
    
    analysis = {
        'dataset_stats': {},
        'feature_distribution': {},
        'correlations': {},
        'business_insights': []
    }
    
    # Dataset statistics
    analysis['dataset_stats'] = {
        'total_records': len(features_df),
        'total_features': features_df.shape[1],
        'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2,
        'date_range': {
            'start': features_df['transaction_date'].min(),
            'end': features_df['transaction_date'].max(),
            'days': (features_df['transaction_date'].max() - features_df['transaction_date'].min()).days
        } if 'transaction_date' in features_df.columns else {},
        'unique_products': features_df['internal_product_id'].nunique() if 'internal_product_id' in features_df.columns else 0,
        'unique_stores': features_df['internal_store_id'].nunique() if 'internal_store_id' in features_df.columns else 0
    }
    
    # Feature type distribution
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns
    datetime_features = features_df.select_dtypes(include=['datetime64']).columns
    
    analysis['feature_distribution'] = {
        'numeric': len(numeric_features),
        'categorical': len(categorical_features),
        'datetime': len(datetime_features),
        'total': features_df.shape[1]
    }
    
    # Feature categories from engines
    engine_features = {
        'temporal': len([f for f in pipeline.temporal_engine.features_created if f in features_df.columns]),
        'aggregation': len([f for f in pipeline.aggregation_engine.features_created if f in features_df.columns]),
        'behavioral': len([f for f in pipeline.behavioral_engine.features_created if f in features_df.columns]),
        'business': len([f for f in pipeline.business_engine.features_created if f in features_df.columns])
    }
    
    analysis['engine_features'] = engine_features
    
    # Business insights based on feature analysis
    insights = []
    
    if 'is_sunday' in features_df.columns:
        sunday_impact = features_df[features_df['is_sunday'] == 1]['quantity'].mean() / features_df[features_df['is_sunday'] == 0]['quantity'].mean()
        insights.append(f"Sunday effect captured: {sunday_impact:.1f}x volume increase on Sundays")
    
    if 'abc_class' in features_df.columns:
        tier_a_share = (features_df['abc_class'] == 'A').mean() * 100
        insights.append(f"ABC classification: {tier_a_share:.1f}% of records are Tier A products")
    
    if 'zero_weeks_ratio' in features_df.columns:
        avg_intermittency = features_df['zero_weeks_ratio'].mean() * 100
        insights.append(f"Intermittency analysis: {avg_intermittency:.1f}% average zero-demand ratio")
    
    if 'profit_margin' in features_df.columns:
        avg_margin = features_df['profit_margin'].mean() * 100
        insights.append(f"Profit analysis: {avg_margin:.1f}% average profit margin")
    
    analysis['business_insights'] = insights
    
    # Create visualizations
    create_feature_visualizations(features_df, pipeline, analysis)
    
    return analysis

def create_feature_visualizations(features_df: pd.DataFrame, pipeline: FeaturePipeline, analysis: dict):
    """Create comprehensive feature visualizations"""
    
    print("[INFO] Creating feature analysis visualizations...")
    
    # Create output directory
    viz_dir = Path("../../data/features/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature importance plot
    if pipeline.feature_importance_scores:
        plt.figure(figsize=(12, 8))
        
        top_features = dict(sorted(
            pipeline.feature_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        feature_names = list(top_features.keys())
        importance_scores = list(top_features.values())
        
        plt.barh(range(len(feature_names)), importance_scores)
        plt.yticks(range(len(feature_names)), [f[:40] + '...' if len(f) > 40 else f for f in feature_names])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Features by Importance Score')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance_top20.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Feature category distribution
    plt.figure(figsize=(10, 6))
    
    engine_features = analysis['engine_features']
    categories = list(engine_features.keys())
    counts = list(engine_features.values())
    
    plt.bar(categories, counts, alpha=0.7)
    plt.title('Features by Engine Category')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'features_by_engine.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pipeline execution timing
    if pipeline.execution_times:
        plt.figure(figsize=(10, 6))
        
        stages = list(pipeline.execution_times.keys())
        times = list(pipeline.execution_times.values())
        
        plt.bar(stages, times, alpha=0.7)
        plt.title('Pipeline Execution Time by Stage')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        total_time = sum(times)
        for i, (stage, time_val) in enumerate(zip(stages, times)):
            pct = time_val / total_time * 100
            plt.text(i, time_val + 0.1, f'{time_val:.1f}s\n({pct:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'pipeline_execution_timing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Data quality overview
    if pipeline.validation_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Missing values
        missing_data = pipeline.validation_results.get('missing_values', {})
        if missing_data:
            top_missing = dict(sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:10])
            axes[0, 0].bar(range(len(top_missing)), list(top_missing.values()))
            axes[0, 0].set_title('Top 10 Features with Missing Values')
            axes[0, 0].set_xticks(range(len(top_missing)))
            axes[0, 0].set_xticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_missing.keys()], rotation=45)
        
        # Quality score
        quality_score = pipeline.validation_results.get('quality_score', 0)
        axes[0, 1].pie([quality_score, 100-quality_score], labels=['Good Quality', 'Issues'], autopct='%1.1f%%')
        axes[0, 1].set_title(f'Overall Feature Quality Score: {quality_score:.1f}/100')
        
        # Feature types
        type_dist = analysis['feature_distribution']
        axes[1, 0].pie(type_dist.values(), labels=type_dist.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Feature Type Distribution')
        
        # Dataset stats
        stats_text = f"""
Dataset Statistics:
• Total Records: {analysis['dataset_stats']['total_records']:,}
• Total Features: {analysis['dataset_stats']['total_features']:,}
• Memory Usage: {analysis['dataset_stats']['memory_usage_mb']:.1f} MB
• Unique Products: {analysis['dataset_stats']['unique_products']:,}
• Unique Stores: {analysis['dataset_stats']['unique_stores']:,}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Dataset Overview')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'data_quality_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Visualizations saved to: {viz_dir}")

def main():
    """Main execution function"""
    
    try:
        # Run complete demonstration
        results = run_feature_pipeline_demo()
        
        if results[0] is not None:
            features_df, pipeline, exported_files, analysis = results
            
            print(f"\n" + "🎉" * 50)
            print("FEATURE ENGINEERING PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 50)
            print(f"\n✅ READY FOR MODELING!")
            print(f"✅ Feature store created with {features_df.shape[1]} features")
            print(f"✅ {len(exported_files)} files exported")
            print(f"✅ Complete pipeline ready for production")
            
            return results
        else:
            print("[ERROR] Demonstration failed")
            return None
            
    except Exception as e:
        print(f"[CRITICAL ERROR] Feature pipeline demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()