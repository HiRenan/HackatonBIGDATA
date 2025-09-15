#!/usr/bin/env python3
"""
Initial Data Exploration - Hackathon Forecast Big Data 2025
Phase 1.1: Understanding the real data structure and characteristics

Following our winning strategy from estrategia_vitoria_completa.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

def explore_parquet_files():
    """
    Explore the three parquet files to understand data structure
    Following Phase 1.1 of our strategy
    """
    data_path = Path("../../data/raw")
    
    # List all parquet files (including snappy compressed ones)
    parquet_files = list(data_path.glob("*.parquet"))
    
    print("INITIAL DATA EXPLORATION")
    print("=" * 50)
    print(f"Found {len(parquet_files)} parquet files")
    
    exploration_results = {}
    
    for i, file_path in enumerate(parquet_files, 1):
        print(f"\nFILE {i}: {file_path.name}")
        print("-" * 40)
        
        try:
            print(f"Reading {file_path.name}... (Size: {file_path.stat().st_size / 1024**2:.1f} MB)")
            # Use PyArrow for efficient reading of large files
            table = pq.read_table(file_path)
            total_rows = table.num_rows
            
            # For analysis, use first 50k rows if file is large
            if total_rows > 50000:
                print(f"Large file ({total_rows:,} rows) - analyzing first 50k rows")
                df = table.slice(0, 50000).to_pandas()
                print(f"Note: Analysis based on sample of {len(df):,} rows out of {total_rows:,}")
            else:
                df = table.to_pandas()
            
            # Basic info
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Data types
            print(f"\nData types:")
            for col, dtype in df.dtypes.items():
                print(f"  {col}: {dtype}")
            
            # Sample data
            print(f"\nFirst few rows:")
            print(df.head(3))
            
            # Missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"\nMissing values:")
                for col, missing_count in missing[missing > 0].items():
                    pct = (missing_count / len(df)) * 100
                    print(f"  {col}: {missing_count} ({pct:.1f}%)")
            else:
                print(f"\nNo missing values [OK]")
            
            # Store results
            exploration_results[file_path.stem] = {
                'shape': df.shape,
                'total_rows': total_rows if 'total_rows' in locals() else df.shape[0],
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': dict(missing[missing > 0]) if missing.any() else {}
            }
            
        except Exception as e:
            print(f"[ERROR] Error reading {file_path.name}: {e}")
            
    return exploration_results

def identify_data_structure(results):
    """
    Based on exploration results, identify what each file contains
    Following our domain expertise from One-Click Order analysis
    """
    print(f"\nDATA STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Analyze column patterns to identify file types
    for file_name, info in results.items():
        columns = info['columns']
        shape = info['shape']
        
        print(f"\n{file_name}:")
        print(f"  Shape: {shape}")
        print(f"  Columns: {columns}")
        
        # Hypothesize file content based on columns and our domain knowledge
        if any(col.lower() in ['data', 'date', 'semana', 'week'] for col in columns):
            if any(col.lower() in ['quantidade', 'quantity', 'vendas', 'sales'] for col in columns):
                print(f"  HYPOTHESIS: TRANSACTIONS DATA (sales by date/week)")
                
        if any(col.lower() in ['produto', 'product', 'sku'] for col in columns):
            if any(col.lower() in ['categoria', 'category', 'descricao'] for col in columns):
                print(f"  HYPOTHESIS: PRODUCT CATALOG (product master data)")
                
        if any(col.lower() in ['pdv', 'store', 'ponto'] for col in columns):
            if any(col.lower() in ['zipcode', 'tipo', 'type'] for col in columns):
                print(f"  HYPOTHESIS: PDV CATALOG (store master data)")

def generate_initial_insights(results):
    """
    Generate initial insights following our strategy
    """
    print(f"\nINITIAL INSIGHTS")
    print("=" * 50)
    
    total_memory = sum(info['memory_mb'] for info in results.values())
    total_rows = sum(info['total_rows'] for info in results.values())
    
    print(f"Data Overview:")
    print(f"  Total memory usage: {total_memory:.1f} MB")
    print(f"  Total rows across files: {total_rows:,}")
    
    if results:
        largest_file = max(results.items(), key=lambda x: x[1]['total_rows'])
        print(f"  Largest file: {largest_file[0]} with {largest_file[1]['total_rows']:,} rows")
    else:
        print(f"  No files found for analysis")
    
    print(f"\nStrategic Implications:")
    print(f"  [OK] Data size manageable for local processing")
    print(f"  [OK] Multiple files suggest proper data structure")
    print(f"  [OK] Ready for comprehensive EDA in Phase 2")
    
    # Resource assessment based on data size
    if total_memory < 1000:  # < 1GB
        resource_level = "minimal"
    elif total_memory < 5000:  # < 5GB
        resource_level = "standard" 
    else:
        resource_level = "optimal"
        
    print(f"  Recommended resource level: {resource_level}")
    
    return {
        'total_memory_mb': total_memory,
        'total_rows': total_rows,
        'recommended_resource_level': resource_level,
        'files_analysis': results
    }

if __name__ == "__main__":
    print("HACKATHON FORECAST BIG DATA 2025")
    print("Phase 1.1: Initial Data Exploration")
    print("Following winning strategy from estrategia_vitoria_completa.md")
    print("=" * 60)
    
    # Explore all parquet files
    results = explore_parquet_files()
    
    # Identify data structure
    identify_data_structure(results)
    
    # Generate insights
    insights = generate_initial_insights(results)
    
    print(f"\nPHASE 1.1 DATA EXPLORATION COMPLETE!")
    print(f"[READY] Ready to proceed to Phase 1.2: Resource Analysis")