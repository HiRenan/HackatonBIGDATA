#!/usr/bin/env python3
"""
CRITICAL Data Relationship Validation - Hackathon Forecast 2025
Validates relationships between PDV, Product, and Transaction datasets

This is CRITICAL for the project success - we need to confirm how datasets relate!
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_datasets_sample():
    """Load representative samples of all datasets"""
    data_path = Path("../../data/raw")
    
    # Get files
    files = list(data_path.glob("*.parquet"))
    print(f"Found {len(files)} parquet files")
    
    datasets = {}
    
    for file_path in files:
        file_size = file_path.stat().st_size / (1024**2)
        print(f"\nLoading: {file_path.name} ({file_size:.1f}MB)")
        
        # Load appropriate sample size
        table = pq.read_table(file_path)
        total_rows = table.num_rows
        
        if total_rows > 100000:
            # Large file - take larger sample for relationship validation
            sample_size = 100000
            df = table.slice(0, sample_size).to_pandas()
            print(f"Sample: {sample_size:,} rows from {total_rows:,} total")
        else:
            # Small file - load all
            df = table.to_pandas()
            print(f"Full dataset: {len(df):,} rows")
        
        # Identify dataset type by columns
        columns = df.columns.tolist()
        if 'pdv' in columns and 'premise' in columns:
            datasets['pdv_catalog'] = df
            print("-> Identified as: PDV CATALOG")
        elif 'internal_store_id' in columns and 'internal_product_id' in columns:
            datasets['transactions'] = df
            print("-> Identified as: TRANSACTION DATA")
        elif 'produto' in columns and 'categoria' in columns:
            datasets['products'] = df
            print("-> Identified as: PRODUCT CATALOG")
        
        print(f"Columns: {columns}")
    
    return datasets

def validate_pdv_relationships(datasets):
    """Validate PDV-Transaction relationships"""
    print("\n" + "="*60)
    print("VALIDATING PDV-TRANSACTION RELATIONSHIPS")
    print("="*60)
    
    if 'pdv_catalog' not in datasets or 'transactions' not in datasets:
        print("[ERROR] Missing datasets for PDV validation")
        return {}
    
    pdv_df = datasets['pdv_catalog']
    trans_df = datasets['transactions']
    
    # Get unique keys from each dataset
    pdv_ids = set(pdv_df['pdv'].astype(str))
    store_ids = set(trans_df['internal_store_id'].astype(str))
    
    print(f"PDV Catalog unique PDVs: {len(pdv_ids):,}")
    print(f"Transactions unique store_ids: {len(store_ids):,}")
    
    # Check intersection
    intersection = pdv_ids.intersection(store_ids)
    print(f"Matching IDs (intersection): {len(intersection):,}")
    
    # Calculate coverage
    pdv_coverage = len(intersection) / len(pdv_ids) * 100 if pdv_ids else 0
    trans_coverage = len(intersection) / len(store_ids) * 100 if store_ids else 0
    
    print(f"PDV Catalog coverage: {pdv_coverage:.1f}% have transactions")
    print(f"Transaction coverage: {trans_coverage:.1f}% have PDV info")
    
    # Check for orphans
    pdv_orphans = pdv_ids - store_ids
    trans_orphans = store_ids - pdv_ids
    
    print(f"\nOrphan Analysis:")
    print(f"PDVs without transactions: {len(pdv_orphans):,}")
    print(f"Transactions without PDV info: {len(trans_orphans):,}")
    
    # Sample some matching and orphan IDs
    if intersection:
        print(f"\nSample matching IDs:")
        for i, id_val in enumerate(sorted(intersection)[:3]):
            print(f"  {id_val}")
    
    if pdv_orphans:
        print(f"\nSample PDV orphans:")
        for i, id_val in enumerate(sorted(pdv_orphans)[:3]):
            print(f"  {id_val}")
    
    if trans_orphans:
        print(f"\nSample Transaction orphans:")
        for i, id_val in enumerate(sorted(trans_orphans)[:3]):
            print(f"  {id_val}")
    
    return {
        'pdv_total': len(pdv_ids),
        'transactions_total': len(store_ids), 
        'intersection': len(intersection),
        'pdv_coverage': pdv_coverage,
        'trans_coverage': trans_coverage,
        'pdv_orphans': len(pdv_orphans),
        'trans_orphans': len(trans_orphans),
        'relationship_quality': 'GOOD' if trans_coverage > 80 else 'POOR'
    }

def validate_product_relationships(datasets):
    """Validate Product-Transaction relationships"""
    print("\n" + "="*60)
    print("VALIDATING PRODUCT-TRANSACTION RELATIONSHIPS")
    print("="*60)
    
    if 'products' not in datasets or 'transactions' not in datasets:
        print("[ERROR] Missing datasets for Product validation")
        return {}
    
    prod_df = datasets['products']
    trans_df = datasets['transactions']
    
    # Get unique keys from each dataset
    product_ids = set(prod_df['produto'].astype(str))
    trans_product_ids = set(trans_df['internal_product_id'].astype(str))
    
    print(f"Product Catalog unique products: {len(product_ids):,}")
    print(f"Transactions unique product_ids: {len(trans_product_ids):,}")
    
    # Check intersection
    intersection = product_ids.intersection(trans_product_ids)
    print(f"Matching IDs (intersection): {len(intersection):,}")
    
    # Calculate coverage
    prod_coverage = len(intersection) / len(product_ids) * 100 if product_ids else 0
    trans_coverage = len(intersection) / len(trans_product_ids) * 100 if trans_product_ids else 0
    
    print(f"Product Catalog coverage: {prod_coverage:.1f}% have transactions")
    print(f"Transaction coverage: {trans_coverage:.1f}% have product info")
    
    # Check for orphans
    prod_orphans = product_ids - trans_product_ids
    trans_orphans = trans_product_ids - product_ids
    
    print(f"\nOrphan Analysis:")
    print(f"Products without transactions: {len(prod_orphans):,}")
    print(f"Transactions without product info: {len(trans_orphans):,}")
    
    # Sample some matching and orphan IDs
    if intersection:
        print(f"\nSample matching IDs:")
        for i, id_val in enumerate(sorted(intersection)[:3]):
            print(f"  {id_val}")
    
    if prod_orphans:
        print(f"\nSample Product orphans:")
        for i, id_val in enumerate(sorted(prod_orphans)[:3]):
            print(f"  {id_val}")
            
    if trans_orphans:
        print(f"\nSample Transaction orphans:")
        for i, id_val in enumerate(sorted(trans_orphans)[:3]):
            print(f"  {id_val}")
    
    return {
        'products_total': len(product_ids),
        'transactions_total': len(trans_product_ids),
        'intersection': len(intersection), 
        'prod_coverage': prod_coverage,
        'trans_coverage': trans_coverage,
        'prod_orphans': len(prod_orphans),
        'trans_orphans': len(trans_orphans),
        'relationship_quality': 'GOOD' if trans_coverage > 80 else 'POOR'
    }

def test_join_operations(datasets):
    """Test actual JOIN operations to validate relationships"""
    print("\n" + "="*60)
    print("TESTING JOIN OPERATIONS")
    print("="*60)
    
    if len(datasets) < 3:
        print("[ERROR] Not all datasets available for JOIN testing")
        return {}
    
    trans_df = datasets['transactions'].head(10000)  # Use smaller sample for JOIN test
    pdv_df = datasets['pdv_catalog']
    prod_df = datasets['products'].head(50000)  # Use smaller sample
    
    join_results = {}
    
    # Test PDV JOIN
    print("\n1. Testing PDV JOIN...")
    try:
        pdv_join = trans_df.merge(
            pdv_df, 
            left_on='internal_store_id', 
            right_on='pdv',
            how='left',
            validate='many_to_one'
        )
        
        matched_pdv = pdv_join['pdv'].notna().sum()
        total_trans = len(trans_df)
        pdv_join_rate = (matched_pdv / total_trans) * 100
        
        print(f"   PDV JOIN SUCCESS: {matched_pdv:,}/{total_trans:,} ({pdv_join_rate:.1f}%) matched")
        join_results['pdv_join_rate'] = pdv_join_rate
        
        # Show sample joined data
        print("   Sample joined data:")
        sample = pdv_join[['internal_store_id', 'pdv', 'categoria_pdv', 'quantity']].head(3)
        print(sample.to_string(index=False))
        
    except Exception as e:
        print(f"   PDV JOIN FAILED: {e}")
        join_results['pdv_join_rate'] = 0
    
    # Test Product JOIN
    print("\n2. Testing Product JOIN...")
    try:
        prod_join = trans_df.merge(
            prod_df,
            left_on='internal_product_id',
            right_on='produto', 
            how='left',
            validate='many_to_one'
        )
        
        matched_prod = prod_join['produto'].notna().sum()
        total_trans = len(trans_df)
        prod_join_rate = (matched_prod / total_trans) * 100
        
        print(f"   Product JOIN SUCCESS: {matched_prod:,}/{total_trans:,} ({prod_join_rate:.1f}%) matched")
        join_results['product_join_rate'] = prod_join_rate
        
        # Show sample joined data
        print("   Sample joined data:")
        sample = prod_join[['internal_product_id', 'produto', 'categoria', 'quantity']].head(3)
        print(sample.to_string(index=False))
        
    except Exception as e:
        print(f"   Product JOIN FAILED: {e}")
        join_results['product_join_rate'] = 0
    
    # Test Full 3-way JOIN
    print("\n3. Testing FULL 3-way JOIN...")
    try:
        full_join = trans_df.merge(
            pdv_df, left_on='internal_store_id', right_on='pdv', how='left'
        ).merge(
            prod_df, left_on='internal_product_id', right_on='produto', how='left'
        )
        
        complete_records = (full_join['pdv'].notna() & full_join['produto'].notna()).sum()
        total_trans = len(trans_df)
        complete_rate = (complete_records / total_trans) * 100
        
        print(f"   FULL JOIN SUCCESS: {complete_records:,}/{total_trans:,} ({complete_rate:.1f}%) complete")
        join_results['full_join_rate'] = complete_rate
        
        # Show sample complete record
        complete_sample = full_join[
            (full_join['pdv'].notna()) & (full_join['produto'].notna())
        ][['internal_store_id', 'categoria_pdv', 'internal_product_id', 'categoria', 'quantity']].head(2)
        
        if len(complete_sample) > 0:
            print("   Sample complete records:")
            print(complete_sample.to_string(index=False))
        
    except Exception as e:
        print(f"   FULL JOIN FAILED: {e}")
        join_results['full_join_rate'] = 0
    
    return join_results

def generate_relationship_report(pdv_results, product_results, join_results):
    """Generate comprehensive relationship report"""
    print("\n" + "="*60)
    print("FINAL RELATIONSHIP VALIDATION REPORT")
    print("="*60)
    
    print(f"\nDATASET COVERAGE ANALYSIS:")
    print(f"   PDV Coverage: {pdv_results.get('trans_coverage', 0):.1f}% of transactions have PDV info")
    print(f"   Product Coverage: {product_results.get('trans_coverage', 0):.1f}% of transactions have product info")
    print(f"   Full Join Rate: {join_results.get('full_join_rate', 0):.1f}% complete records")
    
    print(f"\nDATA QUALITY ASSESSMENT:")
    print(f"   PDV Relationship: {pdv_results.get('relationship_quality', 'UNKNOWN')}")
    print(f"   Product Relationship: {product_results.get('relationship_quality', 'UNKNOWN')}")
    
    # Determine overall data quality
    pdv_good = pdv_results.get('trans_coverage', 0) > 80
    product_good = product_results.get('trans_coverage', 0) > 80
    join_good = join_results.get('full_join_rate', 0) > 70
    
    if pdv_good and product_good and join_good:
        overall_quality = "[EXCELLENT]"
        recommendation = "Proceed with confidence - relationships are solid"
    elif pdv_results.get('trans_coverage', 0) > 50 and product_results.get('trans_coverage', 0) > 50:
        overall_quality = "[ACCEPTABLE]"
        recommendation = "Proceed with caution - some missing relationships"
    else:
        overall_quality = "[POOR]"
        recommendation = "CRITICAL: Investigate relationship issues before proceeding"
    
    print(f"\nOVERALL ASSESSMENT: {overall_quality}")
    print(f"RECOMMENDATION: {recommendation}")
    
    # Action items
    print(f"\nACTION ITEMS:")
    if not pdv_good:
        print(f"   - Investigate PDV relationship gaps ({100-pdv_results.get('trans_coverage', 0):.1f}% missing)")
    if not product_good:
        print(f"   - Investigate Product relationship gaps ({100-product_results.get('trans_coverage', 0):.1f}% missing)")
    if not join_good:
        print(f"   - Address incomplete records for forecasting")
    
    return {
        'overall_quality': overall_quality,
        'recommendation': recommendation,
        'pdv_coverage': pdv_results.get('trans_coverage', 0),
        'product_coverage': product_results.get('trans_coverage', 0),
        'full_join_rate': join_results.get('full_join_rate', 0)
    }

def main():
    """Main validation function"""
    print("HACKATHON FORECAST BIG DATA 2025")
    print("CRITICAL DATA RELATIONSHIP VALIDATION")
    print("="*60)
    print("This analysis is CRITICAL for forecasting success!")
    print("We need to understand how datasets connect.")
    print("="*60)
    
    try:
        # Load datasets
        datasets = load_datasets_sample()
        
        if len(datasets) < 3:
            print(f"[ERROR] Only found {len(datasets)} datasets, need 3")
            print("Available datasets:", list(datasets.keys()))
            return
        
        print(f"\n[SUCCESS] Successfully loaded {len(datasets)} datasets:")
        for name, df in datasets.items():
            print(f"   {name}: {len(df):,} rows")
        
        # Validate relationships
        pdv_results = validate_pdv_relationships(datasets)
        product_results = validate_product_relationships(datasets)
        join_results = test_join_operations(datasets)
        
        # Generate final report
        final_report = generate_relationship_report(pdv_results, product_results, join_results)
        
        print(f"\n[COMPLETE] VALIDATION COMPLETE!")
        print(f"Check the results above to understand data relationships.")
        
        return final_report
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()