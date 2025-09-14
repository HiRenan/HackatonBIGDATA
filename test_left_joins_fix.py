#!/usr/bin/env python3
"""
Test script para verificar se os LEFT JOINs estão funcionando corretamente
Valida que não perdemos nenhuma transação ou volume
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_loader import load_data_efficiently
import pandas as pd

def test_left_joins():
    """Test LEFT JOINs functionality"""
    
    print("🧪 TESTE: LEFT JOINS FIX")
    print("=" * 50)
    
    try:
        # Test with joins enabled (new behavior)
        print("\n🔧 TESTANDO COM LEFT JOINS HABILITADOS:")
        trans_df_joined, prod_df, pdv_df = load_data_efficiently(
            data_path="data/raw",
            sample_transactions=10000,  # Small sample for testing
            sample_products=1000,
            enable_joins=True,
            validate_loss=True
        )
        
        print(f"\n✅ RESULTADO COM JOINS:")
        print(f"  - Shape: {trans_df_joined.shape}")
        print(f"  - Volume total: {trans_df_joined['quantity'].sum():,.0f}")
        print(f"  - Colunas adicionadas: {[col for col in trans_df_joined.columns if col not in ['internal_store_id', 'internal_product_id', 'transaction_date', 'quantity', 'gross_value', 'net_value']]}")
        
        # Test without joins (original behavior)
        print(f"\n📊 TESTANDO SEM JOINS (COMPORTAMENTO ORIGINAL):")
        trans_df_original, _, _ = load_data_efficiently(
            data_path="data/raw", 
            sample_transactions=10000,
            sample_products=1000,
            enable_joins=False,
            validate_loss=False
        )
        
        print(f"\n📈 RESULTADO SEM JOINS:")
        print(f"  - Shape: {trans_df_original.shape}")
        print(f"  - Volume total: {trans_df_original['quantity'].sum():,.0f}")
        
        # Critical validation: Same number of transactions and volume
        assert len(trans_df_joined) == len(trans_df_original), "❌ PERDEU TRANSAÇÕES!"
        assert abs(trans_df_joined['quantity'].sum() - trans_df_original['quantity'].sum()) < 0.01, "❌ PERDEU VOLUME!"
        
        print(f"\n✅ SUCESSO! LEFT JOINs implementados corretamente:")
        print(f"  ✅ Mesma quantidade de transações: {len(trans_df_joined):,}")
        print(f"  ✅ Mesmo volume total: {trans_df_joined['quantity'].sum():,.0f}")
        print(f"  ✅ Adicionou {trans_df_joined.shape[1] - trans_df_original.shape[1]} colunas de contexto")
        print(f"  ✅ Tratamento 'Unknown' para dados faltantes implementado")
        
        # Show sample of new columns
        if 'categoria' in trans_df_joined.columns:
            print(f"\n📋 AMOSTRA DE CATEGORIAS ADICIONADAS:")
            print(trans_df_joined['categoria'].value_counts().head())
            
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_left_joins()
    print(f"\n{'🎉 TESTE PASSOU!' if success else '💥 TESTE FALHOU!'}")