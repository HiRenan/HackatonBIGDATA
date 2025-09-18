#!/usr/bin/env python3
"""
Análise dos Dados Reais do Hackathon
Identificar padrões, distribuições e estrutura dos dados reais
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_transaction_data():
    """Analisar dados de transações"""
    print("=" * 60)
    print("ANÁLISE DOS DADOS REAIS - HACKATHON 2025")
    print("=" * 60)

    data_path = Path("data/raw")

    # Encontrar arquivo de transações
    trans_files = list(data_path.glob("*5196563791502273604*"))
    if not trans_files:
        print("[ERRO] Arquivo de transacoes nao encontrado!")
        return None

    trans_file = trans_files[0]
    print(f"[FILE] Carregando: {trans_file.name}")

    # Carregar amostra dos dados
    print("[LOAD] Carregando amostra (50k registros)...")
    trans_df = pd.read_parquet(trans_file)

    # Amostra para análise
    if len(trans_df) > 50000:
        trans_df = trans_df.sample(n=50000, random_state=42)

    print(f"[DATA] Dados carregados: {len(trans_df):,} registros")
    print(f"[PERIOD] Periodo: {trans_df['transaction_date'].min()} ate {trans_df['transaction_date'].max()}")

    # Análise básica
    print("\n" + "=" * 40)
    print("ESTRUTURA DOS DADOS")
    print("=" * 40)
    print(f"Colunas: {list(trans_df.columns)}")
    print(f"Tipos: {trans_df.dtypes.to_dict()}")
    print(f"Missing values: {trans_df.isnull().sum().to_dict()}")

    # Estatísticas de quantidade
    print("\n" + "=" * 40)
    print("ANÁLISE DE QUANTIDADE")
    print("=" * 40)
    qty_stats = trans_df['quantity'].describe()
    print(f"Quantidade - Estatísticas:")
    for stat, value in qty_stats.items():
        print(f"  {stat}: {value:.2f}")

    print(f"\nZeros: {(trans_df['quantity'] == 0).sum():,} ({(trans_df['quantity'] == 0).mean()*100:.1f}%)")
    print(f"Negativos: {(trans_df['quantity'] < 0).sum():,}")
    print(f"Outliers (>Q99): {(trans_df['quantity'] > trans_df['quantity'].quantile(0.99)).sum():,}")

    # Análise temporal
    print("\n" + "=" * 40)
    print("ANÁLISE TEMPORAL")
    print("=" * 40)
    trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])
    trans_df['year'] = trans_df['transaction_date'].dt.year
    trans_df['month'] = trans_df['transaction_date'].dt.month
    trans_df['week'] = trans_df['transaction_date'].dt.isocalendar().week
    trans_df['weekday'] = trans_df['transaction_date'].dt.day_name()

    print("Anos disponíveis:", sorted(trans_df['year'].unique()))
    print("Meses disponíveis:", sorted(trans_df['month'].unique()))
    print("Range de semanas:", trans_df['week'].min(), "-", trans_df['week'].max())

    # Padrões por dia da semana
    weekday_pattern = trans_df.groupby('weekday')['quantity'].agg(['count', 'mean', 'sum'])
    print("\nPadrão por dia da semana:")
    print(weekday_pattern)

    # Análise de PDVs
    print("\n" + "=" * 40)
    print("ANÁLISE DE PDVs")
    print("=" * 40)
    print(f"Total de PDVs únicos: {trans_df['internal_store_id'].nunique():,}")
    print(f"Range de IDs: {trans_df['internal_store_id'].min()} - {trans_df['internal_store_id'].max()}")

    # Top PDVs por volume
    top_pdvs = trans_df.groupby('internal_store_id')['quantity'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False).head(10)
    print("\nTop 10 PDVs por volume total:")
    print(top_pdvs)

    # Análise de produtos
    print("\n" + "=" * 40)
    print("ANÁLISE DE PRODUTOS")
    print("=" * 40)
    print(f"Total de produtos únicos: {trans_df['internal_product_id'].nunique():,}")
    print(f"Range de IDs: {trans_df['internal_product_id'].min()} - {trans_df['internal_product_id'].max()}")

    # Top produtos por volume
    top_products = trans_df.groupby('internal_product_id')['quantity'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False).head(10)
    print("\nTop 10 produtos por volume total:")
    print(top_products)

    return trans_df

def analyze_target_period(trans_df):
    """Analisar período específico que pode ser o target do hackathon"""
    print("\n" + "=" * 40)
    print("ANÁLISE DO PERÍODO TARGET")
    print("=" * 40)

    # Verificar os últimos dados disponíveis
    latest_date = trans_df['transaction_date'].max()
    earliest_date = trans_df['transaction_date'].min()

    print(f"Período completo: {earliest_date} até {latest_date}")

    # Analisar últimas 5-10 semanas (possível período de teste)
    trans_df_sorted = trans_df.sort_values('transaction_date')

    # Agrupar por semana
    weekly_data = trans_df.groupby([
        trans_df['transaction_date'].dt.to_period('W'),
        'internal_store_id',
        'internal_product_id'
    ])['quantity'].sum().reset_index()

    print(f"\nDados semanais: {len(weekly_data):,} registros")
    print("Últimas 10 semanas por volume:")

    weekly_totals = weekly_data.groupby('transaction_date')['quantity'].agg(['count', 'sum', 'mean']).sort_values('transaction_date')
    print(weekly_totals.tail(10))

    return weekly_data

def identify_hackathon_mapping(trans_df):
    """Identificar como mapear para formato hackathon"""
    print("\n" + "=" * 40)
    print("MAPEAMENTO PARA FORMATO HACKATHON")
    print("=" * 40)

    # Selecionar top PDVs e produtos por volume (candidatos para hackathon)
    top_pdvs = trans_df.groupby('internal_store_id')['quantity'].sum().sort_values(ascending=False).head(20)
    top_products = trans_df.groupby('internal_product_id')['quantity'].sum().sort_values(ascending=False).head(20)

    print("Top 20 PDVs por volume (candidatos para 1001-1005):")
    for i, (pdv_id, volume) in enumerate(top_pdvs.head(20).items()):
        hackathon_id = 1001 + (i % 5)  # Mapear ciclicamente para 1001-1005
        print(f"  {pdv_id} -> {hackathon_id} (volume: {volume:,.0f})")

    print("\nTop 20 produtos por volume (candidatos para 201-210):")
    for i, (prod_id, volume) in enumerate(top_products.head(20).items()):
        hackathon_id = 201 + (i % 10)  # Mapear ciclicamente para 201-210
        print(f"  {prod_id} -> {hackathon_id} (volume: {volume:,.0f})")

    # Criar mapeamento sugerido
    pdv_mapping = {pdv_id: 1001 + (i % 5) for i, pdv_id in enumerate(top_pdvs.head(20).index)}
    product_mapping = {prod_id: 201 + (i % 10) for i, prod_id in enumerate(top_products.head(20).index)}

    return pdv_mapping, product_mapping

def analyze_seasonality_patterns(trans_df):
    """Analisar padrões sazonais"""
    print("\n" + "=" * 40)
    print("PADRÕES SAZONAIS")
    print("=" * 40)

    # Converter para datetime se necessário
    trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])

    # Agrupar por mês
    monthly_pattern = trans_df.groupby(trans_df['transaction_date'].dt.month)['quantity'].agg(['count', 'mean', 'sum'])
    print("Padrão mensal:")
    print(monthly_pattern)

    # Agrupar por semana do ano
    weekly_pattern = trans_df.groupby(trans_df['transaction_date'].dt.isocalendar().week)['quantity'].agg(['count', 'mean', 'sum'])
    print("\nTop 10 semanas por volume:")
    print(weekly_pattern.sort_values('sum', ascending=False).head(10))

    print("\nBottom 10 semanas por volume:")
    print(weekly_pattern.sort_values('sum', ascending=True).head(10))

    return monthly_pattern, weekly_pattern

def generate_insights_summary(trans_df, pdv_mapping, product_mapping):
    """Gerar resumo de insights para modelagem"""
    print("\n" + "=" * 60)
    print("INSIGHTS PARA MODELAGEM")
    print("=" * 60)

    insights = []

    # 1. Distribuição de quantidade
    cv_quantity = trans_df['quantity'].std() / trans_df['quantity'].mean()
    insights.append(f"CV de quantidade: {cv_quantity:.3f} (variabilidade {'alta' if cv_quantity > 1 else 'moderada' if cv_quantity > 0.5 else 'baixa'})")

    # 2. Zeros e outliers
    zero_pct = (trans_df['quantity'] == 0).mean() * 100
    insights.append(f"Zeros: {zero_pct:.1f}% - {'Alto' if zero_pct > 20 else 'Moderado' if zero_pct > 10 else 'Baixo'}")

    # 3. Sazonalidade
    weekday_cv = trans_df.groupby(trans_df['transaction_date'].dt.day_name())['quantity'].mean().std() / trans_df.groupby(trans_df['transaction_date'].dt.day_name())['quantity'].mean().mean()
    insights.append(f"Sazonalidade semanal CV: {weekday_cv:.3f}")

    # 4. Concentração de PDVs/produtos
    pdv_concentration = (trans_df.groupby('internal_store_id')['quantity'].sum().sort_values(ascending=False).head(int(trans_df['internal_store_id'].nunique() * 0.2)).sum() / trans_df['quantity'].sum())
    insights.append(f"Top 20% PDVs representam {pdv_concentration:.1%} do volume")

    product_concentration = (trans_df.groupby('internal_product_id')['quantity'].sum().sort_values(ascending=False).head(int(trans_df['internal_product_id'].nunique() * 0.2)).sum() / trans_df['quantity'].sum())
    insights.append(f"Top 20% produtos representam {product_concentration:.1%} do volume")

    print("\n[INSIGHTS] INSIGHTS PRINCIPAIS:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    # Recomendações para features
    print("\n[FEATURES] RECOMENDACOES PARA FEATURES:")
    print("  • Lag features (1-4 semanas) - dados têm padrão temporal")
    print("  • Rolling means (2-8 semanas) - suavizar volatilidade")
    print("  • Features de concentração PDV/produto - alta concentração")
    print("  • Features sazonais (dia da semana, mês) - padrões identificados")
    print("  • Features de interação PDV×produto - capturar preferências")
    print("  • Tratamento especial para zeros - alta frequência" if zero_pct > 10 else "  • Zeros são raros - usar como está")

    return insights

def main():
    """Função principal"""
    try:
        # 1. Analisar dados de transação
        trans_df = analyze_transaction_data()
        if trans_df is None:
            return

        # 2. Analisar período target
        weekly_data = analyze_target_period(trans_df)

        # 3. Identificar mapeamento
        pdv_mapping, product_mapping = identify_hackathon_mapping(trans_df)

        # 4. Analisar sazonalidade
        monthly_pattern, weekly_pattern = analyze_seasonality_patterns(trans_df)

        # 5. Gerar insights
        insights = generate_insights_summary(trans_df, pdv_mapping, product_mapping)

        # 6. Salvar mapeamentos para uso posterior
        mapping_data = {
            'pdv_mapping': pdv_mapping,
            'product_mapping': product_mapping,
            'analysis_date': datetime.now().isoformat(),
            'sample_size': len(trans_df)
        }

        import json
        with open('hackathon_mapping.json', 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)

        print(f"\n[SAVE] Mapeamentos salvos em: hackathon_mapping.json")
        print("\n[OK] ANALISE CONCLUIDA!")
        print("[READY] Dados prontos para feature engineering e modelagem")

        return trans_df, pdv_mapping, product_mapping, weekly_data

    except Exception as e:
        print(f"[ERRO] ERRO na analise: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()