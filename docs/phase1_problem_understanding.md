# 🎯 Fase 1.1: Entendimento Profundo do Problema

## 📊 Análise Detalhada da Métrica WMAPE

### O que é WMAPE (Weighted Mean Absolute Percentage Error)

```python
# Fórmula WMAPE
WMAPE = (Σ |actual - forecast|) / (Σ |actual|) × 100%

# Características importantes:
# 1. Penaliza erros proporcionalmente ao volume real
# 2. Produtos com maior volume têm mais peso no erro final
# 3. Evita divisão por zero (diferente do MAPE tradicional)
# 4. Mais robusta para produtos intermitentes
```

### Por que WMAPE é Crítica para Retail Forecasting

1. **Volume-Weighted**: Produtos de alto volume impactam mais o resultado
2. **Business-Aligned**: Reflete melhor o impacto financeiro dos erros
3. **Robust**: Não quebra com vendas zero (diferente do MAPE)
4. **Interpretable**: Percentual de erro facilmente compreensível

### Estratégias para Otimizar WMAPE

#### Features Específicas para WMAPE
```python
# Features que minimizam erro percentual
- volume_weighted_features: features ponderadas pelo volume
- relative_performance: performance vs média da categoria
- forecast_difficulty: indicador de dificuldade de previsão
- error_sensitivity: produtos sensíveis a erro percentual
```

#### Tratamento Especializado por Volume
```python
# Segmentação por volume para tratamento diferenciado
high_volume_products = sales > percentile_90  # Foco máximo
medium_volume_products = percentile_50 < sales <= percentile_90
low_volume_products = sales <= percentile_50  # Menos crítico
```

## 🏪 Domain Expertise: One-Click Order & Retail

### Entendimento do Sistema One-Click Order

**Conceito**: Sistema automatizado de reposição que:
- Monitora níveis de estoque em tempo real
- Prevê demanda futura com base em padrões históricos
- Gera pedidos automáticos para evitar rupturas
- Otimiza custos de estoque vs risco de ruptura

### Desafios Específicos do Varejo

#### 1. Sazonalidade Complexa
```python
# Múltiplos ciclos sobrepostos
seasonal_patterns = {
    'weekly': 7,      # Padrão semanal (seg-dom)
    'monthly': 4.33,  # Padrão mensal (início vs fim)
    'quarterly': 13,  # Padrões trimestrais
    'yearly': 52      # Sazonalidade anual
}

# Eventos especiais
special_events = [
    'black_friday', 'natal', 'ano_novo', 'carnaval',
    'festa_junina', 'dia_das_maes', 'volta_as_aulas'
]
```

#### 2. Tipos de PDV e Comportamentos
```python
pdv_characteristics = {
    'c-store': {
        'behavior': 'convenience_focused',
        'peak_hours': ['morning_rush', 'evening_commute'],
        'top_categories': ['beverages', 'snacks', 'cigarettes'],
        'seasonality_strength': 'medium'
    },
    'g-store': {
        'behavior': 'grocery_shopping',
        'peak_hours': ['weekends', 'after_work'],
        'top_categories': ['groceries', 'household'],
        'seasonality_strength': 'high'
    },
    'liquor': {
        'behavior': 'recreational_weekend',
        'peak_hours': ['friday_evening', 'weekends'],
        'top_categories': ['alcoholic_beverages'],
        'seasonality_strength': 'very_high'
    }
}
```

#### 3. Ciclos de Reposição Típicos
```python
# Frequências típicas por categoria
restock_cycles = {
    'beverages': 2,        # 2x por semana
    'snacks': 1,          # 1x por semana
    'cigarettes': 3,      # 3x por semana (alto giro)
    'grocery': 1,         # 1x por semana
    'household': 0.5,     # quinzenal
    'seasonal': 0.25      # mensal
}
```

### Padrões de Demanda no Varejo

#### Produtos por Velocidade de Giro
```python
velocity_classification = {
    'fast_moving': {
        'characteristics': 'high_volume, stable_demand, low_intermittency',
        'forecast_approach': 'time_series_focused',
        'key_features': ['recent_trends', 'seasonal_patterns']
    },
    'medium_moving': {
        'characteristics': 'moderate_volume, some_seasonality',
        'forecast_approach': 'hybrid_approach',
        'key_features': ['category_trends', 'cross_selling']
    },
    'slow_moving': {
        'characteristics': 'low_volume, high_intermittency',
        'forecast_approach': 'similarity_based',
        'key_features': ['category_averages', 'substitution_effects']
    }
}
```

#### Market Basket Analysis Impact
```python
# Produtos frequentemente comprados juntos
cross_selling_pairs = {
    'beer': ['snacks', 'cigarettes'],
    'coffee': ['sugar', 'milk'],
    'bread': ['butter', 'jam'],
    'pasta': ['tomato_sauce', 'cheese']
}

# Produtos substitutos (competem entre si)
substitution_groups = {
    'sodas': ['coca_cola', 'pepsi', 'guarana'],
    'beers': ['brahma', 'skol', 'antarctica'],
    'cigarettes': ['marlboro', 'lucky_strike', 'hollywood']
}
```

## 🔬 Research de Benchmarking

### Modelos Tradicionais (Baseline Esperado)
```python
traditional_approaches = {
    'naive_seasonal': {
        'description': 'Same week last year',
        'pros': 'Simple, captures yearly seasonality',
        'cons': 'No trend, no recent adaptations'
    },
    'moving_average': {
        'description': 'Average of last N periods',
        'pros': 'Smooth, reduces noise',
        'cons': 'Lags behind trends'
    },
    'exponential_smoothing': {
        'description': 'Holt-Winters triple exponential',
        'pros': 'Trend + seasonality',
        'cons': 'Fixed parameters, limited features'
    },
    'arima_sarima': {
        'description': 'Autoregressive integrated moving average',
        'pros': 'Statistical rigor, confidence intervals',
        'cons': 'Univariate, assumes stationarity'
    }
}
```

### Modelos Avançados (Nossa Estratégia)
```python
advanced_approaches = {
    'prophet': {
        'strengths': 'Multiple seasonalities, holidays, changepoints',
        'use_case': 'Products with clear trends and seasonality'
    },
    'lightgbm': {
        'strengths': 'Feature interactions, categorical handling',
        'use_case': 'Rich feature sets, non-linear patterns'
    },
    'lstm_gru': {
        'strengths': 'Long-term dependencies, sequence patterns',
        'use_case': 'Complex temporal patterns'
    },
    'ensemble_stacking': {
        'strengths': 'Combines all approaches optimally',
        'use_case': 'Final solution for maximum accuracy'
    }
}
```

## 🎯 Estratégia de Superação do Baseline

### 1. Feature Engineering Superior
- **Temporal features avançadas**: Múltiplos lags, rolling statistics, fourier components
- **Cross-features inteligentes**: Produto×PDV, Categoria×Região
- **Business features**: Lifecycle stage, complementaridade, substituição

### 2. Tratamento de Casos Especiais
- **Cold start**: Produtos novos via similarity-based forecasting
- **Intermittency**: Zero-inflated models, Croston method
- **Outliers**: Robust models, business rules

### 3. Validação Temporal Robusta
- **Walk-forward validation**: Simula cenário real de previsão
- **Time-based splits**: Nunca usar shuffle em time series
- **Multiple horizons**: Validar consistência em diferentes horizontes

### 4. Domain-Driven Insights
- **Sazonalidade retail**: Padrões específicos de varejo
- **PDV characteristics**: Comportamentos por tipo de estabelecimento
- **Category dynamics**: Interações entre categorias de produtos

## 📋 Próximos Passos

### Análise Imediata dos Dados
1. **Explorar estrutura**: Entender schema dos parquets
2. **Data quality**: Missing values, outliers, consistência
3. **Temporal patterns**: Identificar sazonalidades
4. **Volume distribution**: Entender distribuição ABC

### Features Prioritárias (Fase 3)
1. **Lags temporais**: 1,2,3,4 semanas
2. **Rolling statistics**: médias móveis 4,8,12,26 semanas
3. **Sazonalidade**: week_of_year, month_interactions
4. **Cross-features**: product_pdv_affinity, category_share

### Modelos Prioritários (Fase 4)  
1. **Prophet**: Baseline forte com sazonalidade
2. **LightGBM**: Feature-rich approach
3. **Ensemble**: Stacking dos melhores modelos

---

*Documento criado seguindo a estratégia definitiva para vencer o hackathon*
*Fase 1.1 - Entendimento Profundo do Problema - COMPLETO ✅*