# 🖥️ Fase 1.2: Análise de Recursos Computacionais

## 📊 Caracterização do Dataset Real

### Tamanhos dos Arquivos Identificados
```
1. PDV Catalog: 0.3MB (14,419 lojas)
   - Arquivo: PDV master data
   - Colunas: pdv, premise, categoria_pdv, zipcode
   - Uso: Mapeamento de lojas e características

2. Transaction Data: 132.5MB (6,560,698 transações)
   - Arquivo: Dados de vendas históricas
   - Colunas: internal_store_id, internal_product_id, transaction_date, quantity, values
   - Uso: Core dataset para forecasting

3. Product Catalog: 559.8MB (192,356,316 produtos)
   - Arquivo: Master data de produtos
   - Colunas: produto, categoria, descricao, marca, fabricante
   - Uso: Features de produto e hierarquias
```

### Estimativa de Recursos Necessários

#### Memória RAM
```python
# Estimativas conservadoras para processamento completo
dataset_sizes = {
    'raw_data': 692.6,  # MB (soma dos arquivos)
    'processed_features': 2000,  # MB (após feature engineering)
    'model_training': 4000,  # MB (durante treinamento)
    'safety_buffer': 2000   # MB (buffer de segurança)
}
total_ram_needed = 8000  # MB (~8GB)
recommended_ram = 16000  # MB (~16GB)
```

#### Armazenamento
```python
storage_requirements = {
    'raw_data': 692.6,  # MB
    'processed_data': 1500,  # MB
    'feature_stores': 2000,  # MB
    'models': 500,  # MB (todos os modelos)
    'experiments': 1000,  # MB (MLflow artifacts)
    'submissions': 100   # MB
}
total_storage = 5792.6  # MB (~6GB)
recommended_storage = 20000  # MB (~20GB with safety)
```

## 🏭 Estratégia de Processamento

### Processamento Local (Recomendado)
```python
local_processing_strategy = {
    'approach': 'Chunked processing with optimization',
    'benefits': [
        'Zero cloud costs',
        'Full control over environment', 
        'Faster iteration cycles',
        'No data transfer limitations'
    ],
    'requirements': {
        'ram': '16GB (mínimo 8GB)',
        'storage': '20GB livre',
        'cpu': '8+ cores (recomendado)',
        'gpu': 'Opcional (para deep learning)'
    }
}
```

### Otimizações para Dataset Grande
```python
optimization_techniques = {
    'memory_optimization': {
        'chunked_processing': 'Process data in 100k row chunks',
        'dtype_optimization': 'Use categorical and int32 where possible',
        'lazy_loading': 'Load only needed columns with PyArrow'
    },
    'feature_engineering': {
        'incremental_features': 'Build features incrementally',
        'caching': 'Cache expensive computations',
        'parallel_processing': 'Use multiprocessing for feature creation'
    },
    'model_training': {
        'lightgbm_optimized': 'Use LightGBM memory efficiency',
        'early_stopping': 'Prevent overfitting and save time',
        'model_compression': 'Compress models for storage'
    }
}
```

## ☁️ Cloud Strategy (Backup Plan)

### Google Colab Pro (Se necessário)
```python
colab_strategy = {
    'when_to_use': 'Se RAM local < 8GB',
    'benefits': [
        'High RAM (up to 25GB)',
        'TPU access for deep learning',
        'Pre-installed ML libraries'
    ],
    'limitations': [
        'Session timeouts',
        'File upload/download overhead',
        'Limited persistent storage'
    ],
    'cost': '$10/month (Pro)'
}
```

### AWS/GCP (Para Produção)
```python
cloud_production_strategy = {
    'recommended_instance': 'c5.2xlarge (AWS) or n1-standard-8 (GCP)',
    'specs': {
        'vcpu': 8,
        'ram': '16GB',
        'storage': '100GB SSD'
    },
    'estimated_cost': '$50-100 for hackathon duration',
    'use_case': 'Large-scale hyperparameter tuning'
}
```

## 🔧 Configuração de Ambiente Otimizada

### Dependências Críticas
```python
critical_libraries = {
    'data_processing': [
        'pandas==2.1.0',
        'pyarrow==13.0.0',  # Fast parquet reading
        'numpy==1.24.0',
        'polars==0.19.0'  # Alternative for large data
    ],
    'machine_learning': [
        'lightgbm==4.0.0',  # Primary model
        'prophet==1.1.4',   # Seasonal forecasting  
        'scikit-learn==1.3.0',
        'optuna==3.3.0'     # Hyperparameter tuning
    ],
    'experiment_tracking': [
        'mlflow==2.6.0',
        'wandb==0.15.8'     # Alternative tracking
    ]
}
```

### Configuração de Memória
```python
memory_config = {
    'pandas_options': {
        'pd.options.mode.copy_on_write': True,
        'pd.options.compute.use_numba': True
    },
    'numpy_config': {
        'OMP_NUM_THREADS': 8,
        'MKL_NUM_THREADS': 8
    },
    'lightgbm_config': {
        'max_bin': 255,  # Reduce memory usage
        'num_threads': 8,
        'device_type': 'cpu'
    }
}
```

## 📈 Performance Benchmarks

### Processamento Esperado
```python
processing_benchmarks = {
    'data_loading': {
        'transactions_6.5M_rows': '30 seconds',
        'products_192M_rows': '2-3 minutes',
        'pdv_14k_rows': '< 1 second'
    },
    'feature_engineering': {
        'temporal_features': '10-15 minutes', 
        'cross_features': '20-30 minutes',
        'categorical_encoding': '5-10 minutes'
    },
    'model_training': {
        'lightgbm_single_model': '2-5 minutes',
        'prophet_model': '30-60 minutes',
        'ensemble_training': '10-20 minutes'
    }
}
```

## 🎯 Estratégia Final de Recursos

### Configuração Recomendada
1. **Local Development**: 16GB RAM, 20GB storage, 8+ cores
2. **Processing Strategy**: Chunked processing com otimizações de memória
3. **Backup Plan**: Google Colab Pro se recursos locais insuficientes
4. **Production**: AWS c5.2xlarge para hyperparameter tuning final

### Próximos Passos
1. **Verificar recursos locais** disponíveis
2. **Setup environment** com dependências otimizadas
3. **Implementar chunked processing** para arquivos grandes
4. **Configurar MLflow** para tracking de experimentos

---

*Fase 1.2 - Análise de Recursos Computacionais - COMPLETO ✅*

**DECISÃO**: Prosseguir com processamento local otimizado
**JUSTIFICATIVA**: Dataset manageable com técnicas de otimização
**BACKUP**: Google Colab Pro disponível se necessário