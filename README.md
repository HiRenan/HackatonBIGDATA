# 🏆 Hackathon Forecast Big Data 2025

## 📋 Competição de Previsão de Vendas no Varejo

Projeto desenvolvido para o **Hackathon Forecast Big Data 2025**, focado em previsão de vendas por PDV e SKU usando técnicas avançadas de Machine Learning e Time Series.

### 🎯 Objetivo da Competição
- **Métrica Principal**: WMAPE (Weighted Mean Absolute Percentage Error)
- **Domínio**: Forecasting para sistema One-Click Order no varejo
- **Dados**: 199+ milhões de registros de transações, produtos e PDVs
- **Entrega**: Previsões em CSV + repositório GitHub

## 🗂️ Estrutura do Projeto

```
hackathon_forecast_2025/
├── data/
│   ├── raw/                    # Dados originais (parquets)
│   ├── processed/              # Dados processados
│   ├── features/               # Feature stores
│   └── mock/                   # Dados de teste
├── notebooks/
│   ├── 01_eda/                # Análise exploratória
│   ├── 02_preprocessing/       # Pré-processamento
│   ├── 03_feature_engineering/ # Criação de features
│   └── 04_modeling/           # Modelagem
├── src/
│   ├── config/                # Configurações
│   ├── utils/                 # Utilitários
│   ├── preprocessing/         # Pré-processamento
│   ├── features/              # Feature engineering
│   ├── models/                # Modelos de ML
│   ├── evaluation/            # Métricas e avaliação
│   └── experiment_tracking/   # MLflow setup
├── models/
│   └── trained/               # Modelos treinados
├── submissions/               # Arquivos de submissão
├── docs/                      # Documentação técnica
└── tests/                     # Testes unitários
```

## 📊 Dataset Overview

### Arquivos Identificados
1. **PDV Catalog** (14,419 lojas)
   - Dados: tipos de estabelecimento, localização, categorias
   - Tamanho: 0.3 MB

2. **Transaction Data** (6.5M transações)  
   - Dados: vendas históricas com quantidades e valores
   - Tamanho: 132.5 MB

3. **Product Catalog** (192M produtos)
   - Dados: categorias, marcas, descrições de produtos
   - Tamanho: 559.8 MB

## 🚀 Setup Rápido

### 1. Instalação do Ambiente
```bash
# Clone o repositório
cd hackathon_forecast_2025

# Instalar dependências
pip install -r requirements.txt

# Ou setup automatizado
python setup_environment.py
```

### 2. Verificação dos Dados
```bash
# Executar análise inicial
cd notebooks/01_eda
python initial_data_exploration.py
```

### 3. Iniciar MLflow
```bash
# Tracking de experimentos
mlflow ui
# Acesse: http://localhost:5000
```

## 🔬 Estratégia Técnica

### Abordagem de Modelagem
1. **Prophet**: Baseline com sazonalidade múltipla
2. **LightGBM**: Feature-rich approach com otimizações
3. **Ensemble**: Stacking dos melhores modelos

### Features Principais
- **Temporais**: Lags, rolling statistics, sazonalidade
- **Cross-features**: Produto×PDV, categoria×região
- **Business**: Lifecycle, complementaridade, substituição
- **Volume-weighted**: Features ponderadas por volume

### Otimizações WMAPE
- Tratamento especializado por volume de produto
- Segmentação A/B/C para estratégias diferenciadas
- Features específicas para minimizar erro percentual

## 📈 Métricas de Avaliação

### Métrica Principal
```python
# WMAPE (Weighted Mean Absolute Percentage Error)
WMAPE = (Σ |actual - forecast|) / (Σ |actual|) × 100%
```

### Métricas Secundárias
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric MAPE
- **MAE**: Mean Absolute Error
- **Volume-weighted metrics**: Para produtos de alto giro

## 🛠️ Utilização

### Carregar Dados com Otimização de Memória
```python
from src.utils.data_loader import load_data_efficiently

# Carrega datasets com sample inteligente
trans_df, prod_df, pdv_df = load_data_efficiently(
    data_path="data/raw",
    sample_transactions=500000,
    sample_products=100000
)
```

### Experiment Tracking
```python
from src.experiment_tracking.mlflow_setup import HackathonMLflowTracker

# Inicializar tracker
tracker = HackathonMLflowTracker()

# Começar experimento
run_id = tracker.start_run("lightgbm_v1", model_type="lightgbm")

# Log métricas
tracker.log_training_metrics({"wmape": 15.2, "mape": 18.5})
```

### Avaliação de Modelos
```python
from src.evaluation.metrics import wmape, retail_forecast_evaluation

# Calcular WMAPE (métrica da competição)
score = wmape(y_true, y_pred)

# Avaliação completa
results = retail_forecast_evaluation(y_true_df, y_pred_df)
```

## 🎯 Fases do Desenvolvimento

### ✅ Fase 1: Setup e Entendimento (COMPLETO)
- [x] Análise profunda do problema e métrica WMAPE
- [x] Exploração inicial dos dados (199M+ registros)
- [x] Análise de recursos computacionais
- [x] Setup completo do ambiente de desenvolvimento
- [x] Configuração de tracking de experimentos

### 📝 Próximas Fases
- **Fase 2**: EDA completa e data quality
- **Fase 3**: Feature engineering avançado
- **Fase 4**: Modelagem e otimização
- **Fase 5**: Ensemble e fine-tuning
- **Fase 6**: Validação final e submission

## 📋 Requisitos de Sistema

### Mínimos
- **RAM**: 8GB (16GB recomendado)
- **Storage**: 20GB livres
- **CPU**: 4+ cores
- **Python**: 3.10+

### Dependências Principais
- **pandas**, **pyarrow**: Processamento de dados
- **lightgbm**: Modelo principal
- **prophet**: Forecasting sazonal
- **mlflow**: Experiment tracking
- **scikit-learn**: ML utilities

## 🏁 Submissão

### Formato da Submissão
```csv
store_id,product_id,date,prediction
1,100,2024-01-01,150.5
1,100,2024-01-02,148.2
...
```

### Comando de Submissão
```python
from src.experiment_tracking.mlflow_setup import HackathonMLflowTracker

tracker = HackathonMLflowTracker()
tracker.log_submission(
    submission_df, 
    submission_name="final_ensemble_v1"
)
```

## 📚 Documentação Técnica

- **[Problem Understanding](docs/phase1_problem_understanding.md)**: Análise detalhada da métrica WMAPE
- **[Resource Analysis](docs/phase1_2_resource_analysis.md)**: Estratégia de recursos computacionais
- **[Strategy Document](estrategia_vitoria_completa.md)**: Estratégia completa para vencer

## 🤝 Desenvolvimento

### Estrutura de Commits
```bash
# Seguir padrão
git commit -m "feat: implement lightgbm baseline model

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Experiment Tracking
- Todos experimentos trackados via MLflow
- Métricas padronizadas por tipo de modelo
- Versionamento automático de modelos

---

## 🎯 Status Atual

**Fase 1 COMPLETA** ✅
- Setup e infraestrutura prontos
- Dados analisados e carregados
- Ambiente otimizado para processamento
- Experiment tracking configurado

**Próximo**: Iniciar Fase 2 - EDA Completa

---

*Projeto desenvolvido seguindo estratégia otimizada para competição de forecasting*
*Foco em WMAPE optimization e técnicas de retail forecasting*