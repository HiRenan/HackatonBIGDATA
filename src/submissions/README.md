# Submissions Module - Phase 7: Strategic Competition Management

Este módulo implementa um sistema completo e sofisticado para gerenciamento estratégico de submissões em competições de machine learning, especificamente otimizado para o Hackathon Forecast 2025.

## 🎯 Visão Geral

O sistema de submissões Phase 7 oferece:

- **Estratégias de Submissão**: Múltiplas estratégias desde baseline até ensembles otimizados
- **Gestão de Riscos**: Avaliação automática de riscos de overfitting, vazamento de dados e complexidade
- **Análise Competitiva**: Inteligência competitiva baseada em leaderboards
- **Gestão de Cronograma**: Otimização de timing para submissões estratégicas
- **Pós-processamento**: Aplicação inteligente de regras de negócio e ajustes
- **Pipeline Automatizado**: Execução end-to-end com validações e checkpoints

## 📁 Estrutura do Módulo

```
src/submissions/
├── __init__.py                 # Inicialização do módulo
├── strategy.py                 # Estratégias de submissão
├── risk_manager.py            # Gestão de riscos
├── leaderboard_analyzer.py    # Análise competitiva
├── submission_pipeline.py     # Pipeline principal
├── timeline_manager.py        # Gestão de cronograma
├── post_processor.py          # Pós-processamento
└── README.md                  # Esta documentação

scripts/submissions/
├── submit_model.py            # Script principal de submissão
├── analyze_competition.py     # Análise competitiva
├── manage_timeline.py         # Gestão de cronograma
├── validate_submission.py     # Validação de submissões
└── generate_final_submission.py  # Submissão final

tests/test_submissions/
├── test_strategy.py           # Testes de estratégias
├── test_risk_manager.py       # Testes de gestão de riscos
├── test_leaderboard_analyzer.py  # Testes de análise competitiva
├── test_integration.py        # Testes de integração
└── test_suite.py             # Suite completa de testes
```

## 🚀 Início Rápido

### 1. Submissão Básica

```bash
# Submissão baseline simples
python scripts/submissions/submit_model.py baseline \
    --data-path data/raw \
    --config src/config/environments/submission.yaml

# Submissão com modelo único
python scripts/submissions/submit_model.py single_model \
    --data-path data/raw \
    --leaderboard leaderboard.csv \
    --team-name "nossa_equipe"
```

### 2. Análise Competitiva

```bash
# Análise da competição
python scripts/submissions/analyze_competition.py \
    --leaderboard leaderboard.csv \
    --team-name "nossa_equipe" \
    --competition-days-remaining 5

# Gestão de cronograma
python scripts/submissions/manage_timeline.py \
    --status \
    --competition-end "2025-01-31 23:59"
```

### 3. Uso Programático

```python
from src.submissions.strategy import SubmissionStrategyFactory
from src.submissions.submission_pipeline import create_submission_pipeline

# Criar estratégia
strategy = SubmissionStrategyFactory.create('ensemble', config)

# Executar pipeline
pipeline = create_submission_pipeline(pipeline_config)
result = pipeline.execute_submission_pipeline(
    submission_strategy=strategy,
    train_data=train_df,
    test_data=test_df,
    team_name='nossa_equipe'
)
```

## 🎯 Estratégias de Submissão

### 1. Baseline Strategy
- **Modelo**: Prophet para padrões sazonais
- **Risco**: Baixo
- **Uso**: Submissão inicial rápida

### 2. Single Model Strategy
- **Modelo**: LightGBM otimizado
- **Risco**: Médio
- **Uso**: Modelo principal de produção

### 3. Ensemble Strategy
- **Modelos**: LightGBM + Prophet + Tree Ensemble
- **Método**: Stacking com meta-learner
- **Risco**: Médio
- **Uso**: Melhoria de performance

### 4. Optimized Ensemble Strategy
- **Modelos**: Múltiplos algoritmos
- **Otimização**: Optuna/Grid Search
- **Risco**: Alto
- **Uso**: Máxima performance

### 5. Final Strategy
- **Abordagem**: Ensemble de todas as estratégias anteriores
- **Otimização**: Final optimization
- **Risco**: Alto
- **Uso**: Submissão final de competição

## 🛡️ Gestão de Riscos

### Tipos de Risco Avaliados

1. **Overfitting Risk**
   - Gap entre treino e validação
   - Thresholds configuráveis
   - Recomendações automáticas

2. **Complexity Risk**
   - Número de features
   - Profundidade do modelo
   - Tempo de treinamento

3. **Leakage Risk**
   - Scores suspeitosamente altos
   - Validação de dados

4. **Execution Risk**
   - Uso de memória
   - Tempo de predição
   - Recursos computacionais

### Níveis de Risco

- **LOW**: Submissão segura
- **MEDIUM**: Revisão recomendada
- **HIGH**: Submissão bloqueada

## 🏆 Análise Competitiva

### Métricas Analisadas

- **Posição Atual**: Rank e percentil
- **Zonas Competitivas**: Leader, Contender, Middle Pack, Bottom Tier
- **Gaps**: Distância para top 3, top 10, próxima posição
- **Achievability Score**: Probabilidade de alcançar targets

### Recomendações Estratégicas

- **Leader**: Estratégia conservadora
- **Contender**: Estratégia agressiva balanceada
- **Middle Pack**: Estratégia de melhoria incremental
- **Bottom Tier**: Estratégia de catch-up agressiva

## ⏰ Gestão de Cronograma

### Janelas de Submissão

1. **Baseline** (Dia 14): Submissão inicial
2. **Single Model** (Dia 10): Modelo principal
3. **Initial Ensemble** (Dia 7): Primeiro ensemble
4. **Optimized Ensemble** (Dia 3): Ensemble otimizado
5. **Final Submission** (Último dia): Submissão final

### Alertas Automáticos

- **48h antes**: Alerta de preparação
- **12h antes**: Alerta urgente
- **4h antes**: Alerta crítico

## 🔧 Pós-processamento

### Regras de Negócio

- **Non-negativity**: Elimina valores negativos
- **Growth Rate Limits**: Limita crescimento irreal
- **Seasonality Constraints**: Preserva padrões sazonais

### Ajustes por Outliers

- **Quantile Capping**: Limita valores extremos
- **IQR Method**: Remove outliers estatísticos
- **Z-score**: Normalização de distribuições

### Ajustes Competitivos

- **Leader**: Pós-processamento conservador
- **Contender**: Pós-processamento balanceado
- **Catch-up**: Pós-processamento agressivo

## 📊 Pipeline de Submissão

### Etapas do Pipeline

1. **Data Validation**: Validação de qualidade dos dados
2. **Model Training**: Treinamento do modelo
3. **Risk Assessment**: Avaliação de riscos
4. **Competitive Analysis**: Análise competitiva
5. **Post Processing**: Aplicação de regras
6. **Submission Execution**: Execução final

### Configuração de Ambientes

- **development_submission.yaml**: Desenvolvimento rápido
- **submission.yaml**: Ambiente padrão
- **production_submission.yaml**: Máxima performance

## 🧪 Testes

### Executar Testes

```bash
# Todos os testes
python tests/test_submissions/test_suite.py

# Testes específicos
python tests/test_submissions/test_suite.py --class strategy
python tests/test_submissions/test_suite.py --class risk_manager
python tests/test_submissions/test_suite.py --class leaderboard_analyzer

# Testes de integração
python -m pytest tests/test_submissions/test_integration.py -v
```

### Cobertura de Testes

- **Unit Tests**: Cada componente individual
- **Integration Tests**: Interação entre componentes
- **End-to-End Tests**: Pipeline completo

## 🔧 Configuração

### Arquivo Principal: `submission.yaml`

```yaml
submission_strategies:
  baseline:
    model_config:
      prophet_config:
        daily_seasonality: true
        weekly_seasonality: true
    risk_tolerance: low

risk_management:
  enable_overfitting_assessment: true
  overfitting_config:
    max_acceptable_gap: 0.1

competitive_analysis:
  leaderboard_config:
    top_tier_size: 3
    contender_tier_size: 10
```

### Personalização

- **Strategy Config**: Ajuste modelos e hiperparâmetros
- **Risk Thresholds**: Configure níveis de risco aceitáveis
- **Timeline Windows**: Personalize janelas de submissão
- **Post-processing Rules**: Defina regras de negócio específicas

## 📈 Monitoramento e Logging

### MLflow Integration

- **Experiment Tracking**: Todas as execuções são rastreadas
- **Model Versioning**: Versionamento automático
- **Metrics Logging**: Métricas de performance e risco

### Logs Detalhados

- **Risk Assessment**: Logs detalhados de avaliação de risco
- **Competitive Analysis**: Análise competitiva timestamped
- **Pipeline Execution**: Status de cada etapa

## 🚨 Troubleshooting

### Problemas Comuns

1. **Submissão Bloqueada por Alto Risco**
   - Verifique overfitting gap
   - Reduza complexidade do modelo
   - Ajuste thresholds de risco

2. **Erro na Análise Competitiva**
   - Verifique formato do leaderboard
   - Confirme nome da equipe
   - Valide dados de score

3. **Falha no Pipeline**
   - Verifique logs detalhados
   - Valide dados de entrada
   - Confirme configurações

### Debug Mode

```python
# Ativar logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)

# Executar com configuração de debug
config['debug'] = {
    'enable_detailed_logging': True,
    'save_debug_artifacts': True
}
```

## 🤝 Contribuição

### Adicionar Nova Estratégia

1. Extend `SubmissionStrategy` class
2. Implement abstract methods
3. Register in `SubmissionStrategyFactory`
4. Add configuration to YAML
5. Write unit tests

### Adicionar Novo Tipo de Risco

1. Extend base risk assessment
2. Implement `assess` method
3. Register in `RiskManager`
4. Add configuration options
5. Write unit tests

## 📄 Licença

Este módulo é parte do projeto Hackathon Forecast 2025 e segue as mesmas diretrizes de licenciamento do projeto principal.

---

## 💡 Dicas para Competição

1. **Comece com Baseline**: Sempre faça uma submissão baseline primeiro
2. **Monitore Riscos**: Use a gestão de riscos para evitar overfitting
3. **Análise Competitiva**: Ajuste estratégia baseada na posição no leaderboard
4. **Timing Estratégico**: Use as janelas de submissão otimizadas
5. **Pós-processamento**: Aplique regras de negócio para melhorar scores
6. **Backup Plans**: Sempre tenha uma submissão de backup segura

**Boa sorte na competição! 🏆**