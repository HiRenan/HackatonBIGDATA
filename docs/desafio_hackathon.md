# Desafio Técnico – Hackathon Forecast Big Data 2025

## 🎯 Objetivo

Desenvolver um modelo de previsão de vendas (forecast) para apoiar o varejo na reposição de produtos. A tarefa é prever a **quantidade semanal de vendas por PDV (Ponto de Venda) / SKU (Stock Keeping Unit)** para as **próximas quatro semanas** (primeiras quatro semanas de janeiro/2023), utilizando como base o histórico de vendas de 2022.

*Este é um problema real, baseado no produto One-Click Order da Big Data.*

## 📂 Dados Disponíveis

### Dados de treino (2022)
**1 ano de dados transacionais (2022)** para trabalhar, criar o modelo, fazer testes e desenvolver a solução final:

- **Transações**: Data da transação, PDV, Produto, Quantidade, Faturamento
- **Cadastro de produtos**: Produto, Categoria, Descrição e até 4 características adicionais
- **Cadastro de PDVs**: PDV, On/Off Prem, Categoria PDV (c-store, g-store, liquor, etc.), Zipcode

### Dados de teste (Jan/2023)
- **Janeiro/2023** – mesmo formato dos dados de treino
- **Não será compartilhado** com os participantes
- Usado apenas pela Big Data para avaliar as previsões enviadas

## 📑 Entregáveis

### 1. Arquivo de previsão
Arquivo final **.csv** ou **.parquet** com as colunas: **Semana | PDV | Produto | Quantidade**

```
semana    pdv    produto    quantidade
1         1023   123        120
2         1045   234        85
3         1023   456        110
4         1023   456        95
```

**Especificações CSV:**
- Separador: ";" (ponto e vírgula)
- Encoding: UTF-8
- Exemplo: `1;1023;123;120`

**Campos:**
- `semana` (inteiro): número da semana (1 a 4 de janeiro/2023)
- `pdv` (inteiro): código do ponto de venda
- `produto` (inteiro): código do SKU
- `quantidade` (inteiro): previsão de vendas

### 2. Repositório GitHub
Repositório **público** contendo:
- Código completo e documentação da solução
- Instruções claras de execução (README)

## 📤 Submissões

- **Máximo**: Até 5 submissões durante o período de entrega
- **Critério**: Será considerada a submissão de melhor resultado para efeito de ranqueamento
- **Prazo**: Submissões fora do prazo não serão avaliadas
- **Plataforma**: Site oficial do Hackathon
- **Processamento**: Leaderboard atualizado em até 20 minutos
- **Ranking**: Nome do participante/equipe, WMAPE (%), posição (ordenado crescente - menor é melhor)

## 🧮 Avaliação

### Critérios de Avaliação:
1. **Acurácia** – métrica oficial definida pela Big Data (será divulgada junto com a abertura do desafio)
2. **Qualidade técnica** – clareza, organização e replicabilidade do código
3. **Criatividade na abordagem** – estratégias inovadoras ou eficazes de modelagem
4. **Desempenho em relação ao baseline da Big Data** – o modelo do participante deve superar o resultado obtido pelo algoritmo interno da empresa nos mesmos dados de teste

### ⚠️ Condição Obrigatória
**O código enviado deve ser executável e gerar os resultados apresentados.**

Caso todas as soluções falhem neste requisito, **nenhuma equipe será premiada**.

Mesmo com boa posição no leaderboard, a solução pode ser **invalidada** se não atender aos critérios de execução:
- Código não executável
- Resultado inconsistente ou incompleto