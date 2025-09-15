# Guia de Solução de Problemas - Dependências

## 🚨 Problemas Comuns de Instalação

### 1. **Python 3.13+ Incompatibilidade**

**Problema**: Erros ao instalar Prophet, LightGBM ou outras dependências
```
ERROR: Could not build wheels for prophet
ERROR: Failed building wheel for lightgbm
```

**Solução**:
```bash
# Opção 1: Use Python 3.10 (recomendado)
conda create -n hackathon python=3.10
conda activate hackathon
pip install -r requirements-py310.txt

# Opção 2: Instale apenas dependências essenciais
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 2. **Conflitos de Versões**

**Problema**: Mensagens de dependências conflitantes
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solução**:
```bash
# Use ambiente limpo
conda create -n fresh_env python=3.10
conda activate fresh_env

# Instale por etapas
pip install numpy pandas
pip install scikit-learn lightgbm
pip install matplotlib seaborn plotly
pip install jupyter
```

### 3. **Prophet Falha na Instalação**

**Problema**: Prophet não instala (comum no Windows/Python 3.13)
```bash
# Soluções alternativas:
conda install -c conda-forge prophet  # Opção 1
pip install prophet --no-deps         # Opção 2
# Ou pule Prophet inicialmente
```

### 4. **TensorFlow/PyTorch Conflitos**

**Problema**: TensorFlow e PyTorch causam conflitos
```bash
# Instale apenas um por vez
pip install torch  # Para LSTM
# OU
pip install tensorflow  # Alternativa
```

## 🛠️ Soluções por Sistema Operacional

### Windows
```bash
# Use conda para evitar problemas de compilação
conda install -c conda-forge numpy pandas scikit-learn lightgbm
pip install prophet mlflow jupyter
```

### macOS (Apple Silicon)
```bash
# Versões específicas para M1/M2
pip install --upgrade pip
pip install numpy pandas scikit-learn
conda install -c conda-forge lightgbm
```

### Linux
```bash
# Instale dependências do sistema primeiro
sudo apt-get update
sudo apt-get install build-essential
pip install -r requirements-py310.txt
```

## 📋 Configurações de Fallback

Se nada funcionar, use esta configuração mínima:

```bash
# Configuração mínima garantida
pip install --upgrade pip
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0
pip install jupyter

# Adicione gradualmente:
pip install lightgbm  # Para ML
pip install mlflow     # Para tracking
pip install plotly     # Para visualização
```

## 🔧 Comandos de Depuração

```bash
# Verificar versões instaladas
pip list
python --version

# Verificar conflitos
pip check

# Reinstalar dependência problemática
pip uninstall lightgbm
pip install lightgbm --no-cache-dir

# Limpar cache pip
pip cache purge
```

## 🆘 Último Recurso

Se persistir problemas:

1. **Use Google Colab**:
   - Upload do projeto para Colab
   - Environment pré-configurado
   - Sem problemas de dependências locais

2. **Docker** (se disponível):
   ```bash
   docker-compose up
   # Environment isolado e testado
   ```

3. **Virtual Machine**:
   - Ubuntu 22.04 LTS
   - Python 3.10 pré-instalado
   - Dependências mais estáveis

## 📞 Contato para Suporte

Se nenhuma solução funcionar:
1. Abra uma issue com detalhes do erro
2. Inclua: SO, versão Python, log completo do erro
3. Mencione tentativas de correção realizadas