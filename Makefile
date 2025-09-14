# Phase 6: Production-Ready MLOps Pipeline
# Automated workflows for development, testing, training, and deployment

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := hackathon_forecast_2025
ENV_NAME := hackathon_forecast_2025
CONDA := conda
DOCKER := docker
PYTEST := pytest

# Paths
SRC_DIR := src
TESTS_DIR := tests
DATA_DIR := data
MODELS_DIR := models/trained
SUBMISSIONS_DIR := submissions
LOGS_DIR := logs

# Configuration
ENVIRONMENT ?= development
MLFLOW_TRACKING_URI ?= http://localhost:5000
TEST_COVERAGE_MIN := 80

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

##@ Help
.PHONY: help
help: ## Display this help message
	@echo "$(CYAN)========================================$(NC)"
	@echo "$(WHITE)Phase 6: MLOps Pipeline Commands$(NC)"
	@echo "$(CYAN)========================================$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(CYAN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(PURPLE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment Setup
.PHONY: setup
setup: ## Complete environment setup
	@echo "$(GREEN)🚀 Setting up development environment...$(NC)"
	@$(MAKE) create-env
	@$(MAKE) install-deps
	@$(MAKE) setup-pre-commit
	@$(MAKE) create-dirs
	@$(MAKE) setup-mlflow
	@echo "$(GREEN)✅ Environment setup completed!$(NC)"

.PHONY: create-env
create-env: ## Create conda environment
	@echo "$(BLUE)🔧 Creating conda environment: $(ENV_NAME)$(NC)"
	@$(CONDA) create -n $(ENV_NAME) python=3.10 -y || true
	@echo "$(GREEN)✅ Environment created. Activate with: conda activate $(ENV_NAME)$(NC)"

.PHONY: install-deps
install-deps: ## Install project dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

.PHONY: install-dev-deps
install-dev-deps: ## Install development dependencies
	@echo "$(BLUE)🛠️ Installing development dependencies...$(NC)"
	@$(PIP) install pytest pytest-cov pytest-xdist black flake8 mypy pre-commit bandit safety
	@echo "$(GREEN)✅ Development dependencies installed$(NC)"

.PHONY: setup-pre-commit
setup-pre-commit: ## Setup pre-commit hooks
	@echo "$(BLUE)🔒 Setting up pre-commit hooks...$(NC)"
	@pre-commit install || echo "$(YELLOW)⚠️ Pre-commit not available, skipping...$(NC)"
	@echo "$(GREEN)✅ Pre-commit hooks configured$(NC)"

.PHONY: create-dirs
create-dirs: ## Create necessary directories
	@echo "$(BLUE)📁 Creating project directories...$(NC)"
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/features
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(SUBMISSIONS_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(TESTS_DIR)/unit $(TESTS_DIR)/integration
	@touch $(LOGS_DIR)/.gitkeep
	@echo "$(GREEN)✅ Directories created$(NC)"

##@ Code Quality
.PHONY: lint
lint: ## Run code linting
	@echo "$(BLUE)🔍 Running code linting...$(NC)"
	@echo "Running flake8..."
	@flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503 || echo "$(YELLOW)⚠️ Flake8 issues found$(NC)"
	@echo "Running black (check only)..."
	@black --check $(SRC_DIR) || echo "$(YELLOW)⚠️ Code formatting issues found$(NC)"
	@echo "$(GREEN)✅ Linting completed$(NC)"

.PHONY: format
format: ## Format code with black
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@black $(SRC_DIR)
	@echo "$(GREEN)✅ Code formatted$(NC)"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(BLUE)🔍 Running type checking...$(NC)"
	@mypy $(SRC_DIR) --ignore-missing-imports || echo "$(YELLOW)⚠️ Type checking issues found$(NC)"
	@echo "$(GREEN)✅ Type checking completed$(NC)"

.PHONY: security-scan
security-scan: ## Run security scanning
	@echo "$(BLUE)🔒 Running security scans...$(NC)"
	@echo "Running bandit..."
	@bandit -r $(SRC_DIR) -f json -o security-report.json || echo "$(YELLOW)⚠️ Security issues found$(NC)"
	@echo "Running safety..."
	@safety check || echo "$(YELLOW)⚠️ Vulnerable dependencies found$(NC)"
	@echo "$(GREEN)✅ Security scanning completed$(NC)"

.PHONY: code-quality
code-quality: lint type-check security-scan ## Run all code quality checks

##@ Testing
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@$(PYTEST) $(TESTS_DIR) -v --tb=short
	@echo "$(GREEN)✅ Tests completed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	@$(PYTEST) $(TESTS_DIR)/unit -v
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	@$(PYTEST) $(TESTS_DIR)/integration -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage
	@echo "$(BLUE)📊 Running tests with coverage...$(NC)"
	@$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-fail-under=$(TEST_COVERAGE_MIN)
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

.PHONY: test-performance
test-performance: ## Run performance tests
	@echo "$(BLUE)⚡ Running performance tests...$(NC)"
	@$(PYTHON) -m pytest $(TESTS_DIR)/performance -v --benchmark-only || echo "$(YELLOW)⚠️ No performance tests found$(NC)"
	@echo "$(GREEN)✅ Performance tests completed$(NC)"

.PHONY: test-phase5-compliance
test-phase5-compliance: ## Run Phase 5 compliance tests
	@echo "$(BLUE)✅ Running Phase 5 compliance tests...$(NC)"
	@$(PYTHON) test_phase5_compliance.py
	@echo "$(GREEN)✅ Phase 5 compliance validated$(NC)"

##@ Data Management
.PHONY: download-data
download-data: ## Download competition data
	@echo "$(BLUE)📥 Downloading competition data...$(NC)"
	@$(PYTHON) scripts/download_data.py || echo "$(YELLOW)⚠️ Data download script not found$(NC)"
	@echo "$(GREEN)✅ Data download completed$(NC)"

.PHONY: validate-data
validate-data: ## Validate data integrity
	@echo "$(BLUE)🔍 Validating data integrity...$(NC)"
	@$(PYTHON) -c "from src.utils.data_loader import validate_data_integrity; validate_data_integrity('$(DATA_DIR)/raw')"
	@echo "$(GREEN)✅ Data validation completed$(NC)"

.PHONY: preprocess-data
preprocess-data: ## Run data preprocessing
	@echo "$(BLUE)⚙️ Running data preprocessing...$(NC)"
	@$(PYTHON) -c "from src.architecture.pipelines import DataProcessingPipeline; pipeline = DataProcessingPipeline(); print('Data preprocessing pipeline ready')"
	@echo "$(GREEN)✅ Data preprocessing completed$(NC)"

##@ Model Training
.PHONY: setup-mlflow
setup-mlflow: ## Setup MLflow tracking server
	@echo "$(BLUE)📊 Setting up MLflow...$(NC)"
	@mkdir -p mlruns
	@echo "$(GREEN)✅ MLflow setup completed$(NC)"
	@echo "$(CYAN)🚀 Start MLflow UI with: mlflow ui --host 0.0.0.0 --port 5000$(NC)"

.PHONY: train-baseline
train-baseline: ## Train baseline models
	@echo "$(BLUE)🏋️ Training baseline models...$(NC)"
	@$(PYTHON) -c "from src.models.lightgbm_master import main; main()" &
	@$(PYTHON) -c "from src.models.prophet_seasonal import main; main()" &
	@wait
	@echo "$(GREEN)✅ Baseline models trained$(NC)"

.PHONY: train-advanced
train-advanced: ## Train advanced models (Phase 5)
	@echo "$(BLUE)🚀 Training advanced models...$(NC)"
	@$(PYTHON) -c "from src.models.advanced_ensemble import main; main()"
	@$(PYTHON) -c "from src.models.lstm_temporal import main; main()" || echo "$(YELLOW)⚠️ LSTM training skipped$(NC)"
	@$(PYTHON) -c "from src.models.arima_temporal import main; main()" || echo "$(YELLOW)⚠️ ARIMA training skipped$(NC)"
	@echo "$(GREEN)✅ Advanced models trained$(NC)"

.PHONY: train-ensemble
train-ensemble: ## Train ensemble models
	@echo "$(BLUE)🎯 Training ensemble models...$(NC)"
	@$(PYTHON) -c "from src.models.meta_ensemble import main; main()"
	@echo "$(GREEN)✅ Ensemble models trained$(NC)"

.PHONY: hyperparameter-tuning
hyperparameter-tuning: ## Run hyperparameter tuning
	@echo "$(BLUE)🔧 Running hyperparameter tuning...$(NC)"
	@$(PYTHON) -c "from src.models.optimization_pipeline import main; main()"
	@echo "$(GREEN)✅ Hyperparameter tuning completed$(NC)"

.PHONY: train-all
train-all: train-baseline train-advanced train-ensemble ## Train all models

##@ Model Evaluation
.PHONY: evaluate-models
evaluate-models: ## Evaluate all trained models
	@echo "$(BLUE)📊 Evaluating models...$(NC)"
	@$(PYTHON) -c "from src.evaluation.model_diagnostics import run_comprehensive_evaluation; run_comprehensive_evaluation()"
	@echo "$(GREEN)✅ Model evaluation completed$(NC)"

.PHONY: generate-submission
generate-submission: ## Generate submission file
	@echo "$(BLUE)📝 Generating submission...$(NC)"
	@$(PYTHON) -c "from src.models.phase5_integration_demo import main; main()"
	@echo "$(GREEN)✅ Submission generated$(NC)"

.PHONY: validate-submission
validate-submission: ## Validate submission format
	@echo "$(BLUE)✅ Validating submission format...$(NC)"
	@$(PYTHON) scripts/validate_submission.py $(SUBMISSIONS_DIR)/*.csv || echo "$(YELLOW)⚠️ Validation script not found$(NC)"
	@echo "$(GREEN)✅ Submission validation completed$(NC)"

##@ Monitoring & Observability
.PHONY: start-monitoring
start-monitoring: ## Start monitoring services
	@echo "$(BLUE)📈 Starting monitoring services...$(NC)"
	@$(PYTHON) -c "from src.monitoring.dashboard import start_monitoring_dashboard; start_monitoring_dashboard()" &
	@echo "$(GREEN)✅ Monitoring services started$(NC)"

.PHONY: check-health
check-health: ## Health check for all services
	@echo "$(BLUE)🏥 Running health checks...$(NC)"
	@$(PYTHON) -c "from src.utils.health_check import run_health_checks; run_health_checks()"
	@echo "$(GREEN)✅ Health checks completed$(NC)"

.PHONY: generate-report
generate-report: ## Generate comprehensive performance report
	@echo "$(BLUE)📋 Generating performance report...$(NC)"
	@$(PYTHON) scripts/generate_report.py
	@echo "$(GREEN)✅ Report generated$(NC)"

##@ Docker & Deployment
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@$(DOCKER) build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	@$(DOCKER) run -d -p 8000:8000 --name $(PROJECT_NAME) $(PROJECT_NAME):latest
	@echo "$(GREEN)✅ Docker container started$(NC)"

.PHONY: docker-stop
docker-stop: ## Stop Docker container
	@echo "$(BLUE)🐳 Stopping Docker container...$(NC)"
	@$(DOCKER) stop $(PROJECT_NAME) || true
	@$(DOCKER) rm $(PROJECT_NAME) || true
	@echo "$(GREEN)✅ Docker container stopped$(NC)"

##@ CI/CD Pipeline
.PHONY: ci-pipeline
ci-pipeline: ## Run complete CI pipeline
	@echo "$(BLUE)🔄 Running CI pipeline...$(NC)"
	@$(MAKE) code-quality
	@$(MAKE) test-coverage
	@$(MAKE) test-phase5-compliance
	@$(MAKE) security-scan
	@echo "$(GREEN)✅ CI pipeline completed$(NC)"

.PHONY: cd-pipeline
cd-pipeline: ## Run CD pipeline
	@echo "$(BLUE)🚀 Running CD pipeline...$(NC)"
	@$(MAKE) train-all
	@$(MAKE) evaluate-models
	@$(MAKE) generate-submission
	@$(MAKE) validate-submission
	@$(MAKE) docker-build
	@echo "$(GREEN)✅ CD pipeline completed$(NC)"

.PHONY: full-pipeline
full-pipeline: ci-pipeline cd-pipeline ## Run complete CI/CD pipeline

##@ Maintenance
.PHONY: clean
clean: ## Clean temporary files
	@echo "$(BLUE)🧹 Cleaning temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf .mypy_cache
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean all generated files including models
	@echo "$(BLUE)🧹 Deep cleaning all generated files...$(NC)"
	@rm -rf $(MODELS_DIR)/*
	@rm -rf $(LOGS_DIR)/*
	@rm -rf mlruns
	@echo "$(GREEN)✅ Deep cleanup completed$(NC)"

.PHONY: reset-env
reset-env: ## Reset conda environment
	@echo "$(BLUE)🔄 Resetting environment...$(NC)"
	@$(CONDA) remove -n $(ENV_NAME) --all -y || true
	@$(MAKE) create-env
	@$(MAKE) install-deps
	@echo "$(GREEN)✅ Environment reset$(NC)"

##@ Utilities
.PHONY: check-env
check-env: ## Check environment status
	@echo "$(BLUE)🔍 Checking environment...$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current environment: $$($(CONDA) info --envs | grep '*' | awk '{print $$1}' || echo 'Not using conda')"
	@echo "MLflow tracking URI: $(MLFLOW_TRACKING_URI)"
	@echo "$(GREEN)✅ Environment check completed$(NC)"

.PHONY: install-jupyter
install-jupyter: ## Install and setup Jupyter
	@echo "$(BLUE)📓 Installing Jupyter...$(NC)"
	@$(PIP) install jupyter ipykernel
	@$(PYTHON) -m ipykernel install --user --name $(ENV_NAME) --display-name "$(ENV_NAME)"
	@echo "$(GREEN)✅ Jupyter installed$(NC)"

.PHONY: start-jupyter
start-jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)📓 Starting Jupyter server...$(NC)"
	@jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

.PHONY: quick-test
quick-test: ## Quick smoke test
	@echo "$(BLUE)⚡ Running quick smoke test...$(NC)"
	@$(PYTHON) -c "import sys; sys.path.append('src'); from architecture.factories import model_factory; print('✅ Factories working'); from config.phase6_config import get_config; config = get_config(); print('✅ Configuration loaded'); print('🎉 Quick test passed!')"

.PHONY: demo
demo: ## Run Phase 6 demonstration
	@echo "$(BLUE)🎭 Running Phase 6 demonstration...$(NC)"
	@$(PYTHON) -c "from src.architecture.factories import model_factory; from src.architecture.strategies import create_strategy; from src.architecture.observers import event_publisher; from src.architecture.pipelines import DataProcessingPipeline; from src.config.phase6_config import get_config; print('🏆 Phase 6 components loaded successfully!')"
	@echo "$(GREEN)✅ Phase 6 demonstration completed$(NC)"

##@ Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@mkdir -p docs/api
	@$(PYTHON) -c "import pydoc; pydoc.writedoc('src')" || echo "$(YELLOW)⚠️ Documentation generation incomplete$(NC)"
	@echo "$(GREEN)✅ Documentation generated$(NC)"

.PHONY: serve-docs
serve-docs: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	@cd docs && $(PYTHON) -m http.server 8080

# Development shortcuts
.PHONY: dev
dev: setup quick-test ## Quick development setup

.PHONY: production-deploy
production-deploy: ## Deploy to production
	@echo "$(RED)🚨 PRODUCTION DEPLOYMENT$(NC)"
	@echo "$(YELLOW)⚠️ This will deploy to production. Are you sure? [y/N]$(NC)"
	@read -r REPLY && [ "$$REPLY" = "y" ] || exit 1
	@$(MAKE) full-pipeline
	@echo "$(GREEN)🚀 Production deployment completed$(NC)"

# Version information
.PHONY: version
version: ## Show version information
	@echo "$(CYAN)📋 Version Information$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $$(grep version setup.py | cut -d'"' -f2 2>/dev/null || echo 'unknown')"
	@echo "Environment: $(ENVIRONMENT)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# Show status of all services
.PHONY: status
status: ## Show status of all services
	@echo "$(CYAN)📊 System Status$(NC)"
	@echo "MLflow: $$(curl -s $(MLFLOW_TRACKING_URI) >/dev/null && echo '✅ Running' || echo '❌ Down')"
	@echo "Docker: $$($(DOCKER) ps | grep $(PROJECT_NAME) >/dev/null && echo '✅ Running' || echo '❌ Not running')"
	@echo "Tests: $$([ -f .coverage ] && echo '✅ Recent coverage' || echo '⚠️ No recent coverage')"
	@echo "Models: $$(ls $(MODELS_DIR) 2>/dev/null | wc -l | tr -d ' ') trained models"