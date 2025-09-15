#!/usr/bin/env python3
"""
Phase 6: Pipeline Pattern Implementation
Robust data processing pipelines with error handling and rollback
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime
import pickle
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.execution_time: Optional[float] = None
        self.status = 'not_executed'
        self.error_message: Optional[str] = None

    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any]) -> Any:
        """Execute the pipeline step"""
        pass

    @abstractmethod
    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Rollback the pipeline step (if possible)"""
        pass

    def validate_input(self, data: Any, context: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True

    def validate_output(self, data: Any, context: Dict[str, Any]) -> bool:
        """Validate output data"""
        return True

    def get_step_info(self) -> Dict[str, Any]:
        """Get step information"""
        return {
            'name': self.name,
            'status': self.status,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'config': self.config
        }

class Pipeline(ABC):
    """Abstract base pipeline class"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.steps: List[PipelineStep] = []
        self.execution_context: Dict[str, Any] = {}
        self.checkpoints: List[Any] = []
        self.enable_rollback = self.config.get('enable_rollback', True)
        self.enable_checkpoints = self.config.get('enable_checkpoints', True)

    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute the entire pipeline"""
        start_time = datetime.now()
        logger.info(f"🚀 Starting pipeline: {self.name}")

        try:
            # Initialize context
            self.execution_context = {
                'pipeline_name': self.name,
                'start_time': start_time,
                'input_data_shape': getattr(input_data, 'shape', None),
                'step_results': []
            }

            current_data = input_data
            executed_steps = []

            for i, step in enumerate(self.steps):
                try:
                    logger.info(f"📋 Executing step {i+1}/{len(self.steps)}: {step.name}")

                    # Validate input
                    if not step.validate_input(current_data, self.execution_context):
                        raise ValueError(f"Input validation failed for step: {step.name}")

                    # Create checkpoint if enabled
                    if self.enable_checkpoints:
                        checkpoint = self._create_checkpoint(current_data, step.name)
                        self.checkpoints.append(checkpoint)

                    # Execute step
                    step_start = datetime.now()
                    result_data = step.execute(current_data, self.execution_context)
                    step_end = datetime.now()

                    # Calculate execution time
                    execution_time = (step_end - step_start).total_seconds()
                    step.execution_time = execution_time
                    step.status = 'completed'

                    # Validate output
                    if not step.validate_output(result_data, self.execution_context):
                        raise ValueError(f"Output validation failed for step: {step.name}")

                    # Update context
                    self.execution_context['step_results'].append({
                        'step_name': step.name,
                        'execution_time': execution_time,
                        'output_shape': getattr(result_data, 'shape', None),
                        'status': 'success'
                    })

                    current_data = result_data
                    executed_steps.append(step)

                    logger.info(f"✅ Step completed: {step.name} ({execution_time:.2f}s)")

                except Exception as e:
                    step.status = 'failed'
                    step.error_message = str(e)
                    logger.error(f"❌ Step failed: {step.name} - {e}")

                    if self.enable_rollback:
                        self._rollback_pipeline(executed_steps, input_data)

                    raise RuntimeError(f"Pipeline failed at step '{step.name}': {e}") from e

            # Pipeline completed successfully
            end_time = datetime.now()
            total_execution_time = (end_time - start_time).total_seconds()

            result = {
                'status': 'success',
                'output_data': current_data,
                'execution_time': total_execution_time,
                'steps_executed': len(self.steps),
                'pipeline_name': self.name,
                'context': self.execution_context
            }

            logger.info(f"🎉 Pipeline completed: {self.name} ({total_execution_time:.2f}s)")
            return result

        except Exception as e:
            end_time = datetime.now()
            total_execution_time = (end_time - start_time).total_seconds()

            result = {
                'status': 'failed',
                'error': str(e),
                'execution_time': total_execution_time,
                'steps_executed': sum(1 for step in self.steps if step.status == 'completed'),
                'pipeline_name': self.name,
                'context': self.execution_context
            }

            logger.error(f"💥 Pipeline failed: {self.name} ({total_execution_time:.2f}s)")
            return result

    def _create_checkpoint(self, data: Any, step_name: str) -> Dict[str, Any]:
        """Create a checkpoint of current data"""
        if not self.enable_checkpoints:
            return {}

        try:
            # Create temporary file for checkpoint
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')

            with open(temp_file.name, 'wb') as f:
                pickle.dump(data, f)

            checkpoint = {
                'step_name': step_name,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_file': temp_file.name,
                'data_shape': getattr(data, 'shape', None)
            }

            logger.debug(f"📁 Checkpoint created for {step_name}: {temp_file.name}")
            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to create checkpoint for {step_name}: {e}")
            return {}

    def _rollback_pipeline(self, executed_steps: List[PipelineStep], original_data: Any) -> None:
        """Rollback executed steps"""
        if not self.enable_rollback:
            return

        logger.info("🔄 Starting pipeline rollback...")

        for step in reversed(executed_steps):
            try:
                step.rollback(original_data, self.execution_context)
                logger.info(f"↩️ Rolled back step: {step.name}")
            except Exception as e:
                logger.warning(f"Failed to rollback step {step.name}: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            'name': self.name,
            'total_steps': len(self.steps),
            'step_info': [step.get_step_info() for step in self.steps],
            'config': self.config,
            'context': self.execution_context
        }

# Specific Pipeline Step Implementations

class DataValidationStep(PipelineStep):
    """Step for data validation"""

    def __init__(self, name: str = "data_validation", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.required_columns = self.config.get('required_columns', [])
        self.min_rows = self.config.get('min_rows', 1)
        self.max_nulls_pct = self.config.get('max_nulls_pct', 0.5)

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Execute data validation"""
        logger.info(f"🔍 Validating data: shape {data.shape}")

        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check minimum rows
        if len(data) < self.min_rows:
            raise ValueError(f"Insufficient data: {len(data)} rows (minimum: {self.min_rows})")

        # Check null percentage
        for col in data.columns:
            null_pct = data[col].isnull().sum() / len(data)
            if null_pct > self.max_nulls_pct:
                raise ValueError(f"Column '{col}' has {null_pct:.2%} nulls (max: {self.max_nulls_pct:.2%})")

        logger.info("✅ Data validation passed")
        return data

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Nothing to rollback for validation"""
        return data

class DataCleaningStep(PipelineStep):
    """Step for data cleaning"""

    def __init__(self, name: str = "data_cleaning", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.fill_method = self.config.get('fill_method', 'mean')
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Execute data cleaning"""
        logger.info(f"🧹 Cleaning data: {data.shape}")

        cleaned_data = data.copy()

        # Handle missing values
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns

        if self.fill_method == 'mean':
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].mean())
        elif self.fill_method == 'median':
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].median())
        elif self.fill_method == 'forward':
            cleaned_data = cleaned_data.fillna(method='ffill')

        # Handle outliers
        if self.outlier_method == 'iqr':
            for col in numeric_cols:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                n_outliers = outliers_mask.sum()

                if n_outliers > 0:
                    cleaned_data.loc[outliers_mask, col] = cleaned_data[col].median()
                    logger.info(f"Removed {n_outliers} outliers from {col}")

        logger.info(f"✅ Data cleaned: {cleaned_data.shape}")
        return cleaned_data

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Return original data"""
        return data

class FeatureEngineeringStep(PipelineStep):
    """Step for feature engineering"""

    def __init__(self, name: str = "feature_engineering", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.feature_types = self.config.get('feature_types', ['temporal'])

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Execute feature engineering"""
        logger.info(f"🔧 Engineering features: {self.feature_types}")

        engineered_data = data.copy()

        for feature_type in self.feature_types:
            try:
                if feature_type == 'temporal':
                    engineered_data = self._add_temporal_features(engineered_data)
                elif feature_type == 'aggregation':
                    engineered_data = self._add_aggregation_features(engineered_data)
                elif feature_type == 'business':
                    engineered_data = self._add_business_features(engineered_data)

                logger.info(f"Added {feature_type} features")
            except Exception as e:
                logger.warning(f"Failed to add {feature_type} features: {e}")

        logger.info(f"✅ Feature engineering completed: {engineered_data.shape}")
        return engineered_data

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        if 'date' not in data.columns:
            return data

        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['weekday'] = data['date'].dt.weekday
        data['quarter'] = data['date'].dt.quarter

        return data

    def _add_aggregation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add aggregation features"""
        # Placeholder for aggregation features
        return data

    def _add_business_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add business-specific features"""
        # Placeholder for business features
        return data

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Return data without engineered features"""
        return data

# Concrete Pipeline Implementations

class DataProcessingPipeline(Pipeline):
    """Complete data processing pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("data_processing", config)
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize pipeline steps"""
        # Data validation
        validation_config = self.config.get('validation', {})
        self.add_step(DataValidationStep(config=validation_config))

        # Data cleaning
        cleaning_config = self.config.get('cleaning', {})
        self.add_step(DataCleaningStep(config=cleaning_config))

        # Feature engineering
        feature_config = self.config.get('features', {})
        self.add_step(FeatureEngineeringStep(config=feature_config))

class FeaturePipeline(Pipeline):
    """Specialized feature engineering pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("feature_engineering", config)
        self._initialize_feature_steps()

    def _initialize_feature_steps(self):
        """Initialize feature engineering steps"""
        from ..architecture.factories import feature_factory

        feature_types = self.config.get('feature_types', ['temporal', 'aggregation'])

        for feature_type in feature_types:
            try:
                step_config = self.config.get(f'{feature_type}_config', {})
                feature_step = FeatureGenerationStep(feature_type, step_config)
                self.add_step(feature_step)
            except Exception as e:
                logger.warning(f"Could not add {feature_type} features: {e}")

class FeatureGenerationStep(PipelineStep):
    """Step for generating specific feature types"""

    def __init__(self, feature_type: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(f"generate_{feature_type}_features", config)
        self.feature_type = feature_type

    def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate features"""
        try:
            from ..architecture.factories import feature_factory

            feature_engine = feature_factory.create(self.feature_type, self.config)
            enhanced_data = feature_engine.engineer_features(data)

            logger.info(f"Generated {self.feature_type} features: {enhanced_data.shape}")
            return enhanced_data

        except Exception as e:
            logger.warning(f"Feature generation failed for {self.feature_type}: {e}")
            return data

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Return original data"""
        return data

class ModelTrainingPipeline(Pipeline):
    """Model training pipeline with validation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("model_training", config)
        self._initialize_training_steps()

    def _initialize_training_steps(self):
        """Initialize training steps"""
        # Add training steps
        self.add_step(ModelInitializationStep(config=self.config.get('model', {})))
        self.add_step(ModelTrainingStep(config=self.config.get('training', {})))
        self.add_step(ModelValidationStep(config=self.config.get('validation', {})))

class ModelInitializationStep(PipelineStep):
    """Step for model initialization"""

    def execute(self, data: Tuple[pd.DataFrame, pd.Series], context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Any]:
        """Initialize model"""
        X, y = data

        # Initialize model (placeholder)
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**self.config)

        logger.info(f"✅ Model initialized: {model.__class__.__name__}")
        return X, y, model

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Nothing to rollback"""
        return data

class ModelTrainingStep(PipelineStep):
    """Step for model training"""

    def execute(self, data: Tuple[pd.DataFrame, pd.Series, Any], context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Any]:
        """Train model"""
        X, y, model = data

        logger.info(f"🏋️ Training model on {X.shape[0]} samples")
        model.fit(X, y)

        logger.info("✅ Model training completed")
        return X, y, model

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Reset model"""
        return data

class ModelValidationStep(PipelineStep):
    """Step for model validation"""

    def execute(self, data: Tuple[pd.DataFrame, pd.Series, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model"""
        X, y, model = data

        # Simple validation (placeholder)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')

        validation_result = {
            'model': model,
            'scores': -scores,
            'mean_score': -scores.mean(),
            'std_score': scores.std()
        }

        logger.info(f"✅ Validation completed: {validation_result['mean_score']:.4f} ± {validation_result['std_score']:.4f}")
        return validation_result

    def rollback(self, data: Any, context: Dict[str, Any]) -> Any:
        """Nothing to rollback"""
        return data

if __name__ == "__main__":
    # Demo usage
    print("🔄 Phase 6 Pipeline Pattern Demo")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'value': np.random.randn(1000).cumsum() + 100,
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })

    # Demo data processing pipeline
    print("\n📊 Data Processing Pipeline Demo:")

    data_config = {
        'validation': {
            'required_columns': ['date', 'value'],
            'min_rows': 100
        },
        'cleaning': {
            'fill_method': 'mean',
            'outlier_method': 'iqr'
        },
        'features': {
            'feature_types': ['temporal']
        }
    }

    data_pipeline = DataProcessingPipeline(data_config)
    result = data_pipeline.execute(sample_data)

    print(f"Pipeline status: {result['status']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"Output shape: {result['output_data'].shape}")

    # Pipeline info
    print("\n📋 Pipeline Information:")
    pipeline_info = data_pipeline.get_pipeline_info()
    for step_info in pipeline_info['step_info']:
        print(f"  {step_info['name']}: {step_info['status']} ({step_info.get('execution_time', 0):.3f}s)")

    print("\n✅ Pipeline demo completed!")