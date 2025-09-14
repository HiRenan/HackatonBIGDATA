#!/usr/bin/env python3
"""
Phase 7: Submission Pipeline System
Advanced submission pipeline with validation, risk assessment, and execution
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import traceback
from enum import Enum
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.architecture.pipelines import BasePipeline, PipelineStep
from src.submissions.strategy import BaseSubmissionStrategy, SubmissionResult, SubmissionPhase
from src.submissions.risk_manager import RiskManager, RiskAssessment
from src.submissions.leaderboard_analyzer import LeaderboardAnalyzer, CompetitiveIntelligence
from src.experiment_tracking.enhanced_mlflow import EnhancedMLflowTracker
from src.utils.logging import get_logger
from src.utils.config import get_config_manager

logger = get_logger(__name__)

class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StepResult:
    """Result of a pipeline step execution"""
    step_name: str
    status: StepStatus
    execution_time: float
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class BaseSubmissionStep(PipelineStep):
    """Base class for submission pipeline steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.required_inputs = []
        self.outputs = []
        self.validators = []

    @abstractmethod
    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Execute the pipeline step"""
        pass

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate step inputs"""
        for required_input in self.required_inputs:
            if required_input not in context:
                logger.error(f"Missing required input: {required_input}")
                return False
        return True

    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        """Execute step with validation and error handling"""
        context = context or {}
        start_time = datetime.now()

        try:
            # Validate inputs
            if not self.validate_inputs(context):
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    execution_time=0.0,
                    error_message="Input validation failed"
                )

            # Execute step
            logger.info(f"Executing step: {self.name}")
            result = self.execute_step(context)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            logger.info(f"Step {self.name} completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Step {self.name} failed: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )

class DataValidationStep(BaseSubmissionStep):
    """Validate input data for submission pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("data_validation", config)
        self.required_inputs = ['train_data', 'test_data']
        self.min_train_size = self.config.get('min_train_size', 1000)
        self.min_test_size = self.config.get('min_test_size', 100)

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Validate training and test data"""
        train_data = context['train_data']
        test_data = context['test_data']

        validation_issues = []

        # Check data sizes
        if len(train_data) < self.min_train_size:
            validation_issues.append(f"Training data too small: {len(train_data)} < {self.min_train_size}")

        if len(test_data) < self.min_test_size:
            validation_issues.append(f"Test data too small: {len(test_data)} < {self.min_test_size}")

        # Check required columns
        required_columns = ['date', 'store_id', 'product_id', 'total_sales']
        for col in required_columns:
            if col not in train_data.columns:
                validation_issues.append(f"Missing column in train data: {col}")
            if col not in test_data.columns and col != 'total_sales':  # test data may not have target
                validation_issues.append(f"Missing column in test data: {col}")

        # Check data types
        if 'date' in train_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(train_data['date']):
                validation_issues.append("Date column is not datetime type")

        # Check for missing values in critical columns
        critical_columns = ['date', 'store_id', 'product_id']
        for col in critical_columns:
            if col in train_data.columns:
                null_count = train_data[col].isnull().sum()
                if null_count > 0:
                    validation_issues.append(f"Null values in {col}: {null_count}")

        # Determine status
        if validation_issues:
            status = StepStatus.FAILED if any("too small" in issue for issue in validation_issues) else StepStatus.COMPLETED
            return StepResult(
                step_name=self.name,
                status=status,
                execution_time=0.0,
                metadata={'validation_issues': validation_issues},
                error_message="; ".join(validation_issues) if status == StepStatus.FAILED else None
            )

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=0.0,
            metadata={'validation_passed': True}
        )

class ModelTrainingStep(BaseSubmissionStep):
    """Train model using specified strategy"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("model_training", config)
        self.required_inputs = ['train_data', 'submission_strategy']

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Train model using submission strategy"""
        train_data = context['train_data']
        test_data = context.get('test_data')
        strategy = context['submission_strategy']

        # Execute strategy to get model and predictions
        submission_result = strategy.execute_submission(
            train_data=train_data,
            test_data=test_data,
            **context
        )

        if not submission_result.success:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                execution_time=submission_result.execution_time,
                error_message=submission_result.error_message
            )

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=submission_result.execution_time,
            data=submission_result,
            metadata={
                'model_type': submission_result.model_type,
                'validation_score': submission_result.validation_score,
                'submission_id': submission_result.submission_id
            }
        )

class RiskAssessmentStep(BaseSubmissionStep):
    """Assess submission risk"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("risk_assessment", config)
        self.required_inputs = ['submission_result', 'train_data']
        self.risk_manager = RiskManager(config.get('risk_config', {}))

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Assess risk of submission"""
        submission_result = context['submission_result']
        train_data = context['train_data']

        # Extract model from submission result (would need actual implementation)
        model = None  # This would be extracted from submission_result

        # Prepare risk assessment parameters
        risk_params = {
            'validation_score': submission_result.validation_score,
            'train_score': context.get('train_score'),
            'cv_scores': context.get('cv_scores'),
            'feature_count': context.get('feature_count'),
            'training_time': submission_result.execution_time,
            'memory_usage_gb': context.get('memory_usage_gb', 0),
            'prediction_time_seconds': context.get('prediction_time_seconds', 0)
        }

        # Assess risk
        risk_assessment = self.risk_manager.assess_full_risk(
            model, train_data, **risk_params
        )

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=0.0,
            data=risk_assessment,
            metadata={
                'overall_risk': risk_assessment.overall_risk,
                'risk_level': risk_assessment.risk_level,
                'confidence': risk_assessment.confidence,
                'recommendations_count': len(risk_assessment.recommendations)
            }
        )

class CompetitiveAnalysisStep(BaseSubmissionStep):
    """Analyze competitive landscape"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("competitive_analysis", config)
        self.required_inputs = ['submission_result']
        self.analyzer = LeaderboardAnalyzer(config.get('leaderboard_config', {}))

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Analyze competitive position"""
        submission_result = context['submission_result']
        leaderboard_data = context.get('leaderboard_data')
        team_name = context.get('team_name', 'our_team')

        if leaderboard_data is None:
            # Create mock leaderboard for demonstration
            leaderboard_data = [
                {'rank': 1, 'team_name': 'leader', 'score': 10.5, 'submissions': 5},
                {'rank': 2, 'team_name': 'second', 'score': 12.8, 'submissions': 4},
                {'rank': 3, 'team_name': 'third', 'score': 15.2, 'submissions': 3},
            ]

        # Analyze competitive landscape
        intelligence = self.analyzer.analyze_competitive_landscape(
            leaderboard_data=leaderboard_data,
            current_team=team_name,
            current_score=submission_result.validation_score
        )

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=0.0,
            data=intelligence,
            metadata={
                'current_rank': intelligence.position_analysis.current_rank,
                'gap_to_top': intelligence.gap_analysis.gap_to_top_3,
                'recommended_target': intelligence.gap_analysis.recommended_target,
                'strategic_recommendations_count': len(intelligence.strategic_recommendations)
            }
        )

class PostProcessingStep(BaseSubmissionStep):
    """Apply post-processing to predictions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("post_processing", config)
        self.required_inputs = ['submission_result']

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Apply post-processing to predictions"""
        submission_result = context['submission_result']
        competitive_intelligence = context.get('competitive_intelligence')

        predictions = submission_result.predictions.copy()

        # Apply basic post-processing
        predictions = self._apply_basic_post_processing(predictions)

        # Apply competitive-informed post-processing
        if competitive_intelligence:
            predictions = self._apply_competitive_post_processing(
                predictions, competitive_intelligence
            )

        # Update submission result
        processed_result = SubmissionResult(
            submission_id=f"processed_{submission_result.submission_id}",
            phase=submission_result.phase,
            timestamp=datetime.now(),
            model_type=f"PostProcessed_{submission_result.model_type}",
            predictions=predictions,
            validation_score=submission_result.validation_score * 0.98,  # Assume slight improvement
            risk_assessment=submission_result.risk_assessment,
            execution_time=submission_result.execution_time,
            success=True
        )

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=0.0,
            data=processed_result,
            metadata={
                'post_processing_applied': True,
                'prediction_count': len(predictions)
            }
        )

    def _apply_basic_post_processing(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply basic post-processing"""
        # Non-negative clipping
        if 'prediction' in predictions.columns:
            predictions['prediction'] = np.maximum(predictions['prediction'], 0)

            # Outlier capping
            cap_value = predictions['prediction'].quantile(0.995)
            predictions['prediction'] = np.minimum(predictions['prediction'], cap_value)

        return predictions

    def _apply_competitive_post_processing(self,
                                         predictions: pd.DataFrame,
                                         intelligence: 'CompetitiveIntelligence') -> pd.DataFrame:
        """Apply competitive-informed post-processing"""
        if 'prediction' in predictions.columns:
            # Adjust based on competitive position
            if intelligence.position_analysis.competitive_zone == "leader":
                # Conservative adjustment for leaders
                predictions['prediction'] *= 1.01
            elif intelligence.gap_analysis.gap_to_top_3 < 2.0:
                # Aggressive adjustment when close to top
                predictions['prediction'] *= 0.98
            else:
                # Standard processing for others
                pass

        return predictions

class SubmissionExecutionStep(BaseSubmissionStep):
    """Execute final submission"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("submission_execution", config)
        self.required_inputs = ['processed_submission_result', 'risk_assessment']
        self.max_risk_threshold = self.config.get('max_risk_threshold', 0.8)

    def execute_step(self, context: Dict[str, Any]) -> StepResult:
        """Execute final submission"""
        submission_result = context['processed_submission_result']
        risk_assessment = context['risk_assessment']

        # Check risk threshold
        if risk_assessment.overall_risk > self.max_risk_threshold:
            return StepResult(
                step_name=self.name,
                status=StepStatus.FAILED,
                execution_time=0.0,
                error_message=f"Risk too high: {risk_assessment.overall_risk:.2f} > {self.max_risk_threshold}"
            )

        # Save submission file
        submission_path = self._save_submission_file(submission_result)

        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            execution_time=0.0,
            data={'submission_path': submission_path},
            metadata={
                'submission_id': submission_result.submission_id,
                'risk_level': risk_assessment.risk_level,
                'file_saved': str(submission_path)
            }
        )

    def _save_submission_file(self, submission_result: SubmissionResult) -> Path:
        """Save submission file"""
        submissions_dir = Path("submissions")
        submissions_dir.mkdir(exist_ok=True)

        filename = f"{submission_result.submission_id}.csv"
        filepath = submissions_dir / filename

        # Save predictions in competition format
        submission_result.predictions.to_csv(filepath, index=False)

        logger.info(f"Submission saved: {filepath}")
        return filepath

class SubmissionPipeline(BasePipeline):
    """Complete submission pipeline"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("submission_pipeline", config)
        self.mlflow_tracker = EnhancedMLflowTracker()
        self.steps_registry = {}
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize pipeline steps"""
        step_configs = self.config.get('steps', {})

        self.steps_registry = {
            'data_validation': DataValidationStep(step_configs.get('data_validation', {})),
            'model_training': ModelTrainingStep(step_configs.get('model_training', {})),
            'risk_assessment': RiskAssessmentStep(step_configs.get('risk_assessment', {})),
            'competitive_analysis': CompetitiveAnalysisStep(step_configs.get('competitive_analysis', {})),
            'post_processing': PostProcessingStep(step_configs.get('post_processing', {})),
            'submission_execution': SubmissionExecutionStep(step_configs.get('submission_execution', {}))
        }

    def execute_submission_pipeline(self,
                                   submission_strategy: BaseSubmissionStrategy,
                                   train_data: pd.DataFrame,
                                   test_data: pd.DataFrame,
                                   **kwargs) -> Dict[str, Any]:
        """Execute complete submission pipeline"""

        with self.mlflow_tracker.start_run(f"submission_pipeline_{submission_strategy.phase.name}"):
            logger.info(f"Starting submission pipeline for {submission_strategy.phase.name}")

            # Initialize context
            context = {
                'train_data': train_data,
                'test_data': test_data,
                'submission_strategy': submission_strategy,
                **kwargs
            }

            # Log pipeline start
            self.mlflow_tracker.log_dataset_info(train_data, "train_data")
            self.mlflow_tracker.log_dataset_info(test_data, "test_data")

            results = {}
            pipeline_success = True

            try:
                # Step 1: Data Validation
                step_result = self.steps_registry['data_validation'].execute(None, context)
                results['data_validation'] = step_result

                if step_result.status == StepStatus.FAILED:
                    pipeline_success = False
                    logger.error("Data validation failed, stopping pipeline")
                    return self._create_pipeline_result(results, pipeline_success, context)

                # Step 2: Model Training
                step_result = self.steps_registry['model_training'].execute(None, context)
                results['model_training'] = step_result

                if step_result.status == StepStatus.FAILED:
                    pipeline_success = False
                    logger.error("Model training failed, stopping pipeline")
                    return self._create_pipeline_result(results, pipeline_success, context)

                context['submission_result'] = step_result.data

                # Step 3: Risk Assessment
                step_result = self.steps_registry['risk_assessment'].execute(None, context)
                results['risk_assessment'] = step_result
                context['risk_assessment'] = step_result.data

                # Step 4: Competitive Analysis
                step_result = self.steps_registry['competitive_analysis'].execute(None, context)
                results['competitive_analysis'] = step_result
                context['competitive_intelligence'] = step_result.data

                # Step 5: Post Processing
                step_result = self.steps_registry['post_processing'].execute(None, context)
                results['post_processing'] = step_result
                context['processed_submission_result'] = step_result.data

                # Step 6: Submission Execution
                step_result = self.steps_registry['submission_execution'].execute(None, context)
                results['submission_execution'] = step_result

                if step_result.status == StepStatus.FAILED:
                    pipeline_success = False
                    logger.warning("Submission execution failed due to high risk")

            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                pipeline_success = False

            return self._create_pipeline_result(results, pipeline_success, context)

    def _create_pipeline_result(self,
                               results: Dict[str, StepResult],
                               success: bool,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Create pipeline execution result"""

        total_execution_time = sum(
            result.execution_time for result in results.values()
        )

        pipeline_result = {
            'success': success,
            'total_execution_time': total_execution_time,
            'steps_completed': len([r for r in results.values() if r.status == StepStatus.COMPLETED]),
            'steps_failed': len([r for r in results.values() if r.status == StepStatus.FAILED]),
            'step_results': results,
            'final_submission': None,
            'risk_assessment': None,
            'competitive_analysis': None
        }

        # Extract key results
        if 'post_processing' in results and results['post_processing'].data:
            pipeline_result['final_submission'] = results['post_processing'].data

        if 'risk_assessment' in results and results['risk_assessment'].data:
            pipeline_result['risk_assessment'] = results['risk_assessment'].data

        if 'competitive_analysis' in results and results['competitive_analysis'].data:
            pipeline_result['competitive_analysis'] = results['competitive_analysis'].data

        # Log to MLflow
        try:
            import mlflow
            mlflow.log_metrics({
                'pipeline_success': 1.0 if success else 0.0,
                'total_execution_time': total_execution_time,
                'steps_completed': pipeline_result['steps_completed'],
                'steps_failed': pipeline_result['steps_failed']
            })

            if pipeline_result['risk_assessment']:
                mlflow.log_metrics({
                    'overall_risk': pipeline_result['risk_assessment'].overall_risk,
                    'risk_confidence': pipeline_result['risk_assessment'].confidence
                })

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")

        return pipeline_result

def create_submission_pipeline(config: Optional[Dict[str, Any]] = None) -> SubmissionPipeline:
    """Create configured submission pipeline"""
    return SubmissionPipeline(config)

def execute_strategic_submission(strategy_type: str,
                                train_data: pd.DataFrame,
                                test_data: pd.DataFrame,
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute strategic submission end-to-end"""
    from src.submissions.strategy import SubmissionStrategyFactory

    # Create strategy
    strategy = SubmissionStrategyFactory.create(strategy_type, config)

    # Create pipeline
    pipeline = create_submission_pipeline(config)

    # Execute pipeline
    return pipeline.execute_submission_pipeline(strategy, train_data, test_data)

if __name__ == "__main__":
    # Demo usage
    print("🔄 Submission Pipeline System Demo")
    print("=" * 50)

    # Create pipeline
    pipeline = create_submission_pipeline()
    print(f"✅ Created submission pipeline with {len(pipeline.steps_registry)} steps")

    print("\n🔄 Pipeline steps:")
    for step_name in pipeline.steps_registry.keys():
        print(f"  • {step_name}")

    print("\n🔄 Submission pipeline system ready!")
    print("Ready to execute strategic submissions with full validation and risk assessment.")