#!/usr/bin/env python3
"""
Phase 7: Submission Strategy System
Advanced submission management with competitive intelligence and risk assessment
"""

from .strategy import (
    BaseSubmissionStrategy,
    BaselineSubmissionStrategy,
    SingleModelSubmissionStrategy,
    EnsembleSubmissionStrategy,
    OptimizedEnsembleSubmissionStrategy,
    FinalSubmissionStrategy,
    SubmissionStrategyFactory
)

from .risk_manager import (
    RiskAssessment,
    RiskManager,
    OverfittingRiskAssessor,
    ComplexityRiskAssessor,
    LeakageRiskAssessor,
    ExecutionRiskAssessor
)

from .leaderboard_analyzer import (
    LeaderboardAnalyzer,
    CompetitiveIntelligence,
    PositionAnalysis,
    GapAnalysis
)

from .submission_pipeline import (
    SubmissionPipeline,
    SubmissionStep,
    ValidationStep,
    PostProcessingStep,
    RiskAssessmentStep,
    SubmissionExecutionStep
)

from .timeline_manager import (
    TimelineManager,
    SubmissionSchedule,
    SubmissionWindow,
    DeadlineTracker
)

from .post_processor import (
    BasePostProcessor,
    BusinessRulesPostProcessor,
    OutlierCappingPostProcessor,
    SeasonalAdjustmentPostProcessor,
    EnsembleBlendingPostProcessor,
    PostProcessorPipeline
)

__version__ = "1.0.0"
__author__ = "Hackathon Forecast 2025 Team"

__all__ = [
    # Strategy classes
    "BaseSubmissionStrategy",
    "BaselineSubmissionStrategy",
    "SingleModelSubmissionStrategy",
    "EnsembleSubmissionStrategy",
    "OptimizedEnsembleSubmissionStrategy",
    "FinalSubmissionStrategy",
    "SubmissionStrategyFactory",

    # Risk management
    "RiskAssessment",
    "RiskManager",
    "OverfittingRiskAssessor",
    "ComplexityRiskAssessor",
    "LeakageRiskAssessor",
    "ExecutionRiskAssessor",

    # Competitive analysis
    "LeaderboardAnalyzer",
    "CompetitiveIntelligence",
    "PositionAnalysis",
    "GapAnalysis",

    # Pipeline components
    "SubmissionPipeline",
    "SubmissionStep",
    "ValidationStep",
    "PostProcessingStep",
    "RiskAssessmentStep",
    "SubmissionExecutionStep",

    # Timeline management
    "TimelineManager",
    "SubmissionSchedule",
    "SubmissionWindow",
    "DeadlineTracker",

    # Post-processing
    "BasePostProcessor",
    "BusinessRulesPostProcessor",
    "OutlierCappingPostProcessor",
    "SeasonalAdjustmentPostProcessor",
    "EnsembleBlendingPostProcessor",
    "PostProcessorPipeline"
]

# Module metadata
SUBMISSION_PHASES = {
    1: "Baseline Test - Day 3",
    2: "Single Model - Day 7",
    3: "Initial Ensemble - Day 10",
    4: "Optimized Ensemble - Day 13",
    5: "Final Submission - Last Day"
}

RISK_LEVELS = {
    "LOW": "Safe to submit",
    "MEDIUM": "Additional validation recommended",
    "HIGH": "Consider simpler approach",
    "CRITICAL": "Do not submit"
}

SUBMISSION_STRATEGIES = {
    "conservative": "Prioritize low risk over performance",
    "balanced": "Balance risk and performance",
    "aggressive": "Maximize performance, accept higher risk"
}

def get_submission_info():
    """Get information about the submission system"""
    return {
        "version": __version__,
        "author": __author__,
        "phases": len(SUBMISSION_PHASES),
        "risk_levels": list(RISK_LEVELS.keys()),
        "strategies": list(SUBMISSION_STRATEGIES.keys())
    }

def create_submission_strategy(strategy_type: str, **kwargs):
    """Convenience function to create submission strategy"""
    return SubmissionStrategyFactory.create(strategy_type, **kwargs)

def assess_submission_risk(model, validation_data, **kwargs):
    """Convenience function for risk assessment"""
    risk_manager = RiskManager()
    return risk_manager.assess_full_risk(model, validation_data, **kwargs)

def analyze_leaderboard(leaderboard_data, current_position=None):
    """Convenience function for leaderboard analysis"""
    analyzer = LeaderboardAnalyzer()
    return analyzer.analyze_competitive_landscape(leaderboard_data, current_position)

# Version information
def version():
    """Return version information"""
    return __version__

def phase7_info():
    """Return Phase 7 system information"""
    return {
        "name": "Submission Strategy System",
        "phase": "7",
        "version": __version__,
        "components": [
            "Submission Strategies",
            "Risk Management",
            "Competitive Intelligence",
            "Timeline Management",
            "Post-processing Pipeline",
            "Automated Pipeline"
        ],
        "integration": [
            "Phase 6 Architecture Patterns",
            "MLflow Tracking",
            "Configuration Management",
            "Enterprise Logging"
        ]
    }