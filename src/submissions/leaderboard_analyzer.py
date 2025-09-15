#!/usr/bin/env python3
"""
Phase 7: Leaderboard Analysis & Competitive Intelligence
Advanced competitive analysis with strategic positioning and gap analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from enum import Enum
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import get_logger

logger = get_logger(__name__)

class CompetitiveStrategy(Enum):
    """Competitive strategy types"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CATCH_UP = "catch_up"
    MAINTAIN_LEAD = "maintain_lead"

@dataclass
class LeaderboardEntry:
    """Single leaderboard entry"""
    rank: int
    team_name: str
    score: float
    submissions: int
    last_submission: Optional[datetime] = None
    trend: Optional[str] = None  # "improving", "declining", "stable"

@dataclass
class PositionAnalysis:
    """Analysis of current competitive position"""
    current_rank: int
    current_score: float
    total_teams: int
    percentile: float
    gap_to_top: float
    gap_to_next: float
    points_to_advance: float
    competitive_zone: str  # "leader", "contender", "middle_pack", "bottom"

@dataclass
class GapAnalysis:
    """Detailed gap analysis for strategic planning"""
    gap_to_top_3: float
    gap_to_top_10: float
    gap_to_top_25: float
    improvement_needed: Dict[str, float]
    achievability_score: float
    recommended_target: int

@dataclass
class CompetitiveIntelligence:
    """Comprehensive competitive intelligence"""
    position_analysis: PositionAnalysis
    gap_analysis: GapAnalysis
    trend_analysis: Dict[str, Any]
    strategic_recommendations: List[str]
    risk_assessment: Dict[str, float]
    timestamp: datetime

class LeaderboardAnalyzer:
    """Advanced leaderboard analysis system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.history = []
        self.analysis_cache = {}

        # Configuration
        self.top_tier_size = self.config.get('top_tier_size', 3)
        self.contender_tier_size = self.config.get('contender_tier_size', 10)
        self.improvement_buffer = self.config.get('improvement_buffer', 0.05)  # 5% buffer

    def parse_leaderboard(self, leaderboard_data: Union[pd.DataFrame, List[Dict], str]) -> List[LeaderboardEntry]:
        """Parse leaderboard data into structured format"""
        if isinstance(leaderboard_data, str):
            # Assume it's a file path or JSON string
            try:
                if Path(leaderboard_data).exists():
                    leaderboard_data = pd.read_csv(leaderboard_data)
                else:
                    leaderboard_data = json.loads(leaderboard_data)
            except Exception as e:
                logger.error(f"Error parsing leaderboard data: {str(e)}")
                return []

        if isinstance(leaderboard_data, list):
            leaderboard_data = pd.DataFrame(leaderboard_data)

        entries = []
        for idx, row in leaderboard_data.iterrows():
            entry = LeaderboardEntry(
                rank=row.get('rank', idx + 1),
                team_name=row.get('team_name', f'Team_{idx}'),
                score=float(row.get('score', 0.0)),
                submissions=int(row.get('submissions', 1)),
                last_submission=pd.to_datetime(row.get('last_submission', None), errors='coerce'),
                trend=row.get('trend', 'stable')
            )
            entries.append(entry)

        # Sort by rank
        entries.sort(key=lambda x: x.rank)
        return entries

    def analyze_position(self,
                        leaderboard: List[LeaderboardEntry],
                        current_team: str,
                        current_score: Optional[float] = None) -> PositionAnalysis:
        """Analyze current competitive position"""

        # Find current position
        current_entry = None
        current_rank = None

        for entry in leaderboard:
            if entry.team_name == current_team:
                current_entry = entry
                current_rank = entry.rank
                break

        # If not found and score provided, estimate position
        if current_entry is None and current_score is not None:
            estimated_rank = self._estimate_rank(leaderboard, current_score)
            current_entry = LeaderboardEntry(
                rank=estimated_rank,
                team_name=current_team,
                score=current_score,
                submissions=0
            )
            current_rank = estimated_rank

        if current_entry is None:
            logger.warning(f"Team {current_team} not found in leaderboard")
            return PositionAnalysis(
                current_rank=999,
                current_score=float('inf'),
                total_teams=len(leaderboard),
                percentile=0.0,
                gap_to_top=float('inf'),
                gap_to_next=float('inf'),
                points_to_advance=float('inf'),
                competitive_zone="unknown"
            )

        total_teams = len(leaderboard)
        percentile = (total_teams - current_rank + 1) / total_teams * 100

        # Calculate gaps
        top_score = leaderboard[0].score if leaderboard else current_entry.score
        gap_to_top = abs(current_entry.score - top_score)

        # Gap to next better position
        gap_to_next = 0.0
        if current_rank > 1:
            next_better_idx = current_rank - 2  # Convert to 0-based index
            if next_better_idx < len(leaderboard):
                gap_to_next = abs(current_entry.score - leaderboard[next_better_idx].score)

        # Points needed to advance one position
        points_to_advance = gap_to_next

        # Determine competitive zone
        competitive_zone = self._determine_competitive_zone(current_rank, total_teams)

        return PositionAnalysis(
            current_rank=current_rank,
            current_score=current_entry.score,
            total_teams=total_teams,
            percentile=percentile,
            gap_to_top=gap_to_top,
            gap_to_next=gap_to_next,
            points_to_advance=points_to_advance,
            competitive_zone=competitive_zone
        )

    def analyze_gaps(self,
                    leaderboard: List[LeaderboardEntry],
                    current_position: PositionAnalysis) -> GapAnalysis:
        """Detailed gap analysis for strategic planning"""

        # Calculate gaps to key positions
        top_3_score = leaderboard[min(2, len(leaderboard)-1)].score if len(leaderboard) >= 3 else 0
        top_10_score = leaderboard[min(9, len(leaderboard)-1)].score if len(leaderboard) >= 10 else 0
        top_25_score = leaderboard[min(24, len(leaderboard)-1)].score if len(leaderboard) >= 25 else 0

        gap_to_top_3 = abs(current_position.current_score - top_3_score)
        gap_to_top_10 = abs(current_position.current_score - top_10_score)
        gap_to_top_25 = abs(current_position.current_score - top_25_score)

        # Calculate improvement needed (with buffer)
        improvement_needed = {
            'top_3': gap_to_top_3 * (1 + self.improvement_buffer),
            'top_10': gap_to_top_10 * (1 + self.improvement_buffer),
            'top_25': gap_to_top_25 * (1 + self.improvement_buffer),
            'beat_baseline': max(current_position.current_score * 0.05, 1.0)  # 5% improvement or 1 point
        }

        # Assess achievability
        achievability_score = self._calculate_achievability(
            current_position, improvement_needed, leaderboard
        )

        # Recommend target
        recommended_target = self._recommend_target_position(
            current_position, improvement_needed, achievability_score
        )

        return GapAnalysis(
            gap_to_top_3=gap_to_top_3,
            gap_to_top_10=gap_to_top_10,
            gap_to_top_25=gap_to_top_25,
            improvement_needed=improvement_needed,
            achievability_score=achievability_score,
            recommended_target=recommended_target
        )

    def analyze_trends(self,
                      leaderboard: List[LeaderboardEntry],
                      historical_data: Optional[List[List[LeaderboardEntry]]] = None) -> Dict[str, Any]:
        """Analyze competitive trends"""
        trends = {
            'score_distribution': self._analyze_score_distribution(leaderboard),
            'submission_patterns': self._analyze_submission_patterns(leaderboard),
            'competitive_intensity': self._calculate_competitive_intensity(leaderboard)
        }

        # Historical trend analysis if available
        if historical_data:
            trends['historical_trends'] = self._analyze_historical_trends(historical_data)
            trends['momentum_analysis'] = self._analyze_momentum(historical_data)

        return trends

    def generate_strategic_recommendations(self,
                                         position: PositionAnalysis,
                                         gaps: GapAnalysis,
                                         trends: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        recommendations = []

        # Position-based recommendations
        if position.competitive_zone == "leader":
            recommendations.extend([
                "MAINTAIN LEAD: Focus on consistent performance",
                "Conservative strategy: Avoid high-risk submissions",
                "Monitor competitors closely for catching up attempts"
            ])
        elif position.competitive_zone == "contender":
            recommendations.extend([
                "AGGRESSIVE PUSH: Target top 3 positions",
                "Focus on high-impact optimizations",
                "Take calculated risks for breakthrough performance"
            ])
        elif position.competitive_zone == "middle_pack":
            if gaps.achievability_score > 0.6:
                recommendations.extend([
                    "ADVANCEMENT OPPORTUNITY: Target top 10",
                    "Implement ensemble strategies",
                    "Focus on feature engineering improvements"
                ])
            else:
                recommendations.extend([
                    "INCREMENTAL IMPROVEMENT: Steady progress strategy",
                    "Focus on fundamental model improvements",
                    "Build strong foundation before advanced techniques"
                ])
        else:  # bottom tier
            recommendations.extend([
                "CATCH-UP STRATEGY: Fundamental improvements needed",
                "Focus on basic model validation and feature engineering",
                "Learn from top performers' approaches"
            ])

        # Gap-based recommendations
        if gaps.improvement_needed['top_3'] < 2.0:
            recommendations.append("Close to top 3: Small optimizations may yield big gains")
        elif gaps.improvement_needed['top_10'] < 5.0:
            recommendations.append("Top 10 achievable: Focus on ensemble methods")

        # Trend-based recommendations
        competitive_intensity = trends.get('competitive_intensity', 0.5)
        if competitive_intensity > 0.7:
            recommendations.append("High competition: Be prepared for multiple submissions")
        elif competitive_intensity < 0.3:
            recommendations.append("Low competition: Steady improvement strategy viable")

        return recommendations

    def assess_submission_risk(self,
                             position: PositionAnalysis,
                             gaps: GapAnalysis,
                             proposed_improvement: float) -> Dict[str, float]:
        """Assess risk of submission based on competitive position"""
        risk_factors = {}

        # Position risk
        if position.competitive_zone == "leader":
            risk_factors['position_risk'] = 0.8  # High risk of losing lead
        elif position.competitive_zone == "contender":
            risk_factors['position_risk'] = 0.4  # Moderate risk
        else:
            risk_factors['position_risk'] = 0.2  # Lower position risk

        # Improvement risk
        if proposed_improvement > gaps.improvement_needed['top_3']:
            risk_factors['overreach_risk'] = 0.9  # Very ambitious
        elif proposed_improvement > gaps.improvement_needed['top_10']:
            risk_factors['overreach_risk'] = 0.6  # Ambitious
        else:
            risk_factors['overreach_risk'] = 0.2  # Conservative

        # Competitive pressure risk
        gap_pressure = min(gaps.gap_to_top / 10, 1.0)  # Normalize gap pressure
        risk_factors['competitive_pressure'] = gap_pressure

        return risk_factors

    def analyze_competitive_landscape(self,
                                    leaderboard_data: Any,
                                    current_team: str,
                                    current_score: Optional[float] = None,
                                    historical_data: Optional[List] = None) -> CompetitiveIntelligence:
        """Comprehensive competitive landscape analysis"""

        logger.info(f"Analyzing competitive landscape for {current_team}")

        # Parse leaderboard
        leaderboard = self.parse_leaderboard(leaderboard_data)

        # Analyze position
        position_analysis = self.analyze_position(leaderboard, current_team, current_score)

        # Analyze gaps
        gap_analysis = self.analyze_gaps(leaderboard, position_analysis)

        # Analyze trends
        trend_analysis = self.analyze_trends(leaderboard, historical_data)

        # Generate recommendations
        strategic_recommendations = self.generate_strategic_recommendations(
            position_analysis, gap_analysis, trend_analysis
        )

        # Assess risks
        risk_assessment = self.assess_submission_risk(
            position_analysis, gap_analysis, gap_analysis.improvement_needed['top_10']
        )

        return CompetitiveIntelligence(
            position_analysis=position_analysis,
            gap_analysis=gap_analysis,
            trend_analysis=trend_analysis,
            strategic_recommendations=strategic_recommendations,
            risk_assessment=risk_assessment,
            timestamp=datetime.now()
        )

    def _estimate_rank(self, leaderboard: List[LeaderboardEntry], score: float) -> int:
        """Estimate rank based on score"""
        for i, entry in enumerate(leaderboard):
            if score <= entry.score:  # Assuming lower score is better (WMAPE)
                return i + 1
        return len(leaderboard) + 1

    def _determine_competitive_zone(self, rank: int, total_teams: int) -> str:
        """Determine competitive zone based on rank"""
        if rank <= self.top_tier_size:
            return "leader"
        elif rank <= self.contender_tier_size:
            return "contender"
        elif rank <= total_teams * 0.5:
            return "middle_pack"
        else:
            return "bottom"

    def _calculate_achievability(self,
                               position: PositionAnalysis,
                               improvement_needed: Dict[str, float],
                               leaderboard: List[LeaderboardEntry]) -> float:
        """Calculate achievability score for improvements"""

        # Base achievability on current position
        position_factor = max(0.1, 1.0 - (position.current_rank / position.total_teams))

        # Factor in improvement magnitude
        avg_improvement_needed = np.mean(list(improvement_needed.values()))
        improvement_factor = max(0.1, 1.0 / (1 + avg_improvement_needed / 10))

        # Factor in competitive density
        score_std = np.std([entry.score for entry in leaderboard[:20]])  # Top 20
        density_factor = min(1.0, score_std / 5.0)  # Normalize

        achievability = (position_factor + improvement_factor + density_factor) / 3
        return min(achievability, 1.0)

    def _recommend_target_position(self,
                                 position: PositionAnalysis,
                                 improvement_needed: Dict[str, float],
                                 achievability: float) -> int:
        """Recommend target position based on analysis"""

        current_rank = position.current_rank

        if achievability > 0.8 and improvement_needed['top_3'] < 3.0:
            return min(3, current_rank - 1)
        elif achievability > 0.6 and improvement_needed['top_10'] < 5.0:
            return min(10, current_rank - 2)
        elif achievability > 0.4:
            return max(1, int(current_rank * 0.8))  # 20% improvement
        else:
            return max(1, int(current_rank * 0.9))  # 10% improvement

    def _analyze_score_distribution(self, leaderboard: List[LeaderboardEntry]) -> Dict[str, float]:
        """Analyze score distribution patterns"""
        scores = [entry.score for entry in leaderboard]

        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'range': max(scores) - min(scores),
            'top_10_range': max(scores[:10]) - min(scores[:10]) if len(scores) >= 10 else 0,
            'competitive_density': np.std(scores[:20]) if len(scores) >= 20 else 0
        }

    def _analyze_submission_patterns(self, leaderboard: List[LeaderboardEntry]) -> Dict[str, Any]:
        """Analyze submission patterns"""
        submissions = [entry.submissions for entry in leaderboard]

        return {
            'avg_submissions': np.mean(submissions),
            'max_submissions': max(submissions),
            'submission_efficiency': np.corrcoef(
                [entry.rank for entry in leaderboard],
                [entry.submissions for entry in leaderboard]
            )[0,1] if len(submissions) > 1 else 0
        }

    def _calculate_competitive_intensity(self, leaderboard: List[LeaderboardEntry]) -> float:
        """Calculate competitive intensity"""
        if len(leaderboard) < 10:
            return 0.5

        # Score variance in top 10
        top_10_scores = [entry.score for entry in leaderboard[:10]]
        score_variance = np.var(top_10_scores)

        # Submission activity
        avg_submissions = np.mean([entry.submissions for entry in leaderboard[:20]])

        # Normalize and combine
        intensity = min(1.0, (1.0 / (score_variance + 0.001)) * 0.1 + avg_submissions / 10)
        return intensity

    def _analyze_historical_trends(self, historical_data: List[List[LeaderboardEntry]]) -> Dict[str, Any]:
        """Analyze historical trends"""
        # This would analyze trends over time
        # For now, return placeholder
        return {
            'trend_direction': 'stable',
            'volatility': 0.3,
            'improvement_rate': 0.05
        }

    def _analyze_momentum(self, historical_data: List[List[LeaderboardEntry]]) -> Dict[str, Any]:
        """Analyze momentum patterns"""
        # This would analyze momentum changes
        # For now, return placeholder
        return {
            'momentum_score': 0.6,
            'acceleration': 0.1,
            'consistency': 0.8
        }

def analyze_competition(leaderboard_data: Any,
                       current_team: str,
                       current_score: Optional[float] = None) -> CompetitiveIntelligence:
    """Convenience function for competitive analysis"""
    analyzer = LeaderboardAnalyzer()
    return analyzer.analyze_competitive_landscape(leaderboard_data, current_team, current_score)

if __name__ == "__main__":
    # Demo usage
    print("🏆 Leaderboard Analysis & Competitive Intelligence Demo")
    print("=" * 60)

    # Create mock leaderboard
    mock_leaderboard = [
        {'rank': 1, 'team_name': 'TopTeam', 'score': 12.5, 'submissions': 5},
        {'rank': 2, 'team_name': 'SecondPlace', 'score': 13.2, 'submissions': 4},
        {'rank': 3, 'team_name': 'ThirdPlace', 'score': 14.1, 'submissions': 3},
        {'rank': 4, 'team_name': 'OurTeam', 'score': 16.8, 'submissions': 2},
        {'rank': 5, 'team_name': 'FifthPlace', 'score': 17.3, 'submissions': 4},
    ]

    # Analyze competition
    intelligence = analyze_competition(mock_leaderboard, 'OurTeam')

    print(f"Current Position: Rank {intelligence.position_analysis.current_rank}")
    print(f"Current Score: {intelligence.position_analysis.current_score:.2f}")
    print(f"Gap to Top: {intelligence.gap_analysis.gap_to_top_3:.2f}")
    print(f"Recommended Target: Rank {intelligence.gap_analysis.recommended_target}")

    print(f"\nStrategic Recommendations:")
    for rec in intelligence.strategic_recommendations[:3]:
        print(f"  • {rec}")

    print("\n🏆 Competitive intelligence system ready!")
    print("Ready to dominate the leaderboard with strategic analysis.")