"""
Tests for leaderboard analyzer module
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.submissions.leaderboard_analyzer import (
    PositionAnalysis, GapAnalysis, CompetitiveIntelligence,
    LeaderboardAnalyzer, create_leaderboard_analyzer, analyze_competition
)


class TestPositionAnalysis(unittest.TestCase):
    """Test cases for PositionAnalysis class"""

    def setUp(self):
        """Set up test fixtures"""
        self.position_analysis = PositionAnalysis(
            current_rank=15,
            current_score=18.5,
            total_teams=50,
            percentile=70.0,
            competitive_zone='contender',
            gap_to_top=6.5,
            gap_to_next=0.8,
            points_to_advance=0.8
        )

    def test_initialization(self):
        """Test position analysis initialization"""
        self.assertEqual(self.position_analysis.current_rank, 15)
        self.assertEqual(self.position_analysis.current_score, 18.5)
        self.assertEqual(self.position_analysis.total_teams, 50)
        self.assertEqual(self.position_analysis.percentile, 70.0)
        self.assertEqual(self.position_analysis.competitive_zone, 'contender')
        self.assertEqual(self.position_analysis.gap_to_top, 6.5)
        self.assertEqual(self.position_analysis.gap_to_next, 0.8)
        self.assertEqual(self.position_analysis.points_to_advance, 0.8)

    def test_is_in_top_tier(self):
        """Test top tier detection"""
        top_tier = PositionAnalysis(
            current_rank=2, current_score=12.0, total_teams=50,
            percentile=96.0, competitive_zone='leader',
            gap_to_top=1.0, gap_to_next=0.5, points_to_advance=0.5
        )
        self.assertTrue(top_tier.is_in_top_tier(top_n=3))
        self.assertFalse(self.position_analysis.is_in_top_tier(top_n=3))

    def test_is_leader(self):
        """Test leader detection"""
        leader = PositionAnalysis(
            current_rank=1, current_score=11.0, total_teams=50,
            percentile=98.0, competitive_zone='leader',
            gap_to_top=0.0, gap_to_next=1.5, points_to_advance=0.0
        )
        self.assertTrue(leader.is_leader())
        self.assertFalse(self.position_analysis.is_leader())

    def test_needs_major_improvement(self):
        """Test major improvement need detection"""
        bottom_tier = PositionAnalysis(
            current_rank=45, current_score=28.5, total_teams=50,
            percentile=10.0, competitive_zone='bottom_tier',
            gap_to_top=16.5, gap_to_next=2.3, points_to_advance=2.3
        )
        self.assertTrue(bottom_tier.needs_major_improvement())
        self.assertFalse(self.position_analysis.needs_major_improvement())


class TestGapAnalysis(unittest.TestCase):
    """Test cases for GapAnalysis class"""

    def setUp(self):
        """Set up test fixtures"""
        self.gap_analysis = GapAnalysis(
            gap_to_top_3=4.2,
            gap_to_top_10=1.8,
            gap_to_median=0.5,
            recommended_target=8,
            achievability_score=0.75,
            improvement_needed={
                'top_3': 4.2,
                'top_10': 1.8,
                'next_rank': 0.8
            }
        )

    def test_initialization(self):
        """Test gap analysis initialization"""
        self.assertEqual(self.gap_analysis.gap_to_top_3, 4.2)
        self.assertEqual(self.gap_analysis.gap_to_top_10, 1.8)
        self.assertEqual(self.gap_analysis.gap_to_median, 0.5)
        self.assertEqual(self.gap_analysis.recommended_target, 8)
        self.assertEqual(self.gap_analysis.achievability_score, 0.75)
        self.assertEqual(len(self.gap_analysis.improvement_needed), 3)

    def test_is_close_to_target(self):
        """Test proximity to target detection"""
        self.assertTrue(self.gap_analysis.is_close_to_target('top_10', threshold=2.0))
        self.assertFalse(self.gap_analysis.is_close_to_target('top_3', threshold=2.0))

    def test_get_most_achievable_target(self):
        """Test most achievable target identification"""
        target = self.gap_analysis.get_most_achievable_target()
        self.assertEqual(target, 'next_rank')  # Smallest improvement needed

    def test_is_highly_achievable(self):
        """Test high achievability detection"""
        self.assertTrue(self.gap_analysis.is_highly_achievable())

        low_achievable = GapAnalysis(
            gap_to_top_3=10.0, gap_to_top_10=8.0, gap_to_median=5.0,
            recommended_target=25, achievability_score=0.3,
            improvement_needed={'top_3': 10.0}
        )
        self.assertFalse(low_achievable.is_highly_achievable())


class TestCompetitiveIntelligence(unittest.TestCase):
    """Test cases for CompetitiveIntelligence class"""

    def setUp(self):
        """Set up test fixtures"""
        self.position_analysis = PositionAnalysis(
            current_rank=15, current_score=18.5, total_teams=50,
            percentile=70.0, competitive_zone='contender',
            gap_to_top=6.5, gap_to_next=0.8, points_to_advance=0.8
        )

        self.gap_analysis = GapAnalysis(
            gap_to_top_3=4.2, gap_to_top_10=1.8, gap_to_median=0.5,
            recommended_target=8, achievability_score=0.75,
            improvement_needed={'top_3': 4.2, 'top_10': 1.8, 'next_rank': 0.8}
        )

        self.competitive_intelligence = CompetitiveIntelligence(
            position_analysis=self.position_analysis,
            gap_analysis=self.gap_analysis,
            trend_analysis={'competitive_intensity': 0.65},
            strategic_recommendations=['Focus on ensemble methods', 'Optimize hyperparameters'],
            risk_assessment={'time_pressure': 0.4, 'technical_risk': 0.3}
        )

    def test_initialization(self):
        """Test competitive intelligence initialization"""
        self.assertEqual(self.competitive_intelligence.position_analysis, self.position_analysis)
        self.assertEqual(self.competitive_intelligence.gap_analysis, self.gap_analysis)
        self.assertEqual(len(self.competitive_intelligence.strategic_recommendations), 2)
        self.assertEqual(len(self.competitive_intelligence.risk_assessment), 2)

    def test_get_strategic_priority(self):
        """Test strategic priority assessment"""
        priority = self.competitive_intelligence.get_strategic_priority()
        self.assertIn(priority, ['conservative', 'balanced', 'aggressive', 'experimental'])

    def test_should_take_risks(self):
        """Test risk-taking recommendation"""
        # Should take moderate risks for contender position
        risk_taking = self.competitive_intelligence.should_take_risks()
        self.assertIsInstance(risk_taking, bool)

    def test_get_time_pressure_level(self):
        """Test time pressure assessment"""
        pressure_level = self.competitive_intelligence.get_time_pressure_level()
        self.assertIn(pressure_level, ['low', 'medium', 'high', 'critical'])


class TestLeaderboardAnalyzer(unittest.TestCase):
    """Test cases for LeaderboardAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'leaderboard_config': {
                'top_tier_size': 3,
                'contender_tier_size': 10,
                'improvement_buffer': 0.05
            },
            'timeline_analysis': {
                'critical_days_remaining': 3,
                'urgent_days_remaining': 7,
                'planning_days_remaining': 14
            },
            'strategic_positioning': {
                'leader_strategy': 'conservative',
                'contender_strategy': 'aggressive',
                'middle_pack_strategy': 'balanced',
                'bottom_tier_strategy': 'catch_up'
            }
        }
        self.analyzer = LeaderboardAnalyzer(self.config)

        # Create sample leaderboard data
        np.random.seed(42)
        self.leaderboard_data = pd.DataFrame({
            'rank': range(1, 51),
            'team_name': [f'Team_{i}' if i != 15 else 'our_team' for i in range(1, 51)],
            'score': sorted(np.random.uniform(10, 30, 50)),
            'submissions': np.random.randint(1, 6, 50)
        })

    def test_initialization(self):
        """Test leaderboard analyzer initialization"""
        self.assertEqual(self.analyzer.top_tier_size, 3)
        self.assertEqual(self.analyzer.contender_tier_size, 10)
        self.assertEqual(self.analyzer.improvement_buffer, 0.05)

    def test_find_team_position(self):
        """Test team position finding"""
        position = self.analyzer._find_team_position(self.leaderboard_data, 'our_team')

        self.assertEqual(position['rank'], 15)
        self.assertEqual(position['team_name'], 'our_team')
        self.assertIn('score', position)

    def test_find_team_position_not_found(self):
        """Test team position finding when team not in leaderboard"""
        position = self.analyzer._find_team_position(self.leaderboard_data, 'nonexistent_team')
        self.assertIsNone(position)

    def test_calculate_competitive_zone(self):
        """Test competitive zone calculation"""
        # Test leader zone
        leader_zone = self.analyzer._calculate_competitive_zone(1, 50)
        self.assertEqual(leader_zone, 'leader')

        # Test contender zone
        contender_zone = self.analyzer._calculate_competitive_zone(5, 50)
        self.assertEqual(contender_zone, 'contender')

        # Test middle pack zone
        middle_zone = self.analyzer._calculate_competitive_zone(25, 50)
        self.assertEqual(middle_zone, 'middle_pack')

        # Test bottom tier zone
        bottom_zone = self.analyzer._calculate_competitive_zone(45, 50)
        self.assertEqual(bottom_zone, 'bottom_tier')

    def test_analyze_position(self):
        """Test position analysis"""
        team_position = {
            'rank': 15,
            'team_name': 'our_team',
            'score': 18.5
        }

        position_analysis = self.analyzer._analyze_position(
            self.leaderboard_data, team_position
        )

        self.assertIsInstance(position_analysis, PositionAnalysis)
        self.assertEqual(position_analysis.current_rank, 15)
        self.assertEqual(position_analysis.current_score, 18.5)
        self.assertEqual(position_analysis.total_teams, 50)

    def test_analyze_gaps(self):
        """Test gap analysis"""
        team_position = {
            'rank': 15,
            'team_name': 'our_team',
            'score': 18.5
        }

        gap_analysis = self.analyzer._analyze_gaps(
            self.leaderboard_data, team_position
        )

        self.assertIsInstance(gap_analysis, GapAnalysis)
        self.assertGreater(gap_analysis.gap_to_top_3, 0)
        self.assertGreater(gap_analysis.gap_to_top_10, 0)

    def test_generate_strategic_recommendations(self):
        """Test strategic recommendations generation"""
        position_analysis = PositionAnalysis(
            current_rank=15, current_score=18.5, total_teams=50,
            percentile=70.0, competitive_zone='contender',
            gap_to_top=6.5, gap_to_next=0.8, points_to_advance=0.8
        )

        gap_analysis = GapAnalysis(
            gap_to_top_3=4.2, gap_to_top_10=1.8, gap_to_median=0.5,
            recommended_target=8, achievability_score=0.75,
            improvement_needed={'top_3': 4.2, 'top_10': 1.8}
        )

        recommendations = self.analyzer._generate_strategic_recommendations(
            position_analysis, gap_analysis, days_remaining=7
        )

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_analyze_full_intelligence(self):
        """Test full competitive intelligence analysis"""
        intelligence = self.analyzer.analyze_competitive_intelligence(
            self.leaderboard_data, 'our_team'
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)
        self.assertIsInstance(intelligence.position_analysis, PositionAnalysis)
        self.assertIsInstance(intelligence.gap_analysis, GapAnalysis)
        self.assertGreater(len(intelligence.strategic_recommendations), 0)

    def test_analyze_with_custom_score(self):
        """Test analysis with custom current score"""
        intelligence = self.analyzer.analyze_competitive_intelligence(
            self.leaderboard_data, 'our_team', current_score=16.0
        )

        self.assertEqual(intelligence.position_analysis.current_score, 16.0)

    def test_analyze_team_not_found(self):
        """Test analysis when team is not found in leaderboard"""
        intelligence = self.analyzer.analyze_competitive_intelligence(
            self.leaderboard_data, 'nonexistent_team', current_score=20.0
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)
        self.assertEqual(intelligence.position_analysis.current_score, 20.0)


class TestCreateLeaderboardAnalyzer(unittest.TestCase):
    """Test cases for create_leaderboard_analyzer factory function"""

    def test_create_with_config(self):
        """Test creating analyzer with configuration"""
        config = {
            'leaderboard_config': {
                'top_tier_size': 5,
                'contender_tier_size': 15
            }
        }

        analyzer = create_leaderboard_analyzer(config)

        self.assertIsInstance(analyzer, LeaderboardAnalyzer)
        self.assertEqual(analyzer.top_tier_size, 5)
        self.assertEqual(analyzer.contender_tier_size, 15)

    def test_create_with_defaults(self):
        """Test creating analyzer with default configuration"""
        analyzer = create_leaderboard_analyzer({})

        self.assertIsInstance(analyzer, LeaderboardAnalyzer)
        # Should have reasonable defaults
        self.assertGreater(analyzer.top_tier_size, 0)
        self.assertGreater(analyzer.contender_tier_size, 0)


class TestAnalyzeCompetition(unittest.TestCase):
    """Test cases for analyze_competition function"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.leaderboard_data = pd.DataFrame({
            'rank': range(1, 51),
            'team_name': [f'Team_{i}' if i != 15 else 'our_team' for i in range(1, 51)],
            'score': sorted(np.random.uniform(10, 30, 50)),
            'submissions': np.random.randint(1, 6, 50)
        })

    def test_analyze_competition_with_team_in_leaderboard(self):
        """Test competition analysis with team in leaderboard"""
        intelligence = analyze_competition(
            self.leaderboard_data, 'our_team'
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)
        self.assertEqual(intelligence.position_analysis.current_rank, 15)

    def test_analyze_competition_with_custom_score(self):
        """Test competition analysis with custom score"""
        intelligence = analyze_competition(
            self.leaderboard_data, 'our_team', current_score=16.0
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)
        self.assertEqual(intelligence.position_analysis.current_score, 16.0)

    def test_analyze_competition_team_not_found(self):
        """Test competition analysis when team not found"""
        intelligence = analyze_competition(
            self.leaderboard_data, 'missing_team', current_score=25.0
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)
        self.assertEqual(intelligence.position_analysis.current_score, 25.0)

    def test_analyze_competition_with_config(self):
        """Test competition analysis with custom configuration"""
        config = {
            'leaderboard_config': {
                'top_tier_size': 5,
                'contender_tier_size': 15
            }
        }

        intelligence = analyze_competition(
            self.leaderboard_data, 'our_team', config=config
        )

        self.assertIsInstance(intelligence, CompetitiveIntelligence)


if __name__ == '__main__':
    unittest.main()