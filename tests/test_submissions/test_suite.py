"""
Test suite for submissions module
Runs all tests for the submissions package
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import all test modules
from tests.test_submissions import (
    test_strategy,
    test_risk_manager,
    test_leaderboard_analyzer
)


def create_test_suite():
    """Create comprehensive test suite for submissions module"""

    # Create test suite
    suite = unittest.TestSuite()

    # Add strategy tests
    suite.addTest(unittest.TestLoader().loadTestsFromModule(test_strategy))

    # Add risk manager tests
    suite.addTest(unittest.TestLoader().loadTestsFromModule(test_risk_manager))

    # Add leaderboard analyzer tests
    suite.addTest(unittest.TestLoader().loadTestsFromModule(test_leaderboard_analyzer))

    return suite


def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""

    print("=" * 60)
    print("RUNNING SUBMISSIONS MODULE TEST SUITE")
    print("=" * 60)

    # Create test suite
    suite = create_test_suite()

    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )

    # Run tests
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


def run_specific_test_class(test_class_name):
    """Run tests for a specific test class"""

    # Map test class names to modules
    test_modules = {
        'strategy': test_strategy,
        'risk_manager': test_risk_manager,
        'leaderboard_analyzer': test_leaderboard_analyzer
    }

    if test_class_name.lower() in test_modules:
        module = test_modules[test_class_name.lower()]
        suite = unittest.TestLoader().loadTestsFromModule(module)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1
    else:
        print(f"Unknown test class: {test_class_name}")
        print(f"Available classes: {list(test_modules.keys())}")
        return 1


def main():
    """Main entry point for test suite"""

    import argparse

    parser = argparse.ArgumentParser(description="Run submissions module tests")
    parser.add_argument(
        "--class", "-c", type=str, dest="test_class",
        help="Run tests for specific class (strategy, risk_manager, leaderboard_analyzer)"
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, default=2, choices=[0, 1, 2],
        help="Test verbosity level (0=quiet, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "--failfast", "-f", action="store_true",
        help="Stop on first failure"
    )

    args = parser.parse_args()

    if args.test_class:
        return run_specific_test_class(args.test_class)
    else:
        return run_tests(args.verbosity)


if __name__ == "__main__":
    sys.exit(main())