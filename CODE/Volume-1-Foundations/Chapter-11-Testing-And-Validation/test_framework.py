#!/usr/bin/env python3
"""
AI System Testing Framework

Test and validate AI-powered network automation systems.

From: AI for Networking Engineers - Volume 1, Chapter 11
Author: Eduard Dulharu

Usage:
    from test_framework import AITestFramework

    framework = AITestFramework()
    results = framework.run_accuracy_tests(test_cases)
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import json


@dataclass
class TestCase:
    """Individual test case."""
    name: str
    input: str
    expected_output: Any
    tags: List[str] = None


@dataclass
class TestResult:
    """Test result."""
    test_name: str
    passed: bool
    actual_output: Any
    expected_output: Any
    error: str = None
    execution_time: float = 0.0


class AITestFramework:
    """
    Framework for testing AI network automation systems.

    Test types:
    - Accuracy tests (output correctness)
    - Regression tests (no degradation)
    - Performance tests (latency, throughput)
    - Cost tests (budget compliance)
    """

    def __init__(self):
        """Initialize test framework."""
        self.test_results: List[TestResult] = []

    def run_accuracy_tests(
        self,
        test_cases: List[TestCase],
        system_under_test: Callable[[str], Any]
    ) -> Dict[str, Any]:
        """
        Run accuracy tests.

        Args:
            test_cases: List of test cases
            system_under_test: Function to test

        Returns:
            Test results summary
        """
        results = []
        passed = 0

        for test in test_cases:
            try:
                import time
                start = time.time()

                # Run test
                actual = system_under_test(test.input)

                execution_time = time.time() - start

                # Check result
                success = self._check_accuracy(actual, test.expected_output)

                result = TestResult(
                    test_name=test.name,
                    passed=success,
                    actual_output=actual,
                    expected_output=test.expected_output,
                    execution_time=execution_time
                )

                if success:
                    passed += 1

            except Exception as e:
                result = TestResult(
                    test_name=test.name,
                    passed=False,
                    actual_output=None,
                    expected_output=test.expected_output,
                    error=str(e)
                )

            results.append(result)
            self.test_results.append(result)

        return {
            "total_tests": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "pass_rate": f"{(passed / len(test_cases) * 100):.1f}%",
            "results": results
        }

    def _check_accuracy(self, actual: Any, expected: Any) -> bool:
        """Check if actual matches expected."""
        # Handle different comparison types
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower().strip() == actual.lower().strip()

        elif isinstance(expected, dict) and isinstance(actual, dict):
            # Check key fields match
            for key in expected.keys():
                if key not in actual:
                    return False
                if expected[key] != actual[key]:
                    return False
            return True

        else:
            return actual == expected

    def run_regression_tests(
        self,
        baseline_file: str,
        system_under_test: Callable[[str], Any]
    ) -> Dict[str, Any]:
        """
        Run regression tests against baseline.

        Args:
            baseline_file: JSON file with baseline results
            system_under_test: Function to test

        Returns:
            Regression test results
        """
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        regressions = []
        improvements = []
        unchanged = []

        for test_case in baseline['test_cases']:
            current = system_under_test(test_case['input'])
            baseline_output = test_case['output']

            # Compare
            if current == baseline_output:
                unchanged.append(test_case['name'])
            elif self._is_better(current, baseline_output, test_case['metric']):
                improvements.append({
                    "test": test_case['name'],
                    "before": baseline_output,
                    "after": current
                })
            else:
                regressions.append({
                    "test": test_case['name'],
                    "before": baseline_output,
                    "after": current
                })

        return {
            "total_tests": len(baseline['test_cases']),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": len(unchanged),
            "regression_details": regressions
        }

    def _is_better(self, current: Any, baseline: Any, metric: str) -> bool:
        """Determine if current is better than baseline."""
        if metric == "accuracy":
            # Higher is better
            return current > baseline
        elif metric == "latency":
            # Lower is better
            return current < baseline
        elif metric == "cost":
            # Lower is better
            return current < baseline
        return False

    def generate_test_report(self) -> str:
        """Generate human-readable test report."""
        if not self.test_results:
            return "No tests run yet."

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total - passed

        report = f"""
TEST REPORT
{'='*60}

Summary:
  Total Tests: {total}
  Passed: {passed} ({passed/total*100:.1f}%)
  Failed: {failed} ({failed/total*100:.1f}%)

"""

        if failed > 0:
            report += "\nFailed Tests:\n"
            for result in self.test_results:
                if not result.passed:
                    report += f"  ✗ {result.test_name}\n"
                    report += f"    Expected: {result.expected_output}\n"
                    report += f"    Got: {result.actual_output}\n"
                    if result.error:
                        report += f"    Error: {result.error}\n"
                    report += "\n"

        return report


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    AI Test Framework Demo
    ========================================
    Testing AI network automation
    ========================================
    """)

    # Test 1: Accuracy testing
    print("\nTest 1: Accuracy Tests")
    print("-" * 60)

    def mock_classifier(log: str) -> str:
        """Mock log classifier for testing."""
        if "down" in log.lower() or "failed" in log.lower():
            return "CRITICAL"
        elif "up" in log.lower():
            return "INFO"
        else:
            return "WARNING"

    test_cases = [
        TestCase(
            name="OSPF neighbor down",
            input="%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down",
            expected_output="CRITICAL"
        ),
        TestCase(
            name="Interface up",
            input="%LINK-3-UPDOWN: Interface Gi0/1, changed state to up",
            expected_output="INFO"
        ),
        TestCase(
            name="Config change",
            input="%SYS-5-CONFIG_I: Configured from console",
            expected_output="WARNING"
        )
    ]

    framework = AITestFramework()
    results = framework.run_accuracy_tests(test_cases, mock_classifier)

    print(f"Total: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {results['pass_rate']}")

    print("\nDetailed Results:")
    for r in results['results']:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.test_name}")
        if not r.passed:
            print(f"     Expected: {r.expected_output}")
            print(f"     Got: {r.actual_output}")

    # Test 2: Generate report
    print("\n\nTest 2: Test Report")
    print("-" * 60)

    report = framework.generate_test_report()
    print(report)

    print("✅ Demo complete!")
    print("\n💡 Testing Strategy:")
    print("  1. Unit tests for individual functions")
    print("  2. Integration tests for full workflows")
    print("  3. Regression tests to prevent quality degradation")
    print("  4. Performance tests for latency/cost")
    print("  5. Accuracy tests with real network data")
