#!/usr/bin/env python3
"""
Regression Tests - Prevent Quality Degradation

Track system performance over time and detect regressions in accuracy, latency,
cost, or other key metrics.

From: AI for Networking Engineers - Volume 1, Chapter 11
Author: Eduard Dulharu

Usage:
    from regression_tests import RegressionTester

    tester = RegressionTester()
    result = tester.run_regression_test(current_func, baseline_file="baseline.json")
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class BaselineMetrics:
    """Baseline performance metrics."""
    test_name: str
    timestamp: str
    accuracy: float
    avg_latency_ms: float
    avg_cost: float
    total_tests: int
    version: str


@dataclass
class RegressionResult:
    """Regression test result."""
    test_name: str
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold_percent: float
    passed: bool
    severity: str  # "none", "minor", "major", "critical"


class RegressionTester:
    """
    Test for regressions in AI system performance.

    Features:
    - Track accuracy, latency, cost over time
    - Compare against baseline
    - Detect performance degradation
    - Generate regression reports
    """

    # Regression thresholds (percentage change)
    THRESHOLDS = {
        "accuracy": {
            "minor": 2.0,  # 2% decrease
            "major": 5.0,  # 5% decrease
            "critical": 10.0  # 10% decrease
        },
        "latency": {
            "minor": 20.0,  # 20% increase
            "major": 50.0,  # 50% increase
            "critical": 100.0  # 100% increase (doubled)
        },
        "cost": {
            "minor": 15.0,  # 15% increase
            "major": 30.0,  # 30% increase
            "critical": 50.0  # 50% increase
        }
    }

    def __init__(self, baseline_dir: str = "./baselines"):
        """
        Initialize regression tester.

        Args:
            baseline_dir: Directory for baseline files
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(
        self,
        test_name: str,
        metrics: Dict[str, Any],
        version: str = "1.0"
    ) -> Path:
        """
        Save current metrics as baseline.

        Args:
            test_name: Name of the test
            metrics: Performance metrics
            version: System version

        Returns:
            Path to baseline file
        """
        baseline = BaselineMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            accuracy=metrics.get('accuracy', 0.0),
            avg_latency_ms=metrics.get('avg_latency_ms', 0.0),
            avg_cost=metrics.get('avg_cost', 0.0),
            total_tests=metrics.get('total_tests', 0),
            version=version
        )

        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"

        with open(baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)

        print(f"✓ Baseline saved: {baseline_file}")
        return baseline_file

    def load_baseline(self, test_name: str) -> Optional[BaselineMetrics]:
        """
        Load baseline metrics.

        Args:
            test_name: Name of the test

        Returns:
            BaselineMetrics or None if not found
        """
        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"

        if not baseline_file.exists():
            print(f"⚠ No baseline found for: {test_name}")
            return None

        with open(baseline_file, 'r') as f:
            data = json.load(f)

        return BaselineMetrics(**data)

    def run_regression_test(
        self,
        test_name: str,
        test_function: Callable,
        test_cases: List[Any],
        create_baseline: bool = False,
        version: str = "1.0"
    ) -> Dict[str, Any]:
        """
        Run regression test.

        Args:
            test_name: Name of the test
            test_function: Function to test
            test_cases: Test cases to run
            create_baseline: Create new baseline instead of comparing
            version: System version

        Returns:
            Regression test results
        """
        # Run tests and collect metrics
        current_metrics = self._run_tests(test_function, test_cases)

        if create_baseline:
            # Save as new baseline
            self.save_baseline(test_name, current_metrics, version)
            return {
                "action": "baseline_created",
                "test_name": test_name,
                "metrics": current_metrics
            }

        # Load baseline
        baseline = self.load_baseline(test_name)

        if not baseline:
            print(f"Creating initial baseline for: {test_name}")
            self.save_baseline(test_name, current_metrics, version)
            return {
                "action": "initial_baseline_created",
                "test_name": test_name,
                "metrics": current_metrics
            }

        # Compare with baseline
        regressions = self._detect_regressions(baseline, current_metrics)

        # Determine overall status
        has_critical = any(r.severity == "critical" for r in regressions)
        has_major = any(r.severity == "major" for r in regressions)
        has_minor = any(r.severity == "minor" for r in regressions)

        if has_critical:
            status = "FAILED_CRITICAL"
        elif has_major:
            status = "FAILED_MAJOR"
        elif has_minor:
            status = "WARNING"
        else:
            status = "PASSED"

        return {
            "status": status,
            "test_name": test_name,
            "baseline": asdict(baseline),
            "current_metrics": current_metrics,
            "regressions": [asdict(r) for r in regressions],
            "summary": {
                "critical": sum(1 for r in regressions if r.severity == "critical"),
                "major": sum(1 for r in regressions if r.severity == "major"),
                "minor": sum(1 for r in regressions if r.severity == "minor")
            }
        }

    def _run_tests(
        self,
        test_function: Callable,
        test_cases: List[Any]
    ) -> Dict[str, Any]:
        """
        Run tests and collect metrics.

        Args:
            test_function: Function to test
            test_cases: Test cases

        Returns:
            Performance metrics
        """
        import time

        results = []
        latencies = []
        costs = []

        for test_case in test_cases:
            start = time.time()

            try:
                if isinstance(test_case, dict):
                    result = test_function(test_case['input'])
                    expected = test_case.get('expected')
                else:
                    result = test_function(test_case)
                    expected = None

                latency = (time.time() - start) * 1000  # Convert to ms

                # Mock cost (would be actual API cost in production)
                cost = latency * 0.00001  # Mock: $0.00001 per ms

                latencies.append(latency)
                costs.append(cost)

                # Check correctness if expected provided
                if expected is not None:
                    correct = str(result).lower().strip() == str(expected).lower().strip()
                else:
                    correct = True  # Assume correct if no expected value

                results.append(correct)

            except Exception as e:
                print(f"Test failed: {e}")
                results.append(False)
                latencies.append(0.0)
                costs.append(0.0)

        # Calculate metrics
        accuracy = sum(results) / len(results) if results else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        avg_cost = statistics.mean(costs) if costs else 0.0

        return {
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "avg_cost": avg_cost,
            "total_tests": len(test_cases)
        }

    def _detect_regressions(
        self,
        baseline: BaselineMetrics,
        current_metrics: Dict[str, Any]
    ) -> List[RegressionResult]:
        """
        Detect regressions by comparing with baseline.

        Args:
            baseline: Baseline metrics
            current_metrics: Current metrics

        Returns:
            List of regression results
        """
        regressions = []

        # Check accuracy (decrease is bad)
        accuracy_change = self._calculate_change_percent(
            baseline.accuracy,
            current_metrics['accuracy']
        )

        if accuracy_change < 0:  # Decreased
            severity = self._determine_severity(
                abs(accuracy_change),
                self.THRESHOLDS['accuracy']
            )

            regressions.append(RegressionResult(
                test_name=baseline.test_name,
                metric="accuracy",
                baseline_value=baseline.accuracy,
                current_value=current_metrics['accuracy'],
                change_percent=accuracy_change,
                threshold_percent=self.THRESHOLDS['accuracy']['minor'],
                passed=severity == "none",
                severity=severity
            ))

        # Check latency (increase is bad)
        latency_change = self._calculate_change_percent(
            baseline.avg_latency_ms,
            current_metrics['avg_latency_ms']
        )

        if latency_change > 0:  # Increased
            severity = self._determine_severity(
                latency_change,
                self.THRESHOLDS['latency']
            )

            regressions.append(RegressionResult(
                test_name=baseline.test_name,
                metric="latency",
                baseline_value=baseline.avg_latency_ms,
                current_value=current_metrics['avg_latency_ms'],
                change_percent=latency_change,
                threshold_percent=self.THRESHOLDS['latency']['minor'],
                passed=severity == "none",
                severity=severity
            ))

        # Check cost (increase is bad)
        cost_change = self._calculate_change_percent(
            baseline.avg_cost,
            current_metrics['avg_cost']
        )

        if cost_change > 0:  # Increased
            severity = self._determine_severity(
                cost_change,
                self.THRESHOLDS['cost']
            )

            regressions.append(RegressionResult(
                test_name=baseline.test_name,
                metric="cost",
                baseline_value=baseline.avg_cost,
                current_value=current_metrics['avg_cost'],
                change_percent=cost_change,
                threshold_percent=self.THRESHOLDS['cost']['minor'],
                passed=severity == "none",
                severity=severity
            ))

        return regressions

    def _calculate_change_percent(
        self,
        baseline_value: float,
        current_value: float
    ) -> float:
        """
        Calculate percentage change.

        Args:
            baseline_value: Baseline value
            current_value: Current value

        Returns:
            Percentage change (positive = increase, negative = decrease)
        """
        if baseline_value == 0:
            return 0.0

        change = ((current_value - baseline_value) / baseline_value) * 100
        return change

    def _determine_severity(
        self,
        change_percent: float,
        thresholds: Dict[str, float]
    ) -> str:
        """
        Determine regression severity.

        Args:
            change_percent: Percentage change (absolute value)
            thresholds: Severity thresholds

        Returns:
            Severity level
        """
        if change_percent >= thresholds['critical']:
            return "critical"
        elif change_percent >= thresholds['major']:
            return "major"
        elif change_percent >= thresholds['minor']:
            return "minor"
        else:
            return "none"

    def generate_report(self, regression_result: Dict[str, Any]) -> str:
        """
        Generate human-readable regression report.

        Args:
            regression_result: Regression test result

        Returns:
            Formatted report
        """
        status = regression_result['status']
        test_name = regression_result['test_name']

        # Status icon
        if status == "PASSED":
            status_icon = "✓"
        elif status == "WARNING":
            status_icon = "⚠"
        else:
            status_icon = "✗"

        report = f"""
REGRESSION TEST REPORT
{'='*60}

Test: {test_name}
Status: {status_icon} {status}

"""

        # Metrics comparison
        if 'baseline' in regression_result:
            baseline = regression_result['baseline']
            current = regression_result['current_metrics']

            report += "Metrics Comparison:\n"
            report += "-" * 60 + "\n"

            report += f"\nAccuracy:\n"
            report += f"  Baseline: {baseline['accuracy']:.2%}\n"
            report += f"  Current:  {current['accuracy']:.2%}\n"
            change = ((current['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100 if baseline['accuracy'] > 0 else 0
            report += f"  Change:   {change:+.2f}%\n"

            report += f"\nLatency:\n"
            report += f"  Baseline: {baseline['avg_latency_ms']:.2f} ms\n"
            report += f"  Current:  {current['avg_latency_ms']:.2f} ms\n"
            change = ((current['avg_latency_ms'] - baseline['avg_latency_ms']) / baseline['avg_latency_ms']) * 100 if baseline['avg_latency_ms'] > 0 else 0
            report += f"  Change:   {change:+.2f}%\n"

            report += f"\nCost:\n"
            report += f"  Baseline: ${baseline['avg_cost']:.6f}\n"
            report += f"  Current:  ${current['avg_cost']:.6f}\n"
            change = ((current['avg_cost'] - baseline['avg_cost']) / baseline['avg_cost']) * 100 if baseline['avg_cost'] > 0 else 0
            report += f"  Change:   {change:+.2f}%\n"

        # Regressions
        if regression_result.get('regressions'):
            report += "\nRegressions Detected:\n"
            report += "-" * 60 + "\n"

            for reg in regression_result['regressions']:
                severity_icon = {
                    "critical": "🔴",
                    "major": "🟠",
                    "minor": "🟡",
                    "none": "🟢"
                }.get(reg['severity'], "⚠")

                report += f"\n{severity_icon} {reg['metric'].upper()} ({reg['severity']})\n"
                report += f"  Baseline: {reg['baseline_value']:.4f}\n"
                report += f"  Current:  {reg['current_value']:.4f}\n"
                report += f"  Change:   {reg['change_percent']:+.2f}%\n"

        return report


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Regression Tests Demo
    ========================================
    Prevent quality degradation
    ========================================
    """)

    # Mock classifier
    def log_classifier(log: str) -> str:
        """Mock log classifier."""
        log_lower = log.lower()
        if 'down' in log_lower or 'failed' in log_lower:
            return "CRITICAL"
        elif 'up' in log_lower:
            return "INFO"
        return "WARNING"

    # Test cases
    test_cases = [
        {"input": "%LINK-3-UPDOWN: Interface Gi0/1, changed state to down", "expected": "CRITICAL"},
        {"input": "%LINEPROTO-5-UPDOWN: Line protocol on Interface Gi0/1, changed state to up", "expected": "INFO"},
        {"input": "%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down", "expected": "CRITICAL"},
        {"input": "%SYS-5-CONFIG_I: Configured from console", "expected": "WARNING"},
    ]

    tester = RegressionTester(baseline_dir="./demo_baselines")

    # Test 1: Create initial baseline
    print("\nTest 1: Create Baseline")
    print("-" * 60)

    result = tester.run_regression_test(
        test_name="log_classification",
        test_function=log_classifier,
        test_cases=test_cases,
        create_baseline=True,
        version="1.0"
    )

    print(f"Action: {result['action']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
    print(f"Avg Latency: {result['metrics']['avg_latency_ms']:.2f} ms")

    # Test 2: Run regression test (no regression)
    print("\n\nTest 2: Run Regression Test (No Changes)")
    print("-" * 60)

    result = tester.run_regression_test(
        test_name="log_classification",
        test_function=log_classifier,
        test_cases=test_cases,
        version="1.1"
    )

    print(f"Status: {result['status']}")
    print(f"Summary: {result['summary']}")

    # Test 3: Simulate regression (degraded classifier)
    print("\n\nTest 3: Simulate Regression")
    print("-" * 60)

    def degraded_classifier(log: str) -> str:
        """Degraded classifier - always returns WARNING."""
        import time
        time.sleep(0.01)  # Simulate higher latency
        return "WARNING"  # Less accurate

    result = tester.run_regression_test(
        test_name="log_classification_degraded",
        test_function=degraded_classifier,
        test_cases=test_cases,
        create_baseline=False,
        version="1.2"
    )

    # For this test, we need to create a baseline first
    baseline_result = tester.run_regression_test(
        test_name="log_classification_degraded",
        test_function=log_classifier,
        test_cases=test_cases,
        create_baseline=True,
        version="1.0"
    )

    # Now run with degraded classifier
    result = tester.run_regression_test(
        test_name="log_classification_degraded",
        test_function=degraded_classifier,
        test_cases=test_cases,
        version="1.2"
    )

    print(f"Status: {result['status']}")
    print(f"Regressions found: {len(result['regressions'])}")

    for reg in result['regressions']:
        print(f"  - {reg['metric']}: {reg['severity']} ({reg['change_percent']:+.2f}%)")

    # Test 4: Generate report
    print("\n\nTest 4: Regression Report")
    print("-" * 60)

    report = tester.generate_report(result)
    print(report)

    # Clean up
    import shutil
    shutil.rmtree("./demo_baselines")

    print("\n✅ Demo complete!")
    print("\n💡 Regression Testing Strategy:")
    print("  1. Establish baseline on known-good version")
    print("  2. Run regression tests before each release")
    print("  3. Track metrics over time")
    print("  4. Set appropriate thresholds for your use case")
    print("  5. Investigate and fix regressions immediately")

    print("\n⚠️  Key Metrics to Track:")
    print("  ☐ Accuracy (should not decrease)")
    print("  ☐ Latency (should not increase)")
    print("  ☐ Cost (should not increase)")
    print("  ☐ Error rate (should not increase)")
    print("  ☐ False positive/negative rates")
