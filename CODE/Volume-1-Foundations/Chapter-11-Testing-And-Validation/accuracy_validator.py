#!/usr/bin/env python3
"""
Accuracy Validator - Measure AI System Accuracy

Test AI systems against ground truth data to measure and improve accuracy.

From: AI for Networking Engineers - Volume 1, Chapter 11
Author: Eduard Dulharu

Usage:
    from accuracy_validator import AccuracyValidator

    validator = AccuracyValidator()
    results = validator.validate_classification(test_cases, classifier_func)
    print(f"Accuracy: {results['accuracy']:.2%}")
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import statistics


class MatchType(str, Enum):
    """Match types for validation."""
    EXACT = "exact"
    PARTIAL = "partial"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


@dataclass
class ValidationCase:
    """Single validation test case."""
    input: str
    expected_output: Any
    category: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Individual validation result."""
    test_case: str
    expected: Any
    actual: Any
    passed: bool
    match_score: float  # 0.0 to 1.0
    error: Optional[str] = None


class AccuracyValidator:
    """
    Validate AI system accuracy against ground truth.

    Features:
    - Multiple matching strategies (exact, partial, fuzzy, semantic)
    - Confusion matrix for classification tasks
    - Precision, recall, F1 score
    - Category-based analysis
    - Error analysis
    """

    def __init__(self, match_type: MatchType = MatchType.EXACT):
        """
        Initialize validator.

        Args:
            match_type: Default matching strategy
        """
        self.match_type = match_type
        self.results: List[ValidationResult] = []

    def validate_classification(
        self,
        test_cases: List[ValidationCase],
        classifier: Callable[[str], str],
        match_type: Optional[MatchType] = None
    ) -> Dict[str, Any]:
        """
        Validate classification accuracy.

        Args:
            test_cases: List of test cases
            classifier: Function that classifies input
            match_type: Optional override for match type

        Returns:
            Validation results with accuracy metrics
        """
        match_type = match_type or self.match_type
        results = []

        for case in test_cases:
            try:
                actual = classifier(case.input)
                passed, score = self._check_match(
                    case.expected_output,
                    actual,
                    match_type
                )

                result = ValidationResult(
                    test_case=case.input,
                    expected=case.expected_output,
                    actual=actual,
                    passed=passed,
                    match_score=score
                )

            except Exception as e:
                result = ValidationResult(
                    test_case=case.input,
                    expected=case.expected_output,
                    actual=None,
                    passed=False,
                    match_score=0.0,
                    error=str(e)
                )

            results.append(result)

        self.results = results

        # Calculate metrics
        return self._calculate_metrics(results, test_cases)

    def _check_match(
        self,
        expected: Any,
        actual: Any,
        match_type: MatchType
    ) -> tuple[bool, float]:
        """
        Check if actual matches expected.

        Args:
            expected: Expected value
            actual: Actual value
            match_type: Matching strategy

        Returns:
            Tuple of (passed, match_score)
        """
        if match_type == MatchType.EXACT:
            return self._exact_match(expected, actual)

        elif match_type == MatchType.PARTIAL:
            return self._partial_match(expected, actual)

        elif match_type == MatchType.FUZZY:
            return self._fuzzy_match(expected, actual)

        elif match_type == MatchType.SEMANTIC:
            return self._semantic_match(expected, actual)

        return False, 0.0

    def _exact_match(self, expected: Any, actual: Any) -> tuple[bool, float]:
        """Exact string match (case-insensitive)."""
        if isinstance(expected, str) and isinstance(actual, str):
            match = expected.lower().strip() == actual.lower().strip()
            return match, 1.0 if match else 0.0
        else:
            match = expected == actual
            return match, 1.0 if match else 0.0

    def _partial_match(self, expected: Any, actual: Any) -> tuple[bool, float]:
        """Partial match - expected is substring of actual."""
        if isinstance(expected, str) and isinstance(actual, str):
            expected_lower = expected.lower()
            actual_lower = actual.lower()

            if expected_lower in actual_lower:
                score = len(expected_lower) / len(actual_lower)
                return True, score
            elif actual_lower in expected_lower:
                score = len(actual_lower) / len(expected_lower)
                return True, score

        return False, 0.0

    def _fuzzy_match(self, expected: Any, actual: Any) -> tuple[bool, float]:
        """Fuzzy match using Levenshtein-like distance."""
        if not isinstance(expected, str) or not isinstance(actual, str):
            return expected == actual, 1.0 if expected == actual else 0.0

        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        # Simple fuzzy matching based on common words
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())

        if not expected_words:
            return False, 0.0

        common_words = expected_words & actual_words
        score = len(common_words) / len(expected_words)

        # Consider match if > 80% of words match
        passed = score >= 0.8

        return passed, score

    def _semantic_match(self, expected: Any, actual: Any) -> tuple[bool, float]:
        """
        Semantic match (simplified - production would use embeddings).

        Args:
            expected: Expected value
            actual: Actual value

        Returns:
            Match status and score
        """
        # Simplified semantic matching
        # In production, use sentence embeddings and cosine similarity

        # For now, use fuzzy match as approximation
        return self._fuzzy_match(expected, actual)

    def _calculate_metrics(
        self,
        results: List[ValidationResult],
        test_cases: List[ValidationCase]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive accuracy metrics.

        Args:
            results: Validation results
            test_cases: Original test cases

        Returns:
            Metrics dictionary
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Overall accuracy
        accuracy = passed / total if total > 0 else 0.0

        # Average match score
        avg_score = statistics.mean(r.match_score for r in results) if results else 0.0

        # Error analysis
        errors = [r for r in results if not r.passed]

        # Category-based metrics
        category_metrics = self._calculate_category_metrics(results, test_cases)

        # Confusion matrix (for classification)
        confusion_matrix = self._build_confusion_matrix(results)

        # Precision, Recall, F1
        precision_recall = self._calculate_precision_recall(confusion_matrix)

        return {
            "overall": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "accuracy": accuracy,
                "avg_match_score": avg_score
            },
            "by_category": category_metrics,
            "confusion_matrix": confusion_matrix,
            "precision_recall": precision_recall,
            "errors": [
                {
                    "input": e.test_case,
                    "expected": e.expected,
                    "actual": e.actual,
                    "error": e.error
                }
                for e in errors[:10]  # Limit to first 10
            ]
        }

    def _calculate_category_metrics(
        self,
        results: List[ValidationResult],
        test_cases: List[ValidationCase]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics by category."""
        # Group by category
        categories = {}

        for result, case in zip(results, test_cases):
            category = case.category

            if category not in categories:
                categories[category] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0
                }

            categories[category]["total"] += 1
            if result.passed:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1

        # Calculate accuracy per category
        for category, metrics in categories.items():
            metrics["accuracy"] = metrics["passed"] / metrics["total"] if metrics["total"] > 0 else 0.0

        return categories

    def _build_confusion_matrix(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build confusion matrix for classification.

        Args:
            results: Validation results

        Returns:
            Confusion matrix as nested dict
        """
        matrix = {}

        for result in results:
            expected = str(result.expected)
            actual = str(result.actual)

            if expected not in matrix:
                matrix[expected] = {}

            if actual not in matrix[expected]:
                matrix[expected][actual] = 0

            matrix[expected][actual] += 1

        return matrix

    def _calculate_precision_recall(
        self,
        confusion_matrix: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, F1 for each class.

        Args:
            confusion_matrix: Confusion matrix

        Returns:
            Precision/recall metrics by class
        """
        metrics = {}

        all_classes = set(confusion_matrix.keys())
        for predicted_values in confusion_matrix.values():
            all_classes.update(predicted_values.keys())

        for class_label in all_classes:
            # True positives
            tp = confusion_matrix.get(class_label, {}).get(class_label, 0)

            # False positives (predicted as class_label but actually something else)
            fp = sum(
                confusion_matrix.get(true_label, {}).get(class_label, 0)
                for true_label in all_classes
                if true_label != class_label
            )

            # False negatives (actually class_label but predicted as something else)
            fn = sum(
                confusion_matrix.get(class_label, {}).get(pred_label, 0)
                for pred_label in all_classes
                if pred_label != class_label
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn  # Total actual instances of this class
            }

        return metrics

    def generate_report(self) -> str:
        """
        Generate human-readable validation report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available."

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        accuracy = passed / total if total > 0 else 0.0

        report = f"""
ACCURACY VALIDATION REPORT
{'='*60}

Overall Results:
  Total Tests:  {total}
  Passed:       {passed} ({accuracy:.1%})
  Failed:       {failed} ({(1-accuracy):.1%})

"""

        # Failed tests
        if failed > 0:
            report += "\nFailed Tests:\n"
            report += "-" * 60 + "\n"

            for result in self.results:
                if not result.passed:
                    report += f"\nInput: {result.test_case[:60]}...\n"
                    report += f"  Expected: {result.expected}\n"
                    report += f"  Got:      {result.actual}\n"
                    report += f"  Score:    {result.match_score:.2f}\n"
                    if result.error:
                        report += f"  Error:    {result.error}\n"

        return report


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Accuracy Validator Demo
    ========================================
    Measure AI system accuracy
    ========================================
    """)

    # Mock log classifier for testing
    def log_classifier(log_message: str) -> str:
        """Simple log classifier."""
        log_lower = log_message.lower()

        if 'down' in log_lower or 'failed' in log_lower:
            return "CRITICAL"
        elif 'up' in log_lower:
            return "INFO"
        elif 'warning' in log_lower or 'error' in log_lower:
            return "WARNING"
        else:
            return "INFO"

    # Test cases
    test_cases = [
        ValidationCase(
            input="%LINK-3-UPDOWN: Interface Gi0/1, changed state to down",
            expected_output="CRITICAL",
            category="interface"
        ),
        ValidationCase(
            input="%LINEPROTO-5-UPDOWN: Line protocol on Interface Gi0/1, changed state to up",
            expected_output="INFO",
            category="interface"
        ),
        ValidationCase(
            input="%OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.2 from FULL to DOWN",
            expected_output="CRITICAL",
            category="routing"
        ),
        ValidationCase(
            input="%SYS-5-CONFIG_I: Configured from console",
            expected_output="INFO",
            category="system"
        ),
        ValidationCase(
            input="%SEC-4-IPACCESSLOGDP: list 101 denied tcp 192.168.1.1(4521) -> 10.1.1.1(22)",
            expected_output="WARNING",
            category="security"
        ),
        ValidationCase(
            input="%BGP-3-NOTIFICATION: sent to neighbor 10.1.1.1 4/0 (hold time expired)",
            expected_output="CRITICAL",
            category="routing"
        )
    ]

    # Test 1: Exact matching
    print("\nTest 1: Exact Match Validation")
    print("-" * 60)

    validator = AccuracyValidator(match_type=MatchType.EXACT)
    results = validator.validate_classification(test_cases, log_classifier)

    print(f"Overall Accuracy: {results['overall']['accuracy']:.1%}")
    print(f"Passed: {results['overall']['passed']}/{results['overall']['total_tests']}")

    # Test 2: Category-based metrics
    print("\n\nTest 2: Category-Based Metrics")
    print("-" * 60)

    for category, metrics in results['by_category'].items():
        print(f"{category:12s}: {metrics['accuracy']:.1%} ({metrics['passed']}/{metrics['total']})")

    # Test 3: Confusion matrix
    print("\n\nTest 3: Confusion Matrix")
    print("-" * 60)

    print("\nActual \\ Predicted:")
    for true_label, predictions in results['confusion_matrix'].items():
        print(f"  {true_label:12s}:", end="")
        for pred_label, count in predictions.items():
            print(f"  {pred_label}:{count}", end="")
        print()

    # Test 4: Precision/Recall
    print("\n\nTest 4: Precision, Recall, F1")
    print("-" * 60)

    for class_label, metrics in results['precision_recall'].items():
        print(f"\n{class_label}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1 Score:  {metrics['f1_score']:.2%}")
        print(f"  Support:   {metrics['support']}")

    # Test 5: Full report
    print("\n\nTest 5: Full Validation Report")
    print("-" * 60)

    report = validator.generate_report()
    print(report)

    # Test 6: Fuzzy matching
    print("\n\nTest 6: Fuzzy Matching")
    print("-" * 60)

    fuzzy_validator = AccuracyValidator(match_type=MatchType.FUZZY)

    # Test case with similar but not exact output
    fuzzy_cases = [
        ValidationCase(
            input="Test input",
            expected_output="critical error",
            category="test"
        )
    ]

    def fuzzy_classifier(msg: str) -> str:
        return "critical ERROR detected"  # Similar but not exact

    fuzzy_results = fuzzy_validator.validate_classification(fuzzy_cases, fuzzy_classifier)
    print(f"Fuzzy match accuracy: {fuzzy_results['overall']['accuracy']:.1%}")
    print(f"Average match score: {fuzzy_results['overall']['avg_match_score']:.2f}")

    print("\n✅ Demo complete!")
    print("\n💡 Key Metrics:")
    print("  - Accuracy: Overall correctness")
    print("  - Precision: True positives / (True positives + False positives)")
    print("  - Recall: True positives / (True positives + False negatives)")
    print("  - F1 Score: Harmonic mean of precision and recall")

    print("\n⚠️  Validation Best Practices:")
    print("  ☐ Use diverse test cases")
    print("  ☐ Include edge cases")
    print("  ☐ Test across categories")
    print("  ☐ Track metrics over time")
    print("  ☐ Set minimum accuracy thresholds")
    print("  ☐ Re-validate after changes")
