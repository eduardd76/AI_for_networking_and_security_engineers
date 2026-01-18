#!/usr/bin/env python3
"""
Prompt Testing Framework

Systematically test and evaluate prompt effectiveness.

From: AI for Networking Engineers - Volume 1, Chapter 5
Author: Eduard Dulharu (Ed Harmoosh)

Test prompts against known inputs/outputs to measure accuracy and reliability.

Usage:
    from prompt_tester import PromptTester, TestCase

    # Define test cases
    tests = [
        TestCase(
            input="test input",
            expected_output="expected result",
            description="what this tests"
        )
    ]

    # Run tests
    tester = PromptTester()
    results = tester.run_test_suite(prompt_template, tests)
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A test case for prompt evaluation."""
    input: str
    expected_output: str
    description: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Result of a single test."""
    test_description: str
    passed: bool
    expected: str
    actual: str
    input: str
    tokens_used: int = 0
    cost: float = 0.0
    latency: float = 0.0
    error: Optional[str] = None


class PromptTester:
    """
    Framework for testing and evaluating prompts.

    Runs test suites and generates reports on prompt effectiveness.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize prompt tester.

        Args:
            api_key: Anthropic API key (defaults to env var)
            model: Model to use for testing
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def run_test_suite(
        self,
        prompt_template: Callable[[str], str],
        test_cases: List[TestCase],
        temperature: float = 0.0,
        max_tokens: int = 500,
        matching_mode: str = "contains"
    ) -> Dict[str, Any]:
        """
        Run a suite of tests on a prompt template.

        Args:
            prompt_template: Function that takes input and returns prompt
            test_cases: List of test cases
            temperature: Temperature setting (default: 0.0 for determinism)
            max_tokens: Maximum tokens in response
            matching_mode: How to match expected output
                - "contains": expected text appears in actual output
                - "exact": exact string match
                - "starts": actual output starts with expected
                - "ends": actual output ends with expected

        Returns:
            Test results with pass/fail and metrics
        """
        results = []
        passed = 0
        total = len(test_cases)
        total_cost = 0.0
        total_latency = 0.0

        print(f"\n{'='*80}")
        print(f"PROMPT TEST SUITE ({total} tests)")
        print(f"Model: {self.model}, Temperature: {temperature}")
        print(f"{'='*80}\n")

        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}/{total}: {test.description}")

            # Generate prompt from template
            try:
                prompt = prompt_template(test.input)
            except Exception as e:
                print(f"  ✗ ERROR generating prompt: {e}\n")
                results.append(TestResult(
                    test_description=test.description,
                    passed=False,
                    expected=test.expected_output,
                    actual="",
                    input=test.input,
                    error=f"Prompt generation failed: {e}"
                ))
                continue

            # Call API
            try:
                import time
                start_time = time.time()

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                latency = time.time() - start_time
                actual_output = response.content[0].text.strip()

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (input_tokens / 1_000_000) * 3.0  # Sonnet pricing
                cost += (output_tokens / 1_000_000) * 15.0

                total_cost += cost
                total_latency += latency

                # Check if expected output matches actual
                success = self._check_match(
                    test.expected_output,
                    actual_output,
                    matching_mode
                )

                if success:
                    passed += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"

                print(f"  {status}")
                print(f"  Expected: {test.expected_output[:80]}")
                print(f"  Got: {actual_output[:80]}...")
                print(f"  Latency: {latency:.2f}s, Cost: ${cost:.6f}")
                print()

                results.append(TestResult(
                    test_description=test.description,
                    passed=success,
                    expected=test.expected_output,
                    actual=actual_output,
                    input=test.input,
                    tokens_used=input_tokens + output_tokens,
                    cost=cost,
                    latency=latency
                ))

            except Exception as e:
                print(f"  ✗ ERROR: {e}\n")
                results.append(TestResult(
                    test_description=test.description,
                    passed=False,
                    expected=test.expected_output,
                    actual="",
                    input=test.input,
                    error=str(e)
                ))

        # Summary
        pass_rate = (passed / total) * 100 if total > 0 else 0
        avg_latency = total_latency / passed if passed > 0 else 0

        print(f"{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Passed: {passed}/{total} ({pass_rate:.1f}%)")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Avg Latency: {avg_latency:.2f}s")
        print(f"{'='*80}\n")

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": pass_rate,
            "total_cost": total_cost,
            "avg_latency": avg_latency,
            "results": results
        }

    def _check_match(
        self,
        expected: str,
        actual: str,
        mode: str
    ) -> bool:
        """Check if expected output matches actual output."""
        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        if mode == "contains":
            return expected_lower in actual_lower
        elif mode == "exact":
            return expected_lower == actual_lower
        elif mode == "starts":
            return actual_lower.startswith(expected_lower)
        elif mode == "ends":
            return actual_lower.endswith(expected_lower)
        else:
            raise ValueError(f"Unknown matching mode: {mode}")

    def compare_prompts(
        self,
        prompt1: Callable[[str], str],
        prompt2: Callable[[str], str],
        test_cases: List[TestCase],
        prompt1_name: str = "Prompt A",
        prompt2_name: str = "Prompt B"
    ) -> Dict[str, Any]:
        """
        Compare two different prompts on the same test cases.

        Args:
            prompt1: First prompt template
            prompt2: Second prompt template
            test_cases: Test cases to evaluate
            prompt1_name: Name for first prompt
            prompt2_name: Name for second prompt

        Returns:
            Comparison results
        """
        print(f"\n{'='*80}")
        print(f"PROMPT COMPARISON: {prompt1_name} vs {prompt2_name}")
        print(f"{'='*80}\n")

        print(f"Testing {prompt1_name}...")
        results1 = self.run_test_suite(prompt1, test_cases)

        print(f"\nTesting {prompt2_name}...")
        results2 = self.run_test_suite(prompt2, test_cases)

        # Compare
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")

        print(f"\n{'Metric':<25s} {prompt1_name:>15s} {prompt2_name:>15s} {'Winner':>15s}")
        print("-" * 80)

        # Pass rate
        winner = prompt1_name if results1['pass_rate'] > results2['pass_rate'] else prompt2_name
        if results1['pass_rate'] == results2['pass_rate']:
            winner = "Tie"
        print(f"{'Pass Rate':<25s} {results1['pass_rate']:>14.1f}% {results2['pass_rate']:>14.1f}% {winner:>15s}")

        # Cost
        winner = prompt1_name if results1['total_cost'] < results2['total_cost'] else prompt2_name
        print(f"{'Total Cost':<25s} ${results1['total_cost']:>13.4f} ${results2['total_cost']:>13.4f} {winner:>15s}")

        # Latency
        winner = prompt1_name if results1['avg_latency'] < results2['avg_latency'] else prompt2_name
        print(f"{'Avg Latency':<25s} {results1['avg_latency']:>13.2f}s {results2['avg_latency']:>13.2f}s {winner:>15s}")

        print(f"{'='*80}\n")

        return {
            "prompt1_name": prompt1_name,
            "prompt2_name": prompt2_name,
            "prompt1_results": results1,
            "prompt2_results": results2
        }


# Example usage
if __name__ == "__main__":
    print("""
    ========================================
    Prompt Testing Framework Demo
    ========================================
    Testing log classification prompts
    ========================================
    """)

    # Define test cases
    test_cases = [
        TestCase(
            input="%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down: Dead timer expired",
            expected_output="CRITICAL",
            description="OSPF neighbor down"
        ),
        TestCase(
            input="%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up",
            expected_output="INFO",
            description="Interface up (normal operation)"
        ),
        TestCase(
            input="%SYS-2-MALLOCFAIL: Memory allocation failed for process",
            expected_output="CRITICAL",
            description="Memory allocation failure"
        ),
        TestCase(
            input="%SYS-5-CONFIG_I: Configured from console by admin on vty0",
            expected_output="INFO",
            description="Config change (audit log)"
        ),
        TestCase(
            input="%BGP-3-NOTIFICATION: sent to neighbor 10.2.2.2 (AS 65002) Hold Timer Expired",
            expected_output="ERROR",
            description="BGP notification sent"
        ),
    ]

    # Define prompt template
    def log_classification_prompt(log_entry: str) -> str:
        return f"""Classify this network log into: INFO, WARNING, ERROR, or CRITICAL.

Guidelines:
- INFO: Normal operations, state changes, auditing
- WARNING: Potential issues, non-critical errors
- ERROR: Service-affecting errors, neighbor downs
- CRITICAL: Outages, severe errors, security issues

Log: {log_entry}

Return ONLY the classification, nothing else."""

    # Run tests
    try:
        tester = PromptTester()

        results = tester.run_test_suite(
            prompt_template=log_classification_prompt,
            test_cases=test_cases,
            matching_mode="contains"
        )

        # Show failures
        if results['failed'] > 0:
            print("\nFailed Tests:")
            print("-" * 60)
            for r in results['results']:
                if not r.passed:
                    print(f"  ✗ {r.test_description}")
                    print(f"    Input: {r.input[:60]}...")
                    print(f"    Expected: {r.expected}")
                    print(f"    Got: {r.actual[:60]}...")
                    if r.error:
                        print(f"    Error: {r.error}")
                    print()

        print("✅ Demo complete!")

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tip: Set ANTHROPIC_API_KEY in .env file")
