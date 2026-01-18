"""
Chapter 26: Evaluation & Testing Harness for Network AI
Validate LLM outputs against ground truth and safety rubrics

This module provides comprehensive evaluation of AI-generated network
configurations including exact match, semantic similarity, and safety checks.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


class EvaluationMethod(Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    RUBRIC = "rubric"


class Severity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class EvaluationResult:
    """Complete evaluation result with all metrics."""
    exact_match: bool
    semantic_similarity: float
    rubric_score: int
    rubric_max: int
    rubric_details: List[Dict[str, any]]
    is_safe: bool
    verdict: str
    severity: Severity


class NetworkConfigEvaluator:
    """
    Evaluate AI-generated network configurations.

    Supports:
    - Exact match comparison
    - Semantic similarity (Jaccard)
    - Rubric-based evaluation
    - Vendor syntax validation (Cisco, Juniper, Huawei)
    - Safety checks (ACL, BGP, OSPF)
    """

    def __init__(self):
        self.safety_patterns = {
            'dangerous_permit': re.compile(r'permit\s+any\s+any', re.IGNORECASE),
            'default_route_unfiltered': re.compile(r'default.*route(?!.*route-map)', re.IGNORECASE),
            'telnet_enabled': re.compile(r'transport\s+input\s+telnet', re.IGNORECASE),
            'weak_password': re.compile(r'password\s+(cisco|admin|123)', re.IGNORECASE),
        }

        self.vendor_patterns = {
            'cisco': [
                r'interface\s+\w+',
                r'router\s+(bgp|ospf|eigrp)',
                r'access-list\s+\d+',
                r'ip\s+address\s+\d+\.\d+',
            ],
            'juniper': [
                r'set\s+interfaces',
                r'set\s+protocols\s+(bgp|ospf)',
                r'set\s+routing-options',
                r'set\s+policy-options',
            ],
            'huawei': [
                r'interface\s+\w+',
                r'display\s+',
                r'vlan\s+batch',
                r'ip\s+route-static',
            ]
        }

    def evaluate(
        self,
        model_output: str,
        ground_truth: str,
        criteria: List[str],
        vendor: str = 'cisco'
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of model output.

        Args:
            model_output: AI-generated configuration
            ground_truth: Reference correct configuration
            criteria: Evaluation criteria list
            vendor: Network vendor (cisco, juniper, huawei)

        Returns:
            EvaluationResult with all metrics
        """
        # 1. Exact Match
        exact_match = self._exact_match(model_output, ground_truth)

        # 2. Semantic Similarity
        semantic_sim = self._semantic_similarity(model_output, ground_truth)

        # 3. Rubric Evaluation
        rubric_details = []
        rubric_score = 0
        is_safe = True

        for criterion in criteria:
            passed = self._evaluate_criterion(criterion, model_output, vendor)
            rubric_details.append({
                'criterion': criterion,
                'passed': passed,
                'weight': 1
            })

            if passed:
                rubric_score += 1

            if not passed and any(word in criterion.lower() for word in ['safe', 'security', 'deny']):
                is_safe = False

        # Generate verdict
        verdict, severity = self._generate_verdict(
            exact_match, semantic_sim, rubric_score, len(criteria), is_safe
        )

        return EvaluationResult(
            exact_match=exact_match,
            semantic_similarity=semantic_sim,
            rubric_score=rubric_score,
            rubric_max=len(criteria),
            rubric_details=rubric_details,
            is_safe=is_safe,
            verdict=verdict,
            severity=severity
        )

    def _exact_match(self, text1: str, text2: str) -> bool:
        """Check exact match after normalization."""
        return text1.strip() == text2.strip()

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _evaluate_criterion(self, criterion: str, output: str, vendor: str) -> bool:
        """Evaluate single criterion."""
        criterion_lower = criterion.lower()
        output_lower = output.lower()

        # Safety checks
        if "no 'permit any any'" in criterion_lower or "no permit any" in criterion_lower:
            return not self.safety_patterns['dangerous_permit'].search(output)

        if "default deny" in criterion_lower or "implicit deny" in criterion_lower:
            return any(pattern in output_lower for pattern in ['deny ip any any', 'deny any'])

        if "no telnet" in criterion_lower:
            return not self.safety_patterns['telnet_enabled'].search(output)

        # Format checks
        if "valid json" in criterion_lower:
            return self._is_valid_json(output)

        if f"{vendor} syntax" in criterion_lower:
            return self._validate_vendor_syntax(output, vendor)

        # Content checks
        if "ssh" in criterion_lower and ("deny" in criterion_lower or "block" in criterion_lower):
            return "22" in output or "ssh" in output_lower

        if "bgp" in criterion_lower:
            return "bgp" in output_lower

        if "ospf" in criterion_lower:
            return "ospf" in output_lower

        return True

    def _validate_vendor_syntax(self, config: str, vendor: str) -> bool:
        """Validate vendor-specific syntax."""
        patterns = self.vendor_patterns.get(vendor, [])
        matches = sum(1 for pattern in patterns if re.search(pattern, config, re.IGNORECASE))
        return matches >= 1  # At least one pattern must match

    def _is_valid_json(self, text: str) -> bool:
        """Validate JSON format."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _generate_verdict(
        self,
        exact_match: bool,
        semantic_sim: float,
        rubric_score: int,
        rubric_max: int,
        is_safe: bool
    ) -> Tuple[str, Severity]:
        """Generate human-readable verdict."""

        if not is_safe:
            return (
                "CRITICAL FAILURE: Contains unsafe configuration. DO NOT DEPLOY.",
                Severity.CRITICAL
            )

        if exact_match:
            return (
                "PERFECT: Exact match with ground truth. Safe to deploy.",
                Severity.PASS
            )

        if rubric_score == rubric_max and semantic_sim > 0.8:
            return (
                "PASS: All criteria met. Semantically equivalent to reference.",
                Severity.PASS
            )

        if rubric_score == rubric_max and semantic_sim > 0.6:
            return (
                "PASS: All criteria met. Minor differences from reference.",
                Severity.WARNING
            )

        if rubric_score >= rubric_max * 0.8:
            return (
                f"WARNING: {rubric_score}/{rubric_max} criteria met. Review before deployment.",
                Severity.WARNING
            )

        return (
            f"FAIL: Only {rubric_score}/{rubric_max} criteria met. Needs revision.",
            Severity.FAIL
        )


def example_1_cisco_acl_safety():
    """
    Example 1: Validate Cisco ACL for safety
    """
    print("=" * 60)
    print("Example 1: Cisco ACL Safety Validation")
    print("=" * 60)

    ground_truth = """access-list 101 deny tcp any any eq 22
access-list 101 permit ip 10.0.0.0 0.255.255.255 any
access-list 101 deny ip any any"""

    # Bad output (dangerous)
    bad_output = """access-list 101 permit ip any any"""

    # Good output
    good_output = """access-list 101 deny tcp any any eq 22
access-list 101 permit ip 10.0.0.0 0.255.255.255 any
access-list 101 deny ip any any"""

    criteria = [
        "No 'permit any any' rule",
        "Contains SSH denial (port 22)",
        "Ends with default deny",
        "Valid Cisco syntax"
    ]

    evaluator = NetworkConfigEvaluator()

    print("\n--- Testing Dangerous Config ---")
    result_bad = evaluator.evaluate(bad_output, ground_truth, criteria, vendor='cisco')

    print(f"\nExact Match: {result_bad.exact_match}")
    print(f"Semantic Similarity: {result_bad.semantic_similarity:.2f}")
    print(f"Rubric Score: {result_bad.rubric_score}/{result_bad.rubric_max}")
    print(f"Is Safe: {result_bad.is_safe}")
    print(f"\n{result_bad.verdict}")

    print("\n--- Testing Safe Config ---")
    result_good = evaluator.evaluate(good_output, ground_truth, criteria, vendor='cisco')

    print(f"\nExact Match: {result_good.exact_match}")
    print(f"Semantic Similarity: {result_good.semantic_similarity:.2f}")
    print(f"Rubric Score: {result_good.rubric_score}/{result_good.rubric_max}")
    print(f"Is Safe: {result_good.is_safe}")
    print(f"\n{result_good.verdict}")

    print("\n" + "=" * 60 + "\n")


def example_2_bgp_configuration():
    """
    Example 2: Validate BGP configuration
    """
    print("=" * 60)
    print("Example 2: BGP Configuration Validation")
    print("=" * 60)

    ground_truth = """router bgp 65001
 neighbor 10.1.1.1 remote-as 65002
 neighbor 10.1.1.1 route-map FILTER-IN in
 network 192.168.0.0 mask 255.255.255.0"""

    model_output = """router bgp 65001
 neighbor 10.1.1.1 remote-as 65002
 neighbor 10.1.1.1 route-map FILTER-IN in
 network 192.168.0.0 mask 255.255.255.0"""

    criteria = [
        "Contains BGP configuration",
        "Has neighbor definition",
        "Route filtering applied (route-map)",
        "Network statement present"
    ]

    evaluator = NetworkConfigEvaluator()
    result = evaluator.evaluate(model_output, ground_truth, criteria, vendor='cisco')

    print(f"\nEvaluation Results:")
    print(f"  Exact Match: {'✓' if result.exact_match else '✗'}")
    print(f"  Semantic Similarity: {result.semantic_similarity:.1%}")
    print(f"  Rubric: {result.rubric_score}/{result.rubric_max} criteria met")

    print(f"\nCriterion Breakdown:")
    for detail in result.rubric_details:
        status = "✓ PASS" if detail['passed'] else "✗ FAIL"
        print(f"  {status}: {detail['criterion']}")

    print(f"\nVerdict: {result.verdict}")
    print("\n" + "=" * 60 + "\n")


def example_3_vendor_syntax_validation():
    """
    Example 3: Validate vendor-specific syntax
    """
    print("=" * 60)
    print("Example 3: Multi-Vendor Syntax Validation")
    print("=" * 60)

    cisco_config = """interface GigabitEthernet0/1
 ip address 10.0.0.1 255.255.255.0
 no shutdown"""

    juniper_config = """set interfaces ge-0/0/1 unit 0 family inet address 10.0.0.1/24
set interfaces ge-0/0/1 disable delete"""

    huawei_config = """interface GigabitEthernet0/0/1
 ip address 10.0.0.1 255.255.255.0
 undo shutdown"""

    evaluator = NetworkConfigEvaluator()

    configs = [
        ('cisco', cisco_config),
        ('juniper', juniper_config),
        ('huawei', huawei_config)
    ]

    for vendor, config in configs:
        criteria = [f"Valid {vendor} syntax", "Contains interface configuration"]
        result = evaluator.evaluate(config, config, criteria, vendor=vendor)

        print(f"\n{vendor.upper()} Config:")
        print(f"  Syntax Valid: {'✓' if result.is_safe else '✗'}")
        print(f"  Score: {result.rubric_score}/{result.rubric_max}")

    print("\n" + "=" * 60 + "\n")


def example_4_json_output_validation():
    """
    Example 4: Validate JSON-formatted outputs
    """
    print("=" * 60)
    print("Example 4: JSON Output Validation")
    print("=" * 60)

    ground_truth_json = json.dumps({
        "device": "router-01",
        "interfaces": [
            {"name": "Gi0/1", "ip": "10.0.0.1", "status": "up"},
            {"name": "Gi0/2", "ip": "10.0.0.2", "status": "up"}
        ]
    }, indent=2)

    # Valid JSON
    valid_json = ground_truth_json

    # Invalid JSON
    invalid_json = """{"device": "router-01", "interfaces": [INVALID"""

    criteria = [
        "Valid JSON format",
        "Contains 'device' field",
        "Contains 'interfaces' array"
    ]

    evaluator = NetworkConfigEvaluator()

    print("\n--- Testing Valid JSON ---")
    result_valid = evaluator.evaluate(valid_json, ground_truth_json, criteria)
    print(f"Valid JSON: {result_valid.is_safe}")
    print(f"Score: {result_valid.rubric_score}/{result_valid.rubric_max}")

    print("\n--- Testing Invalid JSON ---")
    result_invalid = evaluator.evaluate(invalid_json, ground_truth_json, criteria)
    print(f"Valid JSON: {result_invalid.is_safe}")
    print(f"Score: {result_invalid.rubric_score}/{result_invalid.rubric_max}")

    print("\n" + "=" * 60 + "\n")


def example_5_llm_generated_config_test():
    """
    Example 5: Test LLM-generated configuration using Claude
    """
    print("=" * 60)
    print("Example 5: Evaluate LLM-Generated Configuration")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM test.")
        print("=" * 60 + "\n")
        return

    prompt = """Generate a secure Cisco ACL that:
1. Denies SSH (port 22) from any source
2. Permits traffic from 10.0.0.0/8 to any destination
3. Ends with an implicit deny all

Output ONLY the ACL configuration, no explanations."""

    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    print("\n🤖 Generating config with Claude...")
    response = llm.invoke(prompt)
    model_output = response.content

    print(f"\nGenerated Config:\n{model_output}\n")

    ground_truth = """access-list 101 deny tcp any any eq 22
access-list 101 permit ip 10.0.0.0 0.255.255.255 any
access-list 101 deny ip any any"""

    criteria = [
        "No 'permit any any' rule",
        "Contains SSH denial (port 22)",
        "Permits 10.0.0.0/8 network",
        "Ends with default deny",
        "Valid Cisco syntax"
    ]

    evaluator = NetworkConfigEvaluator()
    result = evaluator.evaluate(model_output, ground_truth, criteria, vendor='cisco')

    print("\n📊 Evaluation Results:")
    print(f"  Exact Match: {result.exact_match}")
    print(f"  Semantic Similarity: {result.semantic_similarity:.1%}")
    print(f"  Rubric Score: {result.rubric_score}/{result.rubric_max}")
    print(f"  Safety: {'✓ SAFE' if result.is_safe else '✗ UNSAFE'}")

    print(f"\n📋 Detailed Criteria:")
    for detail in result.rubric_details:
        emoji = "✅" if detail['passed'] else "❌"
        print(f"  {emoji} {detail['criterion']}")

    print(f"\n🎯 Final Verdict:")
    print(f"  {result.severity.value.upper()}: {result.verdict}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔬 Chapter 26: Evaluation & Testing Harness")
    print("Validate AI Outputs Against Ground Truth\n")

    try:
        example_1_cisco_acl_safety()
        input("Press Enter to continue...")

        example_2_bgp_configuration()
        input("Press Enter to continue...")

        example_3_vendor_syntax_validation()
        input("Press Enter to continue...")

        example_4_json_output_validation()
        input("Press Enter to continue...")

        example_5_llm_generated_config_test()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Always validate AI outputs before deployment")
        print("- Use rubric-based evaluation for comprehensive testing")
        print("- Safety checks prevent dangerous configurations")
        print("- Semantic similarity catches equivalent but different solutions")
        print("- Vendor syntax validation ensures compatibility\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
