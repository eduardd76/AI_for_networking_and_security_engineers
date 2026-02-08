#!/usr/bin/env python3
"""
Chapter 32: Fine-Tuning Models for Network Data

Production-ready fine-tuning toolkit for network engineering tasks.
Includes data creation, validation, quality assessment, cost calculation, and ROI analysis.

From: AI for Networking Engineers - Volume 3, Chapter 32
Author: Eduard Dulharu
Company: vExpertAI GmbH

Usage:
    python fine_tuning.py
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Data Structures
# ============================================================================

class TaskType(str, Enum):
    """Training task types."""
    TROUBLESHOOTING = "troubleshooting"
    CONFIG_GENERATION = "config_generation"
    LOG_ANALYSIS = "log_analysis"
    SECURITY_AUDIT = "security_audit"
    DOCUMENTATION = "documentation"


class DataQuality(str, Enum):
    """Training data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class TrainingExample:
    """A single training example."""
    system_prompt: str
    user_message: str
    assistant_response: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to training format (JSONL)."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_message},
                {"role": "assistant", "content": self.assistant_response}
            ]
        }

    def estimate_tokens(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        total_chars = len(self.system_prompt) + len(self.user_message) + len(self.assistant_response)
        return total_chars // 4


@dataclass
class ValidationResult:
    """Results from data validation."""
    total_examples: int
    valid_examples: int
    duplicates_removed: int
    incomplete_removed: int
    pii_removed: int
    total_tokens: int
    issues: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of valid examples."""
        if self.total_examples == 0:
            return 0.0
        return (self.valid_examples / self.total_examples) * 100


@dataclass
class CostAnalysis:
    """Fine-tuning cost analysis."""
    training_tokens: int
    fine_tuning_cost: float
    monthly_requests: int
    tokens_without_finetuning: int
    tokens_with_finetuning: int
    input_token_cost_per_1k: float = 0.003
    output_token_cost_per_1k: float = 0.015

    def calculate_roi(self) -> Dict:
        """Calculate complete ROI metrics."""
        # Monthly costs
        cost_without = self._monthly_cost(self.tokens_without_finetuning)
        cost_with = self._monthly_cost(self.tokens_with_finetuning)

        monthly_savings = cost_without - cost_with

        if monthly_savings <= 0:
            return {
                'roi_positive': False,
                'message': 'Fine-tuning would not save money',
                'recommendation': 'STICK WITH PROMPT ENGINEERING'
            }

        breakeven_months = self.fine_tuning_cost / monthly_savings
        annual_savings = monthly_savings * 12
        roi_percentage = ((annual_savings - self.fine_tuning_cost) / self.fine_tuning_cost) * 100

        # Recommendation
        if breakeven_months < 1:
            recommendation = 'FINE-TUNE IMMEDIATELY'
            confidence = 'EXCELLENT'
        elif breakeven_months < 3:
            recommendation = 'FINE-TUNE (GOOD ROI)'
            confidence = 'GOOD'
        elif breakeven_months < 6:
            recommendation = 'MARGINAL - EVALUATE ALTERNATIVES'
            confidence = 'MARGINAL'
        else:
            recommendation = 'STICK WITH PROMPT ENGINEERING'
            confidence = 'POOR'

        return {
            'roi_positive': True,
            'cost_without_finetuning': cost_without,
            'cost_with_finetuning': cost_with,
            'monthly_savings': monthly_savings,
            'breakeven_months': breakeven_months,
            'breakeven_days': breakeven_months * 30,
            'annual_savings': annual_savings,
            'roi_percentage': roi_percentage,
            'recommendation': recommendation,
            'confidence': confidence
        }

    def _monthly_cost(self, avg_tokens_per_request: int) -> float:
        """Calculate monthly API cost."""
        # Assume 50% input, 50% output tokens
        input_tokens = (avg_tokens_per_request / 2) * self.monthly_requests
        output_tokens = (avg_tokens_per_request / 2) * self.monthly_requests

        input_cost = (input_tokens / 1000) * self.input_token_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_token_cost_per_1k

        return input_cost + output_cost


# ============================================================================
# Pydantic Models for LLM Output
# ============================================================================

class QualityIssue(BaseModel):
    """A quality issue in training data."""
    severity: str = Field(description="Severity: critical, high, medium, low")
    issue: str = Field(description="Description of the issue")
    example_id: Optional[int] = Field(description="Which example has the issue")
    recommendation: str = Field(description="How to fix it")


class DataQualityAssessment(BaseModel):
    """LLM assessment of training data quality."""
    overall_quality: str = Field(description="Overall quality: excellent, good, acceptable, poor")
    consistency_score: int = Field(description="Consistency score 1-10")
    accuracy_score: int = Field(description="Accuracy score 1-10")
    diversity_score: int = Field(description="Diversity score 1-10")
    issues: List[QualityIssue] = Field(description="List of quality issues found")
    recommendations: List[str] = Field(description="General recommendations for improvement")


class FineTuningRecommendation(BaseModel):
    """LLM recommendation on whether to fine-tune."""
    should_finetune: bool = Field(description="Whether fine-tuning is recommended")
    confidence: str = Field(description="Confidence level: high, medium, low")
    reasoning: str = Field(description="Explanation of the recommendation")
    alternative_approaches: List[str] = Field(description="Alternative approaches if not fine-tuning")
    estimated_accuracy_improvement: str = Field(description="Expected accuracy improvement percentage")


# ============================================================================
# Training Data Creator
# ============================================================================

class TrainingDataCreator:
    """Create training datasets from various sources."""

    def __init__(self):
        self.examples: List[TrainingExample] = []

    def add_from_ticket(self, problem: str, diagnosis: str, resolution: str,
                       metadata: Optional[Dict] = None) -> None:
        """
        Create training example from troubleshooting ticket.

        Args:
            problem: Problem description from ticket
            diagnosis: Root cause diagnosis
            resolution: How it was resolved
            metadata: Optional metadata (ticket_id, date, etc.)
        """
        system_prompt = (
            "You are a network troubleshooting expert for enterprise networks. "
            "Your task is to diagnose network issues and provide solutions based on "
            "symptoms and diagnostic data. Always provide both diagnosis and resolution steps."
        )

        user_message = f"Troubleshoot this network issue:\n\n{problem}"

        assistant_response = f"""**Diagnosis:**
{diagnosis}

**Resolution:**
{resolution}"""

        example = TrainingExample(
            system_prompt=system_prompt,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )

        self.examples.append(example)

    def add_from_config_pair(self, request: str, config: str, explanation: str,
                            metadata: Optional[Dict] = None) -> None:
        """
        Create training example from config generation request.

        Args:
            request: Natural language config request
            config: Generated configuration
            explanation: Why this config is correct
            metadata: Optional metadata
        """
        system_prompt = (
            "You are a network configuration expert. Generate precise, "
            "production-ready network device configurations following best practices "
            "and security standards. Always include explanations for critical settings."
        )

        user_message = f"Generate configuration:\n\n{request}"

        assistant_response = f"""{config}

# Explanation:
{explanation}"""

        example = TrainingExample(
            system_prompt=system_prompt,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )

        self.examples.append(example)

    def add_from_log_analysis(self, log_entry: str, classification: str,
                             severity: str, action: str,
                             metadata: Optional[Dict] = None) -> None:
        """
        Create training example from log analysis.

        Args:
            log_entry: Raw log entry
            classification: Log classification/category
            severity: Severity level
            action: Recommended action
            metadata: Optional metadata
        """
        system_prompt = (
            "You are a network log analysis expert. Classify log entries, "
            "assess severity, and recommend actions. Be concise and actionable."
        )

        user_message = f"Analyze this log entry:\n\n{log_entry}"

        assistant_response = f"""**Classification:** {classification}
**Severity:** {severity}
**Action:** {action}"""

        example = TrainingExample(
            system_prompt=system_prompt,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )

        self.examples.append(example)

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.examples:
            return {
                'total_examples': 0,
                'total_tokens': 0,
                'avg_tokens_per_example': 0
            }

        total_tokens = sum(ex.estimate_tokens() for ex in self.examples)

        return {
            'total_examples': len(self.examples),
            'total_tokens': total_tokens,
            'avg_tokens_per_example': total_tokens // len(self.examples) if self.examples else 0
        }


# ============================================================================
# Data Validator
# ============================================================================

class DataValidator:
    """Validate and clean training data."""

    def __init__(self):
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b[A-Za-z0-9._%+-]+@(?!example\.com|test\.com)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP Address (consider if PII in your context)'),
            (r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b', 'MAC Address'),
            (r'\b4[0-9]{12}(?:[0-9]{3})?\b', 'Credit Card (Visa)'),
        ]

    def validate_and_clean(self, examples: List[TrainingExample],
                          remove_duplicates: bool = True,
                          check_pii: bool = True) -> Tuple[List[TrainingExample], ValidationResult]:
        """
        Validate and clean training data.

        Args:
            examples: List of training examples
            remove_duplicates: Whether to remove duplicate examples
            check_pii: Whether to check for PII

        Returns:
            Tuple of (cleaned_examples, validation_result)
        """
        result = ValidationResult(
            total_examples=len(examples),
            valid_examples=0,
            duplicates_removed=0,
            incomplete_removed=0,
            pii_removed=0,
            total_tokens=0
        )

        cleaned = examples.copy()

        # Remove duplicates
        if remove_duplicates:
            cleaned, removed = self._remove_duplicates(cleaned)
            result.duplicates_removed = removed

        # Remove incomplete
        cleaned, removed = self._remove_incomplete(cleaned)
        result.incomplete_removed = removed

        # Check for PII
        if check_pii:
            cleaned, removed, pii_issues = self._check_pii(cleaned)
            result.pii_removed = removed
            result.issues.extend(pii_issues)

        # Validate format
        format_issues = self._validate_format(cleaned)
        result.issues.extend(format_issues)

        result.valid_examples = len(cleaned)
        result.total_tokens = sum(ex.estimate_tokens() for ex in cleaned)

        return cleaned, result

    def _remove_duplicates(self, examples: List[TrainingExample]) -> Tuple[List[TrainingExample], int]:
        """Remove duplicate examples based on user message."""
        seen = set()
        unique = []

        for ex in examples:
            key = ex.user_message.strip()
            if key not in seen:
                unique.append(ex)
                seen.add(key)

        return unique, len(examples) - len(unique)

    def _remove_incomplete(self, examples: List[TrainingExample]) -> Tuple[List[TrainingExample], int]:
        """Remove examples with incomplete data."""
        valid = []

        for ex in examples:
            if (ex.system_prompt.strip() and
                ex.user_message.strip() and
                ex.assistant_response.strip() and
                len(ex.assistant_response) > 20):  # Minimum response length
                valid.append(ex)

        return valid, len(examples) - len(valid)

    def _check_pii(self, examples: List[TrainingExample]) -> Tuple[List[TrainingExample], int, List[str]]:
        """Check for PII in examples."""
        clean = []
        issues = []

        for i, ex in enumerate(examples):
            has_pii = False
            pii_found = []

            # Check all content
            all_content = f"{ex.system_prompt} {ex.user_message} {ex.assistant_response}"

            for pattern, pii_type in self.pii_patterns:
                matches = re.findall(pattern, all_content)
                if matches:
                    has_pii = True
                    pii_found.append(f"{pii_type}: {len(matches)} instance(s)")

            if has_pii:
                issues.append(f"Example {i}: Found PII - {', '.join(pii_found)}")
            else:
                clean.append(ex)

        return clean, len(examples) - len(clean), issues

    def _validate_format(self, examples: List[TrainingExample]) -> List[str]:
        """Validate example format."""
        issues = []

        for i, ex in enumerate(examples):
            # Check system prompt
            if len(ex.system_prompt) < 20:
                issues.append(f"Example {i}: System prompt too short")

            # Check user message
            if len(ex.user_message) < 10:
                issues.append(f"Example {i}: User message too short")

            # Check assistant response
            if len(ex.assistant_response) < 20:
                issues.append(f"Example {i}: Assistant response too short")

        return issues


# ============================================================================
# Quality Assessor (LLM-powered)
# ============================================================================

class QualityAssessor:
    """Use LLM to assess training data quality."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def assess_quality(self, examples: List[TrainingExample],
                       sample_size: int = 10) -> DataQualityAssessment:
        """
        Use LLM to assess training data quality.

        Args:
            examples: Training examples to assess
            sample_size: Number of examples to sample for assessment

        Returns:
            Quality assessment from LLM
        """
        # Sample examples
        import random
        sample = random.sample(examples, min(sample_size, len(examples)))

        # Format for LLM
        examples_text = ""
        for i, ex in enumerate(sample, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"System: {ex.system_prompt[:100]}...\n"
            examples_text += f"User: {ex.user_message[:150]}...\n"
            examples_text += f"Assistant: {ex.assistant_response[:150]}...\n"
            examples_text += "-" * 60

        # Create parser
        parser = PydanticOutputParser(pydantic_object=DataQualityAssessment)

        # Create prompt
        template = """You are a machine learning data quality expert. Assess the quality of this training data sample for fine-tuning an LLM.

Training Data Sample ({sample_size} examples from {total_examples} total):
{examples}

Evaluate based on:
1. Consistency: Are examples formatted consistently? Similar style?
2. Accuracy: Are the assistant responses correct and appropriate?
3. Diversity: Do examples cover different scenarios?

{format_instructions}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["examples", "sample_size", "total_examples"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt.format(
            examples=examples_text,
            sample_size=len(sample),
            total_examples=len(examples)
        )

        response = self.llm.invoke(formatted_prompt)
        assessment = parser.parse(response.content)

        return assessment


# ============================================================================
# Cost Calculator
# ============================================================================

class FineTuningCostCalculator:
    """Calculate fine-tuning costs and ROI."""

    def __init__(self,
                 fine_tuning_cost_per_1m_tokens: float = 10.0,
                 epochs: int = 3):
        """
        Initialize calculator.

        Args:
            fine_tuning_cost_per_1m_tokens: Cost per 1M tokens for fine-tuning
            epochs: Number of training epochs
        """
        self.fine_tuning_cost_per_1m = fine_tuning_cost_per_1m_tokens
        self.epochs = epochs

    def calculate_training_cost(self, total_tokens: int) -> float:
        """Calculate one-time training cost."""
        # Training processes data multiple times (epochs)
        training_tokens = total_tokens * self.epochs
        return (training_tokens / 1_000_000) * self.fine_tuning_cost_per_1m

    def create_cost_analysis(self,
                            training_tokens: int,
                            monthly_requests: int,
                            prompt_tokens_without_ft: int,
                            prompt_tokens_with_ft: int,
                            response_tokens: int = 500) -> CostAnalysis:
        """
        Create comprehensive cost analysis.

        Args:
            training_tokens: Total tokens in training data
            monthly_requests: Monthly API requests
            prompt_tokens_without_ft: Avg prompt size without fine-tuning
            prompt_tokens_with_ft: Avg prompt size with fine-tuning
            response_tokens: Avg response size

        Returns:
            CostAnalysis object
        """
        fine_tuning_cost = self.calculate_training_cost(training_tokens)

        # Total tokens per request
        total_without = prompt_tokens_without_ft + response_tokens
        total_with = prompt_tokens_with_ft + response_tokens

        return CostAnalysis(
            training_tokens=training_tokens,
            fine_tuning_cost=fine_tuning_cost,
            monthly_requests=monthly_requests,
            tokens_without_finetuning=total_without,
            tokens_with_finetuning=total_with
        )


# ============================================================================
# ROI Analyzer
# ============================================================================

class ROIAnalyzer:
    """Analyze ROI for fine-tuning decisions."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def get_recommendation(self,
                          cost_analysis: CostAnalysis,
                          current_accuracy: float,
                          target_accuracy: float,
                          task_type: TaskType,
                          data_quality: str) -> FineTuningRecommendation:
        """
        Get LLM recommendation on whether to fine-tune.

        Args:
            cost_analysis: Cost analysis results
            current_accuracy: Current accuracy with prompts (0-100)
            target_accuracy: Target accuracy needed (0-100)
            task_type: Type of task
            data_quality: Quality of training data

        Returns:
            Fine-tuning recommendation
        """
        roi = cost_analysis.calculate_roi()

        # Create parser
        parser = PydanticOutputParser(pydantic_object=FineTuningRecommendation)

        # Create prompt
        template = """You are a machine learning consultant specializing in LLM fine-tuning for production systems.

Evaluate whether fine-tuning is the right approach for this scenario:

**Task Type:** {task_type}
**Current Accuracy:** {current_accuracy}%
**Target Accuracy:** {target_accuracy}%
**Training Data Quality:** {data_quality}

**Cost Analysis:**
- Monthly requests: {monthly_requests:,}
- Fine-tuning cost (one-time): ${fine_tuning_cost:.2f}
- Monthly cost WITHOUT fine-tuning: ${cost_without:.2f}
- Monthly cost WITH fine-tuning: ${cost_with:.2f}
- Break-even: {breakeven_months:.1f} months ({breakeven_days:.0f} days)
- Annual savings: ${annual_savings:.2f}
- ROI: {roi_percentage:.0f}%

**Context:**
- Prompt engineering is free but requires context in every request
- Fine-tuning has upfront cost but reduces prompt length
- Fine-tuning typically improves accuracy by 5-15 percentage points
- Models can overfit to training data if quality is poor

Provide a recommendation with clear reasoning.

{format_instructions}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "task_type", "current_accuracy", "target_accuracy", "data_quality",
                "monthly_requests", "fine_tuning_cost", "cost_without", "cost_with",
                "breakeven_months", "breakeven_days", "annual_savings", "roi_percentage"
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt.format(
            task_type=task_type.value,
            current_accuracy=current_accuracy,
            target_accuracy=target_accuracy,
            data_quality=data_quality,
            monthly_requests=cost_analysis.monthly_requests,
            fine_tuning_cost=cost_analysis.fine_tuning_cost,
            cost_without=roi.get('cost_without_finetuning', 0),
            cost_with=roi.get('cost_with_finetuning', 0),
            breakeven_months=roi.get('breakeven_months', 0),
            breakeven_days=roi.get('breakeven_days', 0),
            annual_savings=roi.get('annual_savings', 0),
            roi_percentage=roi.get('roi_percentage', 0)
        )

        response = self.llm.invoke(formatted_prompt)
        recommendation = parser.parse(response.content)

        return recommendation


# ============================================================================
# Example Functions
# ============================================================================

def example_1_create_training_dataset():
    """
    Example 1: Create Training Dataset from Network Operations

    Shows how to convert tickets, configs, and logs into training data.
    """
    print("\n" + "="*70)
    print("Example 1: Create Training Dataset")
    print("="*70)

    creator = TrainingDataCreator()

    # Add troubleshooting tickets
    print("\nAdding troubleshooting tickets...")

    tickets = [
        {
            'problem': 'Branch office users cannot access datacenter applications. Ping times out to 10.0.0.0/8 network.',
            'diagnosis': 'Primary WAN link (Gi0/0) is down. Traffic routing through backup link but ACL on backup interface (Gi0/1) blocks application ports TCP 8080, 8443.',
            'resolution': 'Updated ACL on Gi0/1 to permit application traffic: "access-list 101 permit tcp any 10.0.0.0 0.255.255.255 eq 8080" and "access-list 101 permit tcp any 10.0.0.0 0.255.255.255 eq 8443". Connectivity restored. Root cause: Incomplete ACL migration when backup link was provisioned. Updated documentation.'
        },
        {
            'problem': 'Switch showing "SPANNING-TREE-2-BLOCK_BPDUGUARD" errors and port Gi0/24 is err-disabled. User cannot connect.',
            'diagnosis': 'PortFast and BPDU Guard enabled on Gi0/24. User mistakenly connected a switch to this access port, triggering BPDU Guard which automatically shut down the port.',
            'resolution': 'Verified Gi0/24 should be access port. Removed user switch. Re-enabled port: "interface Gi0/24", "shutdown", "no shutdown". Added descriptive port description: "description USER_PORT - Building A Floor 2 - NO SWITCHES" to prevent future mistakes.'
        },
        {
            'problem': 'BGP neighbor 203.0.113.5 stuck in Active state, never establishing. "show ip bgp summary" shows Active for 3 hours.',
            'diagnosis': 'BGP neighbor configured with wrong remote-as. Config shows "neighbor 203.0.113.5 remote-as 65002" but actual peer AS is 65003. BGP OPEN message rejected due to AS mismatch in capability negotiation.',
            'resolution': 'Corrected neighbor config: "router bgp 65001", "neighbor 203.0.113.5 remote-as 65003". Session established immediately. Received 150 prefixes as expected. Updated network diagram with correct AS numbers.'
        },
        {
            'problem': 'OSPF adjacency with dist-switch-02 flapping every 30 seconds. "show ip ospf neighbor" shows FULL then DOWN repeatedly.',
            'diagnosis': 'OSPF hello and dead timer mismatch. Local router has default timers (hello 10s, dead 40s). Dist-switch-02 has custom timers (hello 5s, dead 15s). Adjacency forms but fails when dead timer expires on dist-switch-02 side.',
            'resolution': 'Standardized OSPF timers on both devices to company standard (hello 10s, dead 40s): "interface Gi0/1", "ip ospf hello-interval 10", "ip ospf dead-interval 40". Adjacency now stable. Added timer standards to config template.'
        },
        {
            'problem': 'VoIP calls choppy and poor quality on floor 3. Call quality score dropped from 4.2 to 2.1 (MOS score).',
            'diagnosis': 'Access switch on floor 3 (access-sw-03) has no QoS policy applied. Voice traffic (DSCP EF) not prioritized. Competing with data traffic during peak hours (8-10am). Interface GigabitEthernet1/0/1 uplink shows output drops.',
            'resolution': 'Applied standard QoS policy to access-sw-03: "mls qos", "mls qos trust dscp" on all voice VLANs (VLAN 20). Added "priority-queue out" on uplink. Call quality improved to 4.0 MOS immediately. No more output drops observed.'
        }
    ]

    for ticket in tickets:
        creator.add_from_ticket(
            problem=ticket['problem'],
            diagnosis=ticket['diagnosis'],
            resolution=ticket['resolution']
        )

    print(f"  Added {len(tickets)} troubleshooting examples")

    # Add config generation examples
    print("\nAdding config generation examples...")

    configs = [
        {
            'request': 'Configure a secure access port for user devices in VLAN 10',
            'config': '''interface GigabitEthernet0/1
 description USER_PORT - Building A Floor 2
 switchport mode access
 switchport access vlan 10
 switchport port-security
 switchport port-security maximum 3
 switchport port-security violation restrict
 spanning-tree portfast
 spanning-tree bpduguard enable
 no shutdown''',
            'explanation': 'Access port with PortFast for fast link-up, BPDU Guard prevents loops if switch connected, port-security limits MAC addresses to prevent MAC flooding attacks.'
        },
        {
            'request': 'Configure trunk link to distribution switch with specific VLANs',
            'config': '''interface TenGigabitEthernet1/1
 description TRUNK_TO_DIST-SW-01
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 10,20,30,100
 switchport nonegotiate
 no shutdown''',
            'explanation': 'Trunk uses explicit VLAN list (not "vlan all" for security), DTP disabled with "nonegotiate" to prevent negotiation attacks, descriptive naming follows standard.'
        },
        {
            'request': 'Configure BGP peering with ISP using private AS',
            'config': '''router bgp 65001
 bgp log-neighbor-changes
 neighbor 203.0.113.1 remote-as 65000
 neighbor 203.0.113.1 description ISP_PRIMARY
 !
 address-family ipv4
  neighbor 203.0.113.1 activate
  neighbor 203.0.113.1 prefix-list ISP-IN in
  neighbor 203.0.113.1 prefix-list ISP-OUT out
  neighbor 203.0.113.1 maximum-prefix 100000 90 warning-only
 exit-address-family''',
            'explanation': 'BGP with prefix-lists for security (filter what we receive/send), maximum-prefix prevents route table overflow attacks, log neighbor changes for troubleshooting.'
        }
    ]

    for cfg in configs:
        creator.add_from_config_pair(
            request=cfg['request'],
            config=cfg['config'],
            explanation=cfg['explanation']
        )

    print(f"  Added {len(configs)} config generation examples")

    # Add log analysis examples
    print("\nAdding log analysis examples...")

    logs = [
        {
            'entry': '%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired',
            'classification': 'BGP Adjacency Failure',
            'severity': 'CRITICAL',
            'action': 'Check neighbor reachability, verify BGP timers match, check for interface flaps on path to neighbor'
        },
        {
            'entry': '%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up',
            'classification': 'Interface State Change',
            'severity': 'INFO',
            'action': 'Normal operation. Interface came up. Monitor for stability.'
        },
        {
            'entry': '%SEC-6-IPACCESSLOGP: list 100 denied tcp 192.168.1.50(3847) -> 10.0.0.1(22), 1 packet',
            'classification': 'Security ACL Denied Traffic',
            'severity': 'MEDIUM',
            'action': 'Investigate source 192.168.1.50 attempting SSH to 10.0.0.1. Check if authorized. Review ACL 100 for correctness.'
        }
    ]

    for log in logs:
        creator.add_from_log_analysis(
            log_entry=log['entry'],
            classification=log['classification'],
            severity=log['severity'],
            action=log['action']
        )

    print(f"  Added {len(logs)} log analysis examples")

    # Get statistics
    stats = creator.get_statistics()
    print(f"\n{'Dataset Statistics':-^70}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens/example: {stats['avg_tokens_per_example']}")

    return creator


def example_2_validate_and_clean():
    """
    Example 2: Validate and Clean Training Data

    Shows comprehensive data validation including PII detection.
    """
    print("\n" + "="*70)
    print("Example 2: Validate and Clean Training Data")
    print("="*70)

    # Create dataset with some issues
    creator = TrainingDataCreator()

    # Good examples
    creator.add_from_ticket(
        problem="BGP neighbor down",
        diagnosis="Timer mismatch",
        resolution="Fixed timers"
    )

    # Duplicate (same problem)
    creator.add_from_ticket(
        problem="BGP neighbor down",
        diagnosis="Different diagnosis",
        resolution="Different fix"
    )

    # Incomplete (too short)
    creator.examples.append(TrainingExample(
        system_prompt="Short",
        user_message="Too short",
        assistant_response="Bad"
    ))

    # Contains PII
    creator.add_from_ticket(
        problem="User john.doe@company.com cannot connect",
        diagnosis="ACL blocking traffic from 192.168.1.50",
        resolution="Updated ACL"
    )

    # Good example
    creator.add_from_log_analysis(
        log_entry="%SYS-5-CONFIG_I: Configured from console",
        classification="Config change",
        severity="INFO",
        action="Normal operation"
    )

    print(f"\nOriginal dataset: {len(creator.examples)} examples")

    # Validate
    validator = DataValidator()
    cleaned, result = validator.validate_and_clean(creator.examples)

    print(f"\n{'Validation Results':-^70}")
    print(f"Total examples: {result.total_examples}")
    print(f"Valid examples: {result.valid_examples}")
    print(f"Success rate: {result.success_rate:.1f}%")
    print(f"\nRemoved:")
    print(f"  Duplicates: {result.duplicates_removed}")
    print(f"  Incomplete: {result.incomplete_removed}")
    print(f"  Contains PII: {result.pii_removed}")

    if result.issues:
        print(f"\n{'Issues Found':-^70}")
        for issue in result.issues[:5]:  # Show first 5
            print(f"  - {issue}")
        if len(result.issues) > 5:
            print(f"  ... and {len(result.issues) - 5} more")

    print(f"\n{'Final Dataset':-^70}")
    print(f"Clean examples: {len(cleaned)}")
    print(f"Total tokens: {result.total_tokens:,}")

    return cleaned, result


def example_3_roi_analysis():
    """
    Example 3: ROI Analysis for Fine-Tuning Decision

    Shows comprehensive cost-benefit analysis.
    """
    print("\n" + "="*70)
    print("Example 3: ROI Analysis")
    print("="*70)

    calculator = FineTuningCostCalculator(
        fine_tuning_cost_per_1m_tokens=10.0,
        epochs=3
    )

    # Scenario 1: High-volume production use case
    print("\n{'Scenario 1: High-Volume Troubleshooting Assistant':-^70}")

    analysis_high = calculator.create_cost_analysis(
        training_tokens=500_000,  # 500K tokens training data
        monthly_requests=50_000,
        prompt_tokens_without_ft=2_500,  # Large context needed
        prompt_tokens_with_ft=500,  # Model learned context
        response_tokens=500
    )

    roi_high = analysis_high.calculate_roi()

    print(f"\nMonthly Volume: {analysis_high.monthly_requests:,} requests")
    print(f"\nWithout Fine-Tuning:")
    print(f"  Avg tokens/request: {analysis_high.tokens_without_finetuning:,}")
    print(f"  Monthly cost: ${roi_high['cost_without_finetuning']:,.2f}")
    print(f"\nWith Fine-Tuning:")
    print(f"  Avg tokens/request: {analysis_high.tokens_with_finetuning:,}")
    print(f"  Monthly cost: ${roi_high['cost_with_finetuning']:,.2f}")
    print(f"  One-time training cost: ${analysis_high.fine_tuning_cost:,.2f}")
    print(f"\n{'Financial Impact':-^70}")
    print(f"Monthly savings: ${roi_high['monthly_savings']:,.2f}")
    print(f"Annual savings: ${roi_high['annual_savings']:,.2f}")
    print(f"Break-even: {roi_high['breakeven_months']:.1f} months ({roi_high['breakeven_days']:.0f} days)")
    print(f"ROI: {roi_high['roi_percentage']:,.0f}%")
    print(f"\n{'Recommendation':-^70}")
    print(f"{roi_high['recommendation']}")
    print(f"Confidence: {roi_high['confidence']}")

    # Scenario 2: Low-volume use case
    print("\n" + "="*70)
    print("\n{'Scenario 2: Low-Volume Documentation Helper':-^70}")

    analysis_low = calculator.create_cost_analysis(
        training_tokens=200_000,
        monthly_requests=1_000,  # Low volume
        prompt_tokens_without_ft=2_000,
        prompt_tokens_with_ft=400,
        response_tokens=600
    )

    roi_low = analysis_low.calculate_roi()

    print(f"\nMonthly Volume: {analysis_low.monthly_requests:,} requests")
    print(f"\nWithout Fine-Tuning:")
    print(f"  Monthly cost: ${roi_low['cost_without_finetuning']:,.2f}")
    print(f"\nWith Fine-Tuning:")
    print(f"  Monthly cost: ${roi_low['cost_with_finetuning']:,.2f}")
    print(f"  One-time training cost: ${analysis_low.fine_tuning_cost:,.2f}")
    print(f"\n{'Financial Impact':-^70}")
    print(f"Monthly savings: ${roi_low['monthly_savings']:,.2f}")
    print(f"Break-even: {roi_low['breakeven_months']:.1f} months ({roi_low['breakeven_days']:.0f} days)")
    print(f"ROI: {roi_low['roi_percentage']:,.0f}%")
    print(f"\n{'Recommendation':-^70}")
    print(f"{roi_low['recommendation']}")
    print(f"Confidence: {roi_low['confidence']}")

    return analysis_high, analysis_low


def example_4_cost_calculator():
    """
    Example 4: Detailed Cost Calculator

    Shows token-level cost breakdown.
    """
    print("\n" + "="*70)
    print("Example 4: Detailed Cost Calculator")
    print("="*70)

    # Training data scenarios
    scenarios = [
        {
            'name': 'Small Dataset (100 examples)',
            'examples': 100,
            'tokens_per_example': 400,
            'total_tokens': 40_000
        },
        {
            'name': 'Medium Dataset (1,000 examples)',
            'examples': 1_000,
            'tokens_per_example': 400,
            'total_tokens': 400_000
        },
        {
            'name': 'Large Dataset (10,000 examples)',
            'examples': 10_000,
            'tokens_per_example': 400,
            'total_tokens': 4_000_000
        }
    ]

    calculator = FineTuningCostCalculator(
        fine_tuning_cost_per_1m_tokens=10.0,
        epochs=3
    )

    print(f"\n{'Fine-Tuning Training Costs':-^70}")
    print(f"{'Dataset':<35} {'Examples':<12} {'Tokens':<15} {'Cost':<10}")
    print("-" * 70)

    for scenario in scenarios:
        cost = calculator.calculate_training_cost(scenario['total_tokens'])
        print(f"{scenario['name']:<35} {scenario['examples']:<12,} {scenario['total_tokens']:<15,} ${cost:<10.2f}")

    # Monthly usage scenarios
    print(f"\n{'Monthly API Usage Costs (Without Fine-Tuning)':-^70}")
    print(f"{'Volume':<15} {'Tokens/Req':<15} {'Total Tokens':<20} {'Monthly Cost':<15}")
    print("-" * 70)

    volumes = [1_000, 5_000, 10_000, 50_000, 100_000]
    tokens_per_request = 3_000  # Prompt + response

    # Rough cost estimate
    cost_per_1k = 0.009  # Average of input and output

    for volume in volumes:
        total_tokens = volume * tokens_per_request
        monthly_cost = (total_tokens / 1_000) * cost_per_1k
        print(f"{volume:<15,} {tokens_per_request:<15,} {total_tokens:<20,} ${monthly_cost:<15,.2f}")

    # Break-even analysis
    print(f"\n{'Break-Even Analysis':-^70}")
    print("If fine-tuning reduces tokens by 60% (3000 → 1200 tokens/request):")
    print()
    print(f"{'Volume':<15} {'Savings/Month':<20} {'Training Cost':<20} {'Break-Even':<15}")
    print("-" * 70)

    training_cost = 15.0  # Example for medium dataset

    for volume in volumes:
        original_cost = (volume * 3_000 / 1_000) * cost_per_1k
        optimized_cost = (volume * 1_200 / 1_000) * cost_per_1k
        monthly_savings = original_cost - optimized_cost

        if monthly_savings > 0:
            breakeven_months = training_cost / monthly_savings
            breakeven_str = f"{breakeven_months:.1f} months"
        else:
            breakeven_str = "Never"

        print(f"{volume:<15,} ${monthly_savings:<19.2f} ${training_cost:<19.2f} {breakeven_str:<15}")


def example_5_fine_tuning_recommendation():
    """
    Example 5: Get LLM Recommendation on Fine-Tuning

    Uses Claude to provide expert recommendation.
    """
    print("\n" + "="*70)
    print("Example 5: Fine-Tuning Recommendation (LLM-Powered)")
    print("="*70)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not found in environment")
        print("This example requires Claude API access")
        print("\nSet your API key:")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        print("\nSkipping LLM-powered recommendation...")
        return

    # Create cost analysis
    calculator = FineTuningCostCalculator()
    analysis = calculator.create_cost_analysis(
        training_tokens=400_000,
        monthly_requests=25_000,
        prompt_tokens_without_ft=2_000,
        prompt_tokens_with_ft=600,
        response_tokens=400
    )

    # Get LLM recommendation
    print("\nAnalyzing scenario with Claude...")
    analyzer = ROIAnalyzer()

    recommendation = analyzer.get_recommendation(
        cost_analysis=analysis,
        current_accuracy=82.0,
        target_accuracy=95.0,
        task_type=TaskType.TROUBLESHOOTING,
        data_quality="good"
    )

    print(f"\n{'LLM Recommendation':-^70}")
    print(f"\nShould Fine-Tune: {'YES' if recommendation.should_finetune else 'NO'}")
    print(f"Confidence: {recommendation.confidence.upper()}")
    print(f"\n{'Reasoning':-^70}")
    print(recommendation.reasoning)
    print(f"\n{'Expected Improvement':-^70}")
    print(f"Accuracy: {recommendation.estimated_accuracy_improvement}")

    if not recommendation.should_finetune and recommendation.alternative_approaches:
        print(f"\n{'Alternative Approaches':-^70}")
        for i, approach in enumerate(recommendation.alternative_approaches, 1):
            print(f"{i}. {approach}")

    # Also assess data quality
    print("\n" + "="*70)
    print("Assessing Training Data Quality...")
    print("="*70)

    # Create sample dataset
    creator = TrainingDataCreator()
    for i in range(5):
        creator.add_from_ticket(
            problem=f"Network issue {i+1}",
            diagnosis=f"Root cause {i+1}",
            resolution=f"Solution {i+1}"
        )

    assessor = QualityAssessor()
    quality = assessor.assess_quality(creator.examples)

    print(f"\n{'Data Quality Assessment':-^70}")
    print(f"Overall Quality: {quality.overall_quality.upper()}")
    print(f"Consistency Score: {quality.consistency_score}/10")
    print(f"Accuracy Score: {quality.accuracy_score}/10")
    print(f"Diversity Score: {quality.diversity_score}/10")

    if quality.issues:
        print(f"\n{'Issues Found':-^70}")
        for issue in quality.issues[:3]:
            print(f"\n[{issue.severity.upper()}] {issue.issue}")
            print(f"  → {issue.recommendation}")

    if quality.recommendations:
        print(f"\n{'Recommendations':-^70}")
        for i, rec in enumerate(quality.recommendations, 1):
            print(f"{i}. {rec}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Chapter 32: Fine-Tuning Models for Network Data")
    print("="*70)
    print("\nProduction-Ready Fine-Tuning Toolkit")
    print("Created by: Eduard Dulharu (vExpertAI GmbH)")
    print("\n5 Complete Examples:")
    print("  1. Create training dataset from tickets/configs/logs")
    print("  2. Validate and clean data (PII removal, duplicates)")
    print("  3. ROI analysis (high-volume vs low-volume)")
    print("  4. Cost calculator with break-even analysis")
    print("  5. LLM-powered recommendation engine")

    try:
        # Example 1: Create dataset
        creator = example_1_create_training_dataset()
        input("\nPress Enter to continue...")

        # Example 2: Validate
        cleaned, result = example_2_validate_and_clean()
        input("\nPress Enter to continue...")

        # Example 3: ROI analysis
        high_vol, low_vol = example_3_roi_analysis()
        input("\nPress Enter to continue...")

        # Example 4: Cost calculator
        example_4_cost_calculator()
        input("\nPress Enter to continue...")

        # Example 5: LLM recommendation
        example_5_fine_tuning_recommendation()

        # Summary
        print("\n" + "="*70)
        print(" Summary: Key Takeaways")
        print("="*70)
        print("\n1. Fine-tuning makes sense for high-volume, repetitive tasks")
        print("   → Break-even: <3 months = Good ROI")
        print("   → >50K requests/month = Strong candidate")

        print("\n2. Data quality matters more than quantity")
        print("   → 500 excellent examples > 5,000 poor examples")
        print("   → Always validate and clean before training")

        print("\n3. Calculate ROI before committing")
        print("   → One-time training cost: $10-200 typically")
        print("   → Monthly savings from reduced tokens")
        print("   → Factor in retraining costs")

        print("\n4. Typical improvements:")
        print("   → Accuracy: +5-15 percentage points")
        print("   → Token usage: -50-70%")
        print("   → Latency: -20-40%")
        print("   → Cost: -30-50% monthly")

        print("\n5. When NOT to fine-tune:")
        print("   → Low volume (<1K requests/month)")
        print("   → Frequently changing requirements")
        print("   → Already achieving target accuracy")
        print("   → Limited training data (<500 examples)")

        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
