# Chapter 32: Fine-Tuning Models for Network Data

## Introduction

You've been using Claude with prompt engineering for months. It works, but you notice patterns: you keep explaining the same network topology, the same vendor-specific syntax, the same operational procedures. Every API call includes 2,000 tokens of context just to tell the model what a "normal" config looks like in your environment.

**Fine-tuning** teaches the model your network's patterns once, permanently. After fine-tuning, the model already knows your topology, your naming conventions, your config standards. You skip 2,000 tokens of context per request. Faster responses, lower costs, better accuracy.

**The Problem**: Should you fine-tune or stick with prompt engineering? Fine-tuning costs $200-500 upfront. At 50,000 requests/month, you save $300/month in token costs—breaks even in 20 days. At 1,000 requests/month, you save $6/month—breaks even in 3 years. The decision is math, not gut feel.

This chapter builds four versions:
- **V1: ROI Calculator** - Decide if fine-tuning makes financial sense ($0)
- **V2: Training Dataset** - Convert your tickets and configs into training data ($0)
- **V3: Fine-Tuned Model** - Launch AWS Bedrock fine-tuning job ($200 one-time)
- **V4: Benchmark & Deploy** - Measure accuracy gains, deploy in production ($10 testing)

**What You'll Learn**:
- Calculate break-even point for your scenario (V1)
- Create high-quality training data from tickets and configs (V2)
- Fine-tune Claude via AWS Bedrock API (V3)
- Benchmark performance improvements and deploy (V4)

**Prerequisites**: Chapters 5 (Prompt Engineering), 8 (Cost Optimization), 14 (RAG Fundamentals)

---

## When to Fine-Tune: The Decision Factors

Fine-tuning makes sense when **all three** conditions are true:

1. **High volume** (>10,000 requests/month)
   - Same context repeated in every prompt
   - Token savings amortize the upfront cost

2. **Repetitive context** (>1,000 tokens per request)
   - Network topology, naming conventions, config standards
   - Context that doesn't change between requests

3. **Accuracy matters** (need >90%)
   - Critical operations where errors are expensive
   - Base model + prompts isn't good enough

**Don't fine-tune** when:
- Low volume (<1,000 requests/month) - prompt engineering is cheaper
- Requirements change frequently - fine-tuning takes weeks, prompts take minutes
- Already accurate (>95% with prompts) - diminishing returns
- Limited training data (<500 examples) - RAG with few-shot prompts works better

**Decision Matrix:**

| Scenario | Volume/Month | Context Tokens | Accuracy Need | Recommendation |
|----------|--------------|----------------|---------------|----------------|
| Config parsing | 100K+ | 2,000 | 98%+ | **Fine-tune** |
| Log analysis | 50K | 1,500 | 90%+ | **Fine-tune** |
| Troubleshooting | 5K | 1,000 | 85% OK | **Prompt engineering** |
| Documentation | 500 | 500 | 80% OK | **Prompt engineering** |
| Security audit | 10K | 2,000 | 99%+ | **Fine-tune** |

The rest of this chapter shows you how to calculate ROI for your scenario, then walks through the complete fine-tuning process if the numbers make sense.

---

## Version 1: ROI Calculator

**Goal**: Calculate break-even point to decide if fine-tuning makes financial sense.

**What you'll build**: Python tool that compares monthly costs with and without fine-tuning.

**Time**: 45 minutes

**Cost**: $0 (calculations only)

### The Math

**Without fine-tuning**:
- Average prompt: 2,500 tokens (includes 2,000 context + 500 actual query)
- Average response: 500 tokens
- Cost per request: (2,500 × $3/M input) + (500 × $15/M output) = $0.015

**With fine-tuning**:
- Average prompt: 500 tokens (context built into model)
- Average response: 500 tokens
- Cost per request: (500 × $3/M input) + (500 × $15/M output) = $0.009
- One-time fine-tuning: $200

**Break-even**: $200 / ($0.015 - $0.009) = 33,333 requests

At 50,000 requests/month → breaks even in 20 days
At 1,000 requests/month → breaks even in 33 months

### Implementation

```python
"""
Fine-Tuning ROI Calculator
File: roi_calculator.py

Calculate break-even point to decide if fine-tuning makes financial sense.
"""
from anthropic import Anthropic
from typing import Dict
import os


class FineTuningROI:
    """Calculate ROI for fine-tuning decisions."""

    # Claude Sonnet 4 pricing (per million tokens)
    INPUT_COST = 3.00   # $3/M input tokens
    OUTPUT_COST = 15.00  # $15/M output tokens

    def __init__(self,
                 monthly_requests: int,
                 prompt_tokens_without_finetuning: int,
                 prompt_tokens_with_finetuning: int,
                 response_tokens: int,
                 fine_tuning_cost: float = 200.0):
        """
        Args:
            monthly_requests: Number of API requests per month
            prompt_tokens_without_finetuning: Avg prompt size including context
            prompt_tokens_with_finetuning: Avg prompt size (context in model)
            response_tokens: Avg response size
            fine_tuning_cost: One-time cost to fine-tune
        """
        self.monthly_requests = monthly_requests
        self.prompt_without = prompt_tokens_without_finetuning
        self.prompt_with = prompt_tokens_with_finetuning
        self.response = response_tokens
        self.fine_tuning_cost = fine_tuning_cost

    def calculate_monthly_cost(self, use_finetuning: bool) -> float:
        """Calculate monthly API cost."""
        prompt_tokens = self.prompt_with if use_finetuning else self.prompt_without

        # Cost = (input_tokens / 1M × $3) + (output_tokens / 1M × $15)
        input_cost = (self.monthly_requests * prompt_tokens / 1_000_000) * self.INPUT_COST
        output_cost = (self.monthly_requests * self.response / 1_000_000) * self.OUTPUT_COST

        return input_cost + output_cost

    def calculate_roi(self) -> Dict:
        """
        Calculate complete ROI analysis.

        Returns:
            Dict with cost comparison and break-even analysis
        """
        cost_without = self.calculate_monthly_cost(use_finetuning=False)
        cost_with = self.calculate_monthly_cost(use_finetuning=True)

        monthly_savings = cost_without - cost_with

        if monthly_savings <= 0:
            return {
                'roi_positive': False,
                'recommendation': 'DONT_FINETUNE',
                'reason': 'No monthly savings - prompt engineering is already optimal'
            }

        # Calculate break-even
        breakeven_months = self.fine_tuning_cost / monthly_savings
        annual_savings = monthly_savings * 12
        roi_percentage = ((annual_savings - self.fine_tuning_cost) / self.fine_tuning_cost) * 100

        # Determine recommendation
        if breakeven_months < 1:
            recommendation = 'FINETUNE_NOW'
            reason = 'Excellent ROI - breaks even in less than 1 month'
        elif breakeven_months < 3:
            recommendation = 'FINETUNE'
            reason = 'Good ROI - breaks even in under 3 months'
        elif breakeven_months < 6:
            recommendation = 'EVALUATE'
            reason = 'Marginal ROI - consider if volume is growing'
        else:
            recommendation = 'DONT_FINETUNE'
            reason = f'Poor ROI - takes {breakeven_months:.1f} months to break even'

        return {
            'roi_positive': monthly_savings > 0,
            'cost_without_finetuning': cost_without,
            'cost_with_finetuning': cost_with,
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'breakeven_months': breakeven_months,
            'breakeven_days': breakeven_months * 30,
            'roi_percentage': roi_percentage,
            'recommendation': recommendation,
            'reason': reason
        }

    def print_analysis(self):
        """Print detailed ROI analysis."""
        roi = self.calculate_roi()

        print("\n" + "="*70)
        print("FINE-TUNING ROI ANALYSIS")
        print("="*70)

        print(f"\nScenario:")
        print(f"  Monthly requests: {self.monthly_requests:,}")
        print(f"  Prompt tokens (without fine-tuning): {self.prompt_without}")
        print(f"  Prompt tokens (with fine-tuning): {self.prompt_with}")
        print(f"  Response tokens: {self.response}")

        print(f"\nCosts:")
        print(f"  Without fine-tuning: ${roi['cost_without_finetuning']:,.2f}/month")
        print(f"  With fine-tuning: ${roi['cost_with_finetuning']:,.2f}/month")
        print(f"  One-time fine-tuning cost: ${self.fine_tuning_cost:,.2f}")

        if roi['roi_positive']:
            print(f"\nSavings:")
            print(f"  Monthly: ${roi['monthly_savings']:,.2f}")
            print(f"  Annual: ${roi['annual_savings']:,.2f}")

            print(f"\nBreak-Even:")
            print(f"  {roi['breakeven_months']:.1f} months ({roi['breakeven_days']:.0f} days)")

            print(f"\nROI: {roi['roi_percentage']:.0f}%")

        print(f"\nRecommendation: {roi['recommendation']}")
        print(f"  {roi['reason']}")
        print("="*70)


# Example Usage
if __name__ == "__main__":
    print("SCENARIO 1: High-Volume Troubleshooting Assistant")
    print("-" * 70)

    calculator_high = FineTuningROI(
        monthly_requests=50_000,
        prompt_tokens_without_finetuning=2_500,  # 2K context + 500 query
        prompt_tokens_with_finetuning=500,        # Just query
        response_tokens=500
    )
    calculator_high.print_analysis()

    print("\n\nSCENARIO 2: Low-Volume Documentation Assistant")
    print("-" * 70)

    calculator_low = FineTuningROI(
        monthly_requests=1_000,
        prompt_tokens_without_finetuning=2_500,
        prompt_tokens_with_finetuning=500,
        response_tokens=500
    )
    calculator_low.print_analysis()
```

### Example Output

```
SCENARIO 1: High-Volume Troubleshooting Assistant
----------------------------------------------------------------------

======================================================================
FINE-TUNING ROI ANALYSIS
======================================================================

Scenario:
  Monthly requests: 50,000
  Prompt tokens (without fine-tuning): 2,500
  Prompt tokens (with fine-tuning): 500
  Response tokens: 500

Costs:
  Without fine-tuning: $750.00/month
  With fine-tuning: $450.00/month
  One-time fine-tuning cost: $200.00

Savings:
  Monthly: $300.00
  Annual: $3,600.00

Break-Even:
  0.7 months (20 days)

ROI: 1700%

Recommendation: FINETUNE_NOW
  Excellent ROI - breaks even in less than 1 month
======================================================================


SCENARIO 2: Low-Volume Documentation Assistant
----------------------------------------------------------------------

======================================================================
FINE-TUNING ROI ANALYSIS
======================================================================

Scenario:
  Monthly requests: 1,000
  Prompt tokens (without fine-tuning): 2,500
  Prompt tokens (with fine-tuning): 500
  Response tokens: 500

Costs:
  Without fine-tuning: $15.00/month
  With fine-tuning: $9.00/month
  One-time fine-tuning cost: $200.00

Savings:
  Monthly: $6.00
  Annual: $72.00

Break-Even:
  33.3 months (1000 days)

ROI: -64%

Recommendation: DONT_FINETUNE
  Poor ROI - takes 33.3 months to break even
======================================================================
```

### What Just Happened

The ROI calculator shows the exact break-even point for your scenario:

**High volume (50K requests/month)**:
- Saves $300/month in token costs
- Breaks even in 20 days
- ROI: 1,700% in first year
- **Decision: Fine-tune immediately**

**Low volume (1K requests/month)**:
- Saves $6/month in token costs
- Breaks even in 33 months (nearly 3 years)
- ROI: -64% in first year
- **Decision: Stick with prompt engineering**

The math makes the decision obvious. If your break-even is under 3 months, proceed to V2. If it's over 6 months, stop here and optimize your prompts instead.

**Cost**: $0 (calculations only)

---

## Version 2: Training Dataset

**Goal**: Convert historical tickets and configs into high-quality training data.

**What you'll build**: Python tool that generates train.jsonl and validation.jsonl files ready for fine-tuning.

**Time**: 60 minutes

**Cost**: $0 (data preparation only)

### Training Data Requirements

Fine-tuning needs **500+ high-quality examples** in this format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a network troubleshooting expert for Acme Corp."},
    {"role": "user", "content": "BGP neighbor stuck in Active state"},
    {"role": "assistant", "content": "Diagnosis: Remote AS mismatch...\nResolution: Correct neighbor config..."}
  ]
}
```

Each example teaches the model one task. Quality matters more than quantity:
- **Accurate outputs** - Model learns from these, mistakes become permanent
- **Diverse scenarios** - Cover different device types, problems, configs
- **Consistent format** - Same style across all examples
- **No PII** - Remove IP addresses, usernames, sensitive data

### Implementation

```python
"""
Training Data Generator
File: training_data_generator.py

Convert troubleshooting tickets and config standards into fine-tuning dataset.
"""
import json
import re
import random
from typing import List, Dict


class TrainingDataGenerator:
    """Generate training data from network operations."""

    def __init__(self, system_prompt: str):
        """
        Args:
            system_prompt: The system message for all examples
        """
        self.system_prompt = system_prompt
        self.training_examples = []

    def add_troubleshooting_ticket(self,
                                   problem: str,
                                   diagnosis: str,
                                   resolution: str):
        """
        Add a troubleshooting ticket as training example.

        Args:
            problem: The problem description
            diagnosis: Root cause analysis
            resolution: Solution and commands used
        """
        example = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"""**Diagnosis:**
{diagnosis}

**Resolution:**
{resolution}"""
                }
            ]
        }

        self.training_examples.append(example)

    def add_config_standard(self,
                           config_type: str,
                           correct_config: str,
                           explanation: str):
        """
        Add a config standard as training example.

        Args:
            config_type: Type of config (interface, routing, etc.)
            correct_config: Example of standard config
            explanation: Why this is the standard
        """
        example = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Generate a standard {config_type} configuration."},
                {
                    "role": "assistant",
                    "content": f"""{correct_config}

# Standard Explanation:
{explanation}"""
                }
            ]
        }

        self.training_examples.append(example)

    def add_config_validation(self,
                             config: str,
                             issues: List[str],
                             corrected_config: str):
        """
        Add a config validation example.

        Args:
            config: Config with issues
            issues: List of problems found
            corrected_config: Fixed config
        """
        issues_text = "\n".join([f"- {issue}" for issue in issues])

        example = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Validate and fix this configuration:\n\n{config}"},
                {
                    "role": "assistant",
                    "content": f"""**Issues Found:**
{issues_text}

**Corrected Configuration:**
{corrected_config}"""
                }
            ]
        }

        self.training_examples.append(example)

    def validate_and_clean(self):
        """
        Validate training data quality.

        Removes:
        - Duplicate examples
        - Examples with incomplete data
        - Examples with potential PII
        """
        print(f"\nValidating {len(self.training_examples)} examples...")

        # Remove duplicates
        unique_examples = []
        seen_prompts = set()

        for example in self.training_examples:
            prompt = example['messages'][1]['content']
            if prompt not in seen_prompts:
                unique_examples.append(example)
                seen_prompts.add(prompt)

        removed_dupes = len(self.training_examples) - len(unique_examples)
        print(f"  ✓ Removed {removed_dupes} duplicates")

        # Remove incomplete examples
        valid_examples = []
        for example in unique_examples:
            user_msg = example['messages'][1]['content']
            assistant_msg = example['messages'][2]['content']

            # Must have both user input and assistant response
            if len(user_msg) > 10 and len(assistant_msg) > 20:
                valid_examples.append(example)

        removed_incomplete = len(unique_examples) - len(valid_examples)
        print(f"  ✓ Removed {removed_incomplete} incomplete examples")

        # Check for PII (simplified - use real PII detection in production)
        clean_examples = []
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\bpassword\s*[:=]\s*\S+',  # Passwords
        ]

        for example in valid_examples:
            has_pii = False
            for msg in example['messages']:
                for pattern in pii_patterns:
                    if re.search(pattern, msg['content'], re.IGNORECASE):
                        has_pii = True
                        break
                if has_pii:
                    break

            if not has_pii:
                clean_examples.append(example)

        removed_pii = len(valid_examples) - len(clean_examples)
        print(f"  ✓ Removed {removed_pii} examples with potential PII")

        self.training_examples = clean_examples
        print(f"\n✓ Final dataset: {len(self.training_examples)} clean examples")

    def split_train_validation(self, train_ratio: float = 0.9):
        """
        Split dataset into training and validation sets.

        Args:
            train_ratio: Ratio for training (0.9 = 90% train, 10% validation)

        Returns:
            Tuple of (train_examples, validation_examples)
        """
        # Shuffle
        random.shuffle(self.training_examples)

        # Split
        split_point = int(len(self.training_examples) * train_ratio)
        train_examples = self.training_examples[:split_point]
        val_examples = self.training_examples[split_point:]

        print(f"\nSplit dataset:")
        print(f"  Training: {len(train_examples)} examples (90%)")
        print(f"  Validation: {len(val_examples)} examples (10%)")

        return train_examples, val_examples

    def export_to_jsonl(self, train_file: str, validation_file: str):
        """
        Export training data to JSONL files.

        Args:
            train_file: Output file for training data
            validation_file: Output file for validation data
        """
        train_examples, val_examples = self.split_train_validation()

        # Write training set
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Exported {len(train_examples)} training examples to {train_file}")

        # Write validation set
        with open(validation_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Exported {len(val_examples)} validation examples to {validation_file}")

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.training_examples:
            return {}

        # Calculate token counts (rough estimate: 1 token ≈ 4 chars)
        total_tokens = 0
        for example in self.training_examples:
            for msg in example['messages']:
                total_tokens += len(msg['content']) // 4

        return {
            'total_examples': len(self.training_examples),
            'estimated_tokens': total_tokens,
            'avg_tokens_per_example': total_tokens // len(self.training_examples),
            'estimated_cost': self._estimate_fine_tuning_cost(total_tokens)
        }

    def _estimate_fine_tuning_cost(self, total_tokens: int) -> float:
        """
        Estimate fine-tuning cost.

        AWS Bedrock pricing: ~$10 per 1M training tokens (example rate)
        Actual pricing varies by provider and model.
        """
        return (total_tokens / 1_000_000) * 10


# Example Usage
if __name__ == "__main__":
    print("="*70)
    print("TRAINING DATA GENERATOR")
    print("="*70)

    # Initialize generator
    generator = TrainingDataGenerator(
        system_prompt="You are a network troubleshooting expert for Acme Corp. Diagnose issues and provide solutions following company standards."
    )

    # Add troubleshooting tickets
    print("\nAdding troubleshooting tickets...")

    generator.add_troubleshooting_ticket(
        problem="Branch office users cannot access datacenter applications. Ping times out.",
        diagnosis="Primary WAN link (Gi0/0) is down. Traffic routing through backup link but ACL on backup interface blocks application ports.",
        resolution="Updated ACL on Gi0/1 to permit application traffic (TCP 8080, 8443). Connectivity restored. Root cause: Incomplete ACL migration when backup link was provisioned."
    )

    generator.add_troubleshooting_ticket(
        problem="Switch showing SPANNING-TREE-2-BLOCK_BPDUGUARD errors and port Gi0/24 is err-disabled.",
        diagnosis="PortFast and BPDU Guard enabled on Gi0/24. A switch was connected to this access port, triggering BPDU Guard.",
        resolution="Confirmed Gi0/24 should be access port. Re-enabled port with 'shutdown' then 'no shutdown'. Added port description to prevent future mistakes."
    )

    generator.add_troubleshooting_ticket(
        problem="BGP neighbor 203.0.113.5 stuck in Active state, never establishing.",
        diagnosis="Neighbor configured with wrong remote AS. Config shows 'neighbor 203.0.113.5 remote-as 65002' but actual peer AS is 65003. BGP open message rejected due to AS mismatch.",
        resolution="Corrected neighbor config: 'neighbor 203.0.113.5 remote-as 65003'. Session established immediately. Received 150 prefixes as expected."
    )

    generator.add_troubleshooting_ticket(
        problem="OSPF adjacency with dist-switch-02 flapping every 30 seconds.",
        diagnosis="OSPF hello and dead timers mismatch. Router has default timers (hello 10s, dead 40s) but dist-switch-02 has custom timers (hello 5s, dead 15s). Adjacency fails when dead timer expires.",
        resolution="Standardized OSPF timers on both devices to hello 10s, dead 40s. Adjacency stable. Added to config standards document."
    )

    generator.add_troubleshooting_ticket(
        problem="VoIP calls are choppy and have poor quality on floor 3.",
        diagnosis="Access switch on floor 3 has no QoS policy. Voice traffic (marked DSCP EF) not prioritized, competing with data traffic during peak hours.",
        resolution="Applied standard QoS policy to access switch: 'service-policy input QOS-TRUST-DSCP' and 'mls qos trust dscp' on voice VLANs. Call quality improved immediately."
    )

    # Add config standards
    print("Adding config standards...")

    generator.add_config_standard(
        config_type="access port",
        correct_config="""interface GigabitEthernet0/1
 description USER_PORT - Building A Floor 2
 switchport mode access
 switchport access vlan 10
 spanning-tree portfast
 spanning-tree bpduguard enable
 no shutdown""",
        explanation="All access ports use PortFast, BPDU Guard for security, and descriptive naming: USER_PORT - <location>"
    )

    generator.add_config_standard(
        config_type="trunk port",
        correct_config="""interface TenGigabitEthernet1/1
 description TRUNK_TO_DIST-SW-01
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 10,20,30,100
 switchport nonegotiate
 no shutdown""",
        explanation="Trunk ports use explicit VLAN lists (no 'vlan all'), DTP disabled with 'nonegotiate', and descriptive naming: TRUNK_TO_<destination>"
    )

    # Add config validation examples
    print("Adding config validation examples...")

    generator.add_config_validation(
        config="""interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10
 no shutdown""",
        issues=[
            "Missing description (required for all interfaces)",
            "Missing PortFast (required for access ports)",
            "Missing BPDU Guard (required for access ports)"
        ],
        corrected_config="""interface GigabitEthernet0/1
 description USER_PORT - Location TBD
 switchport mode access
 switchport access vlan 10
 spanning-tree portfast
 spanning-tree bpduguard enable
 no shutdown"""
    )

    print(f"✓ Added {len(generator.training_examples)} examples")

    # Validate and clean
    generator.validate_and_clean()

    # Get statistics
    stats = generator.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Avg tokens/example: {stats['avg_tokens_per_example']}")
    print(f"  Estimated fine-tuning cost: ${stats['estimated_cost']:.2f}")

    # Export to files
    print("\nExporting to JSONL files...")
    generator.export_to_jsonl('train.jsonl', 'validation.jsonl')

    print("\n" + "="*70)
    print("✓ Training data ready for fine-tuning")
    print("="*70)
```

### Example Output

```
======================================================================
TRAINING DATA GENERATOR
======================================================================

Adding troubleshooting tickets...
Adding config standards...
Adding config validation examples...
✓ Added 7 examples

Validating 7 examples...
  ✓ Removed 0 duplicates
  ✓ Removed 0 incomplete examples
  ✓ Removed 0 examples with potential PII

✓ Final dataset: 7 clean examples

Dataset Statistics:
  Total examples: 7
  Estimated tokens: 2,625
  Avg tokens/example: 375
  Estimated fine-tuning cost: $0.03

Exporting to JSONL files...

Split dataset:
  Training: 6 examples (90%)
  Validation: 1 examples (10%)
✓ Exported 6 training examples to train.jsonl
✓ Exported 1 validation examples to validation.jsonl

======================================================================
✓ Training data ready for fine-tuning
======================================================================
```

### What Just Happened

You converted operational data into training format:

1. **Troubleshooting tickets** → Examples showing problem → diagnosis → resolution
2. **Config standards** → Examples showing how to generate compliant configs
3. **Config validation** → Examples showing how to find and fix issues

The validator removed:
- Duplicates (same problem multiple times)
- Incomplete examples (missing diagnosis or resolution)
- PII (emails, passwords, sensitive data)

The splitter created:
- `train.jsonl` (90% of data) - Model learns from these
- `validation.jsonl` (10% of data) - Used to check for overfitting

**Real-world scale**: This example shows 7 examples for demonstration. Production fine-tuning needs **500+ examples** for good results. Add more tickets from your ticketing system, more config standards from your documentation, more validation examples from change requests.

**Cost**: $0 (data preparation only)

---

## Version 3: Fine-Tuned Model

**Goal**: Launch AWS Bedrock fine-tuning job and get custom model ID.

**What you'll build**: Python script that uploads training data to S3 and creates fine-tuning job.

**Time**: 90 minutes (your time) + 2-6 hours (training time)

**Cost**: $200 one-time (example: 1M training tokens × 3 epochs)

### AWS Bedrock Setup

Before running the fine-tuning script:

1. **AWS Account**: Sign up at aws.amazon.com
2. **Bedrock Access**: Request access to Claude models in AWS Bedrock console
3. **IAM Role**: Create role with permissions:
   - `bedrock:CreateModelCustomizationJob`
   - `bedrock:GetModelCustomizationJob`
   - `s3:PutObject`, `s3:GetObject`
4. **S3 Bucket**: Create bucket for training data (e.g., `my-bedrock-training-data`)

### Implementation

```python
"""
AWS Bedrock Fine-Tuning
File: bedrock_fine_tune.py

Upload training data and launch fine-tuning job via AWS Bedrock.
"""
import boto3
import json
import time
from typing import Dict


class BedrockFineTuner:
    """Fine-tune Claude models via AWS Bedrock."""

    def __init__(self, region: str = 'us-east-1'):
        """
        Args:
            region: AWS region for Bedrock
        """
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.region = region

    def upload_training_data(self,
                            local_file: str,
                            bucket: str,
                            s3_key: str) -> str:
        """
        Upload training data to S3.

        Args:
            local_file: Path to local JSONL file
            bucket: S3 bucket name
            s3_key: S3 key (path) for the file

        Returns:
            S3 URI (s3://bucket/key)
        """
        print(f"Uploading {local_file} to S3...")

        self.s3.upload_file(local_file, bucket, s3_key)

        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"✓ Uploaded to {s3_uri}")

        return s3_uri

    def create_fine_tuning_job(self,
                              model_id: str,
                              training_data_uri: str,
                              validation_data_uri: str,
                              job_name: str,
                              output_bucket: str,
                              iam_role_arn: str,
                              hyperparameters: Dict = None) -> str:
        """
        Create fine-tuning job in AWS Bedrock.

        Args:
            model_id: Base model (e.g., 'anthropic.claude-3-haiku-20240307-v1:0')
            training_data_uri: S3 URI to train.jsonl
            validation_data_uri: S3 URI to validation.jsonl
            job_name: Name for this fine-tuning job
            output_bucket: S3 bucket for output model
            iam_role_arn: IAM role ARN with Bedrock permissions
            hyperparameters: Optional training hyperparameters

        Returns:
            Job ARN
        """
        print(f"\nCreating fine-tuning job: {job_name}")
        print(f"  Base model: {model_id}")
        print(f"  Training data: {training_data_uri}")
        print(f"  Validation data: {validation_data_uri}")

        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'epochCount': '3',              # Number of training epochs
                'batchSize': '8',               # Batch size
                'learningRate': '0.00001'       # Learning rate
            }

        try:
            response = self.bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=f"{job_name}-model",
                roleArn=iam_role_arn,
                baseModelIdentifier=model_id,
                trainingDataConfig={
                    's3Uri': training_data_uri
                },
                validationDataConfig={
                    's3Uri': validation_data_uri
                },
                outputDataConfig={
                    's3Uri': f"s3://{output_bucket}/models/{job_name}"
                },
                hyperParameters=hyperparameters
            )

            job_arn = response['jobArn']
            print(f"✓ Fine-tuning job created: {job_arn}")
            print(f"\nThis will take 2-6 hours. Check status with get_job_status().")

            return job_arn

        except Exception as e:
            print(f"✗ Error creating fine-tuning job: {e}")
            raise

    def get_job_status(self, job_arn: str) -> Dict:
        """
        Get current status of fine-tuning job.

        Args:
            job_arn: Job ARN from create_fine_tuning_job

        Returns:
            Dict with status and metrics
        """
        response = self.bedrock.get_model_customization_job(
            jobIdentifier=job_arn
        )

        return {
            'status': response['status'],
            'training_metrics': response.get('trainingMetrics', {}),
            'validation_metrics': response.get('validationMetrics', {}),
            'failure_message': response.get('failureMessage', None),
            'output_model_arn': response.get('outputModelArn', None)
        }

    def wait_for_completion(self,
                          job_arn: str,
                          check_interval: int = 300) -> Dict:
        """
        Wait for fine-tuning job to complete (blocking).

        Args:
            job_arn: Job ARN
            check_interval: How often to check status (seconds)

        Returns:
            Final job status
        """
        print(f"\nWaiting for fine-tuning to complete...")
        print(f"Checking every {check_interval} seconds...")

        while True:
            status = self.get_job_status(job_arn)

            print(f"  Status: {status['status']}")

            if status['status'] == 'Completed':
                print(f"\n✓ Fine-tuning completed!")
                print(f"  Custom Model ARN: {status['output_model_arn']}")
                return status

            elif status['status'] in ['Failed', 'Stopped']:
                print(f"\n✗ Fine-tuning failed: {status.get('failure_message', 'Unknown error')}")
                return status

            # Still in progress (InProgress or Pending)
            time.sleep(check_interval)


# Example Usage
if __name__ == "__main__":
    import os

    print("="*70)
    print("AWS BEDROCK FINE-TUNING")
    print("="*70)

    # Configuration
    REGION = 'us-east-1'
    S3_BUCKET = 'my-bedrock-training-data'
    OUTPUT_BUCKET = 'my-bedrock-models'
    IAM_ROLE_ARN = os.environ.get('BEDROCK_IAM_ROLE_ARN')  # Set this env var
    JOB_NAME = 'network-troubleshooting-v1'
    BASE_MODEL = 'anthropic.claude-3-haiku-20240307-v1:0'

    if not IAM_ROLE_ARN:
        print("Error: Set BEDROCK_IAM_ROLE_ARN environment variable")
        print("Example: arn:aws:iam::123456789012:role/BedrockFineTuningRole")
        exit(1)

    # Initialize fine-tuner
    tuner = BedrockFineTuner(region=REGION)

    # Upload training data
    print("\nStep 1: Upload Training Data")
    print("-" * 70)

    training_uri = tuner.upload_training_data(
        local_file='train.jsonl',
        bucket=S3_BUCKET,
        s3_key=f'{JOB_NAME}/train.jsonl'
    )

    validation_uri = tuner.upload_training_data(
        local_file='validation.jsonl',
        bucket=S3_BUCKET,
        s3_key=f'{JOB_NAME}/validation.jsonl'
    )

    # Create fine-tuning job
    print("\nStep 2: Create Fine-Tuning Job")
    print("-" * 70)

    job_arn = tuner.create_fine_tuning_job(
        model_id=BASE_MODEL,
        training_data_uri=training_uri,
        validation_data_uri=validation_uri,
        job_name=JOB_NAME,
        output_bucket=OUTPUT_BUCKET,
        iam_role_arn=IAM_ROLE_ARN,
        hyperparameters={
            'epochCount': '3',
            'batchSize': '8',
            'learningRate': '0.00001'
        }
    )

    # Wait for completion (optional - can check later)
    print("\nStep 3: Wait for Completion (2-6 hours)")
    print("-" * 70)
    print("Options:")
    print("  1. Wait here (blocking)")
    print("  2. Check status later with get_job_status()")

    wait = input("\nWait for completion? (y/n): ").lower()

    if wait == 'y':
        final_status = tuner.wait_for_completion(job_arn, check_interval=300)

        print("\n" + "="*70)
        print("FINE-TUNING RESULTS")
        print("="*70)
        print(f"Status: {final_status['status']}")

        if final_status['status'] == 'Completed':
            print(f"Custom Model ARN: {final_status['output_model_arn']}")
            print(f"\nUse this model ARN in your API calls to use the fine-tuned model.")

            if final_status['training_metrics']:
                print(f"\nTraining Loss: {final_status['training_metrics'].get('loss', 'N/A')}")
            if final_status['validation_metrics']:
                print(f"Validation Loss: {final_status['validation_metrics'].get('loss', 'N/A')}")
    else:
        print(f"\nJob ARN: {job_arn}")
        print("Check status later with: tuner.get_job_status(job_arn)")
```

### Example Output

```
======================================================================
AWS BEDROCK FINE-TUNING
======================================================================

Step 1: Upload Training Data
----------------------------------------------------------------------
Uploading train.jsonl to S3...
✓ Uploaded to s3://my-bedrock-training-data/network-troubleshooting-v1/train.jsonl
Uploading validation.jsonl to S3...
✓ Uploaded to s3://my-bedrock-training-data/network-troubleshooting-v1/validation.jsonl

Step 2: Create Fine-Tuning Job
----------------------------------------------------------------------

Creating fine-tuning job: network-troubleshooting-v1
  Base model: anthropic.claude-3-haiku-20240307-v1:0
  Training data: s3://my-bedrock-training-data/network-troubleshooting-v1/train.jsonl
  Validation data: s3://my-bedrock-training-data/network-troubleshooting-v1/validation.jsonl
✓ Fine-tuning job created: arn:aws:bedrock:us-east-1:123456789012:model-customization-job/abc123

This will take 2-6 hours. Check status with get_job_status().

Step 3: Wait for Completion (2-6 hours)
----------------------------------------------------------------------
Options:
  1. Wait here (blocking)
  2. Check status later with get_job_status()

Wait for completion? (y/n): y

Waiting for fine-tuning to complete...
Checking every 300 seconds...
  Status: InProgress
  Status: InProgress
  Status: InProgress
  [... 2-6 hours later ...]
  Status: Completed

✓ Fine-tuning completed!
  Custom Model ARN: arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123

======================================================================
FINE-TUNING RESULTS
======================================================================
Status: Completed
Custom Model ARN: arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123

Use this model ARN in your API calls to use the fine-tuned model.

Training Loss: 0.234
Validation Loss: 0.256
```

### What Just Happened

You fine-tuned Claude via AWS Bedrock:

1. **Uploaded training data** to S3 (train.jsonl, validation.jsonl)
2. **Created fine-tuning job** with:
   - Base model: Claude 3 Haiku
   - Training: 3 epochs, batch size 8, learning rate 0.00001
   - Output: Custom model ARN
3. **Waited 2-6 hours** for training to complete
4. **Got custom model ARN** to use in API calls

**Training metrics**:
- **Training loss**: 0.234 - Lower is better, measures fit to training data
- **Validation loss**: 0.256 - Should be close to training loss (if much higher, overfitting)

**Using the fine-tuned model**:
```python
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Use your custom model ARN (from Bedrock)
response = client.messages.create(
    model="your-custom-model-arn",  # Replace with actual ARN
    max_tokens=1000,
    messages=[{"role": "user", "content": "BGP neighbor stuck in Active state"}]
)

print(response.content[0].text)
```

**Cost**: $200 one-time (example: 1M training tokens × 3 epochs @ ~$10/M training tokens). Actual cost depends on dataset size.

---

## Version 4: Benchmark & Deploy

**Goal**: Measure performance improvements and deploy fine-tuned model in production.

**What you'll build**: Benchmark suite that compares base model vs fine-tuned model on real test cases.

**Time**: 60 minutes

**Cost**: $5-10 (testing requests)

### Benchmark Metrics

Compare base vs fine-tuned on:
1. **Accuracy** - % of correct answers
2. **Latency** - Response time (shorter prompts = faster)
3. **Token usage** - Input tokens saved (lower cost)
4. **Cost per request** - Total API cost

### Implementation

```python
"""
Fine-Tuning Benchmark
File: benchmark.py

Compare base model vs fine-tuned model performance.
"""
from anthropic import Anthropic
import time
from typing import List, Dict
import os


class FineTuningBenchmark:
    """Benchmark base vs fine-tuned model."""

    # Claude Sonnet 4 pricing
    INPUT_COST = 3.00   # $3/M input tokens
    OUTPUT_COST = 15.00  # $15/M output tokens

    def __init__(self,
                 api_key: str,
                 base_model: str,
                 finetuned_model: str):
        """
        Args:
            api_key: Anthropic API key
            base_model: Base model ID (e.g., 'claude-sonnet-4-20250514')
            finetuned_model: Fine-tuned model ARN from AWS Bedrock
        """
        self.client = Anthropic(api_key=api_key)
        self.base_model = base_model
        self.finetuned_model = finetuned_model

    def run_benchmark(self, test_cases: List[Dict]) -> Dict:
        """
        Run benchmark on test cases.

        Args:
            test_cases: List of dicts with:
                - 'prompt': User query
                - 'expected_keywords': Keywords that should appear in correct answer

        Returns:
            Dict with performance comparison
        """
        print(f"\nRunning benchmark with {len(test_cases)} test cases...")
        print("="*70)

        # Test base model
        print("\nTesting BASE MODEL...")
        base_results = self._test_model(self.base_model, test_cases)

        # Test fine-tuned model
        print("\nTesting FINE-TUNED MODEL...")
        finetuned_results = self._test_model(self.finetuned_model, test_cases)

        # Compare
        return self._compare_results(base_results, finetuned_results)

    def _test_model(self, model: str, test_cases: List[Dict]) -> List[Dict]:
        """Test a model on all test cases."""
        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"  Test {i}/{len(test_cases)}: {test_case['prompt'][:50]}...", end=' ')

            start_time = time.time()

            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": test_case['prompt']}]
                )

                latency = time.time() - start_time
                answer = response.content[0].text.strip()

                # Check if answer contains expected keywords
                expected = test_case.get('expected_keywords', [])
                correct = all(keyword.lower() in answer.lower() for keyword in expected)

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (input_tokens / 1_000_000 * self.INPUT_COST) + \
                       (output_tokens / 1_000_000 * self.OUTPUT_COST)

                results.append({
                    'test_case': i,
                    'correct': correct,
                    'latency': latency,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': cost,
                    'answer': answer
                })

                print(f"{'✓' if correct else '✗'} ({latency:.2f}s, {input_tokens + output_tokens} tokens)")

            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'test_case': i,
                    'correct': False,
                    'error': str(e)
                })

        return results

    def _compare_results(self,
                        base_results: List[Dict],
                        finetuned_results: List[Dict]) -> Dict:
        """Compare base vs fine-tuned performance."""

        # Filter out errors
        base_valid = [r for r in base_results if 'error' not in r]
        finetuned_valid = [r for r in finetuned_results if 'error' not in r]

        # Calculate metrics
        base_accuracy = sum(r['correct'] for r in base_valid) / len(base_valid) if base_valid else 0
        finetuned_accuracy = sum(r['correct'] for r in finetuned_valid) / len(finetuned_valid) if finetuned_valid else 0

        base_avg_latency = sum(r['latency'] for r in base_valid) / len(base_valid) if base_valid else 0
        finetuned_avg_latency = sum(r['latency'] for r in finetuned_valid) / len(finetuned_valid) if finetuned_valid else 0

        base_avg_input = sum(r['input_tokens'] for r in base_valid) / len(base_valid) if base_valid else 0
        finetuned_avg_input = sum(r['input_tokens'] for r in finetuned_valid) / len(finetuned_valid) if finetuned_valid else 0

        base_avg_total = sum(r['total_tokens'] for r in base_valid) / len(base_valid) if base_valid else 0
        finetuned_avg_total = sum(r['total_tokens'] for r in finetuned_valid) / len(finetuned_valid) if finetuned_valid else 0

        base_avg_cost = sum(r['cost'] for r in base_valid) / len(base_valid) if base_valid else 0
        finetuned_avg_cost = sum(r['cost'] for r in finetuned_valid) / len(finetuned_valid) if finetuned_valid else 0

        # Calculate improvements
        accuracy_improvement = (finetuned_accuracy - base_accuracy) * 100
        latency_improvement = ((base_avg_latency - finetuned_avg_latency) / base_avg_latency * 100) if base_avg_latency > 0 else 0
        token_reduction = ((base_avg_total - finetuned_avg_total) / base_avg_total * 100) if base_avg_total > 0 else 0
        cost_reduction = ((base_avg_cost - finetuned_avg_cost) / base_avg_cost * 100) if base_avg_cost > 0 else 0

        return {
            'test_cases': len(base_results),
            'base_model': {
                'accuracy': base_accuracy,
                'avg_latency': base_avg_latency,
                'avg_input_tokens': base_avg_input,
                'avg_total_tokens': base_avg_total,
                'avg_cost': base_avg_cost
            },
            'finetuned_model': {
                'accuracy': finetuned_accuracy,
                'avg_latency': finetuned_avg_latency,
                'avg_input_tokens': finetuned_avg_input,
                'avg_total_tokens': finetuned_avg_total,
                'avg_cost': finetuned_avg_cost
            },
            'improvements': {
                'accuracy_improvement': accuracy_improvement,
                'latency_improvement': latency_improvement,
                'token_reduction': token_reduction,
                'cost_reduction': cost_reduction
            }
        }

    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)

        print(f"\nTest Cases: {results['test_cases']}")

        print(f"\nBase Model Performance:")
        print(f"  Accuracy: {results['base_model']['accuracy']*100:.1f}%")
        print(f"  Avg Latency: {results['base_model']['avg_latency']:.3f}s")
        print(f"  Avg Input Tokens: {results['base_model']['avg_input_tokens']:.0f}")
        print(f"  Avg Total Tokens: {results['base_model']['avg_total_tokens']:.0f}")
        print(f"  Avg Cost: ${results['base_model']['avg_cost']:.4f}")

        print(f"\nFine-Tuned Model Performance:")
        print(f"  Accuracy: {results['finetuned_model']['accuracy']*100:.1f}%")
        print(f"  Avg Latency: {results['finetuned_model']['avg_latency']:.3f}s")
        print(f"  Avg Input Tokens: {results['finetuned_model']['avg_input_tokens']:.0f}")
        print(f"  Avg Total Tokens: {results['finetuned_model']['avg_total_tokens']:.0f}")
        print(f"  Avg Cost: ${results['finetuned_model']['avg_cost']:.4f}")

        print(f"\nImprovements:")
        print(f"  Accuracy: {results['improvements']['accuracy_improvement']:+.1f} percentage points")
        print(f"  Latency: {results['improvements']['latency_improvement']:+.1f}%")
        print(f"  Token Usage: {results['improvements']['token_reduction']:+.1f}%")
        print(f"  Cost: {results['improvements']['cost_reduction']:+.1f}%")

        # Summary
        print(f"\n{'='*70}")

        if results['improvements']['accuracy_improvement'] > 5:
            print("✓ Significant accuracy improvement - fine-tuning is working")
        elif results['improvements']['accuracy_improvement'] > 0:
            print("⚠️  Modest accuracy improvement - evaluate if worth the cost")
        else:
            print("✗ No accuracy improvement - check training data quality")

        if results['improvements']['token_reduction'] > 30:
            print("✓ Significant token savings - cost reduction achieved")
        elif results['improvements']['token_reduction'] > 10:
            print("⚠️  Moderate token savings - some cost reduction")
        else:
            print("✗ Minimal token savings - may not justify fine-tuning cost")


# Example Usage
if __name__ == "__main__":
    import os

    print("="*70)
    print("FINE-TUNING BENCHMARK")
    print("="*70)

    # Test cases (in production, use 50-100 test cases)
    test_cases = [
        {
            'prompt': 'BGP neighbor 203.0.113.5 is stuck in Active state and never establishes. Diagnose the issue.',
            'expected_keywords': ['remote-as', 'mismatch', 'AS', 'config']
        },
        {
            'prompt': 'Switch port Gi0/24 is err-disabled with BPDUGUARD error. How do I fix this?',
            'expected_keywords': ['shutdown', 'no shutdown', 'portfast', 'bpdu']
        },
        {
            'prompt': 'OSPF adjacency with neighbor keeps flapping every 30 seconds. What could cause this?',
            'expected_keywords': ['timer', 'mismatch', 'hello', 'dead']
        },
        {
            'prompt': 'VoIP calls have poor quality during business hours. Network seems fine otherwise.',
            'expected_keywords': ['QoS', 'priority', 'DSCP', 'policy']
        },
        {
            'prompt': 'Generate a standard access port configuration for VLAN 10.',
            'expected_keywords': ['switchport mode access', 'vlan 10', 'portfast', 'bpduguard']
        }
    ]

    # Configuration
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_model = "claude-sonnet-4-20250514"
    finetuned_model = os.environ.get("FINETUNED_MODEL_ARN")  # From Bedrock

    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    if not finetuned_model:
        print("Error: Set FINETUNED_MODEL_ARN environment variable")
        print("Example: arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123")
        exit(1)

    # Run benchmark
    benchmark = FineTuningBenchmark(
        api_key=api_key,
        base_model=base_model,
        finetuned_model=finetuned_model
    )

    results = benchmark.run_benchmark(test_cases)
    benchmark.print_results(results)

    print("\n" + "="*70)
    print("✓ Benchmark complete - review results above")
    print("="*70)
```

### Example Output

```
======================================================================
FINE-TUNING BENCHMARK
======================================================================

Running benchmark with 5 test cases...
======================================================================

Testing BASE MODEL...
  Test 1/5: BGP neighbor 203.0.113.5 is stuck in Active sta... ✓ (1.23s, 2834 tokens)
  Test 2/5: Switch port Gi0/24 is err-disabled with BPDUGUARD... ✓ (1.18s, 2756 tokens)
  Test 3/5: OSPF adjacency with neighbor keeps flapping ever... ✓ (1.31s, 2891 tokens)
  Test 4/5: VoIP calls have poor quality during business hou... ✓ (1.15s, 2723 tokens)
  Test 5/5: Generate a standard access port configuration for... ✓ (0.98s, 2567 tokens)

Testing FINE-TUNED MODEL...
  Test 1/5: BGP neighbor 203.0.113.5 is stuck in Active sta... ✓ (0.67s, 1234 tokens)
  Test 2/5: Switch port Gi0/24 is err-disabled with BPDUGUARD... ✓ (0.61s, 1189 tokens)
  Test 3/5: OSPF adjacency with neighbor keeps flapping ever... ✓ (0.72s, 1298 tokens)
  Test 4/5: VoIP calls have poor quality during business hou... ✓ (0.59s, 1156 tokens)
  Test 5/5: Generate a standard access port configuration for... ✓ (0.54s, 1098 tokens)

======================================================================
BENCHMARK RESULTS
======================================================================

Test Cases: 5

Base Model Performance:
  Accuracy: 100.0%
  Avg Latency: 1.170s
  Avg Input Tokens: 2212
  Avg Total Tokens: 2754
  Avg Cost: $0.0144

Fine-Tuned Model Performance:
  Accuracy: 100.0%
  Avg Latency: 0.626s
  Avg Input Tokens: 743
  Avg Total Tokens: 1195
  Avg Cost: $0.0061

Improvements:
  Accuracy: +0.0 percentage points
  Latency: -46.5%
  Token Usage: -56.6%
  Cost: -57.6%

======================================================================
✓ Significant token savings - cost reduction achieved
======================================================================
✓ Benchmark complete - review results above
======================================================================
```

### What Just Happened

The benchmark compared base model vs fine-tuned on 5 real test cases:

**Performance Gains**:
- **Accuracy**: Same (100% both) - Fine-tuning maintained quality
- **Latency**: -46.5% - Responses 46% faster (2,212 → 743 input tokens)
- **Token Usage**: -56.6% - Used 56% fewer tokens per request
- **Cost**: -57.6% - Reduced cost by 57% per request

**Why the improvements**:
1. **Context built into model** - Don't need 2,000 tokens of topology/standards in every prompt
2. **Shorter prompts** - 743 tokens vs 2,212 tokens average
3. **Faster inference** - Shorter prompts = faster API responses
4. **Lower costs** - Fewer input tokens = lower per-request cost

**Real-world impact at 50,000 requests/month**:
- Before: $750/month (50K × $0.015)
- After: $305/month (50K × $0.0061)
- Savings: $445/month ($5,340/year)
- Fine-tuning cost: $200 one-time
- Breaks even in: 14 days

**When fine-tuning doesn't help**:
- Accuracy drops (check training data quality)
- Token savings <10% (prompt engineering might be better)
- Validation loss >> training loss (overfitting - reduce epochs)

**Cost**: $5-10 (5 test cases × 2 models × $0.50-1.00 per test)

---

## Complete System

You now have four versions that work together:

**V1: ROI Calculator** ($0)
- Input: Monthly volume, token counts
- Output: Break-even analysis
- Decision: Fine-tune or stick with prompts

**V2: Training Dataset** ($0)
- Input: Tickets, configs, standards
- Output: train.jsonl, validation.jsonl
- Quality: 500+ examples, validated, no PII

**V3: Fine-Tuned Model** ($200)
- Input: Training files, hyperparameters
- Output: Custom model ARN
- Time: 2-6 hours training

**V4: Benchmark & Deploy** ($10)
- Input: Test cases
- Output: Performance comparison
- Metrics: Accuracy, latency, tokens, cost

**Production deployment**: Use the fine-tuned model ARN in your API calls. Monitor performance. Retrain every 3-6 months as your network changes.

---

## Labs

### Lab 1: Calculate Your ROI (45 minutes)

Calculate break-even point for your actual scenario.

**Your task**:
1. Identify a high-volume AI task in your network (config parsing, log analysis, troubleshooting)
2. Measure current metrics:
   - Monthly request volume
   - Average prompt tokens (including context)
   - Average response tokens
3. Estimate with fine-tuning:
   - How many prompt tokens could you save? (context that could move into model)
4. Run ROI calculator with your numbers
5. Decide: Fine-tune or optimize prompts?

**Deliverable**: ROI analysis showing break-even point for your scenario.

**Success**: You have a data-driven decision on whether to fine-tune.

---

### Lab 2: Build Training Dataset (60 minutes)

Convert your operational data into training format.

**Your task**:
1. Export 50 resolved tickets from your ticketing system (use training_data_generator.py)
2. Add 20 config standards from your documentation
3. Add 10 config validation examples
4. Run validator to clean dataset
5. Export to train.jsonl and validation.jsonl

**Deliverable**:
- train.jsonl (72 examples)
- validation.jsonl (8 examples)
- Statistics showing estimated training cost

**Success**: Clean, validated training data ready for fine-tuning.

---

### Lab 3: Fine-Tune and Benchmark (90 minutes + 2-6 hours)

Launch fine-tuning job and measure performance.

**Your task**:
1. Set up AWS Bedrock (account, IAM role, S3 bucket)
2. Upload training data to S3
3. Launch fine-tuning job with bedrock_fine_tune.py
4. Wait for completion (can do other work)
5. Create 10 test cases representing real queries
6. Run benchmark comparing base vs fine-tuned
7. Calculate actual ROI from benchmark results

**Deliverable**:
- Custom model ARN
- Benchmark results showing accuracy, latency, token usage, cost
- Actual ROI calculation

**Success**: Fine-tuned model deployed in production with measured performance improvements.

---

## Check Your Understanding

<details>
<summary><strong>1. You have 50,000 requests/month with 2,500-token prompts. Fine-tuning costs $200 and reduces prompts to 500 tokens. Should you fine-tune?</strong></summary>

**Answer: Yes, fine-tune immediately.**

Calculate the ROI:

**Without fine-tuning**:
- Input: 50K × 2,500 tokens = 125M tokens
- Cost: 125M / 1M × $3 = $375/month

**With fine-tuning**:
- Input: 50K × 500 tokens = 25M tokens
- Cost: 25M / 1M × $3 = $75/month
- Savings: $375 - $75 = $300/month

**Break-even**: $200 / $300 = 0.67 months (20 days)

Fine-tuning pays for itself in 20 days and saves $3,600/year. This is an excellent ROI case.

**Key insight**: High volume (>10K/month) + significant context (>1,000 tokens) + repetitive context = perfect fine-tuning candidate.
</details>

<details>
<summary><strong>2. Your fine-tuned model has training loss 0.234 and validation loss 0.567. What does this mean?</strong></summary>

**Answer: Your model is overfitting - it memorized training data but doesn't generalize.**

**What the losses mean**:
- **Training loss**: How well model fits training data (0.234 = good fit)
- **Validation loss**: How well model generalizes to new data (0.567 = poor)

**The problem**: Validation loss is 2.4× higher than training loss. This indicates overfitting - the model memorized specific training examples but can't apply that knowledge to similar but different scenarios.

**How to fix**:
1. **Reduce epochs** - Try 2 epochs instead of 3
2. **Lower learning rate** - Use 0.000005 instead of 0.00001
3. **Add more diverse training data** - 500+ examples covering different scenarios
4. **Use regularization** - Add dropout if provider supports it

**Good metrics**: Validation loss should be within 20% of training loss (e.g., training 0.234, validation 0.280).

**Key insight**: Training loss measures memorization, validation loss measures understanding. You want understanding.
</details>

<details>
<summary><strong>3. Your benchmark shows accuracy improved from 85% to 95%, but tokens only reduced 10%. Is fine-tuning worth it?</strong></summary>

**Answer: Yes, if accuracy matters. The 10% accuracy gain may be more valuable than cost savings.**

**Calculate the value**:

**Scenario**: Config generation for production, 10,000 configs/month

**Without fine-tuning** (85% accuracy):
- Correct: 8,500 configs
- Errors: 1,500 configs need manual review
- Manual review cost: 1,500 × 15 min × $50/hr = $18,750/month

**With fine-tuning** (95% accuracy):
- Correct: 9,500 configs
- Errors: 500 configs need manual review
- Manual review cost: 500 × 15 min × $50/hr = $6,250/month
- Savings: $12,500/month

**Even with minimal token savings**:
- Token savings: 10% × 10K requests × $0.015 = $15/month
- Labor savings: $12,500/month
- Total savings: $12,515/month
- Fine-tuning cost: $200
- Break-even: 0.5 days

**Key insight**: ROI isn't just about token costs. Accuracy improvements can save massive amounts of manual review time. Calculate the value of errors prevented, not just API costs.
</details>

<details>
<summary><strong>4. You have 500 examples of high quality. Is this enough to fine-tune?</strong></summary>

**Answer: Yes, 500 high-quality examples is the minimum threshold for effective fine-tuning.**

**Quality vs quantity**:
- **500 high-quality examples** > 5,000 low-quality examples
- High-quality = accurate outputs, diverse scenarios, consistent format, no errors

**What 500 examples can teach**:
- Your network's naming conventions
- Your config standards
- Common troubleshooting patterns
- Vendor-specific syntax in your environment

**When you need more**:
- **Complex domain knowledge** - May need 1,000+ examples for specialized protocols
- **Multiple vendors** - Need 500+ examples per vendor (Cisco, Juniper, Arista)
- **Diverse scenarios** - If your network has 10 different device roles, need 50+ examples per role

**How to know if you have enough**:
1. Check validation loss - Should stabilize and match training loss
2. Benchmark on unseen test cases - Should maintain accuracy
3. Monitor production - If accuracy drops over time, need more training data

**Key insight**: Start with 500, benchmark thoroughly, add more if validation loss is high or production accuracy drops.
</details>

---

## Lab Time Budget

### Time Investment

**V1: ROI Calculator** (45 min)
- Gather metrics: 15 min
- Run calculator: 5 min
- Interpret results: 10 min
- Document decision: 15 min

**V2: Training Dataset** (60 min)
- Export tickets/configs: 20 min
- Format training data: 25 min
- Validate and clean: 10 min
- Export JSONL: 5 min

**V3: Fine-Tuned Model** (90 min your time + 2-6h training)
- AWS setup: 30 min
- Upload data: 10 min
- Create job: 20 min
- Monitor progress: 30 min (checking periodically)
- Training time: 2-6 hours (automated, no your time)

**V4: Benchmark & Deploy** (60 min)
- Create test cases: 20 min
- Run benchmark: 15 min
- Analyze results: 15 min
- Document findings: 10 min

**Total time investment**: 3.75 hours (your active time)

**Labs**: 3.25 hours
- Lab 1: 45 min
- Lab 2: 60 min
- Lab 3: 90 min

**Total to working system**: 7 hours of your time (spread over days due to training wait)

### Cost Investment

**First year costs**:
- V1: $0 (calculations)
- V2: $0 (data prep)
- V3: $200 (one-time fine-tuning)
- V4: $10 (benchmark testing)
- Retraining: $200 × 2 = $400 (every 6 months as network changes)
- **Total**: $610

### Value Delivered

**Scenario**: 50,000 requests/month, troubleshooting assistant

**Token cost savings**:
- Before: $750/month
- After: $450/month
- Savings: $300/month = $3,600/year

**Accuracy improvements** (85% → 95%):
- Errors reduced: 7,500 → 2,500 per year
- Manual review avoided: 5,000 tickets × 15 min = 1,250 hours
- Labor savings: 1,250 × $50/hr = $62,500/year

**Latency improvements** (1.2s → 0.6s):
- Time saved: 50K × 0.6s × 12 months = 10 hours of cumulative wait time
- User productivity: Engineers waste less time waiting
- Value: ~$500/year (hard to quantify exactly)

**Total value delivered**: $66,600/year

### ROI Calculation

**Investment**: 7 hours × $50/hr + $610 = $960

**Return**: $66,600/year

**ROI**: (($66,600 - $960) / $960) × 100 = **6,837%**

**Break-even**: $960 / ($66,600/12) = 0.17 months = **5 days**

### Why This ROI Is Realistic

Unlike typical software projects that take months to show value, fine-tuning delivers immediate, measurable returns:

1. **Token savings are guaranteed** - Math is simple: fewer tokens = lower cost
2. **Accuracy gains are measurable** - Benchmark shows exact improvement
3. **Latency improvements are automatic** - Shorter prompts = faster responses
4. **Labor savings are quantifiable** - Count manual reviews before vs after

**Best case**: You're Acme Corp with 50K requests/month → ROI in 5 days
**Realistic case**: You're mid-size company with 10K requests/month → ROI in 25 days
**Worst case**: You're small company with 1K requests/month → ROI in 8 months (don't fine-tune yet)

---

## Production Deployment Guide

### Phase 1: Validation (Week 1)

**Test fine-tuned model in staging**:

```python
# test_production.py
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def test_model(model_id: str, test_cases: list) -> dict:
    """Test model on production-like queries."""
    results = {'passed': 0, 'failed': 0, 'errors': []}

    for test in test_cases:
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=1000,
                messages=[{"role": "user", "content": test['query']}]
            )

            # Check if response contains expected keywords
            answer = response.content[0].text.lower()
            if all(kw.lower() in answer for kw in test['expected_keywords']):
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'query': test['query'],
                    'expected': test['expected_keywords'],
                    'got': answer[:200]
                })
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({'query': test['query'], 'error': str(e)})

    results['accuracy'] = results['passed'] / (results['passed'] + results['failed'])
    return results

# Run validation
test_cases = [
    # Add 50-100 real production queries here
]

results = test_model(os.environ.get("FINETUNED_MODEL_ARN"), test_cases)
print(f"Validation accuracy: {results['accuracy']*100:.1f}%")

if results['accuracy'] < 0.90:
    print("❌ Accuracy too low - retrain with more data")
    exit(1)
else:
    print("✅ Validation passed - ready for production")
```

**Week 1 checklist**:
- ✅ Run validation tests (50+ test cases)
- ✅ Accuracy ≥90% on unseen data
- ✅ Latency <2s average
- ✅ No errors or hallucinations
- ✅ Document model version and training data used

### Phase 2: Shadow Mode (Week 2-3)

**Run fine-tuned model in parallel with base model** (don't act on results yet):

```python
# shadow_mode.py
class ShadowModeComparison:
    """Run base and fine-tuned models in parallel for comparison."""

    def __init__(self, base_model: str, finetuned_model: str):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.comparisons = []

    def query(self, user_query: str) -> dict:
        """Send query to both models and compare."""

        # Base model
        base_response = self.client.messages.create(
            model=self.base_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": user_query}]
        )

        # Fine-tuned model
        finetuned_response = self.client.messages.create(
            model=self.finetuned_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": user_query}]
        )

        comparison = {
            'query': user_query,
            'base_answer': base_response.content[0].text,
            'finetuned_answer': finetuned_response.content[0].text,
            'base_tokens': base_response.usage.input_tokens + base_response.usage.output_tokens,
            'finetuned_tokens': finetuned_response.usage.input_tokens + finetuned_response.usage.output_tokens,
            'token_savings': base_response.usage.input_tokens - finetuned_response.usage.input_tokens
        }

        self.comparisons.append(comparison)

        # Return base model result (production still uses base)
        return {'answer': base_response.content[0].text}

    def generate_report(self) -> dict:
        """Generate comparison report."""
        total_token_savings = sum(c['token_savings'] for c in self.comparisons)
        avg_token_savings = total_token_savings / len(self.comparisons)

        return {
            'total_queries': len(self.comparisons),
            'avg_token_savings': avg_token_savings,
            'total_token_savings': total_token_savings,
            'estimated_monthly_savings': (avg_token_savings * 50000 / 1_000_000 * 3)  # Assuming 50K queries/month
        }

# Use in production
shadow = ShadowModeComparison(
    base_model="claude-sonnet-4-20250514",
    finetuned_model=os.environ.get("FINETUNED_MODEL_ARN")
)

# Your production code
result = shadow.query("BGP neighbor stuck in Active state")

# After 2 weeks, generate report
report = shadow.generate_report()
print(f"Shadow mode results: {report['avg_token_savings']:.0f} tokens saved per query")
print(f"Estimated savings: ${report['estimated_monthly_savings']:.2f}/month")
```

**Week 2-3 checklist**:
- ✅ Shadow mode running on 100% of production traffic
- ✅ Collect 10,000+ comparisons
- ✅ Manual review of 100 random comparisons
- ✅ Confirm token savings match benchmark
- ✅ No quality degradation vs base model

### Phase 3: Gradual Rollout (Week 4-6)

**Route increasing % of traffic to fine-tuned model**:

```python
# gradual_rollout.py
import random

class GradualRollout:
    """Route % of traffic to fine-tuned model."""

    def __init__(self, base_model: str, finetuned_model: str, rollout_percentage: int):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.rollout_percentage = rollout_percentage
        self.metrics = {'base_queries': 0, 'finetuned_queries': 0, 'errors': 0}

    def query(self, user_query: str) -> dict:
        """Route query to base or fine-tuned based on rollout %."""

        # Randomly select model based on rollout percentage
        use_finetuned = random.randint(1, 100) <= self.rollout_percentage

        model = self.finetuned_model if use_finetuned else self.base_model

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": user_query}]
            )

            if use_finetuned:
                self.metrics['finetuned_queries'] += 1
            else:
                self.metrics['base_queries'] += 1

            return {
                'answer': response.content[0].text,
                'model_used': 'finetuned' if use_finetuned else 'base'
            }

        except Exception as e:
            self.metrics['errors'] += 1
            # Fallback to base model on error
            response = self.client.messages.create(
                model=self.base_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": user_query}]
            )
            return {'answer': response.content[0].text, 'model_used': 'base_fallback'}

# Week 4: 10% rollout
rollout = GradualRollout(
    base_model="claude-sonnet-4-20250514",
    finetuned_model=os.environ.get("FINETUNED_MODEL_ARN"),
    rollout_percentage=10
)

# Week 5: 50% rollout
# rollout_percentage=50

# Week 6: 100% rollout
# rollout_percentage=100
```

**Rollout schedule**:
- Week 4: 10% traffic → Monitor for 1 week
- Week 5: 50% traffic → Monitor for 1 week
- Week 6: 100% traffic → Fine-tuned model fully deployed

**Week 4-6 checklist**:
- ✅ Week 4: 10% rollout, no issues
- ✅ Week 5: 50% rollout, no issues
- ✅ Week 6: 100% rollout complete
- ✅ Error rate <0.1%
- ✅ User complaints unchanged
- ✅ Cost savings confirmed

### Phase 4: Production Monitoring (Week 7+)

**Monitor performance in production**:

```python
# monitoring.py
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class QueryMetrics:
    """Track metrics for each query."""
    timestamp: datetime
    query: str
    response_time: float
    input_tokens: int
    output_tokens: int
    cost: float
    user_feedback: str = None  # 'helpful' or 'not_helpful'

class ProductionMonitoring:
    """Monitor fine-tuned model in production."""

    def __init__(self):
        self.metrics = []

    def log_query(self, metric: QueryMetrics):
        """Log query metrics."""
        self.metrics.append(metric)

        # Write to metrics file
        with open('metrics.jsonl', 'a') as f:
            f.write(json.dumps({
                'timestamp': metric.timestamp.isoformat(),
                'response_time': metric.response_time,
                'tokens': metric.input_tokens + metric.output_tokens,
                'cost': metric.cost,
                'feedback': metric.user_feedback
            }) + '\n')

    def generate_daily_report(self) -> dict:
        """Generate daily performance report."""
        today_metrics = [m for m in self.metrics
                        if m.timestamp.date() == datetime.now().date()]

        if not today_metrics:
            return {}

        total_queries = len(today_metrics)
        avg_response_time = sum(m.response_time for m in today_metrics) / total_queries
        total_cost = sum(m.cost for m in today_metrics)

        helpful = sum(1 for m in today_metrics if m.user_feedback == 'helpful')
        not_helpful = sum(1 for m in today_metrics if m.user_feedback == 'not_helpful')
        satisfaction = helpful / (helpful + not_helpful) if (helpful + not_helpful) > 0 else None

        return {
            'date': datetime.now().date().isoformat(),
            'total_queries': total_queries,
            'avg_response_time': avg_response_time,
            'total_cost': total_cost,
            'user_satisfaction': satisfaction
        }

# Use in production
monitor = ProductionMonitoring()

# After each query
monitor.log_query(QueryMetrics(
    timestamp=datetime.now(),
    query="BGP neighbor down",
    response_time=0.67,
    input_tokens=500,
    output_tokens=300,
    cost=0.006,
    user_feedback='helpful'
))

# Generate daily report
report = monitor.generate_daily_report()
print(f"Daily Report: {report}")
```

**Week 7+ checklist**:
- ✅ Daily monitoring dashboard
- ✅ Track: queries/day, avg response time, cost, user satisfaction
- ✅ Alert if accuracy drops below 90%
- ✅ Alert if cost exceeds budget
- ✅ Monthly review of training data quality

### Rollback Plan

If fine-tuned model has issues:

```python
# rollback.py
def rollback_to_base_model():
    """Instant rollback to base model."""

    # Update environment variable
    os.environ['MODEL_ID'] = 'claude-sonnet-4-20250514'  # Base model

    # Or use feature flag
    feature_flags['use_finetuned_model'] = False

    print("✅ Rolled back to base model")
```

**Rollback triggers**:
- Accuracy drops below 85%
- Error rate exceeds 1%
- User complaints increase 2×
- Cost exceeds budget by 20%

---

## Common Problems and Solutions

### Problem 1: Model overfits (training accuracy 98%, validation accuracy 65%)

**Symptoms**:
- Training loss very low (< 0.2)
- Validation loss much higher (> 0.5)
- Benchmark shows poor performance on new test cases

**Cause**: Model memorized training data but didn't learn general patterns.

**Solution**:
1. **Reduce epochs** - Try 2 epochs instead of 3
2. **Lower learning rate** - Use 0.000005 instead of 0.00001
3. **Add more diverse training data** - Need 500+ examples covering different scenarios
4. **Increase validation set** - Use 15-20% validation instead of 10%

**Prevention**: Monitor validation loss during training. If it stops improving while training loss keeps decreasing, stop early.

---

### Problem 2: Fine-tuned model "forgot" general knowledge

**Symptoms**:
- Model excellent at your specific tasks
- Model fails at basic networking questions it should know
- Benchmark shows regression on general queries

**Cause**: Training data was too specialized, model lost base knowledge.

**Solution**:
1. **Add general examples to training data** - Include 20% "general networking" queries
2. **Use lower learning rate** - 0.000005 preserves more base knowledge
3. **Reduce epochs** - Fewer epochs = less "forgetting"
4. **Retrain from base model** - Don't fine-tune a fine-tuned model

**Prevention**: Include diverse examples in training data, not just specialized cases.

---

### Problem 3: Training fails with data format errors

**Symptoms**:
- Fine-tuning job fails immediately
- Error: "Invalid training data format"
- Validation shows JSONL format issues

**Cause**: Training data doesn't match required format.

**Solution**:
```python
# validate_format.py
import json

def validate_training_data(file_path: str):
    """Validate JSONL format."""
    errors = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                example = json.loads(line)

                # Check required fields
                if 'messages' not in example:
                    errors.append(f"Line {i}: Missing 'messages' field")
                elif not isinstance(example['messages'], list):
                    errors.append(f"Line {i}: 'messages' must be a list")
                elif len(example['messages']) < 2:
                    errors.append(f"Line {i}: Need at least 2 messages")

                # Check message format
                for msg in example['messages']:
                    if 'role' not in msg or 'content' not in msg:
                        errors.append(f"Line {i}: Message missing 'role' or 'content'")

            except json.JSONDecodeError:
                errors.append(f"Line {i}: Invalid JSON")

    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  {error}")
        return False
    else:
        print("✅ Training data format valid")
        return True

validate_training_data('train.jsonl')
```

**Prevention**: Use the validator before uploading to S3.

---

### Problem 4: ROI calculation was optimistic (actual savings less than projected)

**Symptoms**:
- Projected savings: $300/month
- Actual savings: $100/month
- Token reduction less than expected

**Cause**: Overestimated token savings or underestimated edge cases.

**Solution**:
1. **Run shadow mode** - Compare actual token usage for 2 weeks
2. **Calculate real savings** - Use actual metrics, not estimates
3. **Account for edge cases** - Some queries still need context
4. **Factor in retraining** - Add retraining cost ($200 every 6 months)

**Updated ROI**:
- Actual savings: $100/month
- Fine-tuning cost: $200
- Retraining: $400/year
- Break-even: ($200 + $400) / ($100 × 12) = 50% ROI (still positive)

**Prevention**: Run shadow mode before committing to fine-tuning. Use conservative estimates in ROI calculator.

---

### Problem 5: Fine-tuned model gives inconsistent outputs

**Symptoms**:
- Same query gives different answers each time
- Quality varies significantly
- Some outputs are hallucinations

**Cause**: Training data has inconsistent examples or model temperature too high.

**Solution**:
1. **Check training data consistency**:
   ```python
   # Find inconsistent examples
   def find_inconsistencies(training_file):
       examples = []
       with open(training_file) as f:
           for line in f:
               ex = json.loads(line)
               examples.append(ex)

       # Look for similar prompts with different answers
       for i, ex1 in enumerate(examples):
           for ex2 in examples[i+1:]:
               if similar(ex1['messages'][1]['content'],
                         ex2['messages'][1]['content']):
                   if ex1['messages'][2]['content'] != ex2['messages'][2]['content']:
                       print(f"Inconsistent: {ex1} vs {ex2}")
   ```

2. **Use temperature=0** in API calls for consistency:
   ```python
   response = client.messages.create(
       model=finetuned_model,
       max_tokens=1000,
       temperature=0,  # Deterministic outputs
       messages=[{"role": "user", "content": query}]
   )
   ```

3. **Retrain with cleaned data** - Remove inconsistent examples

**Prevention**: Validate training data for consistency before fine-tuning. Use temperature=0 for production.

---

### Problem 6: Cost exceeds budget (projected $450/month, actual $650/month)

**Symptoms**:
- Monthly bill higher than expected
- Actual query volume higher than estimated
- Some queries use more tokens than benchmark

**Cause**: Query volume higher than estimated or prompt engineering degraded.

**Solution**:
1. **Monitor actual usage**:
   ```python
   # cost_monitoring.py
   class CostMonitoring:
       def __init__(self):
           self.daily_cost = 0
           self.monthly_budget = 450

       def log_query_cost(self, cost: float):
           self.daily_cost += cost

           # Alert if on track to exceed budget
           days_in_month = 30
           projected_monthly = self.daily_cost * days_in_month

           if projected_monthly > self.monthly_budget * 1.1:
               print(f"⚠️  Cost alert: Projected ${projected_monthly:.2f}/month (budget: ${self.monthly_budget})")
   ```

2. **Implement cost controls**:
   - Set max_tokens lower (1000 → 500)
   - Cache frequent queries
   - Use Claude Haiku for simple queries

3. **Optimize prompts further** - Even fine-tuned models benefit from concise prompts

**Prevention**: Monitor costs weekly. Set alerts at 80% of budget.

---

## Summary

You've built a complete fine-tuning system in four versions:

**V1: ROI Calculator** - Calculate break-even point to decide if fine-tuning makes financial sense
**V2: Training Dataset** - Convert tickets and configs into validated training data
**V3: Fine-Tuned Model** - Launch AWS Bedrock fine-tuning job, get custom model ARN
**V4: Benchmark & Deploy** - Measure accuracy gains, token savings, cost reduction

**Key Learnings**:

1. **Fine-tune when**: High volume (>10K/month) + repetitive context (>1K tokens) + accuracy matters
2. **Don't fine-tune when**: Low volume, changing requirements, already accurate, limited training data
3. **Training data quality matters**: 500+ accurate, diverse examples beats 5,000 inconsistent examples
4. **Measure everything**: ROI before fine-tuning, benchmark after, monitor in production
5. **Typical improvements**: -40% latency, -50% tokens, -40% cost, +5-15% accuracy

**When fine-tuning pays off**:
- High-volume troubleshooting assistant: Breaks even in 20 days
- Config generation at scale: Breaks even in 1 month
- Security analysis system: Breaks even in 2 weeks

**When prompt engineering is better**:
- Low-volume documentation queries: Takes 33 months to break even
- Experimental projects: Requirements change weekly
- Already accurate tasks: 95% → 97% not worth effort

**Next chapter**: Multi-agent orchestration - multiple specialized AI agents working together on complex network tasks.

---

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/volume-3/chapter-32/`
