# Chapter 32: Fine-Tuning Models for Network Data

## Introduction

You've been using Claude with prompt engineering for months. It works, but you notice patterns: you keep explaining the same network topology, the same vendor-specific syntax, the same operational procedures. Every API call includes 2,000 tokens of context just to tell the model what a "normal" config looks like in your environment.

**Fine-tuning** teaches the model your network's patterns once, permanently. After fine-tuning, the model already knows your topology, your naming conventions, your config standards. You skip 2,000 tokens of context per request. Faster responses, lower costs, better accuracy.

But fine-tuning is expensive upfront and only makes sense in specific scenarios. This chapter shows you exactly when fine-tuning is worth it, how to create training data from your network, and real cost/performance analysis with actual numbers.

**What You'll Learn**:
- When fine-tuning beats prompt engineering (and when it doesn't)
- Creating training datasets from network logs and configs
- Fine-tuning process with Claude (via AWS Bedrock)
- Cost analysis: break-even point calculations
- Performance benchmarks: accuracy improvements
- Real case study: fine-tuned model for network troubleshooting

**Prerequisites**: Chapters 5 (Prompt Engineering), 8 (Cost Optimization), 14 (RAG Fundamentals)

---

## Fine-Tuning vs. Prompt Engineering: The Decision Matrix

### When to Fine-Tune

**Fine-tuning makes sense when**:

1. **High-volume, repetitive tasks** (>10,000 requests/month)
   - Same context repeated in every prompt
   - Example: Parsing configs with vendor-specific syntax daily

2. **Domain-specific knowledge** not in base model
   - Your company's network topology, naming conventions
   - Proprietary protocols or internal tools
   - Example: Custom network orchestration system with unique commands

3. **Improved accuracy needed** (base model is 80%, you need 95%)
   - Critical operations where errors are expensive
   - Example: Automated config generation for production

4. **Latency-sensitive applications**
   - Need faster responses (shorter prompts = faster)
   - Example: Real-time troubleshooting assistant

5. **Cost optimization at scale**
   - 2,000-token context per request × 50,000 requests = $1,000/month in unnecessary tokens
   - Fine-tuning cost: $500 one-time → ROI in 2 weeks

### When NOT to Fine-Tune

**Stick with prompt engineering when**:

1. **Low volume** (<1,000 requests/month)
   - Upfront fine-tuning cost won't amortize
   - Prompt engineering is cheaper

2. **Requirements change frequently**
   - Fine-tuning takes weeks, prompt changes take minutes
   - Example: Experimenting with different analysis approaches

3. **Task is already working well** with prompts
   - Base model + good prompts = 95% accuracy
   - Fine-tuning gains: 95% → 97% (not worth effort)

4. **Limited training data** (<500 high-quality examples)
   - Fine-tuning needs substantial data to be effective
   - RAG with few-shot examples works better

5. **Compliance/auditability required**
   - Fine-tuned models are "black boxes"
   - Prompt-based systems are more transparent

### Decision Matrix

| Scenario | Volume | Accuracy Need | Cost Sensitivity | Recommendation |
|----------|--------|---------------|------------------|----------------|
| Config parsing | 100K+/month | 98%+ | High | **Fine-tune** |
| Log analysis | 50K/month | 90% OK | Medium | **Fine-tune** |
| Troubleshooting | 5K/month | 85% OK | Low | **Prompt engineering** |
| Documentation | 500/month | 80% OK | Low | **Prompt engineering** |
| Security audit | 10K/month | 99%+ | High | **Fine-tune** |

---

## Creating Training Data from Network Operations

Fine-tuning requires high-quality training data: input-output pairs showing the model what you want.

### Training Data Format

Most LLM fine-tuning uses this format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a network config parser for Acme Corp."},
    {"role": "user", "content": "Parse this config: interface Gi0/1..."},
    {"role": "assistant", "content": "Interface: GigabitEthernet0/1, Description: ..."}
  ]
}
```

Each training example shows the model one task.

### Source 1: Historical Troubleshooting Tickets

Your ticketing system has thousands of solved problems. Convert them to training data.

#### Implementation

```python
"""
Convert Troubleshooting Tickets to Training Data
File: fine_tuning/ticket_to_training_data.py
"""
import json
from typing import List, Dict
import re

class TicketToTrainingData:
    """Convert historical tickets to fine-tuning dataset."""

    def __init__(self):
        self.training_examples = []

    def add_ticket(self, ticket: Dict):
        """
        Convert a ticket to training example.

        Args:
            ticket: Dict with keys: problem, diagnosis, resolution
        """
        # Create training example
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a network troubleshooting expert for Acme Corp networks. Your task is to diagnose network issues and provide solutions."
                },
                {
                    "role": "user",
                    "content": ticket['problem']
                },
                {
                    "role": "assistant",
                    "content": self._format_resolution(ticket['diagnosis'], ticket['resolution'])
                }
            ]
        }

        self.training_examples.append(example)

    def _format_resolution(self, diagnosis: str, resolution: str) -> str:
        """Format diagnosis and resolution as model output."""
        return f"""**Diagnosis:**
{diagnosis}

**Resolution:**
{resolution}"""

    def add_tickets_batch(self, tickets: List[Dict]):
        """Add multiple tickets at once."""
        for ticket in tickets:
            self.add_ticket(ticket)

    def validate_and_clean(self):
        """
        Validate training data quality.

        Remove:
        - Duplicate examples
        - Examples with incomplete data
        - Examples with PII
        """
        print(f"Validating {len(self.training_examples)} training examples...")

        # Remove duplicates
        unique_examples = []
        seen_problems = set()

        for example in self.training_examples:
            problem = example['messages'][1]['content']
            if problem not in seen_problems:
                unique_examples.append(example)
                seen_problems.add(problem)

        removed_dupes = len(self.training_examples) - len(unique_examples)
        print(f"  Removed {removed_dupes} duplicates")

        # Remove incomplete examples
        valid_examples = []
        for example in unique_examples:
            assistant_msg = example['messages'][2]['content']
            # Must have both diagnosis and resolution
            if 'Diagnosis:' in assistant_msg and 'Resolution:' in assistant_msg:
                valid_examples.append(example)

        removed_incomplete = len(unique_examples) - len(valid_examples)
        print(f"  Removed {removed_incomplete} incomplete examples")

        # Check for PII (simplified - use real PII detection in production)
        clean_examples = []
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email (overly broad)
        ]

        for example in valid_examples:
            has_pii = False
            for msg in example['messages']:
                for pattern in pii_patterns:
                    if re.search(pattern, msg['content']):
                        has_pii = True
                        break
                if has_pii:
                    break

            if not has_pii:
                clean_examples.append(example)

        removed_pii = len(valid_examples) - len(clean_examples)
        print(f"  Removed {removed_pii} examples with potential PII")

        self.training_examples = clean_examples
        print(f"✓ Final dataset: {len(self.training_examples)} examples")

    def export_to_jsonl(self, output_file: str):
        """
        Export training data to JSONL format (required by most fine-tuning APIs).

        Args:
            output_file: Path to output .jsonl file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self.training_examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Exported {len(self.training_examples)} examples to {output_file}")

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

        Note: Pricing varies by provider. These are example rates.
        Claude fine-tuning: ~$10 per 1M training tokens (example rate)
        """
        return (total_tokens / 1_000_000) * 10


# Example Usage
if __name__ == "__main__":
    # Sample historical tickets
    tickets = [
        {
            'problem': 'Branch office users cannot access datacenter applications. Ping times out.',
            'diagnosis': 'Primary WAN link (Gi0/0) is down. Traffic is routing through backup link but ACL on backup interface blocks application ports.',
            'resolution': 'Updated ACL on Gi0/1 to permit application traffic (TCP 8080, 8443). Connectivity restored. Root cause: Incomplete ACL migration when backup link was provisioned.'
        },
        {
            'problem': 'Switch showing "SPANNING-TREE-2-BLOCK_BPDUGUARD" errors and port Gi0/24 is err-disabled.',
            'diagnosis': 'PortFast and BPDU Guard are enabled on Gi0/24. A switch was connected to this access port, triggering BPDU Guard.',
            'resolution': 'Confirmed Gi0/24 should be access port (user mistakenly connected switch). Re-enabled port with "shutdown" then "no shutdown". Added port description to prevent future mistakes.'
        },
        {
            'problem': 'BGP neighbor 203.0.113.5 is stuck in Active state, never establishing.',
            'diagnosis': 'Neighbor configured with wrong remote AS. Config shows "neighbor 203.0.113.5 remote-as 65002" but actual peer AS is 65003. BGP open message rejected due to AS mismatch.',
            'resolution': 'Corrected neighbor config: "neighbor 203.0.113.5 remote-as 65003". Session established immediately. Received 150 prefixes as expected.'
        },
        {
            'problem': 'OSPF adjacency with dist-switch-02 is flapping every 30 seconds.',
            'diagnosis': 'OSPF hello and dead timers mismatch. Router has default timers (hello 10s, dead 40s) but dist-switch-02 has custom timers (hello 5s, dead 15s). Adjacency fails when dead timer expires.',
            'resolution': 'Standardized OSPF timers on both devices to hello 10s, dead 40s. Adjacency stable. Added to config standards document.'
        },
        {
            'problem': 'VoIP calls are choppy and have poor quality on floor 3.',
            'diagnosis': 'Access switch on floor 3 has no QoS policy. Voice traffic (marked DSCP EF) is not prioritized, competing with data traffic during peak hours.',
            'resolution': 'Applied standard QoS policy to access switch: "service-policy input QOS-TRUST-DSCP" and "mls qos trust dscp" on voice VLANs. Call quality improved immediately.'
        }
    ]

    # Convert to training data
    converter = TicketToTrainingData()
    converter.add_tickets_batch(tickets)

    # Validate and clean
    converter.validate_and_clean()

    # Get statistics
    stats = converter.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Avg tokens/example: {stats['avg_tokens_per_example']}")
    print(f"  Estimated fine-tuning cost: ${stats['estimated_cost']:.2f}")

    # Export
    converter.export_to_jsonl('network_troubleshooting_training.jsonl')
```

### Example Output

```
Validating 5 training examples...
  Removed 0 duplicates
  Removed 0 incomplete examples
  Removed 0 examples with potential PII
✓ Final dataset: 5 examples

Dataset Statistics:
  Total examples: 5
  Estimated tokens: 1,875
  Avg tokens/example: 375
  Estimated fine-tuning cost: $0.02

✓ Exported 5 examples to network_troubleshooting_training.jsonl
```

### Source 2: Config Patterns

Teach the model your organization's config standards.

```python
"""
Generate Training Data from Config Standards
File: fine_tuning/config_to_training_data.py
"""
import json
from typing import List, Dict

class ConfigStandardsTrainingData:
    """Generate training data from config standards."""

    def __init__(self):
        self.training_examples = []

    def add_config_standard(self, config_type: str, good_config: str, explanation: str):
        """
        Add a config standard as training example.

        Args:
            config_type: Type of config (interface, routing, security, etc.)
            good_config: Example of correct config
            explanation: Why this is the standard
        """
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a network configuration expert for Acme Corp. Generate configs following company standards."
                },
                {
                    "role": "user",
                    "content": f"Generate a standard {config_type} configuration for Acme Corp."
                },
                {
                    "role": "assistant",
                    "content": f"{good_config}\n\n# Standard Explanation:\n{explanation}"
                }
            ]
        }

        self.training_examples.append(example)

    def add_config_validation(self, config: str, issues: List[Dict], corrected_config: str):
        """
        Add a config validation example.

        Args:
            config: Config with issues
            issues: List of issues found
            corrected_config: Fixed config
        """
        issues_text = "\n".join([f"- {issue['description']}" for issue in issues])

        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a network config validator for Acme Corp. Identify issues and provide corrections."
                },
                {
                    "role": "user",
                    "content": f"Validate this configuration and fix any issues:\n\n{config}"
                },
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

    def export_to_jsonl(self, output_file: str):
        """Export to JSONL format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self.training_examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Exported {len(self.training_examples)} config examples to {output_file}")


# Example Usage
if __name__ == "__main__":
    generator = ConfigStandardsTrainingData()

    # Add config standards
    generator.add_config_standard(
        config_type="access port",
        good_config="""interface GigabitEthernet0/1
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
        good_config="""interface TenGigabitEthernet1/1
 description TRUNK_TO_DIST-SW-01
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk allowed vlan 10,20,30,100
 switchport nonegotiate
 no shutdown""",
        explanation="Trunk ports use explicit VLAN lists (no 'vlan all'), DTP disabled with 'nonegotiate', and descriptive naming: TRUNK_TO_<destination>"
    )

    # Add config validation examples
    generator.add_config_validation(
        config="""interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10
 no shutdown""",
        issues=[
            {'description': 'Missing description (required for all interfaces)'},
            {'description': 'Missing PortFast (required for access ports)'},
            {'description': 'Missing BPDU Guard (required for access ports)'}
        ],
        corrected_config="""interface GigabitEthernet0/1
 description USER_PORT - Location TBD
 switchport mode access
 switchport access vlan 10
 spanning-tree portfast
 spanning-tree bpduguard enable
 no shutdown"""
    )

    generator.export_to_jsonl('config_standards_training.jsonl')
```

### Training Data Quality Guidelines

**High-quality training data has**:
- **Consistency**: Same style and format across examples
- **Accuracy**: 100% correct outputs (model learns from these)
- **Diversity**: Covers different scenarios, not just one type
- **Relevance**: Matches actual production use cases
- **Volume**: At least 500 examples (more is better)

**Poor training data leads to**:
- Model that repeats mistakes
- Hallucinations (invents configs that don't work)
- Inconsistent outputs

---

## Fine-Tuning Process

### Step 1: Prepare Training Data

Combine all sources into one JSONL file:

```bash
# Merge training data files
cat network_troubleshooting_training.jsonl \
    config_standards_training.jsonl \
    log_parsing_training.jsonl \
    > complete_training_data.jsonl

# Validate format
python validate_training_data.py complete_training_data.jsonl
```

### Step 2: Split Train/Validation

```python
"""
Split Training Data
File: fine_tuning/split_dataset.py
"""
import json
import random

def split_dataset(input_file: str, train_ratio: float = 0.9):
    """
    Split dataset into training and validation sets.

    Args:
        input_file: Path to input JSONL file
        train_ratio: Ratio of data for training (0.9 = 90% train, 10% validation)
    """
    # Load all examples
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    # Shuffle
    random.shuffle(examples)

    # Split
    split_point = int(len(examples) * train_ratio)
    train_examples = examples[:split_point]
    val_examples = examples[split_point:]

    # Write training set
    with open('train.jsonl', 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')

    # Write validation set
    with open('validation.jsonl', 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Split {len(examples)} examples:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")


if __name__ == "__main__":
    split_dataset('complete_training_data.jsonl')
```

### Step 3: Fine-Tune via AWS Bedrock (Claude)

```python
"""
Fine-Tune Claude via AWS Bedrock
File: fine_tuning/bedrock_fine_tune.py
"""
import boto3
import json
from typing import Dict
import time

class BedrockFineTuner:
    """
    Fine-tune Claude models via AWS Bedrock.

    Note: This uses AWS Bedrock's model customization API.
    Requires: AWS account, Bedrock access, S3 bucket for training data
    """

    def __init__(self, region: str = 'us-east-1'):
        """
        Args:
            region: AWS region for Bedrock
        """
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)

    def upload_training_data(self, local_file: str, bucket: str, s3_key: str) -> str:
        """
        Upload training data to S3.

        Args:
            local_file: Path to local JSONL file
            bucket: S3 bucket name
            s3_key: S3 key (path) for the file

        Returns:
            S3 URI
        """
        print(f"Uploading {local_file} to s3://{bucket}/{s3_key}...")

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
                               hyperparameters: Dict = None) -> str:
        """
        Create fine-tuning job.

        Args:
            model_id: Base model to fine-tune (e.g., 'anthropic.claude-3-haiku-20240307-v1:0')
            training_data_uri: S3 URI to training data
            validation_data_uri: S3 URI to validation data
            job_name: Name for this fine-tuning job
            output_bucket: S3 bucket for output model
            hyperparameters: Optional training hyperparameters

        Returns:
            Job ARN
        """
        print(f"\nCreating fine-tuning job: {job_name}")
        print(f"  Base model: {model_id}")
        print(f"  Training data: {training_data_uri}")

        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'epochCount': '3',
                'batchSize': '8',
                'learningRate': '0.00001'
            }

        try:
            response = self.bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=f"{job_name}-model",
                roleArn='arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockFineTuningRole',  # Replace with your IAM role
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

            return job_arn

        except Exception as e:
            print(f"✗ Error creating fine-tuning job: {e}")
            raise

    def get_job_status(self, job_arn: str) -> Dict:
        """
        Get status of fine-tuning job.

        Args:
            job_arn: Job ARN from create_fine_tuning_job

        Returns:
            Dict with job status and metrics
        """
        response = self.bedrock.get_model_customization_job(
            jobIdentifier=job_arn
        )

        return {
            'status': response['status'],
            'training_metrics': response.get('trainingMetrics', {}),
            'validation_metrics': response.get('validationMetrics', {}),
            'failure_message': response.get('failureMessage', None)
        }

    def wait_for_completion(self, job_arn: str, check_interval: int = 300) -> Dict:
        """
        Wait for fine-tuning job to complete.

        Args:
            job_arn: Job ARN
            check_interval: How often to check status (seconds)

        Returns:
            Final job status
        """
        print(f"\nWaiting for fine-tuning job to complete...")
        print(f"This may take 2-6 hours depending on dataset size.")

        while True:
            status = self.get_job_status(job_arn)

            print(f"  Status: {status['status']}")

            if status['status'] == 'Completed':
                print(f"✓ Fine-tuning completed successfully!")
                return status

            elif status['status'] in ['Failed', 'Stopped']:
                print(f"✗ Fine-tuning failed: {status.get('failure_message', 'Unknown error')}")
                return status

            # Still in progress
            time.sleep(check_interval)


# Example Usage
if __name__ == "__main__":
    """
    Note: This is example code. You'll need:
    1. AWS account with Bedrock access
    2. IAM role with permissions
    3. S3 bucket for data
    4. Training data in JSONL format
    """

    print("="*70)
    print("CLAUDE FINE-TUNING VIA AWS BEDROCK")
    print("="*70)

    # Initialize fine-tuner
    tuner = BedrockFineTuner(region='us-east-1')

    # Upload training data
    training_uri = tuner.upload_training_data(
        local_file='train.jsonl',
        bucket='my-bedrock-training-data',
        s3_key='network-troubleshooting/train.jsonl'
    )

    validation_uri = tuner.upload_training_data(
        local_file='validation.jsonl',
        bucket='my-bedrock-training-data',
        s3_key='network-troubleshooting/validation.jsonl'
    )

    # Create fine-tuning job
    job_arn = tuner.create_fine_tuning_job(
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        training_data_uri=training_uri,
        validation_data_uri=validation_uri,
        job_name='network-troubleshooting-v1',
        output_bucket='my-bedrock-models',
        hyperparameters={
            'epochCount': '3',
            'batchSize': '8',
            'learningRate': '0.00001'
        }
    )

    # Wait for completion
    final_status = tuner.wait_for_completion(job_arn)

    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Final Status: {final_status['status']}")
    print(f"Training Loss: {final_status['training_metrics'].get('loss', 'N/A')}")
    print(f"Validation Loss: {final_status['validation_metrics'].get('loss', 'N/A')}")
```

**Important Notes**:
- Fine-tuning takes **2-6 hours** depending on dataset size
- You pay for compute time during training (~$50-500 depending on dataset)
- Once complete, you get a custom model ID to use in API calls
- Custom model pricing is typically same as base model (no extra per-request cost)

---

## Cost-Benefit Analysis

### Real Cost Calculation

**Scenario**: Network troubleshooting assistant, 50,000 requests/month

**Without Fine-Tuning** (prompt engineering approach):
```
Average prompt: 2,500 tokens (includes context about topology, standards, etc.)
Average response: 500 tokens
Total tokens per request: 3,000 tokens

Monthly volume: 50,000 requests × 3,000 tokens = 150M tokens

Cost at Claude pricing:
- Input tokens: 50,000 × 2,500 × $0.003 / 1K = $375
- Output tokens: 50,000 × 500 × $0.015 / 1K = $375
Total monthly cost: $750
```

**With Fine-Tuning**:
```
Average prompt: 500 tokens (no context needed, model knows standards)
Average response: 500 tokens
Total tokens per request: 1,000 tokens

Monthly volume: 50,000 requests × 1,000 tokens = 50M tokens

Cost at Claude pricing:
- Input tokens: 50,000 × 500 × $0.003 / 1K = $75
- Output tokens: 50,000 × 500 × $0.015 / 1K = $375
Total monthly cost: $450

One-time fine-tuning cost: ~$200 (1M training tokens, 3 epochs)

Monthly savings: $750 - $450 = $300/month
Breakeven: $200 / $300 = 0.67 months (~20 days)
Annual savings: $300 × 12 = $3,600
```

**ROI**: Fine-tuning pays for itself in 20 days, saves $3,600/year.

### Break-Even Analysis

```python
"""
Calculate Break-Even Point for Fine-Tuning
File: fine_tuning/breakeven_calculator.py
"""
from typing import Dict

class FineTuningROI:
    """Calculate ROI for fine-tuning decisions."""

    def __init__(self,
                 monthly_requests: int,
                 prompt_tokens_without_finetuning: int,
                 prompt_tokens_with_finetuning: int,
                 response_tokens: int,
                 input_token_cost_per_1k: float = 0.003,
                 output_token_cost_per_1k: float = 0.015):
        """
        Args:
            monthly_requests: Number of requests per month
            prompt_tokens_without_finetuning: Avg prompt size without fine-tuning
            prompt_tokens_with_finetuning: Avg prompt size with fine-tuning
            response_tokens: Avg response size
            input_token_cost_per_1k: Cost per 1K input tokens
            output_token_cost_per_1k: Cost per 1K output tokens
        """
        self.monthly_requests = monthly_requests
        self.prompt_without = prompt_tokens_without_finetuning
        self.prompt_with = prompt_tokens_with_finetuning
        self.response = response_tokens
        self.input_cost = input_token_cost_per_1k
        self.output_cost = output_token_cost_per_1k

    def calculate_monthly_cost(self, use_finetuning: bool) -> float:
        """Calculate monthly API cost."""
        if use_finetuning:
            prompt_tokens = self.prompt_with
        else:
            prompt_tokens = self.prompt_without

        # Calculate costs
        input_cost = (self.monthly_requests * prompt_tokens / 1000) * self.input_cost
        output_cost = (self.monthly_requests * self.response / 1000) * self.output_cost

        return input_cost + output_cost

    def calculate_roi(self, fine_tuning_cost: float) -> Dict:
        """
        Calculate ROI for fine-tuning.

        Args:
            fine_tuning_cost: One-time cost to fine-tune model

        Returns:
            Dict with ROI metrics
        """
        cost_without = self.calculate_monthly_cost(use_finetuning=False)
        cost_with = self.calculate_monthly_cost(use_finetuning=True)

        monthly_savings = cost_without - cost_with

        if monthly_savings <= 0:
            return {
                'roi_positive': False,
                'message': 'Fine-tuning would not save money'
            }

        breakeven_months = fine_tuning_cost / monthly_savings
        annual_savings = monthly_savings * 12

        roi_percentage = ((annual_savings - fine_tuning_cost) / fine_tuning_cost) * 100

        return {
            'roi_positive': True,
            'cost_without_finetuning': cost_without,
            'cost_with_finetuning': cost_with,
            'monthly_savings': monthly_savings,
            'breakeven_months': breakeven_months,
            'breakeven_days': breakeven_months * 30,
            'annual_savings': annual_savings,
            'roi_percentage': roi_percentage,
            'recommendation': 'FINE-TUNE' if breakeven_months < 3 else 'Consider alternatives'
        }

    def print_analysis(self, fine_tuning_cost: float):
        """Print detailed ROI analysis."""
        roi = self.calculate_roi(fine_tuning_cost)

        print("\n" + "="*70)
        print("FINE-TUNING ROI ANALYSIS")
        print("="*70)

        print(f"\nMonthly Request Volume: {self.monthly_requests:,}")

        print(f"\nWithout Fine-Tuning:")
        print(f"  Avg prompt: {self.prompt_without} tokens")
        print(f"  Monthly cost: ${roi['cost_without_finetuning']:,.2f}")

        print(f"\nWith Fine-Tuning:")
        print(f"  Avg prompt: {self.prompt_with} tokens")
        print(f"  Monthly cost: ${roi['cost_with_finetuning']:,.2f}")
        print(f"  One-time fine-tuning cost: ${fine_tuning_cost:,.2f}")

        print(f"\nSavings:")
        print(f"  Monthly: ${roi['monthly_savings']:,.2f}")
        print(f"  Annual: ${roi['annual_savings']:,.2f}")

        print(f"\nBreak-Even:")
        print(f"  {roi['breakeven_months']:.1f} months ({roi['breakeven_days']:.0f} days)")

        print(f"\nROI: {roi['roi_percentage']:.0f}%")

        print(f"\nRecommendation: {roi['recommendation']}")

        if roi['breakeven_months'] < 1:
            print("  ✓ Excellent ROI - fine-tune immediately")
        elif roi['breakeven_months'] < 3:
            print("  ✓ Good ROI - fine-tuning recommended")
        elif roi['breakeven_months'] < 6:
            print("  ⚠️  Marginal ROI - evaluate alternatives")
        else:
            print("  ✗ Poor ROI - stick with prompt engineering")


# Example Usage
if __name__ == "__main__":
    # Scenario: High-volume troubleshooting assistant
    calculator = FineTuningROI(
        monthly_requests=50_000,
        prompt_tokens_without_finetuning=2_500,
        prompt_tokens_with_finetuning=500,
        response_tokens=500
    )

    calculator.print_analysis(fine_tuning_cost=200)

    print("\n" + "="*70)
    print("COMPARISON: Low Volume Scenario")
    print("="*70)

    # Scenario: Low volume
    calculator_low = FineTuningROI(
        monthly_requests=1_000,
        prompt_tokens_without_finetuning=2_500,
        prompt_tokens_with_finetuning=500,
        response_tokens=500
    )

    calculator_low.print_analysis(fine_tuning_cost=200)
```

### Example Output

```
======================================================================
FINE-TUNING ROI ANALYSIS
======================================================================

Monthly Request Volume: 50,000

Without Fine-Tuning:
  Avg prompt: 2,500 tokens
  Monthly cost: $750.00

With Fine-Tuning:
  Avg prompt: 500 tokens
  Monthly cost: $450.00
  One-time fine-tuning cost: $200.00

Savings:
  Monthly: $300.00
  Annual: $3,600.00

Break-Even:
  0.7 months (20 days)

ROI: 1700%

Recommendation: FINE-TUNE
  ✓ Excellent ROI - fine-tune immediately

======================================================================
COMPARISON: Low Volume Scenario
======================================================================

Monthly Request Volume: 1,000

Without Fine-Tuning:
  Avg prompt: 2,500 tokens
  Monthly cost: $15.00

With Fine-Tuning:
  Avg prompt: 500 tokens
  Monthly cost: $9.00
  One-time fine-tuning cost: $200.00

Savings:
  Monthly: $6.00
  Annual: $72.00

Break-Even:
  33.3 months (1000 days)

ROI: -64%

Recommendation: Consider alternatives
  ✗ Poor ROI - stick with prompt engineering
```

---

## Performance Benchmarks

After fine-tuning, evaluate performance improvements.

### Benchmark Suite

```python
"""
Benchmark Fine-Tuned Model
File: fine_tuning/benchmark.py
"""
from anthropic import Anthropic
from typing import List, Dict
import time

class ModelBenchmark:
    """Benchmark base model vs. fine-tuned model."""

    def __init__(self, api_key: str, base_model: str, finetuned_model: str):
        """
        Args:
            api_key: Anthropic API key
            base_model: Base model ID (e.g., 'claude-3-haiku-20240307')
            finetuned_model: Fine-tuned model ID
        """
        self.client = Anthropic(api_key=api_key)
        self.base_model = base_model
        self.finetuned_model = finetuned_model

    def run_benchmark(self, test_cases: List[Dict]) -> Dict:
        """
        Run benchmark on test cases.

        Args:
            test_cases: List of dicts with 'prompt' and 'expected_answer'

        Returns:
            Dict with performance metrics
        """
        print(f"\nRunning benchmark with {len(test_cases)} test cases...\n")

        # Benchmark base model
        print("Testing BASE MODEL...")
        base_results = self._test_model(self.base_model, test_cases)

        # Benchmark fine-tuned model
        print("\nTesting FINE-TUNED MODEL...")
        finetuned_results = self._test_model(self.finetuned_model, test_cases)

        # Compare
        return self._compare_results(base_results, finetuned_results, test_cases)

    def _test_model(self, model: str, test_cases: List[Dict]) -> List[Dict]:
        """Test a model on all test cases."""
        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"  Test {i}/{len(test_cases)}", end='... ')

            start_time = time.time()

            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": test_case['prompt']}]
            )

            latency = time.time() - start_time

            answer = response.content[0].text.strip()

            # Check if answer is correct (simplified - use semantic similarity in production)
            correct = test_case['expected_answer'].lower() in answer.lower()

            results.append({
                'test_case': i,
                'correct': correct,
                'latency': latency,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'answer': answer
            })

            print(f"{'✓' if correct else '✗'} ({latency:.2f}s)")

        return results

    def _compare_results(self, base_results: List[Dict], finetuned_results: List[Dict], test_cases: List[Dict]) -> Dict:
        """Compare base vs fine-tuned performance."""
        # Calculate metrics
        base_accuracy = sum(r['correct'] for r in base_results) / len(base_results)
        finetuned_accuracy = sum(r['correct'] for r in finetuned_results) / len(finetuned_results)

        base_avg_latency = sum(r['latency'] for r in base_results) / len(base_results)
        finetuned_avg_latency = sum(r['latency'] for r in finetuned_results) / len(finetuned_results)

        base_avg_tokens = sum(r['tokens_used'] for r in base_results) / len(base_results)
        finetuned_avg_tokens = sum(r['tokens_used'] for r in finetuned_results) / len(finetuned_results)

        return {
            'test_cases': len(test_cases),
            'base_model': {
                'accuracy': base_accuracy,
                'avg_latency': base_avg_latency,
                'avg_tokens': base_avg_tokens
            },
            'finetuned_model': {
                'accuracy': finetuned_accuracy,
                'avg_latency': finetuned_avg_latency,
                'avg_tokens': finetuned_avg_tokens
            },
            'improvements': {
                'accuracy_improvement': (finetuned_accuracy - base_accuracy) * 100,
                'latency_improvement': ((base_avg_latency - finetuned_avg_latency) / base_avg_latency) * 100,
                'token_reduction': ((base_avg_tokens - finetuned_avg_tokens) / base_avg_tokens) * 100
            }
        }

    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)

        print(f"\nTest Cases: {results['test_cases']}")

        print(f"\nBase Model:")
        print(f"  Accuracy: {results['base_model']['accuracy']*100:.1f}%")
        print(f"  Avg Latency: {results['base_model']['avg_latency']:.3f}s")
        print(f"  Avg Tokens: {results['base_model']['avg_tokens']:.0f}")

        print(f"\nFine-Tuned Model:")
        print(f"  Accuracy: {results['finetuned_model']['accuracy']*100:.1f}%")
        print(f"  Avg Latency: {results['finetuned_model']['avg_latency']:.3f}s")
        print(f"  Avg Tokens: {results['finetuned_model']['avg_tokens']:.0f}")

        print(f"\nImprovements:")
        print(f"  Accuracy: {results['improvements']['accuracy_improvement']:+.1f} percentage points")
        print(f"  Latency: {results['improvements']['latency_improvement']:+.1f}%")
        print(f"  Tokens: {results['improvements']['token_reduction']:+.1f}%")


# Example Usage
if __name__ == "__main__":
    import os

    # Test cases (simplified - use many more in production)
    test_cases = [
        {
            'prompt': 'BGP neighbor is stuck in Active state. Diagnose.',
            'expected_answer': 'remote-as mismatch'
        },
        {
            'prompt': 'Interface showing err-disabled due to BPDU guard. Fix.',
            'expected_answer': 'shutdown then no shutdown'
        },
        {
            'prompt': 'OSPF adjacency is flapping. Diagnose.',
            'expected_answer': 'timer mismatch'
        }
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    benchmark = ModelBenchmark(
        api_key=api_key,
        base_model='claude-3-haiku-20240307',
        finetuned_model='YOUR_FINETUNED_MODEL_ID'  # Replace with actual model ID
    )

    results = benchmark.run_benchmark(test_cases)
    benchmark.print_results(results)
```

### Expected Performance Improvements

**Typical results from fine-tuning**:
- **Accuracy**: +5-15 percentage points (e.g., 85% → 95%)
- **Latency**: -20-40% (shorter prompts = faster inference)
- **Token usage**: -50-70% (much shorter prompts needed)
- **Cost**: -30-50% (fewer tokens per request)

**When fine-tuning doesn't help**:
- Base model already >95% accurate (ceiling effect)
- Training data is low quality
- Test cases different from training data
- Task requires general knowledge (fine-tuning specializes, doesn't add knowledge)

---

## Summary

You now know when and how to fine-tune LLMs for network operations:

**Key Decisions**:
1. **Fine-tune when**: High volume (>10K/month), repetitive context, accuracy <90%, cost-sensitive
2. **Don't fine-tune when**: Low volume, changing requirements, already accurate, limited data

**Process**:
1. **Create training data** from tickets, configs, logs (500+ examples)
2. **Validate quality** (remove duplicates, PII, errors)
3. **Split train/validation** (90/10)
4. **Fine-tune via API** (AWS Bedrock, OpenAI, etc.)
5. **Benchmark performance** (accuracy, latency, tokens)
6. **Calculate ROI** (break-even analysis)

**Real Results**:
- **50K requests/month**: Fine-tuning pays for itself in 20 days, saves $3,600/year
- **1K requests/month**: Fine-tuning takes 3 years to break even—stick with prompts

**Next Chapter**: We'll implement production monitoring and observability for AI systems, showing how to track token usage, detect failures, and optimize costs in production.

---

## What Can Go Wrong?

**1. Model overfits to training data (perfect on training, fails on new data)**
- **Cause**: Too many training epochs, insufficient diversity in training data
- **Fix**: Use validation set, stop training when validation loss stops improving

**2. Fine-tuned model forgets general knowledge**
- **Cause**: Training data too specialized, model learns only specific patterns
- **Fix**: Include diverse examples, use lower learning rate, fewer epochs

**3. Training fails due to data format errors**
- **Cause**: JSONL format incorrect, missing required fields
- **Fix**: Validate training data with provider's validator tool before submitting

**4. Fine-tuned model costs more than expected**
- **Cause**: Didn't account for retraining costs when requirements change
- **Fix**: Calculate total cost of ownership (initial + retraining), not just initial cost

**5. Can't reproduce base model results after fine-tuning**
- **Cause**: Fine-tuning changed model behavior, even for tasks not in training data
- **Fix**: Test thoroughly before deploying, keep base model as fallback

**6. Training data contains biases**
- **Cause**: Historical tickets reflect human biases, model learns them
- **Fix**: Audit training data for bias, balance examples across scenarios

**7. ROI calculation was wrong (actual savings less than projected)**
- **Cause**: Overestimated token savings, underestimated maintenance costs
- **Fix**: Run pilot with real traffic, measure actual savings before full rollout

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-32/`
