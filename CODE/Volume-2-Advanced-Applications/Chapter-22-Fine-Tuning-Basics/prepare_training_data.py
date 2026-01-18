"""
Chapter 22: Fine-Tuning for Network Tasks
Prepare Training Data and Fine-Tune Models

Learn how to prepare training data and fine-tune LLMs for
specialized network engineering tasks.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def create_training_examples_syslog():
    """
    Example 1: Create training data for syslog classification
    """
    print("=" * 60)
    print("Example 1: Syslog Classification Training Data")
    print("=" * 60)

    training_data = [
        {
            "input": "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired",
            "output": "CRITICAL - BGP neighbor down, network connectivity impacted"
        },
        {
            "input": "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up",
            "output": "INFO - Interface came up, normal operation"
        },
        {
            "input": "%SYS-5-CONFIG_I: Configured from console by admin on vty0",
            "output": "INFO - Configuration change logged"
        },
        {
            "input": "%SEC-6-IPACCESSLOGP: list 100 denied tcp 192.168.1.50 -> 10.0.0.1",
            "output": "MEDIUM - Security ACL denied traffic, investigate source"
        },
        {
            "input": "%OSPF-5-ADJCHG: Process 1, Nbr 10.2.2.2 on Gi0/1 from FULL to DOWN",
            "output": "CRITICAL - OSPF neighbor down, routing affected"
        }
    ]

    # Convert to fine-tuning format (OpenAI style)
    formatted_data = []
    for example in training_data:
        formatted_data.append({
            "messages": [
                {"role": "system", "content": "You are a network log classifier. Classify severity and provide brief analysis."},
                {"role": "user", "content": f"Classify: {example['input']}"},
                {"role": "assistant", "content": example['output']}
            ]
        })

    # Save to JSONL
    output_file = "syslog_training_data.jsonl"
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Created {len(training_data)} training examples")
    print(f"📁 Saved to: {output_file}")
    print("\nSample training example:")
    print("-" * 60)
    print(json.dumps(formatted_data[0], indent=2))

    print("\n" + "=" * 60 + "\n")
    return output_file


def create_training_examples_config_generation():
    """
    Example 2: Training data for config generation
    """
    print("=" * 60)
    print("Example 2: Config Generation Training Data")
    print("=" * 60)

    training_data = [
        {
            "input": "Configure VLAN 10 for users with description USER_VLAN",
            "output": "vlan 10\n name USER_VLAN\n exit"
        },
        {
            "input": "Configure interface Gi0/1 as access port in VLAN 20",
            "output": "interface GigabitEthernet0/1\n switchport mode access\n switchport access vlan 20\n no shutdown"
        },
        {
            "input": "Enable OSPF process 1 on network 10.0.0.0/24 in area 0",
            "output": "router ospf 1\n network 10.0.0.0 0.0.0.255 area 0"
        },
        {
            "input": "Configure BGP 65001 with neighbor 10.1.1.1 in AS 65002",
            "output": "router bgp 65001\n neighbor 10.1.1.1 remote-as 65002\n neighbor 10.1.1.1 activate"
        },
        {
            "input": "Create ACL 100 to permit HTTPS from any to any",
            "output": "access-list 100 permit tcp any any eq 443"
        }
    ]

    formatted_data = []
    for example in training_data:
        formatted_data.append({
            "messages": [
                {"role": "system", "content": "You are a network config generator. Generate precise Cisco IOS commands."},
                {"role": "user", "content": example['input']},
                {"role": "assistant", "content": example['output']}
            ]
        })

    output_file = "config_gen_training_data.jsonl"
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Created {len(training_data)} training examples")
    print(f"📁 Saved to: {output_file}")
    print("\nSample training example:")
    print("-" * 60)
    print(json.dumps(formatted_data[0], indent=2))

    print("\n" + "=" * 60 + "\n")
    return output_file


def validate_training_data(file_path: str):
    """
    Example 3: Validate training data quality
    """
    print("=" * 60)
    print("Example 3: Training Data Validation")
    print("=" * 60)

    print(f"\nValidating: {file_path}\n")

    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Total examples: {len(data)}")

    # Check format
    issues = []
    for i, example in enumerate(data):
        if "messages" not in example:
            issues.append(f"Example {i}: Missing 'messages' field")
            continue

        messages = example["messages"]

        if len(messages) < 2:
            issues.append(f"Example {i}: Need at least user + assistant messages")

        # Check roles
        roles = [msg["role"] for msg in messages]
        if "user" not in roles:
            issues.append(f"Example {i}: Missing 'user' role")
        if "assistant" not in roles:
            issues.append(f"Example {i}: Missing 'assistant' role")

        # Check content
        for msg in messages:
            if not msg.get("content"):
                issues.append(f"Example {i}: Empty content in {msg['role']} message")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✅ All validation checks passed!")

    # Statistics
    print("\nStatistics:")
    print("-" * 60)
    total_tokens = 0
    for example in data:
        for msg in example["messages"]:
            # Rough token estimate
            total_tokens += len(msg["content"].split()) * 1.3

    print(f"Estimated tokens: ~{int(total_tokens)}")
    print(f"Avg tokens/example: ~{int(total_tokens/len(data))}")

    # Cost estimate (OpenAI fine-tuning as reference)
    cost_per_1k = 0.008  # GPT-3.5 fine-tuning cost
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    print(f"Estimated training cost: ${estimated_cost:.4f}")

    print("\n" + "=" * 60 + "\n")


def prepare_real_world_dataset():
    """
    Example 4: Create a larger, realistic training dataset
    """
    print("=" * 60)
    print("Example 4: Real-World Training Dataset")
    print("=" * 60)

    # Simulate creating a dataset from real network operations
    print("\nScenario: Build dataset from 6 months of network operations\n")

    dataset_stats = {
        "syslog_classification": {
            "examples": 10000,
            "sources": ["Core routers", "Edge switches", "Firewalls"],
            "labels": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
            "collection_period": "6 months"
        },
        "config_generation": {
            "examples": 2000,
            "sources": ["Change requests", "Approved configs", "Templates"],
            "types": ["VLAN", "BGP", "OSPF", "ACL", "Interface"],
            "collection_period": "6 months"
        },
        "troubleshooting": {
            "examples": 500,
            "sources": ["Ticket system", "RCA documents", "Runbooks"],
            "types": ["BGP", "OSPF", "Interface", "Hardware", "Config"],
            "collection_period": "12 months"
        }
    }

    print("Dataset Composition:")
    print("-" * 60)

    total_examples = 0
    for task, stats in dataset_stats.items():
        print(f"\n{task.upper()}")
        print(f"  Examples: {stats['examples']:,}")
        print(f"  Sources: {', '.join(stats['sources'])}")
        if 'labels' in stats:
            print(f"  Labels: {', '.join(stats['labels'])}")
        if 'types' in stats:
            print(f"  Types: {', '.join(stats['types'])}")
        print(f"  Period: {stats['collection_period']}")
        total_examples += stats['examples']

    print(f"\nTotal: {total_examples:,} training examples")

    # Best practices
    print("\n📋 Data Preparation Best Practices:")
    print("-" * 60)
    best_practices = [
        "Use real production data (sanitized)",
        "Balance class distribution (avoid 90% INFO logs)",
        "Include edge cases and errors",
        "Validate with domain experts",
        "Split: 80% train, 10% validation, 10% test",
        "Track data version and lineage",
        "Regular data refresh (quarterly)"
    ]
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

    print("\n" + "=" * 60 + "\n")


def fine_tuning_workflow_guide():
    """
    Example 5: Complete fine-tuning workflow
    """
    print("=" * 60)
    print("Example 5: Fine-Tuning Workflow")
    print("=" * 60)

    workflow = """
STEP 1: PREPARE DATA
├── Collect examples from production
├── Clean and normalize data
├── Create train/val/test splits
└── Format as JSONL

STEP 2: VALIDATE DATA
├── Check format (messages array)
├── Verify all examples have user + assistant
├── Validate content quality
└── Estimate token count and cost

STEP 3: UPLOAD & TRAIN (OpenAI Example)
├── Upload training file: openai.File.create()
├── Create fine-tune job: openai.FineTuningJob.create()
├── Monitor progress: openai.FineTuningJob.retrieve()
└── Wait for completion (hours to days)

STEP 4: EVALUATE
├── Test on validation set
├── Compare to base model
├── Check for overfitting
└── Measure task-specific metrics

STEP 5: DEPLOY
├── Update API calls to use fine-tuned model
├── A/B test against base model
├── Monitor performance in production
└── Iterate based on feedback

COST ESTIMATE (GPT-3.5 Turbo):
├── Training: ~$0.008 per 1K tokens
├── 10K examples × 100 tokens = 1M tokens
└── Cost: ~$8 for training data preparation
"""

    print(workflow)

    # Code example
    print("\nCode Example (OpenAI):")
    print("-" * 60)
    print("""
import openai

# 1. Upload training data
with open("training_data.jsonl", "rb") as f:
    response = openai.File.create(file=f, purpose="fine-tune")
    file_id = response["id"]

# 2. Create fine-tuning job
job = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": 3}
)

# 3. Monitor progress
job_id = job["id"]
status = openai.FineTuningJob.retrieve(job_id)
print(f"Status: {status['status']}")

# 4. Use fine-tuned model
fine_tuned_model = status["fine_tuned_model"]
response = openai.ChatCompletion.create(
    model=fine_tuned_model,
    messages=[{"role": "user", "content": "Classify: %BGP-5-ADJCHANGE"}]
)
""")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🎓 Chapter 22: Fine-Tuning for Network Tasks")
    print("Prepare Training Data & Fine-Tune Models\n")

    try:
        # Run examples
        syslog_file = create_training_examples_syslog()
        input("Press Enter to continue...")

        config_file = create_training_examples_config_generation()
        input("Press Enter to continue...")

        validate_training_data(syslog_file)
        input("Press Enter to continue...")

        prepare_real_world_dataset()
        input("Press Enter to continue...")

        fine_tuning_workflow_guide()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Quality data is more important than quantity")
        print("- Start with 100-1000 examples for most tasks")
        print("- Validate data before training")
        print("- Fine-tuning costs $8-50 for typical datasets")
        print("- Always test fine-tuned model vs base model\n")

        print(f"\n📁 Training files created:")
        print(f"  - {syslog_file}")
        print(f"  - {config_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
