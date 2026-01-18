"""
Chapter 23: Fine-Tuning for Specific Network Tasks
Advanced Fine-Tuning Techniques for Network Engineering

Build custom models for network-specific tasks like vendor translation,
config migration, and multi-vendor normalization.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()


def create_vendor_translation_dataset():
    """
    Example 1: Fine-tune for Cisco to Juniper translation
    """
    print("=" * 60)
    print("Example 1: Vendor Translation Training Data")
    print("=" * 60)

    training_pairs = [
        # VLAN configurations
        ("vlan 10\n name USERS", "set vlans USERS vlan-id 10"),
        ("vlan 20\n name SERVERS", "set vlans SERVERS vlan-id 20"),

        # Interface configurations
        (
            "interface GigabitEthernet0/1\n description Uplink\n switchport mode trunk",
            "set interfaces ge-0/0/1 description Uplink\nset interfaces ge-0/0/1 unit 0 family ethernet-switching interface-mode trunk"
        ),
        (
            "interface GigabitEthernet0/2\n switchport mode access\n switchport access vlan 10",
            "set interfaces ge-0/0/2 unit 0 family ethernet-switching interface-mode access\nset interfaces ge-0/0/2 unit 0 family ethernet-switching vlan members 10"
        ),

        # BGP configurations
        (
            "router bgp 65001\n neighbor 10.0.0.1 remote-as 65002",
            "set protocols bgp group EXTERNAL type external\nset protocols bgp group EXTERNAL neighbor 10.0.0.1 peer-as 65002\nset routing-options autonomous-system 65001"
        ),

        # OSPF configurations
        (
            "router ospf 1\n network 10.0.0.0 0.0.0.255 area 0",
            "set protocols ospf area 0.0.0.0 interface ge-0/0/0.0"
        ),
    ]

    formatted_data = []
    for cisco_config, juniper_config in training_pairs:
        formatted_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a network config translator. Convert Cisco IOS to Juniper JunOS commands accurately."
                },
                {
                    "role": "user",
                    "content": f"Translate to Juniper:\n{cisco_config}"
                },
                {
                    "role": "assistant",
                    "content": juniper_config
                }
            ]
        })

    output_file = "vendor_translation_training.jsonl"
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Created {len(training_pairs)} translation pairs")
    print(f"📁 Saved to: {output_file}")
    print("\nSample translation:")
    print("-" * 60)
    print(f"Cisco:\n{training_pairs[0][0]}")
    print(f"\nJuniper:\n{training_pairs[0][1]}")

    print("\n" + "=" * 60 + "\n")
    return output_file


def create_config_migration_dataset():
    """
    Example 2: Fine-tune for legacy to modern config migration
    """
    print("=" * 60)
    print("Example 2: Config Migration Training Data")
    print("=" * 60)

    migrations = [
        {
            "legacy": "access-list 100 permit ip any any",
            "modern": "ip access-list extended ACL_100\n permit ip any any",
            "explanation": "Convert numbered ACL to named ACL"
        },
        {
            "legacy": "line vty 0 4\n transport input telnet",
            "modern": "line vty 0 4\n transport input ssh\n transport output ssh",
            "explanation": "Replace insecure Telnet with SSH"
        },
        {
            "legacy": "snmp-server community public RO",
            "modern": "snmp-server group READONLY v3 auth\nsnmp-server user snmpuser READONLY v3 auth sha AuthPass123 priv aes 128 PrivPass123",
            "explanation": "Upgrade SNMP v2c to v3 with authentication"
        },
        {
            "legacy": "spanning-tree mode pvst",
            "modern": "spanning-tree mode rapid-pvst",
            "explanation": "Upgrade to Rapid PVST+ for faster convergence"
        },
        {
            "legacy": "switchport port-security",
            "modern": "switchport port-security\nswitchport port-security maximum 2\nswitchport port-security violation restrict\nswitchport port-security aging time 120",
            "explanation": "Add comprehensive port security settings"
        },
    ]

    formatted_data = []
    for migration in migrations:
        formatted_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a network config modernization expert. Migrate legacy configs to modern best practices."
                },
                {
                    "role": "user",
                    "content": f"Modernize this config:\n{migration['legacy']}"
                },
                {
                    "role": "assistant",
                    "content": f"{migration['modern']}\n\nExplanation: {migration['explanation']}"
                }
            ]
        })

    output_file = "config_migration_training.jsonl"
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Created {len(migrations)} migration examples")
    print(f"📁 Saved to: {output_file}")
    print("\nSample migration:")
    print("-" * 60)
    print(f"Legacy: {migrations[0]['legacy']}")
    print(f"Modern: {migrations[0]['modern']}")
    print(f"Why: {migrations[0]['explanation']}")

    print("\n" + "=" * 60 + "\n")
    return output_file


def create_troubleshooting_dataset():
    """
    Example 3: Fine-tune for network troubleshooting
    """
    print("=" * 60)
    print("Example 3: Troubleshooting Training Data")
    print("=" * 60)

    troubleshooting_cases = [
        {
            "symptoms": "BGP neighbor stuck in ACTIVE state",
            "diagnosis": "1. Check TCP connectivity on port 179\n2. Verify neighbor IP is reachable\n3. Check ACLs blocking BGP\n4. Confirm AS numbers match",
            "commands": "show ip bgp summary\nshow ip bgp neighbors\nping [neighbor-ip]\nshow access-lists\nshow ip route [neighbor-ip]",
            "root_cause": "Usually ACL blocking TCP/179 or incorrect neighbor IP"
        },
        {
            "symptoms": "OSPF neighbors stuck in EXSTART/EXCHANGE",
            "diagnosis": "1. Check MTU mismatch\n2. Verify network types match\n3. Check area IDs\n4. Verify authentication",
            "commands": "show ip ospf interface\nshow ip ospf neighbor\nshow interfaces | include MTU",
            "root_cause": "Most common: MTU mismatch between neighbors"
        },
        {
            "symptoms": "Interface flapping every few minutes",
            "diagnosis": "1. Check physical layer (cable, SFP)\n2. Review error counters\n3. Check for duplex mismatch\n4. Verify power levels",
            "commands": "show interfaces [interface]\nshow interfaces [interface] counters errors\nshow interfaces [interface] transceiver",
            "root_cause": "Usually bad cable/SFP or duplex mismatch"
        },
    ]

    formatted_data = []
    for case in troubleshooting_cases:
        formatted_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior network troubleshooter. Provide structured diagnosis steps."
                },
                {
                    "role": "user",
                    "content": f"Troubleshoot: {case['symptoms']}"
                },
                {
                    "role": "assistant",
                    "content": f"""Diagnosis Steps:
{case['diagnosis']}

Commands to Run:
{case['commands']}

Most Likely Root Cause:
{case['root_cause']}"""
                }
            ]
        })

    output_file = "troubleshooting_training.jsonl"
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Created {len(troubleshooting_cases)} troubleshooting cases")
    print(f"📁 Saved to: {output_file}")
    print("\nSample case:")
    print("-" * 60)
    print(f"Symptoms: {troubleshooting_cases[0]['symptoms']}")
    print(f"Root Cause: {troubleshooting_cases[0]['root_cause']}")

    print("\n" + "=" * 60 + "\n")
    return output_file


def evaluate_fine_tuned_model():
    """
    Example 4: Evaluate fine-tuned model performance
    """
    print("=" * 60)
    print("Example 4: Model Evaluation Framework")
    print("=" * 60)

    evaluation_metrics = """
EVALUATION METRICS FOR NETWORK TASKS

1. Config Translation:
   - Syntax Accuracy: 95%+ configs should be syntactically valid
   - Semantic Accuracy: 90%+ configs should be functionally equivalent
   - Test: Run configs in simulator (GNS3/EVE-NG)

2. Troubleshooting:
   - Diagnosis Accuracy: Compare to expert diagnoses
   - Command Relevance: % of suggested commands that help
   - Time to Resolution: Track avg time with vs without AI

3. Log Classification:
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1 Score: Harmonic mean of precision and recall
   - Target: F1 > 0.90 for critical events

4. Config Generation:
   - Validation Pass Rate: % configs that pass validation
   - Best Practice Compliance: % configs following standards
   - Manual Edit Rate: % configs needing human correction

TESTING METHODOLOGY:
1. Hold out 10% of data for testing
2. Compare fine-tuned vs base model
3. A/B test in production (shadow mode)
4. Track metrics for 2-4 weeks
5. Iterate based on real-world performance
"""

    print(evaluation_metrics)

    # Sample evaluation results
    print("\n" + "-" * 60)
    print("SAMPLE EVALUATION RESULTS")
    print("-" * 60)

    results = {
        "Base Model (GPT-4o mini)": {
            "Config Translation Accuracy": "78%",
            "Troubleshooting Relevance": "82%",
            "Log Classification F1": "0.85",
            "Avg Latency": "150ms"
        },
        "Fine-Tuned Model": {
            "Config Translation Accuracy": "94%",
            "Troubleshooting Relevance": "91%",
            "Log Classification F1": "0.93",
            "Avg Latency": "140ms"
        }
    }

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric:30s}: {value}")

    print("\n💡 Fine-tuning improved accuracy by 12-16% on network tasks!")
    print("=" * 60 + "\n")


def production_deployment_guide():
    """
    Example 5: Deploy fine-tuned model to production
    """
    print("=" * 60)
    print("Example 5: Production Deployment")
    print("=" * 60)

    deployment_guide = """
PRODUCTION DEPLOYMENT WORKFLOW

Phase 1: PRE-DEPLOYMENT (Week 1)
├── Validate model on test set (95%+ accuracy)
├── Load test API endpoints (100+ req/s)
├── Setup monitoring (latency, errors, costs)
└── Prepare rollback plan

Phase 2: SHADOW DEPLOYMENT (Week 2-3)
├── Run both models in parallel
├── Send production traffic to base model only
├── Log fine-tuned model responses (no user impact)
└── Compare outputs: base vs fine-tuned

Phase 3: CANARY DEPLOYMENT (Week 4)
├── Route 10% of traffic to fine-tuned model
├── Monitor error rates closely
├── Collect user feedback
└── Gradually increase to 50%

Phase 4: FULL DEPLOYMENT (Week 5+)
├── Route 100% traffic to fine-tuned model
├── Monitor for 2 weeks
├── Decommission base model if stable
└── Setup quarterly retraining schedule

MONITORING CHECKLIST:
✓ Latency (p50, p95, p99)
✓ Error rate (4xx, 5xx)
✓ Cost per request
✓ User satisfaction scores
✓ Task completion rate
"""

    print(deployment_guide)

    # Example monitoring code
    print("\n" + "-" * 60)
    print("MONITORING CODE EXAMPLE")
    print("-" * 60)
    print("""
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelMetrics:
    model_name: str
    latency_ms: float
    success: bool
    cost: float
    user_rating: Optional[int] = None

class ModelMonitor:
    def __init__(self):
        self.metrics = []

    def log_request(self, metric: ModelMetrics):
        self.metrics.append(metric)

    def get_stats(self, window_minutes: int = 60):
        recent = [m for m in self.metrics
                  if time.time() - m.timestamp < window_minutes * 60]

        return {
            "requests": len(recent),
            "avg_latency_ms": sum(m.latency_ms for m in recent) / len(recent),
            "success_rate": sum(1 for m in recent if m.success) / len(recent),
            "total_cost": sum(m.cost for m in recent),
            "avg_rating": sum(m.user_rating for m in recent if m.user_rating) /
                         sum(1 for m in recent if m.user_rating)
        }

# Usage
monitor = ModelMonitor()
monitor.log_request(ModelMetrics(
    model_name="ft:gpt-3.5-turbo-network-v1",
    latency_ms=145,
    success=True,
    cost=0.0012,
    user_rating=5
))

stats = monitor.get_stats(window_minutes=60)
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
""")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🚀 Chapter 23: Network-Specific Fine-Tuning")
    print("Advanced Fine-Tuning for Production Use\n")

    try:
        # Run examples
        vendor_file = create_vendor_translation_dataset()
        input("Press Enter to continue...")

        migration_file = create_config_migration_dataset()
        input("Press Enter to continue...")

        troubleshoot_file = create_troubleshooting_dataset()
        input("Press Enter to continue...")

        evaluate_fine_tuned_model()
        input("Press Enter to continue...")

        production_deployment_guide()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Network-specific tasks benefit greatly from fine-tuning")
        print("- Vendor translation can reach 95%+ accuracy")
        print("- Always evaluate before production deployment")
        print("- Use shadow/canary deployment for safety")
        print("- Monitor metrics continuously\n")

        print(f"\n📁 Training files created:")
        print(f"  - {vendor_file}")
        print(f"  - {migration_file}")
        print(f"  - {troubleshoot_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
