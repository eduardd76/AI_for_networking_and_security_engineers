"""
Chapter 3: Model Selection
Choose the Right Model for Your Network Automation Task

This script helps you select the optimal LLM based on:
- Task complexity
- Latency requirements
- Cost constraints
- Data privacy needs

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import time

# Load environment variables
load_dotenv()


# Model Specifications (as of 2024)
MODELS = {
    "gpt-4o": {
        "provider": "OpenAI",
        "type": "proprietary",
        "context_window": 128000,
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "latency_ms": 500,
        "reasoning": "high",
        "use_cases": ["Complex RCA", "Multi-step analysis", "Critical decisions"]
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "type": "proprietary",
        "context_window": 128000,
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "latency_ms": 80,
        "reasoning": "medium",
        "use_cases": ["Log classification", "Config parsing", "Quick answers"]
    },
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "type": "proprietary",
        "context_window": 200000,
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "latency_ms": 400,
        "reasoning": "very-high",
        "use_cases": ["RCA", "Security audits", "Config generation", "Large docs"]
    },
    "claude-3-5-haiku": {
        "provider": "Anthropic",
        "type": "proprietary",
        "context_window": 200000,
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "latency_ms": 50,
        "reasoning": "medium",
        "use_cases": ["Log parsing", "Fast classification", "Real-time alerts"]
    },
    "llama-3.1-8b": {
        "provider": "Meta",
        "type": "open-source",
        "context_window": 128000,
        "input_cost_per_1m": 0.00,  # Self-hosted
        "output_cost_per_1m": 0.00,
        "latency_ms": 20,
        "reasoning": "medium",
        "infrastructure_cost": "$50-100/month (GPU)",
        "use_cases": ["High-volume classification", "On-prem only", "Low latency"]
    },
    "llama-3.1-70b": {
        "provider": "Meta",
        "type": "open-source",
        "context_window": 128000,
        "input_cost_per_1m": 0.00,
        "output_cost_per_1m": 0.00,
        "latency_ms": 150,
        "reasoning": "high",
        "infrastructure_cost": "$500-1000/month (GPU)",
        "use_cases": ["Complex analysis", "PCI-DSS compliance", "Air-gapped"]
    }
}


def analyze_task_requirements(
    task_description: str,
    volume: str,
    complexity: str,
    data_sensitivity: str,
    latency_requirement: str
) -> Dict[str, Any]:
    """
    Example 1: Analyze task and recommend model
    """
    print("=" * 60)
    print("Example 1: Model Selection Analysis")
    print("=" * 60)

    print(f"\nTask: {task_description}")
    print(f"Volume: {volume}")
    print(f"Complexity: {complexity}")
    print(f"Data Sensitivity: {data_sensitivity}")
    print(f"Latency: {latency_requirement}\n")

    # Selection logic
    recommendations = []

    # Rule 1: Data sensitivity trumps everything
    if data_sensitivity == "high" or "PCI" in data_sensitivity or "GDPR" in data_sensitivity:
        recommendations.append({
            "model": "llama-3.1-70b",
            "reason": "Data cannot leave premises (compliance requirement)",
            "priority": 1
        })
        recommendations.append({
            "model": "llama-3.1-8b",
            "reason": "Backup option if complexity allows",
            "priority": 2
        })

    # Rule 2: High volume + low complexity = fast, cheap model
    elif "high" in volume.lower() and complexity.lower() == "low":
        recommendations.append({
            "model": "claude-3-5-haiku",
            "reason": "Ultra-fast (50ms), low cost, handles high volume",
            "priority": 1
        })
        recommendations.append({
            "model": "llama-3.1-8b",
            "reason": "Even cheaper if self-hosted infrastructure available",
            "priority": 2
        })

    # Rule 3: High complexity = best reasoning model
    elif complexity.lower() == "high":
        recommendations.append({
            "model": "claude-3-5-sonnet",
            "reason": "Best reasoning + large context (200K tokens)",
            "priority": 1
        })
        recommendations.append({
            "model": "gpt-4o",
            "reason": "Alternative with strong reasoning",
            "priority": 2
        })

    # Rule 4: Balanced general-purpose
    else:
        recommendations.append({
            "model": "gpt-4o-mini",
            "reason": "Best balance of cost, speed, and capability",
            "priority": 1
        })
        recommendations.append({
            "model": "claude-3-5-haiku",
            "reason": "Faster alternative with good quality",
            "priority": 2
        })

    print("Recommendations:")
    print("-" * 60)
    for rec in recommendations:
        model_info = MODELS[rec["model"]]
        print(f"\n{rec['priority']}. {rec['model'].upper()}")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Cost: ${model_info['input_cost_per_1m']:.2f}/$1M input")
        if "infrastructure_cost" in model_info:
            print(f"   Infrastructure: {model_info['infrastructure_cost']}")
        print(f"   Latency: {model_info['latency_ms']}ms")

    print("\n" + "=" * 60 + "\n")
    return recommendations[0]["model"] if recommendations else "gpt-4o-mini"


def test_model_performance():
    """
    Example 2: Compare model performance on real task
    """
    print("=" * 60)
    print("Example 2: Real Model Performance Comparison")
    print("=" * 60)

    # Test task: Classify syslog severity
    test_log = "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired"
    prompt = f"""Classify this syslog message severity:
{test_log}

Respond with ONLY: CRITICAL, HIGH, MEDIUM, or LOW"""

    models_to_test = []

    # Test Anthropic models if key available
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend([
            ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ])

    # Test OpenAI models if key available
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.extend([
            ("gpt-4o-mini", "GPT-4o mini"),
        ])

    if not models_to_test:
        print("⚠️  No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY to test models.")
        return

    print(f"\nTest Prompt: {prompt}\n")
    print("Performance Results:")
    print("-" * 60)

    results = []

    for model_id, model_name in models_to_test:
        try:
            # Measure latency
            start_time = time.time()

            if "claude" in model_id:
                llm = ChatAnthropic(model=model_id, temperature=0)
            else:
                llm = ChatOpenAI(model=model_id, temperature=0)

            response = llm.invoke(prompt)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            # Estimate tokens (rough)
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response.content.split()) * 1.3

            # Calculate cost
            model_key = model_id.replace("-20241022", "").replace("-", "-")
            if "haiku" in model_id:
                cost = (input_tokens / 1000000 * 0.80) + (output_tokens / 1000000 * 4.00)
            elif "sonnet" in model_id:
                cost = (input_tokens / 1000000 * 3.00) + (output_tokens / 1000000 * 15.00)
            elif "gpt-4o-mini" in model_id:
                cost = (input_tokens / 1000000 * 0.15) + (output_tokens / 1000000 * 0.60)
            else:
                cost = 0

            results.append({
                "model": model_name,
                "response": response.content.strip(),
                "latency": latency,
                "cost": cost
            })

            print(f"\n{model_name}:")
            print(f"  Response: {response.content.strip()}")
            print(f"  Latency:  {latency:.0f}ms")
            print(f"  Cost:     ${cost:.8f}")

        except Exception as e:
            print(f"\n{model_name}: Error - {str(e)}")

    if results:
        print("\n📊 Summary:")
        print("-" * 60)
        fastest = min(results, key=lambda x: x["latency"])
        cheapest = min(results, key=lambda x: x["cost"])

        print(f"Fastest: {fastest['model']} ({fastest['latency']:.0f}ms)")
        print(f"Cheapest: {cheapest['model']} (${cheapest['cost']:.8f})")

    print("\n" + "=" * 60 + "\n")


def cost_projection_calculator():
    """
    Example 3: Calculate costs for production workload
    """
    print("=" * 60)
    print("Example 3: Production Cost Projection")
    print("=" * 60)

    # Scenario parameters
    scenarios = [
        {
            "name": "Syslog Classification",
            "requests_per_day": 100000,
            "avg_input_tokens": 50,
            "avg_output_tokens": 5,
            "description": "Classify 100K syslog messages/day"
        },
        {
            "name": "Config Audit",
            "requests_per_day": 500,
            "avg_input_tokens": 2000,
            "avg_output_tokens": 500,
            "description": "Audit 500 device configs/day"
        },
        {
            "name": "RCA Investigation",
            "requests_per_day": 50,
            "avg_input_tokens": 10000,
            "avg_output_tokens": 2000,
            "description": "50 deep root cause analyses/day"
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"  Volume: {scenario['requests_per_day']:,} requests/day")
        print(f"  Input: ~{scenario['avg_input_tokens']} tokens")
        print(f"  Output: ~{scenario['avg_output_tokens']} tokens\n")

        # Calculate monthly costs for different models
        days_per_month = 30
        monthly_requests = scenario['requests_per_day'] * days_per_month
        monthly_input_tokens = monthly_requests * scenario['avg_input_tokens']
        monthly_output_tokens = monthly_requests * scenario['avg_output_tokens']

        print("  Monthly Cost by Model:")

        for model_name, model_info in MODELS.items():
            if model_info["type"] == "proprietary":
                input_cost = (monthly_input_tokens / 1000000) * model_info["input_cost_per_1m"]
                output_cost = (monthly_output_tokens / 1000000) * model_info["output_cost_per_1m"]
                total_cost = input_cost + output_cost

                print(f"    {model_name:20s}: ${total_cost:>8.2f}/month")

        print("-" * 60)

    print("\n" + "=" * 60 + "\n")


def decision_framework():
    """
    Example 4: Interactive decision framework
    """
    print("=" * 60)
    print("Example 4: Model Selection Decision Framework")
    print("=" * 60)

    print("""
Decision Tree:

1. Is data regulated (PCI-DSS, GDPR, HIPAA)?
   YES → Use open-source model (Llama) self-hosted
   NO → Continue to step 2

2. What is the task complexity?
   LOW (classification, parsing) → Continue to step 3
   HIGH (RCA, generation) → Continue to step 4

3. Low Complexity Tasks:
   - High volume (>10K/day)?
     YES → Claude 3.5 Haiku (fastest + cheap)
     NO → GPT-4o mini (cheapest per call)

4. High Complexity Tasks:
   - Large context needed (>50K tokens)?
     YES → Claude 3.5 Sonnet (200K context)
     NO → GPT-4o (best reasoning)

5. Budget Constraints:
   - Very tight budget → Always use Haiku/mini or self-host Llama
   - No budget constraints → Use Sonnet/GPT-4o for best quality

6. Latency Requirements:
   - < 100ms → Claude Haiku or self-hosted Llama 8B
   - < 500ms → Any proprietary model works
   - > 500ms OK → Consider larger self-hosted models
    """)

    print("=" * 60 + "\n")


def real_world_use_cases():
    """
    Example 5: Map real network tasks to models
    """
    print("=" * 60)
    print("Example 5: Real-World Use Case Mapping")
    print("=" * 60)

    use_cases = [
        {
            "task": "Real-time syslog alerting (100K logs/day)",
            "recommended": "claude-3-5-haiku",
            "reason": "Ultra-fast (50ms), handles high volume, low cost"
        },
        {
            "task": "BGP flap root cause analysis (10 incidents/day)",
            "recommended": "claude-3-5-sonnet",
            "reason": "Complex reasoning, large context for multiple logs"
        },
        {
            "task": "Firewall rule compliance audit (1000 rules, PCI-DSS)",
            "recommended": "llama-3.1-70b",
            "reason": "Data cannot leave premises (compliance), needs good reasoning"
        },
        {
            "task": "Auto-generate change documentation (50 changes/day)",
            "recommended": "gpt-4o-mini",
            "reason": "Good quality output, low cost, medium volume"
        },
        {
            "task": "Security policy generation from requirements",
            "recommended": "claude-3-5-sonnet",
            "reason": "Critical use case, needs best reasoning and accuracy"
        },
        {
            "task": "Interface description from show command",
            "recommended": "claude-3-5-haiku",
            "reason": "Simple extraction task, fast and cheap"
        }
    ]

    print("\nTask → Model Mapping:")
    print("-" * 60)

    for i, uc in enumerate(use_cases, 1):
        model_info = MODELS[uc["recommended"]]
        print(f"\n{i}. {uc['task']}")
        print(f"   Recommended: {uc['recommended'].upper()}")
        print(f"   Reason: {uc['reason']}")
        print(f"   Cost: ${model_info['input_cost_per_1m']:.2f}/$1M")
        if "infrastructure_cost" in model_info:
            print(f"   Infrastructure: {model_info['infrastructure_cost']}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🎯 Chapter 3: Model Selection")
    print("Choose the Right LLM for Your Network Task\n")

    try:
        # Run examples
        analyze_task_requirements(
            task_description="Classify incoming syslog messages",
            volume="100K logs/day",
            complexity="Low",
            data_sensitivity="Public",
            latency_requirement="< 100ms"
        )
        input("Press Enter to continue...")

        test_model_performance()
        input("Press Enter to continue...")

        cost_projection_calculator()
        input("Press Enter to continue...")

        decision_framework()
        input("Press Enter to continue...")

        real_world_use_cases()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Model selection depends on: complexity, volume, cost, data sensitivity")
        print("- Cheap models (Haiku/mini) work well for simple tasks")
        print("- Use Sonnet/GPT-4o for complex reasoning")
        print("- Self-host Llama for regulated data")
        print("- Always test with your actual workload\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
