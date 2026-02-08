"""
Chapter 45: Token Economics & Cost Optimization
Master LLM cost management for production network operations

This module provides comprehensive tools for token analysis, cost tracking,
budget management, and optimization strategies for LLM deployments.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


class CostTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TokenUsage:
    """Token usage metrics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str
    model: str
    operation: str


@dataclass
class BudgetAlert:
    """Budget threshold alert."""
    alert_type: str
    threshold_percent: float
    current_spend: float
    budget_limit: float
    message: str
    timestamp: str


@dataclass
class CostOptimization:
    """Cost optimization recommendation."""
    strategy: str
    potential_savings_usd: float
    potential_savings_percent: float
    implementation_effort: str
    description: str
    example: str


class TokenEconomicsManager:
    """
    Complete token economics and cost management.

    Features:
    - Real-time cost tracking
    - Budget management and alerts
    - Cost forecasting
    - Optimization recommendations
    - Model cost comparison
    - Usage analytics
    """

    def __init__(self, monthly_budget_usd: float = 1000.0):
        self.monthly_budget = monthly_budget_usd
        self.usage_history: List[TokenUsage] = []
        self.total_spend = 0.0

        # Model pricing (per 1M tokens)
        self.model_pricing = {
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'claude-opus-4-20250115': {'input': 15.00, 'output': 75.00},
            'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku': {'input': 0.25, 'output': 1.25},
            'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.00},
        }

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """Calculate cost for token usage."""
        if model not in self.model_pricing:
            model = 'gpt-3.5-turbo'  # Default

        pricing = self.model_pricing[model]

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost

    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        operation: str = "general"
    ) -> TokenUsage:
        """
        Track token usage and cost.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model used
            operation: Operation type

        Returns:
            TokenUsage record
        """
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            timestamp=datetime.now().isoformat(),
            model=model,
            operation=operation
        )

        self.usage_history.append(usage)
        self.total_spend += cost

        return usage

    def get_budget_status(self) -> Dict:
        """Get current budget status."""
        percent_used = (self.total_spend / self.monthly_budget) * 100

        # Determine tier
        if percent_used < 50:
            tier = CostTier.LOW
        elif percent_used < 75:
            tier = CostTier.MEDIUM
        elif percent_used < 90:
            tier = CostTier.HIGH
        else:
            tier = CostTier.CRITICAL

        return {
            'budget': self.monthly_budget,
            'spent': self.total_spend,
            'remaining': self.monthly_budget - self.total_spend,
            'percent_used': percent_used,
            'tier': tier,
            'daily_burn_rate': self._calculate_daily_burn()
        }

    def _calculate_daily_burn(self) -> float:
        """Calculate average daily spend."""
        if not self.usage_history:
            return 0.0

        # Get last 24 hours
        now = datetime.now()
        day_ago = now - timedelta(days=1)

        recent = [
            u for u in self.usage_history
            if datetime.fromisoformat(u.timestamp) > day_ago
        ]

        return sum(u.cost_usd for u in recent)

    def forecast_month_end(self) -> Dict:
        """Forecast end-of-month spend."""
        daily_burn = self._calculate_daily_burn()

        now = datetime.now()
        days_in_month = 30  # Simplified
        days_elapsed = now.day
        days_remaining = days_in_month - days_elapsed

        projected_total = self.total_spend + (daily_burn * days_remaining)
        over_budget = max(0, projected_total - self.monthly_budget)

        return {
            'current_spend': self.total_spend,
            'daily_burn': daily_burn,
            'days_remaining': days_remaining,
            'projected_total': projected_total,
            'over_budget': over_budget,
            'will_exceed': projected_total > self.monthly_budget
        }

    def check_budget_alerts(self) -> List[BudgetAlert]:
        """Check for budget threshold alerts."""
        alerts = []
        status = self.get_budget_status()

        thresholds = [
            (50, "Budget 50% consumed"),
            (75, "Budget 75% consumed - monitor closely"),
            (90, "Budget 90% consumed - implement restrictions"),
            (100, "Budget exceeded - immediate action required"),
        ]

        for threshold, message in thresholds:
            if status['percent_used'] >= threshold:
                alert = BudgetAlert(
                    alert_type=f"threshold_{threshold}",
                    threshold_percent=threshold,
                    current_spend=status['spent'],
                    budget_limit=status['budget'],
                    message=message,
                    timestamp=datetime.now().isoformat()
                )
                alerts.append(alert)

        return alerts

    def get_usage_by_model(self) -> Dict[str, Dict]:
        """Get usage breakdown by model."""
        by_model = defaultdict(lambda: {'tokens': 0, 'cost': 0.0, 'requests': 0})

        for usage in self.usage_history:
            by_model[usage.model]['tokens'] += usage.total_tokens
            by_model[usage.model]['cost'] += usage.cost_usd
            by_model[usage.model]['requests'] += 1

        return dict(by_model)

    def get_usage_by_operation(self) -> Dict[str, Dict]:
        """Get usage breakdown by operation."""
        by_operation = defaultdict(lambda: {'tokens': 0, 'cost': 0.0, 'requests': 0})

        for usage in self.usage_history:
            by_operation[usage.operation]['tokens'] += usage.total_tokens
            by_operation[usage.operation]['cost'] += usage.cost_usd
            by_operation[usage.operation]['requests'] += 1

        return dict(by_operation)

    def recommend_optimizations(self) -> List[CostOptimization]:
        """
        Recommend cost optimization strategies.

        Returns:
            List of optimization recommendations
        """
        optimizations = []

        # Analyze usage patterns
        by_model = self.get_usage_by_model()

        # 1. Model downgrade opportunities
        expensive_models = ['gpt-4', 'claude-opus-4-20250115']
        for model in expensive_models:
            if model in by_model:
                usage = by_model[model]

                # Suggest cheaper alternative
                if model == 'gpt-4':
                    cheaper = 'gpt-4-turbo'
                    savings = usage['cost'] * 0.66  # 66% cheaper
                else:
                    cheaper = 'claude-3-sonnet'
                    savings = usage['cost'] * 0.80  # 80% cheaper

                optimizations.append(CostOptimization(
                    strategy=f"Downgrade from {model} to {cheaper}",
                    potential_savings_usd=savings,
                    potential_savings_percent=(savings / usage['cost']) * 100,
                    implementation_effort="low",
                    description=f"Use {cheaper} for non-critical operations",
                    example=f"Config validation, log parsing → {cheaper}"
                ))

        # 2. Caching opportunities
        total_cost = sum(u['cost'] for u in by_model.values())
        cache_savings = total_cost * 0.30  # 30% cache hit rate

        optimizations.append(CostOptimization(
            strategy="Implement response caching",
            potential_savings_usd=cache_savings,
            potential_savings_percent=30.0,
            implementation_effort="medium",
            description="Cache repetitive queries and configurations",
            example="Cached: common configs, standard topologies, FAQ answers"
        ))

        # 3. Prompt optimization
        prompt_savings = total_cost * 0.20  # 20% through better prompts

        optimizations.append(CostOptimization(
            strategy="Optimize prompt lengths",
            potential_savings_usd=prompt_savings,
            potential_savings_percent=20.0,
            implementation_effort="low",
            description="Reduce input tokens via concise prompts",
            example="Remove examples for simple tasks, use structured input formats"
        ))

        # 4. Batch processing
        batch_savings = total_cost * 0.15

        optimizations.append(CostOptimization(
            strategy="Batch API requests",
            potential_savings_usd=batch_savings,
            potential_savings_percent=15.0,
            implementation_effort="medium",
            description="Group similar requests to reduce overhead",
            example="Batch analyze 10 configs in one call vs 10 separate calls"
        ))

        # Sort by savings
        optimizations.sort(key=lambda x: x.potential_savings_usd, reverse=True)

        return optimizations

    def compare_model_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        models: Optional[List[str]] = None
    ) -> List[Dict]:
        """Compare costs across models."""
        if models is None:
            models = list(self.model_pricing.keys())

        comparison = []

        for model in models:
            cost = self.calculate_cost(input_tokens, output_tokens, model)

            comparison.append({
                'model': model,
                'cost_usd': cost,
                'cost_per_1k_requests': cost * 1000,
                'input_price_per_1m': self.model_pricing[model]['input'],
                'output_price_per_1m': self.model_pricing[model]['output']
            })

        comparison.sort(key=lambda x: x['cost_usd'])

        return comparison


def example_1_basic_cost_tracking():
    """
    Example 1: Basic cost tracking
    """
    print("=" * 60)
    print("Example 1: Token Usage & Cost Tracking")
    print("=" * 60)

    manager = TokenEconomicsManager(monthly_budget_usd=1000.0)

    # Simulate various operations
    operations = [
        (200, 300, 'gpt-4', 'config_validation'),
        (150, 200, 'claude-3-sonnet', 'log_analysis'),
        (100, 150, 'gpt-3.5-turbo', 'quick_query'),
        (500, 800, 'claude-opus-4-20250115', 'complex_troubleshooting'),
    ]

    print("\n📊 Tracking Usage:\n")

    for input_tok, output_tok, model, operation in operations:
        usage = manager.track_usage(input_tok, output_tok, model, operation)

        print(f"Operation: {operation}")
        print(f"  Model: {model}")
        print(f"  Tokens: {usage.total_tokens:,} ({input_tok} in + {output_tok} out)")
        print(f"  Cost: ${usage.cost_usd:.6f}")
        print()

    print(f"💰 Total Spend: ${manager.total_spend:.4f}")

    print("\n" + "=" * 60 + "\n")


def example_2_budget_monitoring():
    """
    Example 2: Budget monitoring and alerts
    """
    print("=" * 60)
    print("Example 2: Budget Monitoring")
    print("=" * 60)

    manager = TokenEconomicsManager(monthly_budget_usd=100.0)

    # Simulate spending
    for i in range(50):
        manager.track_usage(200, 300, 'gpt-4', 'operation')

    status = manager.get_budget_status()

    print("\n💳 Budget Status:\n")
    print(f"  Budget: ${status['budget']:.2f}")
    print(f"  Spent: ${status['spent']:.2f}")
    print(f"  Remaining: ${status['remaining']:.2f}")
    print(f"  Usage: {status['percent_used']:.1f}%")
    print(f"  Tier: {status['tier'].value.upper()}")
    print(f"  Daily Burn: ${status['daily_burn_rate']:.2f}/day")

    # Check alerts
    alerts = manager.check_budget_alerts()

    if alerts:
        print(f"\n🚨 Active Alerts ({len(alerts)}):\n")
        for alert in alerts[-2:]:  # Show last 2
            print(f"  [{alert.threshold_percent}%] {alert.message}")

    # Forecast
    forecast = manager.forecast_month_end()

    print(f"\n📈 Month-End Forecast:")
    print(f"  Days Remaining: {forecast['days_remaining']}")
    print(f"  Daily Burn: ${forecast['daily_burn']:.2f}")
    print(f"  Projected Total: ${forecast['projected_total']:.2f}")

    if forecast['will_exceed']:
        print(f"  ⚠️  Over Budget: ${forecast['over_budget']:.2f}")
    else:
        print(f"  ✅ Within Budget")

    print("\n" + "=" * 60 + "\n")


def example_3_model_comparison():
    """
    Example 3: Compare model costs
    """
    print("=" * 60)
    print("Example 3: Model Cost Comparison")
    print("=" * 60)

    manager = TokenEconomicsManager()

    # Typical request: 200 input, 300 output tokens
    input_tokens = 200
    output_tokens = 300

    print(f"\nScenario: {input_tokens} input tokens, {output_tokens} output tokens\n")

    comparison = manager.compare_model_costs(input_tokens, output_tokens)

    print(f"{'Model':<25} {'Cost/Request':<15} {'Cost/1K Requests':<20}")
    print("─" * 60)

    for item in comparison:
        print(f"{item['model']:<25} ${item['cost_usd']:<14.6f} ${item['cost_per_1k_requests']:<19.4f}")

    cheapest = comparison[0]
    most_expensive = comparison[-1]

    print(f"\n💡 Insights:")
    print(f"  Cheapest: {cheapest['model']} (${cheapest['cost_usd']:.6f})")
    print(f"  Most Expensive: {most_expensive['model']} (${most_expensive['cost_usd']:.6f})")
    print(f"  Cost Ratio: {most_expensive['cost_usd']/cheapest['cost_usd']:.1f}x")

    print("\n" + "=" * 60 + "\n")


def example_4_usage_analytics():
    """
    Example 4: Usage analytics by model and operation
    """
    print("=" * 60)
    print("Example 4: Usage Analytics")
    print("=" * 60)

    manager = TokenEconomicsManager()

    # Simulate realistic workload
    workload = [
        (200, 300, 'gpt-4', 'config_validation', 20),
        (150, 200, 'claude-3-sonnet', 'log_analysis', 50),
        (100, 150, 'gpt-3.5-turbo', 'quick_query', 100),
        (300, 500, 'claude-3-sonnet', 'troubleshooting', 30),
    ]

    for input_tok, output_tok, model, operation, count in workload:
        for _ in range(count):
            manager.track_usage(input_tok, output_tok, model, operation)

    # By model
    by_model = manager.get_usage_by_model()

    print("\n📊 Usage by Model:\n")
    print(f"{'Model':<25} {'Requests':<12} {'Cost':<12}")
    print("─" * 49)

    for model, stats in sorted(by_model.items(), key=lambda x: x[1]['cost'], reverse=True):
        print(f"{model:<25} {stats['requests']:<12} ${stats['cost']:<11.2f}")

    # By operation
    by_operation = manager.get_usage_by_operation()

    print("\n📊 Usage by Operation:\n")
    print(f"{'Operation':<25} {'Requests':<12} {'Cost':<12}")
    print("─" * 49)

    for operation, stats in sorted(by_operation.items(), key=lambda x: x[1]['cost'], reverse=True):
        print(f"{operation:<25} {stats['requests']:<12} ${stats['cost']:<11.2f}")

    print(f"\n💰 Total: ${manager.total_spend:.2f}")

    print("\n" + "=" * 60 + "\n")


def example_5_optimization_recommendations():
    """
    Example 5: Cost optimization recommendations
    """
    print("=" * 60)
    print("Example 5: Cost Optimization Recommendations")
    print("=" * 60)

    manager = TokenEconomicsManager()

    # Simulate heavy GPT-4 usage
    for _ in range(100):
        manager.track_usage(200, 300, 'gpt-4', 'validation')

    for _ in range(50):
        manager.track_usage(300, 500, 'claude-opus-4-20250115', 'analysis')

    print(f"\n💰 Current Monthly Spend: ${manager.total_spend:.2f}\n")

    # Get recommendations
    optimizations = manager.recommend_optimizations()

    print("🎯 Optimization Opportunities:\n")

    total_potential = 0

    for i, opt in enumerate(optimizations[:4], 1):
        print(f"{i}. {opt.strategy}")
        print(f"   💵 Savings: ${opt.potential_savings_usd:.2f} ({opt.potential_savings_percent:.0f}%)")
        print(f"   🔧 Effort: {opt.implementation_effort.upper()}")
        print(f"   📝 {opt.description}")
        print(f"   Example: {opt.example}")
        print()

        total_potential += opt.potential_savings_usd

    print(f"💡 Total Potential Savings: ${total_potential:.2f}/month")
    print(f"   Annual Impact: ${total_potential * 12:.2f}/year")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n💰 Chapter 45: Token Economics & Cost Optimization")
    print("Master LLM Cost Management\n")

    try:
        example_1_basic_cost_tracking()
        input("Press Enter to continue...")

        example_2_budget_monitoring()
        input("Press Enter to continue...")

        example_3_model_comparison()
        input("Press Enter to continue...")

        example_4_usage_analytics()
        input("Press Enter to continue...")

        example_5_optimization_recommendations()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Track every API call for cost visibility")
        print("- Set budget alerts at 50%, 75%, 90%")
        print("- Model selection can reduce costs 10-100x")
        print("- Caching saves 30-50% on repetitive queries")
        print("- Prompt optimization reduces input tokens 20-30%")
        print("- Budget monitoring prevents surprise bills\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
