"""
Chapter 42: Deployment Architectures & Decision Frameworks
Choose the right architecture for AI-powered network operations

This module provides decision frameworks, cost models, and architecture
patterns for deploying LLM systems in network engineering environments.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class DeploymentType(str, Enum):
    CLOUD_API = "cloud_api"
    HYBRID = "hybrid"
    ON_PREMISES = "on_premises"
    EDGE = "edge"


class ScalingPattern(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    AUTO_SCALING = "auto_scaling"


@dataclass
class DeploymentRequirement:
    """Deployment requirements specification."""
    daily_requests: int
    data_residency: str  # "cloud", "on-prem", "hybrid"
    latency_requirement_ms: int
    team_size: int
    budget_monthly_usd: float
    compliance_requirements: List[str]
    uptime_sla: float  # 0.99, 0.999, 0.9999


@dataclass
class ArchitectureOption:
    """Architecture deployment option."""
    name: str
    deployment_type: DeploymentType
    description: str
    initial_cost_usd: float
    monthly_cost_usd: float
    latency_ms: Tuple[int, int]  # (min, max)
    scalability: ScalingPattern
    complexity: str  # "low", "medium", "high"
    pros: List[str]
    cons: List[str]
    best_for: List[str]


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    infrastructure: float
    api_calls: float
    storage: float
    bandwidth: float
    support: float
    total: float


class ArchitectureRecommendation(BaseModel):
    """LLM-powered architecture recommendation."""
    recommended_architecture: str = Field(description="Primary recommendation")
    reasoning: str = Field(description="Why this architecture fits best")
    estimated_monthly_cost: float = Field(description="Monthly cost in USD")
    implementation_steps: List[str] = Field(description="Steps to implement")
    risks: List[str] = Field(description="Potential risks and mitigations")
    alternative_options: List[str] = Field(description="Alternative architectures to consider")


class ArchitectureDecisionFramework:
    """
    Framework for choosing deployment architecture.

    Architectures:
    - Cloud API: OpenAI/Anthropic APIs
    - Hybrid: Cloud + local caching/preprocessing
    - On-Premises: Self-hosted models
    - Edge: Models on network devices
    """

    def __init__(self):
        self.architectures = self._define_architectures()

    def _define_architectures(self) -> List[ArchitectureOption]:
        """Define available architecture options."""
        return [
            ArchitectureOption(
                name="Cloud API (Pay-per-use)",
                deployment_type=DeploymentType.CLOUD_API,
                description="Use cloud provider APIs (OpenAI, Anthropic)",
                initial_cost_usd=0,
                monthly_cost_usd=0,  # Variable based on usage
                latency_ms=(200, 2000),
                scalability=ScalingPattern.AUTO_SCALING,
                complexity="low",
                pros=[
                    "Zero infrastructure management",
                    "Automatic model updates",
                    "Instant scalability",
                    "No upfront costs",
                    "Pay only for what you use"
                ],
                cons=[
                    "Data leaves premises",
                    "Variable latency",
                    "Cost unpredictable at scale",
                    "Vendor lock-in risk",
                    "Rate limits during spikes"
                ],
                best_for=[
                    "Low volume (< 10K requests/day)",
                    "Proof of concept",
                    "Variable workloads",
                    "Small teams",
                    "Quick time-to-market"
                ]
            ),
            ArchitectureOption(
                name="Hybrid (Cloud + Caching)",
                deployment_type=DeploymentType.HYBRID,
                description="Cloud APIs with local caching and preprocessing",
                initial_cost_usd=5_000,
                monthly_cost_usd=2_000,
                latency_ms=(50, 1500),
                scalability=ScalingPattern.HORIZONTAL,
                complexity="medium",
                pros=[
                    "Reduced API costs (caching)",
                    "Lower latency for cached queries",
                    "Better cost predictability",
                    "Local data preprocessing",
                    "Failover to cloud"
                ],
                cons=[
                    "Cache management complexity",
                    "Some data still goes to cloud",
                    "Infrastructure to maintain",
                    "Cache invalidation challenges"
                ],
                best_for=[
                    "Medium volume (10K-100K/day)",
                    "Repetitive queries",
                    "Cost-conscious deployments",
                    "Need local preprocessing"
                ]
            ),
            ArchitectureOption(
                name="On-Premises (Self-hosted)",
                deployment_type=DeploymentType.ON_PREMISES,
                description="Self-hosted models on company infrastructure",
                initial_cost_usd=50_000,
                monthly_cost_usd=5_000,
                latency_ms=(20, 200),
                scalability=ScalingPattern.VERTICAL,
                complexity="high",
                pros=[
                    "Complete data control",
                    "Predictable costs at scale",
                    "No rate limits",
                    "Custom model fine-tuning",
                    "Lowest latency",
                    "No vendor dependency"
                ],
                cons=[
                    "High upfront cost",
                    "Requires ML/GPU expertise",
                    "Manual model updates",
                    "Infrastructure management burden",
                    "Scaling requires hardware"
                ],
                best_for=[
                    "High volume (>100K/day)",
                    "Strict data residency",
                    "Predictable workload",
                    "Long-term deployment",
                    "Compliance requirements"
                ]
            ),
            ArchitectureOption(
                name="Edge Deployment",
                deployment_type=DeploymentType.EDGE,
                description="Models running on network devices or edge servers",
                initial_cost_usd=20_000,
                monthly_cost_usd=1_000,
                latency_ms=(5, 50),
                scalability=ScalingPattern.HORIZONTAL,
                complexity="high",
                pros=[
                    "Ultra-low latency",
                    "No cloud dependency",
                    "Works in air-gapped networks",
                    "Distributed architecture",
                    "Real-time inference"
                ],
                cons=[
                    "Limited model size",
                    "Device resource constraints",
                    "Complex deployment",
                    "Update management difficult",
                    "Requires optimized models"
                ],
                best_for=[
                    "Real-time requirements (<50ms)",
                    "Air-gapped environments",
                    "High security needs",
                    "Distributed sites",
                    "Network automation tasks"
                ]
            ),
        ]

    def calculate_costs(
        self,
        daily_requests: int,
        architecture: ArchitectureOption,
        months: int = 12
    ) -> CostBreakdown:
        """
        Calculate detailed cost breakdown.

        Args:
            daily_requests: Number of AI requests per day
            architecture: Architecture option
            months: Time period for calculation

        Returns:
            CostBreakdown with all costs
        """
        monthly_requests = daily_requests * 30

        if architecture.deployment_type == DeploymentType.CLOUD_API:
            # Cloud API costs (example: GPT-4 pricing)
            avg_tokens_per_request = 500  # 200 input + 300 output
            cost_per_1m_tokens = 15.0  # Average
            api_cost = (monthly_requests * avg_tokens_per_request / 1_000_000) * cost_per_1m_tokens

            return CostBreakdown(
                infrastructure=0,
                api_calls=api_cost,
                storage=50,  # Logs
                bandwidth=100,
                support=0,
                total=api_cost + 150
            )

        elif architecture.deployment_type == DeploymentType.HYBRID:
            # 60% cached, 40% cloud API
            cache_hit_rate = 0.60
            cloud_requests = monthly_requests * (1 - cache_hit_rate)

            avg_tokens = 500
            cost_per_1m = 15.0
            api_cost = (cloud_requests * avg_tokens / 1_000_000) * cost_per_1m

            return CostBreakdown(
                infrastructure=1_500,  # Cache servers
                api_calls=api_cost,
                storage=200,
                bandwidth=150,
                support=150,
                total=1_500 + api_cost + 500
            )

        elif architecture.deployment_type == DeploymentType.ON_PREMISES:
            return CostBreakdown(
                infrastructure=3_500,  # Amortized hardware
                api_calls=0,
                storage=300,
                bandwidth=100,
                support=1_100,  # Staff time
                total=5_000
            )

        else:  # Edge
            return CostBreakdown(
                infrastructure=500,  # Edge devices
                api_calls=0,
                storage=100,
                bandwidth=50,
                support=350,
                total=1_000
            )

    def calculate_roi(
        self,
        current_architecture: ArchitectureOption,
        proposed_architecture: ArchitectureOption,
        daily_requests: int,
        months: int = 24
    ) -> Dict:
        """
        Calculate ROI for switching architectures.

        Returns:
            Dict with ROI analysis
        """
        current_costs = self.calculate_costs(daily_requests, current_architecture, months)
        proposed_costs = self.calculate_costs(daily_requests, proposed_architecture, months)

        # Total costs over period
        current_total = (current_costs.total * months) + current_architecture.initial_cost_usd
        proposed_total = (proposed_costs.total * months) + proposed_architecture.initial_cost_usd

        savings = current_total - proposed_total
        roi_percent = (savings / proposed_architecture.initial_cost_usd * 100) if proposed_architecture.initial_cost_usd > 0 else 0

        # Break-even calculation
        if savings > 0 and proposed_architecture.initial_cost_usd > 0:
            monthly_savings = (current_costs.total - proposed_costs.total)
            breakeven_months = proposed_architecture.initial_cost_usd / monthly_savings if monthly_savings > 0 else float('inf')
        else:
            breakeven_months = float('inf')

        return {
            'current_monthly': current_costs.total,
            'proposed_monthly': proposed_costs.total,
            'total_savings_24mo': savings,
            'roi_percent': roi_percent,
            'breakeven_months': breakeven_months,
            'recommendation': 'Switch' if savings > 0 and breakeven_months < 12 else 'Stay with current'
        }

    def recommend_architecture(
        self,
        requirements: DeploymentRequirement
    ) -> List[ArchitectureOption]:
        """
        Recommend architectures based on requirements.

        Args:
            requirements: Deployment requirements

        Returns:
            List of suitable architectures, ranked
        """
        scored_architectures = []

        for arch in self.architectures:
            score = 0

            # Cost scoring
            monthly_cost = self.calculate_costs(requirements.daily_requests, arch).total

            if monthly_cost <= requirements.budget_monthly_usd:
                score += 30
            elif monthly_cost <= requirements.budget_monthly_usd * 1.2:
                score += 15

            # Data residency
            if requirements.data_residency == "on-prem":
                if arch.deployment_type in [DeploymentType.ON_PREMISES, DeploymentType.EDGE]:
                    score += 25
            elif requirements.data_residency == "cloud":
                if arch.deployment_type == DeploymentType.CLOUD_API:
                    score += 25
            else:  # hybrid
                score += 15

            # Latency
            if arch.latency_ms[1] <= requirements.latency_requirement_ms:
                score += 20

            # Volume appropriateness
            if requirements.daily_requests < 10_000 and arch.deployment_type == DeploymentType.CLOUD_API:
                score += 15
            elif 10_000 <= requirements.daily_requests <= 100_000 and arch.deployment_type == DeploymentType.HYBRID:
                score += 15
            elif requirements.daily_requests > 100_000 and arch.deployment_type in [DeploymentType.ON_PREMISES, DeploymentType.EDGE]:
                score += 15

            # Compliance
            if requirements.compliance_requirements:
                if arch.deployment_type in [DeploymentType.ON_PREMISES, DeploymentType.EDGE]:
                    score += 10

            scored_architectures.append((score, arch))

        # Sort by score
        scored_architectures.sort(key=lambda x: x[0], reverse=True)

        return [arch for score, arch in scored_architectures]


def example_1_basic_comparison():
    """
    Example 1: Compare all architectures
    """
    print("=" * 60)
    print("Example 1: Architecture Comparison")
    print("=" * 60)

    framework = ArchitectureDecisionFramework()

    print("\n📊 Available Deployment Architectures:\n")

    for arch in framework.architectures:
        print(f"━━━ {arch.name} ━━━")
        print(f"Type: {arch.deployment_type.value}")
        print(f"Complexity: {arch.complexity.upper()}")
        print(f"Latency: {arch.latency_ms[0]}-{arch.latency_ms[1]}ms")
        print(f"Initial Cost: ${arch.initial_cost_usd:,.0f}")
        print(f"Monthly Base: ${arch.monthly_cost_usd:,.0f}")

        print(f"\n✅ Pros:")
        for pro in arch.pros[:3]:
            print(f"  • {pro}")

        print(f"\n❌ Cons:")
        for con in arch.cons[:2]:
            print(f"  • {con}")

        print(f"\n🎯 Best for:")
        for use in arch.best_for[:2]:
            print(f"  • {use}")

        print()

    print("=" * 60 + "\n")


def example_2_cost_calculator():
    """
    Example 2: Calculate costs for different volumes
    """
    print("=" * 60)
    print("Example 2: Cost Calculator")
    print("=" * 60)

    framework = ArchitectureDecisionFramework()

    volumes = [1_000, 10_000, 100_000, 500_000]

    print("\n💰 Monthly Cost by Volume:\n")
    print(f"{'Volume/Day':<15} {'Cloud API':<15} {'Hybrid':<15} {'On-Prem':<15} {'Edge':<15}")
    print("─" * 75)

    for volume in volumes:
        costs = []
        for arch in framework.architectures:
            cost = framework.calculate_costs(volume, arch)
            costs.append(f"${cost.total:,.0f}")

        print(f"{volume:>10,} {costs[0]:<15} {costs[1]:<15} {costs[2]:<15} {costs[3]:<15}")

    print("\n📈 Key Insights:")
    print("  • Cloud API cheapest for <10K requests/day")
    print("  • Hybrid optimal for 10K-100K requests/day")
    print("  • On-Prem becomes cost-effective >100K/day")
    print("  • Edge best for ultra-low latency needs")

    print("\n" + "=" * 60 + "\n")


def example_3_roi_analysis():
    """
    Example 3: ROI analysis for migration
    """
    print("=" * 60)
    print("Example 3: ROI Analysis - Cloud to On-Prem")
    print("=" * 60)

    framework = ArchitectureDecisionFramework()

    # Current: Cloud API at 200K requests/day
    cloud = framework.architectures[0]  # Cloud API
    on_prem = framework.architectures[2]  # On-premises

    daily_requests = 200_000

    print(f"\nScenario: {daily_requests:,} requests/day\n")

    roi = framework.calculate_roi(cloud, on_prem, daily_requests, months=24)

    print(f"Current (Cloud API):")
    print(f"  Monthly Cost: ${roi['current_monthly']:,.2f}")
    print(f"  24-Month Total: ${roi['current_monthly'] * 24:,.2f}")

    print(f"\nProposed (On-Premises):")
    print(f"  Initial Investment: ${on_prem.initial_cost_usd:,.2f}")
    print(f"  Monthly Cost: ${roi['proposed_monthly']:,.2f}")
    print(f"  24-Month Total: ${roi['proposed_monthly'] * 24 + on_prem.initial_cost_usd:,.2f}")

    print(f"\n📊 ROI Analysis:")
    print(f"  Total Savings (24mo): ${roi['total_savings_24mo']:,.2f}")
    print(f"  ROI: {roi['roi_percent']:.1f}%")
    print(f"  Break-even: {roi['breakeven_months']:.1f} months")

    print(f"\n💡 Recommendation: {roi['recommendation']}")

    if roi['breakeven_months'] < 12:
        print(f"   Strong business case - payback in <1 year")
    elif roi['breakeven_months'] < 24:
        print(f"   Moderate business case - payback in <2 years")
    else:
        print(f"   Weak business case - long payback period")

    print("\n" + "=" * 60 + "\n")


def example_4_architecture_recommendation():
    """
    Example 4: Get architecture recommendation based on requirements
    """
    print("=" * 60)
    print("Example 4: Architecture Recommendation Engine")
    print("=" * 60)

    framework = ArchitectureDecisionFramework()

    # Define requirements
    requirements = DeploymentRequirement(
        daily_requests=50_000,
        data_residency="hybrid",
        latency_requirement_ms=500,
        team_size=3,
        budget_monthly_usd=3_000,
        compliance_requirements=["GDPR", "SOC2"],
        uptime_sla=0.999
    )

    print("\n📋 Requirements:")
    print(f"  Daily Requests: {requirements.daily_requests:,}")
    print(f"  Data Residency: {requirements.data_residency}")
    print(f"  Max Latency: {requirements.latency_requirement_ms}ms")
    print(f"  Monthly Budget: ${requirements.budget_monthly_usd:,.0f}")
    print(f"  Compliance: {', '.join(requirements.compliance_requirements)}")
    print(f"  Team Size: {requirements.team_size}")

    recommendations = framework.recommend_architecture(requirements)

    print("\n🏆 Ranked Recommendations:\n")

    for i, arch in enumerate(recommendations[:3], 1):
        cost = framework.calculate_costs(requirements.daily_requests, arch)

        print(f"{i}. {arch.name}")
        print(f"   Monthly Cost: ${cost.total:,.2f}")
        print(f"   Latency: {arch.latency_ms[0]}-{arch.latency_ms[1]}ms")
        print(f"   Complexity: {arch.complexity}")

        if i == 1:
            print(f"   ⭐ TOP CHOICE:")
            for reason in arch.best_for[:2]:
                print(f"      • {reason}")

        print()

    print("=" * 60 + "\n")


def example_5_llm_powered_recommendation():
    """
    Example 5: Use LLM for architecture recommendation
    """
    print("=" * 60)
    print("Example 5: LLM-Powered Architecture Advisor")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM example.")
        print("=" * 60 + "\n")
        return

    scenario = """
Company Profile:
- Enterprise network with 5,000 devices
- 150 network engineers globally
- Need AI for: config validation, log analysis, troubleshooting
- Expected 200,000 AI requests/day
- Strict data residency (EU only)
- Compliance: GDPR, ISO 27001
- Budget: $10,000/month
- Need 99.9% uptime
- Team has no ML expertise

Current State:
- Using OpenAI API ($15,000/month and growing)
- Concerns about data leaving EU
- Cost unpredictable
- Rate limiting during incidents

Question: What deployment architecture should we adopt?
"""

    prompt = f"""You are a senior AI architect for network operations. Analyze this scenario:

{scenario}

Provide architecture recommendation with:
- recommended_architecture: Best deployment option
- reasoning: Why this fits their needs
- estimated_monthly_cost: In USD
- implementation_steps: 3-5 key steps
- risks: 2-3 risks with mitigations
- alternative_options: 2 backup options

Focus on: cost, compliance, team skills, and scalability."""

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ArchitectureRecommendation)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\n🤖 Analyzing scenario with Claude...\n")

    response = llm.invoke(full_prompt)
    recommendation = parser.parse(response.content)

    print(f"🏆 Recommended Architecture: {recommendation.recommended_architecture}\n")

    print(f"💡 Reasoning:")
    print(f"   {recommendation.reasoning}\n")

    print(f"💰 Estimated Monthly Cost: ${recommendation.estimated_monthly_cost:,.0f}\n")

    print(f"📝 Implementation Steps:")
    for i, step in enumerate(recommendation.implementation_steps, 1):
        print(f"   {i}. {step}")

    print(f"\n⚠️  Risks & Mitigations:")
    for risk in recommendation.risks:
        print(f"   • {risk}")

    print(f"\n🔄 Alternative Options:")
    for alt in recommendation.alternative_options:
        print(f"   • {alt}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🏗️  Chapter 42: Deployment Architectures")
    print("Decision Frameworks for AI Systems\n")

    try:
        example_1_basic_comparison()
        input("Press Enter to continue...")

        example_2_cost_calculator()
        input("Press Enter to continue...")

        example_3_roi_analysis()
        input("Press Enter to continue...")

        example_4_architecture_recommendation()
        input("Press Enter to continue...")

        example_5_llm_powered_recommendation()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Cloud APIs best for low volume and quick starts")
        print("- Hybrid optimal for 10K-100K requests/day")
        print("- On-premises cost-effective at >100K/day")
        print("- ROI analysis shows break-even typically 6-12 months")
        print("- Architecture choice depends on: volume, budget, compliance, latency\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
