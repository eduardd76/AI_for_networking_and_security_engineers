"""
Chapter 56: Anomaly Detection & Predictive Maintenance
AI-powered network failure prediction and maintenance scheduling

This module demonstrates statistical and AI-based anomaly detection,
predictive failure analysis, and proactive maintenance scheduling.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class AnomalyType(str, Enum):
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    THRESHOLD = "threshold"


class FailureRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricSample:
    """Time-series metric sample."""
    timestamp: str
    value: float
    metric_name: str
    device: str


@dataclass
class AnomalyDetection:
    """Detected anomaly."""
    timestamp: str
    metric_name: str
    device: str
    actual_value: float
    expected_value: float
    deviation_sigma: float
    anomaly_type: AnomalyType
    severity: str


@dataclass
class FailurePrediction:
    """Predicted device failure."""
    device: str
    failure_type: str
    risk_level: FailureRisk
    estimated_time_to_failure_hours: float
    confidence: float
    indicators: List[str]
    recommended_action: str


class MaintenanceRecommendation(BaseModel):
    """LLM-powered maintenance recommendation."""
    device: str = Field(description="Device name")
    priority: str = Field(description="low, medium, high, or critical")
    recommended_action: str = Field(description="Specific maintenance action")
    estimated_downtime_minutes: int = Field(description="Expected downtime")
    best_maintenance_window: str = Field(description="When to perform maintenance")
    rationale: str = Field(description="Why this maintenance is needed")
    risks_if_delayed: List[str] = Field(description="Risks of delaying maintenance")


class PredictiveMaintenanceSystem:
    """
    AI-powered predictive maintenance system.

    Features:
    - Statistical anomaly detection (z-score)
    - Behavioral baseline learning
    - Failure prediction based on patterns
    - Maintenance window optimization
    - Proactive alerting
    """

    def __init__(self, baseline_window: int = 100):
        self.baseline_window = baseline_window
        self.metrics: Dict[str, deque] = {}
        self.baselines: Dict[str, Dict] = {}
        self.anomalies: List[AnomalyDetection] = []

    def add_metric(
        self,
        device: str,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None
    ):
        """Add metric sample to time-series."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        key = f"{device}:{metric_name}"

        if key not in self.metrics:
            self.metrics[key] = deque(maxlen=self.baseline_window)

        self.metrics[key].append(MetricSample(
            timestamp=timestamp,
            value=value,
            metric_name=metric_name,
            device=device
        ))

        # Update baseline
        self._update_baseline(key)

    def _update_baseline(self, key: str):
        """Calculate baseline statistics."""
        samples = list(self.metrics[key])

        if len(samples) < 10:
            return

        values = [s.value for s in samples]

        self.baselines[key] = {
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'samples': len(values)
        }

    def detect_anomaly(
        self,
        device: str,
        metric_name: str,
        current_value: float,
        threshold_sigma: float = 3.0
    ) -> Optional[AnomalyDetection]:
        """
        Detect statistical anomaly using z-score.

        Args:
            device: Device name
            metric_name: Metric being monitored
            current_value: Current metric value
            threshold_sigma: Threshold in standard deviations

        Returns:
            AnomalyDetection if anomaly detected, None otherwise
        """
        key = f"{device}:{metric_name}"

        if key not in self.baselines:
            return None

        baseline = self.baselines[key]

        if baseline['stdev'] == 0:
            return None

        # Calculate z-score
        z_score = abs((current_value - baseline['mean']) / baseline['stdev'])

        if z_score >= threshold_sigma:
            anomaly = AnomalyDetection(
                timestamp=datetime.now().isoformat(),
                metric_name=metric_name,
                device=device,
                actual_value=current_value,
                expected_value=baseline['mean'],
                deviation_sigma=z_score,
                anomaly_type=AnomalyType.STATISTICAL,
                severity='critical' if z_score > 5 else 'high' if z_score > 4 else 'medium'
            )

            self.anomalies.append(anomaly)
            return anomaly

        return None

    def predict_failure(
        self,
        device: str,
        recent_anomalies: List[AnomalyDetection],
        device_age_days: int
    ) -> Optional[FailurePrediction]:
        """
        Predict potential device failure.

        Args:
            device: Device name
            recent_anomalies: Recent anomalies detected
            device_age_days: Age of device in days

        Returns:
            FailurePrediction if failure likely
        """
        # Failure indicators
        indicators = []
        risk_score = 0

        # 1. Frequent anomalies
        if len(recent_anomalies) >= 5:
            indicators.append(f"{len(recent_anomalies)} anomalies in recent period")
            risk_score += 30

        # 2. Critical anomalies
        critical = [a for a in recent_anomalies if a.severity == 'critical']
        if critical:
            indicators.append(f"{len(critical)} critical anomalies")
            risk_score += 40

        # 3. Device age
        if device_age_days > 1825:  # 5 years
            indicators.append(f"Device age: {device_age_days/365:.1f} years")
            risk_score += 20

        # 4. Multiple metrics degraded
        unique_metrics = set(a.metric_name for a in recent_anomalies)
        if len(unique_metrics) >= 3:
            indicators.append(f"Multiple metrics degraded: {', '.join(unique_metrics)}")
            risk_score += 10

        # Determine risk level
        if risk_score < 30:
            risk = FailureRisk.LOW
            time_to_failure = 720  # 30 days
        elif risk_score < 60:
            risk = FailureRisk.MEDIUM
            time_to_failure = 168  # 7 days
        elif risk_score < 80:
            risk = FailureRisk.HIGH
            time_to_failure = 48  # 2 days
        else:
            risk = FailureRisk.CRITICAL
            time_to_failure = 12  # 12 hours

        if risk_score >= 30:
            return FailurePrediction(
                device=device,
                failure_type="hardware_degradation",
                risk_level=risk,
                estimated_time_to_failure_hours=time_to_failure,
                confidence=min(0.95, risk_score / 100),
                indicators=indicators,
                recommended_action=self._get_recommended_action(risk, time_to_failure)
            )

        return None

    def _get_recommended_action(self, risk: FailureRisk, hours: float) -> str:
        """Get recommended action based on risk."""
        if risk == FailureRisk.CRITICAL:
            return f"IMMEDIATE: Schedule emergency maintenance within {hours:.0f} hours"
        elif risk == FailureRisk.HIGH:
            return f"URGENT: Schedule maintenance within {hours/24:.0f} days"
        elif risk == FailureRisk.MEDIUM:
            return f"PLANNED: Schedule maintenance within {hours/24:.0f} days"
        else:
            return "MONITOR: Continue monitoring, no immediate action required"

    def get_maintenance_priority(
        self,
        predictions: List[FailurePrediction]
    ) -> List[Tuple[str, FailurePrediction]]:
        """
        Prioritize devices for maintenance.

        Returns:
            List of (priority_label, prediction) tuples, sorted by priority
        """
        scored = []

        for pred in predictions:
            # Calculate priority score
            risk_weights = {
                FailureRisk.CRITICAL: 100,
                FailureRisk.HIGH: 75,
                FailureRisk.MEDIUM: 50,
                FailureRisk.LOW: 25
            }

            base_score = risk_weights[pred.risk_level]
            time_factor = 100 / max(1, pred.estimated_time_to_failure_hours)
            confidence_factor = pred.confidence

            priority_score = base_score * (1 + time_factor) * confidence_factor

            scored.append((priority_score, pred))

        # Sort by priority
        scored.sort(key=lambda x: x[0], reverse=True)

        # Assign priority labels
        prioritized = []
        for i, (score, pred) in enumerate(scored):
            if i == 0:
                label = "P0 - CRITICAL"
            elif i < 3:
                label = f"P1 - HIGH"
            elif i < 6:
                label = f"P2 - MEDIUM"
            else:
                label = f"P3 - LOW"

            prioritized.append((label, pred))

        return prioritized


def example_1_anomaly_detection():
    """
    Example 1: Statistical anomaly detection
    """
    print("=" * 60)
    print("Example 1: Statistical Anomaly Detection")
    print("=" * 60)

    system = PredictiveMaintenanceSystem()

    # Build baseline (normal CPU usage 20-30%)
    print("\n📊 Building baseline (100 samples)...")

    for i in range(100):
        normal_cpu = 25 + (i % 10) - 5  # 20-30%
        system.add_metric("router-01", "cpu_percent", normal_cpu)

    print(f"✓ Baseline established\n")

    # Inject anomalies
    anomalous_values = [45, 55, 70, 85]

    print("🔍 Testing anomaly detection:\n")

    for value in anomalous_values:
        system.add_metric("router-01", "cpu_percent", value)

        anomaly = system.detect_anomaly("router-01", "cpu_percent", value)

        if anomaly:
            print(f"🚨 ANOMALY DETECTED")
            print(f"   Value: {anomaly.actual_value:.1f}%")
            print(f"   Expected: {anomaly.expected_value:.1f}%")
            print(f"   Deviation: {anomaly.deviation_sigma:.1f}σ")
            print(f"   Severity: {anomaly.severity.upper()}")
            print()

    print("=" * 60 + "\n")


def example_2_failure_prediction():
    """
    Example 2: Predict device failure
    """
    print("=" * 60)
    print("Example 2: Failure Prediction")
    print("=" * 60)

    system = PredictiveMaintenanceSystem()

    # Simulate degrading device
    device = "core-sw-01"

    # Create anomalies indicating degradation
    anomalies = [
        AnomalyDetection(
            timestamp=datetime.now().isoformat(),
            metric_name="cpu_percent",
            device=device,
            actual_value=85,
            expected_value=25,
            deviation_sigma=5.2,
            anomaly_type=AnomalyType.STATISTICAL,
            severity="critical"
        ),
        AnomalyDetection(
            timestamp=datetime.now().isoformat(),
            metric_name="memory_percent",
            device=device,
            actual_value=92,
            expected_value=60,
            deviation_sigma=4.8,
            anomaly_type=AnomalyType.STATISTICAL,
            severity="high"
        ),
        AnomalyDetection(
            timestamp=datetime.now().isoformat(),
            metric_name="interface_errors",
            device=device,
            actual_value=1250,
            expected_value=10,
            deviation_sigma=6.1,
            anomaly_type=AnomalyType.STATISTICAL,
            severity="critical"
        ),
        AnomalyDetection(
            timestamp=datetime.now().isoformat(),
            metric_name="temperature_celsius",
            device=device,
            actual_value=68,
            expected_value=45,
            deviation_sigma=4.2,
            anomaly_type=AnomalyType.STATISTICAL,
            severity="high"
        ),
    ]

    print(f"\n⚠️  Device: {device}")
    print(f"   Recent Anomalies: {len(anomalies)}")
    print(f"   Device Age: 6.2 years\n")

    # Predict failure
    prediction = system.predict_failure(device, anomalies, device_age_days=2263)

    if prediction:
        emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }.get(prediction.risk_level.value, '⚪')

        print(f"{emoji} FAILURE RISK: {prediction.risk_level.value.upper()}")
        print(f"   Confidence: {prediction.confidence:.0%}")
        print(f"   Est. Time to Failure: {prediction.estimated_time_to_failure_hours:.0f} hours")

        print(f"\n🔍 Indicators:")
        for indicator in prediction.indicators:
            print(f"   • {indicator}")

        print(f"\n🔧 Recommended Action:")
        print(f"   {prediction.recommended_action}")

    print("\n" + "=" * 60 + "\n")


def example_3_maintenance_priority():
    """
    Example 3: Prioritize maintenance across fleet
    """
    print("=" * 60)
    print("Example 3: Maintenance Prioritization")
    print("=" * 60)

    system = PredictiveMaintenanceSystem()

    # Multiple devices with varying risk levels
    predictions = [
        FailurePrediction(
            device="core-sw-01",
            failure_type="hardware_degradation",
            risk_level=FailureRisk.CRITICAL,
            estimated_time_to_failure_hours=12,
            confidence=0.92,
            indicators=["Critical CPU", "High temp", "6+ years old"],
            recommended_action="Emergency maintenance required"
        ),
        FailurePrediction(
            device="edge-rtr-05",
            failure_type="interface_degradation",
            risk_level=FailureRisk.HIGH,
            estimated_time_to_failure_hours=48,
            confidence=0.78,
            indicators=["Interface errors increasing", "CRC errors"],
            recommended_action="Schedule maintenance within 2 days"
        ),
        FailurePrediction(
            device="access-sw-12",
            failure_type="power_supply",
            risk_level=FailureRisk.MEDIUM,
            estimated_time_to_failure_hours=168,
            confidence=0.65,
            indicators=["Power supply fluctuations"],
            recommended_action="Plan maintenance within 7 days"
        ),
        FailurePrediction(
            device="edge-rtr-03",
            failure_type="fan_failure",
            risk_level=FailureRisk.LOW,
            estimated_time_to_failure_hours=720,
            confidence=0.52,
            indicators=["Fan speed decreased slightly"],
            recommended_action="Monitor, maintenance in 30 days"
        ),
    ]

    prioritized = system.get_maintenance_priority(predictions)

    print("\n🎯 Maintenance Priority Queue:\n")

    for priority_label, pred in prioritized:
        emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }.get(pred.risk_level.value, '⚪')

        print(f"{emoji} [{priority_label}] {pred.device}")
        print(f"   Risk: {pred.risk_level.value.upper()} (confidence: {pred.confidence:.0%})")
        print(f"   Time to Failure: {pred.estimated_time_to_failure_hours:.0f}h")
        print(f"   Action: {pred.recommended_action}")
        print()

    print("=" * 60 + "\n")


def example_4_behavioral_baseline():
    """
    Example 4: Learn behavioral baselines
    """
    print("=" * 60)
    print("Example 4: Behavioral Baseline Learning")
    print("=" * 60)

    system = PredictiveMaintenanceSystem()

    # Simulate daily patterns (higher during business hours)
    print("\n📚 Learning daily patterns (24 hours)...\n")

    for hour in range(24):
        # Traffic higher during business hours (8-17)
        if 8 <= hour <= 17:
            traffic_mbps = 800 + (hour - 8) * 50  # 800-1250 Mbps
        else:
            traffic_mbps = 200 + hour * 10  # 200-430 Mbps

        system.add_metric("router-01", "traffic_mbps", traffic_mbps)

    baseline = system.baselines.get("router-01:traffic_mbps")

    if baseline:
        print(f"Baseline Statistics:")
        print(f"  Mean: {baseline['mean']:.1f} Mbps")
        print(f"  StdDev: {baseline['stdev']:.1f} Mbps")
        print(f"  Range: {baseline['min']:.1f} - {baseline['max']:.1f} Mbps")
        print(f"  Samples: {baseline['samples']}")

    # Test detection during off-hours spike
    print(f"\n🔍 Testing: 1500 Mbps at 3 AM (unusual)")

    anomaly = system.detect_anomaly("router-01", "traffic_mbps", 1500.0)

    if anomaly:
        print(f"   ✓ Detected as anomaly ({anomaly.deviation_sigma:.1f}σ)")
    else:
        print(f"   ✗ Not detected")

    print("\n" + "=" * 60 + "\n")


def example_5_llm_maintenance_recommendation():
    """
    Example 5: LLM-powered maintenance recommendations
    """
    print("=" * 60)
    print("Example 5: AI Maintenance Recommendations")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM example.")
        print("=" * 60 + "\n")
        return

    device_state = """
Device: core-sw-01
Model: Cisco Catalyst 9500
Age: 6.2 years
Location: Data Center A (Critical)

Recent Anomalies (last 7 days):
- CPU utilization: 85% (expected 25%)
- Memory: 92% (expected 60%)
- Interface errors: 1,250/hr (expected 10/hr)
- Temperature: 68°C (expected 45°C)
- Fan speed: 2,100 RPM (expected 3,000 RPM)

Failure Prediction:
- Risk: CRITICAL
- Estimated time to failure: 12 hours
- Confidence: 92%

Business Impact:
- Handles 40% of data center traffic
- 500+ users dependent
- No redundancy configured
"""

    prompt = f"""You are a network maintenance planner. Analyze this device:

{device_state}

Provide maintenance recommendation with:
- device: Device name
- priority: critical, high, medium, or low
- recommended_action: Specific maintenance steps
- estimated_downtime_minutes: Expected downtime
- best_maintenance_window: When to perform (consider 24/7 ops)
- rationale: Technical reasoning
- risks_if_delayed: What happens if we wait

Be specific and actionable."""

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    parser = PydanticOutputParser(pydantic_object=MaintenanceRecommendation)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\n🤖 Analyzing with Claude...\n")

    response = llm.invoke(full_prompt)
    recommendation = parser.parse(response.content)

    print(f"🔧 Maintenance Plan: {recommendation.device}")
    print(f"   Priority: {recommendation.priority.upper()}")
    print(f"\n📋 Action:")
    print(f"   {recommendation.recommended_action}")
    print(f"\n⏱️  Downtime: {recommendation.estimated_downtime_minutes} minutes")
    print(f"   Best Window: {recommendation.best_maintenance_window}")
    print(f"\n💡 Rationale:")
    print(f"   {recommendation.rationale}")
    print(f"\n⚠️  Risks if Delayed:")
    for risk in recommendation.risks_if_delayed:
        print(f"   • {risk}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔮 Chapter 56: Predictive Maintenance & Anomaly Detection")
    print("AI-Powered Network Failure Prevention\n")

    try:
        example_1_anomaly_detection()
        input("Press Enter to continue...")

        example_2_failure_prediction()
        input("Press Enter to continue...")

        example_3_maintenance_priority()
        input("Press Enter to continue...")

        example_4_behavioral_baseline()
        input("Press Enter to continue...")

        example_5_llm_maintenance_recommendation()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Statistical anomaly detection catches deviations early")
        print("- Multiple anomalies indicate impending failure")
        print("- Prioritize maintenance by risk × time × confidence")
        print("- Behavioral baselines adapt to normal patterns")
        print("- Proactive maintenance reduces unplanned outages")
        print("- AI recommendations optimize maintenance windows\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
