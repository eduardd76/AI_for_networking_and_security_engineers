# Chapter 25: Anomaly Detection with AI

## Knowing "Normal" to Spot "Abnormal"

Every network has a rhythm — traffic peaks during business hours, backups run at midnight, BGP updates settle after maintenance windows. Anomaly detection is about learning that rhythm and flagging when something doesn't fit the pattern.

**Networking analogy**: Think of anomaly detection like BFD (Bidirectional Forwarding Detection) for your entire network state. BFD establishes a baseline ("the link is up and forwarding") and immediately detects when something deviates. Anomaly detection does the same thing, but for traffic patterns, log volumes, error rates, and performance metrics — not just link state.

Traditional monitoring uses fixed thresholds: "alert when CPU > 80%." But what's normal for a core router at 2 PM is different from what's normal at 2 AM. AI-based anomaly detection learns these patterns and adapts.

---

## Types of Network Anomalies

Before building detectors, understand what you're looking for:

| Anomaly Type | Example | Detection Approach |
|-------------|---------|-------------------|
| **Point anomaly** | Single device with 99% CPU when others are at 30% | Compare to fleet baseline |
| **Contextual anomaly** | 90% CPU at 3 AM (normal at 3 PM, not at 3 AM) | Time-aware thresholds |
| **Collective anomaly** | 10 devices all spike to 80% CPU simultaneously | Correlation across fleet |
| **Trend anomaly** | Memory usage growing 1% per day for 30 days | Long-term trend analysis |
| **Pattern anomaly** | BGP updates happening at unusual times | Sequence pattern matching |

---

## Approach 1: Statistical Anomaly Detection

Start with the basics — statistical methods that don't need AI but establish the foundation.

```python
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class NetworkBaselineDetector:
    """Detect anomalies by comparing current values to a learned baseline.

    Like establishing a 'normal' traffic profile for your network
    and alerting when something deviates beyond expected bounds.

    This is the same concept as NetFlow baseline alerting, but
    applied to any metric: CPU, memory, error counts, BGP prefixes, etc.
    """

    def __init__(self, sensitivity: float = 2.0):
        """
        Args:
            sensitivity: Number of standard deviations before flagging.
                         2.0 = ~95% of normal values are within bounds.
                         3.0 = ~99.7% — more conservative, fewer false alarms.
        """
        self.sensitivity = sensitivity
        self.baselines = {}

    def learn_baseline(
        self,
        metric_name: str,
        historical_values: List[float]
    ):
        """Learn what 'normal' looks like for a metric.

        Computes mean and standard deviation from historical data.
        Like building a traffic baseline from a week of NetFlow data.
        """
        values = np.array(historical_values)
        self.baselines[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "samples": len(values)
        }

    def check_value(
        self,
        metric_name: str,
        current_value: float
    ) -> Dict:
        """Check if a current value is anomalous.

        Returns an anomaly report with score and classification.
        """
        if metric_name not in self.baselines:
            return {"status": "no_baseline", "metric": metric_name}

        baseline = self.baselines[metric_name]
        mean = baseline["mean"]
        std = baseline["std"]

        if std == 0:
            # Constant metric — any change is anomalous
            is_anomaly = current_value != mean
            z_score = float('inf') if is_anomaly else 0
        else:
            z_score = abs(current_value - mean) / std
            is_anomaly = z_score > self.sensitivity

        return {
            "metric": metric_name,
            "current_value": current_value,
            "baseline_mean": mean,
            "baseline_std": std,
            "z_score": round(z_score, 2),
            "is_anomaly": is_anomaly,
            "severity": self._classify_severity(z_score),
            "direction": "above" if current_value > mean else "below"
        }

    def _classify_severity(self, z_score: float) -> str:
        """Classify anomaly severity based on z-score."""
        if z_score < 2.0:
            return "normal"
        elif z_score < 3.0:
            return "warning"
        elif z_score < 4.0:
            return "high"
        else:
            return "critical"


# Example: Monitor router fleet metrics
detector = NetworkBaselineDetector(sensitivity=2.5)

# Learn baselines from last 7 days of data (values per hour)
detector.learn_baseline("core_router_cpu", [
    35, 42, 38, 40, 55, 62, 58, 65, 68, 72, 70, 65,  # Business hours
    60, 55, 45, 40, 35, 30, 25, 20, 18, 20, 22, 28   # Off hours
] * 7)  # One week of hourly data

detector.learn_baseline("bgp_prefix_count", [
    820000, 820100, 820050, 819900, 820200, 820150  # Full table ~820K
] * 7)

detector.learn_baseline("interface_errors_per_hour", [
    0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
] * 7)

# Check current values
checks = [
    ("core_router_cpu", 95),           # Way above normal
    ("bgp_prefix_count", 750000),      # 70K prefixes disappeared!
    ("interface_errors_per_hour", 150), # Massive error spike
    ("core_router_cpu", 55),           # Normal
]

for metric, value in checks:
    result = detector.check_value(metric, value)
    status = "ANOMALY" if result['is_anomaly'] else "OK"
    print(f"[{status:7}] {metric}: {value} "
          f"(z-score: {result['z_score']}, severity: {result['severity']})")
```

---

## Approach 2: AI-Enhanced Anomaly Analysis

Statistical detection tells you *that* something is abnormal. AI tells you *what it means* and *what to do about it*.

```python
import os
import json
import re
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY environment variable

def analyze_anomaly(
    anomaly_data: Dict,
    device_context: str = "",
    recent_changes: List[str] = None
) -> Dict:
    """Use AI to analyze a detected anomaly and recommend actions.

    Takes raw anomaly detection output and adds context-aware
    analysis — like having a senior engineer review the alert.
    """

    changes_text = ""
    if recent_changes:
        changes_text = f"\nRecent changes:\n" + "\n".join(
            f"- {c}" for c in recent_changes
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze this network anomaly and provide an assessment.

Anomaly Details:
- Metric: {anomaly_data['metric']}
- Current value: {anomaly_data['current_value']}
- Baseline mean: {anomaly_data['baseline_mean']:.1f}
- Baseline std dev: {anomaly_data['baseline_std']:.1f}
- Z-score: {anomaly_data['z_score']}
- Direction: {anomaly_data['direction']} baseline

Device context: {device_context}
{changes_text}

Provide JSON analysis:
{{
    "assessment": "what this anomaly likely means",
    "possible_causes": ["list of probable causes, most likely first"],
    "risk_level": "critical/high/medium/low",
    "is_likely_false_positive": true/false,
    "false_positive_reason": "if applicable, why this might be a false alarm",
    "recommended_actions": ["ordered list of things to check"],
    "escalation_needed": true/false
}}"""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {"assessment": text}


# Example: Analyze the BGP prefix count anomaly
bgp_anomaly = {
    "metric": "bgp_prefix_count",
    "current_value": 750000,
    "baseline_mean": 820100.0,
    "baseline_std": 100.0,
    "z_score": 701.0,
    "is_anomaly": True,
    "severity": "critical",
    "direction": "below"
}

analysis = analyze_anomaly(
    bgp_anomaly,
    device_context="Core router, dual ISP, full BGP table from both peers",
    recent_changes=[
        "ISP-B maintenance window started at 02:00",
        "No config changes in last 48 hours"
    ]
)

print(f"Assessment: {analysis['assessment']}")
print(f"Risk Level: {analysis['risk_level']}")
print(f"\nPossible causes:")
for cause in analysis.get('possible_causes', []):
    print(f"  - {cause}")
print(f"\nActions:")
for action in analysis.get('recommended_actions', []):
    print(f"  - {action}")
```

---

## Approach 3: Fleet-Wide Anomaly Detection

Instead of monitoring devices individually, compare each device against the fleet. An outlier in the fleet is often more interesting than an absolute threshold breach.

```python
class FleetAnomalyDetector:
    """Detect anomalies by comparing devices against their peers.

    Like comparing OSPF metrics across all routers in an area —
    if one router's SPF count is 10x higher than its neighbors,
    something is wrong with that specific router.
    """

    def __init__(self):
        self.fleet_data = {}

    def update_fleet_metrics(
        self,
        device_metrics: Dict[str, Dict[str, float]]
    ):
        """Update fleet-wide metrics.

        Args:
            device_metrics: Dict mapping hostname to metric dict
                e.g., {"router-01": {"cpu": 45, "memory": 70, "bgp_prefixes": 820000}}
        """
        self.fleet_data = device_metrics

    def find_outliers(self, metric_name: str) -> List[Dict]:
        """Find devices that are outliers for a specific metric.

        Uses the Interquartile Range (IQR) method — same principle
        used in box plots. Outliers are values below Q1 - 1.5*IQR
        or above Q3 + 1.5*IQR.
        """
        values = []
        devices = []

        for hostname, metrics in self.fleet_data.items():
            if metric_name in metrics:
                values.append(metrics[metric_name])
                devices.append(hostname)

        if len(values) < 4:
            return []

        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for device, value in zip(devices, values):
            if value < lower_bound or value > upper_bound:
                outliers.append({
                    "device": device,
                    "metric": metric_name,
                    "value": value,
                    "fleet_median": float(np.median(arr)),
                    "fleet_q1": float(q1),
                    "fleet_q3": float(q3),
                    "bound_exceeded": "lower" if value < lower_bound else "upper"
                })

        return outliers

    def comprehensive_fleet_check(self) -> List[Dict]:
        """Check all metrics across the fleet for outliers."""
        all_outliers = []

        # Get all metric names from any device
        all_metrics = set()
        for metrics in self.fleet_data.values():
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            outliers = self.find_outliers(metric)
            all_outliers.extend(outliers)

        return all_outliers


# Example: Fleet-wide monitoring
fleet_detector = FleetAnomalyDetector()

fleet_detector.update_fleet_metrics({
    "router-core-01": {"cpu": 45, "memory": 68, "bgp_prefixes": 820100, "interface_errors": 2},
    "router-core-02": {"cpu": 42, "memory": 65, "bgp_prefixes": 820050, "interface_errors": 0},
    "router-edge-01": {"cpu": 38, "memory": 62, "bgp_prefixes": 820200, "interface_errors": 1},
    "router-edge-02": {"cpu": 95, "memory": 92, "bgp_prefixes": 750000, "interface_errors": 1500},  # Anomalous!
    "router-edge-03": {"cpu": 40, "memory": 60, "bgp_prefixes": 820000, "interface_errors": 0},
    "router-edge-04": {"cpu": 44, "memory": 66, "bgp_prefixes": 819900, "interface_errors": 3},
})

outliers = fleet_detector.comprehensive_fleet_check()
for outlier in outliers:
    print(f"OUTLIER: {outlier['device']} — {outlier['metric']}: "
          f"{outlier['value']} (fleet median: {outlier['fleet_median']:.0f})")
```

---

## Approach 4: Time-Series Anomaly Detection with AI

For complex time-series data, use AI to understand temporal patterns that statistical methods might miss.

```python
def detect_timeseries_anomalies(
    metric_name: str,
    hourly_values: List[float],
    device_name: str
) -> Dict:
    """Use AI to analyze time-series data for anomalies.

    Better than pure statistical methods for detecting:
    - Seasonal patterns (weekday vs. weekend)
    - Gradual degradation trends
    - Missing expected patterns (e.g., backup didn't run)
    """

    # Format the data as a readable time series
    data_str = "\n".join([
        f"  Hour {i:2d}: {v:.1f}" for i, v in enumerate(hourly_values)
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze this 24-hour time series from a network device
and identify any anomalies.

Device: {device_name}
Metric: {metric_name}

Hourly values (hour 0 = midnight):
{data_str}

Analyze for:
1. Unexpected spikes or drops
2. Missing expected patterns (e.g., backup traffic should appear nightly)
3. Unusual timing (activity at unexpected hours)
4. Gradual trends within the day

Return JSON:
{{
    "anomalies_found": true/false,
    "anomaly_details": [
        {{
            "hour_range": "HH-HH",
            "description": "what's anomalous",
            "expected_behavior": "what you'd normally expect",
            "severity": "critical/warning/informational"
        }}
    ],
    "overall_pattern": "description of the normal daily pattern",
    "health_assessment": "healthy/degraded/concerning"
}}"""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {"anomalies_found": False, "overall_pattern": text}


# Example: Analyze suspicious bandwidth pattern
bandwidth_data = [
    # Normal pattern: low overnight, high during business hours
    5, 3, 2, 1, 2, 8, 25, 60, 75, 80, 82, 78,   # 00:00-11:00
    70, 72, 75, 80, 65, 45, 30, 15, 8, 5, 3, 85  # 12:00-23:00
    # Note: hour 23 shows 85 Gbps — anomalous spike at 11 PM!
]

result = detect_timeseries_anomalies(
    "wan_bandwidth_gbps",
    bandwidth_data,
    "router-core-01"
)

if result['anomalies_found']:
    for anomaly in result.get('anomaly_details', []):
        print(f"[{anomaly['severity']}] Hours {anomaly['hour_range']}: "
              f"{anomaly['description']}")
```

---

## Putting It All Together

```python
class NetworkAnomalySystem:
    """Complete anomaly detection system combining all approaches.

    Architecture:
    1. Statistical detector (fast, runs locally)
    2. Fleet comparison (catches peer outliers)
    3. AI analysis (adds context and recommendations)

    Like a layered defense — each layer catches different things:
    - Statistical = BFD (fast, simple detection)
    - Fleet = OSPF (compares neighbors)
    - AI = Senior engineer (understands context)
    """

    def __init__(self):
        self.baseline_detector = NetworkBaselineDetector()
        self.fleet_detector = FleetAnomalyDetector()

    def monitor_cycle(
        self,
        device_metrics: Dict[str, Dict[str, float]],
        recent_changes: List[str] = None
    ) -> Dict:
        """Run one monitoring cycle across all detection methods."""

        results = {
            "timestamp": datetime.now().isoformat(),
            "devices_checked": len(device_metrics),
            "statistical_anomalies": [],
            "fleet_outliers": [],
            "ai_analyses": []
        }

        # Layer 1: Statistical anomalies per device
        for hostname, metrics in device_metrics.items():
            for metric_name, value in metrics.items():
                check = self.baseline_detector.check_value(
                    f"{hostname}_{metric_name}", value
                )
                if check.get('is_anomaly'):
                    results["statistical_anomalies"].append(check)

        # Layer 2: Fleet outliers
        self.fleet_detector.update_fleet_metrics(device_metrics)
        results["fleet_outliers"] = self.fleet_detector.comprehensive_fleet_check()

        # Layer 3: AI analysis for high-severity anomalies only
        high_severity = [
            a for a in results["statistical_anomalies"]
            if a.get('severity') in ('critical', 'high')
        ]

        for anomaly in high_severity[:5]:  # Limit AI calls
            analysis = analyze_anomaly(
                anomaly,
                recent_changes=recent_changes
            )
            results["ai_analyses"].append({
                "anomaly": anomaly,
                "analysis": analysis
            })

        # Summary
        results["summary"] = {
            "total_anomalies": len(results["statistical_anomalies"]),
            "fleet_outliers": len(results["fleet_outliers"]),
            "critical_count": sum(
                1 for a in results["statistical_anomalies"]
                if a.get('severity') == 'critical'
            )
        }

        return results
```

---

## Cost Considerations

| Detection Layer | Method | Cost | Speed |
|-----------------|--------|------|-------|
| Statistical baseline | Local Python | Free | < 1ms per check |
| Fleet comparison | Local Python | Free | < 10ms per fleet |
| AI anomaly analysis | Claude Sonnet | ~$0.01/anomaly | 2-5 seconds |
| Time-series analysis | Claude Sonnet | ~$0.01/series | 2-5 seconds |

**Key insight**: Run the free statistical and fleet checks continuously. Only invoke AI for anomalies that pass the first two layers. This keeps costs under $5/month even for large networks.

---

## Key Takeaways

1. **Learn baselines first**: Statistical methods are free and catch obvious anomalies. Start here.

2. **Compare against the fleet**: A device that's different from its peers is often more interesting than one that crosses a fixed threshold.

3. **Use AI for context**: AI explains *why* an anomaly matters and *what to do*, not just that it exists.

4. **Layer your detection**: Statistical (fast/free) → fleet comparison (peer analysis) → AI analysis (contextual intelligence). Each layer catches what the others miss.

5. **Monitor the monitors**: Track false positive rates and adjust sensitivity. Too many alerts and people ignore them — just like in your NOC today.

---

## What's Next

In **Chapter 27**, we'll apply these anomaly detection techniques specifically to security — building AI-powered threat detection and security analysis systems.
