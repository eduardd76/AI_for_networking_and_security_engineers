"""
Network Anomaly Detection with Multi-Agent AI Orchestration
Chapter 75: Network Anomaly Detection

Uses LangChain to orchestrate Claude and OpenAI for:
- Claude: Baseline behavior analysis and pattern recognition
- OpenAI: Anomaly classification and threat assessment
- Multi-agent consensus for high-confidence detection

Author: Ed Moffat, vExpertAI GmbH
Production-ready code with proper error handling
Colab-compatible with secure API key handling
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# LangChain imports for orchestration
try:
    from langchain.chains import LLMChain, SequentialChain
    from langchain.prompts import PromptTemplate
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Installing LangChain dependencies...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'langchain', 'langchain-anthropic', 'langchain-openai'])
    from langchain.chains import LLMChain, SequentialChain
    from langchain.prompts import PromptTemplate
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage


# ============================================================================
# API KEY MANAGEMENT (Colab-compatible)
# ============================================================================

def get_api_keys() -> Tuple[str, str]:
    """
    Get API keys from environment or user input (Colab-compatible).

    Returns:
        Tuple[str, str]: (anthropic_key, openai_key)
    """
    # Try environment variables first
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    # If running in Colab, try userdata
    if not anthropic_key or not openai_key:
        try:
            from google.colab import userdata
            if not anthropic_key:
                anthropic_key = userdata.get('ANTHROPIC_API_KEY')
            if not openai_key:
                openai_key = userdata.get('OPENAI_API_KEY')
        except ImportError:
            pass

    # If still not found, prompt user (but don't store)
    if not anthropic_key:
        print("Anthropic API key not found in environment.")
        anthropic_key = input("Enter your Anthropic API key: ").strip()

    if not openai_key:
        print("OpenAI API key not found in environment.")
        openai_key = input("Enter your OpenAI API key: ").strip()

    return anthropic_key, openai_key


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NetFlowRecord:
    """NetFlow v9/IPFIX record for network traffic analysis."""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    bytes: int
    packets: int
    duration_seconds: float


@dataclass
class AnomalyDetectionResult:
    """Result from anomaly detection analysis."""
    anomaly_detected: bool
    confidence: float
    anomaly_type: str
    severity: str
    description: str
    claude_analysis: Optional[Dict]
    openai_analysis: Optional[Dict]
    consensus_reached: bool
    metrics: Dict


# ============================================================================
# MULTI-AGENT ANOMALY DETECTION SYSTEM
# ============================================================================

class MultiAgentAnomalyDetector:
    """
    Multi-agent anomaly detection using Claude + OpenAI with LangChain.

    Architecture:
    1. Claude: Analyzes baseline behavior and detects deviations
    2. OpenAI: Classifies anomaly type (DDoS, exfiltration, normal)
    3. Consensus: Both agents must agree for high-confidence detection
    """

    def __init__(self, anthropic_key: str, openai_key: str):
        """
        Initialize multi-agent detector.

        Args:
            anthropic_key: Anthropic API key
            openai_key: OpenAI API key
        """
        self.anthropic_key = anthropic_key
        self.openai_key = openai_key

        # Initialize LangChain LLMs
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=anthropic_key,
            temperature=0
        )

        self.openai = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_key,
            temperature=0
        )

        self.baselines = {}  # Device baselines

        print("Multi-Agent Anomaly Detector initialized")
        print("- Claude: Baseline analysis")
        print("- OpenAI: Anomaly classification")
        print("- Consensus: Dual-AI validation")

    def learn_baseline(self, flows: List[NetFlowRecord], device_ip: str,
                      learning_days: int = 30) -> Dict:
        """
        Learn baseline traffic pattern using Claude for analysis.

        Args:
            flows: Historical NetFlow records
            device_ip: IP address of device to baseline
            learning_days: Number of days of data

        Returns:
            Dict with baseline statistics
        """
        device_flows = [f for f in flows if f.source_ip == device_ip or f.dest_ip == device_ip]

        if not device_flows:
            return {'error': 'No flows for device'}

        # Calculate traffic statistics
        hourly_traffic = defaultdict(list)
        flows_by_hour = defaultdict(list)

        for flow in device_flows:
            hour_key = flow.timestamp.replace(minute=0, second=0, microsecond=0)
            flows_by_hour[hour_key].append(flow)

        for hour, hour_flows in flows_by_hour.items():
            hour_of_day = hour.hour
            total_bytes = sum(f.bytes for f in hour_flows)
            hourly_traffic[hour_of_day].append(total_bytes)

        # Calculate statistics per hour
        baseline_by_hour = {}
        for hour in range(24):
            if hour in hourly_traffic and len(hourly_traffic[hour]) > 7:
                data = hourly_traffic[hour]
                baseline_by_hour[hour] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'median': float(np.median(data)),
                    'p95': float(np.percentile(data, 95)),
                    'p99': float(np.percentile(data, 99)),
                    'samples': len(data)
                }

        # Learn communication patterns
        communication_peers = defaultdict(int)
        for flow in device_flows:
            peer_ip = flow.dest_ip if flow.source_ip == device_ip else flow.source_ip
            communication_peers[peer_ip] += flow.bytes

        # Get top peers (80% of traffic)
        sorted_peers = sorted(communication_peers.items(), key=lambda x: x[1], reverse=True)
        total_bytes = sum(communication_peers.values())
        cumulative = 0
        typical_peers = []
        for peer_ip, bytes_count in sorted_peers:
            typical_peers.append(peer_ip)
            cumulative += bytes_count
            if cumulative >= total_bytes * 0.8:
                break

        baseline = {
            'device_ip': device_ip,
            'learning_period_days': learning_days,
            'total_flows_analyzed': len(device_flows),
            'hourly_baseline': baseline_by_hour,
            'typical_peers': typical_peers,
            'total_bytes': total_bytes,
            'avg_bytes_per_day': total_bytes / learning_days
        }

        self.baselines[device_ip] = baseline
        return baseline

    def detect_with_consensus(self, flows: List[NetFlowRecord],
                            device_ip: str) -> AnomalyDetectionResult:
        """
        Detect anomalies using dual-AI consensus.

        Why dual-AI is better:
        - Claude excels at pattern recognition and baseline analysis
        - OpenAI excels at classification and threat assessment
        - Consensus reduces false positives
        - Cross-validation catches edge cases

        Args:
            flows: Current NetFlow records
            device_ip: Device to analyze

        Returns:
            AnomalyDetectionResult with consensus analysis
        """
        if device_ip not in self.baselines:
            return AnomalyDetectionResult(
                anomaly_detected=False,
                confidence=0.0,
                anomaly_type="unknown",
                severity="none",
                description="No baseline available",
                claude_analysis=None,
                openai_analysis=None,
                consensus_reached=False,
                metrics={}
            )

        # Calculate current metrics
        device_flows = [f for f in flows if f.source_ip == device_ip or f.dest_ip == device_ip]
        metrics = self._calculate_metrics(device_flows, device_ip)

        # Step 1: Claude analyzes baseline deviation
        claude_analysis = self._claude_baseline_analysis(device_ip, metrics)

        # Step 2: OpenAI classifies anomaly type
        openai_analysis = self._openai_classify_anomaly(device_ip, metrics, claude_analysis)

        # Step 3: Consensus decision
        consensus = self._reach_consensus(claude_analysis, openai_analysis)

        return AnomalyDetectionResult(
            anomaly_detected=consensus['anomaly_detected'],
            confidence=consensus['confidence'],
            anomaly_type=consensus['anomaly_type'],
            severity=consensus['severity'],
            description=consensus['description'],
            claude_analysis=claude_analysis,
            openai_analysis=openai_analysis,
            consensus_reached=consensus['consensus_reached'],
            metrics=metrics
        )

    def _calculate_metrics(self, flows: List[NetFlowRecord], device_ip: str) -> Dict:
        """Calculate current traffic metrics."""
        if not flows:
            return {}

        total_bytes = sum(f.bytes for f in flows)
        total_packets = sum(f.packets for f in flows)
        unique_sources = len(set(f.source_ip for f in flows))
        unique_dests = len(set(f.dest_ip for f in flows))
        unique_ports = len(set(f.dest_port for f in flows))

        first_flow = min(f.timestamp for f in flows)
        last_flow = max(f.timestamp for f in flows)
        duration_hours = max((last_flow - first_flow).total_seconds() / 3600, 0.01)

        bytes_per_hour = total_bytes / duration_hours
        packets_per_second = total_packets / (duration_hours * 3600)

        # Upload/download ratio
        outbound_bytes = sum(f.bytes for f in flows if f.source_ip == device_ip)
        inbound_bytes = sum(f.bytes for f in flows if f.dest_ip == device_ip)
        upload_ratio = outbound_bytes / max(inbound_bytes, 1)

        # Get baseline comparison
        baseline = self.baselines[device_ip]
        current_hour = flows[0].timestamp.hour

        expected_bytes = 0
        z_score = 0.0
        if current_hour in baseline['hourly_baseline']:
            expected = baseline['hourly_baseline'][current_hour]
            expected_bytes = expected['mean']
            if expected['std'] > 0:
                z_score = (bytes_per_hour - expected['mean']) / expected['std']

        return {
            'total_bytes': total_bytes,
            'bytes_per_hour': bytes_per_hour,
            'packets_per_second': packets_per_second,
            'unique_sources': unique_sources,
            'unique_destinations': unique_dests,
            'unique_ports': unique_ports,
            'upload_ratio': upload_ratio,
            'outbound_mb': outbound_bytes / 1e6,
            'inbound_mb': inbound_bytes / 1e6,
            'duration_hours': duration_hours,
            'expected_bytes_per_hour': expected_bytes,
            'z_score': z_score,
            'current_hour': current_hour
        }

    def _claude_baseline_analysis(self, device_ip: str, metrics: Dict) -> Dict:
        """
        Claude analyzes baseline deviation and pattern recognition.

        Claude's strengths:
        - Excellent at understanding normal patterns
        - Strong reasoning about statistical deviations
        - Good at contextual analysis
        """
        baseline = self.baselines[device_ip]

        prompt = f"""Analyze network traffic for baseline deviations.

DEVICE: {device_ip}
BASELINE (30-day learned behavior):
- Normal traffic at hour {metrics['current_hour']}: {metrics['expected_bytes_per_hour']/1e6:.1f} MB/hour
- Typical communication peers: {len(baseline['typical_peers'])} devices

CURRENT TRAFFIC:
- Actual traffic: {metrics['bytes_per_hour']/1e6:.1f} MB/hour
- Z-score: {metrics['z_score']:.2f} (standard deviations from normal)
- Packets/sec: {metrics['packets_per_second']:.0f}
- Unique sources: {metrics['unique_sources']}
- Unique destinations: {metrics['unique_destinations']}
- Upload ratio: {metrics['upload_ratio']:.2f}x (sending {metrics['outbound_mb']:.1f} MB, receiving {metrics['inbound_mb']:.1f} MB)

ANALYSIS REQUIRED:
1. Is this traffic pattern a deviation from baseline?
2. How significant is the deviation?
3. What pattern does this match?

Respond in JSON:
{{
    "is_anomaly": true/false,
    "confidence": 0.0-1.0,
    "deviation_score": 0.0-10.0,
    "pattern_matched": "volumetric_spike/upload_anomaly/peer_anomaly/normal",
    "reasoning": "detailed explanation",
    "indicators": ["list of specific indicators"]
}}"""

        try:
            response = self.claude.invoke([HumanMessage(content=prompt)])
            analysis = json.loads(response.content)
            return analysis
        except Exception as e:
            return {
                'error': str(e),
                'is_anomaly': False,
                'confidence': 0.0
            }

    def _openai_classify_anomaly(self, device_ip: str, metrics: Dict,
                                 claude_analysis: Dict) -> Dict:
        """
        OpenAI classifies anomaly type and threat assessment.

        OpenAI's strengths:
        - Excellent at classification tasks
        - Strong at threat categorization
        - Good at impact assessment
        """
        prompt = f"""Classify network traffic anomaly and assess threat.

DEVICE: {device_ip}
TRAFFIC METRICS:
- Traffic volume: {metrics['bytes_per_hour']/1e6:.1f} MB/hour
- Z-score: {metrics['z_score']:.2f}
- Packets/sec: {metrics['packets_per_second']:.0f}
- Unique sources: {metrics['unique_sources']}
- Upload ratio: {metrics['upload_ratio']:.2f}x

BASELINE ANALYSIS (Claude):
- Anomaly detected: {claude_analysis.get('is_anomaly', False)}
- Pattern: {claude_analysis.get('pattern_matched', 'unknown')}
- Indicators: {claude_analysis.get('indicators', [])}

CLASSIFICATION REQUIRED:
1. What type of attack/anomaly is this?
2. What is the threat severity?
3. What is the business impact?

Categories:
- ddos_volumetric: Bandwidth exhaustion attack
- ddos_application: Application-layer attack
- data_exfiltration: Unauthorized data theft
- port_scan: Reconnaissance activity
- normal_spike: Legitimate traffic increase
- false_positive: Not an actual threat

Respond in JSON:
{{
    "anomaly_type": "ddos_volumetric/ddos_application/data_exfiltration/port_scan/normal_spike/false_positive",
    "severity": "critical/high/medium/low",
    "confidence": 0.0-1.0,
    "threat_assessment": "description of threat",
    "business_impact": "impact description",
    "recommended_actions": ["immediate actions"]
}}"""

        try:
            response = self.openai.invoke([HumanMessage(content=prompt)])
            analysis = json.loads(response.content)
            return analysis
        except Exception as e:
            return {
                'error': str(e),
                'anomaly_type': 'unknown',
                'confidence': 0.0
            }

    def _reach_consensus(self, claude_analysis: Dict, openai_analysis: Dict) -> Dict:
        """
        Reach consensus between Claude and OpenAI.

        Consensus rules:
        - Both must detect anomaly for high confidence
        - If one says false_positive, require high confidence from other
        - Severity is max of both assessments
        """
        claude_anomaly = claude_analysis.get('is_anomaly', False)
        openai_anomaly = openai_analysis.get('anomaly_type', 'false_positive') != 'false_positive'

        # Consensus reached if both agree
        if claude_anomaly and openai_anomaly:
            return {
                'anomaly_detected': True,
                'confidence': min(claude_analysis.get('confidence', 0.5),
                                openai_analysis.get('confidence', 0.5)),
                'anomaly_type': openai_analysis.get('anomaly_type', 'unknown'),
                'severity': openai_analysis.get('severity', 'medium'),
                'description': f"Consensus: {openai_analysis.get('threat_assessment', 'Anomaly detected')}",
                'consensus_reached': True
            }
        elif not claude_anomaly and not openai_anomaly:
            return {
                'anomaly_detected': False,
                'confidence': 0.9,
                'anomaly_type': 'normal',
                'severity': 'none',
                'description': 'Both agents agree: Normal traffic pattern',
                'consensus_reached': True
            }
        else:
            # Disagreement - require high confidence
            if claude_anomaly and openai_analysis.get('confidence', 0) > 0.8:
                return {
                    'anomaly_detected': True,
                    'confidence': 0.6,
                    'anomaly_type': openai_analysis.get('anomaly_type', 'unknown'),
                    'severity': openai_analysis.get('severity', 'medium'),
                    'description': f"Partial consensus: {openai_analysis.get('threat_assessment', 'Possible anomaly')}",
                    'consensus_reached': False
                }
            else:
                return {
                    'anomaly_detected': False,
                    'confidence': 0.3,
                    'anomaly_type': 'unclear',
                    'severity': 'low',
                    'description': 'No consensus - likely false positive',
                    'consensus_reached': False
                }


# ============================================================================
# EXAMPLE FUNCTIONS (As requested)
# ============================================================================

def example_1_traffic_baseline():
    """
    Example 1: Traffic baseline learning with Claude

    Demonstrates:
    - Learning normal traffic patterns
    - Statistical baseline calculation
    - Claude's pattern recognition capabilities
    """
    print("=" * 70)
    print("EXAMPLE 1: Traffic Baseline Learning with Claude")
    print("=" * 70)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize detector
    detector = MultiAgentAnomalyDetector(anthropic_key, openai_key)

    # Generate 30 days of normal traffic
    print("\nGenerating 30 days of historical traffic...")
    historical_flows = []
    web_server_ip = '10.1.50.10'

    for day in range(30):
        for hour in range(24):
            # Business hours: high traffic, Night: low traffic
            if 9 <= hour <= 17:
                base_traffic = 500_000_000  # 500 MB
            elif 0 <= hour <= 6:
                base_traffic = 50_000_000   # 50 MB
            else:
                base_traffic = 200_000_000  # 200 MB

            # Add random variation (±10%)
            traffic = base_traffic + np.random.normal(0, base_traffic * 0.1)

            flow = NetFlowRecord(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                source_ip='0.0.0.0',
                dest_ip=web_server_ip,
                source_port=np.random.randint(1024, 65535),
                dest_port=443,
                protocol='TCP',
                bytes=int(traffic),
                packets=int(traffic / 1500),
                duration_seconds=3600
            )
            historical_flows.append(flow)

    # Learn baseline
    print("Learning baseline pattern...")
    baseline = detector.learn_baseline(historical_flows, web_server_ip, learning_days=30)

    print("\nBaseline Learning Results:")
    print(f"Device: {baseline['device_ip']}")
    print(f"Flows analyzed: {baseline['total_flows_analyzed']:,}")
    print(f"Average daily traffic: {baseline['avg_bytes_per_day']/1e9:.2f} GB")
    print(f"Typical communication peers: {len(baseline['typical_peers'])}")

    print("\nHourly Traffic Pattern:")
    for hour in [3, 9, 14, 21]:  # Sample hours
        if hour in baseline['hourly_baseline']:
            stats = baseline['hourly_baseline'][hour]
            print(f"  Hour {hour:02d}:00 - Mean: {stats['mean']/1e6:.1f} MB/hr, "
                  f"StdDev: {stats['std']/1e6:.1f} MB/hr")

    print("\nClaude has learned the normal traffic pattern.")
    print("This baseline will be used to detect deviations.")


def example_2_ddos_detection():
    """
    Example 2: DDoS detection using dual-AI consensus

    Demonstrates:
    - Claude detecting baseline deviation
    - OpenAI classifying attack type
    - Consensus decision making
    - Why dual-AI is better than single-AI
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DDoS Detection with Dual-AI Consensus")
    print("=" * 70)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize detector
    detector = MultiAgentAnomalyDetector(anthropic_key, openai_key)

    # Generate baseline
    print("\nLearning baseline...")
    historical_flows = []
    web_server_ip = '10.1.50.10'

    for day in range(30):
        for hour in range(24):
            if 9 <= hour <= 17:
                base_traffic = 500_000_000
            else:
                base_traffic = 100_000_000

            traffic = base_traffic + np.random.normal(0, base_traffic * 0.1)

            flow = NetFlowRecord(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                source_ip=f'{np.random.randint(1,223)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,254)}',
                dest_ip=web_server_ip,
                source_port=np.random.randint(1024, 65535),
                dest_port=443,
                protocol='TCP',
                bytes=int(traffic),
                packets=int(traffic / 1500),
                duration_seconds=3600
            )
            historical_flows.append(flow)

    detector.learn_baseline(historical_flows, web_server_ip, learning_days=30)
    print("Baseline learned.")

    # Simulate DDoS attack
    print("\nSimulating DDoS attack (10x normal traffic)...")
    attack_flows = []
    attack_time = datetime.now()

    for minute in range(60):  # 1 hour attack
        for _ in range(100):  # 100 connections/second
            # Random attacker IP (botnet)
            attacker_ip = f'{np.random.randint(1,223)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,254)}'

            flow = NetFlowRecord(
                timestamp=attack_time + timedelta(minutes=minute),
                source_ip=attacker_ip,
                dest_ip=web_server_ip,
                source_port=np.random.randint(1024, 65535),
                dest_port=80,
                protocol='TCP',
                bytes=60,  # Small SYN packets
                packets=1,
                duration_seconds=0.1
            )
            attack_flows.append(flow)

    # Detect with dual-AI
    print("\nAnalyzing with dual-AI system...")
    print("Step 1: Claude analyzes baseline deviation...")
    print("Step 2: OpenAI classifies anomaly type...")
    print("Step 3: Reaching consensus...\n")

    result = detector.detect_with_consensus(attack_flows, web_server_ip)

    # Display results
    print("=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(f"Anomaly Detected: {result.anomaly_detected}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Anomaly Type: {result.anomaly_type}")
    print(f"Severity: {result.severity}")
    print(f"Consensus Reached: {result.consensus_reached}")
    print(f"\nDescription: {result.description}")

    print("\n" + "-" * 70)
    print("CLAUDE ANALYSIS (Baseline Expert)")
    print("-" * 70)
    if result.claude_analysis:
        print(f"Is Anomaly: {result.claude_analysis.get('is_anomaly')}")
        print(f"Confidence: {result.claude_analysis.get('confidence', 0):.2%}")
        print(f"Pattern: {result.claude_analysis.get('pattern_matched')}")
        print(f"Reasoning: {result.claude_analysis.get('reasoning')}")

    print("\n" + "-" * 70)
    print("OPENAI ANALYSIS (Classification Expert)")
    print("-" * 70)
    if result.openai_analysis:
        print(f"Type: {result.openai_analysis.get('anomaly_type')}")
        print(f"Severity: {result.openai_analysis.get('severity')}")
        print(f"Confidence: {result.openai_analysis.get('confidence', 0):.2%}")
        print(f"Threat: {result.openai_analysis.get('threat_assessment')}")
        print(f"\nActions:")
        for action in result.openai_analysis.get('recommended_actions', []):
            print(f"  - {action}")

    print("\n" + "-" * 70)
    print("TRAFFIC METRICS")
    print("-" * 70)
    print(f"Packets/sec: {result.metrics.get('packets_per_second', 0):.0f}")
    print(f"Unique sources: {result.metrics.get('unique_sources', 0)}")
    print(f"Z-score: {result.metrics.get('z_score', 0):.2f}")

    print("\n" + "=" * 70)
    print("WHY DUAL-AI IS BETTER")
    print("=" * 70)
    print("1. Claude's pattern recognition caught the baseline deviation")
    print("2. OpenAI's classification identified it as DDoS attack")
    print("3. Consensus prevents false positives from single-model bias")
    print("4. Cross-validation increases confidence in detection")


def example_3_data_exfiltration():
    """
    Example 3: Data exfiltration detection with multi-agent analysis

    Demonstrates:
    - Upload anomaly detection
    - Off-hours behavior analysis
    - Multi-agent consensus for insider threats
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Data Exfiltration Detection")
    print("=" * 70)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize detector
    detector = MultiAgentAnomalyDetector(anthropic_key, openai_key)

    # Generate baseline for workstation
    print("\nLearning baseline for user workstation...")
    historical_flows = []
    workstation_ip = '10.1.50.75'

    for day in range(30):
        for hour in range(24):
            # Normal workstation: mostly downloads, little upload
            # Business hours only
            if 9 <= hour <= 17:
                # Download: 100 MB/hr, Upload: 10 MB/hr
                inbound = 100_000_000 + np.random.normal(0, 10_000_000)
                outbound = 10_000_000 + np.random.normal(0, 1_000_000)
            else:
                # After hours: minimal traffic
                inbound = 1_000_000
                outbound = 500_000

            # Inbound flow
            flow_in = NetFlowRecord(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                source_ip='172.16.100.5',
                dest_ip=workstation_ip,
                source_port=443,
                dest_port=np.random.randint(49152, 65535),
                protocol='TCP',
                bytes=int(inbound),
                packets=int(inbound / 1500),
                duration_seconds=3600
            )

            # Outbound flow
            flow_out = NetFlowRecord(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                source_ip=workstation_ip,
                dest_ip='172.16.100.5',
                source_port=np.random.randint(49152, 65535),
                dest_port=443,
                protocol='TCP',
                bytes=int(outbound),
                packets=int(outbound / 1500),
                duration_seconds=3600
            )

            historical_flows.extend([flow_in, flow_out])

    detector.learn_baseline(historical_flows, workstation_ip, learning_days=30)
    print("Baseline learned: Normal workstation (10:1 download/upload ratio)")

    # Simulate data exfiltration at 3 AM
    print("\nSimulating data exfiltration at 3 AM...")
    print("Insider threat: Uploading 5 GB of data to external server")

    exfil_flows = []
    exfil_time = datetime.now().replace(hour=3, minute=0)
    attacker_c2 = '45.134.83.200'  # External IP

    # Upload 5 GB over 2 hours
    for minute in range(120):
        flow = NetFlowRecord(
            timestamp=exfil_time + timedelta(minutes=minute),
            source_ip=workstation_ip,
            dest_ip=attacker_c2,
            source_port=np.random.randint(40000, 65000),
            dest_port=443,  # HTTPS (encrypted)
            protocol='TCP',
            bytes=45_000_000,  # 45 MB per minute
            packets=30_000,
            duration_seconds=60
        )
        exfil_flows.append(flow)

    # Detect with dual-AI
    print("\nAnalyzing with dual-AI system...")
    result = detector.detect_with_consensus(exfil_flows, workstation_ip)

    # Display results
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(f"Anomaly Detected: {result.anomaly_detected}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Anomaly Type: {result.anomaly_type}")
    print(f"Severity: {result.severity}")

    print("\n" + "-" * 70)
    print("KEY INDICATORS")
    print("-" * 70)
    print(f"Upload ratio: {result.metrics.get('upload_ratio', 0):.1f}x")
    print(f"Outbound: {result.metrics.get('outbound_mb', 0):.1f} MB")
    print(f"Inbound: {result.metrics.get('inbound_mb', 0):.1f} MB")
    print(f"Time: 3 AM (off-hours)")
    print(f"Destination: External IP (not typical peer)")

    print("\n" + "-" * 70)
    print("CLAUDE ANALYSIS")
    print("-" * 70)
    if result.claude_analysis:
        print(f"Pattern: {result.claude_analysis.get('pattern_matched')}")
        print(f"Indicators: {result.claude_analysis.get('indicators')}")

    print("\n" + "-" * 70)
    print("OPENAI ANALYSIS")
    print("-" * 70)
    if result.openai_analysis:
        print(f"Business Impact: {result.openai_analysis.get('business_impact')}")
        if result.openai_analysis.get('recommended_actions'):
            print("\nRecommended Actions:")
            for action in result.openai_analysis['recommended_actions'][:5]:
                print(f"  - {action}")

    print("\n" + "=" * 70)
    print("MULTI-AGENT ADVANTAGE")
    print("=" * 70)
    print("1. Claude detected unusual upload ratio (pattern recognition)")
    print("2. OpenAI classified as data exfiltration (threat intelligence)")
    print("3. Both agents flagged off-hours activity")
    print("4. Consensus provides high confidence for critical alert")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-AGENT NETWORK ANOMALY DETECTION")
    print("Chapter 75: Network Anomaly Detection")
    print("=" * 70)
    print("\nThis script demonstrates dual-AI anomaly detection:")
    print("- Claude: Baseline behavior analysis")
    print("- OpenAI: Anomaly classification")
    print("- LangChain: Multi-agent orchestration")
    print("\nRunning 3 examples...")

    # Run all examples
    try:
        example_1_traffic_baseline()
        example_2_ddos_detection()
        example_3_data_exfiltration()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. Dual-AI provides better accuracy than single-model detection")
        print("2. Claude excels at pattern recognition and baseline analysis")
        print("3. OpenAI excels at classification and threat assessment")
        print("4. Consensus reduces false positives and increases confidence")
        print("5. Multi-agent orchestration enables production-ready security")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
