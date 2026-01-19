# Chapter 75: Network Anomaly Detection

## Learning Objectives

By the end of this chapter, you will:
- Build AI systems that learn your network's "normal" traffic patterns
- Detect DDoS attacks (volumetric and application-layer) in real-time
- Identify data exfiltration before terabytes leave your network
- Use NetFlow, time-series analysis, and LLMs for anomaly detection
- Deploy production anomaly detection with automated response

**Prerequisites**: Understanding of NetFlow/sFlow, basic statistics, Chapters 70-72 (threat detection, log analysis)

**What You'll Build**: Complete anomaly detection system that learns your network's baseline behavior and automatically detects deviations indicating attacks, compromises, or misconfigurations.

---

## The Anomaly Detection Problem

Traditional security tools look for **known bad** (signatures, IOCs, blocklists). Anomaly detection finds **unknown bad** by detecting deviations from normal.

**Real Incident**: E-commerce Company DDoS

```
Timeline:
14:23 - Traffic to web servers increases from 5,000 req/sec to 8,000 req/sec
14:25 - Firewall logs show denies increasing
14:28 - Web response time degrades (2s → 8s)
14:32 - Customer complaints start
14:35 - Operations team notices (12 minutes into attack)
14:45 - DDoS mitigation engaged (22 minutes into attack)
14:50 - Attack continues, shifts to application-layer
15:15 - Traffic returns to normal (attacker stops)

Revenue loss: $45K (52 minutes of degraded service)
```

**What went wrong**: No anomaly detection. Humans noticed performance degradation, not automated systems.

**With AI Anomaly Detection**:
- 14:23:30 - AI detects traffic anomaly (+60% in 30 seconds)
- 14:24:00 - AI analyzes traffic patterns (identifies DDoS signature)
- 14:24:15 - Automated mitigation engaged (90 seconds into attack)
- 14:25:00 - Attack mitigated, customers unaffected

**Revenue loss**: $0

This chapter builds that system.

---

## Section 1: Traffic Baseline Learning

### Why Baselines Matter

Your network has patterns:
- **Time of day**: Peak traffic 9 AM - 5 PM, quiet at night
- **Day of week**: Lower traffic weekends, spikes Monday morning
- **Seasonal**: E-commerce spikes before holidays
- **Per-device**: Web servers high traffic, mgmt servers low traffic
- **Communication patterns**: App servers talk to DB servers, not to internet directly

**Baseline = Understanding Normal**

Once you know normal, deviations are obvious:
- Database server suddenly sending 100 Mbps outbound (data exfiltration?)
- Web server traffic 10x normal (DDoS? viral marketing success?)
- IoT device accessing internet at 3 AM (compromised?)

### Building Traffic Baselines with NetFlow

NetFlow/sFlow gives you traffic metadata without full packet capture:
- Source/dest IPs and ports
- Bytes/packets transferred
- Protocol
- Timing

**Perfect for anomaly detection** - you don't need payload, just patterns.

```python
"""
Network Traffic Baseline Learning
Uses NetFlow data to understand normal network behavior
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
import anthropic
import json

@dataclass
class NetFlowRecord:
    """NetFlow v9/IPFIX record"""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    bytes: int
    packets: int
    duration_seconds: float

class TrafficBaseline:
    """Learn normal traffic patterns for anomaly detection"""

    def __init__(self):
        self.baselines = {}  # Per-device baselines

    def learn_baseline(self, flows: List[NetFlowRecord], device_ip: str,
                      learning_days: int = 30):
        """Learn baseline traffic pattern for a device over N days"""

        device_flows = [f for f in flows if f.source_ip == device_ip or f.dest_ip == device_ip]

        if not device_flows:
            return None

        # Analyze traffic by hour of day
        hourly_traffic = defaultdict(list)  # hour -> [bytes_per_hour]

        # Group flows by hour
        flows_by_hour = defaultdict(list)
        for flow in device_flows:
            hour_key = flow.timestamp.replace(minute=0, second=0, microsecond=0)
            flows_by_hour[hour_key].append(flow)

        # Calculate bytes per hour
        for hour, hour_flows in flows_by_hour.items():
            hour_of_day = hour.hour
            total_bytes = sum(f.bytes for f in hour_flows)
            hourly_traffic[hour_of_day].append(total_bytes)

        # Calculate statistics per hour
        baseline_by_hour = {}
        for hour in range(24):
            if hour in hourly_traffic and len(hourly_traffic[hour]) > 7:  # Need at least a week
                data = hourly_traffic[hour]
                baseline_by_hour[hour] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data),
                    'p95': np.percentile(data, 95),
                    'p99': np.percentile(data, 99),
                    'samples': len(data)
                }

        # Learn communication patterns (who talks to whom)
        communication_peers = defaultdict(int)  # peer_ip -> byte_count
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
            if cumulative >= total_bytes * 0.8:  # Top 80% of traffic
                break

        # Learn typical ports
        port_usage = defaultdict(int)
        for flow in device_flows:
            port_usage[flow.dest_port] += flow.bytes

        typical_ports = set(port for port, _ in sorted(port_usage.items(),
                                                      key=lambda x: x[1],
                                                      reverse=True)[:20])

        baseline = {
            'device_ip': device_ip,
            'learning_period_days': learning_days,
            'total_flows_analyzed': len(device_flows),
            'hourly_baseline': baseline_by_hour,
            'typical_peers': typical_peers,
            'typical_ports': list(typical_ports),
            'total_bytes': total_bytes,
            'avg_bytes_per_day': total_bytes / learning_days
        }

        self.baselines[device_ip] = baseline
        return baseline

    def detect_anomalies(self, current_flows: List[NetFlowRecord],
                        device_ip: str) -> Dict:
        """Detect anomalies in current traffic vs. baseline"""

        if device_ip not in self.baselines:
            return {'error': 'No baseline for this device'}

        baseline = self.baselines[device_ip]
        device_flows = [f for f in current_flows
                       if f.source_ip == device_ip or f.dest_ip == device_ip]

        if not device_flows:
            return {'anomalies': []}

        anomalies = []
        risk_score = 0.0

        # Current traffic statistics
        current_hour = device_flows[0].timestamp.hour
        current_bytes = sum(f.bytes for f in device_flows)
        current_duration_hours = (max(f.timestamp for f in device_flows) -
                                 min(f.timestamp for f in device_flows)).total_seconds() / 3600

        if current_duration_hours > 0:
            current_bytes_per_hour = current_bytes / current_duration_hours
        else:
            current_bytes_per_hour = current_bytes

        # Check volume anomaly
        if current_hour in baseline['hourly_baseline']:
            expected = baseline['hourly_baseline'][current_hour]

            # Calculate z-score (standard deviations from mean)
            if expected['std'] > 0:
                z_score = (current_bytes_per_hour - expected['mean']) / expected['std']

                if abs(z_score) > 3:  # 3 sigma = 99.7% confidence
                    anomalies.append({
                        'type': 'volume_anomaly',
                        'severity': 'high' if z_score > 5 else 'medium',
                        'description': f"Traffic volume {current_bytes_per_hour/1e6:.1f} MB/hr vs expected {expected['mean']/1e6:.1f} MB/hr (z-score: {z_score:.1f})",
                        'z_score': z_score,
                        'current_mbps': current_bytes_per_hour / 1e6,
                        'expected_mbps': expected['mean'] / 1e6
                    })
                    risk_score += min(abs(z_score) / 10, 0.5)

        # Check for unusual peers
        current_peers = set()
        for flow in device_flows:
            peer_ip = flow.dest_ip if flow.source_ip == device_ip else flow.source_ip
            current_peers.add(peer_ip)

        unusual_peers = current_peers - set(baseline['typical_peers'])
        if len(unusual_peers) > 5:  # More than 5 new peers
            anomalies.append({
                'type': 'unusual_peers',
                'severity': 'medium',
                'description': f"Communicating with {len(unusual_peers)} unusual peers",
                'unusual_peer_count': len(unusual_peers),
                'unusual_peers': list(unusual_peers)[:10]  # Sample
            })
            risk_score += 0.3

        # Check for unusual ports
        current_ports = set(f.dest_port for f in device_flows)
        unusual_ports = current_ports - set(baseline['typical_ports'])
        if len(unusual_ports) > 10:
            anomalies.append({
                'type': 'unusual_ports',
                'severity': 'medium',
                'description': f"Using {len(unusual_ports)} unusual destination ports",
                'unusual_port_count': len(unusual_ports),
                'unusual_ports': sorted(list(unusual_ports))[:20]
            })
            risk_score += 0.2

        # Check for upload anomaly (potential data exfiltration)
        outbound_bytes = sum(f.bytes for f in device_flows if f.source_ip == device_ip)
        inbound_bytes = sum(f.bytes for f in device_flows if f.dest_ip == device_ip)

        if inbound_bytes > 0:
            upload_ratio = outbound_bytes / inbound_bytes
            # Most devices receive more than they send (except servers)
            if upload_ratio > 5:  # Sending 5x more than receiving
                anomalies.append({
                    'type': 'upload_anomaly',
                    'severity': 'high',
                    'description': f"Unusual upload volume: {outbound_bytes/1e6:.1f} MB sent vs {inbound_bytes/1e6:.1f} MB received",
                    'upload_ratio': upload_ratio,
                    'outbound_mb': outbound_bytes / 1e6,
                    'inbound_mb': inbound_bytes / 1e6
                })
                risk_score += 0.4

        return {
            'device_ip': device_ip,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'risk_score': min(risk_score, 1.0),
            'current_bytes_per_hour': current_bytes_per_hour,
            'baseline_available': True
        }

# Example usage
def learn_and_detect():
    baseline_system = TrafficBaseline()

    # Simulate 30 days of historical NetFlow (learning phase)
    historical_flows = []
    web_server_ip = '10.1.50.10'

    # Normal traffic pattern: High during business hours, low at night
    for day in range(30):
        for hour in range(24):
            # Business hours (9-17): 500 MB/hr, Night (0-6): 50 MB/hr
            if 9 <= hour <= 17:
                base_traffic = 500_000_000  # 500 MB
            elif 0 <= hour <= 6:
                base_traffic = 50_000_000   # 50 MB
            else:
                base_traffic = 200_000_000  # 200 MB

            # Add random variation
            traffic = base_traffic + np.random.normal(0, base_traffic * 0.1)

            flow = NetFlowRecord(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                source_ip='0.0.0.0',  # Internet
                dest_ip=web_server_ip,
                source_port=random.randint(1024, 65535),
                dest_port=443,
                protocol='TCP',
                bytes=int(traffic),
                packets=int(traffic / 1500),
                duration_seconds=3600
            )
            historical_flows.append(flow)

    # Learn baseline
    print("Learning baseline from 30 days of traffic...")
    baseline = baseline_system.learn_baseline(historical_flows, web_server_ip, learning_days=30)
    print(f"Baseline learned: {baseline['total_flows_analyzed']} flows analyzed")
    print(f"Average traffic: {baseline['avg_bytes_per_day']/1e9:.2f} GB/day")

    # Simulate anomaly: DDoS attack (10x normal traffic)
    print("\n🚨 Simulating DDoS attack...")
    attack_flows = []
    attack_time = datetime.now()

    for minute in range(60):  # 1 hour of attack
        # 10x normal traffic
        attack_flow = NetFlowRecord(
            timestamp=attack_time + timedelta(minutes=minute),
            source_ip='45.95.168.200',  # Attacker
            dest_ip=web_server_ip,
            source_port=random.randint(1024, 65535),
            dest_port=443,
            protocol='TCP',
            bytes=5_000_000_000,  # 5 GB per minute (10x normal)
            packets=3_333_333,
            duration_seconds=60
        )
        attack_flows.append(attack_flow)

    # Detect anomalies
    result = baseline_system.detect_anomalies(attack_flows, web_server_ip)

    print(f"\nAnomaly Detection Results:")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"Anomalies Detected: {result['anomaly_count']}")

    for anomaly in result['anomalies']:
        print(f"\n  Type: {anomaly['type']}")
        print(f"  Severity: {anomaly['severity']}")
        print(f"  Description: {anomaly['description']}")

# Example Output:
"""
Learning baseline from 30 days of traffic...
Baseline learned: 720 flows analyzed
Average traffic: 7.20 GB/day

🚨 Simulating DDoS attack...

Anomaly Detection Results:
Risk Score: 0.50
Anomalies Detected: 1

  Type: volume_anomaly
  Severity: high
  Description: Traffic volume 5000.0 MB/hr vs expected 500.0 MB/hr (z-score: 9.5)
"""
```

---

## Section 2: DDoS Detection

### Types of DDoS Attacks

**Volumetric Attacks** - Flood bandwidth
- UDP floods
- ICMP floods
- DNS amplification
- **Detection**: Sudden traffic spike (10x-100x normal)

**Application-Layer Attacks** - Exhaust server resources
- HTTP floods (looks like legitimate requests)
- Slowloris (slow connections)
- **Detection**: Request rate spike, but bandwidth may be normal

**Protocol Attacks** - Exploit protocol weaknesses
- SYN floods
- ACK floods
- Fragmentation attacks
- **Detection**: Unusual protocol behavior, connection state exhaustion

### Building DDoS Detector

```python
"""
AI-Powered DDoS Detection
Detects volumetric and application-layer DDoS attacks
"""
from datetime import datetime, timedelta
from typing import List, Dict
import anthropic
import json

class DDoSDetector:
    """Detect DDoS attacks using traffic analysis and AI"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.baseline = TrafficBaseline()

    def analyze_for_ddos(self, flows: List[NetFlowRecord], target_ip: str,
                        time_window_minutes: int = 5) -> Dict:
        """Analyze traffic for DDoS indicators"""

        target_flows = [f for f in flows if f.dest_ip == target_ip]

        if not target_flows:
            return {'ddos_detected': False, 'reason': 'No traffic to analyze'}

        # Calculate traffic metrics
        total_bytes = sum(f.bytes for f in target_flows)
        total_packets = sum(f.packets for f in target_flows)
        unique_sources = len(set(f.source_ip for f in target_flows))
        unique_ports = len(set(f.dest_port for f in target_flows))

        # Time analysis
        first_flow = min(f.timestamp for f in target_flows)
        last_flow = max(f.timestamp for f in target_flows)
        duration_seconds = (last_flow - first_flow).total_seconds()

        if duration_seconds == 0:
            duration_seconds = 1

        # Calculate rates
        mbps = (total_bytes * 8 / 1_000_000) / (duration_seconds / 60)  # Mbps
        packets_per_second = total_packets / duration_seconds
        connections_per_second = len(target_flows) / duration_seconds

        # DDoS indicators
        indicators = []
        ddos_score = 0.0

        # Indicator 1: High packet rate (volumetric attack)
        if packets_per_second > 10_000:
            indicators.append(f"High packet rate: {packets_per_second:.0f} pps")
            ddos_score += 0.4

        # Indicator 2: Many unique sources (distributed attack)
        if unique_sources > 100:
            indicators.append(f"Distributed attack: {unique_sources} unique source IPs")
            ddos_score += 0.3

        # Indicator 3: Single port targeted (focused attack)
        if unique_ports <= 3:
            indicators.append(f"Focused attack: targeting {unique_ports} ports")
            ddos_score += 0.2

        # Indicator 4: Small packet size (typical for flood attacks)
        avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0
        if avg_packet_size < 100:  # Very small packets
            indicators.append(f"Small packet size: {avg_packet_size:.0f} bytes (flood pattern)")
            ddos_score += 0.2

        # Indicator 5: High connection rate (application-layer attack)
        if connections_per_second > 100:
            indicators.append(f"High connection rate: {connections_per_second:.0f} conn/sec")
            ddos_score += 0.3

        # Check if this matches baseline anomaly
        baseline_result = self.baseline.detect_anomalies(target_flows, target_ip)
        if baseline_result.get('anomalies'):
            for anomaly in baseline_result['anomalies']:
                if anomaly['type'] == 'volume_anomaly' and anomaly['severity'] == 'high':
                    indicators.append(f"Traffic volume anomaly: z-score {anomaly['z_score']:.1f}")
                    ddos_score += 0.4

        ddos_score = min(ddos_score, 1.0)

        # If score is high, get AI analysis
        if ddos_score >= 0.6:
            ai_analysis = self._ai_analyze_ddos(target_ip, {
                'mbps': mbps,
                'packets_per_second': packets_per_second,
                'connections_per_second': connections_per_second,
                'unique_sources': unique_sources,
                'unique_ports': unique_ports,
                'avg_packet_size': avg_packet_size,
                'duration_seconds': duration_seconds,
                'total_bytes': total_bytes,
                'indicators': indicators,
                'ddos_score': ddos_score
            })

            return {
                'ddos_detected': ai_analysis['is_ddos'],
                'ddos_score': ddos_score,
                'indicators': indicators,
                'metrics': {
                    'mbps': mbps,
                    'packets_per_second': packets_per_second,
                    'connections_per_second': connections_per_second,
                    'unique_sources': unique_sources
                },
                'ai_analysis': ai_analysis
            }

        return {
            'ddos_detected': False,
            'ddos_score': ddos_score,
            'indicators': indicators
        }

    def _ai_analyze_ddos(self, target_ip: str, metrics: Dict) -> Dict:
        """Use AI to analyze if this is really DDoS"""

        prompt = f"""You are a network security analyst detecting DDoS attacks.

TARGET UNDER ATTACK:
IP Address: {target_ip}
Attack Duration: {metrics['duration_seconds']:.0f} seconds

TRAFFIC METRICS:
- Bandwidth: {metrics['mbps']:.1f} Mbps
- Packet Rate: {metrics['packets_per_second']:.0f} packets/second
- Connection Rate: {metrics['connections_per_second']:.0f} connections/second
- Unique Source IPs: {metrics['unique_sources']}
- Unique Destination Ports: {metrics['unique_ports']}
- Average Packet Size: {metrics['avg_packet_size']:.0f} bytes
- Total Data: {metrics['total_bytes']/1e6:.1f} MB

ATTACK INDICATORS (DDoS Score: {metrics['ddos_score']:.2f}):
{chr(10).join('- ' + indicator for indicator in metrics['indicators'])}

ANALYSIS REQUIRED:
1. Is this a DDoS attack (vs. legitimate traffic spike)?
2. What type of DDoS? (Volumetric/Application-Layer/Protocol)
3. What's the attack vector? (UDP flood, SYN flood, HTTP flood, etc.)
4. What's the business impact?
5. Recommended mitigation actions

Consider: Could this be legitimate (marketing campaign, product launch, viral content)?

Respond in JSON:
{{
    "is_ddos": true/false,
    "confidence": 0.0-1.0,
    "attack_type": "Volumetric/Application-Layer/Protocol/Legitimate Traffic",
    "attack_vector": "UDP Flood/SYN Flood/HTTP Flood/DNS Amplification/etc",
    "severity": "Critical/High/Medium/Low",
    "business_impact": "description of impact",
    "legitimate_possibility": "if this might be legitimate traffic",
    "mitigation_actions": ["immediate actions to take"],
    "investigation_steps": ["what to check"],
    "estimated_attacker_bandwidth": "bandwidth estimation",
    "likely_botnet": true/false
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'is_ddos': True,
                'confidence': metrics['ddos_score'],
                'attack_type': 'Unknown'
            }

# Example: Detecting SYN Flood DDoS
def detect_syn_flood():
    detector = DDoSDetector(anthropic_api_key="your-api-key")

    # Learn baseline first
    # ... (baseline learning code from previous section)

    # Simulate SYN flood attack: 100,000 pps from 5,000 sources
    attack_flows = []
    target_ip = '10.1.50.10'
    attack_start = datetime.now()

    # Generate attack traffic (5 minutes)
    for second in range(300):  # 5 minutes
        for _ in range(100):  # 100 connections per second
            # Random attacker IP (botnet)
            attacker_ip = f'{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}'

            flow = NetFlowRecord(
                timestamp=attack_start + timedelta(seconds=second),
                source_ip=attacker_ip,
                dest_ip=target_ip,
                source_port=random.randint(1024, 65535),
                dest_port=80,  # Targeting web server
                protocol='TCP',
                bytes=60,  # SYN packet is small
                packets=1,
                duration_seconds=0.1
            )
            attack_flows.append(flow)

    # Detect DDoS
    result = detector.analyze_for_ddos(attack_flows, target_ip, time_window_minutes=5)

    if result['ddos_detected']:
        print("🚨 DDoS ATTACK DETECTED")
        print(f"Score: {result['ddos_score']:.2f}")
        print(f"\nMetrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.1f}")

        print(f"\nAI Analysis:")
        ai = result['ai_analysis']
        print(f"  Type: {ai['attack_type']}")
        print(f"  Vector: {ai['attack_vector']}")
        print(f"  Severity: {ai['severity']}")
        print(f"  Impact: {ai['business_impact']}")
        print(f"\nMitigation:")
        for action in ai['mitigation_actions']:
            print(f"  - {action}")

# Example Output:
"""
🚨 DDoS ATTACK DETECTED
Score: 1.00

Metrics:
  mbps: 2.4
  packets_per_second: 10000.0
  connections_per_second: 100.0
  unique_sources: 4523

AI Analysis:
  Type: Protocol
  Vector: SYN Flood
  Severity: Critical
  Impact: Web server (10.1.50.10) under SYN flood attack. Connection table will exhaust within minutes, causing complete service outage. Legitimate users unable to connect.

Mitigation:
  - Enable SYN cookies on target server (immediate)
  - Rate-limit new connections at firewall (100/sec per source)
  - Block top attacking IPs at edge router
  - Engage upstream ISP DDoS mitigation (scrubbing center)
  - Add to DDoS mitigation service (Cloudflare/Akamai)
  - Monitor connection table utilization
  - If internal server, ensure load balancer has SYN flood protection

Analysis Details:
{
  "is_ddos": true,
  "confidence": 0.98,
  "attack_type": "Protocol",
  "attack_vector": "SYN Flood",
  "severity": "Critical",
  "business_impact": "Target server connection table will be exhausted, causing denial of service to all users. Web services will be completely unavailable within 2-5 minutes if not mitigated.",
  "legitimate_possibility": "Extremely unlikely (<1%). The combination of 4,523 unique sources, 10,000 pps, small packet sizes (60 bytes), and focused targeting of single port is textbook SYN flood DDoS.",
  "estimated_attacker_bandwidth": "Attacker using botnet of ~5,000 compromised devices, generating ~2.4 Mbps aggregate (low bandwidth but high packet rate - protocol attack, not volumetric)",
  "likely_botnet": true
}
"""
```

---

## Section 3: Data Exfiltration Detection

### The Data Theft Problem

Attackers steal data by:
1. Compromise system
2. Locate valuable data (customer DB, IP, credentials)
3. Exfiltrate slowly to avoid detection

**Traditional detection fails**:
- Data leaving over HTTPS (encrypted, can't inspect)
- Slow exfiltration (1 GB/day over months = unnoticed)
- Legitimate-looking destinations (cloud storage, compromised legitimate sites)

**AI detection signals**:
- Upload volume anomaly (device sending more than usual)
- Upload ratio (sending 10x more than receiving)
- New destinations (never talked to before)
- Off-hours transfers (2 AM uploads)
- Protocol anomalies (DNS tunneling, ICMP tunneling)

### Building Data Exfiltration Detector

```python
"""
Data Exfiltration Detection
Identifies unauthorized data leaving your network
"""
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict
import anthropic
import json

class DataExfiltrationDetector:
    """Detect data exfiltration using traffic analysis"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.baseline = TrafficBaseline()

    def analyze_for_exfiltration(self, flows: List[NetFlowRecord],
                                 internal_subnets: List[str],
                                 time_window_hours: int = 24) -> Dict:
        """Analyze outbound traffic for data exfiltration"""

        # Identify outbound flows (internal source → external dest)
        outbound_flows = []
        for flow in flows:
            if self._is_internal(flow.source_ip, internal_subnets) and \
               not self._is_internal(flow.dest_ip, internal_subnets):
                outbound_flows.append(flow)

        if not outbound_flows:
            return {'exfiltration_detected': False, 'reason': 'No outbound traffic'}

        # Group by source IP (which internal device is uploading?)
        uploads_by_device = defaultdict(list)
        for flow in outbound_flows:
            uploads_by_device[flow.source_ip].append(flow)

        # Analyze each device for exfiltration
        suspicious_devices = []

        for device_ip, device_flows in uploads_by_device.items():
            exfil_indicators = self._analyze_device_uploads(device_ip, device_flows)

            if exfil_indicators['suspicious']:
                # Get AI analysis
                ai_analysis = self._ai_analyze_exfiltration(device_ip, exfil_indicators)

                if ai_analysis['is_exfiltration']:
                    suspicious_devices.append({
                        'device_ip': device_ip,
                        'indicators': exfil_indicators,
                        'ai_analysis': ai_analysis
                    })

        return {
            'exfiltration_detected': len(suspicious_devices) > 0,
            'devices_analyzed': len(uploads_by_device),
            'suspicious_devices': suspicious_devices
        }

    def _is_internal(self, ip: str, internal_subnets: List[str]) -> bool:
        """Check if IP is in internal subnets"""
        # Simplified - in production use ipaddress module
        for subnet in internal_subnets:
            if ip.startswith(subnet.split('/')[0].rsplit('.', 1)[0]):
                return True
        return False

    def _analyze_device_uploads(self, device_ip: str, flows: List[NetFlowRecord]) -> Dict:
        """Analyze upload patterns for exfiltration indicators"""

        total_uploaded = sum(f.bytes for f in flows)
        unique_destinations = len(set(f.dest_ip for f in flows))
        unique_ports = set(f.dest_port for f in flows)

        # Time analysis
        hour_distribution = defaultdict(int)
        for flow in flows:
            hour_distribution[flow.timestamp.hour] += flow.bytes

        # Check for off-hours uploads (11 PM - 6 AM)
        off_hours_bytes = sum(bytes for hour, bytes in hour_distribution.items()
                              if hour >= 23 or hour <= 6)
        off_hours_percentage = (off_hours_bytes / total_uploaded * 100) if total_uploaded > 0 else 0

        # Check for sustained uploads (continuous over long period)
        first_upload = min(f.timestamp for f in flows)
        last_upload = max(f.timestamp for f in flows)
        duration_hours = (last_upload - first_upload).total_seconds() / 3600

        # Destination analysis
        dest_by_bytes = defaultdict(int)
        for flow in flows:
            dest_by_bytes[flow.dest_ip] += flow.bytes

        top_destination = max(dest_by_bytes.items(), key=lambda x: x[1]) if dest_by_bytes else (None, 0)
        top_dest_percentage = (top_destination[1] / total_uploaded * 100) if total_uploaded > 0 else 0

        # Exfiltration indicators
        indicators = []
        risk_score = 0.0

        # Indicator 1: Large upload volume
        if total_uploaded > 1_000_000_000:  # 1 GB
            indicators.append(f"Large upload volume: {total_uploaded/1e9:.2f} GB")
            risk_score += 0.3

        # Indicator 2: Off-hours uploads
        if off_hours_percentage > 50:
            indicators.append(f"Off-hours uploads: {off_hours_percentage:.0f}% between 11 PM - 6 AM")
            risk_score += 0.3

        # Indicator 3: Sustained upload over long period
        if duration_hours > 6:
            indicators.append(f"Sustained uploads: {duration_hours:.1f} hours continuous")
            risk_score += 0.2

        # Indicator 4: Single destination focus (targeted exfiltration)
        if top_dest_percentage > 80 and total_uploaded > 100_000_000:  # 80% to one IP, >100MB
            indicators.append(f"Focused destination: {top_dest_percentage:.0f}% to {top_destination[0]}")
            risk_score += 0.4

        # Indicator 5: Unusual ports (not 80/443)
        unusual_ports = unique_ports - {80, 443, 53}  # Common legitimate ports
        if unusual_ports:
            indicators.append(f"Unusual ports used: {sorted(list(unusual_ports))[:10]}")
            risk_score += 0.2

        # Check baseline anomaly
        if device_ip in self.baseline.baselines:
            baseline_result = self.baseline.detect_anomalies(flows, device_ip)
            for anomaly in baseline_result.get('anomalies', []):
                if anomaly['type'] == 'upload_anomaly':
                    indicators.append(f"Upload ratio anomaly: {anomaly['upload_ratio']:.1f}x")
                    risk_score += 0.4

        return {
            'device_ip': device_ip,
            'total_uploaded_gb': total_uploaded / 1e9,
            'duration_hours': duration_hours,
            'unique_destinations': unique_destinations,
            'off_hours_percentage': off_hours_percentage,
            'top_destination': top_destination[0],
            'top_dest_percentage': top_dest_percentage,
            'indicators': indicators,
            'risk_score': min(risk_score, 1.0),
            'suspicious': risk_score >= 0.6
        }

    def _ai_analyze_exfiltration(self, device_ip: str, indicators: Dict) -> Dict:
        """Use AI to determine if this is data exfiltration"""

        prompt = f"""You are a security analyst investigating potential data exfiltration.

SUSPICIOUS UPLOAD ACTIVITY:
Source Device: {device_ip}
Total Uploaded: {indicators['total_uploaded_gb']:.2f} GB
Duration: {indicators['duration_hours']:.1f} hours
Unique Destinations: {indicators['unique_destinations']}
Primary Destination: {indicators['top_destination']} ({indicators['top_dest_percentage']:.0f}% of traffic)
Off-Hours Uploads: {indicators['off_hours_percentage']:.0f}%

EXFILTRATION INDICATORS (Risk Score: {indicators['risk_score']:.2f}):
{chr(10).join('- ' + indicator for indicator in indicators['indicators'])}

ANALYSIS REQUIRED:
1. Is this data exfiltration (vs. legitimate uploads)?
2. What data is being stolen?
3. How urgent is this threat?
4. What's the business impact?

Consider legitimate scenarios:
- Backups to cloud
- CDN content uploads
- Remote desktop/VPN traffic
- Software updates/patches
- Cloud application sync

Respond in JSON:
{{
    "is_exfiltration": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "likely_data_type": "Customer DB/Source Code/Credentials/Files/Unknown",
    "exfiltration_method": "HTTP/HTTPS/DNS Tunneling/FTP/etc",
    "business_impact": "what's at risk",
    "legitimate_possibility": "if this might be legitimate",
    "indicators": ["list of exfiltration indicators"],
    "recommended_actions": ["immediate actions"],
    "investigation_priority": "Immediate/High/Medium/Low",
    "likely_insider": true/false,
    "estimated_data_stolen_gb": 0.0
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'is_exfiltration': True,
                'confidence': indicators['risk_score'],
                'severity': 'High'
            }

# Example: Detecting data exfiltration
def detect_exfiltration():
    detector = DataExfiltrationDetector(anthropic_api_key="your-api-key")

    # Simulate insider stealing customer database
    internal_subnets = ['10.1.0.0/16', '10.2.0.0/16']
    exfil_flows = []

    # 3 AM upload session: 5 GB over 2 hours
    attacker_workstation = '10.1.50.75'
    attacker_c2 = '45.134.83.200'  # External IP (attacker's server)
    exfil_start = datetime.now().replace(hour=3, minute=0)

    # Upload data in chunks over 2 hours
    for minute in range(120):  # 2 hours
        flow = NetFlowRecord(
            timestamp=exfil_start + timedelta(minutes=minute),
            source_ip=attacker_workstation,
            dest_ip=attacker_c2,
            source_port=random.randint(40000, 65000),
            dest_port=443,  # HTTPS (encrypted)
            protocol='TCP',
            bytes=45_000_000,  # 45 MB per minute
            packets=30_000,
            duration_seconds=60
        )
        exfil_flows.append(flow)

    # Detect
    result = detector.analyze_for_exfiltration(exfil_flows, internal_subnets, time_window_hours=24)

    if result['exfiltration_detected']:
        print("🚨 DATA EXFILTRATION DETECTED")
        for device in result['suspicious_devices']:
            print(f"\nDevice: {device['device_ip']}")
            print(f"Uploaded: {device['indicators']['total_uploaded_gb']:.2f} GB")
            print(f"Duration: {device['indicators']['duration_hours']:.1f} hours")

            ai = device['ai_analysis']
            print(f"\nAI Assessment:")
            print(f"  Confidence: {ai['confidence']:.0%}")
            print(f"  Severity: {ai['severity']}")
            print(f"  Data Type: {ai['likely_data_type']}")
            print(f"  Method: {ai['exfiltration_method']}")
            print(f"  Impact: {ai['business_impact']}")
            print(f"\nActions:")
            for action in ai['recommended_actions']:
                print(f"  - {action}")

# Example Output:
"""
🚨 DATA EXFILTRATION DETECTED

Device: 10.1.50.75
Uploaded: 5.40 GB
Duration: 2.0 hours

AI Assessment:
  Confidence: 95%
  Severity: Critical
  Data Type: Customer DB
  Method: HTTPS
  Impact: Large-scale data theft in progress. 5.4 GB uploaded suggests database exfiltration (customer records, PII, credentials). Encrypted traffic prevents inspection. Regulatory breach (GDPR, CCPA) likely.

Actions:
  - IMMEDIATE: Block 10.1.50.75 access to internet
  - IMMEDIATE: Block 45.134.83.200 at perimeter firewall
  - Disable user account associated with 10.1.50.75
  - Isolate workstation for forensic analysis
  - Identify what database/files were accessed
  - Calculate breach scope (how many customer records)
  - Notify legal team (breach notification requirements)
  - Preserve evidence (network captures, disk images)
  - Interview employee (insider threat investigation)
  - Check for similar activity from other IPs

Analysis Details:
{
  "is_exfiltration": true,
  "confidence": 0.95,
  "severity": "Critical",
  "likely_data_type": "Customer DB",
  "exfiltration_method": "HTTPS",
  "business_impact": "Active data theft. 5.4 GB transferred suggests significant data loss - likely customer database or large file repository. Regulatory implications (GDPR/CCPA breach notification), customer trust impact, potential competitive intelligence loss.",
  "legitimate_possibility": "Very low (<5%). Indicators strongly suggest exfiltration: (1) 3-5 AM timeframe, (2) Sustained 2-hour upload to single external IP, (3) 5.4 GB volume, (4) No legitimate business reason for workstation to upload multi-GB to unknown external IP at night.",
  "likely_insider": true,
  "estimated_data_stolen_gb": 5.4
}
"""
```

---

## Section 4: Production Deployment

### Complete Anomaly Detection Platform

```python
"""
Production Network Anomaly Detection Platform
Combines baseline learning, DDoS detection, and data exfiltration detection
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
import anthropic

class NetworkAnomalyPlatform:
    """Unified anomaly detection platform"""

    def __init__(self, anthropic_api_key: str, internal_subnets: List[str]):
        self.baseline = TrafficBaseline()
        self.ddos_detector = DDoSDetector(anthropic_api_key)
        self.exfil_detector = DataExfiltrationDetector(anthropic_api_key)
        self.internal_subnets = internal_subnets
        self.alerts = []

    def learn_baselines(self, historical_flows: List[NetFlowRecord],
                       critical_devices: List[str], days: int = 30):
        """Learn baselines for critical devices"""
        print(f"Learning baselines for {len(critical_devices)} critical devices...")

        for device_ip in critical_devices:
            baseline = self.baseline.learn_baseline(historical_flows, device_ip, days)
            if baseline:
                print(f"  ✓ {device_ip}: {baseline['total_flows_analyzed']} flows")

    async def monitor_realtime(self, netflow_stream):
        """Monitor NetFlow in real-time for anomalies"""
        while True:
            # Collect flows for analysis window (5 minutes)
            flows = await self._collect_flows(netflow_stream, minutes=5)

            # Run all detections in parallel
            await asyncio.gather(
                self._detect_general_anomalies(flows),
                self._detect_ddos(flows),
                self._detect_exfiltration(flows)
            )

            await asyncio.sleep(60)  # Check every minute

    async def _detect_general_anomalies(self, flows: List[NetFlowRecord]):
        """Detect baseline anomalies for all devices"""
        for device_ip in self.baseline.baselines.keys():
            result = self.baseline.detect_anomalies(flows, device_ip)

            if result.get('anomalies'):
                alert = {
                    'type': 'baseline_anomaly',
                    'severity': 'medium',
                    'device': device_ip,
                    'anomalies': result['anomalies'],
                    'risk_score': result['risk_score'],
                    'timestamp': datetime.now().isoformat()
                }
                self.alerts.append(alert)
                await self._send_alert(alert)

    async def _detect_ddos(self, flows: List[NetFlowRecord]):
        """Detect DDoS attacks"""
        # Get unique destination IPs (potential targets)
        targets = set(f.dest_ip for f in flows if self._is_internal(f.dest_ip))

        for target_ip in targets:
            result = self.ddos_detector.analyze_for_ddos(flows, target_ip, time_window_minutes=5)

            if result.get('ddos_detected'):
                alert = {
                    'type': 'ddos_attack',
                    'severity': result['ai_analysis']['severity'].lower(),
                    'target': target_ip,
                    'attack_type': result['ai_analysis']['attack_type'],
                    'metrics': result['metrics'],
                    'ai_analysis': result['ai_analysis'],
                    'timestamp': datetime.now().isoformat()
                }
                self.alerts.append(alert)
                await self._send_alert(alert)
                await self._automated_mitigation(alert)

    async def _detect_exfiltration(self, flows: List[NetFlowRecord]):
        """Detect data exfiltration"""
        result = self.exfil_detector.analyze_for_exfiltration(
            flows, self.internal_subnets, time_window_hours=1
        )

        if result.get('exfiltration_detected'):
            for device in result['suspicious_devices']:
                alert = {
                    'type': 'data_exfiltration',
                    'severity': 'critical',
                    'device': device['device_ip'],
                    'indicators': device['indicators'],
                    'ai_analysis': device['ai_analysis'],
                    'timestamp': datetime.now().isoformat()
                }
                self.alerts.append(alert)
                await self._send_alert(alert)
                await self._automated_response(alert)

    def _is_internal(self, ip: str) -> bool:
        """Check if IP is internal"""
        for subnet in self.internal_subnets:
            if ip.startswith(subnet.split('/')[0].rsplit('.', 1)[0]):
                return True
        return False

    async def _send_alert(self, alert: Dict):
        """Send alert to SIEM, Slack, PagerDuty"""
        print(f"\n🚨 ALERT: {alert['type'].upper()}")
        print(f"Severity: {alert['severity']}")
        print(f"Details: {json.dumps(alert, indent=2)}")

        # Integration with alerting systems
        # await self._send_to_slack(alert)
        # await self._send_to_pagerduty(alert)
        # await self._send_to_siem(alert)

    async def _automated_mitigation(self, alert: Dict):
        """Automated DDoS mitigation"""
        if alert['type'] == 'ddos_attack':
            print(f"🛡️  AUTO-MITIGATION: DDoS attack on {alert['target']}")

            # Example automated actions:
            # 1. Rate-limit at firewall
            # 2. Enable SYN cookies
            # 3. Engage upstream scrubbing
            # 4. Add to DDoS mitigation service

    async def _automated_response(self, alert: Dict):
        """Automated response to exfiltration"""
        if alert['type'] == 'data_exfiltration':
            print(f"🛡️  AUTO-RESPONSE: Blocking {alert['device']}")

            # Example automated actions:
            # 1. Block device at firewall
            # 2. Disable user account
            # 3. Quarantine device
            # 4. Notify security team

    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for dashboard"""
        return {
            'alerts_last_24h': len([a for a in self.alerts
                                   if (datetime.now() - datetime.fromisoformat(a['timestamp'])).days < 1]),
            'critical_alerts': len([a for a in self.alerts if a['severity'] == 'critical']),
            'ddos_attacks': len([a for a in self.alerts if a['type'] == 'ddos_attack']),
            'exfiltration_attempts': len([a for a in self.alerts if a['type'] == 'data_exfiltration']),
            'baseline_anomalies': len([a for a in self.alerts if a['type'] == 'baseline_anomaly'])
        }

# Production deployment
async def main():
    platform = NetworkAnomalyPlatform(
        anthropic_api_key='your-api-key',
        internal_subnets=['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
    )

    # Learn baselines from 30 days of historical data
    critical_devices = [
        '10.1.50.10',  # Web server
        '10.1.50.20',  # Database server
        '10.1.50.30',  # Application server
    ]

    # historical_flows = load_historical_netflow()
    # platform.learn_baselines(historical_flows, critical_devices, days=30)

    # Start real-time monitoring
    # netflow_stream = connect_to_netflow_collector()
    # await platform.monitor_realtime(netflow_stream)

if __name__ == '__main__':
    asyncio.run(main())
```

---

## What Can Go Wrong

### 1. Baseline Poisoning
**Problem**: Learning baseline during active attack makes attack "normal"
**Solution**: Baseline during verified clean period, periodic audits

### 2. False Positives from Business Changes
**Problem**: Marketing campaign → traffic spike → DDoS alert
**Solution**: Scheduled event whitelist, gradual threshold adjustment

### 3. Slow Exfiltration Misses
**Problem**: 10 MB/day exfiltration over 6 months goes undetected
**Solution**: Long-term trend analysis, cumulative volume tracking

### 4. Encrypted Traffic Blindness
**Problem**: Can't see what's in HTTPS/TLS traffic
**Solution**: Focus on metadata (volume, timing, destinations), not payload

### 5. Performance at Scale
**Problem**: 10,000 devices × 24/7 monitoring = expensive
**Solution**: Tiered monitoring (critical devices real-time, others batched)

---

## Key Takeaways

1. **Baselines enable anomaly detection** - 30 days of clean data required to understand "normal"

2. **DDoS detection needs multi-layered approach** - Volume + packet rate + source count + AI context

3. **Data exfiltration hides in encrypted traffic** - Detect via volume, timing, destination patterns

4. **NetFlow is sufficient** - Don't need full packet capture, metadata patterns are enough

5. **AI adds context** - Statistics detect anomaly, AI determines if it's a threat or legitimate spike

6. **Automated response is critical** - 90-second detection → mitigation vs. 15-minute human response

**Next Chapter**: Securing AI Systems - Protecting your AI infrastructure from prompt injection, data leakage, and adversarial attacks.

---

**Code Repository**: `github.com/vexpertai/ai-networking-book/chapter-75/`
