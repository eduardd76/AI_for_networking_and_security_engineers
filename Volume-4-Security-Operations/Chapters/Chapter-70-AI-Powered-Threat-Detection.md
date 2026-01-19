# Chapter 70: AI-Powered Threat Detection

## Learning Objectives

By the end of this chapter, you will:
- Detect lateral movement in your network using AI analysis of authentication logs
- Identify Command & Control (C2) beacons in encrypted network traffic
- Spot credential compromise through behavioral analysis
- Build a production threat detection system with real-time alerting
- Understand false positive management and alert tuning

**Prerequisites**: Volume 3 knowledge (monitoring, scaling, production patterns), understanding of network security fundamentals, access to security logs (auth logs, NetFlow, firewall logs)

**What You'll Build**: A complete AI-powered threat detection system that analyzes authentication patterns, network traffic, and user behavior to detect attacks in progress.

---

## The Problem: Traditional Security Tools Miss Modern Attacks

Your security stack costs $500K/year:
- Next-gen firewall ($80K)
- IDS/IPS ($60K)
- SIEM with correlation rules ($200K)
- EDR on endpoints ($150K)
- SOC analyst team ($500K in salaries)

Yet attacks still succeed:

**Real Incident 1**: Ransomware at Manufacturing Company
- Attacker compromised VPN account (weak password)
- Moved laterally to domain controller over 3 days
- Exfiltrated 50GB of data
- Deployed ransomware to 400 workstations
- **Total time undetected**: 72 hours
- **Cost**: $2.8M (ransom + downtime + recovery)

**Why Traditional Tools Failed**:
- Firewall: VPN traffic was "legitimate" (valid credentials)
- IDS: Lateral movement used standard admin tools (PsExec, RDP)
- SIEM: Alerts were lost in 50,000 daily events
- EDR: Only alerted when ransomware executed (too late)

**What AI Would Have Caught**:
- VPN login from unusual country (credential compromise)
- Admin account accessing servers it never touched before (lateral movement)
- Large file transfers to external IP (data exfiltration)
- Unusual PowerShell execution patterns (ransomware prep)

**Detection time with AI**: 12 minutes instead of 72 hours.

This chapter shows you how to build that AI threat detection system.

---

## Section 1: Lateral Movement Detection

### What Is Lateral Movement?

After initial compromise, attackers move through your network:
1. Compromise edge device (VPN, web server, workstation)
2. Steal credentials (keylogger, mimikatz, password reuse)
3. Move to more valuable targets (file servers, databases, domain controllers)
4. Establish persistence and exfiltrate data

**Why It's Hard to Detect**:
- Uses legitimate admin tools (RDP, SSH, PsExec, WMI)
- Valid credentials (stolen, not forged)
- Blends with normal admin activity
- Happens slowly (hours or days to avoid detection)

**Traditional Detection**: Correlation rules like "If admin logs into >10 servers in 5 minutes"
- **Problem**: Attackers move slowly (2-3 servers per hour)
- **Problem**: Rules are rigid (miss variations)
- **Problem**: High false positive rate (legitimate admin work triggers alerts)

**AI Detection**: Learns normal behavior patterns, detects deviations
- What servers does each admin account normally access?
- What times do they typically work?
- What's the typical access pattern?
- What's unusual today?

### Building a Lateral Movement Detector

Let's build an AI system that analyzes authentication logs and identifies lateral movement.

#### Architecture

```
Authentication Logs (SSH, RDP, Windows Event Logs)
        ↓
Log Collector (Splunk, Elastic, Syslog)
        ↓
AI Analyzer (every 5 minutes)
        ↓
Threat Detection Pipeline:
1. Extract authentication events
2. Build user behavior profile
3. Detect unusual access patterns
4. LLM analysis for context
5. Generate alerts

        ↓
Alert (Slack, PagerDuty, SIEM)
```

#### Step 1: Authentication Log Parser

```python
"""
Lateral Movement Detection System
Analyzes authentication logs to detect unusual access patterns
"""
import re
from datetime import datetime, timedelta
from typing import List, Dict, Set
from collections import defaultdict
import anthropic
from dataclasses import dataclass
import json

@dataclass
class AuthEvent:
    """Represents a single authentication event"""
    timestamp: datetime
    user: str
    source_ip: str
    destination_host: str
    auth_type: str  # ssh, rdp, smb, kerberos
    success: bool
    source_host: str = None  # if we can resolve IP to hostname

class AuthLogParser:
    """Parse various authentication log formats"""

    def parse_ssh_log(self, log_line: str) -> AuthEvent:
        """Parse OpenSSH authentication logs"""
        # Example: Jan 18 14:23:45 server1 sshd[12345]: Accepted password for admin from 10.1.1.50 port 52342 ssh2
        pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+sshd.*?(Accepted|Failed)\s+\w+\s+for\s+(\S+)\s+from\s+([\d.]+)'
        match = re.search(pattern, log_line)

        if match:
            timestamp_str, dest_host, result, user, source_ip = match.groups()
            timestamp = datetime.strptime(
                f"{datetime.now().year} {timestamp_str}",
                "%Y %b %d %H:%M:%S"
            )

            return AuthEvent(
                timestamp=timestamp,
                user=user,
                source_ip=source_ip,
                destination_host=dest_host,
                auth_type='ssh',
                success=(result == 'Accepted')
            )
        return None

    def parse_windows_event(self, event: Dict) -> AuthEvent:
        """Parse Windows Event Log 4624 (successful logon)"""
        return AuthEvent(
            timestamp=datetime.fromisoformat(event['timestamp']),
            user=event['TargetUserName'],
            source_ip=event['IpAddress'],
            destination_host=event['WorkstationName'],
            auth_type=event['LogonType'],  # 3=network, 10=RDP
            success=True
        )

    def parse_logs(self, log_source: str, log_type: str) -> List[AuthEvent]:
        """Parse logs from file or SIEM"""
        events = []

        if log_type == 'ssh':
            with open(log_source, 'r') as f:
                for line in f:
                    event = self.parse_ssh_log(line)
                    if event:
                        events.append(event)

        return events
```

#### Step 2: User Behavior Baseline

```python
class UserBehaviorProfile:
    """Build baseline of normal user behavior"""

    def __init__(self, username: str):
        self.username = username
        self.typical_hosts: Set[str] = set()
        self.typical_hours: Set[int] = set()
        self.typical_source_ips: Set[str] = set()
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.events_analyzed = 0

    def learn_from_events(self, events: List[AuthEvent]):
        """Build baseline from historical authentication events"""
        for event in events:
            if event.user == self.username and event.success:
                self.typical_hosts.add(event.destination_host)
                self.typical_hours.add(event.timestamp.hour)
                self.typical_source_ips.add(event.source_ip)
                self.access_frequency[event.destination_host] += 1
                self.events_analyzed += 1

    def is_unusual_host(self, host: str) -> bool:
        """Has user never accessed this host before?"""
        return host not in self.typical_hosts

    def is_unusual_time(self, hour: int) -> bool:
        """Does user typically work at this hour?"""
        # Allow 2-hour buffer
        for typical_hour in self.typical_hours:
            if abs(hour - typical_hour) <= 2:
                return False
        return True

    def is_unusual_source(self, source_ip: str) -> bool:
        """Has user logged in from this IP before?"""
        return source_ip not in self.typical_source_ips

    def get_anomaly_score(self, event: AuthEvent) -> float:
        """Calculate how unusual this authentication is (0-1)"""
        score = 0.0

        # New host (weight: 0.4)
        if self.is_unusual_host(event.destination_host):
            score += 0.4

        # Unusual time (weight: 0.2)
        if self.is_unusual_time(event.timestamp.hour):
            score += 0.2

        # New source IP (weight: 0.3)
        if self.is_unusual_source(event.source_ip):
            score += 0.3

        # Rapid access to multiple hosts (weight: 0.1)
        # Check last 5 minutes
        recent_window = event.timestamp - timedelta(minutes=5)
        # Would need to track recent events - simplified here

        return min(score, 1.0)

class BehaviorBaseline:
    """Manages behavior profiles for all users"""

    def __init__(self):
        self.profiles: Dict[str, UserBehaviorProfile] = {}

    def build_baseline(self, historical_events: List[AuthEvent], days: int = 30):
        """Build baseline from N days of historical authentication logs"""
        print(f"Building baseline from {len(historical_events)} historical events...")

        for event in historical_events:
            if event.user not in self.profiles:
                self.profiles[event.user] = UserBehaviorProfile(event.user)

            self.profiles[event.user].learn_from_events([event])

        print(f"Baseline built for {len(self.profiles)} users")

    def get_profile(self, username: str) -> UserBehaviorProfile:
        """Get or create user profile"""
        if username not in self.profiles:
            self.profiles[username] = UserBehaviorProfile(username)
        return self.profiles[username]
```

#### Step 3: Lateral Movement Detector

```python
class LateralMovementDetector:
    """Detect lateral movement using AI analysis"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.baseline = BehaviorBaseline()
        self.alert_threshold = 0.6  # Anomaly score threshold

    def analyze_authentication_event(self, event: AuthEvent) -> Dict:
        """Analyze single authentication event for lateral movement indicators"""
        profile = self.baseline.get_profile(event.user)
        anomaly_score = profile.get_anomaly_score(event)

        if anomaly_score < self.alert_threshold:
            return {'suspicious': False, 'anomaly_score': anomaly_score}

        # High anomaly score - get LLM analysis
        context = self._build_context(event, profile)
        threat_assessment = self._llm_analyze(event, context, anomaly_score)

        return {
            'suspicious': True,
            'anomaly_score': anomaly_score,
            'threat_assessment': threat_assessment,
            'event': event,
            'context': context
        }

    def _build_context(self, event: AuthEvent, profile: UserBehaviorProfile) -> Dict:
        """Build context about the suspicious authentication"""
        return {
            'user': event.user,
            'destination_host': event.destination_host,
            'source_ip': event.source_ip,
            'timestamp': event.timestamp.isoformat(),
            'typical_hosts': list(profile.typical_hosts)[:10],  # Top 10
            'typical_hours': sorted(list(profile.typical_hours)),
            'has_accessed_before': event.destination_host in profile.typical_hosts,
            'typical_source_ips': list(profile.typical_source_ips),
            'first_time_from_this_ip': event.source_ip not in profile.typical_source_ips
        }

    def _llm_analyze(self, event: AuthEvent, context: Dict, anomaly_score: float) -> Dict:
        """Use LLM to analyze if this is likely lateral movement"""

        prompt = f"""You are a network security analyst detecting lateral movement attacks.

SUSPICIOUS AUTHENTICATION EVENT:
User: {event.user}
Logged into: {event.destination_host}
From IP: {event.source_ip}
Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Hour: {event.timestamp.hour})
Auth Type: {event.auth_type}

USER'S NORMAL BEHAVIOR:
- Typically accesses: {', '.join(context['typical_hosts']) if context['typical_hosts'] else 'No history'}
- Has accessed {event.destination_host} before: {context['has_accessed_before']}
- Typical working hours: {context['typical_hours']}
- Typical source IPs: {', '.join(context['typical_source_ips']) if context['typical_source_ips'] else 'No history'}
- Logging in from new IP: {context['first_time_from_this_ip']}

ANOMALY SCORE: {anomaly_score:.2f} (threshold: {self.alert_threshold})

ANALYSIS REQUIRED:
1. Is this likely lateral movement (attacker using stolen credentials)?
2. What's suspicious about this authentication?
3. What should we check next?
4. Threat severity (Critical/High/Medium/Low)
5. Recommended response actions

Provide analysis in JSON format:
{{
    "is_lateral_movement": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "suspicious_indicators": ["list", "of", "indicators"],
    "explanation": "Why this is/isn't suspicious",
    "recommended_actions": ["action1", "action2"],
    "investigation_steps": ["step1", "step2"]
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON from response
            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'is_lateral_movement': True,  # Fail secure - assume threat
                'confidence': 0.5,
                'severity': 'High',
                'explanation': f'LLM analysis failed, manual review required. Anomaly score: {anomaly_score}'
            }

    def monitor_realtime(self, log_stream):
        """Monitor authentication logs in real-time"""
        parser = AuthLogParser()

        for log_entry in log_stream:
            # Parse log entry
            event = parser.parse_ssh_log(log_entry)  # Or appropriate parser

            if event and event.success:
                # Analyze for lateral movement
                result = self.analyze_authentication_event(event)

                if result['suspicious']:
                    self._generate_alert(result)

    def _generate_alert(self, detection: Dict):
        """Generate security alert"""
        event = detection['event']
        assessment = detection['threat_assessment']

        alert = {
            'alert_type': 'Lateral Movement Detected',
            'severity': assessment.get('severity', 'High'),
            'confidence': assessment.get('confidence', 0.8),
            'timestamp': datetime.now().isoformat(),
            'user': event.user,
            'destination': event.destination_host,
            'source_ip': event.source_ip,
            'anomaly_score': detection['anomaly_score'],
            'indicators': assessment.get('suspicious_indicators', []),
            'explanation': assessment.get('explanation', ''),
            'recommended_actions': assessment.get('recommended_actions', []),
            'investigation_steps': assessment.get('investigation_steps', [])
        }

        # Send to SIEM, Slack, PagerDuty, etc.
        self._send_alert(alert)
        return alert

    def _send_alert(self, alert: Dict):
        """Send alert to security team"""
        # Integration with alerting systems
        print(f"🚨 SECURITY ALERT: {alert['alert_type']}")
        print(f"Severity: {alert['severity']} | Confidence: {alert['confidence']:.0%}")
        print(f"User '{alert['user']}' accessed {alert['destination']} from {alert['source_ip']}")
        print(f"Explanation: {alert['explanation']}")
        print(f"Recommended Actions:")
        for action in alert['recommended_actions']:
            print(f"  - {action}")
```

#### Step 4: Running the Detector

```python
# Example usage
def main():
    # Initialize detector
    detector = LateralMovementDetector(anthropic_api_key="your-api-key")

    # Build baseline from 30 days of historical logs
    parser = AuthLogParser()
    historical_events = parser.parse_logs('/var/log/auth.log.30days', 'ssh')
    detector.baseline.build_baseline(historical_events, days=30)

    # Analyze recent suspicious authentication
    suspicious_auth = AuthEvent(
        timestamp=datetime.now(),
        user='john.admin',
        source_ip='185.220.101.50',  # Tor exit node
        destination_host='dc01.corp.local',  # Domain controller
        auth_type='ssh',
        success=True
    )

    result = detector.analyze_authentication_event(suspicious_auth)

    if result['suspicious']:
        print("Lateral movement detected!")
        print(json.dumps(result['threat_assessment'], indent=2))

if __name__ == '__main__':
    main()
```

**Example Output**:
```json
{
  "alert_type": "Lateral Movement Detected",
  "severity": "Critical",
  "confidence": 0.92,
  "user": "john.admin",
  "destination": "dc01.corp.local",
  "source_ip": "185.220.101.50",
  "anomaly_score": 0.95,
  "indicators": [
    "First time accessing domain controller",
    "Login from Tor exit node (anonymous proxy)",
    "Access at unusual hour (3:24 AM, typical hours: 9-17)",
    "User typically accesses web servers, not AD infrastructure"
  ],
  "explanation": "This authentication shows strong indicators of lateral movement using compromised credentials. The user has never accessed the domain controller before, is logging in from an anonymizing proxy (Tor), and at an unusual hour. This pattern is consistent with an attacker who compromised the user's credentials and is now pivoting to high-value targets.",
  "recommended_actions": [
    "Immediately disable john.admin account",
    "Isolate dc01.corp.local from network",
    "Check dc01 for persistence mechanisms",
    "Review john.admin's recent activities",
    "Force password reset for all admin accounts",
    "Check for data exfiltration from DC"
  ],
  "investigation_steps": [
    "Check when john.admin's credentials were last used legitimately",
    "Review VPN logs for this source IP",
    "Check EDR on dc01 for malicious activity",
    "Review file access logs on domain controller",
    "Check for new user accounts or group modifications"
  ]
}
```

---

## Section 2: Command & Control (C2) Detection

### The C2 Problem

After compromising a system, malware needs to communicate with the attacker's command server:
- Receive commands
- Exfiltrate stolen data
- Download additional tools

**Modern C2 Challenges**:
- Uses HTTPS (encrypted, looks like normal web traffic)
- DNS tunneling (data hidden in DNS queries)
- Uses CDNs (hard to block without breaking legitimate services)
- Low-and-slow (infrequent connections to avoid detection)
- Domain generation algorithms (thousands of backup domains)

**Traditional Detection Fails**:
- Firewalls can't inspect encrypted traffic
- Signature-based IDS misses new malware families
- IP blocklists are ineffective (CDNs, fast-flux DNS)

**AI Detection**: Pattern analysis that doesn't require decryption
- Periodic beaconing patterns
- Unusual DNS query patterns
- Traffic volume anomalies
- Frequency analysis

### Building a C2 Beacon Detector

```python
"""
C2 Beacon Detection using NetFlow analysis and AI
Detects periodic communication patterns indicative of malware beaconing
"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Tuple
import numpy as np
from scipy import stats
import anthropic

@dataclass
class NetFlowRecord:
    """Network flow record"""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    dest_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    duration: float

class C2BeaconDetector:
    """Detect C2 beacons by analyzing network flow patterns"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.beacon_threshold = 0.85  # Periodicity confidence threshold

    def analyze_host_traffic(self, host_ip: str, flows: List[NetFlowRecord],
                            time_window_hours: int = 24) -> Dict:
        """Analyze all traffic from a host for C2 beacons"""

        # Group flows by destination
        dest_flows = defaultdict(list)
        for flow in flows:
            if flow.source_ip == host_ip:
                dest_flows[flow.dest_ip].append(flow)

        # Analyze each destination for periodic beaconing
        suspicious_destinations = []

        for dest_ip, flows_to_dest in dest_flows.items():
            if len(flows_to_dest) < 10:  # Need enough samples
                continue

            periodicity = self._calculate_periodicity(flows_to_dest)

            if periodicity['is_periodic'] and periodicity['confidence'] > self.beacon_threshold:
                # This looks like beaconing - get AI analysis
                ai_analysis = self._analyze_potential_c2(host_ip, dest_ip, flows_to_dest, periodicity)

                if ai_analysis['is_c2']:
                    suspicious_destinations.append({
                        'destination_ip': dest_ip,
                        'periodicity': periodicity,
                        'ai_analysis': ai_analysis,
                        'flows': flows_to_dest
                    })

        return {
            'host_ip': host_ip,
            'analysis_window_hours': time_window_hours,
            'suspicious_destinations': suspicious_destinations,
            'threat_detected': len(suspicious_destinations) > 0
        }

    def _calculate_periodicity(self, flows: List[NetFlowRecord]) -> Dict:
        """Calculate if flows show periodic beaconing pattern"""
        if len(flows) < 10:
            return {'is_periodic': False, 'confidence': 0.0}

        # Extract timestamps and calculate intervals
        timestamps = [flow.timestamp for flow in sorted(flows, key=lambda f: f.timestamp)]
        intervals = []

        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(delta)

        if not intervals:
            return {'is_periodic': False, 'confidence': 0.0}

        # Statistical analysis of intervals
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        coefficient_of_variation = std_interval / mean_interval if mean_interval > 0 else float('inf')

        # Lower CV = more periodic
        # Beacon typically has CV < 0.3
        confidence = max(0, 1 - coefficient_of_variation)
        is_periodic = coefficient_of_variation < 0.3

        return {
            'is_periodic': is_periodic,
            'confidence': confidence,
            'mean_interval_seconds': mean_interval,
            'std_dev_seconds': std_interval,
            'coefficient_of_variation': coefficient_of_variation,
            'connection_count': len(flows),
            'intervals': intervals
        }

    def _analyze_potential_c2(self, source_ip: str, dest_ip: str,
                              flows: List[NetFlowRecord], periodicity: Dict) -> Dict:
        """Use AI to analyze if periodic traffic is actually C2"""

        # Calculate traffic characteristics
        total_bytes_sent = sum(f.bytes_sent for f in flows)
        total_bytes_received = sum(f.bytes_received for f in flows)
        avg_bytes_sent = total_bytes_sent / len(flows)
        avg_bytes_received = total_bytes_received / len(flows)

        ports_used = set(f.dest_port for f in flows)
        protocols_used = set(f.protocol for f in flows)

        # Time span
        first_seen = min(f.timestamp for f in flows)
        last_seen = max(f.timestamp for f in flows)
        duration_hours = (last_seen - first_seen).total_seconds() / 3600

        prompt = f"""You are a network security analyst detecting Command & Control (C2) malware beacons.

PERIODIC TRAFFIC DETECTED:
Source IP: {source_ip}
Destination IP: {dest_ip}
Connection Count: {len(flows)}
Time Span: {duration_hours:.1f} hours

PERIODICITY ANALYSIS:
- Mean interval: {periodicity['mean_interval_seconds']:.1f} seconds ({periodicity['mean_interval_seconds']/60:.1f} minutes)
- Standard deviation: {periodicity['std_dev_seconds']:.1f} seconds
- Coefficient of Variation: {periodicity['coefficient_of_variation']:.3f}
- Periodicity Confidence: {periodicity['confidence']:.2%}

TRAFFIC CHARACTERISTICS:
- Destination Ports: {', '.join(map(str, ports_used))}
- Protocols: {', '.join(protocols_used)}
- Average bytes sent per connection: {avg_bytes_sent:.0f}
- Average bytes received per connection: {avg_bytes_received:.0f}
- Total data sent: {total_bytes_sent / 1024:.1f} KB
- Total data received: {total_bytes_received / 1024:.1f} KB

C2 BEACON INDICATORS:
✓ Highly periodic connections (CV < 0.3)
✓ Consistent intervals (every {periodicity['mean_interval_seconds']/60:.1f} minutes)
? Small data transfers (typical for beacons)
? Long duration (persistent connection over {duration_hours:.1f} hours)

ANALYSIS REQUIRED:
1. Is this likely C2 malware beaconing (vs. legitimate periodic traffic)?
2. What malware families use this beacon pattern?
3. What data is being exfiltrated (if any)?
4. Threat severity and recommended actions

Respond in JSON format:
{{
    "is_c2": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "likely_malware_family": "name or unknown",
    "explanation": "detailed explanation",
    "legitimate_service_possible": "service name if applicable",
    "indicators": ["list of suspicious indicators"],
    "recommended_actions": ["immediate actions to take"]
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
                'is_c2': True,  # Fail secure
                'confidence': 0.7,
                'severity': 'High',
                'explanation': f'AI analysis failed. Periodic traffic detected with CV={periodicity["coefficient_of_variation"]:.3f}'
            }

    def generate_alert(self, detection: Dict):
        """Generate C2 detection alert"""
        for suspicious in detection['suspicious_destinations']:
            dest_ip = suspicious['destination_ip']
            ai_analysis = suspicious['ai_analysis']
            periodicity = suspicious['periodicity']

            alert = {
                'alert_type': 'C2 Beacon Detected',
                'severity': ai_analysis.get('severity', 'Critical'),
                'confidence': ai_analysis.get('confidence', 0.85),
                'timestamp': datetime.now().isoformat(),
                'source_ip': detection['host_ip'],
                'destination_ip': dest_ip,
                'beacon_interval_minutes': periodicity['mean_interval_seconds'] / 60,
                'periodicity_confidence': periodicity['confidence'],
                'connection_count': periodicity['connection_count'],
                'likely_malware': ai_analysis.get('likely_malware_family', 'Unknown'),
                'explanation': ai_analysis.get('explanation', ''),
                'indicators': ai_analysis.get('indicators', []),
                'recommended_actions': ai_analysis.get('recommended_actions', [])
            }

            self._send_alert(alert)
            return alert

# Example usage
def detect_c2_beacons():
    detector = C2BeaconDetector(anthropic_api_key="your-api-key")

    # Simulate NetFlow data showing periodic beaconing
    flows = []
    base_time = datetime.now() - timedelta(hours=24)

    # Simulate C2 beacon every 5 minutes for 24 hours
    for i in range(288):  # 24 hours * 60 min / 5 min
        flow = NetFlowRecord(
            timestamp=base_time + timedelta(minutes=i*5),
            source_ip='10.1.50.75',  # Internal workstation
            dest_ip='45.134.83.12',  # Suspicious external IP
            dest_port=443,
            protocol='TCP',
            bytes_sent=1200,  # Small beacon
            bytes_received=850,  # Small response
            duration=2.3
        )
        flows.append(flow)

    # Analyze
    result = detector.analyze_host_traffic('10.1.50.75', flows, time_window_hours=24)

    if result['threat_detected']:
        alert = detector.generate_alert(result)
        print(json.dumps(alert, indent=2))

# Example Output:
"""
{
  "alert_type": "C2 Beacon Detected",
  "severity": "Critical",
  "confidence": 0.94,
  "source_ip": "10.1.50.75",
  "destination_ip": "45.134.83.12",
  "beacon_interval_minutes": 5.0,
  "periodicity_confidence": 0.96,
  "connection_count": 288,
  "likely_malware": "Cobalt Strike",
  "explanation": "This traffic pattern is highly indicative of C2 beaconing, specifically matching Cobalt Strike's default beacon behavior. The extremely regular 5-minute intervals (CV=0.02) over 24 hours, combined with small consistent data transfers (~1.2KB sent, ~850 bytes received), and use of HTTPS port 443 match known C2 frameworks. The destination IP is not associated with any legitimate service.",
  "indicators": [
    "Highly periodic connections (CV=0.02, nearly perfect periodicity)",
    "Default Cobalt Strike beacon interval (5 minutes)",
    "Small consistent payload sizes (typical for beacons)",
    "24-hour continuous operation without interruption",
    "Destination IP has no legitimate service association",
    "HTTPS traffic but no SNI/certificate validation possible"
  ],
  "recommended_actions": [
    "IMMEDIATE: Isolate 10.1.50.75 from network",
    "Run EDR scan on host for Cobalt Strike artifacts",
    "Capture memory dump for forensic analysis",
    "Block 45.134.83.12 at firewall (add to threat feed)",
    "Check for lateral movement from this host",
    "Review authentication logs for compromised accounts",
    "Search for other hosts beaconing to same destination",
    "Notify incident response team - active breach in progress"
  ]
}
"""
```

---

## Section 3: Credential Compromise Detection

### The Credential Problem

80% of breaches involve compromised credentials (Verizon DBIR 2025). Once attackers have valid usernames and passwords, they look like legitimate users.

**How Credentials Get Compromised**:
- Phishing emails
- Password reuse (LinkedIn breach → corporate access)
- Keyloggers on compromised workstations
- Brute force attacks
- Insider threats

**Detection Challenges**:
- Valid credentials = no failed login alerts
- VPN access looks legitimate
- Hard to distinguish from real user

**AI Detection Signals**:
- Impossible travel (US → China in 30 minutes)
- Unusual login times (admin account at 3 AM)
- Geo-location changes (always Chicago, suddenly Moscow)
- Device fingerprint changes (always MacBook, suddenly Linux)
- Behavioral changes (always accesses HR app, suddenly database queries)

### Building Credential Compromise Detector

```python
"""
Credential Compromise Detection
Identifies stolen credentials being used by attackers
"""
import anthropic
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from geopy.distance import geodesic  # pip install geopy

@dataclass
class LoginEvent:
    """Represents a login attempt"""
    timestamp: datetime
    username: str
    source_ip: str
    geo_location: Tuple[float, float]  # (latitude, longitude)
    city: str
    country: str
    device_type: str  # Windows, Linux, MacOS, iOS, Android
    user_agent: str
    success: bool
    mfa_used: bool = False

class CredentialCompromiseDetector:
    """Detect compromised credentials using behavioral analysis"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.user_history: Dict[str, List[LoginEvent]] = {}

    def analyze_login(self, login: LoginEvent) -> Dict:
        """Analyze login event for compromise indicators"""

        # Get user's historical logins
        history = self.user_history.get(login.username, [])

        if not history:
            # First login we've seen - can't baseline yet
            self._add_to_history(login)
            return {'suspicious': False, 'reason': 'First observed login'}

        # Check for impossible travel
        impossible_travel = self._check_impossible_travel(login, history)

        # Check for unusual location
        unusual_location = self._check_unusual_location(login, history)

        # Check for unusual time
        unusual_time = self._check_unusual_time(login, history)

        # Check for device change
        device_change = self._check_device_change(login, history)

        # Calculate risk score
        risk_factors = []
        risk_score = 0.0

        if impossible_travel['detected']:
            risk_factors.append(f"Impossible travel: {impossible_travel['explanation']}")
            risk_score += 0.4

        if unusual_location['detected']:
            risk_factors.append(f"Unusual location: {unusual_location['explanation']}")
            risk_score += 0.3

        if unusual_time['detected']:
            risk_factors.append(f"Unusual time: {unusual_time['explanation']}")
            risk_score += 0.15

        if device_change['detected']:
            risk_factors.append(f"Device change: {device_change['explanation']}")
            risk_score += 0.15

        # Add to history
        self._add_to_history(login)

        if risk_score >= 0.5:  # Threshold for AI analysis
            ai_analysis = self._ai_analyze_compromise(login, history, risk_factors, risk_score)
            return {
                'suspicious': True,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'ai_analysis': ai_analysis
            }

        return {
            'suspicious': False,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }

    def _check_impossible_travel(self, login: LoginEvent, history: List[LoginEvent]) -> Dict:
        """Check if user traveled impossibly fast between locations"""
        # Get most recent successful login
        recent_logins = [l for l in history if l.success]
        if not recent_logins:
            return {'detected': False}

        last_login = recent_logins[-1]

        # Calculate distance and time
        distance_km = geodesic(last_login.geo_location, login.geo_location).kilometers
        time_diff_hours = (login.timestamp - last_login.timestamp).total_seconds() / 3600

        if time_diff_hours == 0:
            return {'detected': False}

        # Calculate required speed
        speed_kmh = distance_km / time_diff_hours

        # Max realistic speed (including flight time + transit): 900 km/h
        if speed_kmh > 900:
            return {
                'detected': True,
                'distance_km': distance_km,
                'time_hours': time_diff_hours,
                'required_speed_kmh': speed_kmh,
                'explanation': f"Login from {login.city}, {login.country} just {time_diff_hours:.1f} hours after login from {last_login.city}, {last_login.country}. Would require traveling {speed_kmh:.0f} km/h (impossible)."
            }

        return {'detected': False}

    def _check_unusual_location(self, login: LoginEvent, history: List[LoginEvent]) -> Dict:
        """Check if login is from unusual country/city"""
        # Get typical countries
        typical_countries = set(l.country for l in history if l.success)

        if login.country not in typical_countries and len(typical_countries) > 0:
            return {
                'detected': True,
                'explanation': f"First login from {login.country}. User typically logs in from: {', '.join(typical_countries)}"
            }

        return {'detected': False}

    def _check_unusual_time(self, login: LoginEvent, history: List[LoginEvent]) -> Dict:
        """Check if login time is unusual for this user"""
        typical_hours = set(l.timestamp.hour for l in history if l.success)

        if not typical_hours:
            return {'detected': False}

        # Check if current hour is within 2 hours of any typical hour
        current_hour = login.timestamp.hour
        for typical_hour in typical_hours:
            if abs(current_hour - typical_hour) <= 2:
                return {'detected': False}

        return {
            'detected': True,
            'explanation': f"Login at {login.timestamp.strftime('%H:%M')} ({current_hour}:00 hour). User typically logs in during hours: {sorted(typical_hours)}"
        }

    def _check_device_change(self, login: LoginEvent, history: List[LoginEvent]) -> Dict:
        """Check if device type changed"""
        typical_devices = set(l.device_type for l in history if l.success)

        if login.device_type not in typical_devices and len(typical_devices) > 0:
            return {
                'detected': True,
                'explanation': f"Login from {login.device_type}. User typically uses: {', '.join(typical_devices)}"
            }

        return {'detected': False}

    def _ai_analyze_compromise(self, login: LoginEvent, history: List[LoginEvent],
                               risk_factors: List[str], risk_score: float) -> Dict:
        """Use AI to analyze if credentials are compromised"""

        # Build context
        recent_history = history[-10:] if len(history) > 10 else history
        history_summary = "\n".join([
            f"- {l.timestamp.strftime('%Y-%m-%d %H:%M')} from {l.city}, {l.country} ({l.device_type})"
            for l in recent_history
        ])

        prompt = f"""You are a security analyst detecting compromised credentials.

SUSPICIOUS LOGIN:
Username: {login.username}
Timestamp: {login.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Location: {login.city}, {login.country}
IP Address: {login.source_ip}
Device: {login.device_type}
User Agent: {login.user_agent}
MFA Used: {login.mfa_used}

RISK FACTORS DETECTED (Risk Score: {risk_score:.2f}):
{chr(10).join('- ' + factor for factor in risk_factors)}

USER'S RECENT LOGIN HISTORY:
{history_summary}

ANALYSIS REQUIRED:
1. Are these credentials compromised (stolen/phished)?
2. Is this the legitimate user or an attacker?
3. What attack scenario is most likely?
4. Threat severity and immediate actions needed

Respond in JSON format:
{{
    "credentials_compromised": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "likely_scenario": "description of attack",
    "explanation": "detailed reasoning",
    "indicators": ["list of compromise indicators"],
    "recommended_actions": ["immediate actions"],
    "false_positive_possibility": "explanation if might be legitimate"
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'credentials_compromised': True,
                'confidence': risk_score,
                'severity': 'High',
                'explanation': f'AI analysis failed. Risk score: {risk_score:.2f}'
            }

    def _add_to_history(self, login: LoginEvent):
        """Add login to user's history"""
        if login.username not in self.user_history:
            self.user_history[login.username] = []
        self.user_history[login.username].append(login)

        # Keep last 100 logins per user
        if len(self.user_history[login.username]) > 100:
            self.user_history[login.username] = self.user_history[login.username][-100:]

# Example usage
def detect_credential_compromise():
    detector = CredentialCompromiseDetector(anthropic_api_key="your-api-key")

    # Simulate normal user behavior (Chicago, business hours, Windows)
    for i in range(20):
        normal_login = LoginEvent(
            timestamp=datetime.now() - timedelta(days=20-i, hours=9),
            username='alice.smith',
            source_ip='8.8.8.8',
            geo_location=(41.8781, -87.6298),  # Chicago
            city='Chicago',
            country='USA',
            device_type='Windows',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            success=True,
            mfa_used=True
        )
        detector.analyze_login(normal_login)

    # Suspicious login from different country, different device, no MFA
    suspicious_login = LoginEvent(
        timestamp=datetime.now(),
        username='alice.smith',
        source_ip='91.241.19.67',
        geo_location=(55.7558, 37.6173),  # Moscow
        city='Moscow',
        country='Russia',
        device_type='Linux',
        user_agent='curl/7.68.0',  # Command-line tool
        success=True,
        mfa_used=False  # MFA bypassed somehow
    )

    result = detector.analyze_login(suspicious_login)

    if result['suspicious']:
        print("🚨 COMPROMISED CREDENTIALS DETECTED")
        print(json.dumps(result, indent=2))

# Example Output:
"""
{
  "suspicious": true,
  "risk_score": 0.85,
  "risk_factors": [
    "Unusual location: First login from Russia. User typically logs in from: USA",
    "Device change: Login from Linux. User typically uses: Windows",
    "Unusual time: Login at 03:45 (3:00 hour). User typically logs in during hours: [9, 10, 11, 14, 15]"
  ],
  "ai_analysis": {
    "credentials_compromised": true,
    "confidence": 0.93,
    "severity": "Critical",
    "likely_scenario": "Phished credentials used by attacker. The change from Windows to Linux (specifically curl user-agent) suggests automated tooling rather than interactive user. Moscow location combined with no MFA and off-hours access strongly indicates compromise.",
    "explanation": "This login shows multiple strong indicators of credential compromise: (1) Geographic impossibility - user can't be in Moscow 12 hours after being in Chicago, (2) Device change from Windows desktop to Linux command-line tool suggests attacker using scripts, (3) No MFA when user always used MFA previously indicates MFA was bypassed or credentials stolen include session tokens, (4) 3:45 AM local time is inconsistent with user's work hours (9 AM - 5 PM).",
    "indicators": [
      "Login from Russia (user never logged in from outside USA)",
      "curl user-agent (command-line tool, not browser)",
      "MFA not used (always used previously)",
      "3:45 AM login (always 9 AM - 5 PM previously)",
      "Linux device (always Windows previously)"
    ],
    "recommended_actions": [
      "IMMEDIATE: Disable alice.smith account",
      "Force password reset after account review",
      "Review all recent actions by this account",
      "Check for data exfiltration",
      "Enable conditional access policies (geo-fencing)",
      "Require MFA re-enrollment",
      "Contact Alice Smith to confirm if she's traveling (unlikely given timing)",
      "Search logs for initial compromise vector (phishing email, credential dump)",
      "Check other user accounts for similar patterns"
    ],
    "false_positive_possibility": "Very low (<5%). The combination of geographic impossibility, device change to command-line tool, MFA bypass, and off-hours access makes legitimate user activity extremely unlikely. Would need Alice Smith to confirm she's (1) traveling to Russia, (2) using Linux command-line tools, and (3) working at 3:45 AM."
  }
}
"""
```

---

## Section 4: Production Integration

### Complete Threat Detection Platform

```python
"""
Production Threat Detection Platform
Combines lateral movement, C2, and credential compromise detection
"""
import asyncio
from typing import List
from datetime import datetime

class ThreatDetectionPlatform:
    """Unified threat detection platform"""

    def __init__(self, anthropic_api_key: str):
        self.lateral_movement_detector = LateralMovementDetector(anthropic_api_key)
        self.c2_detector = C2BeaconDetector(anthropic_api_key)
        self.credential_detector = CredentialCompromiseDetector(anthropic_api_key)
        self.alerts = []

    async def monitor(self):
        """Monitor all threat vectors continuously"""
        tasks = [
            self._monitor_authentication_logs(),
            self._monitor_netflow(),
            self._monitor_vpn_logins()
        ]
        await asyncio.gather(*tasks)

    async def _monitor_authentication_logs(self):
        """Monitor SSH/RDP/Windows auth logs for lateral movement"""
        # Integration with syslog, Windows Event Collector, etc.
        pass

    async def _monitor_netflow(self):
        """Monitor NetFlow for C2 beacons"""
        # Integration with NetFlow collector
        pass

    async def _monitor_vpn_logins(self):
        """Monitor VPN logins for credential compromise"""
        # Integration with VPN logs
        pass

    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for security dashboard"""
        return {
            'alerts_last_24h': len([a for a in self.alerts
                                   if (datetime.now() - a['timestamp']).days < 1]),
            'critical_alerts': len([a for a in self.alerts if a['severity'] == 'Critical']),
            'lateral_movement_detected': len([a for a in self.alerts
                                             if a['alert_type'] == 'Lateral Movement Detected']),
            'c2_beacons_detected': len([a for a in self.alerts
                                       if a['alert_type'] == 'C2 Beacon Detected']),
            'compromised_credentials': len([a for a in self.alerts
                                           if 'credentials_compromised' in str(a)])
        }
```

---

## What Can Go Wrong

### 1. False Positives

**Problem**: AI flags legitimate activity as threats
- Sysadmin accessing many servers = lateral movement alert
- Windows Update server beaconing = C2 alert
- User traveling for work = credential compromise alert

**Solution**:
- Whitelist known patterns
- Lower thresholds initially, tune based on feedback
- Require human validation for critical alerts
- Build feedback loop to improve baselines

### 2. Baseline Quality

**Problem**: Bad baseline = bad detections
- If you baseline during an active breach, attacker behavior becomes "normal"
- Limited historical data = poor behavioral models
- Network changes invalidate baselines

**Solution**:
- Baseline from verified clean period (post-incident)
- Require minimum 30 days of data
- Continuous baseline updates
- Periodic baseline audits

### 3. Performance at Scale

**Problem**: 10,000 devices = millions of events/day
- LLM analysis of every event is too expensive
- Real-time processing falls behind
- Alert fatigue from volume

**Solution**:
- Pre-filter with statistical analysis (only AI analyze high anomaly scores)
- Batch processing for non-critical detections
- Tiered alerting (critical = immediate, high = 15 min batching)
- Caching common patterns

### 4. Evasion

**Problem**: Attackers adapt to detection
- Slow lateral movement (1 server/day instead of 10/hour)
- Irregular beaconing intervals
- Credential use mimicking normal behavior

**Solution**:
- Multi-layered detection (can't evade all)
- Threat intelligence integration
- Behavioral analysis (not just pattern matching)
- Assume breach posture (detection + containment)

---

## Key Takeaways

1. **AI detects threats traditional tools miss** - Lateral movement, C2 beacons, and credential compromise are hard for signature-based systems but perfect for AI behavioral analysis

2. **Baseline matters** - 30+ days of clean historical data is critical for accurate detection. Bad baselines = false positives.

3. **Combine statistical + AI analysis** - Use statistics to filter (periodicity, anomaly scores), use AI to interpret context and reduce false positives

4. **Tuning is essential** - Start with high thresholds, tune down based on false positives. Every network is different.

5. **Cost management** - Don't AI-analyze every log entry. Pre-filter with rules, analyze only suspicious activity.

6. **Integration is key** - Must integrate with SIEM, SOAR, EDR, firewall for automated response. Detection without response is theater.

**Next Chapter**: We'll build AI-powered security log analysis and SIEM integration to correlate these threat detections across your entire security stack.

---

**Code Repository**: Complete working examples at `github.com/vexpertai/ai-networking-book/chapter-70/`
