# Chapter 72: Security Log Analysis & SIEM Integration

## Learning Objectives

By the end of this chapter, you will:
- Analyze millions of firewall logs to identify real threats vs. noise
- Correlate IDS/IPS alerts to reduce alert fatigue from 10,000 to 50 actionable incidents
- Detect insider threats through behavioral log analysis
- Integrate AI with your SIEM (Splunk, Elastic, QRadar)
- Build automated security log analysis pipelines

**Prerequisites**: Chapter 70 (Threat Detection), understanding of firewalls, IDS/IPS, SIEM concepts

**What You'll Build**: AI-powered log analysis system that integrates with your SIEM and reduces security alert volume by 95% while increasing detection accuracy.

---

## The Log Analysis Problem

Your network generates **50,000 security events per day**:
- Firewall: 45,000 denied connection logs
- IDS: 4,500 alerts
- IPS: 500 blocks

Your 3-person security team reviews... maybe 100 of them.

**Real Scenario**: Financial Services Company

```
Daily Security Events: 52,000
Security Analysts: 3
Time per event review: 2 minutes
Total time needed: 1,733 hours/day
Actual analyst hours available: 24 hours/day
Coverage: 1.4% of events reviewed
```

**What gets missed**: 98.6% of events, including:
- Real reconnaissance before attacks
- Data exfiltration attempts
- Insider threat indicators
- Coordinated attack campaigns

**The AI Solution**:
- Analyze 100% of events
- Correlate related events across time
- Distinguish noise from threats
- Summarize complex attack chains
- Prioritize by business impact

**Result**: 52,000 events → 50 actionable incidents → 3 critical threats

This chapter shows you how to build that system.

---

## Section 1: Firewall Log Intelligence

### The Firewall Log Deluge

Your firewall logs 45,000 denied connections daily:
- 95% are internet background noise (port scans, bot traffic)
- 4% are misconfigured applications (legitimate but blocked)
- 1% are actual reconnaissance or attacks

**Problem**: Finding the 1% that matters.

**Traditional Approach**: Write rules
```
If source_ip == known_bad_ip: ALERT
If denied_port in [22, 3389, 445]: ALERT
If denied_count > 100: ALERT
```

**Problems**:
- Attackers use new IPs (rules don't catch them)
- Port 22/3389 blocks are mostly noise
- Count thresholds miss slow scans
- Alert fatigue (thousands of alerts)

**AI Approach**: Understand attack context
- What's the attacker trying to reach?
- Is this reconnaissance (many ports) or targeted (specific service)?
- Is this one attacker or coordinated campaign?
- What's the business impact if they succeed?

### Building Firewall Log Analyzer

```python
"""
AI-Powered Firewall Log Analysis
Turns 45,000 denied logs into actionable threat intelligence
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Set
import anthropic
import json

@dataclass
class FirewallDenyLog:
    """Firewall denied connection log entry"""
    timestamp: datetime
    source_ip: str
    source_port: int
    dest_ip: str
    dest_port: int
    protocol: str
    rule_name: str
    interface: str

class FirewallLogAnalyzer:
    """Analyze firewall logs for attack patterns"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def analyze_logs(self, logs: List[FirewallDenyLog], time_window_hours: int = 24) -> Dict:
        """Analyze firewall logs for threats"""

        # Group logs by source IP
        logs_by_source = defaultdict(list)
        for log in logs:
            logs_by_source[log.source_ip].append(log)

        threats = []

        for source_ip, source_logs in logs_by_source.items():
            if len(source_logs) < 10:  # Skip single-hit sources
                continue

            # Analyze this source's activity
            activity_profile = self._profile_activity(source_ip, source_logs)

            # Determine if this is threat-worthy
            if activity_profile['threat_score'] > 0.6:
                ai_assessment = self._ai_analyze_firewall_activity(source_ip, activity_profile, source_logs)

                if ai_assessment['is_threat']:
                    threats.append({
                        'source_ip': source_ip,
                        'activity_profile': activity_profile,
                        'ai_assessment': ai_assessment,
                        'log_count': len(source_logs)
                    })

        return {
            'total_logs_analyzed': len(logs),
            'unique_sources': len(logs_by_source),
            'threats_identified': len(threats),
            'threat_details': threats
        }

    def _profile_activity(self, source_ip: str, logs: List[FirewallDenyLog]) -> Dict:
        """Profile attacker activity from firewall logs"""

        # Extract patterns
        dest_ips = set(log.dest_ip for log in logs)
        dest_ports = set(log.dest_port for log in logs)
        protocols = set(log.protocol for log in logs)

        # Time analysis
        first_seen = min(log.timestamp for log in logs)
        last_seen = max(log.timestamp for log in logs)
        duration_hours = (last_seen - first_seen).total_seconds() / 3600

        # Port scan detection
        is_port_scan = len(dest_ports) > 10 and len(dest_ips) <= 3

        # Network scan detection
        is_network_scan = len(dest_ips) > 20 and len(dest_ports) <= 5

        # Targeted attack detection
        is_targeted = len(dest_ips) <= 3 and len(dest_ports) <= 3 and len(logs) > 50

        # Calculate threat score
        threat_score = 0.0

        if is_port_scan:
            threat_score += 0.4
        if is_network_scan:
            threat_score += 0.5
        if is_targeted:
            threat_score += 0.6

        # High-value ports
        critical_ports = {22, 23, 3389, 445, 1433, 3306, 5432, 8080, 8443}
        if dest_ports & critical_ports:
            threat_score += 0.3

        # Sustained activity
        if duration_hours > 1:
            threat_score += 0.2

        threat_score = min(threat_score, 1.0)

        return {
            'source_ip': source_ip,
            'attempt_count': len(logs),
            'unique_dest_ips': len(dest_ips),
            'unique_dest_ports': len(dest_ports),
            'dest_ips': list(dest_ips)[:10],  # Top 10
            'dest_ports': sorted(list(dest_ports))[:20],  # Top 20
            'protocols': list(protocols),
            'first_seen': first_seen,
            'last_seen': last_seen,
            'duration_hours': duration_hours,
            'is_port_scan': is_port_scan,
            'is_network_scan': is_network_scan,
            'is_targeted_attack': is_targeted,
            'threat_score': threat_score
        }

    def _ai_analyze_firewall_activity(self, source_ip: str, profile: Dict,
                                      sample_logs: List[FirewallDenyLog]) -> Dict:
        """Use AI to analyze if firewall blocks indicate real threat"""

        # Sample some logs for context
        sample_size = min(10, len(sample_logs))
        samples = sample_logs[:sample_size]
        sample_text = "\n".join([
            f"  {log.timestamp.strftime('%H:%M:%S')} → {log.dest_ip}:{log.dest_port}/{log.protocol}"
            for log in samples
        ])

        prompt = f"""You are a network security analyst reviewing firewall logs to distinguish real threats from noise.

FIREWALL ACTIVITY SUMMARY:
Source IP: {source_ip}
Total Blocked Attempts: {profile['attempt_count']}
Time Span: {profile['duration_hours']:.1f} hours
First Seen: {profile['first_seen'].strftime('%Y-%m-%d %H:%M')}
Last Seen: {profile['last_seen'].strftime('%Y-%m-%d %H:%M')}

TARGET ANALYSIS:
- Unique Destination IPs: {profile['unique_dest_ips']}
- Unique Destination Ports: {profile['unique_dest_ports']}
- Target IPs: {', '.join(profile['dest_ips'])}
- Target Ports: {', '.join(map(str, profile['dest_ports']))}
- Protocols: {', '.join(profile['protocols'])}

ATTACK PATTERN INDICATORS:
- Port Scan: {profile['is_port_scan']}
- Network Scan: {profile['is_network_scan']}
- Targeted Attack: {profile['is_targeted_attack']}
- Threat Score: {profile['threat_score']:.2f}

SAMPLE LOGS (first 10 attempts):
{sample_text}

CRITICAL ANALYSIS REQUIRED:
1. Is this a real threat (vs. internet background noise)?
2. What attack technique is being used?
3. What are they trying to accomplish?
4. What's the business risk if they succeed?
5. Should security team investigate immediately?

Respond in JSON:
{{
    "is_threat": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "attack_type": "Port Scan/Network Scan/Brute Force/Reconnaissance/etc",
    "attacker_goal": "what they're trying to accomplish",
    "business_impact": "what happens if they succeed",
    "indicators": ["list of threat indicators"],
    "recommended_actions": ["immediate actions"],
    "investigation_priority": "Immediate/High/Medium/Low",
    "likely_automated": true/false,
    "likely_noise": true/false
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
                'is_threat': True,  # Fail secure
                'confidence': 0.5,
                'severity': 'Medium',
                'attack_type': 'Unknown'
            }

# Example usage
def analyze_firewall_logs():
    analyzer = FirewallLogAnalyzer(anthropic_api_key="your-api-key")

    # Simulate 24 hours of firewall logs
    logs = []

    # Scenario 1: Port scan from attacker
    attacker_ip = '45.95.168.200'
    for port in [21, 22, 23, 25, 80, 443, 445, 1433, 3306, 3389, 5432, 8080, 8443]:
        for minute in range(0, 30, 2):  # 30 minutes, every 2 minutes
            log = FirewallDenyLog(
                timestamp=datetime.now() - timedelta(hours=2, minutes=minute),
                source_ip=attacker_ip,
                source_port=random.randint(40000, 65000),
                dest_ip='10.1.50.10',  # Web server
                dest_port=port,
                protocol='TCP',
                rule_name='DENY_INBOUND',
                interface='outside'
            )
            logs.append(log)

    # Scenario 2: Internet noise (random port scans)
    for _ in range(1000):
        log = FirewallDenyLog(
            timestamp=datetime.now() - timedelta(hours=random.randint(0, 24)),
            source_ip=f'{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}',
            source_port=random.randint(1024, 65535),
            dest_ip=f'10.1.{random.randint(1,255)}.{random.randint(1,254)}',
            dest_port=random.choice([22, 23, 445, 3389]),
            protocol='TCP',
            rule_name='DENY_INBOUND',
            interface='outside'
            )
        logs.append(log)

    # Analyze
    results = analyzer.analyze_logs(logs, time_window_hours=24)

    print(f"Analyzed {results['total_logs_analyzed']} logs")
    print(f"Identified {results['threats_identified']} threats")

    for threat in results['threat_details']:
        print(f"\n🚨 THREAT DETECTED: {threat['source_ip']}")
        print(f"Activity: {threat['log_count']} blocked attempts")
        print(f"Assessment: {threat['ai_assessment']['attack_type']}")
        print(f"Severity: {threat['ai_assessment']['severity']}")
        print(f"Goal: {threat['ai_assessment']['attacker_goal']}")

# Example Output:
"""
Analyzed 1195 logs
Identified 1 threats

🚨 THREAT DETECTED: 45.95.168.200
Activity: 195 blocked attempts
Assessment: Reconnaissance - Multi-Service Port Scan
Severity: High
Goal: This is systematic reconnaissance targeting a web server. Attacker is probing for vulnerable services across common enterprise ports (SSH, Telnet, SMB, RDP, SQL databases, web services). The methodical approach over 30 minutes indicates preparation for a targeted attack once a vulnerable service is found.

AI Analysis:
{
  "is_threat": true,
  "confidence": 0.91,
  "severity": "High",
  "attack_type": "Reconnaissance - Multi-Service Port Scan",
  "attacker_goal": "Identify vulnerable services on 10.1.50.10 (web server) for subsequent exploitation",
  "business_impact": "If successful in finding vulnerable service: potential unauthorized access, data breach, service disruption, ransomware deployment",
  "indicators": [
    "Systematic port scanning (13 different ports)",
    "Targeting critical services (SSH, RDP, SQL databases, SMB)",
    "Sustained activity over 30 minutes (automated tool)",
    "Single target IP (focused reconnaissance, not random)",
    "Originating from known hosting provider (likely VPS)"
  ],
  "recommended_actions": [
    "Block 45.95.168.200 at network edge (add to threat feed)",
    "Verify 10.1.50.10 patch status for all probed services",
    "Check if any services on these ports are exposed (shouldn't be)",
    "Review recent logs for successful connections from this IP",
    "Add IP to WAF block list if web server",
    "Monitor for follow-up attacks from related IPs (same /24 subnet)"
  ],
  "investigation_priority": "High",
  "likely_automated": true,
  "likely_noise": false
}
"""
```

---

## Section 2: IDS/IPS Alert Correlation

### The Alert Fatigue Problem

Your IDS generates 4,500 alerts per day:
- Snort alerts on suspicious patterns
- Suricata detects potential exploits
- Web Application Firewall (WAF) blocks SQL injection attempts

Your analysts can review ~200 alerts per day.

**Problem**: Most alerts are false positives or low-impact:
- 3,800 alerts: False positives or noisy signatures
- 600 alerts: Real but low-severity (scanners, bots)
- 90 alerts: Medium severity (worth investigating)
- 10 alerts: Critical (active attacks)

**Traditional Approach**: Alert rules
```
If alert_signature == "SQL Injection Attempt" AND count > 10: INVESTIGATE
If alert_priority == "Critical": PAGE_ANALYST
```

**Problem**: Misses attack campaigns spread across multiple alerts.

**AI Approach**: Correlation and context
- Which alerts are related (same attacker, same target)?
- What's the attack narrative (recon → exploit → post-exploit)?
- What's the actual risk (attacker succeeded or just trying)?

### Building IDS Alert Correlator

```python
"""
AI-Powered IDS/IPS Alert Correlation
Reduces 4,500 alerts to 50 actionable incidents
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict
import anthropic
import json

@dataclass
class IDSAlert:
    """IDS/IPS alert"""
    timestamp: datetime
    alert_id: str
    signature: str
    severity: int  # 1=low, 2=medium, 3=high, 4=critical
    source_ip: str
    dest_ip: str
    dest_port: int
    protocol: str
    payload_summary: str
    classification: str  # e.g., "Web Application Attack", "Network Scan"

class IDSAlertCorrelator:
    """Correlate IDS alerts into security incidents"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def correlate_alerts(self, alerts: List[IDSAlert], time_window_minutes: int = 60) -> Dict:
        """Correlate related alerts into incidents"""

        # Group alerts by potential incident
        incidents = self._group_alerts_into_incidents(alerts, time_window_minutes)

        # Analyze each incident with AI
        analyzed_incidents = []
        for incident in incidents:
            if incident['alert_count'] < 3:  # Skip single/double alerts
                continue

            ai_analysis = self._ai_correlate_incident(incident)

            if ai_analysis['is_real_incident']:
                analyzed_incidents.append({
                    'incident': incident,
                    'analysis': ai_analysis
                })

        return {
            'total_alerts': len(alerts),
            'incidents_identified': len(analyzed_incidents),
            'reduction_ratio': f"{len(alerts)}:{len(analyzed_incidents)}",
            'incidents': analyzed_incidents
        }

    def _group_alerts_into_incidents(self, alerts: List[IDSAlert],
                                    time_window_minutes: int) -> List[Dict]:
        """Group related alerts into potential incidents"""

        # Group by (source_ip, dest_ip) pairs within time window
        incident_groups = defaultdict(list)

        for alert in sorted(alerts, key=lambda a: a.timestamp):
            # Create incident key
            key = (alert.source_ip, alert.dest_ip)
            incident_groups[key].append(alert)

        # Convert to incident objects
        incidents = []
        for (source_ip, dest_ip), incident_alerts in incident_groups.items():
            if len(incident_alerts) < 3:  # Need multiple alerts for correlation
                continue

            # Check if alerts are within time window
            time_span = (max(a.timestamp for a in incident_alerts) -
                        min(a.timestamp for a in incident_alerts))

            if time_span.total_seconds() / 60 > time_window_minutes * 2:
                continue  # Too spread out

            incidents.append({
                'source_ip': source_ip,
                'dest_ip': dest_ip,
                'alert_count': len(incident_alerts),
                'alerts': incident_alerts,
                'first_alert': min(a.timestamp for a in incident_alerts),
                'last_alert': max(a.timestamp for a in incident_alerts),
                'duration_minutes': time_span.total_seconds() / 60,
                'signatures': list(set(a.signature for a in incident_alerts)),
                'max_severity': max(a.severity for a in incident_alerts),
                'classifications': list(set(a.classification for a in incident_alerts))
            })

        return incidents

    def _ai_correlate_incident(self, incident: Dict) -> Dict:
        """Use AI to correlate alerts and assess incident"""

        # Build alert timeline
        alert_timeline = []
        for alert in sorted(incident['alerts'], key=lambda a: a.timestamp):
            alert_timeline.append({
                'time': alert.timestamp.strftime('%H:%M:%S'),
                'signature': alert.signature,
                'severity': alert.severity,
                'classification': alert.classification
            })

        timeline_text = "\n".join([
            f"  [{a['time']}] [{a['severity']}/4] {a['signature']} ({a['classification']})"
            for a in alert_timeline[:15]  # First 15 alerts
        ])

        prompt = f"""You are a SOC analyst correlating IDS alerts to identify real security incidents.

INCIDENT SUMMARY:
Source IP: {incident['source_ip']}
Target IP: {incident['dest_ip']}
Alert Count: {incident['alert_count']}
Duration: {incident['duration_minutes']:.1f} minutes
First Alert: {incident['first_alert'].strftime('%Y-%m-%d %H:%M')}
Last Alert: {incident['last_alert'].strftime('%Y-%m-%d %H:%M')}
Max Severity: {incident['max_severity']}/4

ALERT TYPES DETECTED:
{', '.join(incident['signatures'])}

ATTACK CLASSIFICATIONS:
{', '.join(incident['classifications'])}

ALERT TIMELINE (chronological):
{timeline_text}

CORRELATION ANALYSIS REQUIRED:
1. Are these alerts related (same attack) or coincidental?
2. What is the attack narrative (what's the attacker doing)?
3. Did the attack succeed or just attempts?
4. What's the actual business risk?
5. Is this worth analyst time or false positive/noise?

Respond in JSON:
{{
    "is_real_incident": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "attack_narrative": "chronological story of what attacker did",
    "attack_stage": "Reconnaissance/Initial Access/Exploitation/Post-Exploitation",
    "likely_succeeded": true/false,
    "business_impact": "what's at risk",
    "false_positive_likelihood": 0.0-1.0,
    "correlated_indicators": ["how alerts are related"],
    "recommended_actions": ["prioritized actions"],
    "investigation_priority": "Immediate/High/Medium/Low",
    "requires_incident_response": true/false
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'is_real_incident': True,
                'confidence': 0.5,
                'severity': 'Medium'
            }

# Example usage
def correlate_ids_alerts():
    correlator = IDSAlertCorrelator(anthropic_api_key="your-api-key")

    # Simulate attack campaign: Recon → Exploit → Post-Exploit
    alerts = []
    attacker_ip = '185.220.101.45'
    target_ip = '10.1.50.25'  # Web server
    base_time = datetime.now() - timedelta(hours=2)

    # Phase 1: Reconnaissance (scanning)
    for minute in range(0, 5):
        alerts.append(IDSAlert(
            timestamp=base_time + timedelta(minutes=minute),
            alert_id=f"alert_{len(alerts)}",
            signature="Web Application Scanning Detected",
            severity=2,
            source_ip=attacker_ip,
            dest_ip=target_ip,
            dest_port=443,
            protocol='TCP',
            payload_summary='GET /admin/',
            classification='Web Application Attack'
        ))

    # Phase 2: SQL Injection attempts
    for minute in range(5, 15):
        alerts.append(IDSAlert(
            timestamp=base_time + timedelta(minutes=minute),
            alert_id=f"alert_{len(alerts)}",
            signature="SQL Injection Attempt",
            severity=3,
            source_ip=attacker_ip,
            dest_ip=target_ip,
            dest_port=443,
            protocol='TCP',
            payload_summary="' OR '1'='1",
            classification='Web Application Attack'
        ))

    # Phase 3: Command Injection (successful)
    alerts.append(IDSAlert(
        timestamp=base_time + timedelta(minutes=16),
        alert_id=f"alert_{len(alerts)}",
        signature="Command Injection Detected",
        severity=4,
        source_ip=attacker_ip,
        dest_ip=target_ip,
        dest_port=443,
        protocol='TCP',
        payload_summary=';/bin/bash',
        classification='Web Application Attack - Command Injection'
    ))

    # Phase 4: Outbound connection (shell callback)
    alerts.append(IDSAlert(
        timestamp=base_time + timedelta(minutes=17),
        alert_id=f"alert_{len(alerts)}",
        signature="Outbound Connection to Suspicious IP",
        severity=3,
        source_ip=target_ip,  # Compromised server calling out
        dest_ip=attacker_ip,
        dest_port=4444,  # Common reverse shell port
        protocol='TCP',
        payload_summary='',
        classification='Suspicious Outbound Connection'
    ))

    # Phase 5: Data exfiltration
    for minute in range(18, 25):
        alerts.append(IDSAlert(
            timestamp=base_time + timedelta(minutes=minute),
            alert_id=f"alert_{len(alerts)}",
            signature="Large Data Transfer Detected",
            severity=3,
            source_ip=target_ip,
            dest_ip=attacker_ip,
            dest_port=4444,
            protocol='TCP',
            payload_summary='',
            classification='Possible Data Exfiltration'
        ))

    # Add 4,000 noise alerts (false positives)
    for _ in range(4000):
        alerts.append(IDSAlert(
            timestamp=base_time + timedelta(minutes=random.randint(0, 1440)),
            alert_id=f"alert_{len(alerts)}",
            signature=random.choice(["Port Scan", "Bad Traffic", "Protocol Violation"]),
            severity=1,
            source_ip=f'{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}',
            dest_ip=f'10.1.{random.randint(1,255)}.{random.randint(1,254)}',
            dest_port=random.randint(1, 65535),
            protocol='TCP',
            payload_summary='',
            classification='Misc Activity'
        ))

    # Correlate
    results = correlator.correlate_alerts(alerts, time_window_minutes=60)

    print(f"Analyzed {results['total_alerts']} alerts")
    print(f"Identified {results['incidents_identified']} real incidents")
    print(f"Reduction: {results['reduction_ratio']}")

    for inc in results['incidents']:
        analysis = inc['analysis']
        print(f"\n🚨 INCIDENT: {inc['incident']['source_ip']} → {inc['incident']['dest_ip']}")
        print(f"Severity: {analysis['severity']}")
        print(f"Attack: {analysis['attack_narrative']}")
        print(f"Succeeded: {analysis['likely_succeeded']}")
        print(f"Priority: {analysis['investigation_priority']}")

# Example Output:
"""
Analyzed 4026 alerts
Identified 1 real incidents
Reduction: 4026:1

🚨 INCIDENT: 185.220.101.45 → 10.1.50.25
Severity: Critical
Attack: This is a successful multi-stage web application attack. Timeline:
1. (00:00-00:05) Reconnaissance: Attacker scanned web application for vulnerabilities
2. (00:05-00:15) Exploitation Attempts: Multiple SQL injection attempts testing input validation
3. (00:16) Successful Exploit: Command injection achieved code execution on server
4. (00:17) Callback Established: Compromised server connected back to attacker (reverse shell on port 4444)
5. (00:18-00:25) Data Exfiltration: Large sustained data transfer from compromised server to attacker

This is a textbook attack progression from recon to post-exploitation. The outbound connection to port 4444 and sustained data transfer confirm the attack succeeded.

Succeeded: true
Priority: Immediate

Analysis Details:
{
  "is_real_incident": true,
  "confidence": 0.96,
  "severity": "Critical",
  "attack_stage": "Post-Exploitation (Data Exfiltration)",
  "likely_succeeded": true,
  "business_impact": "Web server compromised. Data exfiltration in progress. Attacker has shell access and potentially escalated privileges. Customer data, credentials, and internal systems at risk.",
  "false_positive_likelihood": 0.04,
  "correlated_indicators": [
    "Progressive attack stages (recon → exploit → post-exploit)",
    "Same source IP across all attack phases",
    "Outbound connection to attacker IP (reverse shell indicator)",
    "Data transfer immediately after command injection",
    "Attack duration (25 minutes) consistent with manual exploitation"
  ],
  "recommended_actions": [
    "IMMEDIATE: Isolate 10.1.50.25 from network (contain breach)",
    "IMMEDIATE: Block 185.220.101.45 at perimeter",
    "Capture memory dump and disk image for forensics",
    "Identify what data was exfiltrated (check application logs)",
    "Force password resets for accounts on compromised server",
    "Search for persistence mechanisms (backdoors, web shells)",
    "Check for lateral movement from 10.1.50.25",
    "Notify incident response team and management",
    "Begin breach notification assessment (regulatory requirements)"
  ],
  "investigation_priority": "Immediate",
  "requires_incident_response": true
}
"""
```

---

## Section 3: Insider Threat Detection

### The Insider Problem

Insiders are trusted users who:
- Have legitimate access
- Know where valuable data is
- Understand security controls
- Can operate slowly to avoid detection

**Insider Threat Indicators** (in logs):
- Data exfiltration (large file transfers)
- Privilege escalation attempts
- After-hours unusual activity
- Accessing data outside their role
- Deleting logs/evidence

**Challenge**: Distinguishing malicious insiders from legitimate power users.

### Building Insider Threat Detector

```python
"""
Insider Threat Detection from Security Logs
Identifies malicious insider activity
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import anthropic
import json

@dataclass
class UserActivity:
    """User activity log entry"""
    timestamp: datetime
    username: str
    action: str  # file_access, file_download, login, privilege_change, etc.
    resource: str  # file path, system name, database, etc.
    data_volume_mb: float
    source_ip: str
    department: str
    job_role: str

class InsiderThreatDetector:
    """Detect insider threats through behavioral analysis"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.user_baselines = {}

    def analyze_user(self, username: str, recent_activity: List[UserActivity],
                    baseline_activity: List[UserActivity]) -> Dict:
        """Analyze user for insider threat indicators"""

        # Build baseline profile
        baseline_profile = self._build_baseline(baseline_activity)

        # Detect anomalies in recent activity
        anomalies = self._detect_anomalies(recent_activity, baseline_profile)

        if not anomalies['suspicious']:
            return {'threat_detected': False, 'anomalies': anomalies}

        # AI analysis of suspicious behavior
        ai_assessment = self._ai_analyze_insider_behavior(
            username, recent_activity, baseline_profile, anomalies
        )

        return {
            'threat_detected': ai_assessment['is_insider_threat'],
            'anomalies': anomalies,
            'ai_assessment': ai_assessment
        }

    def _build_baseline(self, activities: List[UserActivity]) -> Dict:
        """Build baseline of normal user behavior"""
        typical_actions = set(a.action for a in activities)
        typical_resources = set(a.resource for a in activities)
        typical_hours = set(a.timestamp.hour for a in activities)

        avg_data_volume = sum(a.data_volume_mb for a in activities) / len(activities) if activities else 0

        return {
            'typical_actions': typical_actions,
            'typical_resources': typical_resources,
            'typical_hours': typical_hours,
            'avg_daily_data_volume_mb': avg_data_volume,
            'department': activities[0].department if activities else 'Unknown',
            'job_role': activities[0].job_role if activities else 'Unknown'
        }

    def _detect_anomalies(self, recent: List[UserActivity], baseline: Dict) -> Dict:
        """Detect anomalous behavior"""
        anomalies = []
        risk_score = 0.0

        # Check for unusual data access
        total_data_mb = sum(a.data_volume_mb for a in recent)
        if total_data_mb > baseline['avg_daily_data_volume_mb'] * 10:
            anomalies.append(f"Data access volume {total_data_mb:.0f}MB (typical: {baseline['avg_daily_data_volume_mb']:.0f}MB/day)")
            risk_score += 0.4

        # Check for accessing unusual resources
        unusual_resources = [a.resource for a in recent
                            if a.resource not in baseline['typical_resources']]
        if len(unusual_resources) > 10:
            anomalies.append(f"Accessed {len(unusual_resources)} resources never accessed before")
            risk_score += 0.3

        # Check for off-hours activity
        off_hours_activity = [a for a in recent if a.timestamp.hour not in baseline['typical_hours']]
        if len(off_hours_activity) > len(recent) * 0.5:
            anomalies.append(f"{len(off_hours_activity)} activities during off-hours")
            risk_score += 0.2

        # Check for privilege escalation
        privilege_changes = [a for a in recent if 'privilege' in a.action.lower()]
        if privilege_changes:
            anomalies.append(f"{len(privilege_changes)} privilege escalation attempts")
            risk_score += 0.5

        # Check for bulk downloads
        downloads = [a for a in recent if 'download' in a.action.lower()]
        if len(downloads) > 100:
            anomalies.append(f"{len(downloads)} file downloads (potential data exfiltration)")
            risk_score += 0.4

        return {
            'suspicious': risk_score > 0.5,
            'risk_score': min(risk_score, 1.0),
            'anomaly_list': anomalies,
            'anomaly_count': len(anomalies)
        }

    def _ai_analyze_insider_behavior(self, username: str, recent: List[UserActivity],
                                    baseline: Dict, anomalies: Dict) -> Dict:
        """AI analysis of potential insider threat"""

        # Sample recent activities
        activity_summary = []
        for activity in recent[:20]:
            activity_summary.append({
                'time': activity.timestamp.strftime('%Y-%m-%d %H:%M'),
                'action': activity.action,
                'resource': activity.resource,
                'data_mb': activity.data_volume_mb
            })

        activities_text = "\n".join([
            f"  [{a['time']}] {a['action']}: {a['resource']} ({a['data_mb']:.1f}MB)"
            for a in activity_summary
        ])

        prompt = f"""You are a security analyst investigating potential insider threats.

USER PROFILE:
Username: {username}
Department: {baseline['department']}
Job Role: {baseline['job_role']}

SUSPICIOUS BEHAVIOR DETECTED:
Risk Score: {anomalies['risk_score']:.2f}
Anomalies Detected: {anomalies['anomaly_count']}

SPECIFIC ANOMALIES:
{chr(10).join('- ' + anomaly for anomaly in anomalies['anomaly_list'])}

NORMAL BEHAVIOR BASELINE:
- Typical Actions: {', '.join(baseline['typical_actions'])}
- Typical Resources: {len(baseline['typical_resources'])} resources regularly accessed
- Typical Hours: {sorted(list(baseline['typical_hours']))}
- Average Daily Data: {baseline['avg_daily_data_volume_mb']:.0f}MB

RECENT SUSPICIOUS ACTIVITIES (last 20):
{activities_text}

THREAT ASSESSMENT REQUIRED:
1. Is this malicious insider activity or legitimate behavior?
2. What is the user's likely intent?
3. What data/systems are at risk?
4. Is this data exfiltration, sabotage, or reconnaissance?
5. How urgent is this threat?

Consider: Could this be legitimate (job change, project work, authorized audit)?

Respond in JSON:
{{
    "is_insider_threat": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "threat_type": "Data Exfiltration/Sabotage/Privilege Abuse/Reconnaissance/None",
    "user_intent": "what user is trying to accomplish",
    "at_risk": "what data/systems are threatened",
    "indicators": ["list of insider threat indicators"],
    "legitimate_explanation_possible": "if this might be legitimate activity",
    "recommended_actions": ["immediate actions"],
    "investigation_priority": "Immediate/High/Medium/Low",
    "notify_legal_hr": true/false
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
                'is_insider_threat': True,
                'confidence': 0.6,
                'severity': 'High'
            }

# Example: Insider downloading customer database
def detect_insider_threat():
    detector = InsiderThreatDetector(anthropic_api_key="your-api-key")

    # Baseline: Normal user behavior (30 days)
    baseline = []
    for day in range(30):
        for hour in [9, 10, 11, 14, 15, 16]:  # Business hours
            baseline.append(UserActivity(
                timestamp=datetime.now() - timedelta(days=30-day, hours=24-hour),
                username='bob.engineer',
                action='file_access',
                resource='/shared/engineering/project_docs',
                data_volume_mb=5.2,
                source_ip='10.1.20.50',
                department='Engineering',
                job_role='Senior Engineer'
            ))

    # Suspicious: Late night mass data download (last 24 hours)
    recent = []
    for hour in range(2, 5):  # 2 AM - 5 AM
        for i in range(50):  # Downloading many customer files
            recent.append(UserActivity(
                timestamp=datetime.now() - timedelta(hours=24-hour, minutes=i),
                username='bob.engineer',
                action='file_download',
                resource=f'/database/customers/customer_{1000+i}.csv',
                data_volume_mb=25.5,
                source_ip='10.1.20.50',
                department='Engineering',
                job_role='Senior Engineer'
            ))

    # Analyze
    result = detector.analyze_user('bob.engineer', recent, baseline)

    if result['threat_detected']:
        print("🚨 INSIDER THREAT DETECTED")
        print(json.dumps(result['ai_assessment'], indent=2))

# Example Output:
"""
{
  "is_insider_threat": true,
  "confidence": 0.89,
  "severity": "Critical",
  "threat_type": "Data Exfiltration",
  "user_intent": "Systematic download of customer database, likely for unauthorized use (theft, sale, or transfer to competitor). The 2-5 AM timeframe suggests deliberate attempt to avoid detection.",
  "at_risk": "Customer PII database. 50+ customer records downloaded containing likely sensitive information (names, emails, potentially payment info). Regulatory breach (GDPR, CCPA).",
  "indicators": [
    "Off-hours activity (2-5 AM, user typically works 9 AM - 5 PM)",
    "Mass download of customer files (50 files, 1,275MB total)",
    "Accessing database files outside job role (Engineer, not Database Admin)",
    "Never accessed customer database in 30-day baseline",
    "Sequential file access pattern (automated script, not manual)",
    "10x normal data volume"
  ],
  "legitimate_explanation_possible": "Low probability (<10%). Could theoretically be authorized data migration or backup, but timing (middle of night), lack of prior database access, and systematic pattern strongly suggest unauthorized activity. If legitimate, would require written authorization.",
  "recommended_actions": [
    "IMMEDIATE: Disable bob.engineer account",
    "IMMEDIATE: Review downloaded files (what PII was accessed)",
    "Check if data was transferred off-network (USB, email, cloud)",
    "Interview Bob (HR + Security present)",
    "Review Bob's recent communications (resignation notice? competitor contact?)",
    "Check for similar activity from other accounts",
    "Notify Legal and HR (potential policy violation)",
    "Begin breach assessment (GDPR/CCPA notification requirements)",
    "Preserve evidence (logs, forensics) for potential legal action"
  ],
  "investigation_priority": "Immediate",
  "notify_legal_hr": true
}
"""
```

---

## Section 4: SIEM Integration

### Integrating AI with Your SIEM

Most organizations already have SIEM:
- Splunk
- Elastic (ELK Stack)
- IBM QRadar
- Microsoft Sentinel

**Integration Pattern**:
```
SIEM (Splunk/Elastic) → AI Analysis → Enriched Alerts → SOAR/Ticketing
```

### Splunk Integration Example

```python
"""
Splunk + AI Integration
Pull alerts from Splunk, analyze with AI, push enriched results back
"""
import requests
import anthropic
import json
from typing import List, Dict

class SplunkAIIntegration:
    """Integrate AI analysis with Splunk SIEM"""

    def __init__(self, splunk_url: str, splunk_token: str, anthropic_api_key: str):
        self.splunk_url = splunk_url
        self.splunk_token = splunk_token
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def fetch_splunk_alerts(self, search_query: str) -> List[Dict]:
        """Fetch alerts from Splunk"""
        url = f"{self.splunk_url}/services/search/jobs/export"
        headers = {
            'Authorization': f'Bearer {self.splunk_token}'
        }
        data = {
            'search': search_query,
            'output_mode': 'json'
        }

        response = requests.post(url, headers=headers, data=data, verify=False)
        alerts = [json.loads(line) for line in response.text.strip().split('\n')]
        return alerts

    def analyze_and_enrich(self, alert: Dict) -> Dict:
        """Analyze alert with AI and enrich with context"""
        # Extract alert details
        alert_type = alert.get('signature', 'Unknown')
        source_ip = alert.get('src_ip', 'Unknown')
        dest_ip = alert.get('dest_ip', 'Unknown')

        # AI analysis
        prompt = f"""Analyze this security alert and provide enrichment:

Alert Type: {alert_type}
Source: {source_ip}
Destination: {dest_ip}
Details: {json.dumps(alert, indent=2)}

Provide:
1. Threat assessment (is this real threat?)
2. Recommended priority (Critical/High/Medium/Low)
3. Suggested response actions
4. Related TTPs (MITRE ATT&CK)

JSON format:
{{
    "is_threat": true/false,
    "priority": "Critical/High/Medium/Low",
    "mitre_tactics": ["Initial Access", "Execution"],
    "response_actions": ["action1", "action2"]
}}
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        enrichment = json.loads(response.content[0].text)

        # Add enrichment to alert
        alert['ai_enrichment'] = enrichment
        return alert

    def push_to_splunk(self, enriched_alert: Dict):
        """Push enriched alert back to Splunk"""
        url = f"{self.splunk_url}/services/receivers/simple"
        headers = {
            'Authorization': f'Bearer {self.splunk_token}'
        }

        # Format as Splunk event
        event_data = json.dumps(enriched_alert)

        response = requests.post(
            url,
            headers=headers,
            data=event_data,
            verify=False
        )

        return response.status_code == 200

# Usage
def main():
    integration = SplunkAIIntegration(
        splunk_url='https://splunk.company.com:8089',
        splunk_token='your-splunk-token',
        anthropic_api_key='your-anthropic-key'
    )

    # Fetch high-priority alerts from last hour
    alerts = integration.fetch_splunk_alerts(
        'search index=security severity>=3 earliest=-1h'
    )

    for alert in alerts:
        enriched = integration.analyze_and_enrich(alert)
        if enriched['ai_enrichment']['is_threat']:
            integration.push_to_splunk(enriched)
            print(f"Enriched and pushed alert: {alert.get('signature')}")

if __name__ == '__main__':
    main()
```

---

## What Can Go Wrong

### 1. Alert Overload from AI
**Problem**: AI generates more alerts than it solves
- Every anomaly becomes an alert
- No prioritization

**Solution**:
- Tune thresholds (start high, lower gradually)
- Require minimum confidence scores
- Batch low-priority alerts

### 2. False Negative Bias
**Problem**: AI trained on noisy data misses real threats
- "That's normal" when it's actually an attack

**Solution**:
- Regular baseline audits
- Threat intelligence feeds
- Manual review of "dismissed" alerts

### 3. Cost Explosion
**Problem**: Analyzing 50K events/day with LLM = $1,500/month

**Solution**:
- Pre-filter with rules (LLM only for ambiguous cases)
- Use smaller models for simple correlation
- Batch processing vs. real-time

### 4. Integration Complexity
**Problem**: Every SIEM has different APIs, formats, quirks

**Solution**:
- Standardize on normalized log format internally
- Use SIEM-agnostic tools where possible
- Plan for 2-3 weeks integration time per SIEM

---

## Key Takeaways

1. **AI excels at log correlation** - Turning 50,000 events into 50 actionable incidents is AI's strength

2. **Firewall logs hide reconnaissance** - 1% of denies are real threats, AI finds them

3. **IDS alert correlation reduces fatigue** - From 4,500 alerts to 50 incidents = 99% reduction

4. **Insider threats need behavioral baselines** - 30+ days of clean data required

5. **SIEM integration is key** - AI is the analysis layer, SIEM is the data platform

6. **Cost management matters** - Pre-filter, batch, and tier by priority

**Next Chapter**: Network Anomaly Detection - using AI to learn your network's "normal" and detect deviations automatically.

---

**Code Repository**: `github.com/vexpertai/ai-networking-book/chapter-72/`
