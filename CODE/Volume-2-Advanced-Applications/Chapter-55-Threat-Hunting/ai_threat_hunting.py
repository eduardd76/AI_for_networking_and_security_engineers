"""
Chapter 55: AI-Powered Threat Hunting for Networks
Behavioral Detection and MITRE ATT&CK Mapping

Detect threats in network logs using AI-powered behavioral analysis,
anomaly detection, and mapping to MITRE ATT&CK tactics.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

load_dotenv()


class ThreatLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"


class MITRETactic(str, Enum):
    RECONNAISSANCE = "Reconnaissance"
    INITIAL_ACCESS = "Initial Access"
    EXECUTION = "Execution"
    PERSISTENCE = "Persistence"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    LATERAL_MOVEMENT = "Lateral Movement"
    EXFILTRATION = "Exfiltration"
    COMMAND_AND_CONTROL = "Command and Control"


class ThreatIndicator(BaseModel):
    indicator_type: str = Field(description="Type: IP, User, Pattern, etc")
    value: str
    confidence: int = Field(description="Confidence 0-100")


class ThreatDetection(BaseModel):
    threat_name: str
    threat_level: ThreatLevel
    mitre_tactics: List[MITRETactic]
    indicators: List[ThreatIndicator]
    affected_assets: List[str]
    recommended_actions: List[str]
    investigation_priority: int = Field(description="Priority 1-10, 10=highest")


def detect_ssh_brute_force():
    """
    Example 1: Detect SSH brute force attempts
    """
    print("=" * 60)
    print("Example 1: SSH Brute Force Detection")
    print("=" * 60)

    logs = """
    Jan 18 10:15:01 firewall sshd[12345]: Failed password for admin from 192.168.1.100 port 52341 ssh2
    Jan 18 10:15:03 firewall sshd[12346]: Failed password for admin from 192.168.1.100 port 52342 ssh2
    Jan 18 10:15:05 firewall sshd[12347]: Failed password for admin from 192.168.1.100 port 52343 ssh2
    Jan 18 10:15:07 firewall sshd[12348]: Failed password for root from 192.168.1.100 port 52344 ssh2
    Jan 18 10:15:09 firewall sshd[12349]: Failed password for root from 192.168.1.100 port 52345 ssh2
    Jan 18 10:15:11 firewall sshd[12350]: Failed password for user from 192.168.1.100 port 52346 ssh2
    Jan 18 10:15:13 firewall sshd[12351]: Failed password for admin from 192.168.1.100 port 52347 ssh2
    Jan 18 10:15:15 firewall sshd[12352]: Accepted password for admin from 192.168.1.100 port 52348 ssh2
    """

    prompt = f"""Analyze these SSH logs for security threats:

{logs}

Identify:
1. Type of attack
2. Threat level
3. MITRE ATT&CK tactics
4. Indicators of compromise
5. Affected systems
6. Recommended immediate actions

Provide structured threat detection output."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ThreatDetection)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing SSH logs for threats...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"🚨 Threat Detected: {result.threat_name}")
    print(f"   Severity: {result.threat_level.value.upper()}")
    print(f"   Priority: {result.investigation_priority}/10")

    print(f"\n📋 MITRE ATT&CK Tactics:")
    for tactic in result.mitre_tactics:
        print(f"   • {tactic.value}")

    print(f"\n🔍 Indicators of Compromise:")
    for ioc in result.indicators:
        print(f"   • {ioc.indicator_type}: {ioc.value} (confidence: {ioc.confidence}%)")

    print(f"\n💻 Affected Assets:")
    for asset in result.affected_assets:
        print(f"   • {asset}")

    print(f"\n🔧 Recommended Actions:")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"   {i}. {action}")

    print("\n" + "=" * 60 + "\n")
    return result


def detect_data_exfiltration():
    """
    Example 2: Detect potential data exfiltration
    """
    print("=" * 60)
    print("Example 2: Data Exfiltration Detection")
    print("=" * 60)

    network_flow = """
    NetFlow Summary (Last Hour):

    Source: 192.168.10.50 (File Server)
    Destination: 203.0.113.45 (External IP - Unknown)
    Protocol: HTTPS (443)
    Data Transferred: 15.2 GB
    Duration: 45 minutes
    Packets: 12,450,000
    Time: 02:00 AM - 02:45 AM

    Historical Context:
    - This server typically transfers 100-500 MB/hour
    - Previous max transfer: 2GB (during backup window)
    - This IP (203.0.113.45) never seen before
    - Transfer occurred outside business hours
    - User account: service_account_backup
    """

    prompt = f"""Analyze this network flow data for potential data exfiltration:

{network_flow}

Assess:
1. Likelihood of data exfiltration
2. Threat level and urgency
3. MITRE ATT&CK mapping
4. IOCs (IP addresses, accounts, patterns)
5. Investigation steps
6. Containment actions"""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ThreatDetection)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing network flow for exfiltration...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"⚠️  Threat: {result.threat_name}")
    print(f"   Level: {result.threat_level.value.upper()}")
    print(f"   Investigation Priority: {result.investigation_priority}/10")

    print(f"\n🎯 MITRE Tactics:")
    for tactic in result.mitre_tactics:
        print(f"   → {tactic.value}")

    print(f"\n🚩 Red Flags (IoCs):")
    for ioc in result.indicators:
        print(f"   • {ioc.indicator_type}: {ioc.value}")
        print(f"     Confidence: {ioc.confidence}%")

    print(f"\n🔐 Containment Actions:")
    for action in result.recommended_actions:
        print(f"   • {action}")

    print("\n" + "=" * 60 + "\n")
    return result


def detect_lateral_movement():
    """
    Example 3: Detect lateral movement attempts
    """
    print("=" * 60)
    print("Example 3: Lateral Movement Detection")
    print("=" * 60)

    logs = """
    Authentication Logs (Last 30 minutes):

    10:00 AM: admin@workstation-01 logged in (normal)
    10:05 AM: admin@workstation-01 -> SSH to server-db-01 (SUCCESS)
    10:07 AM: admin@server-db-01 -> SSH to server-web-01 (SUCCESS)
    10:10 AM: admin@server-web-01 -> SSH to server-file-01 (SUCCESS)
    10:12 AM: admin@server-file-01 -> SMB access to \\\\dc-01\\SYSVOL (SUCCESS)
    10:15 AM: admin@server-file-01 -> RDP to dc-01 (FAILED - permission denied)
    10:16 AM: admin@server-file-01 -> RDP to dc-01 (FAILED - permission denied)
    10:18 AM: Administrator@server-file-01 -> RDP to dc-01 (SUCCESS)

    Context:
    - User 'admin' is a junior network engineer
    - User typically only accesses workstations and switches
    - No change tickets for server access
    - Database, web, and domain controller access is unusual
    - Privilege escalation from 'admin' to 'Administrator' account
    """

    prompt = f"""Analyze these authentication logs for lateral movement:

{logs}

Identify:
1. Attack pattern (is this lateral movement?)
2. Threat severity
3. MITRE ATT&CK tactics
4. Compromised accounts/systems
5. Next likely targets
6. Immediate containment steps"""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ThreatDetection)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing for lateral movement...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"🚨 ALERT: {result.threat_name}")
    print(f"   Severity: {result.threat_level.value.upper()}")
    print(f"   Priority: {result.investigation_priority}/10")

    print(f"\n📍 Attack Chain (MITRE):")
    for tactic in result.mitre_tactics:
        print(f"   {tactic.value}")

    print(f"\n🔴 Compromised:")
    for asset in result.affected_assets:
        print(f"   • {asset}")

    print(f"\n⚡ IMMEDIATE ACTIONS:")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"   {i}. {action}")

    print("\n" + "=" * 60 + "\n")
    return result


def hunt_for_c2_communication():
    """
    Example 4: Hunt for Command & Control traffic
    """
    print("=" * 60)
    print("Example 4: C2 Communication Detection")
    print("=" * 60)

    network_patterns = """
    Suspicious Network Patterns Detected:

    Internal Host: 192.168.10.75 (employee-laptop-42)

    Pattern 1: Regular Beaconing
    - Destination: 198.51.100.23:8443
    - Interval: Every 10 minutes (±30 seconds)
    - Payload: Small encrypted packets (120-150 bytes)
    - Duration: Last 3 days
    - Time: 24/7 including nights/weekends

    Pattern 2: DNS Tunneling Indicators
    - High volume of DNS TXT queries
    - Queries to unusual domains: x7h2k9.example-domain.com
    - Subdomain labels are long hex strings
    - Average 200+ DNS queries/hour (normal: 10-20)

    Pattern 3: Data Staging
    - Internal file sharing to temporary directories
    - Compression of multiple files
    - No legitimate business application detected

    Host Details:
    - User: john.doe@company.com
    - Last seen: Today, 9:45 AM
    - OS: Windows 10
    - Recent activity: Normal web browsing, email
    """

    prompt = f"""Analyze these network patterns for C2 communication:

{network_patterns}

Determine:
1. Is this likely C2 traffic?
2. Confidence level
3. MITRE ATT&CK tactics
4. Forensic investigation steps
5. Containment priority
6. Remediation actions"""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ThreatDetection)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nHunting for C2 indicators...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"⚠️  Detection: {result.threat_name}")
    print(f"   Threat Level: {result.threat_level.value.upper()}")
    print(f"   Investigation Priority: {result.investigation_priority}/10")

    print(f"\n🔗 C2 Indicators:")
    for ioc in result.indicators:
        print(f"   • {ioc.indicator_type}: {ioc.value}")
        print(f"     Confidence: {ioc.confidence}%")

    print(f"\n🎯 MITRE Tactics:")
    for tactic in result.mitre_tactics:
        print(f"   → {tactic.value}")

    print(f"\n🚨 CRITICAL ACTIONS:")
    for action in result.recommended_actions:
        print(f"   ⚡ {action}")

    print("\n" + "=" * 60 + "\n")
    return result


def automated_threat_hunting_workflow():
    """
    Example 5: Automated threat hunting workflow
    """
    print("=" * 60)
    print("Example 5: Automated Threat Hunting Workflow")
    print("=" * 60)

    workflow = """
AUTOMATED THREAT HUNTING WORKFLOW:

Phase 1: DATA COLLECTION
├── Ingest network flows (NetFlow/IPFIX)
├── Collect authentication logs (RADIUS/TACACS)
├── Parse firewall logs
├── Monitor DNS queries
└── Aggregate endpoint telemetry

Phase 2: BASELINE ESTABLISHMENT
├── Learn normal behavior patterns (per user, per host, per time)
├── Calculate statistical baselines (traffic volume, connection patterns)
├── Build user/entity behavior profiles
└── Update baselines continuously

Phase 3: ANOMALY DETECTION
├── Statistical anomalies (volume, frequency, timing)
├── Behavioral anomalies (unusual access patterns)
├── Geolocation anomalies (impossible travel)
├── Protocol anomalies (misuse, tunneling)
└── Temporal anomalies (off-hours activity)

Phase 4: THREAT CORRELATION
├── Group related anomalies into potential incidents
├── Map to MITRE ATT&CK tactics
├── Calculate threat score (0-100)
├── Enrich with threat intel (IPs, domains, hashes)
└── Prioritize by business impact

Phase 5: AI ANALYSIS
├── LLM analyzes aggregated indicators
├── Generates threat hypothesis
├── Identifies attack patterns
├── Provides confidence score
└── Suggests investigation steps

Phase 6: AUTOMATED RESPONSE
├── High confidence + low risk → Auto-block
├── High confidence + high risk → Alert SOC
├── Medium confidence → Monitor closely
├── Low confidence → Log for analysis
└── Generate incident ticket

DETECTION RULES (Examples):

Rule: SSH Brute Force
├── Trigger: >5 failed SSH attempts in 60s
├── Action: Alert + Temp block (15 min)
└── Confidence: High

Rule: Data Exfiltration
├── Trigger: >5GB to new external IP after hours
├── Action: Alert SOC + Block connection
└── Confidence: Medium (needs investigation)

Rule: Lateral Movement
├── Trigger: Admin account accessing >3 servers in <10min
├── Action: Alert SOC + Force MFA
└── Confidence: High

Rule: C2 Beaconing
├── Trigger: Regular outbound connections (±5% variance) to suspicious IP
├── Action: Alert SOC + Isolate host
└── Confidence: Very High

INTEGRATION POINTS:
• SIEM (Splunk, ELK, Sentinel)
• SOAR (Automation & Response)
• EDR (Endpoint Detection)
• Firewall (Block rules)
• IPAM (Asset context)
• Threat Intel Feeds (enrichment)
"""

    print(workflow)

    print("\nCode Integration Example:")
    print("-" * 60)
    print("""
from threat_hunter import ThreatHuntingAgent

agent = ThreatHuntingAgent(
    llm_model="claude-sonnet-4-20250514",
    threat_intel_feeds=["alienvault", "abuse.ch"],
    mitre_mapping=True,
    auto_response_enabled=True
)

# Continuous threat hunting
while True:
    # Collect logs (last 5 minutes)
    logs = collect_network_logs(window_minutes=5)

    # Detect threats
    threats = agent.hunt_threats(logs)

    for threat in threats:
        if threat.level == "critical" and threat.confidence > 80:
            # Auto-block high confidence threats
            agent.block_ioc(threat.indicators)
            alert_soc(threat)

        elif threat.level in ["high", "critical"]:
            # Alert for manual review
            alert_soc(threat)

        else:
            # Log for trending
            log_threat(threat)

    time.sleep(300)  # Every 5 minutes
""")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔐 Chapter 55: AI-Powered Threat Hunting")
    print("Behavioral Detection & MITRE ATT&CK Mapping\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        exit(1)

    try:
        # Run examples
        detect_ssh_brute_force()
        input("Press Enter to continue...")

        detect_data_exfiltration()
        input("Press Enter to continue...")

        detect_lateral_movement()
        input("Press Enter to continue...")

        hunt_for_c2_communication()
        input("Press Enter to continue...")

        automated_threat_hunting_workflow()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- AI can detect behavioral anomalies humans miss")
        print("- Map threats to MITRE ATT&CK for standardization")
        print("- Automate response for high-confidence threats")
        print("- Continuous hunting catches threats early")
        print("- Combine statistical + AI analysis for best results\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
