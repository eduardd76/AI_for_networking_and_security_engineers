"""
Chapter 27: AI-Powered Log Triage & Classification
Classify, correlate, and prioritize network logs automatically

This module demonstrates production log analysis using rule-based and
AI-powered classification, event correlation, and incident triage.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class SeverityLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CorrelationType(str, Enum):
    TEMPORAL = "temporal"
    TOPOLOGICAL = "topological"
    CAUSAL = "causal"


@dataclass
class LogEvent:
    """Structured log event with classification."""
    timestamp: str
    hostname: str
    message: str
    raw: str
    category: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    method: str = 'embedding'


@dataclass
class CorrelationGroup:
    """Group of related events forming an incident."""
    id: str
    root_cause: LogEvent
    related_events: List[LogEvent]
    correlation_type: CorrelationType
    severity: str
    title: str
    impact: str
    recommended_action: str
    device_count: int = 1


class IncidentReport(BaseModel):
    """Structured incident analysis from LLM."""
    title: str = Field(description="Incident title")
    root_cause: str = Field(description="Root cause analysis")
    impact: str = Field(description="Business impact")
    affected_devices: List[str] = Field(description="List of affected devices")
    recommended_actions: List[str] = Field(description="Remediation steps")
    severity: str = Field(description="critical, high, medium, or low")


class LogTriageClassifier:
    """
    AI-powered log classification and correlation.

    Features:
    - Rule-based classification (fast, high confidence)
    - Embedding-based classification (slower, handles unknowns)
    - Temporal correlation (events within time window)
    - Topological correlation (events from related devices)
    - Causal analysis (event A causes event B)
    """

    def __init__(self):
        # Classification rules (pattern → category)
        self.classification_rules = [
            ('authentication_failure', re.compile(
                r'authentication\s+failed|login\s+failed|invalid\s+(password|credentials)|ssh.*failed',
                re.IGNORECASE
            )),
            ('routing_protocol_event', re.compile(
                r'bgp.*(down|flap)|ospf.*(down|neighbor)|adjacency.*(lost|down)|neighbor.*down',
                re.IGNORECASE
            )),
            ('interface_event', re.compile(
                r'interface.*(down|flap)|link.*(down|loss)|lineproto.*down|carrier.*loss',
                re.IGNORECASE
            )),
            ('resource_exhaustion', re.compile(
                r'cpu.*(high|utilization|threshold)|memory.*(threshold|exhausted)|buffer.*full',
                re.IGNORECASE
            )),
            ('hardware_failure', re.compile(
                r'fan.*fail|power.*supply|temperature.*critical|hardware.*error|system.*restart',
                re.IGNORECASE
            )),
            ('security_event', re.compile(
                r'acl.*denied|firewall.*blocked|intrusion.*detected|security.*violation',
                re.IGNORECASE
            )),
        ]

        # Severity mapping
        self.severity_map = {
            'authentication_failure': SeverityLevel.WARNING,
            'routing_protocol_event': SeverityLevel.ERROR,
            'interface_event': SeverityLevel.ERROR,
            'resource_exhaustion': SeverityLevel.ERROR,
            'hardware_failure': SeverityLevel.CRITICAL,
            'security_event': SeverityLevel.ERROR,
        }

    def classify_logs(self, logs: List[Dict]) -> List[LogEvent]:
        """
        Classify raw syslog messages.

        Args:
            logs: List of dicts with 'timestamp', 'hostname', 'message', 'raw'

        Returns:
            List of LogEvent objects with categories and confidence
        """
        classified = []

        for log in logs:
            event = LogEvent(
                timestamp=log['timestamp'],
                hostname=log['hostname'],
                message=log['message'],
                raw=log['raw']
            )

            # Try rule-based first (fast path)
            for category, pattern in self.classification_rules:
                if pattern.search(log['message']):
                    event.category = category
                    event.method = 'rule'
                    event.confidence = 0.95
                    break

            # Fallback to AI classification
            if event.category is None:
                event.category = self._classify_by_ai(log['message'])
                event.method = 'embedding'
                event.confidence = 0.75

            # Assign severity
            event.severity = self._predict_severity(event)

            classified.append(event)

        return classified

    def _classify_by_ai(self, message: str) -> str:
        """Classify using semantic understanding."""
        message_lower = message.lower()

        # Simplified heuristics (in production: use vector embeddings)
        if any(word in message_lower for word in ['down', 'unreachable', 'failed', 'loss']):
            return 'routing_protocol_event'
        elif any(word in message_lower for word in ['cpu', 'memory', 'threshold']):
            return 'resource_exhaustion'
        elif any(word in message_lower for word in ['denied', 'blocked', 'violation']):
            return 'security_event'

        return 'unknown'

    def _predict_severity(self, event: LogEvent) -> str:
        """Predict severity with context."""
        base_severity = self.severity_map.get(event.category, SeverityLevel.INFO)

        message_lower = event.message.lower()

        # Context-based adjustments
        if 'restart' in message_lower or 'crash' in message_lower:
            base_severity = SeverityLevel.CRITICAL

        elif 'critical' in message_lower:
            base_severity = SeverityLevel.CRITICAL

        elif any(word in message_lower for word in ['95%', '98%', '99%']):
            base_severity = SeverityLevel.ERROR

        return base_severity.value

    def correlate_events(
        self,
        logs: List[LogEvent],
        time_window_secs: int = 300
    ) -> List[CorrelationGroup]:
        """
        Correlate related events into incident groups.

        Args:
            logs: Classified log events
            time_window_secs: Time window for temporal correlation

        Returns:
            List of CorrelationGroup objects
        """
        groups = []
        processed = set()

        for i, log in enumerate(logs):
            if log.timestamp in processed:
                continue

            # Find related events
            related = [log]
            processed.add(log.timestamp)

            # Temporal correlation
            for other in logs[i+1:]:
                if self._are_temporally_related(log, other, time_window_secs):
                    related.append(other)
                    processed.add(other.timestamp)

            # Create group if significant
            if len(related) > 1 or log.severity in ['critical', 'error']:
                group = self._create_group(log, related)
                groups.append(group)

        return groups

    def _are_temporally_related(
        self,
        log1: LogEvent,
        log2: LogEvent,
        window_secs: int
    ) -> bool:
        """Check if events are within time window."""
        try:
            t1 = datetime.fromisoformat(log1.timestamp)
            t2 = datetime.fromisoformat(log2.timestamp)
            return abs((t2 - t1).total_seconds()) <= window_secs
        except:
            return False

    def _create_group(
        self,
        root: LogEvent,
        related: List[LogEvent]
    ) -> CorrelationGroup:
        """Create correlation group with incident report."""

        # Determine correlation type
        same_host = all(e.hostname == root.hostname for e in related)
        correlation_type = CorrelationType.CAUSAL if same_host else CorrelationType.TOPOLOGICAL

        # Generate incident details
        title, impact, action = self._generate_report(root, related)

        # Find max severity
        severity_order = ['debug', 'info', 'warning', 'error', 'critical']
        max_severity = max(
            (e.severity for e in related),
            key=lambda s: severity_order.index(s) if s in severity_order else 0
        )

        return CorrelationGroup(
            id=f"incident-{hash(root.timestamp) % 10000}",
            root_cause=root,
            related_events=related,
            correlation_type=correlation_type,
            severity=max_severity,
            title=title,
            impact=impact,
            recommended_action=action,
            device_count=len(set(e.hostname for e in related))
        )

    def _generate_report(
        self,
        root: LogEvent,
        related: List[LogEvent]
    ) -> Tuple[str, str, str]:
        """Generate incident title, impact, and action."""

        if root.category == 'hardware_failure':
            return (
                f"Device Failure: {root.hostname}",
                "Complete device outage with potential network segmentation",
                "Page NOC immediately. Check hardware health and initiate RMA if needed."
            )

        elif root.category == 'authentication_failure':
            failed_count = len(related)
            return (
                f"Brute Force Attack: {failed_count} attempts",
                f"Security incident - {failed_count} failed login attempts detected",
                "Block source IP at firewall. Review security logs for compromise indicators."
            )

        elif root.category == 'routing_protocol_event':
            return (
                f"Routing Instability: {root.hostname}",
                "BGP/OSPF adjacency loss causing potential traffic blackholing",
                "Check interface status, verify routing configuration, review recent changes."
            )

        elif root.category == 'resource_exhaustion':
            return (
                f"Resource Exhaustion: {root.hostname}",
                "High CPU/memory usage - risk of packet loss and service degradation",
                "Investigate top processes. Check for routing loops or DDoS traffic."
            )

        else:
            return (
                f"Incident: {root.category}",
                f"Multiple related events detected across {len(set(e.hostname for e in related))} devices",
                "Investigate root cause and assess impact before remediation."
            )


def example_1_classify_syslogs():
    """
    Example 1: Classify raw syslog messages
    """
    print("=" * 60)
    print("Example 1: Log Classification")
    print("=" * 60)

    raw_logs = [
        {
            'timestamp': '2026-01-18T10:15:01',
            'hostname': 'edge-rtr-01',
            'message': 'Failed password for admin from 192.168.1.100',
            'raw': 'Jan 18 10:15:01 edge-rtr-01 sshd[12345]: Failed password for admin from 192.168.1.100'
        },
        {
            'timestamp': '2026-01-18T10:20:15',
            'hostname': 'core-sw-01',
            'message': '%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down',
            'raw': 'Jan 18 10:20:15 core-sw-01 %BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down'
        },
        {
            'timestamp': '2026-01-18T10:25:30',
            'hostname': 'edge-rtr-02',
            'message': '%SYS-2-CPURISINGTHRESHOLD: CPU utilization for process is at 95%',
            'raw': 'Jan 18 10:25:30 edge-rtr-02 %SYS-2-CPURISINGTHRESHOLD: CPU utilization for process is at 95%'
        },
    ]

    classifier = LogTriageClassifier()

    print("\n📥 Processing logs...")
    classified = classifier.classify_logs(raw_logs)

    print(f"\n✅ Classified {len(classified)} events:\n")

    for event in classified:
        emoji = {
            'critical': '🔴',
            'error': '🟠',
            'warning': '🟡',
            'info': '🟢'
        }.get(event.severity, '⚪')

        method_badge = "🔧 RULE" if event.method == 'rule' else "🤖 AI"

        print(f"{emoji} [{event.severity.upper()}] {event.hostname}")
        print(f"   Category: {event.category}")
        print(f"   Method: {method_badge} (confidence: {event.confidence:.0%})")
        print(f"   Message: {event.message[:60]}...")
        print()

    print("=" * 60 + "\n")


def example_2_correlate_related_events():
    """
    Example 2: Correlate related events into incidents
    """
    print("=" * 60)
    print("Example 2: Event Correlation")
    print("=" * 60)

    # Simulated cascade: Router restart → BGP down → Network unreachable
    raw_logs = [
        {
            'timestamp': '2026-01-18T10:00:00',
            'hostname': 'core-rtr-01',
            'message': '%SYS-5-RESTART: System restarted',
            'raw': 'Jan 18 10:00:00 core-rtr-01 %SYS-5-RESTART: System restarted'
        },
        {
            'timestamp': '2026-01-18T10:00:15',
            'hostname': 'core-rtr-01',
            'message': '%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down',
            'raw': 'Jan 18 10:00:15 core-rtr-01 %BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down'
        },
        {
            'timestamp': '2026-01-18T10:00:30',
            'hostname': 'edge-rtr-01',
            'message': '%ROUTING: Network 192.168.0.0/16 unreachable via core-rtr-01',
            'raw': 'Jan 18 10:00:30 edge-rtr-01 %ROUTING: Network 192.168.0.0/16 unreachable via core-rtr-01'
        },
    ]

    classifier = LogTriageClassifier()

    classified = classifier.classify_logs(raw_logs)
    groups = classifier.correlate_events(classified, time_window_secs=60)

    print(f"\n🔗 Found {len(groups)} correlated incident(s):\n")

    for group in groups:
        print(f"📋 {group.title}")
        print(f"   Severity: {group.severity.upper()}")
        print(f"   Root Cause: {group.root_cause.message[:50]}...")
        print(f"   Related Events: {len(group.related_events)}")
        print(f"   Devices Affected: {group.device_count}")
        print(f"   Correlation Type: {group.correlation_type.value}")
        print(f"\n   💡 Impact: {group.impact}")
        print(f"   🔧 Action: {group.recommended_action}\n")

    print("=" * 60 + "\n")


def example_3_brute_force_detection():
    """
    Example 3: Detect brute force attack
    """
    print("=" * 60)
    print("Example 3: Brute Force Attack Detection")
    print("=" * 60)

    # Multiple failed SSH attempts
    raw_logs = []
    base_time = datetime.fromisoformat('2026-01-18T10:00:00')

    for i in range(10):
        timestamp = (base_time + timedelta(seconds=i*3)).isoformat()
        raw_logs.append({
            'timestamp': timestamp,
            'hostname': 'firewall-01',
            'message': f'Failed password for admin from 203.0.113.50 port {50000+i}',
            'raw': f'Jan 18 10:00:{i*3:02d} firewall-01 sshd: Failed password for admin from 203.0.113.50'
        })

    classifier = LogTriageClassifier()

    classified = classifier.classify_logs(raw_logs)
    groups = classifier.correlate_events(classified, time_window_secs=60)

    print(f"\n🚨 Security Alert!\n")

    for group in groups:
        print(f"Title: {group.title}")
        print(f"Failed Attempts: {len(group.related_events)}")
        print(f"Source: 203.0.113.50")
        print(f"Time Window: 30 seconds")
        print(f"\n⚠️  {group.impact}")
        print(f"🛡️  {group.recommended_action}")

    print("\n" + "=" * 60 + "\n")


def example_4_resource_exhaustion():
    """
    Example 4: Detect resource exhaustion cascade
    """
    print("=" * 60)
    print("Example 4: Resource Exhaustion Detection")
    print("=" * 60)

    raw_logs = [
        {
            'timestamp': '2026-01-18T14:00:00',
            'hostname': 'core-sw-01',
            'message': 'CPU utilization has reached 95%',
            'raw': 'Jan 18 14:00:00 core-sw-01 %SYS: CPU utilization has reached 95%'
        },
        {
            'timestamp': '2026-01-18T14:00:10',
            'hostname': 'core-sw-01',
            'message': 'Memory threshold exceeded - 98% used',
            'raw': 'Jan 18 14:00:10 core-sw-01 %SYS: Memory threshold exceeded - 98% used'
        },
        {
            'timestamp': '2026-01-18T14:00:20',
            'hostname': 'core-sw-01',
            'message': 'Buffer allocation failures detected',
            'raw': 'Jan 18 14:00:20 core-sw-01 %SYS: Buffer allocation failures detected'
        },
    ]

    classifier = LogTriageClassifier()

    classified = classifier.classify_logs(raw_logs)
    groups = classifier.correlate_events(classified)

    print("\n⚠️  Resource Exhaustion Detected:\n")

    for group in groups:
        print(f"Device: core-sw-01")
        print(f"Severity: {group.severity.upper()}")
        print(f"Symptoms: CPU 95%, Memory 98%, Buffer failures")
        print(f"\nImpact: {group.impact}")
        print(f"Action: {group.recommended_action}")

    print("\n" + "=" * 60 + "\n")


def example_5_llm_incident_analysis():
    """
    Example 5: Use LLM for deep incident analysis
    """
    print("=" * 60)
    print("Example 5: LLM-Powered Incident Analysis")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM analysis.")
        print("=" * 60 + "\n")
        return

    # Complex incident with multiple symptoms
    incident_logs = """
1. [10:00:00] core-rtr-01: Interface GigabitEthernet0/1 down
2. [10:00:15] core-rtr-01: %BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down
3. [10:00:30] edge-rtr-01: %ROUTING: 192.168.0.0/16 unreachable
4. [10:00:45] edge-rtr-02: %ROUTING: 192.168.0.0/16 unreachable
5. [10:01:00] monitoring: ALERT - 50% packet loss to core network
"""

    prompt = f"""You are a senior network operations engineer. Analyze this incident:

{incident_logs}

Provide structured analysis:
- Title: Short incident description
- Root cause: What triggered this cascade?
- Impact: Business and technical impact
- Affected devices: List all impacted devices
- Recommended actions: Step-by-step remediation (3-5 steps)
- Severity: critical, high, medium, or low

Be specific and actionable."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=IncidentReport)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\n🤖 Analyzing incident with Claude...\n")

    response = llm.invoke(full_prompt)
    report = parser.parse(response.content)

    print(f"📋 {report.title}")
    print(f"   Severity: {report.severity.upper()}\n")

    print(f"🔍 Root Cause:")
    print(f"   {report.root_cause}\n")

    print(f"💥 Impact:")
    print(f"   {report.impact}\n")

    print(f"🖥️  Affected Devices:")
    for device in report.affected_devices:
        print(f"   • {device}")

    print(f"\n🔧 Recommended Actions:")
    for i, action in enumerate(report.recommended_actions, 1):
        print(f"   {i}. {action}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔍 Chapter 27: AI-Powered Log Triage")
    print("Classify, Correlate, and Prioritize Network Logs\n")

    try:
        example_1_classify_syslogs()
        input("Press Enter to continue...")

        example_2_correlate_related_events()
        input("Press Enter to continue...")

        example_3_brute_force_detection()
        input("Press Enter to continue...")

        example_4_resource_exhaustion()
        input("Press Enter to continue...")

        example_5_llm_incident_analysis()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Rule-based classification is fast and accurate for known patterns")
        print("- AI classification handles unknown log formats")
        print("- Temporal correlation groups cascading failures")
        print("- Automated triage reduces MTTR significantly")
        print("- LLM analysis provides context-aware recommendations\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
