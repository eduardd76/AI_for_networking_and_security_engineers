#!/usr/bin/env python3
"""
AI-Powered Network Log Analysis

Analyze network logs to detect issues, patterns, and anomalies.

From: AI for Networking Engineers - Volume 2, Chapter 24
Author: Eduard Dulharu

Usage:
    python log_analyzer.py
"""

from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define output schemas
class LogIssue(BaseModel):
    """A detected issue in logs."""
    severity: str = Field(description="Severity: critical, high, medium, low")
    category: str = Field(description="Category: security, performance, config, hardware")
    issue: str = Field(description="Description of the issue")
    affected_devices: List[str] = Field(description="List of affected device IPs or hostnames")
    first_seen: str = Field(description="When issue first appeared")
    count: int = Field(description="Number of occurrences")
    recommendation: str = Field(description="How to fix or investigate")


class LogAnalysis(BaseModel):
    """Complete log analysis report."""
    summary: str = Field(description="Executive summary of findings")
    issues: List[LogIssue] = Field(description="List of detected issues")
    patterns: List[str] = Field(description="Notable patterns or trends")
    anomalies: List[str] = Field(description="Unusual events worth investigating")


class NetworkLogAnalyzer:
    """
    AI-powered network log analyzer.

    Analyzes syslog, device logs, and event logs to detect issues.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

    def analyze_logs(self, logs: List[str], context: str = "") -> LogAnalysis:
        """
        Analyze a batch of logs.

        Args:
            logs: List of log entries
            context: Optional context about the environment

        Returns:
            Analysis report
        """
        # Create parser
        parser = PydanticOutputParser(pydantic_object=LogAnalysis)

        # Create prompt
        template = """You are a network operations expert analyzing logs.

Context: {context}

Analyze these network logs and identify:
1. Critical issues that need immediate attention
2. Patterns (repeated events, trends)
3. Anomalies (unusual behavior)
4. Root causes and correlations

Logs:
{logs}

{format_instructions}

Provide actionable insights."""

        prompt = ChatPromptTemplate.from_template(template)

        # Combine logs
        log_text = "\n".join(logs[:100])  # Limit to avoid token limits

        # Get analysis
        formatted_prompt = prompt.format(
            context=context or "Production network environment",
            logs=log_text,
            format_instructions=parser.get_format_instructions()
        )

        response = self.llm.invoke(formatted_prompt)
        analysis = parser.parse(response.content)

        return analysis

    def find_error_patterns(self, logs: List[str]) -> Dict[str, int]:
        """
        Find common error patterns in logs.

        Args:
            logs: List of log entries

        Returns:
            Dictionary of error patterns and counts
        """
        import re
        from collections import Counter

        error_patterns = {
            "interface_down": r"%LINK-3-UPDOWN.*down",
            "ospf_neighbor_down": r"%OSPF-5-ADJCHG.*Down",
            "bgp_neighbor_down": r"%BGP-5-ADJCHANGE.*Down",
            "stp_topology_change": r"%SPANTREE.*Topology.*Change",
            "dhcp_snooping_violation": r"DHCP_SNOOPING.*violation",
            "port_security_violation": r"PSECURE.*violation",
            "authentication_failure": r"AUTH-FAIL|Authentication.*failed",
            "memory_warning": r"%SYS.*Memory",
            "cpu_high": r"%SYS.*CPU"
        }

        matches = Counter()

        for log in logs:
            for pattern_name, pattern in error_patterns.items():
                if re.search(pattern, log, re.IGNORECASE):
                    matches[pattern_name] += 1

        return dict(matches)

    def detect_security_events(self, logs: List[str]) -> List[Dict]:
        """
        Detect potential security events.

        Args:
            logs: List of log entries

        Returns:
            List of security events
        """
        import re

        security_patterns = [
            {
                "name": "Multiple login failures",
                "pattern": r"AUTH.*FAIL|LOGIN.*FAIL",
                "severity": "high"
            },
            {
                "name": "Port security violation",
                "pattern": r"PSECURE.*violation",
                "severity": "medium"
            },
            {
                "name": "DHCP snooping violation",
                "pattern": r"DHCP_SNOOPING.*violation",
                "severity": "medium"
            },
            {
                "name": "Unauthorized access attempt",
                "pattern": r"UNAUTH|UNAUTHORIZED",
                "severity": "high"
            },
            {
                "name": "Config change without authorization",
                "pattern": r"SYS-5-CONFIG_I.*CONSOLE",
                "severity": "medium"
            }
        ]

        events = []

        for log in logs:
            for pattern_info in security_patterns:
                if re.search(pattern_info["pattern"], log, re.IGNORECASE):
                    # Extract timestamp and device
                    timestamp_match = re.search(r'\d{2}:\d{2}:\d{2}', log)
                    timestamp = timestamp_match.group(0) if timestamp_match else "Unknown"

                    events.append({
                        "event": pattern_info["name"],
                        "severity": pattern_info["severity"],
                        "timestamp": timestamp,
                        "log": log[:100]
                    })

        return events

    def correlate_events(self, logs: List[str]) -> List[Dict]:
        """
        Find correlated events that might indicate a problem.

        Args:
            logs: List of log entries

        Returns:
            List of correlated event groups
        """
        import re
        from collections import defaultdict
        from datetime import datetime, timedelta

        # Group logs by device
        device_logs = defaultdict(list)

        for log in logs:
            # Extract device/IP
            ip_match = re.search(r'\d+\.\d+\.\d+\.\d+', log)
            if ip_match:
                device = ip_match.group(0)
                device_logs[device].append(log)

        # Look for correlated issues
        correlations = []

        for device, device_log_list in device_logs.items():
            # Check for interface flapping
            link_changes = [log for log in device_log_list if "UPDOWN" in log or "LINEPROTO" in log]
            if len(link_changes) >= 3:
                correlations.append({
                    "type": "Interface Flapping",
                    "device": device,
                    "count": len(link_changes),
                    "description": f"Interface on {device} changed state {len(link_changes)} times"
                })

            # Check for routing protocol flapping
            ospf_changes = [log for log in device_log_list if "OSPF" in log and ("Down" in log or "Up" in log)]
            if len(ospf_changes) >= 2:
                correlations.append({
                    "type": "OSPF Neighbor Instability",
                    "device": device,
                    "count": len(ospf_changes),
                    "description": f"OSPF neighbor on {device} unstable ({len(ospf_changes)} changes)"
                })

        return correlations


def create_sample_logs():
    """Generate sample network logs for testing."""
    return [
        "Jan 18 10:15:32 10.1.1.1 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down",
        "Jan 18 10:15:33 10.1.1.1 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down",
        "Jan 18 10:15:35 10.1.1.2 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 on GigabitEthernet0/0 from FULL to DOWN, Neighbor Down: Interface down",
        "Jan 18 10:16:01 10.1.1.5 %DHCP_SNOOPING-4-DHCP_SNOOPING_RATE_LIMIT_EXCEEDED: DHCP snooping rate limit exceeded on Gi0/5",
        "Jan 18 10:16:15 10.1.1.3 %SEC-6-IPACCESSLOGP: list 101 denied tcp 192.168.1.100(45678) -> 10.1.1.10(22), 1 packet",
        "Jan 18 10:17:22 10.1.1.1 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up",
        "Jan 18 10:17:23 10.1.1.1 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up",
        "Jan 18 10:17:30 10.1.1.2 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 on GigabitEthernet0/0 from LOADING to FULL, Loading Done",
        "Jan 18 10:20:45 10.1.1.4 %PARSER-4-BADCFG: Bad configuration command: interface g0/99",
        "Jan 18 10:21:10 10.1.1.3 %SYS-2-MALLOCFAIL: Memory allocation of 65536 bytes failed",
        "Jan 18 10:22:33 10.1.1.6 %SEC_LOGIN-4-LOGIN_FAILED: Login failed for user admin from 192.168.1.200",
        "Jan 18 10:22:45 10.1.1.6 %SEC_LOGIN-4-LOGIN_FAILED: Login failed for user admin from 192.168.1.200",
        "Jan 18 10:22:58 10.1.1.6 %SEC_LOGIN-4-LOGIN_FAILED: Login failed for user admin from 192.168.1.200",
        "Jan 18 10:25:00 10.1.1.7 %SPANTREE-2-ROOTCHANGE: Root Changed for VLAN 10: New Root Port is Gi0/1",
        "Jan 18 10:30:15 10.1.1.1 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down",
        "Jan 18 10:30:16 10.1.1.1 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down",
        "Jan 18 10:30:20 10.1.1.2 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 on GigabitEthernet0/0 from FULL to DOWN, Neighbor Down: Interface down",
        "Jan 18 10:35:12 10.1.1.8 %BGP-5-ADJCHANGE: neighbor 203.0.113.1 Down - BGP Notification sent",
        "Jan 18 10:40:05 10.1.1.5 %PORT_SECURITY-2-PSECURE_VIOLATION: Security violation on port Gi0/10, MAC address 0000.1234.5678",
        "Jan 18 10:45:22 10.1.1.9 %SYS-4-CONFIG_NEWER_CFGFILE_DETECTED: Config file is newer than current config"
    ]


def main():
    """Demo the log analyzer."""
    print("="*60)
    print("AI-Powered Network Log Analysis")
    print("="*60)

    try:
        # Create analyzer
        analyzer = NetworkLogAnalyzer()

        # Get sample logs
        logs = create_sample_logs()

        print(f"\nAnalyzing {len(logs)} log entries...")

        # Example 1: Pattern detection
        print("\n" + "="*60)
        print("Example 1: Error Pattern Detection")
        print("="*60)

        patterns = analyzer.find_error_patterns(logs)

        print("\nDetected patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern:30s}: {count} occurrences")

        # Example 2: Security event detection
        print("\n" + "="*60)
        print("Example 2: Security Event Detection")
        print("="*60)

        security_events = analyzer.detect_security_events(logs)

        print(f"\nFound {len(security_events)} security events:")
        for event in security_events[:5]:  # Show first 5
            print(f"\n  [{event['severity'].upper()}] {event['event']}")
            print(f"  Time: {event['timestamp']}")
            print(f"  Log: {event['log'][:80]}...")

        # Example 3: Event correlation
        print("\n" + "="*60)
        print("Example 3: Event Correlation")
        print("="*60)

        correlations = analyzer.correlate_events(logs)

        print(f"\nFound {len(correlations)} correlated event groups:")
        for correlation in correlations:
            print(f"\n  Type: {correlation['type']}")
            print(f"  Device: {correlation['device']}")
            print(f"  Count: {correlation['count']}")
            print(f"  Description: {correlation['description']}")

        # Example 4: AI-powered analysis
        print("\n" + "="*60)
        print("Example 4: AI-Powered Analysis")
        print("="*60)

        context = "Production network with 10 switches, 5 routers, OSPF + BGP"
        print("\nRunning AI analysis (this may take a moment)...")

        analysis = analyzer.analyze_logs(logs, context)

        print(f"\n{'='*60}")
        print("ANALYSIS REPORT")
        print(f"{'='*60}")

        print(f"\nSummary:")
        print(f"{analysis.summary}")

        print(f"\nIssues Found ({len(analysis.issues)}):")
        for i, issue in enumerate(analysis.issues, 1):
            print(f"\n{i}. [{issue.severity.upper()}] {issue.category}")
            print(f"   Issue: {issue.issue}")
            print(f"   Devices: {', '.join(issue.affected_devices[:3])}")
            print(f"   First seen: {issue.first_seen}")
            print(f"   Count: {issue.count}")
            print(f"   Recommendation: {issue.recommendation}")

        print(f"\nPatterns Detected ({len(analysis.patterns)}):")
        for i, pattern in enumerate(analysis.patterns, 1):
            print(f"{i}. {pattern}")

        if analysis.anomalies:
            print(f"\nAnomalies ({len(analysis.anomalies)}):")
            for i, anomaly in enumerate(analysis.anomalies, 1):
                print(f"{i}. {anomaly}")

        print("\n" + "="*60)
        print("✓ Log analysis completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
