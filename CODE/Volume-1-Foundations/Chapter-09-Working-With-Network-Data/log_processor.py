#!/usr/bin/env python3
"""
Log Processor - AI-Powered Network Log Analysis

Process and analyze network logs using AI to identify patterns, anomalies, and issues.

From: AI for Networking Engineers - Volume 1, Chapter 9
Author: Eduard Dulharu

Usage:
    from log_processor import LogProcessor

    processor = LogProcessor()
    analysis = processor.analyze_logs(log_file="syslog.txt")
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import Counter


class LogSeverity(str, Enum):
    """Log severity levels."""
    EMERGENCY = "emergency"
    ALERT = "alert"
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFO = "info"
    DEBUG = "debug"


class LogCategory(str, Enum):
    """Log categories."""
    INTERFACE = "interface"
    ROUTING = "routing"
    SECURITY = "security"
    SYSTEM = "system"
    CONFIG = "config"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


@dataclass
class ParsedLog:
    """Parsed log entry."""
    timestamp: Optional[datetime]
    severity: LogSeverity
    facility: Optional[str]
    message: str
    category: LogCategory
    device: Optional[str]
    interface: Optional[str]
    raw_log: str


class LogProcessor:
    """
    Process and analyze network device logs using AI.

    Features:
    - Parse syslog format
    - Classify by severity and category
    - Identify patterns and anomalies
    - Extract actionable insights
    - Timeline analysis
    """

    # Common log patterns
    SYSLOG_PATTERN = re.compile(
        r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+)\s+'
        r'(?P<device>\S+)\s+'
        r'%?(?P<facility>\w+)-(?P<severity>\d+)-(?P<mnemonic>\w+):\s*'
        r'(?P<message>.+)'
    )

    # Severity mapping (syslog numeric to enum)
    SEVERITY_MAP = {
        '0': LogSeverity.EMERGENCY,
        '1': LogSeverity.ALERT,
        '2': LogSeverity.CRITICAL,
        '3': LogSeverity.ERROR,
        '4': LogSeverity.WARNING,
        '5': LogSeverity.NOTICE,
        '6': LogSeverity.INFO,
        '7': LogSeverity.DEBUG
    }

    # Category detection keywords
    CATEGORY_KEYWORDS = {
        LogCategory.INTERFACE: ['interface', 'link', 'updown', 'lineproto'],
        LogCategory.ROUTING: ['ospf', 'bgp', 'eigrp', 'rip', 'route', 'neighbor', 'adjacency'],
        LogCategory.SECURITY: ['acl', 'denied', 'sec', 'auth', 'login', 'aaa'],
        LogCategory.SYSTEM: ['reload', 'restart', 'memory', 'cpu', 'temperature'],
        LogCategory.CONFIG: ['config', 'nvram', 'configured'],
        LogCategory.HARDWARE: ['power', 'fan', 'module', 'transceiver']
    }

    def __init__(self):
        """Initialize log processor."""
        self.parsed_logs: List[ParsedLog] = []

    def parse_log_file(self, log_file: str) -> List[ParsedLog]:
        """
        Parse log file.

        Args:
            log_file: Path to log file

        Returns:
            List of parsed log entries
        """
        parsed_logs = []

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parsed = self._parse_log_line(line)
                    if parsed:
                        parsed_logs.append(parsed)

        except FileNotFoundError:
            print(f"Error: Log file not found: {log_file}")
            return []

        self.parsed_logs = parsed_logs
        return parsed_logs

    def _parse_log_line(self, log_line: str) -> Optional[ParsedLog]:
        """
        Parse individual log line.

        Args:
            log_line: Raw log line

        Returns:
            ParsedLog object or None if parsing failed
        """
        # Try syslog format
        match = self.SYSLOG_PATTERN.search(log_line)

        if match:
            timestamp_str = match.group('timestamp')
            device = match.group('device')
            facility = match.group('facility')
            severity_num = match.group('severity')
            message = match.group('message')

            # Parse timestamp (add current year as syslog doesn't include it)
            try:
                current_year = datetime.now().year
                timestamp = datetime.strptime(
                    f"{current_year} {timestamp_str}",
                    "%Y %b %d %H:%M:%S"
                )
            except ValueError:
                timestamp = None

            # Map severity
            severity = self.SEVERITY_MAP.get(severity_num, LogSeverity.INFO)

            # Detect category
            category = self._detect_category(message, facility)

            # Extract interface if present
            interface = self._extract_interface(message)

            return ParsedLog(
                timestamp=timestamp,
                severity=severity,
                facility=facility,
                message=message,
                category=category,
                device=device,
                interface=interface,
                raw_log=log_line
            )

        # Fallback: basic parsing
        return ParsedLog(
            timestamp=None,
            severity=LogSeverity.INFO,
            facility=None,
            message=log_line,
            category=LogCategory.UNKNOWN,
            device=None,
            interface=None,
            raw_log=log_line
        )

    def _detect_category(self, message: str, facility: Optional[str]) -> LogCategory:
        """
        Detect log category from message content.

        Args:
            message: Log message
            facility: Syslog facility

        Returns:
            LogCategory enum
        """
        message_lower = message.lower()

        # Check facility first
        if facility:
            facility_lower = facility.lower()
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                if facility_lower in [k.lower() for k in keywords]:
                    return category

        # Check message content
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return category

        return LogCategory.UNKNOWN

    def _extract_interface(self, message: str) -> Optional[str]:
        """
        Extract interface name from log message.

        Args:
            message: Log message

        Returns:
            Interface name or None
        """
        # Common interface patterns
        interface_patterns = [
            r'Interface\s+([A-Za-z]+\d+(?:/\d+)*(?:\.\d+)?)',
            r'([A-Za-z]+Ethernet\d+(?:/\d+)*)',
            r'(Gi\d+(?:/\d+)*)',
            r'(Fa\d+(?:/\d+)*)',
            r'(Te\d+(?:/\d+)*)'
        ]

        for pattern in interface_patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)

        return None

    def analyze_logs(self) -> Dict[str, Any]:
        """
        Analyze parsed logs for insights.

        Returns:
            Analysis results with statistics and findings
        """
        if not self.parsed_logs:
            return {"error": "No logs to analyze"}

        total_logs = len(self.parsed_logs)

        # Severity distribution
        severity_counts = Counter(log.severity.value for log in self.parsed_logs)

        # Category distribution
        category_counts = Counter(log.category.value for log in self.parsed_logs)

        # Device statistics
        device_counts = Counter(
            log.device for log in self.parsed_logs if log.device
        )

        # Critical/error logs
        critical_logs = [
            log for log in self.parsed_logs
            if log.severity in [LogSeverity.EMERGENCY, LogSeverity.ALERT, LogSeverity.CRITICAL]
        ]

        error_logs = [
            log for log in self.parsed_logs
            if log.severity == LogSeverity.ERROR
        ]

        # Interface issues
        interface_issues = self._identify_interface_issues()

        # Routing issues
        routing_issues = self._identify_routing_issues()

        # Time-based patterns
        hourly_distribution = self._calculate_hourly_distribution()

        return {
            "summary": {
                "total_logs": total_logs,
                "critical_count": len(critical_logs),
                "error_count": len(error_logs),
                "warning_count": severity_counts.get(LogSeverity.WARNING.value, 0),
                "devices": len(device_counts)
            },
            "severity_distribution": dict(severity_counts),
            "category_distribution": dict(category_counts),
            "top_devices": dict(device_counts.most_common(5)),
            "critical_logs": [
                {
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "device": log.device,
                    "severity": log.severity.value,
                    "message": log.message
                }
                for log in critical_logs[:10]
            ],
            "interface_issues": interface_issues,
            "routing_issues": routing_issues,
            "hourly_distribution": hourly_distribution
        }

    def _identify_interface_issues(self) -> List[Dict[str, Any]]:
        """
        Identify interface-related issues.

        Returns:
            List of interface issues
        """
        issues = []

        # Find interface state changes
        interface_logs = [
            log for log in self.parsed_logs
            if log.category == LogCategory.INTERFACE
        ]

        # Count flaps (interface going up/down repeatedly)
        interface_changes = Counter(log.interface for log in interface_logs if log.interface)

        for interface, count in interface_changes.most_common(10):
            if count >= 2:  # Multiple state changes
                issues.append({
                    "interface": interface,
                    "event_count": count,
                    "issue_type": "potential_flapping" if count >= 5 else "state_changes",
                    "severity": "high" if count >= 5 else "medium"
                })

        return issues

    def _identify_routing_issues(self) -> List[Dict[str, Any]]:
        """
        Identify routing protocol issues.

        Returns:
            List of routing issues
        """
        issues = []

        routing_logs = [
            log for log in self.parsed_logs
            if log.category == LogCategory.ROUTING
        ]

        # Neighbor down events
        neighbor_down = [
            log for log in routing_logs
            if 'down' in log.message.lower() and 'neighbor' in log.message.lower()
        ]

        if neighbor_down:
            issues.append({
                "issue_type": "neighbor_adjacency_loss",
                "count": len(neighbor_down),
                "severity": "high",
                "sample_messages": [log.message for log in neighbor_down[:3]]
            })

        # Route flaps
        route_changes = [
            log for log in routing_logs
            if any(word in log.message.lower() for word in ['add', 'delete', 'change'])
        ]

        if len(route_changes) > 10:
            issues.append({
                "issue_type": "route_instability",
                "count": len(route_changes),
                "severity": "medium",
                "description": "High number of routing table changes"
            })

        return issues

    def _calculate_hourly_distribution(self) -> Dict[int, int]:
        """
        Calculate log distribution by hour.

        Returns:
            Dictionary mapping hour (0-23) to log count
        """
        hourly = Counter()

        for log in self.parsed_logs:
            if log.timestamp:
                hourly[log.timestamp.hour] += 1

        return dict(sorted(hourly.items()))

    def generate_summary_report(self) -> str:
        """
        Generate human-readable summary report.

        Returns:
            Formatted report string
        """
        if not self.parsed_logs:
            return "No logs to analyze."

        analysis = self.analyze_logs()
        summary = analysis['summary']

        report = f"""
LOG ANALYSIS REPORT
{'='*60}

Summary:
  Total Logs: {summary['total_logs']}
  Critical:   {summary['critical_count']}
  Errors:     {summary['error_count']}
  Warnings:   {summary['warning_count']}
  Devices:    {summary['devices']}

Severity Distribution:
"""

        for severity, count in analysis['severity_distribution'].items():
            percentage = (count / summary['total_logs']) * 100
            report += f"  {severity:12s}: {count:5d} ({percentage:5.1f}%)\n"

        report += "\nCategory Distribution:\n"
        for category, count in analysis['category_distribution'].items():
            percentage = (count / summary['total_logs']) * 100
            report += f"  {category:12s}: {count:5d} ({percentage:5.1f}%)\n"

        # Interface issues
        if analysis['interface_issues']:
            report += "\nInterface Issues:\n"
            for issue in analysis['interface_issues'][:5]:
                report += f"  ⚠ {issue['interface']}: {issue['event_count']} events ({issue['issue_type']})\n"

        # Routing issues
        if analysis['routing_issues']:
            report += "\nRouting Issues:\n"
            for issue in analysis['routing_issues']:
                report += f"  ⚠ {issue['issue_type']}: {issue['count']} occurrences\n"

        # Critical logs
        if analysis['critical_logs']:
            report += f"\nTop Critical Logs:\n"
            for log in analysis['critical_logs'][:5]:
                report += f"  [{log['device']}] {log['message'][:60]}...\n"

        return report

    def filter_logs(
        self,
        severity: Optional[LogSeverity] = None,
        category: Optional[LogCategory] = None,
        device: Optional[str] = None,
        interface: Optional[str] = None
    ) -> List[ParsedLog]:
        """
        Filter logs by criteria.

        Args:
            severity: Filter by severity level
            category: Filter by category
            device: Filter by device name
            interface: Filter by interface

        Returns:
            Filtered list of logs
        """
        filtered = self.parsed_logs

        if severity:
            filtered = [log for log in filtered if log.severity == severity]

        if category:
            filtered = [log for log in filtered if log.category == category]

        if device:
            filtered = [log for log in filtered if log.device == device]

        if interface:
            filtered = [log for log in filtered if log.interface == interface]

        return filtered


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Log Processor Demo
    ========================================
    Process and analyze network logs
    ========================================
    """)

    # Create sample log file
    sample_logs = """
Jan 18 10:15:42 CORE-RTR-01 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
Jan 18 10:15:43 CORE-RTR-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
Jan 18 10:16:01 CORE-RTR-01 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.2 on GigabitEthernet0/1 from FULL to DOWN, Neighbor Down: Interface down or detached
Jan 18 10:16:15 CORE-RTR-01 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
Jan 18 10:16:16 CORE-RTR-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up
Jan 18 10:16:45 CORE-RTR-01 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.2 on GigabitEthernet0/1 from LOADING to FULL, Loading Done
Jan 18 10:17:22 CORE-RTR-01 %SEC-6-IPACCESSLOGP: list 101 denied tcp 203.0.113.50(4521) -> 10.1.1.100(22), 1 packet
Jan 18 10:18:05 CORE-RTR-01 %SYS-5-CONFIG_I: Configured from console by admin on vty0 (10.0.0.5)
Jan 18 10:20:33 CORE-RTR-02 %BGP-3-NOTIFICATION: sent to neighbor 203.0.113.100 4/0 (hold time expired) 0 bytes
Jan 18 10:21:15 CORE-RTR-02 %BGP-5-ADJCHANGE: neighbor 203.0.113.100 Down BGP Notification sent
Jan 18 10:25:42 CORE-RTR-01 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
Jan 18 10:25:43 CORE-RTR-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
"""

    # Write sample log file
    sample_file = "sample_network.log"
    with open(sample_file, 'w') as f:
        f.write(sample_logs)

    # Test 1: Parse logs
    print("\nTest 1: Parse Log File")
    print("-" * 60)

    processor = LogProcessor()
    parsed_logs = processor.parse_log_file(sample_file)

    print(f"Parsed {len(parsed_logs)} log entries")
    print(f"\nFirst 3 parsed logs:")
    for log in parsed_logs[:3]:
        print(f"  [{log.severity.value:8s}] [{log.category.value:10s}] {log.message[:50]}...")

    # Test 2: Analyze logs
    print("\n\nTest 2: Log Analysis")
    print("-" * 60)

    analysis = processor.analyze_logs()

    print(f"\nSummary:")
    for key, value in analysis['summary'].items():
        print(f"  {key:15s}: {value}")

    print(f"\nSeverity Distribution:")
    for severity, count in analysis['severity_distribution'].items():
        print(f"  {severity:12s}: {count}")

    print(f"\nCategory Distribution:")
    for category, count in analysis['category_distribution'].items():
        print(f"  {category:12s}: {count}")

    # Test 3: Interface issues
    print("\n\nTest 3: Interface Issues")
    print("-" * 60)

    if analysis['interface_issues']:
        for issue in analysis['interface_issues']:
            print(f"  Interface: {issue['interface']}")
            print(f"  Events: {issue['event_count']}")
            print(f"  Type: {issue['issue_type']}")
            print()

    # Test 4: Filtering
    print("\n\nTest 4: Filter Critical Logs")
    print("-" * 60)

    critical = processor.filter_logs(severity=LogSeverity.CRITICAL)
    print(f"Found {len(critical)} critical logs")

    routing_logs = processor.filter_logs(category=LogCategory.ROUTING)
    print(f"Found {len(routing_logs)} routing logs")

    # Test 5: Summary report
    print("\n\nTest 5: Summary Report")
    print("-" * 60)

    report = processor.generate_summary_report()
    print(report)

    # Clean up
    import os
    os.remove(sample_file)

    print("\n✅ Demo complete!")
    print("\n💡 Log Processing Capabilities:")
    print("  - Parse syslog format")
    print("  - Classify by severity and category")
    print("  - Identify interface flapping")
    print("  - Detect routing issues")
    print("  - Filter and search logs")
    print("  - Generate summary reports")

    print("\n⚠️  Production Tips:")
    print("  ☐ Use with AI for deeper analysis")
    print("  ☐ Set up alerting for critical logs")
    print("  ☐ Track patterns over time")
    print("  ☐ Correlate with network topology")
    print("  ☐ Archive logs for compliance")
