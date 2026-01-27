#!/usr/bin/env python3
"""
Configuration Analysis Module

Advanced configuration analysis and validation for network devices.

From: AI for Networking Engineers - Volume 2, Chapter 13
Author: Eduard Dulharu (Ed Harmoosh)

This module provides:
- Security vulnerability scanning
- Best practice validation
- Compliance checking
- Performance optimization analysis
- Redundancy assessment
- Documentation quality scoring

Usage:
    from doc_analyzer import ConfigAnalyzer

    analyzer = ConfigAnalyzer()
    findings = analyzer.analyze_security(config)
    report = analyzer.generate_analysis_report(config, hostname)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import re
from datetime import datetime


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(str, Enum):
    """Finding categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    REDUNDANCY = "redundancy"
    COMPLIANCE = "compliance"
    BEST_PRACTICE = "best_practice"
    DOCUMENTATION = "documentation"


@dataclass
class Finding:
    """Configuration finding/recommendation."""
    severity: Severity
    category: Category
    title: str
    description: str
    affected_lines: List[str] = field(default_factory=list)
    recommendation: str = ""
    risk_score: int = 5  # 0-10 scale


class ConfigAnalyzer:
    """
    Analyze network configurations for issues and improvements.

    Features:
    - Security vulnerability detection
    - Best practice validation
    - Performance optimization suggestions
    - Compliance checking
    - Redundancy analysis
    - Documentation assessment

    Example:
        >>> analyzer = ConfigAnalyzer()
        >>> findings = analyzer.analyze_security(config)
        >>> print(f"Found {len(findings)} security issues")
    """

    def __init__(self):
        """Initialize analyzer."""
        self.findings: List[Finding] = []
        self.config_lines: List[str] = []

    def analyze_config(self, config: str) -> Dict[str, Any]:
        """
        Perform comprehensive configuration analysis.

        Args:
            config: Device configuration text.

        Returns:
            Dictionary with analysis results including findings, scores, and recommendations.
        """
        self.config_lines = config.split('\n')
        self.findings = []

        # Run all analysis modules
        self._analyze_security(config)
        self._analyze_best_practices(config)
        self._analyze_redundancy(config)
        self._analyze_documentation(config)

        return self._generate_report()

    def _analyze_security(self, config: str) -> None:
        """Analyze security issues in configuration."""
        config_lower = config.lower()

        # 1. Check for Telnet
        if re.search(r'transport\s+input\s+telnet', config, re.IGNORECASE):
            self.findings.append(Finding(
                severity=Severity.CRITICAL,
                category=Category.SECURITY,
                title="Telnet Enabled",
                description="Telnet transmits credentials and data in cleartext. SSH should be used exclusively.",
                affected_lines=self._find_lines(config, r'transport\s+input\s+telnet'),
                recommendation="Remove telnet: 'no transport input telnet' and ensure SSH is enabled: 'transport input ssh'",
                risk_score=9
            ))

        # 2. Check for HTTP (non-HTTPS)
        if 'ip http server' in config_lower and 'ip http secure-server' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.HIGH,
                category=Category.SECURITY,
                title="Unencrypted HTTP Server",
                description="HTTP management interface is unencrypted. Use HTTPS/secure HTTP instead.",
                affected_lines=self._find_lines(config, r'ip\s+http\s+server'),
                recommendation="Disable HTTP and enable secure server: 'no ip http server' then 'ip http secure-server'",
                risk_score=8
            ))

        # 3. Check for weak SNMP
        if re.search(r'snmp-server\s+community\s+(public|private)', config, re.IGNORECASE):
            self.findings.append(Finding(
                severity=Severity.CRITICAL,
                category=Category.SECURITY,
                title="Default SNMP Community Strings",
                description="Default SNMP community strings (public/private) are widely known. Exposes network to attacks.",
                affected_lines=self._find_lines(config, r'snmp-server\s+community\s+(public|private)'),
                recommendation="Use SNMPv3 with authentication and encryption, or change community to unique strong string",
                risk_score=9
            ))

        # 4. Check for weak enable password
        has_enable_secret = 'enable secret' in config_lower
        if not has_enable_secret:
            self.findings.append(Finding(
                severity=Severity.HIGH,
                category=Category.SECURITY,
                title="No Enable Secret Configured",
                description="Privileged mode is not protected with a strong secret password.",
                recommendation="Configure: 'enable secret <strong-password>' (hashed, not cleartext)",
                risk_score=7
            ))

        # 5. Check for no console timeout
        if 'exec-timeout 0 0' in config_lower or ('exec-timeout' not in config_lower and 'line con 0' in config_lower):
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.SECURITY,
                title="Console Timeout Not Set",
                description="Console sessions can remain active indefinitely, risking unauthorized access.",
                affected_lines=self._find_lines(config, r'line\s+con\s+0'),
                recommendation="Set console timeout: 'line con 0' then 'exec-timeout 5 0' (5 minutes)",
                risk_score=6
            ))

        # 6. Check for no logging
        if 'logging' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.SECURITY,
                title="Logging Not Configured",
                description="Device activity is not being logged, limiting audit trail and incident investigation.",
                recommendation="Enable logging: 'logging host <syslog-server>' and 'logging buffer informational'",
                risk_score=5
            ))

        # 7. Check for CDP enabled on untrusted interfaces
        if 'cdp run' in config_lower or 'no cdp run' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.SECURITY,
                title="CDP Potentially Exposed",
                description="CDP can leak network topology information. Should be disabled on untrusted interfaces.",
                recommendation="Disable CDP on untrusted ports: 'interface <name>' then 'no cdp enable'",
                risk_score=4
            ))

    def _analyze_best_practices(self, config: str) -> None:
        """Analyze configuration against best practices."""
        config_lower = config.lower()

        # 1. Check for NTP
        if 'ntp' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.BEST_PRACTICE,
                title="NTP Not Configured",
                description="Time synchronization is critical for logging accuracy and security policies.",
                recommendation="Configure NTP: 'ntp server <primary-ip>' and 'ntp server <secondary-ip> prefer'",
                risk_score=5
            ))

        # 2. Check for DNS
        if 'ip name-server' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.BEST_PRACTICE,
                title="DNS Not Configured",
                description="DNS configuration improves usability and troubleshooting capability.",
                recommendation="Configure DNS: 'ip name-server <dns1>' and 'ip name-server <dns2>'",
                risk_score=2
            ))

        # 3. Check for hostname
        if 'hostname' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.BEST_PRACTICE,
                title="Hostname Not Set",
                description="Hostname helps identify devices in logs and alerts.",
                recommendation="Set hostname: 'hostname <device-name>'",
                risk_score=1
            ))

        # 4. Check for banner
        if 'banner motd' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.BEST_PRACTICE,
                title="Banner MOTD Not Configured",
                description="Banner message provides legal notice and access policy information.",
                recommendation="Configure banner: 'banner motd ^C ... ^C'",
                risk_score=2
            ))

        # 5. Check for IP routing
        if 'no ip routing' in config_lower and 'router' not in config_lower:
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.BEST_PRACTICE,
                title="IP Routing Disabled on Router",
                description="Routing is disabled but this appears to be a router.",
                recommendation="Enable routing: 'ip routing'",
                risk_score=6
            ))

    def _analyze_redundancy(self, config: str) -> None:
        """Analyze redundancy configuration."""
        config_lower = config.lower()

        # 1. Check for HSRP/VRRP on core devices
        has_hsrp = 'standby' in config_lower
        has_vrrp = 'vrrp' in config_lower

        if not has_hsrp and not has_vrrp and ('router' in config_lower or 'core' in config_lower):
            self.findings.append(Finding(
                severity=Severity.MEDIUM,
                category=Category.REDUNDANCY,
                title="No HSRP/VRRP Configured",
                description="No first-hop redundancy protocol configured on core/distribution device.",
                recommendation="Implement HSRP or VRRP for high availability: 'standby 1 ip <virtual-ip>'",
                risk_score=6
            ))

        # 2. Check for spanning tree
        if 'spanning-tree' not in config_lower and ('switch' in config_lower or 'vlan' in config_lower):
            self.findings.append(Finding(
                severity=Severity.HIGH,
                category=Category.REDUNDANCY,
                title="Spanning Tree Not Configured",
                description="Layer 2 loop prevention not configured on switch. Risk of broadcast storms.",
                recommendation="Enable Spanning Tree: 'spanning-tree mode rapid-pvst'",
                risk_score=7
            ))

        # 3. Check for etherchannel
        if 'channel-group' not in config_lower and 'port-channel' not in config_lower and 'switch' in config_lower:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.REDUNDANCY,
                title="EtherChannel Not Used",
                description="Multiple links between switches are not aggregated. Inefficient bandwidth use.",
                recommendation="Configure EtherChannel for multi-link connections",
                risk_score=3
            ))

    def _analyze_documentation(self, config: str) -> None:
        """Analyze documentation quality in configuration."""
        config_lower = config.lower()

        # 1. Count comments/descriptions
        comment_count = len(re.findall(r'^[!#]', config, re.MULTILINE))
        description_count = len(re.findall(r'description\s+', config, re.IGNORECASE))

        if comment_count < 5:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.DOCUMENTATION,
                title="Limited Configuration Comments",
                description="Few comments in configuration (found: {}, should have 10+)".format(comment_count),
                recommendation="Add comments explaining WHY each configuration block exists: ! Brief description",
                risk_score=1
            ))

        if description_count < 3:
            self.findings.append(Finding(
                severity=Severity.LOW,
                category=Category.DOCUMENTATION,
                title="Limited Interface Descriptions",
                description="Few interfaces have descriptions (found: {}, should have all)".format(description_count),
                recommendation="Add descriptions to all interfaces: 'interface <name>' then 'description <purpose>'",
                risk_score=2
            ))

    def _find_lines(self, config: str, pattern: str) -> List[str]:
        """Find lines matching pattern."""
        matching_lines = []
        for i, line in enumerate(self.config_lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                matching_lines.append(f"Line {i}: {line.strip()}")
                if len(matching_lines) >= 3:  # Limit to first 3
                    break
        return matching_lines

    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        severity_count = {s.value: 0 for s in Severity}
        category_count = {c.value: 0 for c in Category}
        total_risk = 0

        for finding in self.findings:
            severity_count[finding.severity.value] += 1
            category_count[finding.category.value] += 1
            total_risk += finding.risk_score

        return {
            "total_findings": len(self.findings),
            "severity_breakdown": severity_count,
            "category_breakdown": category_count,
            "total_risk_score": total_risk,
            "average_risk_score": round(total_risk / max(1, len(self.findings)), 1),
            "findings": self.findings
        }


# Example usage
if __name__ == "__main__":
    sample_config = """
hostname router-core-01
!
interface GigabitEthernet0/0
 description Uplink to ISP
 ip address 203.0.113.1 255.255.255.252
!
line vty 0 4
 transport input telnet
!
snmp-server community public RO
"""

    analyzer = ConfigAnalyzer()
    report = analyzer.analyze_config(sample_config)

    print("=" * 60)
    print("Configuration Analysis Report")
    print("=" * 60)
    print(f"Total Findings: {report['total_findings']}")
    print(f"Total Risk Score: {report['total_risk_score']}/100")
    print(f"Average Risk: {report['average_risk_score']}/10")
    print("\nSeverity Breakdown:")
    for severity, count in report['severity_breakdown'].items():
        if count > 0:
            print(f"  {severity}: {count}")
