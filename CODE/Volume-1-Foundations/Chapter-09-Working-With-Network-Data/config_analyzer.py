#!/usr/bin/env python3
"""
Config Analyzer - AI-Powered Network Configuration Analysis

Comprehensive configuration analysis using AI to identify issues, best practices,
and optimization opportunities.

From: AI for Networking Engineers - Volume 1, Chapter 9
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from config_analyzer import ConfigAnalyzer

    analyzer = ConfigAnalyzer()
    analysis = analyzer.analyze_config(config_text, device_type="cisco_ios")
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class FindingSeverity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    """Finding categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    REDUNDANCY = "redundancy"
    COMPLIANCE = "compliance"
    BEST_PRACTICE = "best_practice"
    CONFIGURATION = "configuration"


@dataclass
class Finding:
    """Single configuration finding."""
    severity: FindingSeverity
    category: FindingCategory
    title: str
    description: str
    affected_lines: List[str]
    recommendation: str
    risk_score: int  # 0-10


class ConfigAnalyzer:
    """
    Analyze network configurations for issues and improvements.

    Features:
    - Security vulnerability detection
    - Best practice validation
    - Performance optimization suggestions
    - Compliance checking
    - Redundancy analysis
    """

    def __init__(self):
        """Initialize analyzer."""
        self.findings: List[Finding] = []

    def analyze_config(
        self,
        config: str,
        device_type: str = "cisco_ios",
        check_compliance: bool = True,
        check_security: bool = True,
        check_best_practices: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze configuration comprehensively.

        Args:
            config: Configuration text
            device_type: Device platform
            check_compliance: Run compliance checks
            check_security: Run security checks
            check_best_practices: Run best practice checks

        Returns:
            Analysis results with findings and recommendations
        """
        self.findings = []

        # Run analysis modules
        if check_security:
            self._check_security_issues(config, device_type)

        if check_best_practices:
            self._check_best_practices(config, device_type)

        if check_compliance:
            self._check_compliance(config, device_type)

        # Additional checks
        self._check_redundancy(config, device_type)
        self._check_performance(config, device_type)
        self._check_documentation(config)

        # Generate summary
        return self._generate_analysis_report()

    def _check_security_issues(self, config: str, device_type: str) -> None:
        """
        Check for security vulnerabilities.

        Args:
            config: Configuration text
            device_type: Device platform
        """
        config_lower = config.lower()

        # Check 1: Telnet enabled
        if 'transport input telnet' in config_lower or 'line vty' in config_lower:
            # Check if telnet is explicitly allowed
            if re.search(r'transport\s+input\s+telnet', config, re.IGNORECASE):
                self.findings.append(Finding(
                    severity=FindingSeverity.CRITICAL,
                    category=FindingCategory.SECURITY,
                    title="Telnet Enabled",
                    description="Telnet transmits credentials in cleartext",
                    affected_lines=self._extract_matching_lines(config, r'transport\s+input\s+telnet'),
                    recommendation="Use SSH only: 'transport input ssh'",
                    risk_score=9
                ))

        # Check 2: Weak SNMP
        if re.search(r'snmp-server\s+community\s+\w+\s+RW', config, re.IGNORECASE):
            self.findings.append(Finding(
                severity=FindingSeverity.HIGH,
                category=FindingCategory.SECURITY,
                title="SNMP Read-Write Community Detected",
                description="SNMP v1/v2c with RW access is insecure",
                affected_lines=self._extract_matching_lines(config, r'snmp-server\s+community.*RW'),
                recommendation="Use SNMPv3 with authentication and encryption",
                risk_score=8
            ))

        # Check for public/private default communities
        if re.search(r'snmp-server\s+community\s+(public|private)', config, re.IGNORECASE):
            self.findings.append(Finding(
                severity=FindingSeverity.CRITICAL,
                category=FindingCategory.SECURITY,
                title="Default SNMP Community Strings",
                description="Using default SNMP community strings (public/private)",
                affected_lines=self._extract_matching_lines(config, r'snmp-server\s+community\s+(public|private)'),
                recommendation="Change to unique community strings",
                risk_score=10
            ))

        # Check 3: HTTP server enabled
        if 'ip http server' in config_lower and 'ip http secure-server' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.HIGH,
                category=FindingCategory.SECURITY,
                title="HTTP Server Enabled",
                description="Unencrypted HTTP management interface is exposed",
                affected_lines=self._extract_matching_lines(config, r'ip\s+http\s+server'),
                recommendation="Disable HTTP or use HTTPS: 'no ip http server' and 'ip http secure-server'",
                risk_score=7
            ))

        # Check 4: No enable secret
        has_enable_secret = 'enable secret' in config_lower
        has_enable_password = 'enable password' in config_lower and 'enable secret' not in config_lower

        if has_enable_password:
            self.findings.append(Finding(
                severity=FindingSeverity.CRITICAL,
                category=FindingCategory.SECURITY,
                title="Weak Enable Password",
                description="Using 'enable password' instead of 'enable secret'",
                affected_lines=self._extract_matching_lines(config, r'enable\s+password'),
                recommendation="Use 'enable secret' with strong password",
                risk_score=8
            ))
        elif not has_enable_secret and not has_enable_password:
            self.findings.append(Finding(
                severity=FindingSeverity.HIGH,
                category=FindingCategory.SECURITY,
                title="No Enable Secret Configured",
                description="No privileged mode password protection",
                affected_lines=[],
                recommendation="Configure 'enable secret' with strong password",
                risk_score=7
            ))

        # Check 5: Weak encryption
        if re.search(r'password\s+\d+\s+\w+', config):
            # Check for type 7 passwords (weak)
            type7_passwords = re.findall(r'password\s+7\s+\w+', config)
            if type7_passwords:
                self.findings.append(Finding(
                    severity=FindingSeverity.MEDIUM,
                    category=FindingCategory.SECURITY,
                    title="Type 7 Password Encryption",
                    description=f"Found {len(type7_passwords)} Type 7 encrypted passwords (easily reversible)",
                    affected_lines=type7_passwords[:3],
                    recommendation="Use 'service password-encryption' and 'enable secret' (Type 5)",
                    risk_score=6
                ))

        # Check 6: No AAA
        if 'aaa new-model' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.MEDIUM,
                category=FindingCategory.SECURITY,
                title="AAA Not Configured",
                description="Authentication, Authorization, and Accounting (AAA) is not enabled",
                affected_lines=[],
                recommendation="Configure AAA for centralized authentication: 'aaa new-model'",
                risk_score=5
            ))

        # Check 7: Overly permissive ACLs
        permit_any_rules = re.findall(r'permit\s+ip\s+any\s+any', config, re.IGNORECASE)
        if permit_any_rules:
            self.findings.append(Finding(
                severity=FindingSeverity.HIGH,
                category=FindingCategory.SECURITY,
                title="Overly Permissive ACL Rules",
                description=f"Found {len(permit_any_rules)} 'permit any any' rules",
                affected_lines=permit_any_rules[:3],
                recommendation="Restrict ACLs to specific source/destination networks",
                risk_score=7
            ))

    def _check_best_practices(self, config: str, device_type: str) -> None:
        """
        Check best practice compliance.

        Args:
            config: Configuration text
            device_type: Device platform
        """
        config_lower = config.lower()

        # Check 1: No NTP configured
        if 'ntp server' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.MEDIUM,
                category=FindingCategory.BEST_PRACTICE,
                title="NTP Not Configured",
                description="No NTP server configured for time synchronization",
                affected_lines=[],
                recommendation="Configure NTP servers: 'ntp server <ip>'",
                risk_score=4
            ))

        # Check 2: No logging
        if 'logging' not in config_lower or 'logging buffered' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.MEDIUM,
                category=FindingCategory.BEST_PRACTICE,
                title="Insufficient Logging",
                description="No logging or buffered logging configured",
                affected_lines=[],
                recommendation="Configure logging: 'logging buffered' and 'logging host <syslog-server>'",
                risk_score=5
            ))

        # Check 3: No interface descriptions
        interfaces = re.findall(r'^interface\s+\S+', config, re.MULTILINE | re.IGNORECASE)
        interface_descriptions = re.findall(r'^\s+description\s+', config, re.MULTILINE | re.IGNORECASE)

        if len(interfaces) > 0 and len(interface_descriptions) < len(interfaces) * 0.5:
            self.findings.append(Finding(
                severity=FindingSeverity.LOW,
                category=FindingCategory.BEST_PRACTICE,
                title="Missing Interface Descriptions",
                description=f"Only {len(interface_descriptions)}/{len(interfaces)} interfaces have descriptions",
                affected_lines=[],
                recommendation="Add descriptions to all interfaces for documentation",
                risk_score=2
            ))

        # Check 4: No banner
        if 'banner motd' not in config_lower and 'banner login' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.LOW,
                category=FindingCategory.BEST_PRACTICE,
                title="No Login Banner",
                description="No login banner configured",
                affected_lines=[],
                recommendation="Configure login banner with legal notice",
                risk_score=2
            ))

        # Check 5: CDP enabled on all interfaces
        if 'no cdp run' not in config_lower:
            # CDP is on by default, check if disabled
            self.findings.append(Finding(
                severity=FindingSeverity.LOW,
                category=FindingCategory.BEST_PRACTICE,
                title="CDP Globally Enabled",
                description="Cisco Discovery Protocol exposes device information",
                affected_lines=[],
                recommendation="Disable CDP on external interfaces or globally if not needed",
                risk_score=3
            ))

    def _check_compliance(self, config: str, device_type: str) -> None:
        """
        Check compliance requirements.

        Args:
            config: Configuration text
            device_type: Device platform
        """
        config_lower = config.lower()

        # Check 1: Password complexity
        if 'security passwords min-length' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.MEDIUM,
                category=FindingCategory.COMPLIANCE,
                title="No Minimum Password Length",
                description="Password complexity policy not enforced",
                affected_lines=[],
                recommendation="Configure: 'security passwords min-length 8'",
                risk_score=5
            ))

        # Check 2: Login attempts
        if 'login block-for' not in config_lower:
            self.findings.append(Finding(
                severity=FindingSeverity.LOW,
                category=FindingCategory.COMPLIANCE,
                title="No Login Rate Limiting",
                description="No protection against brute force attacks",
                affected_lines=[],
                recommendation="Configure: 'login block-for 300 attempts 3 within 60'",
                risk_score=4
            ))

        # Check 3: Session timeout
        vty_lines = self._extract_section(config, r'^line vty')
        if vty_lines:
            if 'exec-timeout' not in vty_lines.lower():
                self.findings.append(Finding(
                    severity=FindingSeverity.MEDIUM,
                    category=FindingCategory.COMPLIANCE,
                    title="No VTY Session Timeout",
                    description="VTY lines do not have exec timeout configured",
                    affected_lines=["line vty 0 4"],
                    recommendation="Configure: 'exec-timeout 10 0' (10 minutes)",
                    risk_score=5
                ))

    def _check_redundancy(self, config: str, device_type: str) -> None:
        """
        Check redundancy configuration.

        Args:
            config: Configuration text
            device_type: Device platform
        """
        # Check for HSRP/VRRP
        has_hsrp = 'standby' in config.lower()
        has_vrrp = 'vrrp' in config.lower()

        if not has_hsrp and not has_vrrp:
            # Check if device has multiple interfaces (might need redundancy)
            interfaces = re.findall(r'^interface\s+\S+', config, re.MULTILINE | re.IGNORECASE)
            if len(interfaces) > 2:
                self.findings.append(Finding(
                    severity=FindingSeverity.INFO,
                    category=FindingCategory.REDUNDANCY,
                    title="No Gateway Redundancy Protocol",
                    description="No HSRP/VRRP configured for gateway redundancy",
                    affected_lines=[],
                    recommendation="Consider HSRP/VRRP for critical gateway interfaces",
                    risk_score=3
                ))

    def _check_performance(self, config: str, device_type: str) -> None:
        """
        Check performance-related configuration.

        Args:
            config: Configuration text
            device_type: Device platform
        """
        config_lower = config.lower()

        # Check for spanning-tree optimization
        if 'spanning-tree' in config_lower:
            if 'spanning-tree portfast' not in config_lower:
                self.findings.append(Finding(
                    severity=FindingSeverity.INFO,
                    category=FindingCategory.PERFORMANCE,
                    title="PortFast Not Configured",
                    description="PortFast can reduce convergence time for access ports",
                    affected_lines=[],
                    recommendation="Enable PortFast on access ports: 'spanning-tree portfast'",
                    risk_score=2
                ))

    def _check_documentation(self, config: str) -> None:
        """
        Check configuration documentation.

        Args:
            config: Configuration text
        """
        # Check for comments
        comments = re.findall(r'^\s*!.*\S', config, re.MULTILINE)
        config_lines = len([l for l in config.splitlines() if l.strip()])

        if len(comments) < config_lines * 0.1:  # Less than 10% commented
            self.findings.append(Finding(
                severity=FindingSeverity.INFO,
                category=FindingCategory.BEST_PRACTICE,
                title="Insufficient Documentation",
                description="Configuration has minimal comments/documentation",
                affected_lines=[],
                recommendation="Add comments to document complex sections",
                risk_score=1
            ))

    def _extract_matching_lines(self, config: str, pattern: str) -> List[str]:
        """Extract lines matching regex pattern."""
        matches = re.findall(pattern, config, re.IGNORECASE | re.MULTILINE)
        return matches[:5]  # Limit to 5 samples

    def _extract_section(self, config: str, start_pattern: str) -> str:
        """Extract configuration section."""
        lines = config.splitlines()
        section_lines = []
        in_section = False

        for line in lines:
            if re.match(start_pattern, line, re.IGNORECASE):
                in_section = True
                section_lines.append(line)
            elif in_section:
                if line.startswith(' ') or line.startswith('\t'):
                    section_lines.append(line)
                else:
                    break

        return '\n'.join(section_lines)

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Returns:
            Analysis results dictionary
        """
        # Sort findings by severity and risk score
        sorted_findings = sorted(
            self.findings,
            key=lambda f: (
                ['critical', 'high', 'medium', 'low', 'info'].index(f.severity.value),
                -f.risk_score
            )
        )

        # Count by severity
        severity_counts = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.HIGH: 0,
            FindingSeverity.MEDIUM: 0,
            FindingSeverity.LOW: 0,
            FindingSeverity.INFO: 0
        }

        for finding in sorted_findings:
            severity_counts[finding.severity] += 1

        # Count by category
        category_counts = {}
        for finding in sorted_findings:
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        # Calculate overall risk score
        overall_risk = sum(f.risk_score for f in sorted_findings) / len(sorted_findings) if sorted_findings else 0

        return {
            "summary": {
                "total_findings": len(sorted_findings),
                "critical": severity_counts[FindingSeverity.CRITICAL],
                "high": severity_counts[FindingSeverity.HIGH],
                "medium": severity_counts[FindingSeverity.MEDIUM],
                "low": severity_counts[FindingSeverity.LOW],
                "info": severity_counts[FindingSeverity.INFO],
                "overall_risk_score": round(overall_risk, 1)
            },
            "severity_counts": {k.value: v for k, v in severity_counts.items()},
            "category_counts": {k.value: v for k, v in category_counts.items()},
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "title": f.title,
                    "description": f.description,
                    "affected_lines": f.affected_lines,
                    "recommendation": f.recommendation,
                    "risk_score": f.risk_score
                }
                for f in sorted_findings
            ]
        }

    def generate_text_report(self) -> str:
        """
        Generate human-readable text report.

        Returns:
            Formatted report string
        """
        report_data = self._generate_analysis_report()
        summary = report_data['summary']

        report = f"""
CONFIGURATION ANALYSIS REPORT
{'='*60}

Overall Risk Score: {summary['overall_risk_score']}/10

Summary:
  Total Findings: {summary['total_findings']}
  Critical:       {summary['critical']}
  High:           {summary['high']}
  Medium:         {summary['medium']}
  Low:            {summary['low']}
  Info:           {summary['info']}

"""

        # Group findings by severity
        for severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH, FindingSeverity.MEDIUM]:
            severity_findings = [f for f in self.findings if f.severity == severity]

            if severity_findings:
                report += f"\n{severity.value.upper()} Severity Findings:\n"
                report += "-" * 60 + "\n"

                for finding in severity_findings:
                    report += f"\n  ⚠ {finding.title} (Risk: {finding.risk_score}/10)\n"
                    report += f"     {finding.description}\n"
                    report += f"     → {finding.recommendation}\n"

        return report


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Config Analyzer Demo
    ========================================
    Comprehensive configuration analysis
    ========================================
    """)

    # Sample configuration with issues
    sample_config = """
hostname CORE-RTR-01

enable password cisco123

interface GigabitEthernet0/0
 ip address 10.1.1.1 255.255.255.0
 no shutdown

interface GigabitEthernet0/1
 ip address 10.2.2.1 255.255.255.0
 shutdown

interface GigabitEthernet0/2
 switchport access vlan 10
 no shutdown

line vty 0 4
 password cisco
 transport input telnet
 login

ip http server

snmp-server community public RO
snmp-server community private RW

ip access-list extended PERMIT_ALL
 permit ip any any

!
end
"""

    # Test: Analyze configuration
    print("\nAnalyzing Configuration...")
    print("-" * 60)

    analyzer = ConfigAnalyzer()
    analysis = analyzer.analyze_config(
        sample_config,
        device_type="cisco_ios",
        check_security=True,
        check_best_practices=True,
        check_compliance=True
    )

    # Display summary
    print(f"\nAnalysis Summary:")
    for key, value in analysis['summary'].items():
        print(f"  {key:20s}: {value}")

    # Display findings by severity
    print(f"\n\nCritical Findings:")
    critical = [f for f in analysis['findings'] if f['severity'] == 'critical']
    for finding in critical:
        print(f"  ⚠ {finding['title']}")
        print(f"     {finding['description']}")
        print(f"     → {finding['recommendation']}")
        print()

    print(f"\nHigh Severity Findings:")
    high = [f for f in analysis['findings'] if f['severity'] == 'high']
    for finding in high:
        print(f"  ⚠ {finding['title']}")
        print(f"     {finding['description']}")
        print(f"     → {finding['recommendation']}")
        print()

    # Generate text report
    print("\n" + "="*60)
    print("FULL TEXT REPORT")
    print("="*60)
    text_report = analyzer.generate_text_report()
    print(text_report)

    print("\n✅ Demo complete!")
    print("\n💡 Analysis Capabilities:")
    print("  - Security vulnerability detection")
    print("  - Best practice validation")
    print("  - Compliance checking")
    print("  - Redundancy analysis")
    print("  - Performance optimization")
    print("  - Risk scoring")

    print("\n⚠️  Top Security Issues Detected:")
    print("  ☐ Telnet enabled (use SSH)")
    print("  ☐ Default SNMP communities (public/private)")
    print("  ☐ Weak enable password")
    print("  ☐ HTTP server enabled")
    print("  ☐ Overly permissive ACLs")
