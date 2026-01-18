#!/usr/bin/env python3
"""
Guardrails - Safety Constraints for AI Network Operations

Implement safety guardrails to prevent AI from making dangerous or
policy-violating changes to network infrastructure.

From: AI for Networking Engineers - Volume 1, Chapter 12
Author: Eduard Dulharu

Usage:
    from guardrails import SafetyGuardrails

    guardrails = SafetyGuardrails()
    result = guardrails.validate_change(proposed_config, device_info)
    if result.safe:
        # Apply change
        pass
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ViolationType(str, Enum):
    """Safety violation types."""
    CRITICAL_DEVICE = "critical_device"
    PRODUCTION_HOURS = "production_hours"
    SYNTAX_ERROR = "syntax_error"
    DESTRUCTIVE_COMMAND = "destructive_command"
    SECURITY_POLICY = "security_policy"
    COMPLIANCE = "compliance"
    MISSING_APPROVAL = "missing_approval"


@dataclass
class SafetyViolation:
    """Safety violation details."""
    violation_type: ViolationType
    severity: str  # "block", "warn", "info"
    message: str
    affected_lines: List[str]
    recommendation: str


@dataclass
class SafetyResult:
    """Safety validation result."""
    safe: bool
    violations: List[SafetyViolation]
    warnings: List[SafetyViolation]
    risk_score: int  # 0-10


class SafetyGuardrails:
    """
    Enforce safety guardrails for AI network operations.

    Features:
    - Prevent changes to critical devices
    - Block destructive commands
    - Enforce production change windows
    - Validate against security policies
    - Syntax validation
    - Rollback capability requirements
    """

    # Critical devices that require extra approval
    CRITICAL_DEVICES = [
        r'core-rtr',
        r'border-rtr',
        r'fw-\d+',
        r'vpn-concentrator'
    ]

    # Destructive commands that should be blocked
    DESTRUCTIVE_COMMANDS = [
        r'reload',
        r'erase\s+startup-config',
        r'write\s+erase',
        r'format',
        r'delete\s+/force',
        r'no\s+interface',
        r'shutdown\s+system',
        r'reload\s+/force'
    ]

    # Commands requiring special approval
    HIGH_RISK_COMMANDS = [
        r'no\s+router',
        r'no\s+ip\s+route',
        r'access-list.*deny.*any',
        r'shutdown',  # Interface shutdown
        r'clear\s+ip\s+bgp',
        r'clear\s+ip\s+ospf',
    ]

    # Minimum required configuration elements for compliance
    REQUIRED_SECURITY_FEATURES = [
        'enable secret',
        'aaa',
        'logging',
        'ntp server'
    ]

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize safety guardrails.

        Args:
            config_file: Optional configuration file for custom policies
        """
        self.policies = self._load_policies(config_file)

    def _load_policies(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load safety policies from configuration."""
        # Default policies
        return {
            "block_critical_devices_without_approval": True,
            "block_destructive_commands": True,
            "enforce_change_windows": True,
            "require_rollback_plan": True,
            "enforce_security_baseline": True,
            "allowed_change_window": {
                "start_hour": 0,  # 12 AM
                "end_hour": 6     # 6 AM
            }
        }

    def validate_change(
        self,
        proposed_config: str,
        device_info: Dict[str, Any],
        has_approval: bool = False,
        has_rollback_plan: bool = False
    ) -> SafetyResult:
        """
        Validate proposed configuration change against safety policies.

        Args:
            proposed_config: Proposed configuration or commands
            device_info: Device information (hostname, role, criticality)
            has_approval: Whether change has required approval
            has_rollback_plan: Whether rollback plan exists

        Returns:
            SafetyResult with validation outcome
        """
        violations = []
        warnings = []

        # Check 1: Critical device protection
        if self._is_critical_device(device_info.get('hostname', '')):
            if not has_approval:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.CRITICAL_DEVICE,
                    severity="block",
                    message=f"Changes to critical device require approval",
                    affected_lines=[device_info.get('hostname', 'unknown')],
                    recommendation="Obtain approval from network admin before proceeding"
                ))

        # Check 2: Destructive commands
        destructive = self._check_destructive_commands(proposed_config)
        if destructive:
            violations.extend(destructive)

        # Check 3: High-risk commands
        high_risk = self._check_high_risk_commands(proposed_config)
        if high_risk and not has_approval:
            warnings.extend(high_risk)

        # Check 4: Change window enforcement
        if self.policies['enforce_change_windows']:
            window_check = self._check_change_window()
            if window_check:
                warnings.append(window_check)

        # Check 5: Rollback plan requirement
        if self.policies['require_rollback_plan'] and not has_rollback_plan:
            violations.append(SafetyViolation(
                violation_type=ViolationType.MISSING_APPROVAL,
                severity="block",
                message="Rollback plan required for configuration changes",
                affected_lines=[],
                recommendation="Create rollback plan before applying changes"
            ))

        # Check 6: Security baseline compliance
        if self.policies['enforce_security_baseline']:
            security_violations = self._check_security_baseline(proposed_config)
            warnings.extend(security_violations)

        # Check 7: Syntax validation
        syntax_errors = self._check_syntax(proposed_config)
        if syntax_errors:
            violations.extend(syntax_errors)

        # Calculate risk score
        risk_score = self._calculate_risk_score(violations, warnings)

        # Determine if safe
        blocking_violations = [v for v in violations if v.severity == "block"]
        safe = len(blocking_violations) == 0

        return SafetyResult(
            safe=safe,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score
        )

    def _is_critical_device(self, hostname: str) -> bool:
        """Check if device is critical."""
        hostname_lower = hostname.lower()

        for pattern in self.CRITICAL_DEVICES:
            if re.search(pattern, hostname_lower):
                return True

        return False

    def _check_destructive_commands(
        self,
        config: str
    ) -> List[SafetyViolation]:
        """Check for destructive commands."""
        violations = []

        for pattern in self.DESTRUCTIVE_COMMANDS:
            matches = re.findall(pattern, config, re.IGNORECASE | re.MULTILINE)

            if matches:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.DESTRUCTIVE_COMMAND,
                    severity="block",
                    message=f"Destructive command detected: {pattern}",
                    affected_lines=matches,
                    recommendation="Remove destructive command or obtain special approval"
                ))

        return violations

    def _check_high_risk_commands(
        self,
        config: str
    ) -> List[SafetyViolation]:
        """Check for high-risk commands."""
        warnings = []

        for pattern in self.HIGH_RISK_COMMANDS:
            matches = re.findall(pattern, config, re.IGNORECASE | re.MULTILINE)

            if matches:
                warnings.append(SafetyViolation(
                    violation_type=ViolationType.DESTRUCTIVE_COMMAND,
                    severity="warn",
                    message=f"High-risk command detected: {pattern}",
                    affected_lines=matches[:3],  # Limit to 3 examples
                    recommendation="Review carefully - this command can impact service"
                ))

        return warnings

    def _check_change_window(self) -> Optional[SafetyViolation]:
        """Check if current time is within allowed change window."""
        from datetime import datetime

        current_hour = datetime.now().hour
        start_hour = self.policies['allowed_change_window']['start_hour']
        end_hour = self.policies['allowed_change_window']['end_hour']

        # Check if current time is outside change window
        if not (start_hour <= current_hour < end_hour):
            return SafetyViolation(
                violation_type=ViolationType.PRODUCTION_HOURS,
                severity="warn",
                message=f"Changes outside approved window ({start_hour}:00-{end_hour}:00)",
                affected_lines=[],
                recommendation="Schedule changes during maintenance window or obtain emergency approval"
            )

        return None

    def _check_security_baseline(
        self,
        config: str
    ) -> List[SafetyViolation]:
        """Check security baseline compliance."""
        warnings = []
        config_lower = config.lower()

        for required_feature in self.REQUIRED_SECURITY_FEATURES:
            if required_feature.lower() not in config_lower:
                # Only warn if this is a complete config (not a partial change)
                if len(config.splitlines()) > 20:  # Heuristic for complete config
                    warnings.append(SafetyViolation(
                        violation_type=ViolationType.SECURITY_POLICY,
                        severity="warn",
                        message=f"Security baseline: '{required_feature}' not configured",
                        affected_lines=[],
                        recommendation=f"Configure {required_feature} for security compliance"
                    ))

        return warnings

    def _check_syntax(self, config: str) -> List[SafetyViolation]:
        """Basic syntax validation."""
        errors = []

        lines = config.splitlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()

            if not line or line.startswith('!'):
                continue

            # Check for common syntax errors
            # Unmatched quotes
            if line.count('"') % 2 != 0:
                errors.append(SafetyViolation(
                    violation_type=ViolationType.SYNTAX_ERROR,
                    severity="block",
                    message=f"Syntax error: Unmatched quotes on line {i}",
                    affected_lines=[line],
                    recommendation="Fix quote syntax"
                ))

            # Invalid IP address format (basic check)
            ip_matches = re.findall(r'\d+\.\d+\.\d+\.\d+', line)
            for ip in ip_matches:
                octets = ip.split('.')
                if any(int(octet) > 255 for octet in octets if octet.isdigit()):
                    errors.append(SafetyViolation(
                        violation_type=ViolationType.SYNTAX_ERROR,
                        severity="block",
                        message=f"Invalid IP address: {ip}",
                        affected_lines=[line],
                        recommendation="Correct IP address format"
                    ))

        return errors

    def _calculate_risk_score(
        self,
        violations: List[SafetyViolation],
        warnings: List[SafetyViolation]
    ) -> int:
        """Calculate overall risk score."""
        score = 0

        # Blocking violations
        for violation in violations:
            if violation.severity == "block":
                score += 3

        # Warnings
        for warning in warnings:
            if warning.severity == "warn":
                score += 1

        return min(score, 10)  # Cap at 10

    def generate_report(self, result: SafetyResult) -> str:
        """
        Generate human-readable safety report.

        Args:
            result: SafetyResult

        Returns:
            Formatted report
        """
        status = "✓ SAFE" if result.safe else "✗ BLOCKED"

        report = f"""
SAFETY VALIDATION REPORT
{'='*60}

Status: {status}
Risk Score: {result.risk_score}/10

"""

        if result.violations:
            report += "\nVIOLATIONS (Blocking):\n"
            report += "-" * 60 + "\n"

            for violation in result.violations:
                report += f"\n🔴 {violation.message}\n"
                report += f"   Type: {violation.violation_type.value}\n"

                if violation.affected_lines:
                    report += f"   Affected: {', '.join(violation.affected_lines[:3])}\n"

                report += f"   → {violation.recommendation}\n"

        if result.warnings:
            report += "\nWARNINGS:\n"
            report += "-" * 60 + "\n"

            for warning in result.warnings:
                report += f"\n🟡 {warning.message}\n"
                report += f"   Type: {warning.violation_type.value}\n"

                if warning.affected_lines:
                    report += f"   Affected: {', '.join(warning.affected_lines[:3])}\n"

                report += f"   → {warning.recommendation}\n"

        if result.safe and not result.warnings:
            report += "\n✓ No safety violations detected.\n"

        return report


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Safety Guardrails Demo
    ========================================
    Prevent dangerous AI operations
    ========================================
    """)

    guardrails = SafetyGuardrails()

    # Test 1: Safe change
    print("\nTest 1: Safe Configuration Change")
    print("-" * 60)

    safe_config = """
interface GigabitEthernet0/1
 description Updated by AI
 no shutdown
"""

    result = guardrails.validate_change(
        proposed_config=safe_config,
        device_info={"hostname": "access-sw-01", "role": "access_switch"},
        has_approval=False,
        has_rollback_plan=True
    )

    print(f"Safe: {result.safe}")
    print(f"Risk Score: {result.risk_score}/10")
    print(f"Violations: {len(result.violations)}")
    print(f"Warnings: {len(result.warnings)}")

    # Test 2: Destructive command
    print("\n\nTest 2: Destructive Command (Blocked)")
    print("-" * 60)

    destructive_config = """
reload
erase startup-config
"""

    result = guardrails.validate_change(
        proposed_config=destructive_config,
        device_info={"hostname": "access-sw-01"},
        has_approval=False,
        has_rollback_plan=False
    )

    print(f"Safe: {result.safe}")
    print(f"Violations: {len(result.violations)}")

    for violation in result.violations:
        print(f"  - {violation.message}")

    # Test 3: Critical device without approval
    print("\n\nTest 3: Critical Device Without Approval")
    print("-" * 60)

    result = guardrails.validate_change(
        proposed_config=safe_config,
        device_info={"hostname": "core-rtr-01", "role": "core_router"},
        has_approval=False,
        has_rollback_plan=True
    )

    print(f"Safe: {result.safe}")
    print(f"Violations: {len(result.violations)}")

    for violation in result.violations:
        print(f"  - {violation.message}")
        print(f"    → {violation.recommendation}")

    # Test 4: High-risk command with approval
    print("\n\nTest 4: High-Risk Command With Approval")
    print("-" * 60)

    risky_config = """
interface GigabitEthernet0/1
 shutdown
no router ospf 1
"""

    result = guardrails.validate_change(
        proposed_config=risky_config,
        device_info={"hostname": "distribution-sw-01"},
        has_approval=True,  # Has approval
        has_rollback_plan=True
    )

    print(f"Safe: {result.safe}")
    print(f"Warnings: {len(result.warnings)}")

    for warning in result.warnings:
        print(f"  ⚠ {warning.message}")

    # Test 5: Syntax errors
    print("\n\nTest 5: Syntax Errors")
    print("-" * 60)

    bad_syntax = """
interface GigabitEthernet0/1
 ip address 192.168.1.300 255.255.255.0
 description "Unclosed quote
"""

    result = guardrails.validate_change(
        proposed_config=bad_syntax,
        device_info={"hostname": "test-device"},
        has_approval=True,
        has_rollback_plan=True
    )

    print(f"Safe: {result.safe}")
    print(f"Syntax Errors: {len([v for v in result.violations if v.violation_type == ViolationType.SYNTAX_ERROR])}")

    for violation in result.violations:
        print(f"  - {violation.message}")

    # Test 6: Full report
    print("\n\nTest 6: Full Safety Report")
    print("-" * 60)

    mixed_config = """
interface GigabitEthernet0/1
 shutdown
 ip address 10.1.1.1 255.255.255.0

no router bgp 65000
reload in 5
"""

    result = guardrails.validate_change(
        proposed_config=mixed_config,
        device_info={"hostname": "core-rtr-01", "role": "core_router"},
        has_approval=False,
        has_rollback_plan=False
    )

    report = guardrails.generate_report(result)
    print(report)

    print("\n✅ Demo complete!")
    print("\n💡 Safety Guardrails Purpose:")
    print("  - Prevent accidental service disruptions")
    print("  - Enforce security policies")
    print("  - Protect critical infrastructure")
    print("  - Ensure compliance with change management")
    print("  - Provide early warning of risky changes")

    print("\n⚠️  Guardrail Categories:")
    print("  ☐ Device protection (critical devices)")
    print("  ☐ Command filtering (destructive commands)")
    print("  ☐ Time-based controls (change windows)")
    print("  ☐ Syntax validation")
    print("  ☐ Security baseline enforcement")
    print("  ☐ Approval requirements")
