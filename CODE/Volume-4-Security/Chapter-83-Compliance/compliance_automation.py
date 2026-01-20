"""
Chapter 83: Compliance Automation
Dual-AI compliance validation using Claude and OpenAI with LangChain orchestration

This module automates SOC2, PCI-DSS, and GDPR compliance checking for network infrastructure.
Uses Claude for complex compliance reasoning and OpenAI for structured validation.

Author: Ed @ vExpertAI
Production-ready code with proper error handling and type hints
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Google Colab compatibility
try:
    from google.colab import userdata  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


@dataclass
class ComplianceViolation:
    """Compliance violation finding"""
    control_id: str
    severity: str  # critical, high, medium, low
    device: str
    description: str
    remediation: str
    evidence: str
    detected_by: str  # claude or openai
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DualAIComplianceChecker:
    """
    Dual-AI compliance checker using both Claude and OpenAI

    Architecture:
    - Claude: Complex reasoning for SOC2 and GDPR (better at contextual analysis)
    - OpenAI: Structured validation for PCI-DSS (better at rule-based checking)
    - LangChain: Orchestrates both models
    """

    def __init__(self, anthropic_key: Optional[str] = None, openai_key: Optional[str] = None):
        """Initialize dual-AI compliance checker"""

        # Get API keys
        if IN_COLAB:
            self.anthropic_key = anthropic_key or userdata.get('ANTHROPIC_API_KEY')
            self.openai_key = openai_key or userdata.get('OPENAI_API_KEY')
        else:
            self.anthropic_key = anthropic_key or os.getenv('ANTHROPIC_API_KEY')
            self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')

        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found")

        # Initialize LangChain chat models with correct API
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Use 'model' not 'model_name'
            api_key=self.anthropic_key,
            max_tokens=2000,
            temperature=0
        )

        self.openai = ChatOpenAI(
            model="gpt-4o",
            api_key=self.openai_key,
            temperature=0
        )

    def check_soc2_compliance(self, device_config: str, device_name: str) -> Dict:
        """
        Example 1: SOC2 compliance automation with dual-AI validation

        Uses Claude for primary analysis (better at contextual reasoning)
        Uses OpenAI for validation (second opinion on critical findings)
        """
        print(f"\n[SOC2] Checking compliance for {device_name}...")

        violations = []

        # Quick pattern checks first (no AI needed)
        basic_violations = self._basic_soc2_checks(device_config, device_name)
        violations.extend(basic_violations)

        # Claude analysis for complex compliance reasoning
        claude_violations = self._claude_soc2_analysis(device_config, device_name)
        violations.extend(claude_violations)

        # OpenAI validation for critical findings
        if any(v.severity == 'critical' for v in violations):
            print(f"[SOC2] Critical violations found - running OpenAI validation...")
            openai_validation = self._openai_validate_critical(device_config, device_name, violations)
            violations.extend(openai_validation)

        return {
            'device_name': device_name,
            'framework': 'SOC2',
            'compliant': len(violations) == 0,
            'violation_count': len(violations),
            'violations': [asdict(v) for v in violations],
            'checked_controls': ['CC6.1', 'CC6.6', 'CC6.7', 'CC7.2'],
            'timestamp': datetime.now().isoformat()
        }

    def _basic_soc2_checks(self, config: str, device: str) -> List[ComplianceViolation]:
        """Basic SOC2 checks using pattern matching (no AI needed)"""
        violations = []

        # CC6.1: Check for MFA/centralized auth
        has_tacacs = re.search(r'tacacs-server\s+host', config, re.IGNORECASE)
        has_radius = re.search(r'radius-server\s+host', config, re.IGNORECASE)
        local_auth = re.search(r'username\s+\S+\s+(?:password|secret)', config, re.IGNORECASE)

        if local_auth and not (has_tacacs or has_radius):
            violations.append(ComplianceViolation(
                control_id='CC6.1',
                severity='critical',
                device=device,
                description='Local authentication without centralized auth. SOC2 requires MFA.',
                remediation='Configure TACACS+ or RADIUS with MFA enforcement.',
                evidence='Config contains local username/password only',
                detected_by='pattern_match'
            ))

        # CC6.7: Check for logging
        has_logging = re.search(r'logging\s+(?:host|server|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
                               config, re.IGNORECASE)

        if not has_logging:
            violations.append(ComplianceViolation(
                control_id='CC6.7',
                severity='critical',
                device=device,
                description='No centralized logging configured. SOC2 requires all admin actions logged.',
                remediation='Configure syslog server: logging host <syslog-ip>',
                evidence='No logging host/server found in config',
                detected_by='pattern_match'
            ))

        # CC7.2: Check for insecure SNMP
        insecure_snmp = re.search(r'snmp-server\s+community\s+(public|private)\s',
                                 config, re.IGNORECASE)

        if insecure_snmp:
            violations.append(ComplianceViolation(
                control_id='CC7.2',
                severity='critical',
                device=device,
                description=f'SNMP with default community string "{insecure_snmp.group(1)}"',
                remediation='Migrate to SNMPv3 with authentication and encryption',
                evidence=f'snmp-server community {insecure_snmp.group(1)}',
                detected_by='pattern_match'
            ))

        return violations

    def _claude_soc2_analysis(self, config: str, device: str) -> List[ComplianceViolation]:
        """Use Claude for complex SOC2 compliance analysis"""

        config_sample = config[:2000]  # First 2000 chars

        prompt = f"""You are a SOC2 auditor reviewing network device configurations.

DEVICE: {device}

CONFIGURATION (sample):
{config_sample}

SOC2 REQUIREMENTS:
- CC6.1: MFA for admin access, no shared accounts
- CC6.6: Least privilege access, network segmentation
- CC6.7: Comprehensive logging, configuration management
- CC7.2: Security monitoring, encryption for management protocols

ANALYSIS REQUIRED:
Look for additional compliance issues beyond basic pattern matching:
1. Weak encryption protocols (Telnet, HTTP, SSHv1)
2. Overly permissive access rules (permit any any)
3. Missing security features (no timeout, no privilege levels)
4. Shared accounts or default credentials

Respond in JSON format:
{{
    "violations": [
        {{
            "control_id": "CC6.7",
            "severity": "high",
            "description": "Clear description of issue",
            "remediation": "How to fix it",
            "evidence": "Config snippet showing issue"
        }}
    ],
    "overall_assessment": "Brief assessment",
    "audit_readiness": "Ready/Not Ready"
}}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.claude.invoke(messages)

            # Extract text content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse JSON response
            analysis = json.loads(response_text)

            # Convert to ComplianceViolation objects
            violations = []
            for v in analysis.get('violations', []):
                violations.append(ComplianceViolation(
                    control_id=v['control_id'],
                    severity=v['severity'],
                    device=device,
                    description=v['description'],
                    remediation=v['remediation'],
                    evidence=v['evidence'],
                    detected_by='claude'
                ))

            return violations

        except json.JSONDecodeError as e:
            print(f"[ERROR] Claude returned invalid JSON: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] Claude analysis failed: {e}")
            return []

    def _openai_validate_critical(self, config: str, device: str,
                                  violations: List[ComplianceViolation]) -> List[ComplianceViolation]:
        """Use OpenAI to validate critical findings (second opinion)"""

        critical_violations = [v for v in violations if v.severity == 'critical']
        if not critical_violations:
            return []

        violations_text = "\n".join([
            f"- {v.control_id}: {v.description}" for v in critical_violations
        ])

        prompt = f"""You are validating SOC2 compliance findings for a network device.

DEVICE: {device}

CRITICAL VIOLATIONS FOUND:
{violations_text}

CONFIGURATION (first 1500 chars):
{config[:1500]}

VALIDATION REQUIRED:
Review each critical violation. For each one, determine:
1. Is this a real compliance issue? (true/false)
2. Is the severity correct? (critical/high/medium/low)
3. Any additional context or findings?

Respond in JSON:
{{
    "validated_issues": [
        {{
            "control_id": "CC6.1",
            "is_real_issue": true,
            "confirmed_severity": "critical",
            "additional_context": "Explanation"
        }}
    ],
    "new_findings": [
        {{
            "control_id": "CC6.7",
            "severity": "high",
            "description": "New issue found",
            "remediation": "How to fix",
            "evidence": "Config snippet"
        }}
    ]
}}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.openai.invoke(messages)

            # Extract text content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse JSON response
            validation = json.loads(response_text)

            # Convert new findings to violations
            new_violations = []
            for finding in validation.get('new_findings', []):
                new_violations.append(ComplianceViolation(
                    control_id=finding['control_id'],
                    severity=finding['severity'],
                    device=device,
                    description=f"[OpenAI Validation] {finding['description']}",
                    remediation=finding['remediation'],
                    evidence=finding['evidence'],
                    detected_by='openai'
                ))

            return new_violations

        except json.JSONDecodeError as e:
            print(f"[ERROR] OpenAI returned invalid JSON: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] OpenAI validation failed: {e}")
            return []

    def check_pci_segmentation(self, firewall_rules: List[Dict],
                               cde_subnets: List[str]) -> Dict:
        """
        Example 2: PCI-DSS network segmentation validation

        Uses OpenAI for rule-based checking (better at structured analysis)
        Uses Claude for contextual assessment (better at complex reasoning)
        """
        print(f"\n[PCI-DSS] Validating network segmentation...")

        violations = []

        # Basic checks using pattern matching
        basic_violations = self._basic_pci_checks(firewall_rules, cde_subnets)
        violations.extend(basic_violations)

        # OpenAI for structured rule analysis
        openai_violations = self._openai_pci_analysis(firewall_rules, cde_subnets)
        violations.extend(openai_violations)

        # Claude for overall segmentation assessment
        claude_assessment = self._claude_pci_assessment(firewall_rules, cde_subnets, violations)

        return {
            'framework': 'PCI-DSS',
            'requirement': '1.2.1 (Network Segmentation)',
            'compliant': len(violations) == 0,
            'violation_count': len(violations),
            'violations': [asdict(v) for v in violations],
            'segmentation_assessment': claude_assessment,
            'timestamp': datetime.now().isoformat()
        }

    def _basic_pci_checks(self, rules: List[Dict], cde_subnets: List[str]) -> List[ComplianceViolation]:
        """Basic PCI-DSS checks using pattern matching"""
        violations = []

        for rule in rules:
            dest = rule.get('destination', '')
            source = rule.get('source', '')
            action = rule.get('action', '')
            service = rule.get('service', '')

            # Check for rules allowing traffic to CDE
            if action == 'permit' and any(cde in dest for cde in cde_subnets):

                # Check for overly permissive service
                if service in ['any', 'all']:
                    violations.append(ComplianceViolation(
                        control_id='PCI-DSS 1.2.1',
                        severity='high',
                        device='firewall',
                        description=f'Rule {rule.get("rule_id")} allows ANY service to CDE',
                        remediation='Specify exact services/ports required. Remove "permit any"',
                        evidence=f'Rule: {source} -> {dest} service {service}',
                        detected_by='pattern_match'
                    ))

                # Check for traffic from any source
                if source in ['any', '0.0.0.0/0', 'internet']:
                    violations.append(ComplianceViolation(
                        control_id='PCI-DSS 1.2.1',
                        severity='critical',
                        device='firewall',
                        description=f'Rule {rule.get("rule_id")} allows traffic from ANY source to CDE',
                        remediation='Specify authorized source IPs/subnets only',
                        evidence=f'Rule: {source} -> {dest}',
                        detected_by='pattern_match'
                    ))

        return violations

    def _openai_pci_analysis(self, rules: List[Dict], cde_subnets: List[str]) -> List[ComplianceViolation]:
        """Use OpenAI for structured PCI-DSS rule analysis"""

        # Format rules for analysis (first 20)
        rules_text = "\n".join([
            f"Rule {r.get('rule_id', i)}: {r.get('action')} {r.get('source')} -> {r.get('destination')} ({r.get('service')})"
            for i, r in enumerate(rules[:20], 1)
        ])

        prompt = f"""You are a PCI-DSS QSA (Qualified Security Assessor) reviewing firewall rules.

CDE SUBNETS: {', '.join(cde_subnets)}

FIREWALL RULES (sample):
{rules_text}

PCI-DSS REQUIREMENT 1.2.1:
Restrict inbound and outbound traffic to that which is necessary for the cardholder
data environment (CDE), and specifically deny all other traffic.

ANALYSIS REQUIRED:
1. Identify rules that violate least-privilege access to CDE
2. Check for missing egress filtering from CDE
3. Verify DMZ segmentation (Internet -> DMZ -> CDE, not direct)
4. Look for overly broad allow rules

Respond in JSON:
{{
    "violations": [
        {{
            "severity": "critical",
            "description": "Issue description",
            "remediation": "How to fix",
            "evidence": "Rule details"
        }}
    ],
    "segmentation_score": "Good/Needs Improvement/Non-Compliant"
}}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.openai.invoke(messages)

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            analysis = json.loads(response_text)

            violations = []
            for v in analysis.get('violations', []):
                violations.append(ComplianceViolation(
                    control_id='PCI-DSS 1.2.1',
                    severity=v['severity'],
                    device='firewall',
                    description=v['description'],
                    remediation=v['remediation'],
                    evidence=v['evidence'],
                    detected_by='openai'
                ))

            return violations

        except Exception as e:
            print(f"[ERROR] OpenAI PCI analysis failed: {e}")
            return []

    def _claude_pci_assessment(self, rules: List[Dict], cde_subnets: List[str],
                               violations: List[ComplianceViolation]) -> str:
        """Use Claude for overall PCI-DSS segmentation assessment"""

        violations_summary = "\n".join([
            f"- [{v.severity.upper()}] {v.description}" for v in violations
        ]) if violations else "No violations detected"

        prompt = f"""You are a PCI-DSS assessor providing an overall segmentation assessment.

CDE SUBNETS: {', '.join(cde_subnets)}
FIREWALL RULES COUNT: {len(rules)}

VIOLATIONS FOUND:
{violations_summary}

Provide a brief overall assessment (2-3 sentences):
1. Is the CDE properly segmented?
2. What is the overall risk level?
3. Is the organization ready for PCI-DSS audit?

Respond with plain text assessment (not JSON)."""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.claude.invoke(messages)

            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            print(f"[ERROR] Claude assessment failed: {e}")
            return "Assessment unavailable due to error"

    def check_gdpr_data_flows(self, systems: List[Dict], data_flows: List[Dict]) -> Dict:
        """
        Example 3: GDPR data flow mapping with multi-agent analysis

        Uses both Claude and OpenAI to analyze data flows for GDPR compliance
        """
        print(f"\n[GDPR] Mapping personal data flows...")

        # Claude analyzes data flow patterns
        claude_analysis = self._claude_gdpr_analysis(systems, data_flows)

        # OpenAI validates international transfers
        openai_validation = self._openai_gdpr_validation(systems, data_flows)

        # Combine findings
        violations = []
        violations.extend(claude_analysis.get('violations', []))
        violations.extend(openai_validation.get('violations', []))

        return {
            'framework': 'GDPR',
            'article': 'Article 30 (Records of Processing)',
            'systems_checked': len(systems),
            'data_flows_checked': len(data_flows),
            'compliant': len(violations) == 0,
            'violations': [asdict(v) for v in violations],
            'international_transfers': openai_validation.get('international_transfers', []),
            'risk_assessment': claude_analysis.get('risk_assessment', ''),
            'timestamp': datetime.now().isoformat()
        }

    def _claude_gdpr_analysis(self, systems: List[Dict], data_flows: List[Dict]) -> Dict:
        """Use Claude for GDPR data flow pattern analysis"""

        systems_text = "\n".join([
            f"- {s['name']} ({s['location']}): {', '.join(s.get('data_types', []))}"
            for s in systems
        ])

        flows_text = "\n".join([
            f"- {f['source']} -> {f['destination']}: {f.get('data_type', 'unknown')}"
            for f in data_flows
        ])

        prompt = f"""You are a GDPR compliance officer analyzing data flows.

SYSTEMS:
{systems_text}

DATA FLOWS:
{flows_text}

GDPR REQUIREMENTS:
- Article 30: Document what personal data you process, where it comes from, where it goes
- Article 44-46: Restrictions on international transfers (EU -> non-EU requires safeguards)

ANALYSIS REQUIRED:
1. Identify systems processing personal data
2. Map data flows involving personal data
3. Flag international transfers requiring safeguards
4. Assess overall GDPR compliance risk

Respond in JSON:
{{
    "violations": [
        {{
            "severity": "high",
            "description": "Issue description",
            "remediation": "Required action",
            "evidence": "System/flow details"
        }}
    ],
    "risk_assessment": "Low/Medium/High risk assessment"
}}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.claude.invoke(messages)

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            analysis = json.loads(response_text)

            violations = []
            for v in analysis.get('violations', []):
                violations.append(ComplianceViolation(
                    control_id='GDPR Art.30',
                    severity=v['severity'],
                    device='data_flow',
                    description=v['description'],
                    remediation=v['remediation'],
                    evidence=v['evidence'],
                    detected_by='claude'
                ))

            return {
                'violations': violations,
                'risk_assessment': analysis.get('risk_assessment', 'Unknown')
            }

        except Exception as e:
            print(f"[ERROR] Claude GDPR analysis failed: {e}")
            return {'violations': [], 'risk_assessment': 'Error'}

    def _openai_gdpr_validation(self, systems: List[Dict], data_flows: List[Dict]) -> Dict:
        """Use OpenAI for GDPR international transfer validation"""

        eu_countries = ['DE', 'FR', 'UK', 'ES', 'IT', 'NL', 'BE', 'AT', 'SE', 'DK', 'FI', 'IE', 'PT', 'GR', 'PL']

        # Find international transfers
        international_transfers = []
        for flow in data_flows:
            source_sys = next((s for s in systems if s['name'] == flow['source']), None)
            dest_sys = next((s for s in systems if s['name'] == flow['destination']), None)

            if source_sys and dest_sys:
                from_location = source_sys.get('location', 'Unknown')
                to_location = dest_sys.get('location', 'Unknown')

                # Check if EU -> non-EU
                if from_location in eu_countries and to_location not in eu_countries:
                    international_transfers.append({
                        'from': flow['source'],
                        'to': flow['destination'],
                        'from_location': from_location,
                        'to_location': to_location,
                        'data_type': flow.get('data_type', 'unknown')
                    })

        if not international_transfers:
            return {'violations': [], 'international_transfers': []}

        transfers_text = "\n".join([
            f"- {t['from']} ({t['from_location']}) -> {t['to']} ({t['to_location']}): {t['data_type']}"
            for t in international_transfers
        ])

        prompt = f"""You are validating GDPR international data transfers.

INTERNATIONAL TRANSFERS DETECTED (EU -> non-EU):
{transfers_text}

GDPR REQUIREMENTS:
- Transfers from EU to non-EU require safeguards (SCCs, DPF, Adequacy Decision)
- Without safeguards = GDPR violation

VALIDATION REQUIRED:
For each transfer, determine:
1. Does it involve personal data? (true/false)
2. What safeguard mechanism is required? (SCCs, DPF, etc.)
3. Is this a compliance violation without proper safeguards?

Respond in JSON:
{{
    "violations": [
        {{
            "severity": "high",
            "description": "Transfer violation description",
            "remediation": "Required safeguard mechanism",
            "evidence": "Transfer details"
        }}
    ],
    "transfers_requiring_safeguards": ["transfer1", "transfer2"]
}}"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.openai.invoke(messages)

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            validation = json.loads(response_text)

            violations = []
            for v in validation.get('violations', []):
                violations.append(ComplianceViolation(
                    control_id='GDPR Art.44',
                    severity=v['severity'],
                    device='international_transfer',
                    description=v['description'],
                    remediation=v['remediation'],
                    evidence=v['evidence'],
                    detected_by='openai'
                ))

            return {
                'violations': violations,
                'international_transfers': international_transfers
            }

        except Exception as e:
            print(f"[ERROR] OpenAI GDPR validation failed: {e}")
            return {'violations': [], 'international_transfers': international_transfers}

    def generate_compliance_report(self, results: List[Dict]) -> str:
        """Generate unified compliance report across all frameworks"""

        report = f"""
MULTI-FRAMEWORK COMPLIANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Powered by Dual-AI Analysis (Claude + OpenAI)

"""

        for result in results:
            framework = result.get('framework', 'Unknown')
            compliant = result.get('compliant', False)
            violation_count = result.get('violation_count', 0)

            report += f"\n{framework} COMPLIANCE\n"
            report += "=" * 80 + "\n"
            report += f"Status: {'COMPLIANT' if compliant else 'NON-COMPLIANT'}\n"
            report += f"Violations: {violation_count}\n\n"

            if violation_count > 0:
                # Group by severity
                violations = result.get('violations', [])
                by_severity = {}
                for v in violations:
                    severity = v['severity']
                    if severity not in by_severity:
                        by_severity[severity] = []
                    by_severity[severity].append(v)

                for severity in ['critical', 'high', 'medium', 'low']:
                    if severity in by_severity:
                        report += f"\n{severity.upper()} SEVERITY ({len(by_severity[severity])}):\n"
                        for v in by_severity[severity]:
                            report += f"\n  [{v['control_id']}] {v['description']}\n"
                            report += f"  Remediation: {v['remediation']}\n"
                            report += f"  Detected by: {v['detected_by']}\n"
                            report += f"  Evidence: {v['evidence']}\n"

            report += "\n" + "-" * 80 + "\n"

        return report


# Example functions

def example_1():
    """Example 1: SOC2 compliance automation with dual-AI validation"""

    print("\n" + "=" * 80)
    print("EXAMPLE 1: SOC2 Compliance Automation (Dual-AI)")
    print("=" * 80)

    checker = DualAIComplianceChecker()

    # Non-compliant device config (realistic example)
    bad_config = """
hostname router-edge-01
!
username admin password MyPassword123
username backup password Backup456
!
snmp-server community public RO
snmp-server community private RW
!
interface GigabitEthernet0/0
 ip address 10.1.1.1 255.255.255.0
!
access-list 100 permit ip any any
!
line vty 0 4
 login local
 transport input telnet ssh
!
no logging host
!
"""

    result = checker.check_soc2_compliance(bad_config, 'router-edge-01')

    print(f"\nDevice: {result['device_name']}")
    print(f"Compliant: {result['compliant']}")
    print(f"Violations: {result['violation_count']}\n")

    for violation in result['violations']:
        print(f"[{violation['severity'].upper()}] {violation['control_id']} (detected by {violation['detected_by']})")
        print(f"  Issue: {violation['description']}")
        print(f"  Fix: {violation['remediation']}")
        print()

    return result


def example_2():
    """Example 2: PCI-DSS network segmentation validation"""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: PCI-DSS Network Segmentation (Dual-AI)")
    print("=" * 80)

    checker = DualAIComplianceChecker()

    # Define firewall rules (some violate PCI-DSS)
    firewall_rules = [
        {
            'rule_id': 'R001',
            'source': 'internet',
            'destination': '10.0.1.0/24',  # DMZ
            'service': 'https',
            'action': 'permit'
        },
        {
            'rule_id': 'R002',
            'source': 'any',  # BAD: Any source
            'destination': '10.0.10.0/24',  # CDE
            'service': 'any',  # BAD: Any service
            'action': 'permit'
        },
        {
            'rule_id': 'R003',
            'source': '10.0.1.0/24',
            'destination': '10.0.10.0/24',  # CDE
            'service': 'mysql',
            'action': 'permit'
        },
        {
            'rule_id': 'R004',
            'source': 'internet',
            'destination': '10.0.10.5',  # BAD: Direct internet to CDE
            'service': 'ssh',
            'action': 'permit'
        }
    ]

    cde_subnets = ['10.0.10.0/24']

    result = checker.check_pci_segmentation(firewall_rules, cde_subnets)

    print(f"\nFramework: {result['framework']}")
    print(f"Requirement: {result['requirement']}")
    print(f"Compliant: {result['compliant']}")
    print(f"Violations: {result['violation_count']}\n")

    for violation in result['violations']:
        print(f"[{violation['severity'].upper()}] (detected by {violation['detected_by']})")
        print(f"  Issue: {violation['description']}")
        print(f"  Fix: {violation['remediation']}")
        print()

    print(f"\nSegmentation Assessment:\n{result['segmentation_assessment']}")

    return result


def example_3():
    """Example 3: GDPR data flow mapping with multi-agent analysis"""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: GDPR Data Flow Mapping (Multi-Agent)")
    print("=" * 80)

    checker = DualAIComplianceChecker()

    # Define systems
    systems = [
        {
            'name': 'web-app-eu',
            'location': 'DE',
            'data_types': ['customer_name', 'email', 'ip_address'],
            'purpose': 'Web application'
        },
        {
            'name': 'database-eu',
            'location': 'FR',
            'data_types': ['customer_name', 'email', 'purchase_history'],
            'purpose': 'Data storage'
        },
        {
            'name': 'analytics-us',
            'location': 'US',
            'data_types': ['user_behavior', 'session_data'],
            'purpose': 'Analytics'
        },
        {
            'name': 'backup-us',
            'location': 'US',
            'data_types': ['customer_name', 'email'],  # Personal data!
            'purpose': 'Backups'
        }
    ]

    # Define data flows
    data_flows = [
        {
            'source': 'web-app-eu',
            'destination': 'database-eu',
            'data_type': 'customer_data'
        },
        {
            'source': 'web-app-eu',
            'destination': 'analytics-us',
            'data_type': 'anonymized_behavior'
        },
        {
            'source': 'database-eu',
            'destination': 'backup-us',  # BAD: EU -> US with personal data
            'data_type': 'customer_data'
        }
    ]

    result = checker.check_gdpr_data_flows(systems, data_flows)

    print(f"\nFramework: {result['framework']}")
    print(f"Systems Checked: {result['systems_checked']}")
    print(f"Data Flows Checked: {result['data_flows_checked']}")
    print(f"Compliant: {result['compliant']}")
    print(f"Violations: {len(result['violations'])}\n")

    if result['international_transfers']:
        print("INTERNATIONAL TRANSFERS DETECTED:")
        for transfer in result['international_transfers']:
            print(f"  {transfer['from']} ({transfer['from_location']}) -> {transfer['to']} ({transfer['to_location']})")
        print()

    for violation in result['violations']:
        print(f"[{violation['severity'].upper()}] {violation['control_id']} (detected by {violation['detected_by']})")
        print(f"  Issue: {violation['description']}")
        print(f"  Fix: {violation['remediation']}")
        print()

    print(f"\nRisk Assessment: {result['risk_assessment']}")

    return result


def main():
    """Run all examples"""

    print("\n" + "=" * 80)
    print("CHAPTER 83: COMPLIANCE AUTOMATION WITH DUAL-AI")
    print("=" * 80)
    print("\nThis demonstrates why dual-AI is better for compliance:")
    print("- Claude: Better at complex reasoning and contextual analysis")
    print("- OpenAI: Better at structured validation and rule-based checking")
    print("- Together: More comprehensive coverage and higher accuracy")
    print()

    results = []

    # Run all examples
    try:
        result1 = example_1()
        results.append(result1)
    except Exception as e:
        print(f"[ERROR] Example 1 failed: {e}")

    try:
        result2 = example_2()
        results.append(result2)
    except Exception as e:
        print(f"[ERROR] Example 2 failed: {e}")

    try:
        result3 = example_3()
        results.append(result3)
    except Exception as e:
        print(f"[ERROR] Example 3 failed: {e}")

    # Generate unified report
    if results:
        checker = DualAIComplianceChecker()
        report = checker.generate_compliance_report(results)

        print("\n" + "=" * 80)
        print("UNIFIED COMPLIANCE REPORT")
        print("=" * 80)
        print(report)


if __name__ == "__main__":
    main()
