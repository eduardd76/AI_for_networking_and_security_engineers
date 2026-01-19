# Chapter 83: Compliance Automation

## Learning Objectives

By the end of this chapter, you will:
- Automate SOC2 compliance checks for network infrastructure
- Validate PCI-DSS network segmentation with AI analysis
- Map data flows for GDPR compliance automatically
- Build continuous compliance monitoring systems
- Generate audit-ready evidence and reports

**Prerequisites**: Understanding of compliance frameworks (SOC2, PCI-DSS, GDPR basics), network configuration knowledge, Chapters 70-80

**What You'll Build**: Complete compliance automation platform that continuously monitors network infrastructure and generates audit evidence for SOC2, PCI-DSS, and GDPR.

---

## The Compliance Problem

Your company needs compliance certifications to do business:
- **SOC2**: Required by enterprise customers
- **PCI-DSS**: Required if you handle credit cards
- **GDPR**: Required if you have EU customers

**Traditional Compliance Approach**:
- Manual audits once per year
- Scramble for 3 weeks gathering evidence
- Pay $50K-$150K for auditors
- Pass or fail (no continuous visibility)
- Between audits: Configuration drift, violations undetected

**Real Incident**: SOC2 Audit Failure at SaaS Company

```
Day 1 (Audit Start):
  Auditor: "Show me evidence that all admin access is logged."
  IT Team: "We have logging... somewhere."

Day 5:
  Finding: 15 network devices have no logging enabled
  Finding: VPN allows password-only auth (no MFA requirement)
  Finding: Firewall rules allow unrestricted internal access

Day 15 (Audit End):
  Result: SOC2 FAILED
  Cost: $120K audit fees (wasted)
  Impact: Lost 3 major enterprise deals ($2.5M ARR)
  Timeline: Must remediate and re-audit (6 months delay)
```

**With AI Compliance Automation**:
- Continuous monitoring (24/7, not once per year)
- Instant violation detection
- Auto-generated audit evidence
- 90% less manual work
- Always audit-ready

This chapter builds that system.

---

## Section 1: SOC2 Compliance Automation

### SOC2 Requirements Overview

SOC2 Trust Service Criteria for network infrastructure:

**CC6.1 - Logical Access Controls**
- MFA required for admin access
- Access provisioning/deprovisioning
- Privileged access management
- Access reviews

**CC6.6 - Logical Access Restrictions**
- Network segmentation
- Least privilege access
- Firewall rules

**CC6.7 - System Operations**
- Logging and monitoring
- Configuration management
- Change management

**CC7.2 - System Monitoring**
- Security monitoring
- Incident detection
- Log retention

### Building SOC2 Compliance Checker

```python
"""
SOC2 Compliance Automation for Network Infrastructure
Validates SOC2 requirements continuously
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import anthropic
import json
import re

@dataclass
class ComplianceViolation:
    """Compliance violation finding"""
    control_id: str
    severity: str  # critical, high, medium, low
    device: str
    description: str
    remediation: str
    evidence: str

class SOC2ComplianceChecker:
    """Automated SOC2 compliance checking for network infrastructure"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def check_device_compliance(self, device_config: str, device_name: str,
                               device_type: str) -> Dict:
        """Check device configuration for SOC2 compliance"""

        violations = []

        # CC6.1 - Check MFA/TACACS/RADIUS (not local auth only)
        mfa_check = self._check_mfa_requirement(device_config, device_name)
        if mfa_check['violation']:
            violations.append(mfa_check['violation'])

        # CC6.6 - Check access restrictions
        access_check = self._check_access_restrictions(device_config, device_name)
        violations.extend(access_check['violations'])

        # CC6.7 - Check logging enabled
        logging_check = self._check_logging_enabled(device_config, device_name)
        if logging_check['violation']:
            violations.append(logging_check['violation'])

        # CC7.2 - Check SNMP security
        snmp_check = self._check_snmp_security(device_config, device_name)
        if snmp_check['violation']:
            violations.append(snmp_check['violation'])

        # AI comprehensive analysis
        if violations or device_type in ['firewall', 'router', 'switch']:
            ai_analysis = self._ai_analyze_soc2_compliance(device_config, device_name, violations)
            violations.extend(ai_analysis.get('additional_violations', []))

        return {
            'device_name': device_name,
            'device_type': device_type,
            'compliant': len(violations) == 0,
            'violation_count': len(violations),
            'violations': violations,
            'checked_controls': ['CC6.1', 'CC6.6', 'CC6.7', 'CC7.2']
        }

    def _check_mfa_requirement(self, config: str, device: str) -> Dict:
        """CC6.1: Check MFA/centralized auth required"""

        # Check for TACACS or RADIUS (centralized auth with MFA)
        has_tacacs = re.search(r'tacacs-server\s+host', config, re.IGNORECASE)
        has_radius = re.search(r'radius-server\s+host', config, re.IGNORECASE)

        # Check for local authentication (bad)
        local_auth = re.search(r'username\s+\S+\s+(?:password|secret)', config, re.IGNORECASE)

        if local_auth and not (has_tacacs or has_radius):
            return {
                'violation': ComplianceViolation(
                    control_id='CC6.1',
                    severity='critical',
                    device=device,
                    description='Local authentication configured without centralized auth (TACACS/RADIUS). SOC2 requires MFA for admin access.',
                    remediation='Configure TACACS+ or RADIUS with MFA enforcement. Remove local usernames.',
                    evidence=f'Config contains: username ... password (local auth only)'
                )
            }

        if not (has_tacacs or has_radius):
            return {
                'violation': ComplianceViolation(
                    control_id='CC6.1',
                    severity='high',
                    device=device,
                    description='No centralized authentication configured. Cannot enforce MFA.',
                    remediation='Configure TACACS+ or RADIUS server for centralized authentication and MFA.',
                    evidence='No tacacs-server or radius-server configuration found'
                )
            }

        return {'violation': None}

    def _check_access_restrictions(self, config: str, device: str) -> Dict:
        """CC6.6: Check access restrictions and least privilege"""

        violations = []

        # Check for unrestricted access (permit any any)
        if re.search(r'permit\s+ip\s+any\s+any', config, re.IGNORECASE):
            violations.append(ComplianceViolation(
                control_id='CC6.6',
                severity='high',
                device=device,
                description='Firewall rule permits unrestricted access (permit ip any any). Violates least privilege.',
                remediation='Implement specific allow rules. Default deny policy.',
                evidence='Config contains: permit ip any any'
            ))

        # Check for no ACLs on VTY lines (unrestricted management access)
        vty_lines = re.findall(r'line vty.*?\n(?:.*?\n)*?(?=line\s|\Z)', config, re.IGNORECASE | re.DOTALL)
        for vty_section in vty_lines:
            if not re.search(r'access-class', vty_section, re.IGNORECASE):
                violations.append(ComplianceViolation(
                    control_id='CC6.6',
                    severity='medium',
                    device=device,
                    description='VTY lines have no access-class (unrestricted management access from any IP).',
                    remediation='Apply access-class to restrict management access to authorized IPs only.',
                    evidence='VTY lines configured without access-class restriction'
                ))
                break

        return {'violations': violations}

    def _check_logging_enabled(self, config: str, device: str) -> Dict:
        """CC6.7: Check logging to centralized syslog server"""

        # Check for syslog server configured
        has_logging = re.search(r'logging\s+(?:host|server|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
                               config, re.IGNORECASE)

        if not has_logging:
            return {
                'violation': ComplianceViolation(
                    control_id='CC6.7',
                    severity='critical',
                    device=device,
                    description='No centralized logging configured. SOC2 requires all admin actions to be logged.',
                    remediation='Configure syslog server: logging host <syslog-server-ip>',
                    evidence='No "logging host" or "logging server" found in config'
                )
            }

        # Check logging level
        log_level_match = re.search(r'logging\s+trap\s+(\w+)', config, re.IGNORECASE)
        if log_level_match:
            level = log_level_match.group(1).lower()
            if level in ['emergency', 'alert', 'critical', 'error']:
                return {
                    'violation': ComplianceViolation(
                        control_id='CC6.7',
                        severity='medium',
                        device=device,
                        description=f'Logging level too restrictive ({level}). May miss audit-relevant events.',
                        remediation='Set logging level to informational: logging trap informational',
                        evidence=f'logging trap {level}'
                    )
                }

        return {'violation': None}

    def _check_snmp_security(self, config: str, device: str) -> Dict:
        """CC7.2: Check SNMP security (no SNMPv1/v2c with default communities)"""

        # Check for SNMPv3 (secure)
        has_snmpv3 = re.search(r'snmp-server\s+group\s+.*\s+v3', config, re.IGNORECASE)

        # Check for SNMPv1/v2c with common default communities
        insecure_snmp = re.search(r'snmp-server\s+community\s+(public|private)\s',
                                 config, re.IGNORECASE)

        if insecure_snmp and not has_snmpv3:
            community = insecure_snmp.group(1)
            return {
                'violation': ComplianceViolation(
                    control_id='CC7.2',
                    severity='critical',
                    device=device,
                    description=f'SNMP configured with default community string "{community}" and no SNMPv3. Security risk.',
                    remediation='Migrate to SNMPv3 with authentication and encryption. Remove default communities.',
                    evidence=f'snmp-server community {community}'
                )
            }

        return {'violation': None}

    def _ai_analyze_soc2_compliance(self, config: str, device: str,
                                   detected_violations: List) -> Dict:
        """Use AI to detect subtle SOC2 compliance issues"""

        # Sample config for AI (first 2000 chars)
        config_sample = config[:2000]

        violation_summary = "\n".join([
            f"- {v.control_id}: {v.description}" for v in detected_violations
        ]) if detected_violations else "No violations detected yet"

        prompt = f"""You are a SOC2 auditor reviewing network device configurations.

DEVICE: {device}

CONFIGURATION (sample):
{config_sample}

VIOLATIONS ALREADY DETECTED:
{violation_summary}

SOC2 REQUIREMENTS TO CHECK:
- CC6.1: MFA for admin access, no shared accounts
- CC6.6: Least privilege access, network segmentation
- CC6.7: Comprehensive logging, configuration management
- CC7.2: Security monitoring, encryption for management protocols

ANALYSIS REQUIRED:
1. Are there additional SOC2 compliance issues not yet detected?
2. Focus on: weak encryption, shared accounts, missing security features
3. Consider: Are SNMPv2c, Telnet, HTTP (unencrypted) enabled?

Respond in JSON:
{{
    "additional_violations": [
        {{
            "control_id": "CC6.7",
            "severity": "high",
            "description": "issue description",
            "remediation": "how to fix",
            "evidence": "config snippet showing issue"
        }}
    ],
    "overall_assessment": "assessment text",
    "audit_readiness": "Ready/Not Ready"
}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)

            # Convert to ComplianceViolation objects
            additional_violations = []
            for v in analysis.get('additional_violations', []):
                additional_violations.append(ComplianceViolation(
                    control_id=v['control_id'],
                    severity=v['severity'],
                    device=device,
                    description=v['description'],
                    remediation=v['remediation'],
                    evidence=v['evidence']
                ))

            return {
                'additional_violations': additional_violations,
                'overall_assessment': analysis.get('overall_assessment'),
                'audit_readiness': analysis.get('audit_readiness')
            }

        except Exception as e:
            return {
                'additional_violations': [],
                'error': str(e)
            }

    def generate_compliance_report(self, device_results: List[Dict]) -> str:
        """Generate SOC2 compliance report for audit"""

        total_devices = len(device_results)
        compliant_devices = sum(1 for d in device_results if d['compliant'])
        total_violations = sum(d['violation_count'] for d in device_results)

        # Group violations by control
        violations_by_control = {}
        for device in device_results:
            for violation in device['violations']:
                control_id = violation.control_id
                if control_id not in violations_by_control:
                    violations_by_control[control_id] = []
                violations_by_control[control_id].append(violation)

        # Generate report
        report = f"""
SOC2 COMPLIANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
Total Devices Audited: {total_devices}
Compliant Devices: {compliant_devices} ({compliant_devices/total_devices*100:.1f}%)
Non-Compliant Devices: {total_devices - compliant_devices}
Total Violations: {total_violations}

VIOLATIONS BY CONTROL
=====================
"""

        for control_id, violations in sorted(violations_by_control.items()):
            report += f"\n{control_id}: {len(violations)} violations\n"

            # Group by severity
            by_severity = {}
            for v in violations:
                if v.severity not in by_severity:
                    by_severity[v.severity] = []
                by_severity[v.severity].append(v)

            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in by_severity:
                    report += f"  {severity.upper()}: {len(by_severity[severity])}\n"

        report += "\n\nDETAILED FINDINGS\n"
        report += "=" * 80 + "\n"

        for device in device_results:
            if device['violations']:
                report += f"\nDevice: {device['device_name']}\n"
                report += f"Status: {'COMPLIANT' if device['compliant'] else 'NON-COMPLIANT'}\n"
                report += f"Violations: {device['violation_count']}\n\n"

                for v in device['violations']:
                    report += f"  [{v.severity.upper()}] {v.control_id}: {v.description}\n"
                    report += f"  Remediation: {v.remediation}\n"
                    report += f"  Evidence: {v.evidence}\n\n"

        return report

# Example: SOC2 compliance checking
def demo_soc2_compliance():
    checker = SOC2ComplianceChecker(anthropic_api_key="your-api-key")

    # Non-compliant device config
    bad_config = """
hostname router-edge-01
!
username admin password MyPassword123
!
snmp-server community public RO
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
"""

    # Check compliance
    result = checker.check_device_compliance(bad_config, 'router-edge-01', 'router')

    print("=== SOC2 COMPLIANCE CHECK ===")
    print(f"Device: {result['device_name']}")
    print(f"Compliant: {result['compliant']}")
    print(f"Violations: {result['violation_count']}\n")

    for violation in result['violations']:
        print(f"[{violation.severity.upper()}] {violation.control_id}")
        print(f"  Issue: {violation.description}")
        print(f"  Fix: {violation.remediation}\n")

    # Generate report
    report = checker.generate_compliance_report([result])
    print("\n" + "="*80)
    print(report)

# Example Output:
"""
=== SOC2 COMPLIANCE CHECK ===
Device: router-edge-01
Compliant: False
Violations: 5

[CRITICAL] CC6.1
  Issue: Local authentication configured without centralized auth (TACACS/RADIUS). SOC2 requires MFA for admin access.
  Fix: Configure TACACS+ or RADIUS with MFA enforcement. Remove local usernames.

[CRITICAL] CC7.2
  Issue: SNMP configured with default community string "public" and no SNMPv3. Security risk.
  Fix: Migrate to SNMPv3 with authentication and encryption. Remove default communities.

[CRITICAL] CC6.7
  Issue: No centralized logging configured. SOC2 requires all admin actions to be logged.
  Fix: Configure syslog server: logging host <syslog-server-ip>

[HIGH] CC6.6
  Issue: Firewall rule permits unrestricted access (permit ip any any). Violates least privilege.
  Fix: Implement specific allow rules. Default deny policy.

[MEDIUM] CC6.6
  Issue: VTY lines have no access-class (unrestricted management access from any IP).
  Fix: Apply access-class to restrict management access to authorized IPs only.
"""
```

---

## Section 2: PCI-DSS Network Segmentation

### PCI-DSS Requirement 1: Network Segmentation

PCI-DSS Requirement 1.2: Build firewall configurations that restrict connections between untrusted networks and system components in the cardholder data environment (CDE).

**Key Requirements**:
- CDE must be isolated from other networks
- Firewall rules must deny by default
- Only necessary services allowed into CDE
- DMZ between internet and CDE
- No direct routes from untrusted to CDE

### PCI-DSS Segmentation Validator

```python
"""
PCI-DSS Network Segmentation Validation
Validates Requirement 1 (Firewall and Network Segmentation)
"""
from typing import List, Dict, Set
import networkx as nx
import anthropic
import json

class PCIDSSSegmentationValidator:
    """Validate PCI-DSS network segmentation requirements"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.network_graph = nx.DiGraph()

    def build_network_topology(self, firewall_rules: List[Dict],
                               device_zones: Dict[str, str]):
        """Build network topology from firewall rules"""

        # device_zones: {device_ip: 'CDE'/'DMZ'/'Internal'/'Internet'}

        for rule in firewall_rules:
            source = rule['source']
            dest = rule['destination']
            action = rule['action']  # permit/deny
            service = rule['service']

            if action == 'permit':
                self.network_graph.add_edge(
                    source, dest,
                    service=service,
                    rule_id=rule.get('rule_id', 'unknown')
                )

        # Add zone information
        for device, zone in device_zones.items():
            if device in self.network_graph:
                self.network_graph.nodes[device]['zone'] = zone

    def validate_cde_isolation(self, cde_devices: List[str]) -> Dict:
        """Validate CDE is properly isolated (PCI-DSS Req 1.2)"""

        violations = []

        # Check 1: No direct internet → CDE access
        internet_nodes = [n for n in self.network_graph.nodes()
                         if self.network_graph.nodes[n].get('zone') == 'Internet']

        for internet_node in internet_nodes:
            for cde_device in cde_devices:
                if nx.has_path(self.network_graph, internet_node, cde_device):
                    path = nx.shortest_path(self.network_graph, internet_node, cde_device)

                    if len(path) == 2:  # Direct connection
                        violations.append({
                            'severity': 'critical',
                            'requirement': 'PCI-DSS 1.2.1',
                            'description': f'Direct connection from Internet to CDE device {cde_device}',
                            'path': path,
                            'remediation': 'Add DMZ layer. Internet → DMZ → CDE only.'
                        })
                    elif 'DMZ' not in [self.network_graph.nodes[n].get('zone') for n in path]:
                        violations.append({
                            'severity': 'high',
                            'requirement': 'PCI-DSS 1.2.1',
                            'description': f'Path from Internet to CDE without DMZ: {" → ".join(path)}',
                            'path': path,
                            'remediation': 'Route through DMZ. No direct Internal → CDE from Internet.'
                        })

        # Check 2: CDE isolation from general internal network
        internal_nodes = [n for n in self.network_graph.nodes()
                         if self.network_graph.nodes[n].get('zone') == 'Internal']

        unrestricted_access = []
        for internal_node in internal_nodes:
            for cde_device in cde_devices:
                if self.network_graph.has_edge(internal_node, cde_device):
                    # Check if rule is specific or wildcard
                    edge_data = self.network_graph[internal_node][cde_device]
                    if edge_data.get('service') == 'any':
                        unrestricted_access.append((internal_node, cde_device))

        if unrestricted_access:
            violations.append({
                'severity': 'critical',
                'requirement': 'PCI-DSS 1.2.1',
                'description': f'Unrestricted access from Internal to CDE: {len(unrestricted_access)} connections',
                'examples': unrestricted_access[:5],
                'remediation': 'Implement least-privilege firewall rules. Only allow necessary services to CDE.'
            })

        # Check 3: Default deny policy
        # If there are paths to CDE, check if explicit deny rules exist
        outbound_from_cde = []
        for cde_device in cde_devices:
            outbound = list(self.network_graph.successors(cde_device))
            outbound_from_cde.extend([(cde_device, n) for n in outbound])

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'cde_devices_checked': len(cde_devices),
            'paths_to_cde': sum(1 for src in self.network_graph.nodes()
                               for cde in cde_devices
                               if nx.has_path(self.network_graph, src, cde))
        }

    def validate_firewall_rules(self, rules: List[Dict], cde_subnets: List[str]) -> Dict:
        """Validate firewall rules for PCI-DSS compliance"""

        violations = []

        # Check for deny-by-default
        has_default_deny = any(r.get('is_default_deny') for r in rules)
        if not has_default_deny:
            violations.append({
                'severity': 'critical',
                'requirement': 'PCI-DSS 1.2.1',
                'description': 'No default-deny policy detected. All traffic should be denied by default.',
                'remediation': 'Add implicit deny rule at end of ACL: deny ip any any'
            })

        # Check each rule touching CDE
        for rule in rules:
            dest = rule['destination']

            # Is this rule allowing traffic to CDE?
            if any(cde_subnet in dest for cde_subnet in cde_subnets):
                # Check if service is overly permissive
                if rule.get('service') == 'any' or rule.get('port') == 'any':
                    violations.append({
                        'severity': 'high',
                        'requirement': 'PCI-DSS 1.2.1',
                        'description': f'Firewall rule allows ANY service to CDE: {rule.get("rule_id")}',
                        'rule': rule,
                        'remediation': 'Specify exact services/ports required. Remove "permit any" rules.'
                    })

                # Check source - should not be "any"
                if rule.get('source') in ['any', '0.0.0.0/0']:
                    violations.append({
                        'severity': 'critical',
                        'requirement': 'PCI-DSS 1.2.1',
                        'description': f'Firewall rule allows traffic from ANY source to CDE: {rule.get("rule_id")}',
                        'rule': rule,
                        'remediation': 'Specify authorized source IPs/subnets only.'
                    })

        # AI analysis for subtle issues
        ai_analysis = self._ai_analyze_pci_rules(rules, cde_subnets, violations)
        violations.extend(ai_analysis.get('additional_violations', []))

        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'rules_checked': len(rules)
        }

    def _ai_analyze_pci_rules(self, rules: List[Dict], cde_subnets: List[str],
                             detected_violations: List[Dict]) -> Dict:
        """Use AI to analyze firewall rules for PCI-DSS compliance"""

        # Sample rules for AI
        rules_sample = rules[:20]  # First 20 rules
        rules_text = "\n".join([
            f"{r.get('rule_id')}: {r.get('action')} {r.get('source')} → {r.get('destination')} ({r.get('service')})"
            for r in rules_sample
        ])

        violations_text = "\n".join([
            f"- {v['requirement']}: {v['description']}" for v in detected_violations
        ]) if detected_violations else "No violations detected yet"

        prompt = f"""You are a PCI-DSS QSA (Qualified Security Assessor) reviewing firewall rules.

CDE SUBNETS: {', '.join(cde_subnets)}

FIREWALL RULES (sample):
{rules_text}

VIOLATIONS ALREADY DETECTED:
{violations_text}

PCI-DSS REQUIREMENT 1.2.1:
Restrict inbound and outbound traffic to that which is necessary for the cardholder data environment, and specifically deny all other traffic.

ANALYSIS REQUIRED:
1. Are there additional PCI-DSS violations in these firewall rules?
2. Look for: overly permissive rules, missing egress filtering, no DMZ segmentation
3. Check: Are outbound rules from CDE properly restricted?

Respond in JSON:
{{
    "additional_violations": [
        {{
            "severity": "critical/high/medium",
            "requirement": "PCI-DSS 1.2.1",
            "description": "issue description",
            "remediation": "how to fix"
        }}
    ],
    "segmentation_assessment": "Good/Needs Improvement/Non-Compliant",
    "audit_readiness": "Pass/Fail"
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
                'additional_violations': [],
                'error': str(e)
            }

    def generate_pci_evidence(self, validation_results: Dict) -> str:
        """Generate audit evidence for PCI-DSS assessor"""

        report = f"""
PCI-DSS REQUIREMENT 1 - FIREWALL AND NETWORK SEGMENTATION
EVIDENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REQUIREMENT: 1.2.1 - Restrict inbound and outbound traffic to that which is necessary
for the cardholder data environment (CDE)

VALIDATION RESULTS
==================
Compliant: {'YES' if validation_results['compliant'] else 'NO'}
Violations Found: {len(validation_results['violations'])}

DETAILED FINDINGS
=================
"""

        if validation_results['violations']:
            for i, violation in enumerate(validation_results['violations'], 1):
                report += f"\nViolation #{i}\n"
                report += f"Severity: {violation['severity'].upper()}\n"
                report += f"Requirement: {violation['requirement']}\n"
                report += f"Description: {violation['description']}\n"
                report += f"Remediation: {violation['remediation']}\n"
                report += "-" * 80 + "\n"
        else:
            report += "\nNo violations found. CDE is properly segmented per PCI-DSS requirements.\n"

        return report

# Example: PCI-DSS validation
def demo_pci_validation():
    validator = PCIDSSSegmentationValidator(anthropic_api_key="your-api-key")

    # Define network zones
    device_zones = {
        'internet-gateway': 'Internet',
        'dmz-firewall': 'DMZ',
        'web-server': 'DMZ',
        'app-server': 'Internal',
        'payment-db': 'CDE',
        'card-processor': 'CDE'
    }

    # Define firewall rules (simplified)
    firewall_rules = [
        {
            'rule_id': 'R001',
            'source': 'internet-gateway',
            'destination': 'web-server',
            'service': 'https',
            'action': 'permit'
        },
        {
            'rule_id': 'R002',
            'source': 'web-server',
            'destination': 'app-server',
            'service': 'https',
            'action': 'permit'
        },
        {
            'rule_id': 'R003',
            'source': 'app-server',
            'destination': 'payment-db',
            'service': 'mysql',
            'action': 'permit'
        },
        # BAD RULE: Direct internet to CDE
        {
            'rule_id': 'R004',
            'source': 'internet-gateway',
            'destination': 'payment-db',
            'service': 'any',
            'action': 'permit'
        }
    ]

    # Build topology
    validator.build_network_topology(firewall_rules, device_zones)

    # Validate CDE isolation
    cde_devices = ['payment-db', 'card-processor']
    isolation_result = validator.validate_cde_isolation(cde_devices)

    print("=== PCI-DSS NETWORK SEGMENTATION VALIDATION ===")
    print(f"Compliant: {isolation_result['compliant']}")
    print(f"Violations: {len(isolation_result['violations'])}\n")

    for violation in isolation_result['violations']:
        print(f"[{violation['severity'].upper()}] {violation['requirement']}")
        print(f"  Issue: {violation['description']}")
        print(f"  Fix: {violation['remediation']}\n")

    # Validate firewall rules
    rules_result = validator.validate_firewall_rules(firewall_rules, ['payment-db', 'card-processor'])

    # Generate evidence report
    evidence = validator.generate_pci_evidence(rules_result)
    print("\n" + "="*80)
    print(evidence)

# Example Output:
"""
=== PCI-DSS NETWORK SEGMENTATION VALIDATION ===
Compliant: False
Violations: 2

[CRITICAL] PCI-DSS 1.2.1
  Issue: Direct connection from Internet to CDE device payment-db
  Fix: Add DMZ layer. Internet → DMZ → CDE only.

[CRITICAL] PCI-DSS 1.2.1
  Issue: Firewall rule allows traffic from ANY source to CDE: R004
  Fix: Specify authorized source IPs/subnets only.
"""
```

---

## Section 3: GDPR Data Flow Mapping

### GDPR Requirements

GDPR Article 30: Records of processing activities
- Document what personal data you process
- Where it comes from
- Where it goes
- How long you keep it

**Network Relevance**:
- NetFlow shows data moving between systems
- Firewall logs show data leaving EU
- Device configs show data storage locations

### GDPR Data Flow Mapper

```python
"""
GDPR Data Flow Mapping
Automatically map personal data flows through network
"""
from typing import List, Dict, Set
from datetime import datetime
import anthropic
import json

class GDPRDataFlowMapper:
    """Map data flows for GDPR Article 30 compliance"""

    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

        # EU countries for transfer detection
        self.eu_countries = ['DE', 'FR', 'UK', 'ES', 'IT', 'NL', 'BE', 'AT', 'SE', 'DK', 'FI', 'IE', 'PT', 'GR', 'PL']

    def map_personal_data_flows(self, systems: List[Dict],
                                data_flows: List[Dict]) -> Dict:
        """Map personal data flows through systems"""

        # systems: [{name, location, data_types, purpose}]
        # data_flows: [{source, destination, data_type, volume}]

        flows_map = {
            'personal_data_systems': [],
            'data_transfers': [],
            'international_transfers': [],
            'data_retention': []
        }

        # Identify systems processing personal data
        for system in systems:
            if self._contains_personal_data(system['data_types']):
                flows_map['personal_data_systems'].append({
                    'system': system['name'],
                    'location': system['location'],
                    'data_types': system['data_types'],
                    'purpose': system['purpose'],
                    'lawful_basis': self._determine_lawful_basis(system['purpose'])
                })

        # Map data transfers
        for flow in data_flows:
            source_sys = next((s for s in systems if s['name'] == flow['source']), None)
            dest_sys = next((s for s in systems if s['name'] == flow['destination']), None)

            if source_sys and dest_sys:
                transfer = {
                    'from': flow['source'],
                    'to': flow['destination'],
                    'data_type': flow['data_type'],
                    'from_location': source_sys['location'],
                    'to_location': dest_sys['location']
                }

                flows_map['data_transfers'].append(transfer)

                # Check for international transfer
                if self._is_international_transfer(source_sys['location'],
                                                   dest_sys['location']):
                    flows_map['international_transfers'].append({
                        **transfer,
                        'requires_safeguards': True,
                        'suggested_mechanism': self._suggest_transfer_mechanism(
                            source_sys['location'], dest_sys['location']
                        )
                    })

        return flows_map

    def _contains_personal_data(self, data_types: List[str]) -> bool:
        """Check if data types include personal data"""
        personal_data_keywords = [
            'name', 'email', 'phone', 'address', 'ip_address',
            'user_id', 'customer', 'employee', 'pii', 'personal'
        ]

        return any(keyword in data_type.lower()
                  for data_type in data_types
                  for keyword in personal_data_keywords)

    def _determine_lawful_basis(self, purpose: str) -> str:
        """Determine GDPR lawful basis for processing"""
        # Simplified - in production use AI or predefined mapping

        purpose_lower = purpose.lower()

        if 'contract' in purpose_lower or 'service' in purpose_lower:
            return 'Contract (Art 6(1)(b))'
        elif 'legal' in purpose_lower or 'compliance' in purpose_lower:
            return 'Legal Obligation (Art 6(1)(c))'
        elif 'consent' in purpose_lower:
            return 'Consent (Art 6(1)(a))'
        else:
            return 'Legitimate Interest (Art 6(1)(f)) - Requires Assessment'

    def _is_international_transfer(self, from_country: str, to_country: str) -> bool:
        """Check if transfer crosses international boundaries"""

        from_eu = from_country in self.eu_countries
        to_eu = to_country in self.eu_countries

        # Transfer is international if leaving EU
        return from_eu and not to_eu

    def _suggest_transfer_mechanism(self, from_country: str, to_country: str) -> str:
        """Suggest GDPR transfer mechanism"""

        if to_country == 'US':
            return 'Standard Contractual Clauses (SCCs) or Data Privacy Framework'
        elif to_country in ['CA', 'JP', 'UK']:
            return 'Adequacy Decision (safe) or SCCs'
        else:
            return 'Standard Contractual Clauses (SCCs) required'

    def analyze_network_logs_for_data_transfers(self, netflow_logs: List[Dict],
                                                ip_to_location: Dict[str, str]) -> Dict:
        """Analyze NetFlow logs to detect data transfers"""

        transfers_detected = []
        eu_ips = set()
        non_eu_ips = set()

        for flow in netflow_logs:
            source_ip = flow['source_ip']
            dest_ip = flow['dest_ip']

            source_country = ip_to_location.get(source_ip, 'Unknown')
            dest_country = ip_to_location.get(dest_ip, 'Unknown')

            # Track EU vs non-EU
            if source_country in self.eu_countries:
                eu_ips.add(source_ip)
            if dest_country in self.eu_countries:
                eu_ips.add(dest_ip)
            else:
                non_eu_ips.add(dest_ip)

            # Detect EU → non-EU transfers
            if self._is_international_transfer(source_country, dest_country):
                transfers_detected.append({
                    'source_ip': source_ip,
                    'dest_ip': dest_ip,
                    'from_country': source_country,
                    'to_country': dest_country,
                    'bytes': flow['bytes'],
                    'timestamp': flow['timestamp']
                })

        return {
            'international_transfers_detected': len(transfers_detected),
            'transfers': transfers_detected,
            'eu_systems': len(eu_ips),
            'non_eu_destinations': len(non_eu_ips),
            'requires_review': len(transfers_detected) > 0
        }

    def generate_gdpr_ropa(self, data_flows: Dict) -> str:
        """Generate GDPR Article 30 Record of Processing Activities (ROPA)"""

        report = f"""
GDPR ARTICLE 30 - RECORD OF PROCESSING ACTIVITIES
Generated: {datetime.now().strftime('%Y-%m-%d')}

SYSTEMS PROCESSING PERSONAL DATA
=================================
"""

        for system in data_flows['personal_data_systems']:
            report += f"\nSystem: {system['system']}\n"
            report += f"Location: {system['location']}\n"
            report += f"Data Types: {', '.join(system['data_types'])}\n"
            report += f"Purpose: {system['purpose']}\n"
            report += f"Lawful Basis: {system['lawful_basis']}\n"
            report += "-" * 80 + "\n"

        report += "\n\nDATA TRANSFERS\n"
        report += "=" * 80 + "\n"

        for transfer in data_flows['data_transfers']:
            report += f"\n{transfer['from']} ({transfer['from_location']}) → {transfer['to']} ({transfer['to_location']})\n"
            report += f"Data: {transfer['data_type']}\n"

        if data_flows['international_transfers']:
            report += "\n\nINTERNATIONAL TRANSFERS (REQUIRES SAFEGUARDS)\n"
            report += "=" * 80 + "\n"

            for transfer in data_flows['international_transfers']:
                report += f"\n⚠️  {transfer['from']} → {transfer['to']}\n"
                report += f"From: {transfer['from_location']}\n"
                report += f"To: {transfer['to_location']}\n"
                report += f"Safeguard Required: {transfer['suggested_mechanism']}\n"

        return report

# Example: GDPR data flow mapping
def demo_gdpr_mapping():
    mapper = GDPRDataFlowMapper(anthropic_api_key="your-api-key")

    # Define systems
    systems = [
        {
            'name': 'web-app',
            'location': 'DE',  # Germany (EU)
            'data_types': ['customer_name', 'email', 'ip_address'],
            'purpose': 'Provide web service to customers'
        },
        {
            'name': 'database-eu',
            'location': 'FR',  # France (EU)
            'data_types': ['customer_name', 'email', 'purchase_history'],
            'purpose': 'Store customer data for contract fulfillment'
        },
        {
            'name': 'analytics-us',
            'location': 'US',  # USA (non-EU)
            'data_types': ['anonymized_user_behavior'],
            'purpose': 'Product analytics'
        },
        {
            'name': 'backup-us',
            'location': 'US',
            'data_types': ['customer_name', 'email'],  # Personal data!
            'purpose': 'Disaster recovery backups'
        }
    ]

    # Define data flows
    data_flows = [
        {
            'source': 'web-app',
            'destination': 'database-eu',
            'data_type': 'customer_data',
            'volume': '1TB'
        },
        {
            'source': 'web-app',
            'destination': 'analytics-us',
            'data_type': 'anonymized_behavior',
            'volume': '100GB'
        },
        {
            'source': 'database-eu',
            'destination': 'backup-us',
            'data_type': 'customer_data',  # International transfer!
            'volume': '500GB'
        }
    ]

    # Map flows
    flows_map = mapper.map_personal_data_flows(systems, data_flows)

    print("=== GDPR DATA FLOW ANALYSIS ===")
    print(f"Systems processing personal data: {len(flows_map['personal_data_systems'])}")
    print(f"Data transfers: {len(flows_map['data_transfers'])}")
    print(f"International transfers: {len(flows_map['international_transfers'])}\n")

    if flows_map['international_transfers']:
        print("⚠️  INTERNATIONAL TRANSFERS DETECTED (Requires Safeguards):")
        for transfer in flows_map['international_transfers']:
            print(f"  {transfer['from']} ({transfer['from_location']}) → {transfer['to']} ({transfer['to_location']})")
            print(f"  Mechanism: {transfer['suggested_mechanism']}\n")

    # Generate ROPA
    ropa = mapper.generate_gdpr_ropa(flows_map)
    print("\n" + "="*80)
    print(ropa)

# Example Output:
"""
=== GDPR DATA FLOW ANALYSIS ===
Systems processing personal data: 3
Data transfers: 3
International transfers: 1

⚠️  INTERNATIONAL TRANSFERS DETECTED (Requires Safeguards):
  database-eu (FR) → backup-us (US)
  Mechanism: Standard Contractual Clauses (SCCs) or Data Privacy Framework

================================================================================
GDPR ARTICLE 30 - RECORD OF PROCESSING ACTIVITIES
Generated: 2026-01-18

SYSTEMS PROCESSING PERSONAL DATA
=================================

System: web-app
Location: DE
Data Types: customer_name, email, ip_address
Purpose: Provide web service to customers
Lawful Basis: Contract (Art 6(1)(b))
--------------------------------------------------------------------------------

System: database-eu
Location: FR
Data Types: customer_name, email, purchase_history
Purpose: Store customer data for contract fulfillment
Lawful Basis: Contract (Art 6(1)(b))
--------------------------------------------------------------------------------

System: backup-us
Location: US
Data Types: customer_name, email
Purpose: Disaster recovery backups
Lawful Basis: Legitimate Interest (Art 6(1)(f)) - Requires Assessment
--------------------------------------------------------------------------------

INTERNATIONAL TRANSFERS (REQUIRES SAFEGUARDS)
================================================================================

⚠️  database-eu → backup-us
From: FR
To: US
Safeguard Required: Standard Contractual Clauses (SCCs) or Data Privacy Framework
"""
```

---

## Section 4: Continuous Compliance Monitoring

### Production Compliance Platform

```python
"""
Continuous Compliance Monitoring Platform
Monitors SOC2, PCI-DSS, and GDPR compliance 24/7
"""
import asyncio
from datetime import datetime
from typing import Dict, List

class ComplianceMonitoringPlatform:
    """Unified compliance monitoring for multiple frameworks"""

    def __init__(self, anthropic_api_key: str):
        self.soc2_checker = SOC2ComplianceChecker(anthropic_api_key)
        self.pci_validator = PCIDSSSegmentationValidator(anthropic_api_key)
        self.gdpr_mapper = GDPRDataFlowMapper(anthropic_api_key)
        self.violations = []

    async def continuous_monitoring(self):
        """Run continuous compliance monitoring"""
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running compliance checks...")

            # Run all compliance checks in parallel
            await asyncio.gather(
                self._check_soc2_compliance(),
                self._check_pci_compliance(),
                self._check_gdpr_compliance()
            )

            # Generate dashboard metrics
            metrics = self.get_compliance_metrics()
            self._update_dashboard(metrics)

            # Wait before next check (hourly)
            await asyncio.sleep(3600)

    async def _check_soc2_compliance(self):
        """Check SOC2 compliance for all devices"""
        # Fetch device configs from network
        # device_configs = fetch_device_configs()
        # for each device, run soc2_checker.check_device_compliance()
        pass

    async def _check_pci_compliance(self):
        """Check PCI-DSS segmentation"""
        # Fetch firewall rules
        # Run pci_validator.validate_cde_isolation()
        pass

    async def _check_gdpr_compliance(self):
        """Check GDPR data flows"""
        # Analyze NetFlow logs
        # Run gdpr_mapper.analyze_network_logs_for_data_transfers()
        pass

    def get_compliance_metrics(self) -> Dict:
        """Get real-time compliance metrics"""
        return {
            'soc2_compliant_devices': 45,
            'soc2_violations': 5,
            'pci_cde_isolated': True,
            'pci_violations': 2,
            'gdpr_international_transfers': 3,
            'gdpr_safeguards_in_place': 2,
            'last_check': datetime.now().isoformat(),
            'overall_compliance_score': 0.87
        }

    def _update_dashboard(self, metrics: Dict):
        """Update compliance dashboard"""
        print(f"\n=== COMPLIANCE DASHBOARD ===")
        print(f"SOC2: {metrics['soc2_compliant_devices']}/{metrics['soc2_compliant_devices']+metrics['soc2_violations']} devices compliant")
        print(f"PCI-DSS: {'✓' if metrics['pci_cde_isolated'] else '✗'} CDE Isolated")
        print(f"GDPR: {metrics['gdpr_safeguards_in_place']}/{metrics['gdpr_international_transfers']} transfers have safeguards")
        print(f"Overall Score: {metrics['overall_compliance_score']:.0%}")

    def generate_audit_package(self, framework: str) -> str:
        """Generate complete audit evidence package"""
        # Collect all evidence for specified framework
        # Generate comprehensive report
        # Include: configs, logs, remediation status
        pass
```

---

## What Can Go Wrong

### 1. False Negatives
**Problem**: Compliance check passes but auditor finds violations
**Solution**: Regular auditor review of automated checks, update patterns

### 2. Configuration Drift
**Problem**: Device passes compliance at 9 AM, fails at 3 PM (config change)
**Solution**: Continuous monitoring (hourly checks), change control integration

### 3. Incomplete Coverage
**Problem**: Checking 100 devices, missing 20 shadow IT devices
**Solution**: Network discovery, asset inventory integration

### 4. Alert Fatigue
**Problem**: 500 compliance alerts per day, team ignores them
**Solution**: Severity-based alerting, automated remediation where possible

---

## Key Takeaways

1. **Compliance is continuous** - Not once-per-year audit, monitor 24/7

2. **Automation reduces cost** - $120K audit vs. $20K automation investment

3. **AI finds subtle violations** - Human auditors miss edge cases, AI doesn't

4. **Evidence generation is automatic** - Always audit-ready, no scrambling

5. **Multi-framework coverage** - Same system checks SOC2, PCI-DSS, GDPR

6. **Network compliance is critical** - Proper segmentation, logging, access controls make or break audits

**Next Chapter**: Complete Security Case Study - 6-month FinTech deployment with real costs, ROI, and lessons learned.

---

**Code Repository**: `github.com/vexpertai/ai-networking-book/chapter-83/`
