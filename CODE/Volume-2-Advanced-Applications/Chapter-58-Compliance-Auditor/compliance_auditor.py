"""
Chapter 58: AI Compliance Auditor
Automated compliance checking for GDPR, SOC2, HIPAA, ISO 27001
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


# ============================================================================
# Data Models
# ============================================================================

class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    GDPR = "GDPR"
    SOC2 = "SOC2"
    HIPAA = "HIPAA"
    ISO27001 = "ISO27001"
    PCI_DSS = "PCI_DSS"


class ComplianceStatus(str, Enum):
    """Compliance check status"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class RiskLevel(str, Enum):
    """Risk level for non-compliance"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class RemediationPriority(str, Enum):
    """Remediation priority"""
    P0_IMMEDIATE = "P0_IMMEDIATE"  # Fix within 24h
    P1_URGENT = "P1_URGENT"  # Fix within 1 week
    P2_HIGH = "P2_HIGH"  # Fix within 1 month
    P3_MEDIUM = "P3_MEDIUM"  # Fix within 3 months
    P4_LOW = "P4_LOW"  # Plan for next quarter


@dataclass
class ComplianceRequirement:
    """Single compliance requirement"""
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    control_category: str  # Access Control, Encryption, Logging, etc.
    validation_criteria: List[str]
    evidence_required: List[str]


@dataclass
class ComplianceCheck:
    """Result of a compliance check"""
    requirement: ComplianceRequirement
    status: ComplianceStatus
    risk_level: RiskLevel
    findings: List[str]
    evidence_found: List[str]
    evidence_missing: List[str]
    remediation_steps: List[str]
    priority: RemediationPriority
    checked_at: datetime
    next_check_due: datetime


@dataclass
class DataAsset:
    """Data asset subject to compliance"""
    asset_id: str
    name: str
    data_type: str  # PII, PHI, Financial, etc.
    location: str
    owner: str
    retention_days: int
    encryption_enabled: bool
    access_logs_enabled: bool
    last_accessed: datetime
    created_at: datetime


@dataclass
class AIDecision:
    """AI system decision requiring audit trail"""
    decision_id: str
    timestamp: datetime
    system_name: str
    input_data: Dict
    output_decision: str
    confidence_score: float
    model_version: str
    explanation: str
    human_review_required: bool
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


@dataclass
class AuditTrail:
    """Audit trail entry"""
    event_id: str
    timestamp: datetime
    user: str
    action: str
    resource: str
    result: str  # SUCCESS, FAILURE, DENIED
    ip_address: str
    details: Dict
    compliance_relevant: bool = False


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA)"""
    pia_id: str
    project_name: str
    data_processed: List[str]
    processing_purpose: str
    legal_basis: str  # Consent, Contract, Legal Obligation, etc.
    data_subjects: List[str]
    third_parties: List[str]
    retention_period: int
    security_measures: List[str]
    risks_identified: List[str]
    mitigation_measures: List[str]
    dpo_approval: bool
    approved_at: Optional[datetime] = None


@dataclass
class ComplianceReport:
    """Overall compliance report"""
    report_id: str
    framework: ComplianceFramework
    generated_at: datetime
    scope: str
    total_requirements: int
    compliant: int
    non_compliant: int
    partially_compliant: int
    not_applicable: int
    requires_review: int
    overall_compliance_score: float  # 0-100
    critical_findings: List[ComplianceCheck]
    recommendations: List[str]
    next_audit_due: datetime


# ============================================================================
# Pydantic Models for LLM Structured Outputs
# ============================================================================

class ComplianceAnalysis(BaseModel):
    """LLM analysis of compliance status"""
    status: str = Field(description="COMPLIANT, NON_COMPLIANT, PARTIALLY_COMPLIANT, or REQUIRES_REVIEW")
    risk_level: str = Field(description="CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL")
    findings: List[str] = Field(description="List of specific findings (2-5 items)")
    evidence_found: List[str] = Field(description="Evidence that supports compliance")
    evidence_missing: List[str] = Field(description="Missing evidence or gaps")
    remediation_steps: List[str] = Field(description="Concrete steps to achieve compliance (2-5 items)")
    priority: str = Field(description="P0_IMMEDIATE, P1_URGENT, P2_HIGH, P3_MEDIUM, or P4_LOW")


class AIExplainability(BaseModel):
    """LLM explanation of AI decision"""
    decision_summary: str = Field(description="Brief summary of what the AI decided")
    key_factors: List[str] = Field(description="Key factors that influenced the decision (3-5 items)")
    confidence_explanation: str = Field(description="Why this confidence score (one sentence)")
    potential_biases: List[str] = Field(description="Potential biases or limitations (1-3 items)")
    human_review_reasoning: str = Field(description="Why human review is or isn't needed (one sentence)")
    compliance_considerations: List[str] = Field(description="Relevant compliance considerations (2-3 items)")


class DataRetentionRecommendation(BaseModel):
    """LLM recommendation for data retention"""
    should_retain: bool = Field(description="True if data should be retained, False if eligible for deletion")
    reasoning: str = Field(description="Legal/business reasoning for retention decision")
    applicable_regulations: List[str] = Field(description="Which regulations apply (e.g., GDPR Art. 5, HIPAA 164.316)")
    retention_period_days: int = Field(description="Recommended retention period in days")
    deletion_method: str = Field(description="Recommended deletion method (e.g., secure wipe, anonymization)")
    risk_assessment: str = Field(description="Risk of retaining vs. deleting this data")


# ============================================================================
# Main Compliance Auditor
# ============================================================================

class ComplianceAuditor:
    """AI-powered compliance auditing system"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=self.api_key,
            temperature=0  # Deterministic for compliance
        )

        # Compliance requirements database (sample)
        self.requirements = self._load_requirements()

        # Audit trails
        self.audit_trails: List[AuditTrail] = []

    def _load_requirements(self) -> Dict[ComplianceFramework, List[ComplianceRequirement]]:
        """Load compliance requirements (simplified for examples)"""
        return {
            ComplianceFramework.GDPR: [
                ComplianceRequirement(
                    requirement_id="GDPR-5.1.f",
                    framework=ComplianceFramework.GDPR,
                    title="Data Integrity and Confidentiality",
                    description="Personal data must be processed securely using encryption and access controls",
                    control_category="Security",
                    validation_criteria=[
                        "Data at rest encryption enabled",
                        "Data in transit encryption (TLS 1.2+)",
                        "Access controls implemented",
                        "Regular security audits conducted"
                    ],
                    evidence_required=[
                        "Encryption configuration",
                        "Access control logs",
                        "Security audit reports"
                    ]
                ),
                ComplianceRequirement(
                    requirement_id="GDPR-5.1.e",
                    framework=ComplianceFramework.GDPR,
                    title="Storage Limitation",
                    description="Personal data kept in a form which permits identification for no longer than necessary",
                    control_category="Data Retention",
                    validation_criteria=[
                        "Data retention policies defined",
                        "Automated deletion processes in place",
                        "Regular data inventory reviews"
                    ],
                    evidence_required=[
                        "Retention policy document",
                        "Deletion logs",
                        "Data inventory reports"
                    ]
                ),
                ComplianceRequirement(
                    requirement_id="GDPR-15",
                    framework=ComplianceFramework.GDPR,
                    title="Right of Access",
                    description="Data subjects have the right to access their personal data",
                    control_category="Data Subject Rights",
                    validation_criteria=[
                        "Subject access request (SAR) process documented",
                        "SAR response within 30 days",
                        "Data portability capability"
                    ],
                    evidence_required=[
                        "SAR process documentation",
                        "SAR response logs",
                        "Data export functionality"
                    ]
                )
            ],
            ComplianceFramework.SOC2: [
                ComplianceRequirement(
                    requirement_id="CC6.1",
                    framework=ComplianceFramework.SOC2,
                    title="Logical and Physical Access Controls",
                    description="The entity implements logical access security software and controls",
                    control_category="Access Control",
                    validation_criteria=[
                        "Multi-factor authentication enabled",
                        "Role-based access control (RBAC)",
                        "Access review process (quarterly)",
                        "Privileged access management"
                    ],
                    evidence_required=[
                        "Access control configuration",
                        "Access review reports",
                        "MFA enforcement logs"
                    ]
                ),
                ComplianceRequirement(
                    requirement_id="CC7.2",
                    framework=ComplianceFramework.SOC2,
                    title="System Monitoring",
                    description="The entity monitors system components and alerts on anomalies",
                    control_category="Monitoring",
                    validation_criteria=[
                        "Security monitoring tools deployed",
                        "Alert thresholds configured",
                        "24/7 monitoring coverage",
                        "Incident response procedures"
                    ],
                    evidence_required=[
                        "Monitoring tool configuration",
                        "Alert logs",
                        "Incident response records"
                    ]
                )
            ],
            ComplianceFramework.HIPAA: [
                ComplianceRequirement(
                    requirement_id="164.312(a)(1)",
                    framework=ComplianceFramework.HIPAA,
                    title="Access Control",
                    description="Implement technical policies to allow only authorized access to ePHI",
                    control_category="Access Control",
                    validation_criteria=[
                        "Unique user identification",
                        "Emergency access procedure",
                        "Automatic logoff",
                        "Encryption and decryption"
                    ],
                    evidence_required=[
                        "Access control policies",
                        "User access logs",
                        "Encryption evidence"
                    ]
                ),
                ComplianceRequirement(
                    requirement_id="164.312(b)",
                    framework=ComplianceFramework.HIPAA,
                    title="Audit Controls",
                    description="Implement hardware, software, and procedural mechanisms to record and examine ePHI access",
                    control_category="Logging",
                    validation_criteria=[
                        "Audit logs capture all ePHI access",
                        "Logs retained for 6 years",
                        "Regular log review process",
                        "Tamper-proof log storage"
                    ],
                    evidence_required=[
                        "Audit log configuration",
                        "Log retention policy",
                        "Log review reports"
                    ]
                )
            ]
        }

    def check_requirement(self, requirement: ComplianceRequirement,
                         system_config: Dict, evidence: List[str]) -> ComplianceCheck:
        """
        Check a single compliance requirement using AI analysis

        Args:
            requirement: The compliance requirement to check
            system_config: Current system configuration
            evidence: Available evidence documents/logs

        Returns:
            ComplianceCheck with detailed findings
        """
        # Prepare prompt for LLM
        parser = PydanticOutputParser(pydantic_object=ComplianceAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a compliance auditor specializing in {framework}.
Analyze the system against the requirement and provide structured findings.
Be strict but fair - only mark COMPLIANT if all criteria are fully met.

{format_instructions}"""),
            ("human", """Requirement: {requirement_title}
Description: {requirement_description}

Validation Criteria:
{validation_criteria}

Evidence Required:
{evidence_required}

Current System Configuration:
{system_config}

Available Evidence:
{evidence}

Analyze compliance status and provide detailed findings.""")
        ])

        chain = prompt | self.llm | parser

        try:
            analysis = chain.invoke({
                "framework": requirement.framework.value,
                "format_instructions": parser.get_format_instructions(),
                "requirement_title": requirement.title,
                "requirement_description": requirement.description,
                "validation_criteria": "\n".join(f"- {c}" for c in requirement.validation_criteria),
                "evidence_required": "\n".join(f"- {e}" for e in requirement.evidence_required),
                "system_config": str(system_config),
                "evidence": "\n".join(f"- {e}" for e in evidence)
            })

            # Convert to ComplianceCheck
            now = datetime.now()
            check = ComplianceCheck(
                requirement=requirement,
                status=ComplianceStatus[analysis.status],
                risk_level=RiskLevel[analysis.risk_level],
                findings=analysis.findings,
                evidence_found=analysis.evidence_found,
                evidence_missing=analysis.evidence_missing,
                remediation_steps=analysis.remediation_steps,
                priority=RemediationPriority[analysis.priority],
                checked_at=now,
                next_check_due=now + timedelta(days=90)  # Quarterly reviews
            )

            return check

        except Exception as e:
            # Fallback for errors
            return ComplianceCheck(
                requirement=requirement,
                status=ComplianceStatus.REQUIRES_REVIEW,
                risk_level=RiskLevel.MEDIUM,
                findings=[f"Automated check failed: {str(e)}"],
                evidence_found=[],
                evidence_missing=requirement.evidence_required,
                remediation_steps=["Manual review required"],
                priority=RemediationPriority.P2_HIGH,
                checked_at=datetime.now(),
                next_check_due=datetime.now() + timedelta(days=7)
            )

    def audit_framework(self, framework: ComplianceFramework,
                       system_config: Dict, evidence: Dict[str, List[str]]) -> ComplianceReport:
        """
        Audit entire compliance framework

        Args:
            framework: Which framework to audit (GDPR, SOC2, HIPAA)
            system_config: Current system configuration
            evidence: Available evidence by requirement_id

        Returns:
            ComplianceReport with all findings
        """
        requirements = self.requirements.get(framework, [])
        checks: List[ComplianceCheck] = []

        for req in requirements:
            req_evidence = evidence.get(req.requirement_id, [])
            check = self.check_requirement(req, system_config, req_evidence)
            checks.append(check)

        # Calculate statistics
        total = len(checks)
        compliant = len([c for c in checks if c.status == ComplianceStatus.COMPLIANT])
        non_compliant = len([c for c in checks if c.status == ComplianceStatus.NON_COMPLIANT])
        partially_compliant = len([c for c in checks if c.status == ComplianceStatus.PARTIALLY_COMPLIANT])
        not_applicable = len([c for c in checks if c.status == ComplianceStatus.NOT_APPLICABLE])
        requires_review = len([c for c in checks if c.status == ComplianceStatus.REQUIRES_REVIEW])

        # Compliance score (compliant + 0.5*partially / total applicable)
        applicable = total - not_applicable
        score = 0.0
        if applicable > 0:
            score = ((compliant + 0.5 * partially_compliant) / applicable) * 100

        # Critical findings
        critical = [c for c in checks if c.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]]
        critical.sort(key=lambda c: (c.risk_level.value, c.priority.value))

        # Generate recommendations
        recommendations = self._generate_recommendations(checks, framework)

        return ComplianceReport(
            report_id=f"{framework.value}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            framework=framework,
            generated_at=datetime.now(),
            scope="Full system audit",
            total_requirements=total,
            compliant=compliant,
            non_compliant=non_compliant,
            partially_compliant=partially_compliant,
            not_applicable=not_applicable,
            requires_review=requires_review,
            overall_compliance_score=score,
            critical_findings=critical[:10],  # Top 10
            recommendations=recommendations,
            next_audit_due=datetime.now() + timedelta(days=90)
        )

    def _generate_recommendations(self, checks: List[ComplianceCheck],
                                  framework: ComplianceFramework) -> List[str]:
        """Generate top recommendations based on findings"""
        recommendations = []

        # Group by priority
        p0_count = len([c for c in checks if c.priority == RemediationPriority.P0_IMMEDIATE])
        p1_count = len([c for c in checks if c.priority == RemediationPriority.P1_URGENT])

        if p0_count > 0:
            recommendations.append(f"URGENT: {p0_count} critical issues require immediate remediation (within 24h)")

        if p1_count > 0:
            recommendations.append(f"HIGH PRIORITY: {p1_count} urgent issues require remediation within 1 week")

        # Common gaps
        encryption_issues = [c for c in checks if "encryption" in c.requirement.title.lower()
                            and c.status != ComplianceStatus.COMPLIANT]
        if encryption_issues:
            recommendations.append("Encryption gaps identified - review data protection measures")

        logging_issues = [c for c in checks if "log" in c.requirement.title.lower()
                         and c.status != ComplianceStatus.COMPLIANT]
        if logging_issues:
            recommendations.append("Audit logging requires improvement - ensure comprehensive activity tracking")

        access_issues = [c for c in checks if "access" in c.requirement.title.lower()
                        and c.status != ComplianceStatus.COMPLIANT]
        if access_issues:
            recommendations.append("Access control enhancements needed - review IAM policies and MFA enforcement")

        return recommendations[:5]  # Top 5

    def explain_ai_decision(self, decision: AIDecision) -> str:
        """
        Generate compliance-friendly explanation of AI decision

        Args:
            decision: The AI decision to explain

        Returns:
            Human-readable explanation suitable for audit trails
        """
        parser = PydanticOutputParser(pydantic_object=AIExplainability)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI explainability expert.
Provide clear, compliance-friendly explanations of AI decisions.
Focus on transparency, fairness, and regulatory requirements (GDPR Art. 22, etc.).

{format_instructions}"""),
            ("human", """AI System: {system_name}
Model Version: {model_version}
Timestamp: {timestamp}

Input Data:
{input_data}

Output Decision: {output_decision}
Confidence: {confidence_score:.2%}

Provide a compliance-ready explanation of this decision.""")
        ])

        chain = prompt | self.llm | parser

        explanation = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "system_name": decision.system_name,
            "model_version": decision.model_version,
            "timestamp": decision.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "input_data": str(decision.input_data),
            "output_decision": decision.output_decision,
            "confidence_score": decision.confidence_score
        })

        # Format as audit-friendly text
        audit_text = f"""AI DECISION EXPLANATION
========================
Decision ID: {decision.decision_id}
System: {decision.system_name} (v{decision.model_version})
Timestamp: {decision.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

DECISION SUMMARY
{explanation.decision_summary}

KEY FACTORS
{chr(10).join(f'{i+1}. {factor}' for i, factor in enumerate(explanation.key_factors))}

CONFIDENCE EXPLANATION
{explanation.confidence_explanation}

POTENTIAL LIMITATIONS
{chr(10).join(f'- {bias}' for bias in explanation.potential_biases)}

HUMAN REVIEW
{explanation.human_review_reasoning}

COMPLIANCE CONSIDERATIONS
{chr(10).join(f'- {consideration}' for consideration in explanation.compliance_considerations)}

Original Decision: {decision.output_decision}
Confidence: {decision.confidence_score:.2%}
"""

        return audit_text

    def check_data_retention(self, asset: DataAsset) -> Dict:
        """
        Check if data should be retained or deleted per compliance rules

        Args:
            asset: The data asset to evaluate

        Returns:
            Retention recommendation with reasoning
        """
        parser = PydanticOutputParser(pydantic_object=DataRetentionRecommendation)

        # Calculate age
        age_days = (datetime.now() - asset.created_at).days
        days_since_access = (datetime.now() - asset.last_accessed).days

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data privacy officer expert in GDPR, HIPAA, and data retention regulations.
Evaluate whether data should be retained or deleted based on legal requirements and business needs.

{format_instructions}"""),
            ("human", """Data Asset: {name}
Type: {data_type}
Owner: {owner}
Current Retention Policy: {retention_days} days

Age: {age_days} days
Last Accessed: {days_since_access} days ago
Encryption: {encryption_enabled}
Access Logs: {access_logs_enabled}

Should this data be retained or deleted?""")
        ])

        chain = prompt | self.llm | parser

        recommendation = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "name": asset.name,
            "data_type": asset.data_type,
            "owner": asset.owner,
            "retention_days": asset.retention_days,
            "age_days": age_days,
            "days_since_access": days_since_access,
            "encryption_enabled": "Yes" if asset.encryption_enabled else "No",
            "access_logs_enabled": "Yes" if asset.access_logs_enabled else "No"
        })

        return {
            "asset_id": asset.asset_id,
            "asset_name": asset.name,
            "should_retain": recommendation.should_retain,
            "reasoning": recommendation.reasoning,
            "applicable_regulations": recommendation.applicable_regulations,
            "recommended_retention_days": recommendation.retention_period_days,
            "deletion_method": recommendation.deletion_method,
            "risk_assessment": recommendation.risk_assessment,
            "action_required": "RETAIN" if recommendation.should_retain else "DELETE",
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_audit_trail(self, event_type: str, user: str, resource: str,
                            result: str, details: Dict) -> AuditTrail:
        """
        Generate audit trail entry

        Args:
            event_type: Type of event (ACCESS, MODIFY, DELETE, EXPORT, etc.)
            user: User who performed action
            resource: Resource accessed/modified
            result: SUCCESS, FAILURE, or DENIED
            details: Additional details

        Returns:
            AuditTrail entry
        """
        # Determine if compliance-relevant
        compliance_keywords = ["pii", "phi", "personal", "sensitive", "confidential",
                              "export", "delete", "access", "modify"]
        is_compliance = any(keyword in resource.lower() or keyword in event_type.lower()
                           for keyword in compliance_keywords)

        trail = AuditTrail(
            event_id=f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(user) % 10000:04d}",
            timestamp=datetime.now(),
            user=user,
            action=event_type,
            resource=resource,
            result=result,
            ip_address=details.get("ip_address", "unknown"),
            details=details,
            compliance_relevant=is_compliance
        )

        self.audit_trails.append(trail)
        return trail


# ============================================================================
# Examples
# ============================================================================

def example_1_gdpr_compliance_check():
    """Example 1: Check GDPR compliance"""
    print("=" * 80)
    print("Example 1: GDPR Compliance Check")
    print("=" * 80)

    auditor = ComplianceAuditor()

    # Sample system configuration
    system_config = {
        "encryption_at_rest": True,
        "encryption_in_transit": "TLS 1.3",
        "access_control": "RBAC enabled",
        "mfa_enabled": True,
        "data_retention_policy": "90 days for logs, 2 years for customer data",
        "sar_process": "Documented, average response time 25 days",
        "deletion_automation": "Monthly cleanup jobs"
    }

    # Sample evidence
    evidence = {
        "GDPR-5.1.f": [
            "Encryption: AES-256 for data at rest",
            "TLS 1.3 enforced for all API endpoints",
            "Access logs show RBAC enforcement",
            "Security audit completed Q4 2025"
        ],
        "GDPR-5.1.e": [
            "Retention policy document v2.1",
            "Automated deletion logs show monthly cleanup",
            "Data inventory reviewed quarterly"
        ],
        "GDPR-15": [
            "SAR process documented in DPA handbook",
            "15 SARs processed in last 90 days, avg response 22 days",
            "Data export API available"
        ]
    }

    print("\nRunning GDPR audit...")
    print(f"System: {system_config['encryption_at_rest']}, {system_config['mfa_enabled']}")

    report = auditor.audit_framework(ComplianceFramework.GDPR, system_config, evidence)

    print(f"\n{'='*80}")
    print(f"GDPR COMPLIANCE REPORT")
    print(f"{'='*80}")
    print(f"Report ID: {report.report_id}")
    print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall Compliance Score: {report.overall_compliance_score:.1f}%\n")

    print(f"Requirements Status:")
    print(f"  ✓ Compliant:           {report.compliant}")
    print(f"  ✗ Non-Compliant:       {report.non_compliant}")
    print(f"  ⚠ Partially Compliant: {report.partially_compliant}")
    print(f"  - Not Applicable:      {report.not_applicable}")
    print(f"  ? Requires Review:     {report.requires_review}")

    if report.critical_findings:
        print(f"\nCritical Findings ({len(report.critical_findings)}):")
        for i, finding in enumerate(report.critical_findings[:3], 1):
            print(f"\n{i}. {finding.requirement.requirement_id}: {finding.requirement.title}")
            print(f"   Status: {finding.status.value} | Risk: {finding.risk_level.value} | Priority: {finding.priority.value}")
            print(f"   Findings:")
            for f in finding.findings[:2]:
                print(f"   - {f}")

    if report.recommendations:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

    print(f"\nNext Audit Due: {report.next_audit_due.strftime('%Y-%m-%d')}")


def example_2_ai_decision_explainability():
    """Example 2: Explain AI decision for compliance"""
    print("\n" + "=" * 80)
    print("Example 2: AI Decision Explainability")
    print("=" * 80)

    auditor = ComplianceAuditor()

    # Sample AI decision
    decision = AIDecision(
        decision_id="DEC-20260118-001",
        timestamp=datetime.now(),
        system_name="Network Access Control AI",
        input_data={
            "user": "contractor@external.com",
            "requested_access": "Production Database",
            "time": "02:30 AM",
            "location": "Unknown IP (Germany)",
            "risk_score": 0.78
        },
        output_decision="ACCESS DENIED",
        confidence_score=0.92,
        model_version="v2.3.1",
        explanation="High-risk access attempt detected",
        human_review_required=True
    )

    print("\nAI Decision:")
    print(f"System: {decision.system_name}")
    print(f"Decision: {decision.output_decision} (confidence: {decision.confidence_score:.2%})")
    print(f"Input: Contractor requesting production DB access at 2:30 AM")

    print("\nGenerating compliance-friendly explanation...")
    explanation = auditor.explain_ai_decision(decision)

    print(f"\n{explanation}")


def example_3_data_retention_check():
    """Example 3: Check data retention compliance"""
    print("\n" + "=" * 80)
    print("Example 3: Data Retention Compliance")
    print("=" * 80)

    auditor = ComplianceAuditor()

    # Sample data assets
    assets = [
        DataAsset(
            asset_id="DA-001",
            name="Customer PII Database",
            data_type="PII",
            location="EU-West-1",
            owner="CustomerSuccess",
            retention_days=730,  # 2 years
            encryption_enabled=True,
            access_logs_enabled=True,
            last_accessed=datetime.now() - timedelta(days=15),
            created_at=datetime.now() - timedelta(days=800)  # Over retention
        ),
        DataAsset(
            asset_id="DA-002",
            name="Debug Logs 2023",
            data_type="Operational Logs",
            location="US-East-1",
            owner="Engineering",
            retention_days=90,
            encryption_enabled=False,
            access_logs_enabled=True,
            last_accessed=datetime.now() - timedelta(days=400),
            created_at=datetime.now() - timedelta(days=900)
        )
    ]

    print("\nChecking data retention policies...\n")

    for asset in assets:
        age_days = (datetime.now() - asset.created_at).days
        print(f"Asset: {asset.name}")
        print(f"  Type: {asset.data_type}")
        print(f"  Age: {age_days} days (policy: {asset.retention_days} days)")
        print(f"  Last accessed: {(datetime.now() - asset.last_accessed).days} days ago")

        recommendation = auditor.check_data_retention(asset)

        print(f"\n  Recommendation: {recommendation['action_required']}")
        print(f"  Reasoning: {recommendation['reasoning']}")
        print(f"  Applicable Regulations: {', '.join(recommendation['applicable_regulations'])}")
        print(f"  Recommended Retention: {recommendation['recommended_retention_days']} days")
        if not recommendation['should_retain']:
            print(f"  Deletion Method: {recommendation['deletion_method']}")
        print(f"  Risk Assessment: {recommendation['risk_assessment']}")
        print()


def example_4_audit_trail_generation():
    """Example 4: Generate audit trails"""
    print("\n" + "=" * 80)
    print("Example 4: Audit Trail Generation")
    print("=" * 80)

    auditor = ComplianceAuditor()

    # Simulate various events
    events = [
        {
            "type": "ACCESS",
            "user": "admin@company.com",
            "resource": "Customer PII Database",
            "result": "SUCCESS",
            "details": {"ip_address": "10.0.1.50", "records_accessed": 15, "query": "SELECT * FROM customers WHERE country='DE'"}
        },
        {
            "type": "EXPORT",
            "user": "analyst@company.com",
            "resource": "PHI Records",
            "result": "SUCCESS",
            "details": {"ip_address": "10.0.2.100", "records_exported": 250, "format": "CSV"}
        },
        {
            "type": "DELETE",
            "user": "dpo@company.com",
            "resource": "Customer PII (SAR Request)",
            "result": "SUCCESS",
            "details": {"ip_address": "10.0.1.75", "reason": "GDPR Article 17 - Right to erasure", "customer_id": "CUST-12345"}
        },
        {
            "type": "ACCESS",
            "user": "contractor@external.com",
            "resource": "Production Database",
            "result": "DENIED",
            "details": {"ip_address": "203.0.113.50", "reason": "Insufficient privileges", "attempted_action": "SELECT"}
        }
    ]

    print("\nGenerating audit trails for recent events...\n")

    for event in events:
        trail = auditor.generate_audit_trail(
            event_type=event["type"],
            user=event["user"],
            resource=event["resource"],
            result=event["result"],
            details=event["details"]
        )

        compliance_marker = "🔒 COMPLIANCE" if trail.compliance_relevant else "📝 OPERATIONAL"
        result_marker = "✓" if trail.result == "SUCCESS" else "✗"

        print(f"{compliance_marker} | {result_marker} {trail.action}")
        print(f"  Event ID: {trail.event_id}")
        print(f"  User: {trail.user}")
        print(f"  Resource: {trail.resource}")
        print(f"  Result: {trail.result}")
        print(f"  IP: {trail.details.get('ip_address', 'unknown')}")
        print(f"  Timestamp: {trail.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Details: {', '.join(f'{k}={v}' for k, v in trail.details.items() if k != 'ip_address')}")
        print()

    # Summary
    compliance_count = len([t for t in auditor.audit_trails if t.compliance_relevant])
    print(f"\nAudit Trail Summary:")
    print(f"  Total Events: {len(auditor.audit_trails)}")
    print(f"  Compliance-Relevant: {compliance_count}")
    print(f"  Operational: {len(auditor.audit_trails) - compliance_count}")


def example_5_privacy_impact_assessment():
    """Example 5: Conduct Privacy Impact Assessment"""
    print("\n" + "=" * 80)
    print("Example 5: Privacy Impact Assessment (PIA)")
    print("=" * 80)

    # Sample PIA
    pia = PrivacyImpactAssessment(
        pia_id="PIA-2026-001",
        project_name="AI-Powered Customer Support Chatbot",
        data_processed=["Customer name", "Email address", "Support ticket history", "Chat transcripts"],
        processing_purpose="Provide automated customer support and escalate to human agents",
        legal_basis="Legitimate Interest (GDPR Art. 6.1.f) + Consent for chat history storage",
        data_subjects=["Customers", "Support agents"],
        third_parties=["Claude API (Anthropic)", "AWS S3 (chat storage)", "Slack (internal notifications)"],
        retention_period=365,  # 1 year
        security_measures=[
            "End-to-end encryption for chat transcripts",
            "Role-based access control",
            "Anonymization after 90 days of inactivity",
            "Regular security audits",
            "Data residency: EU-only storage"
        ],
        risks_identified=[
            "AI model may hallucinate incorrect support information",
            "Potential data leakage to third-party AI provider",
            "Insufficient consent mechanism for chat history",
            "Cross-border data transfer risks"
        ],
        mitigation_measures=[
            "Human review for all critical responses (refunds, account changes)",
            "Data Processing Agreement (DPA) with Anthropic",
            "Explicit opt-in checkbox for chat history storage",
            "Use EU region for Claude API and S3 storage",
            "Implement rate limiting and content filtering"
        ],
        dpo_approval=False
    )

    print(f"\nPrivacy Impact Assessment")
    print(f"{'='*80}")
    print(f"PIA ID: {pia.pia_id}")
    print(f"Project: {pia.project_name}")
    print(f"\nData Processing:")
    print(f"  Purpose: {pia.processing_purpose}")
    print(f"  Legal Basis: {pia.legal_basis}")
    print(f"  Data Types: {', '.join(pia.data_processed)}")
    print(f"  Data Subjects: {', '.join(pia.data_subjects)}")
    print(f"  Retention: {pia.retention_period} days")

    print(f"\nThird Parties:")
    for party in pia.third_parties:
        print(f"  - {party}")

    print(f"\nSecurity Measures:")
    for measure in pia.security_measures:
        print(f"  ✓ {measure}")

    print(f"\nRisks Identified ({len(pia.risks_identified)}):")
    for i, risk in enumerate(pia.risks_identified, 1):
        print(f"  {i}. ⚠ {risk}")

    print(f"\nMitigation Measures ({len(pia.mitigation_measures)}):")
    for i, mitigation in enumerate(pia.mitigation_measures, 1):
        print(f"  {i}. 🛡 {mitigation}")

    print(f"\nApproval Status:")
    if pia.dpo_approval:
        print(f"  ✓ Approved by DPO on {pia.approved_at.strftime('%Y-%m-%d')}")
    else:
        print(f"  ⏳ Pending DPO review and approval")

    # Assessment
    print(f"\nRisk Assessment:")
    risk_score = len(pia.risks_identified) * 10
    mitigation_score = len(pia.mitigation_measures) * 10
    net_risk = max(0, risk_score - mitigation_score)

    if net_risk < 30:
        risk_level = "LOW"
        action = "Proceed with standard monitoring"
    elif net_risk < 60:
        risk_level = "MEDIUM"
        action = "Implement additional safeguards before launch"
    else:
        risk_level = "HIGH"
        action = "REQUIRES DPO approval and additional risk mitigation"

    print(f"  Risk Score: {risk_score} | Mitigation Score: {mitigation_score}")
    print(f"  Net Risk: {net_risk} → {risk_level}")
    print(f"  Recommended Action: {action}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 58: AI Compliance Auditor")
    print("Automated compliance checking for GDPR, SOC2, HIPAA, ISO 27001")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_gdpr_compliance_check()
    example_2_ai_decision_explainability()
    example_3_data_retention_check()
    example_4_audit_trail_generation()
    example_5_privacy_impact_assessment()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
