"""
Chapter 31: Document Extraction & Parsing
Extract structured data from unstructured network documentation
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


# ============================================================================
# Data Models
# ============================================================================

class DocumentType(str, Enum):
    """Document type"""
    NETWORK_DIAGRAM = "NETWORK_DIAGRAM"
    CONFIG_FILE = "CONFIG_FILE"
    RUNBOOK = "RUNBOOK"
    RFC = "RFC"
    VENDOR_DOC = "VENDOR_DOC"
    INCIDENT_REPORT = "INCIDENT_REPORT"
    DESIGN_DOC = "DESIGN_DOC"


@dataclass
class ExtractedDevice:
    """Extracted network device"""
    name: str
    device_type: str  # router, switch, firewall
    model: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)


@dataclass
class ExtractedConnection:
    """Extracted connection between devices"""
    source_device: str
    source_interface: Optional[str]
    target_device: str
    target_interface: Optional[str]
    connection_type: str  # physical, logical, vpn
    protocol: Optional[str] = None


@dataclass
class ExtractedConfig:
    """Extracted configuration snippet"""
    config_type: str  # interface, routing, acl, etc.
    content: str
    device: Optional[str] = None
    validation_status: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Document metadata"""
    title: str
    document_type: DocumentType
    version: Optional[str] = None
    last_updated: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    """Complete extracted document"""
    metadata: DocumentMetadata
    devices: List[ExtractedDevice]
    connections: List[ExtractedConnection]
    configs: List[ExtractedConfig]
    key_findings: List[str]
    summary: str


# ============================================================================
# Pydantic Models for LLM Outputs
# ============================================================================

class DeviceExtraction(BaseModel):
    """LLM-extracted device"""
    name: str = Field(description="Device name or hostname")
    device_type: str = Field(description="router, switch, firewall, load_balancer, server")
    model: Optional[str] = Field(description="Device model (e.g., Cisco ASR 9000)")
    ip_address: Optional[str] = Field(description="Management IP address")
    location: Optional[str] = Field(description="Physical location")
    interfaces: List[str] = Field(description="Interface names (e.g., Eth1/1, GigE0/0)")
    protocols: List[str] = Field(description="Protocols used (BGP, OSPF, VLAN, etc.)")


class ConnectionExtraction(BaseModel):
    """LLM-extracted connection"""
    source_device: str = Field(description="Source device name")
    source_interface: Optional[str] = Field(description="Source interface")
    target_device: str = Field(description="Target device name")
    target_interface: Optional[str] = Field(description="Target interface")
    connection_type: str = Field(description="physical, logical, or vpn")
    protocol: Optional[str] = Field(description="Protocol used on connection")


class FullExtraction(BaseModel):
    """Complete document extraction"""
    devices: List[DeviceExtraction] = Field(description="All devices found (2-8 devices)")
    connections: List[ConnectionExtraction] = Field(description="All connections (1-6 connections)")
    key_findings: List[str] = Field(description="Important findings or issues (2-4 items)")
    summary: str = Field(description="One-paragraph summary of the document")


class ConfigValidation(BaseModel):
    """Configuration validation"""
    is_valid: bool = Field(description="True if configuration is valid")
    issues_found: List[str] = Field(description="Issues or errors found (0-5 items)")
    recommendations: List[str] = Field(description="Recommendations for improvement (2-4 items)")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, LOW, or OK")


class IncidentExtraction(BaseModel):
    """Incident report extraction"""
    incident_id: str = Field(description="Incident identifier")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    affected_devices: List[str] = Field(description="Devices affected (1-5 items)")
    root_cause: str = Field(description="Root cause description (one sentence)")
    timeline: List[str] = Field(description="Key timeline events (3-5 items)")
    resolution: str = Field(description="Resolution steps (one paragraph)")
    lessons_learned: List[str] = Field(description="Lessons learned (2-3 items)")


# ============================================================================
# Document Extractor
# ============================================================================

class DocumentExtractor:
    """Extract structured data from network documentation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=self.api_key,
            temperature=0
        )

    def extract_from_text(self, text: str, doc_type: DocumentType) -> ExtractedDocument:
        """
        Extract structured information from text

        Args:
            text: Document text
            doc_type: Type of document

        Returns:
            Extracted document with devices, connections, configs
        """
        parser = PydanticOutputParser(pydantic_object=FullExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network documentation parser.
Extract all devices, connections, and key information from the document.
Focus on technical details: device names, IPs, interfaces, protocols.

{format_instructions}"""),
            ("human", """Document Type: {doc_type}

Document Content:
{text}

Extract all devices, connections, and key findings.""")
        ])

        chain = prompt | self.llm | parser

        extraction = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "doc_type": doc_type.value,
            "text": text[:4000]  # Limit to 4000 chars
        })

        # Convert to internal format
        devices = [
            ExtractedDevice(
                name=d.name,
                device_type=d.device_type,
                model=d.model,
                ip_address=d.ip_address,
                location=d.location,
                interfaces=d.interfaces,
                protocols=d.protocols
            )
            for d in extraction.devices
        ]

        connections = [
            ExtractedConnection(
                source_device=c.source_device,
                source_interface=c.source_interface,
                target_device=c.target_device,
                target_interface=c.target_interface,
                connection_type=c.connection_type,
                protocol=c.protocol
            )
            for c in extraction.connections
        ]

        metadata = DocumentMetadata(
            title="Extracted Document",
            document_type=doc_type,
            tags=[]
        )

        return ExtractedDocument(
            metadata=metadata,
            devices=devices,
            connections=connections,
            configs=[],
            key_findings=extraction.key_findings,
            summary=extraction.summary
        )

    def extract_config_blocks(self, text: str) -> List[ExtractedConfig]:
        """
        Extract configuration blocks from text

        Args:
            text: Text containing configs

        Returns:
            List of extracted config blocks
        """
        configs = []

        # Pattern matching for common config blocks
        patterns = {
            "interface": r"interface\s+[\w/]+\s*\n(?:.*\n)*?!",
            "router bgp": r"router bgp\s+\d+\s*\n(?:.*\n)*?!",
            "router ospf": r"router ospf\s+\d+\s*\n(?:.*\n)*?!",
            "access-list": r"access-list\s+\d+\s+(?:permit|deny).*",
            "ip route": r"ip route\s+[\d./]+\s+[\d./]+.*"
        }

        for config_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                configs.append(ExtractedConfig(
                    config_type=config_type,
                    content=match.strip()
                ))

        return configs

    def validate_config(self, config: str, vendor: str = "cisco") -> Dict:
        """
        Validate network configuration

        Args:
            config: Configuration text
            vendor: Vendor (cisco, juniper, arista)

        Returns:
            Validation result
        """
        parser = PydanticOutputParser(pydantic_object=ConfigValidation)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network configuration validator.
Check for syntax errors, security issues, and best practices.
Focus on: permit any any, weak crypto, missing descriptions, hardcoded IPs.

{format_instructions}"""),
            ("human", """Vendor: {vendor}

Configuration:
{config}

Validate this configuration and report issues.""")
        ])

        chain = prompt | self.llm | parser

        validation = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "vendor": vendor,
            "config": config[:2000]
        })

        return {
            "is_valid": validation.is_valid,
            "issues_found": validation.issues_found,
            "recommendations": validation.recommendations,
            "severity": validation.severity
        }

    def extract_incident_report(self, text: str) -> Dict:
        """
        Extract structured data from incident report

        Args:
            text: Incident report text

        Returns:
            Structured incident data
        """
        parser = PydanticOutputParser(pydantic_object=IncidentExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an incident report analyzer.
Extract key information: incident ID, severity, affected devices, root cause, timeline, resolution.

{format_instructions}"""),
            ("human", """Incident Report:
{text}

Extract structured incident information.""")
        ])

        chain = prompt | self.llm | parser

        incident = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "text": text[:3000]
        })

        return {
            "incident_id": incident.incident_id,
            "severity": incident.severity,
            "affected_devices": incident.affected_devices,
            "root_cause": incident.root_cause,
            "timeline": incident.timeline,
            "resolution": incident.resolution,
            "lessons_learned": incident.lessons_learned
        }

    def extract_ip_addresses(self, text: str) -> List[Dict[str, str]]:
        """
        Extract all IP addresses from text

        Args:
            text: Text to search

        Returns:
            List of IP addresses with context
        """
        # Regex for IPv4
        ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'

        matches = []
        for match in re.finditer(ipv4_pattern, text):
            ip = match.group()
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()

            matches.append({
                "ip_address": ip,
                "context": context
            })

        return matches

    def extract_device_names(self, text: str) -> List[str]:
        """
        Extract device hostnames from text

        Args:
            text: Text to search

        Returns:
            List of unique device names
        """
        # Common patterns for device names
        patterns = [
            r'\b(?:router|switch|firewall|fw|core|dist|access)-[\w-]+\b',
            r'\b[\w-]+(?:-r\d+|-sw\d+|-fw\d+)\b',
            r'hostname\s+([\w-]+)',
        ]

        devices = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            devices.update([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])

        return sorted(list(devices))


# ============================================================================
# Examples
# ============================================================================

def example_1_extract_from_design_doc():
    """Example 1: Extract from network design document"""
    print("=" * 80)
    print("Example 1: Extract from Network Design Document")
    print("=" * 80)

    extractor = DocumentExtractor()

    document = """
    Network Design - Data Center A

    Core Layer:
    - core-router-01 (Cisco ASR 9010, 10.0.0.1) located in DC-A-Row1
      Interfaces: GigE0/0/0 (to dist-sw-01), GigE0/0/1 (to dist-sw-02)
      Protocols: BGP AS 65001, OSPF Area 0

    - core-router-02 (Cisco ASR 9010, 10.0.0.2) located in DC-A-Row1
      Interfaces: GigE0/0/0 (to dist-sw-01), GigE0/0/1 (to dist-sw-02)
      Protocols: BGP AS 65001, OSPF Area 0

    Distribution Layer:
    - dist-sw-01 (Arista 7280, 10.0.1.1) located in DC-A-Row2
      Uplinks: Eth1 to core-router-01, Eth2 to core-router-02
      Downlinks: Eth10-20 to access switches
      VLANs: 100, 200, 300

    Connections:
    - core-router-01 GigE0/0/0 <---> dist-sw-01 Eth1 (10G fiber)
    - core-router-02 GigE0/0/1 <---> dist-sw-01 Eth2 (10G fiber)
    """

    print("\nExtracting structured data from design document...")
    print(f"Document size: {len(document)} characters\n")

    extracted = extractor.extract_from_text(document, DocumentType.DESIGN_DOC)

    print(f"Extracted {len(extracted.devices)} devices:")
    for device in extracted.devices:
        print(f"\n  Device: {device.name}")
        print(f"    Type: {device.device_type}")
        if device.model:
            print(f"    Model: {device.model}")
        if device.ip_address:
            print(f"    IP: {device.ip_address}")
        if device.interfaces:
            print(f"    Interfaces: {', '.join(device.interfaces[:3])}")
        if device.protocols:
            print(f"    Protocols: {', '.join(device.protocols)}")

    print(f"\nExtracted {len(extracted.connections)} connections:")
    for conn in extracted.connections:
        print(f"  {conn.source_device} ({conn.source_interface or 'N/A'}) --> {conn.target_device} ({conn.target_interface or 'N/A'})")

    print(f"\nKey Findings:")
    for finding in extracted.key_findings:
        print(f"  - {finding}")

    print(f"\nSummary:")
    print(f"  {extracted.summary}")


def example_2_extract_config_blocks():
    """Example 2: Extract configuration blocks"""
    print("\n" + "=" * 80)
    print("Example 2: Extract Configuration Blocks")
    print("=" * 80)

    extractor = DocumentExtractor()

    config_text = """
    hostname core-router-01
    !
    interface GigabitEthernet0/0/0
     description Link to dist-sw-01
     ip address 10.1.1.1 255.255.255.252
     no shutdown
    !
    interface GigabitEthernet0/0/1
     description Link to dist-sw-02
     ip address 10.1.1.5 255.255.255.252
     no shutdown
    !
    router bgp 65001
     bgp log-neighbor-changes
     neighbor 10.0.0.2 remote-as 65001
     neighbor 10.0.0.2 update-source Loopback0
    !
    router ospf 1
     router-id 10.0.0.1
     network 10.0.0.0 0.0.0.255 area 0
     network 10.1.1.0 0.0.0.255 area 0
    !
    ip route 0.0.0.0 0.0.0.0 192.168.1.1
    !
    """

    print("\nExtracting configuration blocks...")
    configs = extractor.extract_config_blocks(config_text)

    print(f"\nFound {len(configs)} configuration blocks:\n")

    for i, config in enumerate(configs, 1):
        print(f"{i}. Type: {config.config_type}")
        print(f"   Content:")
        print(f"   {config.content[:100]}...")
        print()


def example_3_validate_configuration():
    """Example 3: Validate network configuration"""
    print("\n" + "=" * 80)
    print("Example 3: Validate Configuration")
    print("=" * 80)

    extractor = DocumentExtractor()

    # Sample config with issues
    config = """
    access-list 100 permit ip any any
    !
    enable password cisco123
    !
    interface GigabitEthernet0/0
     ip address 10.1.1.1 255.255.255.0
     no shutdown
    !
    crypto isakmp policy 10
     encryption des
     hash md5
    !
    """

    print("\nValidating configuration...")
    print(f"Config size: {len(config)} bytes\n")

    validation = extractor.validate_config(config, vendor="cisco")

    print(f"{'='*80}")
    print(f"VALIDATION RESULT")
    print(f"{'='*80}")
    print(f"Valid: {'✓' if validation['is_valid'] else '✗'}")
    print(f"Severity: {validation['severity']}")

    if validation['issues_found']:
        print(f"\nIssues Found ({len(validation['issues_found'])}):")
        for i, issue in enumerate(validation['issues_found'], 1):
            print(f"  {i}. {issue}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(validation['recommendations'], 1):
        print(f"  {i}. {rec}")


def example_4_extract_incident_report():
    """Example 4: Extract from incident report"""
    print("\n" + "=" * 80)
    print("Example 4: Extract Incident Report")
    print("=" * 80)

    extractor = DocumentExtractor()

    incident_report = """
    INCIDENT REPORT #INC-2026-001

    Severity: CRITICAL
    Reported: 2026-01-18 02:30 UTC

    Summary:
    Complete network outage in Data Center A affecting all production services.

    Affected Systems:
    - core-router-01 (primary core router)
    - dist-sw-01, dist-sw-02 (distribution switches)
    - All downstream access switches and servers

    Timeline:
    - 02:30 UTC: Monitoring alerts for core-router-01 unreachable
    - 02:32 UTC: NOC engineer confirmed BGP sessions down
    - 02:35 UTC: Found memory leak in BGP process (100% memory utilization)
    - 02:40 UTC: Reloaded BGP process on core-router-01
    - 02:45 UTC: BGP sessions restored, traffic recovering
    - 03:00 UTC: All services fully operational

    Root Cause:
    Software bug in IOS-XR 7.3.1 causing memory leak in BGP process when processing
    large routing tables with > 500K routes. Memory exhaustion led to process crash.

    Resolution:
    1. Emergency reload of BGP process
    2. Applied temporary route filtering to reduce table size
    3. Scheduled upgrade to IOS-XR 7.3.2 (patched version)

    Lessons Learned:
    - Implement stricter memory monitoring with lower thresholds
    - Always test new IOS versions in lab before production deployment
    - Need redundant core routers with better failover mechanisms
    """

    print("\nExtracting incident report data...")
    print(f"Report size: {len(incident_report)} characters\n")

    incident = extractor.extract_incident_report(incident_report)

    print(f"{'='*80}")
    print(f"INCIDENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Incident ID: {incident['incident_id']}")
    print(f"Severity: {incident['severity']}")

    print(f"\nAffected Devices ({len(incident['affected_devices'])}):")
    for device in incident['affected_devices']:
        print(f"  - {device}")

    print(f"\nRoot Cause:")
    print(f"  {incident['root_cause']}")

    print(f"\nTimeline:")
    for event in incident['timeline']:
        print(f"  - {event}")

    print(f"\nResolution:")
    print(f"  {incident['resolution']}")

    print(f"\nLessons Learned:")
    for lesson in incident['lessons_learned']:
        print(f"  - {lesson}")


def example_5_extract_ip_and_devices():
    """Example 5: Extract IPs and device names"""
    print("\n" + "=" * 80)
    print("Example 5: Extract IPs and Device Names")
    print("=" * 80)

    extractor = DocumentExtractor()

    text = """
    Network documentation excerpt:

    Core routers (core-r1, core-r2) are configured with management IPs:
    - core-r1: 10.0.0.1
    - core-r2: 10.0.0.2

    Distribution switches:
    - dist-sw-01 at 10.0.1.1
    - dist-sw-02 at 10.0.1.2

    Access layer devices (access-sw-01 through access-sw-10) use
    the 10.0.2.0/24 range.

    Firewall fw-01 (192.168.100.1) protects the perimeter.
    """

    print("\nExtracting IP addresses...")
    ips = extractor.extract_ip_addresses(text)

    print(f"\nFound {len(ips)} IP addresses:")
    for ip_info in ips[:8]:  # Show first 8
        print(f"\n  IP: {ip_info['ip_address']}")
        print(f"  Context: {ip_info['context'][:80]}...")

    print("\n" + "-" * 80)
    print("\nExtracting device names...")
    devices = extractor.extract_device_names(text)

    print(f"\nFound {len(devices)} device names:")
    for device in devices:
        print(f"  - {device}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 31: Document Extraction & Parsing")
    print("Extract structured data from network documentation")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_extract_from_design_doc()
    example_2_extract_config_blocks()
    example_3_validate_configuration()
    example_4_extract_incident_report()
    example_5_extract_ip_and_devices()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
