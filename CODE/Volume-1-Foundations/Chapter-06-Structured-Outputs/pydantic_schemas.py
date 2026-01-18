#!/usr/bin/env python3
"""
Pydantic Schemas for Network Data Validation

Type-safe data models for network configurations, security findings,
and analysis results from AI systems.

From: AI for Networking Engineers - Volume 1, Chapter 6
Author: Eduard Dulharu

Usage:
    from pydantic_schemas import InterfaceConfig, DeviceConfig

    # Parse and validate
    interface = InterfaceConfig(**json_data)

    # Access with type safety
    print(interface.name)  # IDE autocomplete works!
    print(interface.ip_address)

    # Export back to dict/JSON
    data = interface.model_dump()
    json_str = interface.model_dump_json()
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Literal
from enum import Enum
from ipaddress import IPv4Address, IPv4Network
from datetime import datetime


# Enums for controlled vocabularies
class InterfaceStatus(str, Enum):
    """Interface operational status."""
    UP = "up"
    DOWN = "down"
    ADMIN_DOWN = "admin_down"
    TESTING = "testing"


class DeviceType(str, Enum):
    """Network device types."""
    ROUTER = "router"
    SWITCH = "switch"
    FIREWALL = "firewall"
    LOAD_BALANCER = "load_balancer"
    ACCESS_POINT = "access_point"


class SeverityLevel(str, Enum):
    """Security finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RoutingProtocol(str, Enum):
    """Routing protocols."""
    STATIC = "static"
    OSPF = "ospf"
    EIGRP = "eigrp"
    BGP = "bgp"
    IS_IS = "is-is"
    RIP = "rip"


# Interface Models
class InterfaceConfig(BaseModel):
    """
    Network interface configuration.

    Example:
        {
            "name": "GigabitEthernet0/1",
            "ip_address": "10.1.1.1",
            "subnet_mask": "255.255.255.0",
            "status": "up",
            "description": "Uplink to Core"
        }
    """
    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(..., description="Interface name")
    ip_address: Optional[str] = Field(None, description="IPv4 address")
    subnet_mask: Optional[str] = Field(None, description="Subnet mask")
    status: InterfaceStatus = Field(..., description="Operational status")
    description: Optional[str] = Field(None, description="Interface description")
    vlan: Optional[int] = Field(None, ge=1, le=4094, description="VLAN ID")
    speed: Optional[str] = Field(None, description="Interface speed (e.g., 1000)")
    duplex: Optional[Literal["full", "half", "auto"]] = Field(None, description="Duplex mode")

    @field_validator('ip_address')
    @classmethod
    def validate_ip(cls, v: Optional[str]) -> Optional[str]:
        """Validate IP address format."""
        if v is None:
            return v
        try:
            IPv4Address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")

    @field_validator('subnet_mask')
    @classmethod
    def validate_mask(cls, v: Optional[str]) -> Optional[str]:
        """Validate subnet mask."""
        if v is None:
            return v

        # Allow CIDR notation
        if v.startswith('/'):
            try:
                prefix = int(v[1:])
                if not (0 <= prefix <= 32):
                    raise ValueError(f"Invalid prefix length: {v}")
                return v
            except ValueError:
                raise ValueError(f"Invalid subnet mask: {v}")

        # Validate dotted decimal
        try:
            IPv4Address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid subnet mask: {v}")


class InterfaceStats(BaseModel):
    """Interface statistics and counters."""
    interface_name: str
    packets_in: int = Field(ge=0)
    packets_out: int = Field(ge=0)
    bytes_in: int = Field(ge=0)
    bytes_out: int = Field(ge=0)
    errors_in: int = Field(ge=0)
    errors_out: int = Field(ge=0)
    drops_in: int = Field(ge=0)
    drops_out: int = Field(ge=0)


# Device Models
class DeviceConfig(BaseModel):
    """
    Complete device configuration.

    Example:
        {
            "hostname": "CORE-RTR-01",
            "mgmt_ip": "192.168.1.1",
            "device_type": "router",
            "model": "Cisco ISR 4331",
            "interfaces": [...]
        }
    """
    hostname: str = Field(..., min_length=1, max_length=255)
    mgmt_ip: str = Field(..., description="Management IP address")
    device_type: DeviceType
    model: Optional[str] = Field(None, description="Device model")
    ios_version: Optional[str] = Field(None, description="IOS/software version")
    location: Optional[str] = Field(None, description="Physical location")
    interfaces: List[InterfaceConfig] = Field(default_factory=list)
    serial_number: Optional[str] = None
    uptime: Optional[str] = None

    @field_validator('mgmt_ip')
    @classmethod
    def validate_mgmt_ip(cls, v: str) -> str:
        """Validate management IP address."""
        try:
            IPv4Address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid management IP: {v}")

    def get_interface(self, name: str) -> Optional[InterfaceConfig]:
        """Get interface by name."""
        for interface in self.interfaces:
            if interface.name == name:
                return interface
        return None


# Security Models
class SecurityFinding(BaseModel):
    """
    Security vulnerability or issue found during analysis.

    Example:
        {
            "issue": "Telnet enabled on VTY lines",
            "severity": "critical",
            "line": "line vty 0 4",
            "fix": "transport input ssh",
            "impact": "Unencrypted remote access"
        }
    """
    issue: str = Field(..., description="Description of security issue")
    severity: SeverityLevel
    line: Optional[str] = Field(None, description="Config line with issue")
    fix: str = Field(..., description="How to fix the issue")
    impact: Optional[str] = Field(None, description="Potential impact")
    cve_id: Optional[str] = Field(None, description="CVE identifier if applicable")


class SecurityAnalysisResult(BaseModel):
    """Complete security analysis results."""
    device_hostname: str
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    critical_findings: List[SecurityFinding] = Field(default_factory=list)
    high_findings: List[SecurityFinding] = Field(default_factory=list)
    medium_findings: List[SecurityFinding] = Field(default_factory=list)
    low_findings: List[SecurityFinding] = Field(default_factory=list)
    summary: Optional[str] = None

    def total_findings(self) -> int:
        """Get total number of findings."""
        return (len(self.critical_findings) +
                len(self.high_findings) +
                len(self.medium_findings) +
                len(self.low_findings))

    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return len(self.critical_findings) > 0


# Routing Models
class Route(BaseModel):
    """Single routing table entry."""
    network: str = Field(..., description="Destination network")
    next_hop: str = Field(..., description="Next hop IP")
    interface: Optional[str] = Field(None, description="Exit interface")
    metric: int = Field(ge=0, description="Route metric")
    protocol: RoutingProtocol
    administrative_distance: int = Field(ge=0, le=255)

    @field_validator('network')
    @classmethod
    def validate_network(cls, v: str) -> str:
        """Validate network format."""
        try:
            IPv4Network(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid network: {v}")


class RoutingTable(BaseModel):
    """Complete routing table."""
    device_hostname: str
    routes: List[Route]
    default_route: Optional[Route] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    def get_routes_by_protocol(self, protocol: RoutingProtocol) -> List[Route]:
        """Filter routes by protocol."""
        return [r for r in self.routes if r.protocol == protocol]


# Log Analysis Models
class LogEntry(BaseModel):
    """Parsed syslog entry."""
    timestamp: datetime
    severity: SeverityLevel
    facility: str
    message: str
    hostname: Optional[str] = None
    process: Optional[str] = None
    raw_log: str


class LogAnalysisResult(BaseModel):
    """Analysis results for log entries."""
    total_logs: int = Field(ge=0)
    critical_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    info_count: int = Field(ge=0)
    top_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# Change Management Models
class ConfigChange(BaseModel):
    """Individual configuration change."""
    change_type: Literal["added", "removed", "modified"]
    line_number: Optional[int] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    impact: str
    risk_level: SeverityLevel


class ChangeImpactAnalysis(BaseModel):
    """Complete change impact analysis."""
    change_description: str
    overall_risk: SeverityLevel
    affected_services: List[str] = Field(default_factory=list)
    changes: List[ConfigChange] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    rollback_plan: str
    estimated_downtime_minutes: int = Field(ge=0)
    requires_maintenance_window: bool


# Example usage and testing
if __name__ == "__main__":
    import json

    print("""
    ========================================
    Pydantic Schemas Demo
    ========================================
    Testing type-safe data models
    ========================================
    """)

    # Test 1: Interface validation
    print("\nTest 1: Interface Configuration")
    print("-" * 60)

    interface_data = {
        "name": "GigabitEthernet0/1",
        "ip_address": "10.1.1.1",
        "subnet_mask": "255.255.255.0",
        "status": "up",
        "description": "Uplink to Core",
        "vlan": 10
    }

    try:
        interface = InterfaceConfig(**interface_data)
        print(f"✓ Created: {interface.name}")
        print(f"  IP: {interface.ip_address}/{interface.subnet_mask}")
        print(f"  Status: {interface.status}")
        print(f"  VLAN: {interface.vlan}")

        # Export back to dict
        print(f"\n  Export to dict: {interface.model_dump()}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Invalid IP address
    print("\n\nTest 2: Invalid IP Address (should fail)")
    print("-" * 60)

    invalid_interface = {
        "name": "Gi0/2",
        "ip_address": "999.999.999.999",  # Invalid!
        "status": "up"
    }

    try:
        interface = InterfaceConfig(**invalid_interface)
        print(f"✗ Should have failed!")
    except Exception as e:
        print(f"✓ Validation caught error: {e}")

    # Test 3: Device with interfaces
    print("\n\nTest 3: Complete Device Configuration")
    print("-" * 60)

    device_data = {
        "hostname": "CORE-RTR-01",
        "mgmt_ip": "192.168.1.1",
        "device_type": "router",
        "model": "Cisco ISR 4331",
        "ios_version": "16.9.5",
        "interfaces": [
            {
                "name": "Gi0/0/0",
                "ip_address": "10.1.1.1",
                "subnet_mask": "255.255.255.252",
                "status": "up"
            },
            {
                "name": "Gi0/0/1",
                "ip_address": "10.1.2.1",
                "subnet_mask": "255.255.255.252",
                "status": "up"
            }
        ]
    }

    try:
        device = DeviceConfig(**device_data)
        print(f"✓ Device: {device.hostname}")
        print(f"  Type: {device.device_type}")
        print(f"  Interfaces: {len(device.interfaces)}")

        # Access nested data with type safety
        for iface in device.interfaces:
            print(f"    - {iface.name}: {iface.ip_address} ({iface.status})")

        # Get specific interface
        gi0 = device.get_interface("Gi0/0/0")
        if gi0:
            print(f"\n  Found interface: {gi0.name} at {gi0.ip_address}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Security findings
    print("\n\nTest 4: Security Analysis Results")
    print("-" * 60)

    security_data = {
        "device_hostname": "EDGE-FW-01",
        "critical_findings": [
            {
                "issue": "Telnet enabled on VTY lines",
                "severity": "critical",
                "line": "line vty 0 4",
                "fix": "no transport input telnet\ntransport input ssh",
                "impact": "Unencrypted remote access allows credential theft"
            }
        ],
        "high_findings": [
            {
                "issue": "Weak SNMP community string",
                "severity": "high",
                "line": "snmp-server community public RO",
                "fix": "no snmp-server community public\nsnmp-server community <strong-string> RO",
                "impact": "Information disclosure"
            }
        ],
        "summary": "2 critical security issues found requiring immediate attention"
    }

    try:
        analysis = SecurityAnalysisResult(**security_data)
        print(f"✓ Analysis for: {analysis.device_hostname}")
        print(f"  Total findings: {analysis.total_findings()}")
        print(f"  Critical issues: {analysis.has_critical_issues()}")

        if analysis.critical_findings:
            print(f"\n  Critical Issues:")
            for finding in analysis.critical_findings:
                print(f"    - {finding.issue}")
                print(f"      Fix: {finding.fix}")
                print(f"      Impact: {finding.impact}")

        # Export to JSON
        json_output = analysis.model_dump_json(indent=2)
        print(f"\n  JSON export (truncated):\n{json_output[:200]}...")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 5: Change impact analysis
    print("\n\nTest 5: Change Impact Analysis")
    print("-" * 60)

    change_data = {
        "change_description": "Add ACL to block guest VLAN from management subnet",
        "overall_risk": "medium",
        "affected_services": ["Guest WiFi", "Management Access"],
        "changes": [
            {
                "change_type": "added",
                "new_value": "ip access-list extended GUEST_BLOCK",
                "impact": "Creates new ACL",
                "risk_level": "low"
            }
        ],
        "prerequisites": ["Verify guest VLAN subnet", "Confirm management subnet"],
        "rollback_plan": "no ip access-list extended GUEST_BLOCK",
        "estimated_downtime_minutes": 0,
        "requires_maintenance_window": False
    }

    try:
        change = ChangeImpactAnalysis(**change_data)
        print(f"✓ Change: {change.change_description}")
        print(f"  Overall risk: {change.overall_risk}")
        print(f"  Downtime: {change.estimated_downtime_minutes} minutes")
        print(f"  Maintenance window: {change.requires_maintenance_window}")
        print(f"  Affected services: {', '.join(change.affected_services)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n✅ Demo complete!")
    print("\n💡 Benefits of Pydantic:")
    print("  - Type safety (IDE autocomplete)")
    print("  - Automatic validation")
    print("  - Clear error messages")
    print("  - Easy serialization (dict/JSON)")
    print("  - Self-documenting schemas")
