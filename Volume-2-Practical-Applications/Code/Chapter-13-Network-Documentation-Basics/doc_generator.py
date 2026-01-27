#!/usr/bin/env python3
"""
Configuration Documentation Generator

Auto-generate comprehensive documentation from network device configurations.

From: AI for Networking Engineers - Volume 2, Chapter 13
Author: Eduard Dulharu (Ed Harmoosh)

This module provides:
- Device overview extraction (role, IPs, protocols, features)
- Interface table generation (markdown format)
- Routing documentation (protocols, neighbors, policies)
- Security documentation (ACLs, AAA, management access)
- VLAN documentation with descriptions
- Hardware inventory extraction
- Complete device documentation assembly
- Multi-vendor support (Cisco, Juniper, Arista)
- Template-based output (markdown, HTML, JSON)

Usage:
    from doc_generator import ConfigDocumentationGenerator, OutputFormat

    generator = ConfigDocumentationGenerator()
    doc = generator.generate_complete_documentation(
        config=config_text,
        hostname="router-core-01",
        output_file="router-core-01.md"
    )

    # Or generate specific sections
    overview = generator.generate_device_overview(config, "router-01")
    interfaces = generator.generate_interface_documentation(config)
    vlans = generator.generate_vlan_documentation(config)
"""

from anthropic import Anthropic
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
import os
import re


class OutputFormat(str, Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class DeviceType(str, Enum):
    """Supported device types."""
    CISCO_IOS = "cisco_ios"
    CISCO_NXOS = "cisco_nxos"
    CISCO_ASA = "cisco_asa"
    JUNIPER_JUNOS = "juniper_junos"
    ARISTA_EOS = "arista_eos"
    PALO_ALTO = "palo_alto"
    FORTINET = "fortinet"
    UNKNOWN = "unknown"


class DeviceRole(str, Enum):
    """Device roles in network architecture."""
    CORE_ROUTER = "Core Router"
    DISTRIBUTION_ROUTER = "Distribution Router"
    EDGE_ROUTER = "Edge Router"
    ACCESS_SWITCH = "Access Switch"
    DISTRIBUTION_SWITCH = "Distribution Switch"
    CORE_SWITCH = "Core Switch"
    FIREWALL = "Firewall"
    LOAD_BALANCER = "Load Balancer"
    WIRELESS_CONTROLLER = "Wireless Controller"
    UNKNOWN = "Unknown"


@dataclass
class InterfaceInfo:
    """Interface information."""
    name: str
    ip_address: Optional[str] = None
    subnet_mask: Optional[str] = None
    description: Optional[str] = None
    status: str = "Unknown"
    vlan: Optional[int] = None
    vrf: Optional[str] = None
    speed: Optional[str] = None
    duplex: Optional[str] = None
    mtu: Optional[int] = None


@dataclass
class VlanInfo:
    """VLAN information."""
    vlan_id: int
    name: Optional[str] = None
    description: Optional[str] = None
    ip_address: Optional[str] = None
    state: str = "active"
    interfaces: List[str] = field(default_factory=list)


@dataclass
class RoutingProtocol:
    """Routing protocol information."""
    protocol: str
    process_id: Optional[str] = None
    router_id: Optional[str] = None
    as_number: Optional[int] = None
    networks: List[str] = field(default_factory=list)
    neighbors: List[Dict[str, str]] = field(default_factory=list)
    redistributed_from: List[str] = field(default_factory=list)


@dataclass
class DeviceOverview:
    """Complete device overview."""
    hostname: str
    role: DeviceRole
    device_type: DeviceType
    management_ip: Optional[str] = None
    software_version: Optional[str] = None
    serial_number: Optional[str] = None
    uptime: Optional[str] = None
    routing_protocols: List[str] = field(default_factory=list)
    key_features: List[str] = field(default_factory=list)
    interface_count: int = 0
    vlan_count: int = 0
    notable_config: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConfigDocumentationGenerator:
    """
    Generate documentation automatically from network configs.

    Uses Claude to analyze device configurations and produce
    structured, consistent documentation in multiple formats.

    Attributes:
        client: Anthropic API client instance
        model: Claude model to use (default: claude-3-5-sonnet)
        temperature: Model temperature (0 for deterministic output)

    Example:
        >>> generator = ConfigDocumentationGenerator()
        >>> overview = generator.generate_device_overview(config, "router-01")
        >>> print(overview.role)
        DeviceRole.CORE_ROUTER
    """

    # Model pricing (per 1M tokens) - January 2026
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    }

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0
    ):
        """
        Initialize the documentation generator.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use.
            temperature: Temperature for generation (0 = deterministic).
        """
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model
        self.temperature = temperature
        self._usage_stats = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def _call_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Make API call and track usage.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens in response.

        Returns:
            Response text from Claude.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track usage
        self._usage_stats["input_tokens"] += response.usage.input_tokens
        self._usage_stats["output_tokens"] += response.usage.output_tokens
        self._usage_stats["requests"] += 1

        return response.content[0].text

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Dictionary with input/output tokens, requests, and estimated cost.
        """
        pricing = self.PRICING.get(self.model, {"input": 3.00, "output": 15.00})
        cost = (
            (self._usage_stats["input_tokens"] / 1_000_000) * pricing["input"] +
            (self._usage_stats["output_tokens"] / 1_000_000) * pricing["output"]
        )
        return {
            **self._usage_stats,
            "estimated_cost_usd": round(cost, 6),
            "model": self.model
        }

    def detect_device_type(self, config: str) -> DeviceType:
        """
        Detect device type from configuration.

        Args:
            config: Device configuration text.

        Returns:
            Detected DeviceType enum value.
        """
        config_lower = config.lower()

        # Cisco IOS patterns
        if 'ios' in config_lower or re.search(r'version \d+\.\d+', config):
            if 'feature' in config_lower and 'vdc' in config_lower:
                return DeviceType.CISCO_NXOS
            return DeviceType.CISCO_IOS

        # Cisco NX-OS patterns
        if 'feature' in config_lower and 'vpc' in config_lower:
            return DeviceType.CISCO_NXOS

        # Cisco ASA patterns
        if 'access-group' in config_lower and 'security-level' in config_lower:
            return DeviceType.CISCO_ASA

        # Juniper patterns
        if 'set system' in config_lower or 'routing-options' in config_lower:
            return DeviceType.JUNIPER_JUNOS

        # Arista patterns
        if 'arista' in config_lower or 'management api http' in config_lower:
            return DeviceType.ARISTA_EOS

        # Palo Alto patterns
        if 'set deviceconfig' in config_lower or 'set network' in config_lower:
            return DeviceType.PALO_ALTO

        # Fortinet patterns
        if 'config system global' in config_lower:
            return DeviceType.FORTINET

        return DeviceType.UNKNOWN

    def extract_hostname(self, config: str) -> str:
        """
        Extract hostname from configuration.

        Args:
            config: Device configuration text.

        Returns:
            Extracted hostname or "unknown".
        """
        # Cisco IOS/NX-OS
        match = re.search(r'^hostname\s+(\S+)', config, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)

        # Juniper
        match = re.search(r'set system host-name\s+(\S+)', config, re.IGNORECASE)
        if match:
            return match.group(1)

        # Arista
        match = re.search(r'^hostname\s+(\S+)', config, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)

        return "unknown"

    def extract_interfaces_basic(self, config: str) -> List[InterfaceInfo]:
        """
        Extract basic interface information using regex (no API call).

        Useful for quick parsing without API cost.

        Args:
            config: Device configuration text.

        Returns:
            List of InterfaceInfo objects.
        """
        interfaces = []

        # Cisco IOS interface pattern
        pattern = r'^interface\s+(\S+).*?(?=^interface|\Z)'
        matches = re.finditer(pattern, config, re.MULTILINE | re.DOTALL)

        for match in matches:
            interface_block = match.group(0)
            interface_name = match.group(1)

            # Extract IP
            ip_match = re.search(r'ip address\s+(\d+\.\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+\.\d+)', interface_block)
            ip = ip_match.group(1) if ip_match else None
            mask = ip_match.group(2) if ip_match else None

            # Extract description
            desc_match = re.search(r'description\s+(.+)$', interface_block, re.MULTILINE)
            description = desc_match.group(1).strip() if desc_match else None

            # Check shutdown status
            status = "Down" if 'shutdown' in interface_block.lower() and 'no shutdown' not in interface_block.lower() else "Up"

            # Extract VLAN
            vlan_match = re.search(r'switchport access vlan\s+(\d+)', interface_block)
            vlan = int(vlan_match.group(1)) if vlan_match else None

            interfaces.append(InterfaceInfo(
                name=interface_name,
                ip_address=ip,
                subnet_mask=mask,
                description=description,
                status=status,
                vlan=vlan
            ))

        return interfaces

    def extract_vlans_basic(self, config: str) -> List[VlanInfo]:
        """
        Extract VLAN information using regex (no API call).

        Args:
            config: Device configuration text.

        Returns:
            List of VlanInfo objects.
        """
        vlans = []

        # VLAN database pattern
        pattern = r'^vlan\s+(\d+)\s*\n(?:\s+name\s+(\S+))?'
        matches = re.finditer(pattern, config, re.MULTILINE)

        for match in matches:
            vlan_id = int(match.group(1))
            name = match.group(2) if match.group(2) else f"VLAN{vlan_id:04d}"

            vlans.append(VlanInfo(
                vlan_id=vlan_id,
                name=name
            ))

        return vlans

    def generate_device_overview(self, config: str, hostname: str = None) -> DeviceOverview:
        """
        Generate high-level device documentation.

        Uses AI to analyze the configuration and extract key information
        about the device's role, features, and configuration.

        Args:
            config: Full device configuration text.
            hostname: Device hostname (auto-detected if not provided).

        Returns:
            DeviceOverview dataclass with all extracted information.

        Example:
            >>> overview = generator.generate_device_overview(config)
            >>> print(f"Role: {overview.role.value}")
            >>> print(f"Protocols: {', '.join(overview.routing_protocols)}")
        """
        if not hostname:
            hostname = self.extract_hostname(config)

        device_type = self.detect_device_type(config)

        prompt = f"""Analyze this {device_type.value} network device configuration and extract documentation.

Hostname: {hostname}
Configuration:
```
{config}
```

Extract and return as JSON:
{{
    "role": "One of: Core Router, Distribution Router, Edge Router, Access Switch, Distribution Switch, Core Switch, Firewall, Load Balancer, Wireless Controller, Unknown",
    "management_ip": "Primary management IP address or null",
    "software_version": "Software version if visible or null",
    "routing_protocols": ["List of routing protocols: BGP, OSPF, EIGRP, ISIS, RIP, Static, etc."],
    "key_features": ["List of enabled features: HSRP, VRRP, VRF, QoS, NAT, VPN, MPLS, etc."],
    "interface_count": number_of_interfaces,
    "vlan_count": number_of_vlans,
    "notable_config": "One sentence about anything important or unusual"
}}

Return ONLY valid JSON, no markdown or explanation."""

        response = self._call_api(prompt, max_tokens=1000)

        # Parse JSON response
        try:
            # Clean response (remove markdown if present)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```\w*\n', '', clean_response)
                clean_response = re.sub(r'\n```$', '', clean_response)

            data = json.loads(clean_response)
        except json.JSONDecodeError:
            # Fallback to basic extraction
            data = {
                "role": "Unknown",
                "management_ip": None,
                "routing_protocols": [],
                "key_features": [],
                "interface_count": len(self.extract_interfaces_basic(config)),
                "vlan_count": len(self.extract_vlans_basic(config)),
                "notable_config": None
            }

        # Map role string to enum
        role_map = {v.value.lower(): v for v in DeviceRole}
        role_str = data.get("role", "Unknown").lower()
        role = role_map.get(role_str, DeviceRole.UNKNOWN)

        return DeviceOverview(
            hostname=hostname,
            role=role,
            device_type=device_type,
            management_ip=data.get("management_ip"),
            software_version=data.get("software_version"),
            routing_protocols=data.get("routing_protocols", []),
            key_features=data.get("key_features", []),
            interface_count=data.get("interface_count", 0),
            vlan_count=data.get("vlan_count", 0),
            notable_config=data.get("notable_config")
        )

    def generate_interface_documentation(
        self,
        config: str,
        format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str:
        """
        Generate interface documentation.

        Creates a detailed table of all interfaces with their configuration.

        Args:
            config: Full device configuration text.
            format: Output format (MARKDOWN, HTML, JSON).

        Returns:
            Formatted interface documentation.
        """
        prompt = f"""Extract all interfaces from this config and create documentation.

Config:
```
{config}
```

Create a markdown table with columns:
| Interface | IP Address | Subnet | Status | Description | VLAN/VRF | Speed | Notes |

Include ALL interfaces:
- Physical (GigabitEthernet, FastEthernet, Ethernet, etc.)
- Loopback
- Tunnel
- VLAN (SVI)
- Port-channel/LAG
- Management

For Status: "Up" if no shutdown or status not specified, "Down" if shutdown
For missing values: "N/A"
For Notes: Any special config (HSRP, trunk, access, etc.)

Return ONLY the markdown table."""

        response = self._call_api(prompt, max_tokens=3000)

        if format == OutputFormat.JSON:
            interfaces = self.extract_interfaces_basic(config)
            return json.dumps([asdict(i) for i in interfaces], indent=2)

        return response

    def generate_routing_documentation(self, config: str) -> str:
        """
        Generate routing configuration documentation.

        Creates comprehensive routing documentation including all protocols,
        neighbors, networks, and policies.

        Args:
            config: Full device configuration text.

        Returns:
            Markdown-formatted routing documentation.
        """
        prompt = f"""Document the routing configuration comprehensively.

Config:
```
{config}
```

Create documentation with these sections:

## Routing Summary
Brief overview of routing design

## Routing Protocols

### BGP (if configured)
- AS Number
- Router ID
- Neighbors table: | Neighbor IP | Remote AS | Description | State |
- Address families
- Route policies applied

### OSPF (if configured)
- Process ID
- Router ID
- Areas: | Area | Type | Networks |
- Neighbors (if visible)
- Special configurations (stub, NSSA, etc.)

### EIGRP (if configured)
- AS Number
- Router ID
- Networks
- Authentication

### Static Routes
Table: | Destination | Next-Hop | AD | Tag | Description |

## Route Redistribution
What routes are redistributed between protocols

## Route Policies
- Route-maps with descriptions
- Prefix-lists
- AS-path access-lists
- Community lists

If a section is not configured, write "Not configured".
Use proper markdown formatting with headers and tables."""

        return self._call_api(prompt, max_tokens=3000)

    def generate_security_documentation(self, config: str) -> str:
        """
        Generate security configuration documentation.

        Documents all security-related configuration including access controls,
        authentication, encryption, and security features.

        Args:
            config: Full device configuration text.

        Returns:
            Markdown-formatted security documentation.
        """
        prompt = f"""Document the security configuration comprehensively.

Config:
```
{config}
```

Create documentation with these sections:

## Security Summary
Overall security posture assessment (Good/Moderate/Needs Improvement)

## Access Control Lists
For each ACL:
- Name/Number
- Purpose (inferred from name, application, or rules)
- Where applied (interface, VTY, etc.)
- Key rules summary

Table: | ACL Name | Type | Applied To | Direction | Purpose |

## Authentication & Authorization
- AAA configuration
- TACACS+/RADIUS servers (IP, purpose)
- Local user accounts (no passwords!)
- Login authentication methods
- Authorization settings
- Accounting configuration

## Management Security
- SSH configuration (version, timeout, retries)
- Telnet status (enabled/disabled)
- HTTP/HTTPS configuration
- Console/AUX security
- VTY line security (access-class, transport)

## Layer 2 Security (if switch)
- Port security
- DHCP snooping
- Dynamic ARP Inspection
- IP Source Guard
- Storm control
- BPDU guard/filter

## Encryption & VPN
- IPsec policies
- Crypto maps
- VPN tunnels
- SSL/TLS settings

## Security Recommendations
List any missing security best practices.

IMPORTANT: Never include actual passwords, keys, or community strings.
If a section is not configured, write "Not configured"."""

        return self._call_api(prompt, max_tokens=3000)

    def generate_vlan_documentation(self, config: str) -> str:
        """
        Generate VLAN documentation.

        Args:
            config: Full device configuration text.

        Returns:
            Markdown-formatted VLAN documentation.
        """
        prompt = f"""Document the VLAN configuration.

Config:
```
{config}
```

Create documentation with:

## VLAN Summary
Total VLANs, naming convention used, organization

## VLAN Table
| VLAN ID | Name | Description | SVI IP | State | Purpose |

## VLAN Interfaces
Which interfaces are in which VLANs (access ports)

## Trunk Ports
| Interface | Native VLAN | Allowed VLANs |

## Voice VLANs
Any voice VLAN configurations

If VLANs are not configured, write "No VLANs configured (likely a router)"."""

        return self._call_api(prompt, max_tokens=2000)

    def generate_complete_documentation(
        self,
        config: str,
        hostname: str = None,
        output_file: str = None,
        include_vlans: bool = True,
        format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str:
        """
        Generate complete device documentation.

        Combines all documentation sections into a comprehensive document.

        Args:
            config: Full device configuration text.
            hostname: Device hostname (auto-detected if not provided).
            output_file: Optional path to save the documentation.
            include_vlans: Include VLAN section (disable for routers).
            format: Output format.

        Returns:
            Complete markdown documentation string.

        Example:
            >>> doc = generator.generate_complete_documentation(
            ...     config=router_config,
            ...     hostname="router-core-01",
            ...     output_file="docs/router-core-01.md"
            ... )
        """
        print(f"Generating documentation...")

        # Get overview
        print("  → Extracting device overview...")
        overview = self.generate_device_overview(config, hostname)

        print("  → Building interface documentation...")
        interfaces = self.generate_interface_documentation(config, format)

        print("  → Documenting routing configuration...")
        routing = self.generate_routing_documentation(config)

        print("  → Documenting security configuration...")
        security = self.generate_security_documentation(config)

        vlans_section = ""
        if include_vlans:
            print("  → Documenting VLAN configuration...")
            vlans = self.generate_vlan_documentation(config)
            vlans_section = f"""
---

## VLAN Configuration

{vlans}
"""

        # Build complete doc
        doc = f"""# {overview.hostname} - Network Device Documentation

| Property | Value |
|----------|-------|
| **Generated** | {overview.generated_at} |
| **Device Role** | {overview.role.value} |
| **Device Type** | {overview.device_type.value} |
| **Management IP** | {overview.management_ip or 'N/A'} |
| **Software Version** | {overview.software_version or 'N/A'} |

---

## Executive Summary

**{overview.hostname}** is a **{overview.role.value}** running {overview.device_type.value}.

**Routing Protocols**: {', '.join(overview.routing_protocols) if overview.routing_protocols else 'None configured'}

**Key Features**: {', '.join(overview.key_features) if overview.key_features else 'Standard configuration'}

**Interface Count**: {overview.interface_count}
**VLAN Count**: {overview.vlan_count}

{f"**Notable**: {overview.notable_config}" if overview.notable_config else ""}

---

## Interface Configuration

{interfaces}

---

## Routing Configuration

{routing}
{vlans_section}
---

## Security Configuration

{security}

---

## Documentation Metadata

| Field | Value |
|-------|-------|
| **Generated At** | {overview.generated_at} |
| **Generator** | AI Documentation Pipeline (Chapter 13) |
| **Source** | Running configuration |
| **AI Model** | {self.model} |

### API Usage for This Document
```
Input Tokens:  {self._usage_stats['input_tokens']}
Output Tokens: {self._usage_stats['output_tokens']}
API Calls:     {self._usage_stats['requests']}
Est. Cost:     ${self.get_usage_stats()['estimated_cost_usd']:.4f}
```

---

*This documentation is auto-generated from device configuration.*
*Review for accuracy before use in production decisions.*
"""

        # Save if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(doc)
            print(f"  ✓ Saved to {output_file}")

        print(f"  ✓ Documentation complete!")
        return doc


# =============================================================================
# Example Usage and Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 13: Network Documentation Basics")
    print("Configuration Documentation Generator Demo")
    print("=" * 70)

    # Sample Cisco IOS configuration for testing
    SAMPLE_CONFIG = '''
hostname router-core-01
!
version 15.7
!
enable secret 9 $9$xxxxxxxxxxxx
!
aaa new-model
aaa authentication login default group tacacs+ local
aaa authorization exec default group tacacs+ local
aaa accounting exec default start-stop group tacacs+
!
ip domain-name company.local
ip name-server 10.0.0.10
ip name-server 10.0.0.11
!
interface Loopback0
 description Management Loopback
 ip address 192.168.1.1 255.255.255.255
!
interface GigabitEthernet0/0
 description Uplink to ISP Primary
 ip address 203.0.113.1 255.255.255.252
 ip access-group EDGE_INBOUND in
 no shutdown
!
interface GigabitEthernet0/1
 description Connection to Core Switch
 ip address 10.0.1.1 255.255.255.0
 ip ospf authentication message-digest
 ip ospf message-digest-key 1 md5 7 xxxxxxxx
 no shutdown
!
interface GigabitEthernet0/2
 description Connection to DR Site
 ip address 10.0.2.1 255.255.255.0
 no shutdown
!
interface GigabitEthernet0/3
 description Unused - Shutdown
 shutdown
!
router ospf 1
 router-id 192.168.1.1
 auto-cost reference-bandwidth 100000
 passive-interface default
 no passive-interface GigabitEthernet0/1
 network 10.0.0.0 0.0.255.255 area 0
 network 192.168.1.1 0.0.0.0 area 0
!
router bgp 65001
 bgp router-id 192.168.1.1
 bgp log-neighbor-changes
 neighbor 203.0.113.2 remote-as 65002
 neighbor 203.0.113.2 description ISP_PRIMARY
 neighbor 203.0.113.2 password 7 xxxxxxxx
 !
 address-family ipv4
  network 10.0.0.0 mask 255.255.0.0
  neighbor 203.0.113.2 activate
  neighbor 203.0.113.2 prefix-list ADVERTISE_TO_ISP out
 exit-address-family
!
ip prefix-list ADVERTISE_TO_ISP seq 10 permit 10.0.0.0/16
!
ip access-list extended EDGE_INBOUND
 remark Block RFC1918 from Internet
 deny ip 10.0.0.0 0.255.255.255 any
 deny ip 172.16.0.0 0.15.255.255 any
 deny ip 192.168.0.0 0.0.255.255 any
 remark Allow established connections
 permit tcp any any established
 remark Allow specific services
 permit tcp any host 203.0.113.1 eq 22
 permit icmp any any echo-reply
 deny ip any any log
!
ip access-list extended MANAGEMENT_ACCESS
 permit tcp 10.0.0.0 0.0.255.255 any eq 22
 permit tcp 10.0.0.0 0.0.255.255 any eq 23
 deny ip any any log
!
snmp-server community R3@dOnly RO
snmp-server location "Main Data Center, Rack A1"
snmp-server contact "NOC Team - noc@company.com"
snmp-server enable traps
!
logging buffered 16384 informational
logging host 10.0.0.20
logging source-interface Loopback0
!
ntp server 10.0.0.15 prefer
ntp server 10.0.0.16
!
line con 0
 exec-timeout 5 0
 logging synchronous
line vty 0 4
 access-class MANAGEMENT_ACCESS in
 exec-timeout 15 0
 logging synchronous
 transport input ssh
line vty 5 15
 access-class MANAGEMENT_ACCESS in
 exec-timeout 15 0
 transport input ssh
!
end
'''

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set.")
        print("   Set it to run the full demo:\n")
        print("   export ANTHROPIC_API_KEY='your-key-here'\n")
        print("   Showing config parsing only (no API calls)...\n")

        # Demo basic parsing
        generator = ConfigDocumentationGenerator.__new__(ConfigDocumentationGenerator)
        generator.client = None

        print("-" * 70)
        print("Basic Parsing Demo (No API)")
        print("-" * 70)

        hostname = generator.extract_hostname(SAMPLE_CONFIG)
        print(f"\nHostname: {hostname}")

        device_type = generator.detect_device_type(SAMPLE_CONFIG)
        print(f"Device Type: {device_type.value}")

        interfaces = generator.extract_interfaces_basic(SAMPLE_CONFIG)
        print(f"\nInterfaces Found: {len(interfaces)}")
        for iface in interfaces[:5]:
            print(f"  - {iface.name}: {iface.ip_address or 'No IP'} ({iface.status})")

        vlans = generator.extract_vlans_basic(SAMPLE_CONFIG)
        print(f"\nVLANs Found: {len(vlans)}")

    else:
        # Full demo with API
        generator = ConfigDocumentationGenerator()

        print("\n" + "-" * 70)
        print("Full Documentation Demo")
        print("-" * 70)

        # Generate complete documentation
        doc = generator.generate_complete_documentation(
            config=SAMPLE_CONFIG,
            hostname="router-core-01",
            output_file="router-core-01-doc.md",
            include_vlans=False  # Router config, no VLANs
        )

        print("\n" + "-" * 70)
        print("Documentation Preview (first 2000 chars)")
        print("-" * 70)
        print(doc[:2000])
        print("\n...[truncated]...\n")

        # Show usage stats
        stats = generator.get_usage_stats()
        print("-" * 70)
        print("API Usage Statistics")
        print("-" * 70)
        print(f"  Model:         {stats['model']}")
        print(f"  API Calls:     {stats['requests']}")
        print(f"  Input Tokens:  {stats['input_tokens']:,}")
        print(f"  Output Tokens: {stats['output_tokens']:,}")
        print(f"  Est. Cost:     ${stats['estimated_cost_usd']:.4f}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
