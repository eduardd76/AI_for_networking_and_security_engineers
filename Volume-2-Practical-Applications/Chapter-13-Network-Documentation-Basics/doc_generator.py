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
- Complete device documentation assembly

Usage:
    from doc_generator import ConfigDocumentationGenerator

    generator = ConfigDocumentationGenerator()
    doc = generator.generate_complete_documentation(
        config=config_text,
        hostname="router-core-01",
        output_file="router-core-01.md"
    )
"""

from anthropic import Anthropic
from typing import Dict
import json
from datetime import datetime
import os


class ConfigDocumentationGenerator:
    """
    Generate documentation automatically from network configs.

    Uses Claude to analyze device configurations and produce
    structured, consistent documentation in markdown format.

    Attributes:
        client: Anthropic API client instance

    Example:
        >>> generator = ConfigDocumentationGenerator()
        >>> overview = generator.generate_device_overview(config, "router-01")
        >>> print(overview['role'])
        'Core Router'
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the documentation generator.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate_device_overview(self, config: str, hostname: str) -> Dict:
        """
        Generate high-level device documentation.

        Extracts key information about the device including its role,
        management IP, routing protocols, and notable configurations.

        Args:
            config: Full device configuration text
            hostname: Device hostname for identification

        Returns:
            Dictionary containing:
                - hostname: Device name
                - role: Device role (e.g., "Core Router", "Access Switch")
                - management_ip: Primary management IP address
                - routing_protocols: List of routing protocols in use
                - key_features: List of enabled features
                - interface_summary: Brief description of interfaces
                - notable_config: Important configuration notes
                - generated_at: ISO timestamp of generation
        """
        prompt = f"""Analyze this network device configuration and create documentation.

Device: {hostname}
Configuration:
{config}

Extract and document:
1. Device role (core router, access switch, firewall, etc.)
2. Management IP address
3. Routing protocols in use (BGP, OSPF, EIGRP, static)
4. Key features enabled (HSRP, VRF, QoS, etc.)
5. Interface count and types
6. Notable configurations or policies

Return as JSON:
{{
    "hostname": "device name",
    "role": "device role",
    "management_ip": "IP address",
    "routing_protocols": ["list of protocols"],
    "key_features": ["list of features"],
    "interface_summary": "summary of interfaces",
    "notable_config": "anything important to know"
}}

JSON:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        doc_data = json.loads(response.content[0].text)
        doc_data['generated_at'] = datetime.now().isoformat()

        return doc_data

    def generate_interface_table(self, config: str) -> str:
        """
        Generate markdown table of all interfaces.

        Parses the configuration to extract all interface definitions
        and formats them as a readable markdown table.

        Args:
            config: Full device configuration text

        Returns:
            Markdown-formatted table with columns:
            Interface | IP Address | Status | Description | VLAN/VRF
        """
        prompt = f"""Extract all interfaces from this config and create a markdown table.

Config:
{config}

Create a table with columns:
| Interface | IP Address | Status | Description | VLAN/VRF |

Include ALL interfaces (physical, loopback, tunnel, VLAN, etc.)
For Status, use "Up" if "no shutdown" or status not specified, "Down" if "shutdown"
For missing values, use "N/A"

Markdown table:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_routing_documentation(self, config: str) -> str:
        """
        Document routing configuration.

        Extracts and documents all routing-related configuration including
        protocols, static routes, redistribution, and routing policies.

        Args:
            config: Full device configuration text

        Returns:
            Markdown-formatted routing documentation with sections for:
            - Routing Protocols
            - Static Routes
            - Route Redistribution
            - Routing Policies
        """
        prompt = f"""Document the routing configuration from this device.

Config:
{config}

Create documentation covering:

## Routing Protocols
- Which protocols are enabled
- Process IDs, AS numbers
- Router IDs
- Networks advertised

## Static Routes
- Destination networks
- Next hops
- Administrative distance if non-default
- Purpose (if determinable from description)

## Route Redistribution
- What's redistributed where
- Route-maps or filters applied
- Metrics set

## Routing Policies
- Route-maps defined
- Prefix-lists
- AS-path access-lists
- Community lists

Format as markdown with sections and bullet points.
If a section has no configuration, state "Not configured"

Documentation:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_security_documentation(self, config: str) -> str:
        """
        Document security features and policies.

        Extracts security-related configuration including access controls,
        authentication settings, and security features.

        Args:
            config: Full device configuration text

        Returns:
            Markdown-formatted security documentation with sections for:
            - Access Control
            - Authentication
            - Management Access
            - Security Features
        """
        prompt = f"""Document the security configuration from this device.

Config:
{config}

Cover:

## Access Control
- ACLs defined and their purpose
- Where ACLs are applied (interfaces, VTY, etc.)
- Traffic filtering policies

## Authentication
- AAA configuration
- TACACS+/RADIUS servers
- Local users (without passwords)
- Authentication methods

## Management Access
- SSH/Telnet configuration
- Allowed management networks
- VTY line configuration
- Console/AUX settings

## Security Features
- Port security
- DHCP snooping
- Dynamic ARP Inspection
- IP Source Guard
- Control-plane policing
- Any other security features

Format as markdown with sections and bullet points.
If a section has no configuration, state "Not configured"
Never include actual passwords or community strings in output.

Documentation:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_complete_documentation(
        self,
        config: str,
        hostname: str,
        output_file: str = None
    ) -> str:
        """
        Generate complete device documentation.

        Combines all documentation sections into a single comprehensive
        markdown document suitable for a documentation repository.

        Args:
            config: Full device configuration text
            hostname: Device hostname for identification
            output_file: Optional path to save the documentation

        Returns:
            Complete markdown documentation string

        Example:
            >>> doc = generator.generate_complete_documentation(
            ...     config=router_config,
            ...     hostname="router-core-01",
            ...     output_file="docs/router-core-01.md"
            ... )
            Generating documentation for router-core-01...
            Documentation saved to docs/router-core-01.md
        """
        print(f"Generating documentation for {hostname}...")

        # Get all sections
        print("  → Extracting device overview...")
        overview = self.generate_device_overview(config, hostname)

        print("  → Building interface table...")
        interfaces = self.generate_interface_table(config)

        print("  → Documenting routing configuration...")
        routing = self.generate_routing_documentation(config)

        print("  → Documenting security configuration...")
        security = self.generate_security_documentation(config)

        # Build complete doc
        doc = f"""# {hostname} - Device Documentation

**Generated**: {overview['generated_at']}
**Device Role**: {overview['role']}
**Management IP**: {overview['management_ip']}

---

## Overview

**Routing Protocols**: {', '.join(overview['routing_protocols']) if overview['routing_protocols'] else 'None configured'}
**Key Features**: {', '.join(overview['key_features']) if overview['key_features'] else 'Standard configuration'}

{overview.get('notable_config', '')}

---

## Interfaces

{interfaces}

---

## Routing Configuration

{routing}

---

## Security Configuration

{security}

---

## Maintenance Notes

**Last Config Update**: {overview['generated_at']}
**Documentation Source**: Auto-generated from running configuration
**Next Review**: [Schedule quarterly reviews]

---

*This documentation is auto-generated from device configuration using AI.*
*To update, regenerate from current config.*
"""

        # Save if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(doc)
            print(f"  ✓ Documentation saved to {output_file}")

        print(f"  ✓ Complete!")
        return doc


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Sample configuration for testing
    sample_config = """
hostname router-core-01
!
interface Loopback0
 ip address 192.168.1.1 255.255.255.255
!
interface GigabitEthernet0/0
 description Uplink to ISP
 ip address 203.0.113.1 255.255.255.252
 no shutdown
!
interface GigabitEthernet0/1
 description Connection to Datacenter
 ip address 10.0.1.1 255.255.255.0
 no shutdown
!
router ospf 1
 router-id 192.168.1.1
 network 10.0.0.0 0.0.255.255 area 0
 network 192.168.1.1 0.0.0.0 area 0
!
router bgp 65001
 bgp router-id 192.168.1.1
 neighbor 203.0.113.2 remote-as 65002
 neighbor 203.0.113.2 description ISP_PRIMARY
 !
 address-family ipv4
  network 10.0.0.0 mask 255.255.0.0
  neighbor 203.0.113.2 activate
 exit-address-family
!
ip access-list extended MANAGEMENT_ACCESS
 permit tcp 10.0.0.0 0.0.255.255 any eq 22
 deny ip any any log
!
line vty 0 4
 access-class MANAGEMENT_ACCESS in
 transport input ssh
!
"""

    print("=" * 60)
    print("Chapter 13: Network Documentation Basics")
    print("Configuration Documentation Generator Demo")
    print("=" * 60)

    # Initialize generator
    generator = ConfigDocumentationGenerator()

    # Generate complete documentation
    doc = generator.generate_complete_documentation(
        config=sample_config,
        hostname="router-core-01",
        output_file="router-core-01-doc.md"
    )

    print("\n" + "=" * 60)
    print("DOCUMENTATION PREVIEW (first 1000 chars):")
    print("=" * 60)
    print(doc[:1000] + "...\n")
