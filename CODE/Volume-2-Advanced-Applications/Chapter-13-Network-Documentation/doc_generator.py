"""
Chapter 13: Network Documentation Basics
Configuration Documentation Generator

Auto-generate documentation from network device configurations.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

from anthropic import Anthropic
from typing import Dict, List
import json
from datetime import datetime


class ConfigDocumentationGenerator:
    """Generate documentation automatically from network configs."""

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate_device_overview(self, config: str, hostname: str) -> Dict:
        """Generate high-level device documentation."""

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
        """Generate markdown table of all interfaces."""

        prompt = f"""Extract all interfaces from this config and create a markdown table.

Config:
{config}

Create a table with columns:
| Interface | IP Address | Status | Description | VLAN/VRF |

Include ALL interfaces (physical, loopback, tunnel, etc.)

Markdown table:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_routing_documentation(self, config: str) -> str:
        """Document routing configuration."""

        prompt = f"""Document the routing configuration from this device.

Config:
{config}

Create documentation covering:

## Routing Protocols
- Which protocols are enabled
- Process IDs, AS numbers
- Router IDs

## Static Routes
- Destination networks
- Next hops
- Purpose

## Route Redistribution
- What's redistributed where
- Filters applied

## Routing Policies
- Route-maps
- Prefix-lists
- Access-lists affecting routing

Format as markdown with sections and bullet points.

Documentation:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_security_documentation(self, config: str) -> str:
        """Document security features and policies."""

        prompt = f"""Document the security configuration from this device.

Config:
{config}

Cover:

## Access Control
- ACLs defined and their purpose
- Where they're applied

## Authentication
- AAA configuration
- TACACS/RADIUS servers
- Local users

## Management Access
- SSH/Telnet configuration
- Allowed management networks
- VTY line configuration

## Security Features
- Port security
- DHCP snooping
- DAI, IP Source Guard
- Any other security features

Markdown documentation:"""

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
        """Generate complete device documentation."""

        print(f"Generating documentation for {hostname}...")

        # Get all sections
        overview = self.generate_device_overview(config, hostname)
        interfaces = self.generate_interface_table(config)
        routing = self.generate_routing_documentation(config)
        security = self.generate_security_documentation(config)

        # Build complete doc
        doc = f"""# {hostname} - Device Documentation

**Generated**: {overview['generated_at']}
**Device Role**: {overview['role']}
**Management IP**: {overview['management_ip']}

---

## Overview

**Routing Protocols**: {', '.join(overview['routing_protocols'])}
**Key Features**: {', '.join(overview['key_features'])}

{overview['notable_config']}

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

*This documentation is auto-generated from device configuration. To update, regenerate from current config.*
"""

        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(doc)
            print(f"Documentation saved to {output_file}")

        return doc


# Example usage
if __name__ == "__main__":
    generator = ConfigDocumentationGenerator()

    # Sample config
    config = """
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
ip access-list extended MANAGEMENT_ACCESS
 permit tcp 10.0.0.0 0.0.255.255 any eq 22
 deny ip any any log
!
line vty 0 4
 access-class MANAGEMENT_ACCESS in
 transport input ssh
    """

    # Generate full documentation
    doc = generator.generate_complete_documentation(
        config=config,
        hostname="router-core-01",
        output_file="router-core-01-doc.md"
    )

    print("\n" + "="*60)
    print("DOCUMENTATION GENERATED:")
    print("="*60)
    print(doc[:500] + "...\n")
