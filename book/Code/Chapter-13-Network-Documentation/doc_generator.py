"""
Chapter 13: Network Documentation Basics
Configuration Documentation Generator

Auto-generate documentation from network device configurations using Claude AI.

How it works:
    1. You feed a device's running/startup config to this script
    2. Claude reads the config and extracts structured data (like a senior
       engineer would, but in seconds)
    3. Output is clean Markdown documentation: device overview, interface
       tables, routing details, and security audit

Think of it like "show tech-support" but the output is human-readable
documentation instead of raw CLI dumps.

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List

from anthropic import Anthropic

# Model to use for documentation generation.
# Claude Sonnet is the best balance of quality and cost for structured extraction.
# For simpler tasks (e.g., just pulling interface lists), you could use Haiku
# to cut costs by ~10x.
MODEL = "claude-sonnet-4-20250514"


class ConfigDocumentationGenerator:
    """Generate documentation automatically from network configs.

    This class sends device configurations to Claude and gets back
    structured documentation. Each method handles a different section
    (overview, interfaces, routing, security).

    Networking analogy: This is like having a junior engineer read every
    config and fill out a standardized documentation template -- except
    it runs in seconds and never misses a section.
    """

    def __init__(self, api_key: str = None):
        # Uses ANTHROPIC_API_KEY environment variable if no key is passed.
        # This is the recommended approach -- never hardcode API keys.
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def _parse_json_response(self, text: str) -> dict:
        """Safely extract JSON from an LLM response.

        LLMs sometimes wrap JSON in markdown code fences or add
        explanation text before/after. This method handles those cases.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code fences
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")

    def generate_device_overview(self, config: str, hostname: str) -> Dict:
        """Generate high-level device documentation.

        Takes a raw device config and returns a structured dictionary
        with the device's role, IPs, protocols, and notable features.

        Args:
            config: The full running/startup configuration text
            hostname: Device hostname (used in the prompt for context)

        Returns:
            Dictionary with keys: hostname, role, management_ip,
            routing_protocols, key_features, interface_summary,
            notable_config, generated_at
        """

        prompt = f"""Analyze this network device configuration and create documentation.

Device: {hostname}
Configuration:
{config}

Extract and document:
1. Device role (core router, access switch, firewall, etc.)
2. Management IP address (Loopback0 or management interface)
3. Routing protocols in use (BGP, OSPF, EIGRP, static)
4. Key features enabled (HSRP, VRF, QoS, etc.)
5. Interface count and types
6. Notable configurations or policies

Return ONLY valid JSON with no additional text:
{{
    "hostname": "device name",
    "role": "device role",
    "management_ip": "IP address",
    "routing_protocols": ["list of protocols"],
    "key_features": ["list of features"],
    "interface_summary": "summary of interfaces",
    "notable_config": "anything important to know"
}}"""

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        doc_data = self._parse_json_response(response.content[0].text)
        doc_data['generated_at'] = datetime.now().isoformat()

        return doc_data

    def generate_interface_table(self, config: str) -> str:
        """Generate markdown table of all interfaces.

        Extracts every interface from the config and formats them into
        a Markdown table -- similar to "show ip interface brief" but
        with descriptions and VLAN/VRF info included.
        """

        prompt = f"""Extract all interfaces from this network device config and create a markdown table.

Config:
{config}

Create a table with these exact columns:
| Interface | IP Address | Status | Description | VLAN/VRF |

Rules:
- Include ALL interfaces: physical, loopback, tunnel, VLAN, port-channel
- If no IP address is configured, show "L2" or "N/A"
- For status, use "up" if "no shutdown" or "down" if "shutdown"
- If no description is configured, show "-"

Return ONLY the markdown table, no other text."""

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_routing_documentation(self, config: str) -> str:
        """Document routing configuration.

        Produces a structured breakdown of all routing protocols,
        static routes, redistribution, and route policies found
        in the config.
        """

        prompt = f"""Document the routing configuration from this network device.

Config:
{config}

Create documentation covering these sections. If a section does not
apply (e.g., no redistribution configured), write "None configured."

## Routing Protocols
- Which protocols are enabled (OSPF, BGP, EIGRP, IS-IS)
- Process IDs, AS numbers, area assignments
- Router IDs
- Neighbor statements and peer groups

## Static Routes
- Destination networks and next hops
- Administrative distance overrides
- Purpose (if evident from description or context)

## Route Redistribution
- What is redistributed into what
- Route-maps or filters applied to redistribution

## Routing Policies
- Route-maps and their match/set clauses
- Prefix-lists and community-lists
- Access-lists that affect routing decisions

Format as clean markdown with sections and bullet points."""

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_security_documentation(self, config: str) -> str:
        """Document security features and policies.

        Reviews the config for security-relevant settings: ACLs, AAA,
        management access controls, and L2/L3 security features.
        """

        prompt = f"""Document the security configuration from this network device.
Focus on what a security auditor or a new team member would need to know.

Config:
{config}

Cover these sections. If a section does not apply, write "Not configured."

## Access Control Lists
- ACLs defined, their type (standard/extended/named), and purpose
- Where each ACL is applied (interface, VTY, route-map, etc.)

## Authentication & Authorization
- AAA configuration (authentication, authorization, accounting)
- TACACS+/RADIUS server definitions
- Local user accounts (note: do NOT include passwords in docs)

## Management Plane Security
- SSH/Telnet configuration and version
- Allowed management source networks
- VTY line access restrictions
- Console and aux port settings

## Data Plane Security Features
- Port security settings
- DHCP snooping, Dynamic ARP Inspection (DAI)
- IP Source Guard, uRPF
- Storm control, BPDU guard
- Any other security features

Format as clean markdown."""

        response = self.client.messages.create(
            model=MODEL,
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
        """Generate complete device documentation.

        Calls each section generator and assembles the final Markdown
        document. This makes 4 API calls total (overview, interfaces,
        routing, security).

        Args:
            config: Full device configuration text
            hostname: Device hostname
            output_file: Optional path to save the documentation

        Returns:
            Complete Markdown documentation string
        """

        print(f"Generating documentation for {hostname}...")

        # Get all sections -- each is a separate API call.
        # Why separate calls instead of one big prompt?
        # 1. Each section gets focused attention from the model
        # 2. If one section fails, the others still succeed
        # 3. Easier to update individual section prompts
        print(f"  [1/4] Device overview...")
        overview = self.generate_device_overview(config, hostname)
        print(f"  [2/4] Interface table...")
        interfaces = self.generate_interface_table(config)
        print(f"  [3/4] Routing documentation...")
        routing = self.generate_routing_documentation(config)
        print(f"  [4/4] Security documentation...")
        security = self.generate_security_documentation(config)

        # Build complete doc
        doc = f"""# {hostname} - Device Documentation

**Generated**: {overview['generated_at']}
**Device Role**: {overview['role']}
**Management IP**: {overview['management_ip']}

---

## Overview

**Routing Protocols**: {', '.join(overview.get('routing_protocols', []))}
**Key Features**: {', '.join(overview.get('key_features', []))}

{overview.get('notable_config', 'No notable configurations.')}

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

*This documentation was auto-generated by sending the device configuration
to Claude AI. To update, re-run this script against the current config.*
"""

        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(doc)
            print(f"  Documentation saved to {output_file}")

        return doc


# ---------------------------------------------------------------------------
# Example usage -- run this file directly to see it in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("")
        print("Get a key at: https://console.anthropic.com/")
        exit(1)

    generator = ConfigDocumentationGenerator()

    # Sample Cisco IOS config -- replace with your own device configs
    config = """
hostname router-core-01
!
interface Loopback0
 ip address 192.168.1.1 255.255.255.255
!
interface GigabitEthernet0/0
 description Uplink to ISP-A (Lumen)
 ip address 203.0.113.1 255.255.255.252
 no shutdown
!
interface GigabitEthernet0/1
 description Trunk to DC-SPINE-01
 ip address 10.0.1.1 255.255.255.0
 no shutdown
!
interface GigabitEthernet0/2
 description Management OOB
 vrf forwarding MGMT
 ip address 172.16.0.1 255.255.255.0
 no shutdown
!
router ospf 1
 router-id 192.168.1.1
 network 10.0.0.0 0.0.255.255 area 0
 network 192.168.1.1 0.0.0.0 area 0
 passive-interface default
 no passive-interface GigabitEthernet0/1
!
router bgp 65001
 bgp router-id 192.168.1.1
 bgp log-neighbor-changes
 neighbor 203.0.113.2 remote-as 65002
 neighbor 203.0.113.2 description ISP-A_Lumen
 neighbor 203.0.113.2 route-map ISP-A-IN in
 neighbor 203.0.113.2 route-map ISP-A-OUT out
!
ip prefix-list ADVERTISE-TO-ISP seq 10 permit 198.51.100.0/24
!
route-map ISP-A-OUT permit 10
 match ip address prefix-list ADVERTISE-TO-ISP
!
route-map ISP-A-IN permit 10
 set local-preference 200
!
ip access-list extended MANAGEMENT_ACCESS
 permit tcp 10.0.0.0 0.0.255.255 any eq 22
 permit tcp 172.16.0.0 0.0.0.255 any eq 22
 deny ip any any log
!
line vty 0 4
 access-class MANAGEMENT_ACCESS in
 transport input ssh
!
ntp server 10.0.0.10
ntp server 10.0.0.11
!
logging host 10.0.0.20
logging trap informational
    """

    # Generate full documentation
    doc = generator.generate_complete_documentation(
        config=config,
        hostname="router-core-01",
        output_file="router-core-01-doc.md"
    )

    print("\n" + "="*60)
    print("DOCUMENTATION GENERATED SUCCESSFULLY")
    print("="*60)
    print(doc[:500] + "...\n")
