#!/usr/bin/env python3
"""
Multi-Vendor Config Parser - Universal Network Config Analysis

Parse configurations from Cisco, Juniper, Arista, and other vendors using AI.

From: AI for Networking Engineers - Volume 1, Chapter 9
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from multi_vendor_parser import MultiVendorParser

    parser = MultiVendorParser()
    result = parser.parse_config(config_text, vendor="cisco")
"""

import re
from typing import Dict, Any, Optional, List
from enum import Enum


class Vendor(str, Enum):
    """Supported vendors."""
    CISCO_IOS = "cisco_ios"
    CISCO_NXOS = "cisco_nxos"
    JUNIPER = "juniper"
    ARISTA = "arista"
    UNKNOWN = "unknown"


class MultiVendorParser:
    """
    Parse network configurations from multiple vendors.

    Features:
    - Auto-detect vendor
    - Normalize data structures
    - Extract common information (interfaces, routing, ACLs)
    - Handle vendor-specific quirks
    """

    def __init__(self):
        """Initialize parser."""
        self.vendor_patterns = {
            Vendor.CISCO_IOS: [
                r'Cisco IOS Software',
                r'hostname\s+\S+',
                r'interface (GigabitEthernet|FastEthernet)'
            ],
            Vendor.CISCO_NXOS: [
                r'Cisco Nexus Operating System',
                r'feature (ospf|bgp|interface-vlan)'
            ],
            Vendor.JUNIPER: [
                r'version \d+\.\d+',
                r'set system host-name',
                r'set interfaces'
            ],
            Vendor.ARISTA: [
                r'Arista',
                r'!Command: show running-config'
            ]
        }

    def detect_vendor(self, config: str) -> Vendor:
        """
        Auto-detect vendor from config.

        Args:
            config: Configuration text

        Returns:
            Detected vendor enum
        """
        config_lower = config.lower()

        for vendor, patterns in self.vendor_patterns.items():
            matches = sum(1 for pattern in patterns
                         if re.search(pattern, config, re.IGNORECASE))

            # If majority of patterns match, it's this vendor
            if matches >= len(patterns) / 2:
                return vendor

        return Vendor.UNKNOWN

    def parse_config(
        self,
        config: str,
        vendor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse configuration and extract structured data.

        Args:
            config: Configuration text
            vendor: Optional vendor override (auto-detect if not provided)

        Returns:
            Normalized configuration data
        """
        # Detect vendor if not specified
        if not vendor:
            detected_vendor = self.detect_vendor(config)
        else:
            detected_vendor = Vendor(vendor)

        # Parse based on vendor
        if detected_vendor in [Vendor.CISCO_IOS, Vendor.CISCO_NXOS]:
            return self._parse_cisco(config, detected_vendor)
        elif detected_vendor == Vendor.JUNIPER:
            return self._parse_juniper(config)
        elif detected_vendor == Vendor.ARISTA:
            return self._parse_arista(config)
        else:
            return {"error": "Unknown vendor", "raw_config": config}

    def _parse_cisco(self, config: str, vendor: Vendor) -> Dict[str, Any]:
        """Parse Cisco IOS/NXOS configuration."""
        data = {
            "vendor": vendor.value,
            "hostname": self._extract_hostname_cisco(config),
            "interfaces": self._extract_interfaces_cisco(config),
            "routing": self._extract_routing_cisco(config),
            "acls": self._extract_acls_cisco(config),
            "vlans": self._extract_vlans_cisco(config)
        }
        return data

    def _parse_juniper(self, config: str) -> Dict[str, Any]:
        """Parse Juniper configuration."""
        data = {
            "vendor": Vendor.JUNIPER.value,
            "hostname": self._extract_hostname_juniper(config),
            "interfaces": self._extract_interfaces_juniper(config),
            "routing": [],
            "acls": [],
            "vlans": []
        }
        return data

    def _parse_arista(self, config: str) -> Dict[str, Any]:
        """Parse Arista EOS configuration."""
        # Arista uses similar syntax to Cisco IOS
        data = self._parse_cisco(config, Vendor.CISCO_IOS)
        data["vendor"] = Vendor.ARISTA.value
        return data

    def _extract_hostname_cisco(self, config: str) -> Optional[str]:
        """Extract hostname from Cisco config."""
        match = re.search(r'hostname\s+(\S+)', config)
        return match.group(1) if match else None

    def _extract_hostname_juniper(self, config: str) -> Optional[str]:
        """Extract hostname from Juniper config."""
        match = re.search(r'set system host-name\s+(\S+)', config)
        return match.group(1) if match else None

    def _extract_interfaces_cisco(self, config: str) -> List[Dict[str, Any]]:
        """Extract interface configurations from Cisco."""
        interfaces = []
        current_interface = None
        current_config = []

        for line in config.splitlines():
            line = line.strip()

            # Check for interface declaration
            if line.startswith('interface '):
                # Save previous interface
                if current_interface:
                    interfaces.append(self._parse_interface_block_cisco(
                        current_interface,
                        current_config
                    ))

                # Start new interface
                current_interface = line.replace('interface ', '')
                current_config = [line]

            elif current_interface and (line.startswith(' ') or not line):
                # Part of current interface
                current_config.append(line)

            elif current_interface:
                # End of interface block
                interfaces.append(self._parse_interface_block_cisco(
                    current_interface,
                    current_config
                ))
                current_interface = None
                current_config = []

        # Save last interface
        if current_interface:
            interfaces.append(self._parse_interface_block_cisco(
                current_interface,
                current_config
            ))

        return interfaces

    def _parse_interface_block_cisco(
        self,
        name: str,
        config_lines: List[str]
    ) -> Dict[str, Any]:
        """Parse individual interface configuration block."""
        config_text = '\n'.join(config_lines)

        interface = {
            "name": name,
            "description": None,
            "ip_address": None,
            "subnet_mask": None,
            "status": "unknown",
            "vlan": None
        }

        # Extract description
        desc_match = re.search(r'description\s+(.+)', config_text)
        if desc_match:
            interface["description"] = desc_match.group(1).strip()

        # Extract IP address
        ip_match = re.search(r'ip address\s+(\S+)\s+(\S+)', config_text)
        if ip_match:
            interface["ip_address"] = ip_match.group(1)
            interface["subnet_mask"] = ip_match.group(2)

        # Determine status
        if 'shutdown' in config_text and 'no shutdown' not in config_text:
            interface["status"] = "admin_down"
        elif 'no shutdown' in config_text:
            interface["status"] = "up"

        # Extract VLAN (for switchport)
        vlan_match = re.search(r'switchport access vlan\s+(\d+)', config_text)
        if vlan_match:
            interface["vlan"] = int(vlan_match.group(1))

        return interface

    def _extract_interfaces_juniper(self, config: str) -> List[Dict[str, Any]]:
        """Extract interface configurations from Juniper."""
        interfaces = []

        # Juniper uses: set interfaces ge-0/0/0 unit 0 family inet address 10.1.1.1/24
        pattern = r'set interfaces\s+(\S+)\s+unit\s+\d+\s+family\s+inet\s+address\s+(\S+)'

        for match in re.finditer(pattern, config):
            interface_name = match.group(1)
            ip_with_prefix = match.group(2)

            # Parse IP and prefix
            if '/' in ip_with_prefix:
                ip, prefix = ip_with_prefix.split('/')
                interfaces.append({
                    "name": interface_name,
                    "ip_address": ip,
                    "prefix_length": int(prefix),
                    "status": "unknown"
                })

        return interfaces

    def _extract_routing_cisco(self, config: str) -> List[Dict[str, Any]]:
        """Extract routing protocol configuration."""
        routing = []

        # OSPF
        if 'router ospf' in config:
            routing.append({"protocol": "ospf", "enabled": True})

        # BGP
        bgp_match = re.search(r'router bgp\s+(\d+)', config)
        if bgp_match:
            routing.append({
                "protocol": "bgp",
                "enabled": True,
                "as_number": int(bgp_match.group(1))
            })

        # EIGRP
        eigrp_match = re.search(r'router eigrp\s+(\d+)', config)
        if eigrp_match:
            routing.append({
                "protocol": "eigrp",
                "enabled": True,
                "as_number": int(eigrp_match.group(1))
            })

        return routing

    def _extract_acls_cisco(self, config: str) -> List[Dict[str, Any]]:
        """Extract ACL configurations."""
        acls = []

        # Find ACL declarations
        acl_pattern = r'ip access-list\s+(extended|standard)\s+(\S+)'

        for match in re.finditer(acl_pattern, config):
            acl_type = match.group(1)
            acl_name = match.group(2)

            acls.append({
                "name": acl_name,
                "type": acl_type,
                "entries": []  # Could extract individual rules
            })

        return acls

    def _extract_vlans_cisco(self, config: str) -> List[Dict[str, Any]]:
        """Extract VLAN configurations."""
        vlans = []

        # VLAN declarations
        vlan_pattern = r'vlan\s+(\d+)'

        for match in re.finditer(vlan_pattern, config):
            vlan_id = int(match.group(1))

            # Look for VLAN name
            vlan_name = None
            name_pattern = rf'vlan\s+{vlan_id}\s+name\s+(\S+)'
            name_match = re.search(name_pattern, config)
            if name_match:
                vlan_name = name_match.group(1)

            vlans.append({
                "vlan_id": vlan_id,
                "name": vlan_name
            })

        return vlans


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Multi-Vendor Parser Demo
    ========================================
    Parse configs from Cisco, Juniper, Arista
    ========================================
    """)

    parser = MultiVendorParser()

    # Test 1: Cisco IOS config
    print("\nTest 1: Parse Cisco IOS Configuration")
    print("-" * 60)

    cisco_config = """
hostname CORE-RTR-01

interface GigabitEthernet0/0
 description WAN Interface
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description LAN Interface
 ip address 10.1.1.1 255.255.255.0
 no shutdown

interface GigabitEthernet0/2
 description Unused
 shutdown

router ospf 1
 network 10.1.1.0 0.0.0.255 area 0

router bgp 65001
 neighbor 203.0.113.2 remote-as 65002

ip access-list extended PERMIT_INTERNAL
 permit ip 10.1.1.0 0.0.0.255 any
 deny ip any any log

vlan 10
 name USERS

vlan 20
 name SERVERS
"""

    detected = parser.detect_vendor(cisco_config)
    print(f"Detected vendor: {detected.value}")

    parsed = parser.parse_config(cisco_config)
    print(f"\nHostname: {parsed['hostname']}")
    print(f"Interfaces: {len(parsed['interfaces'])}")
    for iface in parsed['interfaces']:
        status_icon = "✓" if iface['status'] == "up" else "✗"
        ip_info = f"{iface['ip_address']}/{iface['subnet_mask']}" if iface['ip_address'] else "No IP"
        print(f"  {status_icon} {iface['name']:20s} {ip_info:20s} ({iface['status']})")

    print(f"\nRouting protocols: {len(parsed['routing'])}")
    for protocol in parsed['routing']:
        print(f"  - {protocol['protocol'].upper()}: enabled")

    print(f"\nACLs: {len(parsed['acls'])}")
    for acl in parsed['acls']:
        print(f"  - {acl['name']} ({acl['type']})")

    print(f"\nVLANs: {len(parsed['vlans'])}")
    for vlan in parsed['vlans']:
        name = vlan['name'] or 'unnamed'
        print(f"  - VLAN {vlan['vlan_id']}: {name}")

    # Test 2: Juniper config
    print("\n\nTest 2: Parse Juniper Configuration")
    print("-" * 60)

    juniper_config = """
set system host-name JUNIPER-RTR-01
set interfaces ge-0/0/0 unit 0 family inet address 10.1.1.1/24
set interfaces ge-0/0/1 unit 0 family inet address 10.2.2.1/24
"""

    detected = parser.detect_vendor(juniper_config)
    print(f"Detected vendor: {detected.value}")

    parsed = parser.parse_config(juniper_config)
    print(f"\nHostname: {parsed['hostname']}")
    print(f"Interfaces: {len(parsed['interfaces'])}")
    for iface in parsed['interfaces']:
        print(f"  - {iface['name']:15s} {iface['ip_address']}/{iface['prefix_length']}")

    print("\n✅ Demo complete!")
    print("\n💡 Supported Vendors:")
    print("  - Cisco IOS/IOS-XE")
    print("  - Cisco NX-OS")
    print("  - Juniper Junos")
    print("  - Arista EOS")
