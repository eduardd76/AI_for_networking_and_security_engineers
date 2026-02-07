#!/usr/bin/env python3
"""
Network Topology Diagrammer

Generate network diagrams from CDP/LLDP neighbor data using AI.

From: AI for Networking Engineers - Volume 2, Chapter 13
Author: Eduard Dulharu (Ed Harmoosh)

This module provides:
- CDP/LLDP output parsing
- Mermaid diagram generation
- Connection details extraction
- Complete topology documentation

Usage:
    from topology_diagrammer import NetworkTopologyDiagrammer

    diagrammer = NetworkTopologyDiagrammer()
    topology_doc = diagrammer.create_topology_documentation(cdp_data)
"""

from anthropic import Anthropic
from typing import List, Dict
from datetime import datetime
import json


class NetworkTopologyDiagrammer:
    """
    Generate network diagrams from CDP/LLDP data.

    Parses neighbor discovery protocol output and generates
    visual topology diagrams in Mermaid format.

    Mermaid diagrams render automatically in:
    - GitHub README files
    - GitLab wikis
    - Notion pages
    - Confluence (with plugin)
    - Many documentation platforms

    Attributes:
        client: Anthropic API client instance

    Example:
        >>> diagrammer = NetworkTopologyDiagrammer()
        >>> neighbors = diagrammer.extract_neighbors_from_cdp(cdp_output)
        >>> print(len(neighbors))
        3
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the topology diagrammer.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def extract_neighbors_from_cdp(self, cdp_output: str) -> List[Dict]:
        """
        Extract neighbor information from CDP/LLDP output.

        Parses raw CDP or LLDP command output and extracts structured
        neighbor information.

        Args:
            cdp_output: Raw output from "show cdp neighbors detail" or
                       "show lldp neighbors detail" command

        Returns:
            List of dictionaries, each containing:
                - local_device: Name of the local device
                - local_interface: Interface on the local device
                - remote_device: Neighbor device hostname
                - remote_interface: Interface on the neighbor
                - platform: Neighbor device platform/model

        Example:
            >>> cdp = "Device ID: switch-01\\nInterface: Gi0/1..."
            >>> neighbors = diagrammer.extract_neighbors_from_cdp(cdp)
            >>> print(neighbors[0]['remote_device'])
            'switch-01'
        """
        prompt = f"""Extract neighbor information from this CDP/LLDP output.

Output:
{cdp_output}

Return JSON array of neighbors:
[
    {{
        "local_device": "this device name (extract from context or use 'local')",
        "local_interface": "interface on this device",
        "remote_device": "neighbor device name",
        "remote_interface": "interface on neighbor",
        "platform": "device platform/model"
    }}
]

If platform is not available, use "Unknown".
Normalize interface names (e.g., "Gi" -> "GigabitEthernet").

JSON:"""

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # Haiku for cost efficiency
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)

    def generate_mermaid_diagram(
        self,
        devices: List[str],
        connections: List[Dict]
    ) -> str:
        """
        Generate Mermaid diagram syntax from topology data.

        Creates a flowchart diagram showing network devices and
        their interconnections with interface labels.

        Args:
            devices: List of device hostnames
            connections: List of connection dictionaries from extract_neighbors_from_cdp

        Returns:
            Mermaid flowchart syntax string ready for rendering

        Note:
            Device shapes are chosen based on name patterns:
            - Routers (rtr/router): Rectangle
            - Switches (sw/switch): Cylinder
            - Firewalls (fw/firewall): Hexagon
        """
        devices_str = ", ".join(devices)
        connections_str = "\n".join([
            f"- {c['local_device']} ({c['local_interface']}) <--> "
            f"{c['remote_device']} ({c['remote_interface']})"
            for c in connections
        ])

        prompt = f"""Create a network topology diagram using Mermaid syntax.

Devices:
{devices_str}

Connections:
{connections_str}

Generate Mermaid flowchart syntax showing:
1. All devices as nodes with appropriate shapes:
   - Routers: rectangle [Router Name]
   - Switches: cylinder [(Switch Name)]
   - Firewalls: hexagon {{{{Firewall Name}}}}
   - Unknown: rectangle
2. All connections between them with bidirectional arrows
3. Interface labels on links (abbreviated, e.g., Gi0/1)
4. Use subgraphs to group by tier (Core, Distribution, Access) if determinable

Start with: flowchart TB

Mermaid syntax only (no markdown code fences):"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def create_topology_documentation(
        self,
        cdp_outputs: Dict[str, str]
    ) -> str:
        """
        Create complete topology documentation with diagram.

        Processes CDP/LLDP output from multiple devices and generates
        comprehensive topology documentation including a visual diagram
        and connection details table.

        Args:
            cdp_outputs: Dictionary mapping device hostname to CDP/LLDP output
                        Example: {"router-01": "Device ID: switch-01..."}

        Returns:
            Markdown-formatted topology documentation including:
            - Generation metadata
            - Mermaid topology diagram
            - Connection details table

        Example:
            >>> cdp_data = {
            ...     "router-01": cdp_from_router,
            ...     "switch-01": cdp_from_switch
            ... }
            >>> doc = diagrammer.create_topology_documentation(cdp_data)
        """
        print("Analyzing topology...")

        # Extract all neighbors
        all_neighbors = []
        all_devices = set()

        for device_name, cdp_output in cdp_outputs.items():
            print(f"  → Processing {device_name}...")
            neighbors = self.extract_neighbors_from_cdp(cdp_output)

            # Update local_device if not set
            for n in neighbors:
                if n['local_device'] == 'local':
                    n['local_device'] = device_name

            all_neighbors.extend(neighbors)
            all_devices.add(device_name)
            for n in neighbors:
                all_devices.add(n['remote_device'])

        print(f"  → Found {len(all_devices)} devices, {len(all_neighbors)} connections")

        # Remove duplicate connections (A->B and B->A)
        seen_connections = set()
        unique_neighbors = []
        for conn in all_neighbors:
            # Create a normalized key for the connection
            key = tuple(sorted([
                f"{conn['local_device']}:{conn['local_interface']}",
                f"{conn['remote_device']}:{conn['remote_interface']}"
            ]))
            if key not in seen_connections:
                seen_connections.add(key)
                unique_neighbors.append(conn)

        print(f"  → {len(unique_neighbors)} unique connections after deduplication")

        # Generate diagram
        print("  → Generating Mermaid diagram...")
        diagram = self.generate_mermaid_diagram(
            devices=list(all_devices),
            connections=unique_neighbors
        )

        # Create documentation
        doc = f"""# Network Topology Documentation

**Generated**: {datetime.now().isoformat()}
**Total Devices**: {len(all_devices)}
**Total Connections**: {len(unique_neighbors)}

---

## Topology Diagram

```mermaid
{diagram}
```

---

## Device List

| Device | Type | Connections |
|--------|------|-------------|
"""
        # Count connections per device
        device_connections = {}
        for conn in unique_neighbors:
            device_connections[conn['local_device']] = device_connections.get(conn['local_device'], 0) + 1
            device_connections[conn['remote_device']] = device_connections.get(conn['remote_device'], 0) + 1

        for device in sorted(all_devices):
            # Determine device type from name
            device_lower = device.lower()
            if 'rtr' in device_lower or 'router' in device_lower:
                device_type = "Router"
            elif 'sw' in device_lower or 'switch' in device_lower:
                device_type = "Switch"
            elif 'fw' in device_lower or 'firewall' in device_lower:
                device_type = "Firewall"
            else:
                device_type = "Unknown"

            conn_count = device_connections.get(device, 0)
            doc += f"| {device} | {device_type} | {conn_count} |\n"

        doc += """
---

## Connection Details

| Local Device | Local Interface | Remote Device | Remote Interface | Platform |
|--------------|-----------------|---------------|------------------|----------|
"""

        for conn in sorted(unique_neighbors, key=lambda x: (x['local_device'], x['local_interface'])):
            doc += f"| {conn['local_device']} | {conn['local_interface']} | "
            doc += f"{conn['remote_device']} | {conn['remote_interface']} | "
            doc += f"{conn.get('platform', 'N/A')} |\n"

        doc += """
---

*Topology auto-generated from CDP/LLDP neighbor data using AI.*
"""

        print("  ✓ Topology documentation complete!")
        return doc


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 13: Network Documentation Basics")
    print("Network Topology Diagrammer Demo")
    print("=" * 60)

    diagrammer = NetworkTopologyDiagrammer()

    # Simulate CDP output from multiple devices
    cdp_data = {
        "router-core-01": """
Device ID: switch-dist-01
  IP address: 10.0.1.2
  Interface: GigabitEthernet0/1,  Port ID (outgoing port): GigabitEthernet1/0/1
  Platform: cisco WS-C3850-24T,  Capabilities: Switch IGMP

Device ID: router-core-02
  IP address: 10.0.0.2
  Interface: GigabitEthernet0/2,  Port ID (outgoing port): GigabitEthernet0/2
  Platform: cisco ISR4451-X,  Capabilities: Router

Device ID: firewall-01
  IP address: 10.0.0.10
  Interface: GigabitEthernet0/3,  Port ID (outgoing port): Ethernet1/1
  Platform: Palo Alto Networks PA-5250,  Capabilities: Router
        """,
        "switch-dist-01": """
Device ID: router-core-01
  IP address: 10.0.1.1
  Interface: GigabitEthernet1/0/1,  Port ID (outgoing port): GigabitEthernet0/1
  Platform: cisco ISR4451-X,  Capabilities: Router

Device ID: switch-access-01
  IP address: 10.0.2.1
  Interface: GigabitEthernet1/0/10,  Port ID (outgoing port): GigabitEthernet0/1
  Platform: cisco WS-C2960X-24TS-L,  Capabilities: Switch IGMP

Device ID: switch-access-02
  IP address: 10.0.2.2
  Interface: GigabitEthernet1/0/11,  Port ID (outgoing port): GigabitEthernet0/1
  Platform: cisco WS-C2960X-24TS-L,  Capabilities: Switch IGMP
        """
    }

    # Generate topology documentation
    topology_doc = diagrammer.create_topology_documentation(cdp_data)

    print("\n" + "=" * 60)
    print("TOPOLOGY DOCUMENTATION:")
    print("=" * 60)
    print(topology_doc)

    # Save to file
    with open("network-topology.md", "w") as f:
        f.write(topology_doc)

    print("\n✓ Topology documentation saved to network-topology.md")
