"""
Chapter 13: Network Documentation Basics
Network Topology Diagrammer

Generate network diagrams from CDP/LLDP data using AI.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

from anthropic import Anthropic
from typing import List, Dict
from datetime import datetime
import json


class NetworkTopologyDiagrammer:
    """Generate network diagrams from configs and CDP/LLDP data."""

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def extract_neighbors_from_cdp(self, cdp_output: str) -> List[Dict]:
        """Extract neighbor information from CDP/LLDP output."""

        prompt = f"""Extract neighbor information from this CDP/LLDP output.

Output:
{cdp_output}

Return JSON array of neighbors:
[
    {{
        "local_device": "this device name",
        "local_interface": "interface on this device",
        "remote_device": "neighbor device name",
        "remote_interface": "interface on neighbor",
        "platform": "device platform/model"
    }}
]

JSON:"""

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku for cost efficiency
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
        """Generate Mermaid diagram syntax from topology data."""

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
1. All devices as nodes
2. All connections between them
3. Interface labels on links
4. Use appropriate shapes (rectangle for routers, cylinder for switches, etc.)

Mermaid syntax:"""

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
        """Create complete topology documentation with diagram."""

        print("Analyzing topology...")

        # Extract all neighbors
        all_neighbors = []
        all_devices = set()

        for device_name, cdp_output in cdp_outputs.items():
            neighbors = self.extract_neighbors_from_cdp(cdp_output)
            all_neighbors.extend(neighbors)
            all_devices.add(device_name)
            for n in neighbors:
                all_devices.add(n['remote_device'])

        print(f"Found {len(all_devices)} devices, {len(all_neighbors)} connections")

        # Generate diagram
        diagram = self.generate_mermaid_diagram(
            devices=list(all_devices),
            connections=all_neighbors
        )

        # Create documentation
        doc = f"""# Network Topology Documentation

**Generated**: {datetime.now().isoformat()}
**Total Devices**: {len(all_devices)}
**Total Connections**: {len(all_neighbors)}

---

## Topology Diagram

```mermaid
{diagram}
```

---

## Connection Details

"""

        # Add connection table
        doc += "| Local Device | Local Interface | Remote Device | Remote Interface | Platform |\n"
        doc += "|--------------|-----------------|---------------|------------------|----------|\n"

        for conn in all_neighbors:
            doc += f"| {conn['local_device']} | {conn['local_interface']} | "
            doc += f"{conn['remote_device']} | {conn['remote_interface']} | "
            doc += f"{conn.get('platform', 'N/A')} |\n"

        doc += "\n---\n\n"
        doc += "*Topology auto-generated from CDP/LLDP neighbor data.*\n"

        return doc


# Example usage
if __name__ == "__main__":
    diagrammer = NetworkTopologyDiagrammer()

    # Simulate CDP output from multiple devices
    cdp_data = {
        "router-core-01": """
Device ID: switch-dist-01
Interface: GigabitEthernet0/1, Port ID: GigabitEthernet1/0/1
Platform: cisco WS-C3850

Device ID: router-core-02
Interface: GigabitEthernet0/2, Port ID: GigabitEthernet0/2
Platform: Cisco 4451-X
        """,
        "switch-dist-01": """
Device ID: router-core-01
Interface: GigabitEthernet1/0/1, Port ID: GigabitEthernet0/1
Platform: Cisco 4451-X

Device ID: switch-access-01
Interface: GigabitEthernet1/0/10, Port ID: GigabitEthernet1/0/1
Platform: cisco WS-C2960X
        """
    }

    topology_doc = diagrammer.create_topology_documentation(cdp_data)

    print(topology_doc)

    # Save to file
    with open("network-topology.md", "w") as f:
        f.write(topology_doc)

    print("\n✓ Topology documentation saved to network-topology.md")
