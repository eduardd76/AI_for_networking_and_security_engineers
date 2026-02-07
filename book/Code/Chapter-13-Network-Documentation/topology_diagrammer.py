"""
Chapter 13: Network Documentation Basics
Network Topology Diagrammer

Generate network topology diagrams from CDP/LLDP neighbor data using Claude AI.

How it works:
    1. Collect CDP/LLDP output from your devices (e.g., "show cdp neighbors detail")
    2. Claude parses the raw CLI output into structured neighbor data
    3. From that data, it generates a Mermaid diagram you can render in
       GitHub, Notion, Confluence, or any Markdown viewer

Think of it as: "show cdp neighbors" across your entire network, automatically
turned into a visual topology map.

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

# Use Haiku for simple extraction (cheap + fast), Sonnet for diagram generation
EXTRACTION_MODEL = "claude-haiku-4-20250514"
DIAGRAM_MODEL = "claude-sonnet-4-20250514"


class NetworkTopologyDiagrammer:
    """Generate network diagrams from CDP/LLDP neighbor data.

    Networking analogy: This automates what you'd normally do by SSHing
    into every device, running "show cdp neighbors detail", copying the
    output into a Visio diagram, and drawing lines between boxes.
    Except this does it in seconds.
    """

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def _parse_json_response(self, text: str) -> list:
        """Safely extract JSON array from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fences
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON array boundaries
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")

    def extract_neighbors_from_cdp(self, cdp_output: str) -> List[Dict]:
        """Extract neighbor information from CDP/LLDP output.

        Takes raw "show cdp neighbors detail" or "show lldp neighbors detail"
        output and returns structured data.

        Args:
            cdp_output: Raw CLI output from the device

        Returns:
            List of neighbor dictionaries with keys: local_device,
            local_interface, remote_device, remote_interface, platform
        """

        prompt = f"""Extract neighbor information from this CDP/LLDP output.

Output:
{cdp_output}

Return a JSON array of neighbor entries. Each entry must have these exact keys:
[
    {{
        "local_device": "this device name",
        "local_interface": "interface on this device",
        "remote_device": "neighbor device name",
        "remote_interface": "interface on neighbor",
        "platform": "device platform/model"
    }}
]

Return ONLY the JSON array, no other text."""

        response = self.client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_json_response(response.content[0].text)

    def generate_mermaid_diagram(
        self,
        devices: List[str],
        connections: List[Dict]
    ) -> str:
        """Generate Mermaid diagram syntax from topology data.

        Mermaid is a text-based diagramming language that renders in
        GitHub README files, Confluence, Notion, and many other tools.
        No Visio license needed.

        Args:
            devices: List of all device hostnames
            connections: List of connection dictionaries from extract_neighbors_from_cdp

        Returns:
            Mermaid diagram syntax string
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

Generate Mermaid flowchart syntax following these rules:
1. All devices as named nodes
2. Connections as bidirectional links with interface labels
3. Use shapes to indicate device roles:
   - Routers: rectangle [router-name]
   - Switches: stadium shape ([switch-name])
   - Firewalls: hexagon {{{{firewall-name}}}}
4. Use subgraphs to group devices by site/layer if the naming suggests it
5. Start the diagram with "graph TB" (top to bottom layout)

Return ONLY the Mermaid syntax starting with "graph TB", no code fences."""

        response = self.client.messages.create(
            model=DIAGRAM_MODEL,
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Strip any code fences the model may add despite instructions
        text = response.content[0].text.strip()
        text = re.sub(r'^```(?:mermaid)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return text

    def create_topology_documentation(
        self,
        cdp_outputs: Dict[str, str]
    ) -> str:
        """Create complete topology documentation with diagram.

        This is the main entry point. Pass in CDP output from multiple
        devices and get back a complete Markdown document with a
        rendered diagram and connection table.

        Args:
            cdp_outputs: Dict mapping device hostname to its CDP output
                         e.g., {"router-01": "show cdp neighbors detail output..."}

        Returns:
            Complete Markdown documentation string
        """

        print("Analyzing topology from CDP/LLDP data...")

        # Extract all neighbors from every device's CDP output
        all_neighbors = []
        all_devices = set()

        for device_name, cdp_output in cdp_outputs.items():
            try:
                neighbors = self.extract_neighbors_from_cdp(cdp_output)
                all_neighbors.extend(neighbors)
                all_devices.add(device_name)
                for n in neighbors:
                    all_devices.add(n['remote_device'])
                print(f"  {device_name}: {len(neighbors)} neighbors found")
            except Exception as e:
                print(f"  {device_name}: ERROR parsing CDP - {e}")

        print(f"  Total: {len(all_devices)} devices, {len(all_neighbors)} links")

        # Deduplicate bidirectional links (A->B and B->A are the same link)
        seen_links = set()
        unique_neighbors = []
        for conn in all_neighbors:
            link_key = tuple(sorted([
                f"{conn['local_device']}:{conn['local_interface']}",
                f"{conn['remote_device']}:{conn['remote_interface']}"
            ]))
            if link_key not in seen_links:
                seen_links.add(link_key)
                unique_neighbors.append(conn)

        print(f"  Unique links (after dedup): {len(unique_neighbors)}")

        # Generate diagram
        diagram = self.generate_mermaid_diagram(
            devices=list(all_devices),
            connections=unique_neighbors
        )

        # Create documentation
        doc = f"""# Network Topology Documentation

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Devices**: {len(all_devices)}
**Total Links**: {len(unique_neighbors)}
**Source**: CDP/LLDP neighbor discovery

---

## Topology Diagram

```mermaid
{diagram}
```

> To render this diagram: paste into GitHub, Notion, or any Mermaid-compatible
> Markdown viewer. You can also use https://mermaid.live to view it online.

---

## Connection Details

"""

        # Add connection table
        doc += "| Local Device | Local Interface | Remote Device | Remote Interface | Platform |\n"
        doc += "|:-------------|:----------------|:--------------|:-----------------|:---------|\n"

        for conn in unique_neighbors:
            doc += f"| {conn['local_device']} | {conn['local_interface']} | "
            doc += f"{conn['remote_device']} | {conn['remote_interface']} | "
            doc += f"{conn.get('platform', 'N/A')} |\n"

        doc += "\n---\n\n"
        doc += "*Topology auto-generated from CDP/LLDP neighbor data using Claude AI.*\n"

        return doc


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    diagrammer = NetworkTopologyDiagrammer()

    # Simulate "show cdp neighbors detail" output from multiple devices.
    # In production, you'd collect this via Netmiko/NAPALM/Ansible.
    cdp_data = {
        "router-core-01": """
Device ID: switch-dist-01.example.com
  IP address: 10.0.1.2
  Platform: cisco WS-C3850-24T, Capabilities: Switch IGMP
  Interface: GigabitEthernet0/1, Port ID (outgoing port): GigabitEthernet1/0/1
  Holdtime: 160 sec

Device ID: router-core-02.example.com
  IP address: 10.0.2.1
  Platform: Cisco ISR4451-X/K9, Capabilities: Router Switch IGMP
  Interface: GigabitEthernet0/2, Port ID (outgoing port): GigabitEthernet0/2
  Holdtime: 145 sec

Device ID: firewall-01.example.com
  IP address: 10.0.3.1
  Platform: Palo Alto Networks PA-5250, Capabilities: Router
  Interface: GigabitEthernet0/3, Port ID (outgoing port): ethernet1/1
  Holdtime: 120 sec
        """,
        "switch-dist-01": """
Device ID: router-core-01.example.com
  IP address: 10.0.1.1
  Platform: Cisco ISR4451-X/K9, Capabilities: Router Switch IGMP
  Interface: GigabitEthernet1/0/1, Port ID (outgoing port): GigabitEthernet0/1
  Holdtime: 155 sec

Device ID: switch-access-01.example.com
  IP address: 10.0.10.1
  Platform: cisco WS-C2960X-24TS-L, Capabilities: Switch IGMP
  Interface: GigabitEthernet1/0/10, Port ID (outgoing port): GigabitEthernet0/1
  Holdtime: 170 sec

Device ID: switch-access-02.example.com
  IP address: 10.0.10.2
  Platform: cisco WS-C2960X-24TS-L, Capabilities: Switch IGMP
  Interface: GigabitEthernet1/0/11, Port ID (outgoing port): GigabitEthernet0/1
  Holdtime: 165 sec
        """
    }

    topology_doc = diagrammer.create_topology_documentation(cdp_data)

    print(topology_doc)

    # Save to file
    with open("network-topology.md", "w") as f:
        f.write(topology_doc)

    print("\nTopology documentation saved to network-topology.md")
