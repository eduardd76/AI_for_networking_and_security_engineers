# Chapter 13: Network Documentation Basics

## Why This Chapter Matters

Your network has 500 devices. Documentation exists in:
- Word docs on SharePoint (last updated 2019)
- Wiki pages (conflicting information)
- Tribal knowledge (in Bob's head, and Bob just left)
- Comments in configs (sometimes)
- Nowhere (most of the time)

A new engineer asks: "What's the BGP policy for AWS?" Nobody knows. It takes 3 days to figure out.

**Cost**: New engineer productivity delay, risk of misconfiguration, slow incident response.

**Solution**: AI-generated, always-current documentation that updates automatically from your network.

This chapter shows you how to:
- Auto-generate documentation from configs
- Create network diagrams with AI
- Build searchable knowledge bases
- Keep docs synchronized with reality

**The goal**: Documentation that's actually useful because it's always accurate.

---

## Section 1: The Documentation Problem

### Why Documentation Fails

**Problem 1: Manual Updates**
```
1. Engineer makes config change
2. Engineer forgets to update docs
3. Docs are now wrong
4. Next engineer follows wrong docs
5. Incident occurs
```

**Problem 2: Multiple Sources of Truth**
```
SharePoint: "VLAN 10 is for servers"
Wiki: "VLAN 10 is for guest WiFi"
Config: "VLAN 10 description QUARANTINE"
Reality: Who knows?
```

**Problem 3: Format Inconsistency**
```
Device A doc: Markdown, detailed, current
Device B doc: Word, minimal, 2 years old
Device C doc: Email thread from 2017
Device D doc: Doesn't exist
```

### What Good Documentation Looks Like

**Characteristics**:
- **Accurate**: Matches current network state
- **Complete**: Covers all devices, not just some
- **Searchable**: Find answers in seconds, not hours
- **Versioned**: Track changes over time
- **Accessible**: One place, not scattered
- **Maintainable**: Updates automatically, not manually

**The AI Advantage**: Generate docs from configs, not from memory.

---

## Section 2: Auto-Generating Documentation from Configs

### Extract Structured Data

```python
# doc_generator.py
from anthropic import Anthropic
from typing import Dict, List
import json
from datetime import datetime

class ConfigDocumentationGenerator:
    """Generate documentation automatically from network configs."""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

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
    generator = ConfigDocumentationGenerator(api_key="your-api-key")

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
```

**Output**: `router-core-01-doc.md` with complete, formatted documentation.

---

## Section 3: Creating Network Diagrams with AI

### Topology Discovery

```python
# topology_diagrammer.py
from anthropic import Anthropic
from typing import List, Dict
import re

class NetworkTopologyDiagrammer:
    """Generate network diagrams from configs and CDP/LLDP data."""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

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
            model="claude-3-5-haiku-20241022",  # Haiku for cost efficiency
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
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
            model="claude-3-5-sonnet-20241022",
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
    diagrammer = NetworkTopologyDiagrammer(api_key="your-api-key")

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
```

**Output**: Network diagram in Mermaid format that renders in Markdown viewers, GitHub, etc.

---

## Section 4: Building a Documentation Pipeline

### Automated Documentation Workflow

```python
# documentation_pipeline.py
from pathlib import Path
from typing import Dict, List
import schedule
import time
from datetime import datetime
import git

class DocumentationPipeline:
    """Automated pipeline for network documentation."""

    def __init__(
        self,
        api_key: str,
        config_dir: str,
        output_dir: str,
        git_repo: str = None
    ):
        self.generator = ConfigDocumentationGenerator(api_key)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.git_repo = git_repo

        if git_repo:
            self.repo = git.Repo(git_repo)

    def fetch_configs(self) -> Dict[str, str]:
        """Fetch current configs from devices or config management system."""
        # In production: use Netmiko, NAPALM, or pull from Git

        configs = {}
        for config_file in self.config_dir.glob("*.cfg"):
            hostname = config_file.stem
            with open(config_file, 'r') as f:
                configs[hostname] = f.read()

        return configs

    def generate_all_documentation(self):
        """Generate documentation for all devices."""

        print(f"\n{'='*60}")
        print(f"Documentation Generation Started: {datetime.now()}")
        print(f"{'='*60}\n")

        configs = self.fetch_configs()
        print(f"Found {len(configs)} device configs")

        generated_files = []

        for hostname, config in configs.items():
            try:
                output_file = self.output_dir / f"{hostname}.md"

                self.generator.generate_complete_documentation(
                    config=config,
                    hostname=hostname,
                    output_file=str(output_file)
                )

                generated_files.append(output_file)
                print(f"  ✓ {hostname}")

            except Exception as e:
                print(f"  ✗ {hostname}: {e}")

        # Generate index
        self.generate_index(generated_files)

        # Commit to Git if configured
        if self.git_repo:
            self.commit_changes()

        print(f"\n{'='*60}")
        print(f"Documentation Generation Complete")
        print(f"Total files: {len(generated_files)}")
        print(f"{'='*60}\n")

    def generate_index(self, doc_files: List[Path]):
        """Generate index page linking to all device docs."""

        index_content = f"""# Network Documentation Index

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Devices**: {len(doc_files)}

---

## Device Documentation

"""

        for doc_file in sorted(doc_files):
            hostname = doc_file.stem
            index_content += f"- [{hostname}]({doc_file.name})\n"

        index_content += """

---

## Documentation Standards

This documentation is automatically generated from device configurations.

**Update Frequency**: Daily at 2:00 AM
**Source**: Device running configurations
**Generator**: AI-powered documentation pipeline

To update manually: `python documentation_pipeline.py --generate-now`

---

*Auto-generated network documentation*
"""

        index_file = self.output_dir / "index.md"
        with open(index_file, 'w') as f:
            f.write(index_content)

        print(f"  ✓ Generated index: {index_file}")

    def commit_changes(self):
        """Commit documentation updates to Git."""

        try:
            self.repo.git.add(all=True)

            if self.repo.is_dirty():
                commit_message = f"Auto-update network documentation - {datetime.now()}"
                self.repo.index.commit(commit_message)

                # Push to remote (optional)
                # self.repo.remote(name='origin').push()

                print(f"  ✓ Changes committed to Git")
            else:
                print(f"  → No changes to commit")

        except Exception as e:
            print(f"  ✗ Git commit failed: {e}")

    def schedule_daily_updates(self):
        """Schedule automatic daily documentation updates."""

        # Run daily at 2 AM
        schedule.every().day.at("02:00").do(self.generate_all_documentation)

        print("Documentation pipeline scheduled (daily at 2:00 AM)")
        print("Press Ctrl+C to stop")

        while True:
            schedule.run_pending()
            time.sleep(60)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Network Documentation Pipeline")
    parser.add_argument("--generate-now", action="store_true",
                       help="Generate documentation immediately")
    parser.add_argument("--schedule", action="store_true",
                       help="Run scheduled updates")
    parser.add_argument("--config-dir", default="./configs",
                       help="Directory containing device configs")
    parser.add_argument("--output-dir", default="./docs",
                       help="Output directory for documentation")
    parser.add_argument("--git-repo", default=None,
                       help="Git repository path for versioning")

    args = parser.parse_args()

    pipeline = DocumentationPipeline(
        api_key="your-api-key",
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        git_repo=args.git_repo
    )

    if args.generate_now:
        pipeline.generate_all_documentation()

    elif args.schedule:
        pipeline.schedule_daily_updates()

    else:
        print("Use --generate-now or --schedule")
```

**Usage**:
```bash
# Generate documentation now
python documentation_pipeline.py --generate-now

# Schedule daily updates
python documentation_pipeline.py --schedule

# With Git versioning
python documentation_pipeline.py --generate-now --git-repo /path/to/docs/repo
```

---

## Section 5: Living Documentation Pattern

### Keep Docs in Sync with Reality

**The Problem**: Docs get out of date the moment they're written.

**The Solution**: Generate docs on every config change.

**Integration Points**:
1. **CI/CD Pipeline**: Generate docs when configs committed to Git
2. **Config Management**: Hook into Ansible/Terraform apply
3. **Network Automation**: Generate after device changes
4. **Scheduled**: Daily automated generation

### CI/CD Integration Example

```yaml
# .github/workflows/generate-docs.yml
name: Generate Network Documentation

on:
  push:
    paths:
      - 'configs/**'
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install anthropic gitpython

      - name: Generate documentation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python documentation_pipeline.py --generate-now

      - name: Commit and push changes
        run: |
          git config --local user.email "docs-bot@company.com"
          git config --local user.name "Documentation Bot"
          git add docs/
          git diff --quiet && git diff --staged --quiet || \
            git commit -m "Auto-update network documentation [skip ci]"
          git push
```

**Result**: Every config change triggers automatic documentation update.

---

## What Can Go Wrong

**1. Stale configs**
- Pipeline reads old configs
- Documentation doesn't match reality
- Solution: Fetch live configs from devices

**2. API costs for large networks**
- 500 devices × $0.05/doc = $25 per run
- Daily runs = $750/month
- Solution: Generate only changed devices, use cheaper models for simple extraction

**3. Git conflicts**
- Multiple updates at once
- Merge conflicts in documentation
- Solution: Single automated process, not multiple sources

**4. Incomplete extraction**
- AI misses some config sections
- Documentation has gaps
- Solution: Template validation, human review of first generation

**5. Sensitive data exposure**
- Docs include passwords, keys
- Published to public repo
- Solution: Sanitize configs before documentation (Chapter 12 patterns)

---

## Key Takeaways

1. **Auto-generate docs from configs** - Always accurate because it's generated from source of truth
2. **Create visual topology diagrams** - Mermaid diagrams from CDP/LLDP data
3. **Build documentation pipelines** - Automatic generation on schedule or trigger
4. **Version in Git** - Track changes, review history
5. **Living documentation** - Stays current through automation, not manual updates

Documentation doesn't have to be a burden. Make the computer do it.

Next chapter: RAG Fundamentals - Making this documentation searchable with AI.
