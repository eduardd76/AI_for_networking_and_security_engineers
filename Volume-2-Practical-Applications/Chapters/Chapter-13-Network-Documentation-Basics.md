# Chapter 13: Network Documentation Basics

## Introduction

Network documentation is often called the "necessary evil" of IT operations. Everyone knows it's important. Everyone claims they'll maintain it. And yet, in nearly every organization, documentation exists in a fragmented state—scattered across shared drives, wikis, email archives, and the institutional knowledge of departing team members.

This chapter addresses one of the most persistent problems in network operations: **keeping documentation accurate and up-to-date as networks constantly evolve**.

**Networking analogy**: Think of AI-generated documentation like SNMP polling versus manual inventory. You *could* manually track every device's uptime, interface counters, and firmware version in a spreadsheet — or you could let SNMP poll the devices and populate a dashboard automatically. AI documentation does the same thing, but for *everything in the config* — routing protocols, ACLs, interface assignments, QoS policies — all extracted and formatted without human effort.

### The Documentation Crisis

Let's be realistic: traditional network documentation fails for a simple reason—**it requires constant manual maintenance**. Every configuration change means someone must:

1. Make the change to the device
2. Test that it works
3. Update the documentation
4. Get approval from reviewers
5. Publish the changes
6. Notify stakeholders

By step 3, most teams are exhausted. The documentation update either gets postponed indefinitely or done hastily and inaccurately.

### The Problem Statement

Consider this scenario: Your company manages 500 network devices across 12 data centers and 47 branch offices. A new engineer (hired last week) is asked to verify the BGP configuration for your AWS peering connection. They need to know:

- What's the current BGP AS number?
- What neighbors are configured?
- Which networks are advertised?
- Are there route filters applied?
- When was this last changed?

Where do they look?

**Option A**: Search SharePoint for "BGP AWS"
- Finds a document last updated in 2019
- Contains different AS numbers than what's actually running
- Decision: **Not trustworthy**

**Option B**: Check the wiki
- Multiple conflicting entries
- Different people describing the same thing differently
- No indication which is correct
- Decision: **Too confusing**

**Option C**: Ask the network team
- Senior engineer shows the running config
- Explains the current state verbally
- New engineer writes notes by hand
- Later realizes they missed critical details
- Decision: **Error-prone and doesn't scale**

**Option D**: Ask ChatGPT
- It confidently returns completely wrong information
- Junior engineer trusts it anyway
- Causes an outage when applied to production
- Decision: **Dangerous**

**The real answer should be Option E**: Access auto-generated, always-current documentation that was built directly from the actual running configurations.

### Why This Matters

Documentation problems manifest as:

| Impact | Cost |
|--------|------|
| **Onboarding delays** | New engineers spend 3-5 days understanding the network instead of 1 day |
| **Troubleshooting slowness** | Engineers waste hours tracing through configs to understand intent |
| **Risk of misconfiguration** | Without clear documentation, changes are made blindly |
| **Compliance violations** | Auditors find undocumented configurations and flag as non-compliant |
| **Knowledge loss** | When experienced engineer leaves, critical knowledge walks out the door |
| **Incident response delays** | During an outage, time spent understanding the topology compounds the problem |

---

## The Solution: AI-Powered Documentation

Instead of fighting to maintain documentation manually, what if **the documentation was automatically generated from the source of truth—the device configurations themselves**?

This is the premise of this chapter: **Use AI to analyze network device configurations and automatically generate comprehensive, accurate documentation.**

### Why AI is the Right Tool

AI models like Claude excel at tasks that require:
- **Understanding context** - Recognizing that `ip ospf cost 100` has different meaning on a WAN link vs LAN link
- **Synthesis** - Combining scattered configuration blocks into coherent narrative (BGP config spread across 20 lines of CLI into organized documentation)
- **Inference** - Deducing a device's role from its configuration ("This is a core router because it has BGP, OSPF, and connects multiple sites")
- **Multi-vendor compatibility** - Understanding Cisco IOS, NX-OS, Juniper Junos, Arista EOS with the same approach

### The Philosophy

The code in this chapter follows one principle: **Documentation should be generated, not written.**

- **Generated from configs**: Always in sync with running state
- **Generated on schedule**: Daily/hourly, before config changes take effect
- **Generated on change**: Before/after comparisons for auditing
- **Generated programmatically**: No human bottleneck

---

## Real-World Use Case: Enterprise Network Documentation

### The Scenario

TechCorp manages a large enterprise network:
- **500 devices** across 15 locations
- **3 NOC shifts** handling operations 24/7
- **12 network engineers** maintaining the network
- **Quarterly audits** requiring full configuration documentation

**The problem they faced:**
- Documentation took 2 weeks of manual work per quarter
- Different people documented things differently
- Formats were inconsistent (some markdown, some Word docs)
- Auditors requested printed documentation (causing more manual work)
- When urgent changes needed, documentation was never updated

**The solution:**
- Automated documentation generation running daily
- Generated documents stored in Git (version control + history)
- Reports automatically generated for audits (PDF from markdown)
- Change validation: Compare before/after configs
- Onboarding: New engineer sees exactly what's configured, not outdated docs

**The results:**
- Documentation generation takes 2 hours instead of 80 hours
- Quality improved (AI catches things humans miss)
- Compliance ready at any time (not after manual scramble)
- Onboarding time reduced from 5 days to 2 days

---

## Core Concepts

### Multi-Vendor Support

The code in this chapter handles multiple vendors:

```
Cisco IOS      → Standard enterprise routing/switching
Cisco NX-OS    → Data center environments
Cisco ASA      → Firewall configurations
Juniper Junos  → High-end routers and switches
Arista EOS     → Modern cloud networking
Palo Alto      → Next-gen firewalls
Fortinet       → SMB firewalls
```

The generator auto-detects the vendor from config characteristics.

### Four-Step Documentation Pipeline

```
┌─────────────────────────────────────────────────────┐
│ STEP 1: Fetch Configurations                       │
│ Pull running configs from devices (Netmiko, NAPALM) │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ STEP 2: Analyze Configurations                     │
│ Extract device role, interfaces, routing, security │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ STEP 3: Generate Documentation                     │
│ Create markdown docs with tables, diagrams, notes  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ STEP 4: Distribute & Version                       │
│ Commit to Git, publish to wiki, alert stakeholders │
└─────────────────────────────────────────────────────┘
```

---

## Files in This Chapter

| File | Purpose | Size | Key Methods |
|------|---------|------|-------------|
| `doc_generator.py` | Core documentation generation | ~800 lines | `generate_device_overview()`, `generate_complete_documentation()`, device type detection |
| `doc_analyzer.py` | Configuration analysis & validation | ~400 lines | `analyze_security()`, `analyze_best_practices()`, `analyze_redundancy()` |
| `topology_diagrammer.py` | Network diagram generation | ~400 lines | `extract_neighbors_from_cdp()`, `generate_mermaid_diagram()` |
| `documentation_pipeline.py` | Automation orchestration | ~500 lines | `generate_all_documentation()`, batch processing, scheduling |

**Total**: 2,100+ lines of production-ready code

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# API key from Anthropic (https://console.anthropic.com)
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### Install Dependencies

```bash
cd Chapter-13-Network-Documentation-Basics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install anthropic python-dotenv schedule gitpython
```

### Configure

```bash
# Copy example environment file
cp ../../.env.example .env

# Edit .env and add your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-api03-...
```

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

## Part 1: Documentation Generation (`doc_generator.py`)

This module handles the core task of analyzing device configs and generating documentation.

### Core Classes

**ConfigDocumentationGenerator**
- Main class for documentation generation
- Handles multi-vendor device support
- Tracks API usage and costs
- Supports multiple output formats

**Data Classes**
- `DeviceOverview` - High-level device info
- `InterfaceInfo` - Interface details
- `VlanInfo` - VLAN configuration
- `RoutingProtocol` - Routing protocol info

### Auto-Detecting Device Type

The generator automatically identifies device type from configuration patterns:

```python
from doc_generator import ConfigDocumentationGenerator

generator = ConfigDocumentationGenerator()

# Auto-detects from config patterns
device_type = generator.detect_device_type(config_text)
print(device_type)  # DeviceType.CISCO_IOS

hostname = generator.extract_hostname(config_text)
print(hostname)  # "router-core-01"
```

### Extracting Information Without API Calls

For cost optimization, the generator uses regex-based extraction for basic info:

```python
# These don't require API calls (fast, free)
interfaces = generator.extract_interfaces_basic(config)
vlans = generator.extract_vlans_basic(config)
hostname = generator.extract_hostname(config)

for interface in interfaces:
    print(f"{interface.name}: {interface.ip_address or 'No IP'}")
```

## Section 2: Auto-Generating Documentation from Configs

### Extract Structured Data

```python
# doc_generator.py
import os
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
            model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
    generator = ConfigDocumentationGenerator(api_key=os.environ["ANTHROPIC_API_KEY"])

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
            model="claude-haiku-4-20250514",  # Haiku for cost efficiency
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
    from datetime import datetime
    
    diagrammer = NetworkTopologyDiagrammer(api_key=os.environ["ANTHROPIC_API_KEY"])

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
        api_key=os.environ["ANTHROPIC_API_KEY"],
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

## Advanced Usage Patterns

### Pattern 1: Pre/Post Change Documentation

```bash
# Before change
python doc_generator.py --config before.cfg --output before-doc.md

# Make change to device
ssh router make-change

# After change
python doc_generator.py --config after.cfg --output after-doc.md

# Diff to see what changed
diff before-doc.md after-doc.md
```

### Pattern 2: Documentation as Validation

Use the analyzer to ensure changes don't introduce security issues:

```python
analyzer = ConfigAnalyzer()
report = analyzer.analyze_config(new_config)

if any(f.severity == Severity.CRITICAL for f in report['findings']):
    print("❌ Critical findings - blocking deployment")
    exit(1)

print("✓ Config passed validation - safe to deploy")
```

### Pattern 3: Documentation API

Generate docs via REST API:

```python
from fastapi import FastAPI
from doc_generator import ConfigDocumentationGenerator

app = FastAPI()
generator = ConfigDocumentationGenerator()

@app.post("/generate-docs")
async def generate_docs(hostname: str, config: str):
    doc = generator.generate_complete_documentation(config, hostname)
    return {"documentation": doc}
```

### Pattern 4: Multi-Format Export

```python
from doc_generator import OutputFormat

# Generate in all formats
doc_md = generator.generate_complete_documentation(config, format=OutputFormat.MARKDOWN)
doc_html = generator.generate_complete_documentation(config, format=OutputFormat.HTML)
doc_json = generator.generate_complete_documentation(config, format=OutputFormat.JSON)
```

---

## Production Deployment Guide

### Prerequisites Checklist

- [ ] API key stored in secrets manager (not .env)
- [ ] Configs accessible (Netmiko/NAPALM configured)
- [ ] Output directory has Git repo initialized
- [ ] Email alerts configured for errors
- [ ] Monitoring in place for API usage

### Step 1: Test Locally

```bash
# Generate docs for single device
python doc_generator.py

# Review output
cat router-core-01-doc.md

# Check for issues
python doc_analyzer.py
```

### Step 2: Schedule Batch Run

```bash
# Create cron job for daily generation
crontab -e
# Add: 0 2 * * * cd /path && python documentation_pipeline.py --generate-now
```

### Step 3: Monitor & Alert

```python
# Monitor API costs
stats = generator.get_usage_stats()
if stats['estimated_cost_usd'] > BUDGET:
    send_alert("Documentation generation exceeding budget")

# Monitor for errors
try:
    pipeline.generate_all_documentation()
except Exception as e:
    send_slack_alert(f"Documentation generation failed: {e}")
```

### Step 4: Archive & Backup

```bash
# Backup documentation Git repo weekly
git push backup master
```

---

## Troubleshooting

### "Config too large for context"
**Symptom**: Error about exceeding token limits
**Solution**: Chunk large configs or use models with larger context (Opus: 200K tokens)

### "Rate limit exceeded"
**Symptom**: API returns 429 error
**Solution**: Add delays between devices or use rate limiter:
```python
from time import sleep
for device in devices:
    generate_doc(device)
    sleep(1)  # 1 second between calls
```

### "Incomplete extraction"
**Symptom**: Some configuration sections missing from docs
**Solution**: Review AI prompt quality, add specific extraction instructions, use two-pass extraction

### "Sensitive data in docs"
**Symptom**: Passwords appearing in documentation
**Solution**: Sanitize configs before processing (remove passwords, remove secrets)

---

## Exercises for Practice

### Exercise 1: Basic Documentation Generation
Generate documentation for a provided router config file. Compare with manual documentation to identify differences.

### Exercise 2: Configuration Analysis
Run the analyzer on a config and fix all CRITICAL findings. Re-run to verify.

### Exercise 3: Multi-Device Pipeline
Set up the documentation pipeline for 5 test device configs. Observe change detection when you modify one.

### Exercise 4: API Integration
Create a simple REST API using FastAPI that accepts a config and returns documentation.

### Exercise 5: Production Deployment
Set up scheduled documentation generation with Git versioning for a simulated network environment.

---

## Best Practices Summary

✅ **Automate everything** - Schedule daily generation
✅ **Version control** - Keep documentation in Git
✅ **Validate before deploy** - Use analyzer to check for issues
✅ **Document WHY not WHAT** - Add context to auto-generated docs
✅ **Cache and optimize** - Only regenerate changed devices
✅ **Monitor costs** - Track API usage
✅ **Alert on errors** - Know when generation fails
✅ **Review periodically** - Ensure accuracy
✅ **Involve the team** - Get feedback on docs

---

## What Can Go Wrong

| Scenario | Risk | Mitigation |
|----------|------|-----------|
| Stale configs | Docs don't match reality | Fetch live configs from devices every time |
| API rate limits | Generation fails | Use rate limiting, lower frequency, upgrade API tier |
| Sensitive data exposure | Security breach | Sanitize all configs before sending to AI |
| Cost explosion | Unexpected API bills | Monitor usage, set budget alerts, cache results |
| Git conflicts | Multiple update sources | Single automated process only |
| Incomplete extraction | Missing documentation | Validate first run, add prompt improvements |
| Accuracy issues | Trust in docs decreases | Include confidence scores, require human review |

---

## Chapter Summary

This chapter showed how to solve a critical network operations problem: **keeping documentation accurate and current**.

### Key Takeaways

1. **Auto-generate from source of truth** - Configs, not memory
2. **Use AI for synthesis** - It handles complexity humans miss
3. **Build pipelines, not scripts** - Automation at scale
4. **Validate everything** - Analyzer catches issues before they hurt
5. **Version and track** - Git provides history and accountability
6. **Monitor and optimize** - Know your costs
7. **Make it production-ready** - This isn't a one-off tool

### Costs vs Benefits

| Aspect | Cost | Benefit |
|--------|------|---------|
| **API calls** | $50-200/month | Documentation always current |
| **Implementation** | 4-8 hours | Saves 40+ hours of manual work/quarter |
| **Maintenance** | Minimal | Scales to 1000s of devices |
| **Learning curve** | 2-4 hours | Empowers entire team |

---

## Next Chapter

**Chapter 14: RAG Fundamentals** - Make this documentation searchable with AI.

Your auto-generated documentation is only useful if people can find answers in it. Chapter 14 shows how to build a Retrieval-Augmented Generation (RAG) system that lets anyone ask questions in plain English and get answers from your documentation.

**Preview**: "What's the OSPF configuration on the backup core router?"
```
>> RAG system searches documentation
>> Finds relevant sections
>> Claude synthesizes the answer
<< "The backup core router (router-core-02) runs OSPF process 1 with..."
```

---

## Resources

### Code Files
- `doc_generator.py` - 800 lines, documentation generation
- `doc_analyzer.py` - 400 lines, configuration analysis
- `topology_diagrammer.py` - 400 lines, network diagram generation
- `documentation_pipeline.py` - 500 lines, automation orchestration

### External References
- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [Mermaid Diagram Documentation](https://mermaid.js.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Netmiko - Device Automation](https://github.com/ktbyers/netmiko)
- [NAPALM - Multi-vendor Automation](https://napalm.readthedocs.io/)

### Related Topics
- Configuration management (Ansible, Terraform)
- Network monitoring (Prometheus, Grafana)
- Compliance automation
- Change management
- Knowledge management

---

**Chapter 13 Complete** ✓

*Generated documentation beats manual documentation every single time.*
