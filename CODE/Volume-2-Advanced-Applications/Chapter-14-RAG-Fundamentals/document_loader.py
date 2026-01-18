#!/usr/bin/env python3
"""
Document Loader for Network Documentation

Simple utilities to load network docs from various formats.

From: AI for Networking Engineers - Volume 2, Chapter 14
Author: Eduard Dulharu

Usage:
    from document_loader import load_docs_from_directory
    docs = load_docs_from_directory("./network_docs")
"""

import os
from pathlib import Path
from typing import List, Dict


def load_text_file(file_path: str) -> Dict[str, str]:
    """
    Load a text file.

    Args:
        file_path: Path to text file

    Returns:
        Dict with 'content' and 'metadata'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return {
        "content": content,
        "metadata": {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "type": "text"
        }
    }


def load_markdown_file(file_path: str) -> Dict[str, str]:
    """
    Load a markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        Dict with 'content' and 'metadata'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract title from first h1 if present
    title = None
    for line in content.split('\n'):
        if line.startswith('# '):
            title = line.replace('# ', '').strip()
            break

    return {
        "content": content,
        "metadata": {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "title": title,
            "type": "markdown"
        }
    }


def load_config_file(file_path: str) -> Dict[str, str]:
    """
    Load a network config file.

    Args:
        file_path: Path to config file (.cfg, .conf, etc.)

    Returns:
        Dict with 'content' and 'metadata'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try to extract hostname
    hostname = None
    for line in content.split('\n'):
        if line.strip().startswith('hostname '):
            hostname = line.strip().replace('hostname ', '')
            break

    return {
        "content": content,
        "metadata": {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "hostname": hostname,
            "type": "config"
        }
    }


def load_docs_from_directory(directory: str, extensions: List[str] = None) -> List[Dict]:
    """
    Load all documents from a directory.

    Args:
        directory: Path to directory
        extensions: List of file extensions to load (default: ['.txt', '.md', '.cfg', '.conf'])

    Returns:
        List of document dicts
    """
    if extensions is None:
        extensions = ['.txt', '.md', '.cfg', '.conf']

    docs = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return docs

    # Find all matching files
    for ext in extensions:
        for file_path in dir_path.rglob(f'*{ext}'):
            try:
                if ext in ['.txt']:
                    doc = load_text_file(str(file_path))
                elif ext in ['.md']:
                    doc = load_markdown_file(str(file_path))
                elif ext in ['.cfg', '.conf']:
                    doc = load_config_file(str(file_path))
                else:
                    continue

                docs.append(doc)
                print(f"✓ Loaded: {file_path.name}")

            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")

    print(f"\nTotal: {len(docs)} documents loaded")
    return docs


def create_sample_docs(output_dir: str = "./sample_network_docs"):
    """
    Create sample network documentation files for testing.

    Args:
        output_dir: Directory to create sample files in
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample 1: VLAN Guide
    with open(f"{output_dir}/vlan_guide.md", 'w') as f:
        f.write("""# VLAN Configuration Guide

## Overview
VLANs (Virtual LANs) segment network traffic at Layer 2.

## VLAN Numbering Scheme
- VLAN 1: Management (default)
- VLAN 10-99: User networks
- VLAN 100-199: Voice networks
- VLAN 200-299: Server networks
- VLAN 999: Quarantine

## Configuration Example
```
vlan 10
  name USERS
vlan 20
  name SERVERS
```

## Best Practices
1. Don't use VLAN 1 for user traffic
2. Document all VLANs in IPAM
3. Use meaningful names
4. Implement VLAN pruning on trunks
""")

    # Sample 2: BGP Standards
    with open(f"{output_dir}/bgp_standards.md", 'w') as f:
        f.write("""# BGP Configuration Standards

## AS Numbers
- Primary AS: 65001
- Backup AS: 65002

## Peering Requirements
1. MD5 authentication required
2. Prefix-list filtering
3. Maximum prefix limits (1000 for peers)
4. BFD for fast convergence

## Configuration Template
```
router bgp 65001
  neighbor 203.0.113.1 remote-as 174
  neighbor 203.0.113.1 password <MD5>
  neighbor 203.0.113.1 maximum-prefix 1000
```
""")

    # Sample 3: Router Config
    with open(f"{output_dir}/router_config.cfg", 'w') as f:
        f.write("""hostname CORE-RTR-01

interface GigabitEthernet0/0
 description WAN_UPLINK
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description INTERNAL
 ip address 10.0.0.1 255.255.255.0
 no shutdown

router ospf 1
 router-id 10.0.0.1
 network 10.0.0.0 0.0.0.255 area 0

router bgp 65001
 neighbor 203.0.113.2 remote-as 174
""")

    print(f"✓ Created sample docs in {output_dir}/")


# Demo
if __name__ == "__main__":
    print("="*60)
    print("Document Loader Demo")
    print("="*60)

    # Create sample documents
    print("\n1. Creating sample network docs...")
    create_sample_docs()

    # Load documents
    print("\n2. Loading documents...")
    docs = load_docs_from_directory("./sample_network_docs")

    # Display loaded docs
    print("\n3. Loaded Documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  Document {i}:")
        print(f"    Filename: {doc['metadata']['filename']}")
        print(f"    Type: {doc['metadata']['type']}")
        print(f"    Content length: {len(doc['content'])} characters")
        if 'title' in doc['metadata'] and doc['metadata']['title']:
            print(f"    Title: {doc['metadata']['title']}")
        if 'hostname' in doc['metadata'] and doc['metadata']['hostname']:
            print(f"    Hostname: {doc['metadata']['hostname']}")

    # Cleanup instructions
    print("\n" + "="*60)
    print("Demo complete!")
    print("\nTo clean up: rm -rf ./sample_network_docs")
