# Chapter 13: Network Documentation Basics

**Auto-generate always-accurate documentation from network configs**

## The Problem

Documentation scattered across:
- SharePoint (last updated 2019)
- Wiki pages (conflicting information)
- Tribal knowledge (gone when Bob leaves)
- Nowhere (most devices)

## The Solution

Generate documentation directly from configs - always accurate because it's from the source of truth.

## Files

| File | Description |
|------|-------------|
| `doc_generator.py` | Generate device documentation from configs |
| `topology_diagrammer.py` | Create network diagrams from CDP/LLDP |
| `documentation_pipeline.py` | Automated pipeline for continuous doc generation |

## Quick Start

```bash
# Install dependencies
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key"

# Generate docs for a single device
python doc_generator.py

# Generate topology diagrams
python topology_diagrammer.py

# Run the full pipeline
python documentation_pipeline.py --generate-now --config-dir ./configs
```

## Usage Examples

### Generate Device Documentation

```python
from doc_generator import ConfigDocumentationGenerator

generator = ConfigDocumentationGenerator()

config = """
hostname my-router
interface GigabitEthernet0/0
 ip address 10.0.0.1 255.255.255.0
router ospf 1
 network 10.0.0.0 0.0.255.255 area 0
"""

doc = generator.generate_complete_documentation(
    config=config,
    hostname="my-router",
    output_file="my-router.md"
)
```

### Create Topology Diagram

```python
from topology_diagrammer import NetworkTopologyDiagrammer

diagrammer = NetworkTopologyDiagrammer()

cdp_data = {
    "router-01": "Device ID: switch-01\nInterface: Gi0/1..."
}

topology = diagrammer.create_topology_documentation(cdp_data)
```

### Run Automated Pipeline

```bash
# One-time generation
python documentation_pipeline.py --generate-now

# Scheduled daily at 2 AM
python documentation_pipeline.py --schedule

# With Git versioning
python documentation_pipeline.py --generate-now --git-repo /path/to/repo
```

## CI/CD Integration

See the GitHub Actions workflow in the chapter for automated doc generation on config changes.

## What Gets Generated

- **Device Overview**: Role, management IP, protocols, features
- **Interface Table**: All interfaces in markdown format
- **Routing Documentation**: Protocols, neighbors, policies
- **Security Documentation**: ACLs, AAA, management access
- **Topology Diagrams**: Mermaid diagrams from CDP/LLDP

## Dependencies

- `anthropic` - Claude API client
- `schedule` - For scheduled runs (optional)
- `gitpython` - For Git versioning (optional)

## Cost Considerations

| Operation | Model | Approx. Cost/Device |
|-----------|-------|---------------------|
| Simple extraction (CDP parse, interface list) | Claude Haiku | ~$0.001 |
| Full documentation (4 sections) | Claude Sonnet | ~$0.01-0.05 |

For 500 devices daily: ~$5-25/run depending on config size.

**Cost optimization**: Only regenerate docs for devices whose configs actually changed (diff against last run).

## Next Chapter

**Chapter 14: RAG Fundamentals** - Make this documentation searchable with AI.
