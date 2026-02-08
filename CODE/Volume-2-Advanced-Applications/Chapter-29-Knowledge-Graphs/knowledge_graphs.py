"""
Chapter 29: Knowledge Graphs for Network Documentation
Build structured knowledge graphs from unstructured network docs
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import networkx as nx
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import json


# ============================================================================
# Data Models
# ============================================================================

class EntityType(str, Enum):
    """Network entity types"""
    ROUTER = "ROUTER"
    SWITCH = "SWITCH"
    FIREWALL = "FIREWALL"
    LOAD_BALANCER = "LOAD_BALANCER"
    SERVER = "SERVER"
    INTERFACE = "INTERFACE"
    SUBNET = "SUBNET"
    VLAN = "VLAN"
    VRF = "VRF"
    BGP_PEER = "BGP_PEER"
    OSPF_AREA = "OSPF_AREA"


class RelationType(str, Enum):
    """Relationship types between entities"""
    CONNECTED_TO = "CONNECTED_TO"
    HAS_INTERFACE = "HAS_INTERFACE"
    BELONGS_TO = "BELONGS_TO"
    PEERS_WITH = "PEERS_WITH"
    ROUTES_TO = "ROUTES_TO"
    AGGREGATES = "AGGREGATES"
    REDUNDANT_WITH = "REDUNDANT_WITH"
    DEPENDS_ON = "DEPENDS_ON"


@dataclass
class Entity:
    """Network entity"""
    entity_id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class Relationship:
    """Relationship between entities"""
    source_id: str
    target_id: str
    relationship_type: RelationType
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_entity(self, entity: Entity):
        """Add entity to graph"""
        self.entities[entity.entity_id] = entity
        self.graph.add_node(entity.entity_id, **entity.__dict__)

    def add_relationship(self, relationship: Relationship):
        """Add relationship to graph"""
        self.relationships.append(relationship)
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            type=relationship.relationship_type.value,
            **relationship.properties
        )


# ============================================================================
# Pydantic Models for LLM Structured Outputs
# ============================================================================

class ExtractedEntity(BaseModel):
    """LLM-extracted entity"""
    name: str = Field(description="Entity name (e.g., 'core-router-01', 'VLAN-100')")
    entity_type: str = Field(description="Type: ROUTER, SWITCH, FIREWALL, SERVER, SUBNET, VLAN, etc.")
    properties: Dict[str, str] = Field(description="Key properties (model, location, IP, etc.)")


class ExtractedRelationship(BaseModel):
    """LLM-extracted relationship"""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type: CONNECTED_TO, HAS_INTERFACE, PEERS_WITH, etc.")
    properties: Dict[str, str] = Field(description="Relationship properties (interface, protocol, etc.)")


class DocumentExtraction(BaseModel):
    """Complete extraction from document"""
    entities: List[ExtractedEntity] = Field(description="All entities found (3-10 entities)")
    relationships: List[ExtractedRelationship] = Field(description="All relationships (2-8 relationships)")


class PathQuery(BaseModel):
    """LLM answer to path query"""
    path_exists: bool = Field(description="True if a path exists between source and target")
    path_description: str = Field(description="Human-readable path description")
    path_nodes: List[str] = Field(description="List of node names in the path")
    hop_count: int = Field(description="Number of hops in the path")
    technologies: List[str] = Field(description="Technologies involved (BGP, OSPF, VLAN, etc.)")


class ImpactAnalysis(BaseModel):
    """LLM impact analysis"""
    affected_entities: List[str] = Field(description="Entities directly affected (2-6 items)")
    downstream_impact: List[str] = Field(description="Downstream services/systems impacted (2-5 items)")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    mitigation_steps: List[str] = Field(description="Steps to mitigate impact (2-4 items)")
    estimated_blast_radius: int = Field(description="Number of entities in blast radius")


# ============================================================================
# Knowledge Graph Builder
# ============================================================================

class KnowledgeGraphBuilder:
    """Build knowledge graphs from network documentation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=self.api_key,
            temperature=0
        )
        self.kg = KnowledgeGraph()

    def extract_from_document(self, document: str) -> DocumentExtraction:
        """
        Extract entities and relationships from document using LLM

        Args:
            document: Network documentation text

        Returns:
            DocumentExtraction with entities and relationships
        """
        parser = PydanticOutputParser(pydantic_object=DocumentExtraction)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network documentation parser.
Extract all network entities (devices, interfaces, subnets, VLANs) and their relationships.
Focus on physical and logical connections.

{format_instructions}"""),
            ("human", """Parse this network documentation and extract entities and relationships:

{document}

Extract all devices, interfaces, subnets, VLANs, and their connections.""")
        ])

        chain = prompt | self.llm | parser

        extraction = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "document": document
        })

        return extraction

    def build_graph_from_extraction(self, extraction: DocumentExtraction) -> KnowledgeGraph:
        """
        Build knowledge graph from extraction

        Args:
            extraction: Extracted entities and relationships

        Returns:
            Knowledge graph
        """
        # Add entities
        for ext_entity in extraction.entities:
            try:
                entity_type = EntityType[ext_entity.entity_type.upper().replace(" ", "_")]
            except KeyError:
                entity_type = EntityType.ROUTER  # Default fallback

            entity = Entity(
                entity_id=ext_entity.name.lower().replace(" ", "_"),
                entity_type=entity_type,
                name=ext_entity.name,
                properties=ext_entity.properties
            )
            self.kg.add_entity(entity)

        # Add relationships
        for ext_rel in extraction.relationships:
            source_id = ext_rel.source.lower().replace(" ", "_")
            target_id = ext_rel.target.lower().replace(" ", "_")

            # Verify entities exist
            if source_id not in self.kg.entities or target_id not in self.kg.entities:
                continue

            try:
                rel_type = RelationType[ext_rel.relationship_type.upper().replace(" ", "_")]
            except KeyError:
                rel_type = RelationType.CONNECTED_TO  # Default fallback

            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                properties=ext_rel.properties
            )
            self.kg.add_relationship(relationship)

        return self.kg

    def query_path(self, source: str, target: str) -> Dict:
        """
        Query path between two entities

        Args:
            source: Source entity name
            target: Target entity name

        Returns:
            Path information with AI explanation
        """
        source_id = source.lower().replace(" ", "_")
        target_id = target.lower().replace(" ", "_")

        # Check if entities exist
        if source_id not in self.kg.entities or target_id not in self.kg.entities:
            return {
                "path_exists": False,
                "error": "Source or target entity not found in graph"
            }

        # Find shortest path
        try:
            path = nx.shortest_path(self.kg.graph, source_id, target_id)
            path_exists = True
        except nx.NetworkXNoPath:
            path = []
            path_exists = False

        # Build context for LLM
        path_context = []
        if path_exists:
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                edge_data = self.kg.graph.get_edge_data(src, dst)
                src_entity = self.kg.entities[src]
                dst_entity = self.kg.entities[dst]

                path_context.append({
                    "hop": i + 1,
                    "from": src_entity.name,
                    "from_type": src_entity.entity_type.value,
                    "to": dst_entity.name,
                    "to_type": dst_entity.entity_type.value,
                    "relationship": edge_data.get("type", "UNKNOWN"),
                    "properties": edge_data
                })

        # Get AI explanation
        parser = PydanticOutputParser(pydantic_object=PathQuery)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network path analyzer.
Analyze the path between network entities and explain the connectivity.

{format_instructions}"""),
            ("human", """Query: Find path from {source} to {target}

Path found: {path_exists}
Path details:
{path_context}

Explain this network path in plain English.""")
        ])

        chain = prompt | self.llm | parser

        result = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "source": source,
            "target": target,
            "path_exists": path_exists,
            "path_context": json.dumps(path_context, indent=2) if path_context else "No path found"
        })

        return {
            "source": source,
            "target": target,
            "path_exists": result.path_exists,
            "path_description": result.path_description,
            "path_nodes": result.path_nodes,
            "hop_count": result.hop_count,
            "technologies": result.technologies,
            "raw_path": [self.kg.entities[node].name for node in path] if path else []
        }

    def analyze_impact(self, failed_entity: str) -> Dict:
        """
        Analyze impact of entity failure

        Args:
            failed_entity: Entity that failed

        Returns:
            Impact analysis with blast radius
        """
        entity_id = failed_entity.lower().replace(" ", "_")

        if entity_id not in self.kg.entities:
            return {"error": "Entity not found"}

        entity = self.kg.entities[entity_id]

        # Find all downstream entities (using DFS)
        downstream = list(nx.descendants(self.kg.graph, entity_id))
        downstream_entities = [self.kg.entities[eid] for eid in downstream]

        # Find direct connections
        direct_connections = list(self.kg.graph.successors(entity_id))
        direct_entities = [self.kg.entities[eid] for eid in direct_connections]

        # Build context for LLM
        context = {
            "failed_entity": {
                "name": entity.name,
                "type": entity.entity_type.value,
                "properties": entity.properties
            },
            "direct_connections": [
                {"name": e.name, "type": e.entity_type.value}
                for e in direct_entities
            ],
            "downstream_count": len(downstream_entities),
            "downstream_entities": [
                {"name": e.name, "type": e.entity_type.value}
                for e in downstream_entities[:10]  # Limit to 10
            ]
        }

        # Get AI analysis
        parser = PydanticOutputParser(pydantic_object=ImpactAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network impact analyst.
Analyze the blast radius and impact of a network entity failure.

{format_instructions}"""),
            ("human", """Analyze the impact of this failure:

Failed Entity: {failed_entity_name} ({failed_entity_type})
Properties: {failed_entity_props}

Direct Connections: {direct_count}
{direct_connections}

Total Downstream Impact: {downstream_count} entities
Sample Downstream Entities:
{downstream_entities}

Provide comprehensive impact analysis.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "failed_entity_name": entity.name,
            "failed_entity_type": entity.entity_type.value,
            "failed_entity_props": json.dumps(entity.properties),
            "direct_count": len(direct_entities),
            "direct_connections": "\n".join([f"- {e.name} ({e.entity_type.value})" for e in direct_entities]),
            "downstream_count": len(downstream_entities),
            "downstream_entities": "\n".join([f"- {e.name} ({e.entity_type.value})" for e in downstream_entities[:10]])
        })

        return {
            "failed_entity": entity.name,
            "entity_type": entity.entity_type.value,
            "affected_entities": analysis.affected_entities,
            "downstream_impact": analysis.downstream_impact,
            "severity": analysis.severity,
            "mitigation_steps": analysis.mitigation_steps,
            "estimated_blast_radius": analysis.estimated_blast_radius,
            "actual_downstream_count": len(downstream_entities),
            "direct_connections_count": len(direct_entities)
        }

    def find_redundant_paths(self, source: str, target: str) -> Dict:
        """
        Find all paths between entities (redundancy analysis)

        Args:
            source: Source entity
            target: Target entity

        Returns:
            All paths and redundancy assessment
        """
        source_id = source.lower().replace(" ", "_")
        target_id = target.lower().replace(" ", "_")

        if source_id not in self.kg.entities or target_id not in self.kg.entities:
            return {"error": "Entity not found"}

        # Find all simple paths
        try:
            all_paths = list(nx.all_simple_paths(self.kg.graph, source_id, target_id, cutoff=10))
        except nx.NetworkXNoPath:
            all_paths = []

        # Convert to entity names
        paths_with_names = []
        for path in all_paths:
            path_names = [self.kg.entities[node].name for node in path]
            paths_with_names.append(path_names)

        return {
            "source": source,
            "target": target,
            "path_count": len(all_paths),
            "redundancy": "HIGH" if len(all_paths) >= 3 else "MEDIUM" if len(all_paths) == 2 else "LOW" if len(all_paths) == 1 else "NONE",
            "paths": paths_with_names[:5],  # Limit to 5 paths
            "shortest_path_length": len(all_paths[0]) if all_paths else 0
        }

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """
        Get all neighbors of an entity up to specified depth

        Args:
            entity_name: Entity to query
            depth: How many hops to traverse

        Returns:
            Neighbor entities by depth
        """
        entity_id = entity_name.lower().replace(" ", "_")

        if entity_id not in self.kg.entities:
            return {"error": "Entity not found"}

        neighbors_by_depth = {}

        for d in range(1, depth + 1):
            if d == 1:
                # Direct neighbors
                neighbors = list(self.kg.graph.neighbors(entity_id))
            else:
                # Use BFS to get neighbors at exact depth d
                all_paths = nx.single_source_shortest_path_length(self.kg.graph, entity_id, cutoff=d)
                neighbors = [node for node, dist in all_paths.items() if dist == d]

            neighbor_entities = [
                {
                    "name": self.kg.entities[nid].name,
                    "type": self.kg.entities[nid].entity_type.value,
                    "properties": self.kg.entities[nid].properties
                }
                for nid in neighbors if nid in self.kg.entities
            ]

            neighbors_by_depth[f"depth_{d}"] = neighbor_entities

        return {
            "entity": entity_name,
            "query_depth": depth,
            "neighbors": neighbors_by_depth,
            "total_neighbors": sum(len(v) for v in neighbors_by_depth.values())
        }


# ============================================================================
# Examples
# ============================================================================

def example_1_extract_from_document():
    """Example 1: Extract knowledge graph from documentation"""
    print("=" * 80)
    print("Example 1: Extract Knowledge Graph from Documentation")
    print("=" * 80)

    builder = KnowledgeGraphBuilder()

    # Sample network documentation
    document = """
    Network Infrastructure Documentation - Data Center A

    Core Routers:
    - core-router-01 (Cisco ASR 9010, 10.0.0.1)
      Connected to: core-router-02 (redundant link), dist-switch-01, dist-switch-02
      BGP Peering: AS 65001

    - core-router-02 (Cisco ASR 9010, 10.0.0.2)
      Connected to: core-router-01 (redundant link), dist-switch-01, dist-switch-02
      BGP Peering: AS 65001

    Distribution Switches:
    - dist-switch-01 (Arista 7280, 10.0.1.1)
      Connected to: core-router-01, core-router-02, access-switch-01, access-switch-02
      VLAN: 100, 200, 300

    - dist-switch-02 (Arista 7280, 10.0.1.2)
      Connected to: core-router-01, core-router-02, access-switch-03, access-switch-04
      VLAN: 100, 200, 300

    Access Switches:
    - access-switch-01 (Cisco Catalyst 3850, 10.0.2.1)
      Connected to: dist-switch-01
      VLAN 100: Servers
      Servers: web-server-01, web-server-02

    - access-switch-02 (Cisco Catalyst 3850, 10.0.2.2)
      Connected to: dist-switch-01
      VLAN 200: Database servers
      Servers: db-server-01, db-server-02
    """

    print("\nExtracting entities and relationships from documentation...")
    print(f"Document length: {len(document)} characters\n")

    extraction = builder.extract_from_document(document)

    print(f"Extracted {len(extraction.entities)} entities:")
    for entity in extraction.entities[:8]:  # Show first 8
        print(f"  - {entity.name} ({entity.entity_type})")
        if entity.properties:
            props = ", ".join([f"{k}={v}" for k, v in list(entity.properties.items())[:2]])
            print(f"    Properties: {props}")

    print(f"\nExtracted {len(extraction.relationships)} relationships:")
    for rel in extraction.relationships[:8]:  # Show first 8
        print(f"  - {rel.source} --[{rel.relationship_type}]--> {rel.target}")

    # Build graph
    print("\nBuilding knowledge graph...")
    kg = builder.build_graph_from_extraction(extraction)

    print(f"\nKnowledge Graph Statistics:")
    print(f"  Total Entities: {len(kg.entities)}")
    print(f"  Total Relationships: {len(kg.relationships)}")
    print(f"  Graph Nodes: {kg.graph.number_of_nodes()}")
    print(f"  Graph Edges: {kg.graph.number_of_edges()}")


def example_2_query_path():
    """Example 2: Query path between entities"""
    print("\n" + "=" * 80)
    print("Example 2: Query Network Path")
    print("=" * 80)

    builder = KnowledgeGraphBuilder()

    # Build simple test graph
    entities = [
        Entity("core-router-01", EntityType.ROUTER, "core-router-01", {"ip": "10.0.0.1"}),
        Entity("dist-switch-01", EntityType.SWITCH, "dist-switch-01", {"ip": "10.0.1.1"}),
        Entity("access-switch-01", EntityType.SWITCH, "access-switch-01", {"ip": "10.0.2.1"}),
        Entity("web-server-01", EntityType.SERVER, "web-server-01", {"ip": "10.1.1.10"}),
    ]

    relationships = [
        Relationship("core-router-01", "dist-switch-01", RelationType.CONNECTED_TO, {"interface": "Eth1/1"}),
        Relationship("dist-switch-01", "access-switch-01", RelationType.CONNECTED_TO, {"interface": "Eth1/2"}),
        Relationship("access-switch-01", "web-server-01", RelationType.CONNECTED_TO, {"vlan": "100"}),
    ]

    for entity in entities:
        builder.kg.add_entity(entity)

    for rel in relationships:
        builder.kg.add_relationship(rel)

    print("\nGraph built:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")

    # Query path
    print("\nQuerying path: core-router-01 → web-server-01")
    result = builder.query_path("core-router-01", "web-server-01")

    if result.get("path_exists"):
        print(f"\n✓ Path found!")
        print(f"  Hops: {result['hop_count']}")
        print(f"  Path: {' → '.join(result['path_nodes'])}")
        print(f"  Technologies: {', '.join(result['technologies'])}")
        print(f"\nDescription:")
        print(f"  {result['path_description']}")
    else:
        print(f"\n✗ No path found")


def example_3_impact_analysis():
    """Example 3: Analyze failure impact"""
    print("\n" + "=" * 80)
    print("Example 3: Failure Impact Analysis")
    print("=" * 80)

    builder = KnowledgeGraphBuilder()

    # Build test topology
    entities = [
        Entity("core-router-01", EntityType.ROUTER, "core-router-01", {"critical": "yes"}),
        Entity("dist-switch-01", EntityType.SWITCH, "dist-switch-01", {}),
        Entity("dist-switch-02", EntityType.SWITCH, "dist-switch-02", {}),
        Entity("access-switch-01", EntityType.SWITCH, "access-switch-01", {}),
        Entity("access-switch-02", EntityType.SWITCH, "access-switch-02", {}),
        Entity("web-server-01", EntityType.SERVER, "web-server-01", {"service": "web"}),
        Entity("web-server-02", EntityType.SERVER, "web-server-02", {"service": "web"}),
        Entity("db-server-01", EntityType.SERVER, "db-server-01", {"service": "database"}),
    ]

    relationships = [
        Relationship("core-router-01", "dist-switch-01", RelationType.CONNECTED_TO, {}),
        Relationship("core-router-01", "dist-switch-02", RelationType.CONNECTED_TO, {}),
        Relationship("dist-switch-01", "access-switch-01", RelationType.CONNECTED_TO, {}),
        Relationship("dist-switch-02", "access-switch-02", RelationType.CONNECTED_TO, {}),
        Relationship("access-switch-01", "web-server-01", RelationType.CONNECTED_TO, {}),
        Relationship("access-switch-01", "web-server-02", RelationType.CONNECTED_TO, {}),
        Relationship("access-switch-02", "db-server-01", RelationType.CONNECTED_TO, {}),
    ]

    for entity in entities:
        builder.kg.add_entity(entity)

    for rel in relationships:
        builder.kg.add_relationship(rel)

    print("\nTopology built:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")

    # Analyze impact
    print("\nAnalyzing impact of: core-router-01 failure")
    impact = builder.analyze_impact("core-router-01")

    print(f"\n{'='*80}")
    print(f"IMPACT ANALYSIS")
    print(f"{'='*80}")
    print(f"Failed Entity: {impact['failed_entity']} ({impact['entity_type']})")
    print(f"Severity: {impact['severity']}")
    print(f"Blast Radius: {impact['estimated_blast_radius']} entities")
    print(f"Direct Connections: {impact['direct_connections_count']}")
    print(f"Total Downstream: {impact['actual_downstream_count']}")

    print(f"\nAffected Entities:")
    for entity in impact['affected_entities']:
        print(f"  - {entity}")

    print(f"\nDownstream Impact:")
    for service in impact['downstream_impact']:
        print(f"  - {service}")

    print(f"\nMitigation Steps:")
    for i, step in enumerate(impact['mitigation_steps'], 1):
        print(f"  {i}. {step}")


def example_4_redundancy_analysis():
    """Example 4: Analyze path redundancy"""
    print("\n" + "=" * 80)
    print("Example 4: Redundancy Analysis")
    print("=" * 80)

    builder = KnowledgeGraphBuilder()

    # Build topology with redundant paths
    entities = [
        Entity("router-a", EntityType.ROUTER, "router-a", {}),
        Entity("router-b", EntityType.ROUTER, "router-b", {}),
        Entity("router-c", EntityType.ROUTER, "router-c", {}),
        Entity("router-d", EntityType.ROUTER, "router-d", {}),
        Entity("server-01", EntityType.SERVER, "server-01", {}),
    ]

    # Create redundant paths: A → D and A → B → C → D and A → B → D
    relationships = [
        Relationship("router-a", "router-d", RelationType.CONNECTED_TO, {"path": "direct"}),
        Relationship("router-a", "router-b", RelationType.CONNECTED_TO, {"path": "via-b"}),
        Relationship("router-b", "router-c", RelationType.CONNECTED_TO, {"path": "via-c"}),
        Relationship("router-b", "router-d", RelationType.CONNECTED_TO, {"path": "via-d"}),
        Relationship("router-c", "router-d", RelationType.CONNECTED_TO, {"path": "via-c"}),
        Relationship("router-d", "server-01", RelationType.CONNECTED_TO, {}),
    ]

    for entity in entities:
        builder.kg.add_entity(entity)

    for rel in relationships:
        builder.kg.add_relationship(rel)

    print("\nTopology with redundant paths:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")

    # Analyze redundancy
    print("\nAnalyzing redundancy: router-a → router-d")
    redundancy = builder.find_redundant_paths("router-a", "router-d")

    print(f"\n{'='*80}")
    print(f"REDUNDANCY ANALYSIS")
    print(f"{'='*80}")
    print(f"Source: {redundancy['source']}")
    print(f"Target: {redundancy['target']}")
    print(f"Available Paths: {redundancy['path_count']}")
    print(f"Redundancy Level: {redundancy['redundancy']}")
    print(f"Shortest Path Length: {redundancy['shortest_path_length']} hops")

    print(f"\nAll Paths:")
    for i, path in enumerate(redundancy['paths'], 1):
        print(f"  Path {i}: {' → '.join(path)}")


def example_5_neighbor_discovery():
    """Example 5: Discover entity neighbors"""
    print("\n" + "=" * 80)
    print("Example 5: Neighbor Discovery")
    print("=" * 80)

    builder = KnowledgeGraphBuilder()

    # Build test topology
    entities = [
        Entity("core-01", EntityType.ROUTER, "core-01", {"role": "core"}),
        Entity("dist-01", EntityType.SWITCH, "dist-01", {"role": "distribution"}),
        Entity("dist-02", EntityType.SWITCH, "dist-02", {"role": "distribution"}),
        Entity("access-01", EntityType.SWITCH, "access-01", {"role": "access"}),
        Entity("access-02", EntityType.SWITCH, "access-02", {"role": "access"}),
        Entity("server-01", EntityType.SERVER, "server-01", {"service": "web"}),
        Entity("server-02", EntityType.SERVER, "server-02", {"service": "db"}),
    ]

    relationships = [
        Relationship("core-01", "dist-01", RelationType.CONNECTED_TO, {}),
        Relationship("core-01", "dist-02", RelationType.CONNECTED_TO, {}),
        Relationship("dist-01", "access-01", RelationType.CONNECTED_TO, {}),
        Relationship("dist-02", "access-02", RelationType.CONNECTED_TO, {}),
        Relationship("access-01", "server-01", RelationType.CONNECTED_TO, {}),
        Relationship("access-02", "server-02", RelationType.CONNECTED_TO, {}),
    ]

    for entity in entities:
        builder.kg.add_entity(entity)

    for rel in relationships:
        builder.kg.add_relationship(rel)

    print("\nTopology built:")
    print(f"  Entities: {len(entities)}")

    # Discover neighbors
    print("\nDiscovering neighbors of: core-01 (depth=3)")
    neighbors = builder.get_entity_neighbors("core-01", depth=3)

    print(f"\n{'='*80}")
    print(f"NEIGHBOR DISCOVERY")
    print(f"{'='*80}")
    print(f"Entity: {neighbors['entity']}")
    print(f"Query Depth: {neighbors['query_depth']}")
    print(f"Total Neighbors: {neighbors['total_neighbors']}")

    for depth_key, entities_list in neighbors['neighbors'].items():
        depth = depth_key.split('_')[1]
        print(f"\nDepth {depth} ({len(entities_list)} entities):")
        for entity in entities_list:
            print(f"  - {entity['name']} ({entity['type']})")
            if entity.get('properties'):
                props = ", ".join([f"{k}={v}" for k, v in entity['properties'].items()])
                print(f"    {props}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 29: Knowledge Graphs for Network Documentation")
    print("Build structured knowledge graphs from unstructured docs")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_extract_from_document()
    example_2_query_path()
    example_3_impact_analysis()
    example_4_redundancy_analysis()
    example_5_neighbor_discovery()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
