"""
Chapter 37: Graph RAG for Network Topology
Production-ready hybrid RAG system using knowledge graphs for network topology analysis

Combines:
- Network topology as knowledge graph (NetworkX)
- Graph-based retrieval for topology queries
- Vector embeddings for semantic search
- Path finding and dependency mapping
- Impact analysis using graph traversal

Author: Eduard Dulharu (@eduardd76)
Company: vExpertAI GmbH
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import networkx as nx
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json


# ============================================================================
# Data Models
# ============================================================================

class DeviceType(str, Enum):
    """Network device types"""
    ROUTER = "ROUTER"
    SWITCH = "SWITCH"
    FIREWALL = "FIREWALL"
    LOAD_BALANCER = "LOAD_BALANCER"
    SERVER = "SERVER"
    WAN_ACCELERATOR = "WAN_ACCELERATOR"


class ConnectionType(str, Enum):
    """Connection types between devices"""
    L2_TRUNK = "L2_TRUNK"
    L2_ACCESS = "L2_ACCESS"
    L3_LINK = "L3_LINK"
    BGP_PEER = "BGP_PEER"
    OSPF_NEIGHBOR = "OSPF_NEIGHBOR"
    REDUNDANT_PAIR = "REDUNDANT_PAIR"
    AGGREGATION = "AGGREGATION"


@dataclass
class NetworkDevice:
    """Network device node"""
    device_id: str
    device_type: DeviceType
    hostname: str
    management_ip: str
    properties: Dict[str, Any] = field(default_factory=dict)
    interfaces: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)


@dataclass
class NetworkConnection:
    """Connection between devices"""
    source_id: str
    target_id: str
    connection_type: ConnectionType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class TopologyQuery:
    """Topology query result"""
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    retrieval_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pydantic Models for LLM Outputs
# ============================================================================

class TopologyQueryAnalysis(BaseModel):
    """LLM analysis of topology query"""
    query_type: str = Field(description="PATH_FINDING, IMPACT_ANALYSIS, NEIGHBOR_DISCOVERY, or DEPENDENCY_MAPPING")
    source_entity: Optional[str] = Field(description="Source device/entity if applicable")
    target_entity: Optional[str] = Field(description="Target device/entity if applicable")
    key_entities: List[str] = Field(description="Key entities to focus on (2-5 items)")
    recommended_strategy: str = Field(description="GRAPH_TRAVERSAL, VECTOR_SEARCH, or HYBRID")
    reasoning: str = Field(description="Why this strategy is recommended")


class PathAnalysis(BaseModel):
    """LLM analysis of network path"""
    path_exists: bool = Field(description="Whether a path exists")
    path_description: str = Field(description="Human-readable path explanation (2-3 sentences)")
    critical_hops: List[str] = Field(description="Critical devices in the path (2-4 items)")
    path_type: str = Field(description="L2_ONLY, L3_ROUTED, MIXED, or REDUNDANT")
    technologies: List[str] = Field(description="Technologies used (VLAN, OSPF, BGP, etc.)")
    latency_estimate: str = Field(description="LOW, MEDIUM, or HIGH")


class ImpactAnalysisResult(BaseModel):
    """LLM impact analysis result"""
    blast_radius: int = Field(description="Number of directly affected devices")
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    affected_services: List[str] = Field(description="Services that would be impacted (2-6 items)")
    downstream_devices: List[str] = Field(description="Downstream devices affected (3-8 items)")
    mitigation_options: List[str] = Field(description="Mitigation strategies (2-4 items)")
    recovery_time_estimate: str = Field(description="Estimated recovery time")


class DependencyMapResult(BaseModel):
    """LLM dependency mapping result"""
    primary_dependencies: List[str] = Field(description="Direct dependencies (2-5 items)")
    secondary_dependencies: List[str] = Field(description="Indirect dependencies (2-5 items)")
    single_points_of_failure: List[str] = Field(description="SPOFs identified (1-3 items)")
    redundancy_level: str = Field(description="HIGH, MEDIUM, LOW, or NONE")
    recommendations: List[str] = Field(description="Improvement recommendations (2-4 items)")


class HybridRAGAnswer(BaseModel):
    """LLM answer using hybrid RAG"""
    answer: str = Field(description="Complete answer (3-5 sentences)")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    graph_insights: List[str] = Field(description="Insights from graph traversal (2-4 items)")
    vector_insights: List[str] = Field(description="Insights from semantic search (2-4 items)")
    recommendations: List[str] = Field(description="Actionable recommendations (2-3 items)")


# ============================================================================
# Network Topology Graph
# ============================================================================

class NetworkTopologyGraph:
    """Network topology represented as a knowledge graph"""

    def __init__(self, api_key_anthropic: Optional[str] = None,
                 api_key_openai: Optional[str] = None):
        self.api_key_anthropic = api_key_anthropic or os.environ.get("ANTHROPIC_API_KEY")
        self.api_key_openai = api_key_openai or os.environ.get("OPENAI_API_KEY")

        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=self.api_key_anthropic,
            temperature=0
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key_openai
        )

        # Storage
        self.devices: Dict[str, NetworkDevice] = {}
        self.connections: List[NetworkConnection] = []
        self.graph = nx.DiGraph()

        # Embeddings for semantic search
        self.device_embeddings: Dict[str, np.ndarray] = {}
        self.device_descriptions: Dict[str, str] = {}

    def add_device(self, device: NetworkDevice):
        """
        Add device to topology graph

        Args:
            device: Network device to add
        """
        self.devices[device.device_id] = device

        # Add to graph
        self.graph.add_node(
            device.device_id,
            device_type=device.device_type.value,
            hostname=device.hostname,
            management_ip=device.management_ip,
            **device.properties
        )

        # Create description for embeddings
        description = f"{device.hostname} ({device.device_type.value}) at {device.management_ip}"
        if device.properties:
            props = ", ".join([f"{k}={v}" for k, v in device.properties.items()])
            description += f" - {props}"
        if device.protocols:
            description += f" - Protocols: {', '.join(device.protocols)}"

        self.device_descriptions[device.device_id] = description

        # Create embedding
        embedding = np.array(self.embeddings.embed_query(description))
        self.device_embeddings[device.device_id] = embedding

    def add_connection(self, connection: NetworkConnection):
        """
        Add connection to topology graph

        Args:
            connection: Network connection to add
        """
        self.connections.append(connection)

        # Add to graph
        self.graph.add_edge(
            connection.source_id,
            connection.target_id,
            connection_type=connection.connection_type.value,
            weight=connection.weight,
            **connection.properties
        )

    def analyze_query(self, query: str) -> TopologyQueryAnalysis:
        """
        Analyze query to determine best approach

        Args:
            query: User query about network topology

        Returns:
            Query analysis with recommended strategy
        """
        parser = PydanticOutputParser(pydantic_object=TopologyQueryAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network topology query analyzer.

Analyze queries and determine:
- Query type: PATH_FINDING, IMPACT_ANALYSIS, NEIGHBOR_DISCOVERY, or DEPENDENCY_MAPPING
- Key entities involved
- Best retrieval strategy:
  - GRAPH_TRAVERSAL: For path finding, neighbor discovery, direct relationships
  - VECTOR_SEARCH: For semantic/similarity queries, "devices like X"
  - HYBRID: For complex queries needing both

{format_instructions}"""),
            ("human", """Analyze this network topology query:

{query}

Determine query type, key entities, and recommended retrieval strategy.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query
        })

        return analysis

    def find_path(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """
        Find path between two devices

        Args:
            source_id: Source device ID
            target_id: Target device ID

        Returns:
            Path information with AI analysis
        """
        if source_id not in self.devices or target_id not in self.devices:
            return {"error": "Device not found", "path_exists": False}

        # Find shortest path
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            path_exists = True
        except nx.NetworkXNoPath:
            return {"error": "No path exists", "path_exists": False}

        # Build path details
        path_details = []
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            edge_data = self.graph.get_edge_data(src, dst)
            src_device = self.devices[src]
            dst_device = self.devices[dst]

            path_details.append({
                "hop": i + 1,
                "from": src_device.hostname,
                "from_type": src_device.device_type.value,
                "from_ip": src_device.management_ip,
                "to": dst_device.hostname,
                "to_type": dst_device.device_type.value,
                "to_ip": dst_device.management_ip,
                "connection_type": edge_data.get("connection_type", "UNKNOWN"),
                "properties": edge_data
            })

        # Get AI analysis
        parser = PydanticOutputParser(pydantic_object=PathAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network path analyzer.
Analyze the network path and explain connectivity, technologies, and characteristics.

{format_instructions}"""),
            ("human", """Analyze this network path:

Source: {source}
Target: {target}

Path Details:
{path_details}

Provide comprehensive path analysis.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "source": self.devices[source_id].hostname,
            "target": self.devices[target_id].hostname,
            "path_details": json.dumps(path_details, indent=2)
        })

        return {
            "path_exists": True,
            "path": [self.devices[node].hostname for node in path],
            "path_ids": path,
            "hop_count": len(path) - 1,
            "path_description": analysis.path_description,
            "critical_hops": analysis.critical_hops,
            "path_type": analysis.path_type,
            "technologies": analysis.technologies,
            "latency_estimate": analysis.latency_estimate,
            "detailed_hops": path_details
        }

    def analyze_impact(self, device_id: str) -> Dict[str, Any]:
        """
        Analyze impact of device failure

        Args:
            device_id: Device that failed

        Returns:
            Impact analysis with blast radius
        """
        if device_id not in self.devices:
            return {"error": "Device not found"}

        device = self.devices[device_id]

        # Find downstream devices
        try:
            downstream = list(nx.descendants(self.graph, device_id))
        except:
            downstream = []

        # Find direct neighbors
        direct_neighbors = list(self.graph.successors(device_id))

        # Find redundant paths for critical services
        redundancy_info = []
        for target_id in downstream[:5]:  # Check first 5 downstream
            if target_id in self.devices:
                # Check if alternative path exists
                temp_graph = self.graph.copy()
                temp_graph.remove_node(device_id)

                # Find paths from any predecessor to this target
                predecessors = list(self.graph.predecessors(device_id))
                has_backup = False
                for pred in predecessors:
                    try:
                        nx.shortest_path(temp_graph, pred, target_id)
                        has_backup = True
                        break
                    except:
                        pass

                redundancy_info.append({
                    "device": self.devices[target_id].hostname,
                    "has_backup_path": has_backup
                })

        # Build context for LLM
        context = {
            "failed_device": {
                "hostname": device.hostname,
                "type": device.device_type.value,
                "ip": device.management_ip,
                "properties": device.properties
            },
            "direct_neighbors": [
                {"hostname": self.devices[nid].hostname, "type": self.devices[nid].device_type.value}
                for nid in direct_neighbors
            ],
            "downstream_count": len(downstream),
            "downstream_devices": [
                {"hostname": self.devices[did].hostname, "type": self.devices[did].device_type.value}
                for did in downstream[:10]
            ],
            "redundancy_info": redundancy_info
        }

        # Get AI analysis
        parser = PydanticOutputParser(pydantic_object=ImpactAnalysisResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network impact analyst.
Analyze blast radius and impact of device failures.

{format_instructions}"""),
            ("human", """Analyze failure impact:

Failed Device: {hostname} ({device_type})
IP: {ip}
Properties: {properties}

Direct Neighbors: {neighbor_count}
{neighbors}

Downstream Impact: {downstream_count} devices
{downstream}

Redundancy Information:
{redundancy}

Provide comprehensive impact analysis.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "hostname": device.hostname,
            "device_type": device.device_type.value,
            "ip": device.management_ip,
            "properties": json.dumps(device.properties),
            "neighbor_count": len(direct_neighbors),
            "neighbors": "\n".join([f"- {d['hostname']} ({d['type']})" for d in context['direct_neighbors']]),
            "downstream_count": len(downstream),
            "downstream": "\n".join([f"- {d['hostname']} ({d['type']})" for d in context['downstream_devices']]),
            "redundancy": json.dumps(redundancy_info, indent=2)
        })

        return {
            "device": device.hostname,
            "device_type": device.device_type.value,
            "blast_radius": analysis.blast_radius,
            "severity": analysis.severity,
            "affected_services": analysis.affected_services,
            "downstream_devices": analysis.downstream_devices,
            "mitigation_options": analysis.mitigation_options,
            "recovery_time_estimate": analysis.recovery_time_estimate,
            "actual_downstream_count": len(downstream),
            "direct_neighbors_count": len(direct_neighbors),
            "redundancy_analysis": redundancy_info
        }

    def map_dependencies(self, device_id: str) -> Dict[str, Any]:
        """
        Map all dependencies for a device

        Args:
            device_id: Device to analyze

        Returns:
            Dependency map with SPOFs identified
        """
        if device_id not in self.devices:
            return {"error": "Device not found"}

        device = self.devices[device_id]

        # Find all ancestors (devices this depends on)
        try:
            ancestors = list(nx.ancestors(self.graph, device_id))
        except:
            ancestors = []

        # Separate by depth
        primary_deps = list(self.graph.predecessors(device_id))
        secondary_deps = [a for a in ancestors if a not in primary_deps]

        # Identify single points of failure
        spofs = []
        for dep_id in primary_deps:
            # Check if this is the only path
            temp_graph = self.graph.copy()
            temp_graph.remove_node(dep_id)

            # Try to reach device from any other ancestor
            other_ancestors = [a for a in ancestors if a != dep_id]
            has_alternative = False

            for ancestor in other_ancestors:
                try:
                    nx.shortest_path(temp_graph, ancestor, device_id)
                    has_alternative = True
                    break
                except:
                    pass

            if not has_alternative:
                spofs.append(dep_id)

        # Build context
        context = {
            "primary": [
                {"hostname": self.devices[did].hostname, "type": self.devices[did].device_type.value}
                for did in primary_deps
            ],
            "secondary": [
                {"hostname": self.devices[did].hostname, "type": self.devices[did].device_type.value}
                for did in secondary_deps[:10]
            ],
            "spofs": [
                {"hostname": self.devices[did].hostname, "type": self.devices[did].device_type.value}
                for did in spofs
            ]
        }

        # Get AI analysis
        parser = PydanticOutputParser(pydantic_object=DependencyMapResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network dependency analyst.
Map dependencies and identify single points of failure.

{format_instructions}"""),
            ("human", """Analyze dependencies for:

Device: {hostname} ({device_type})

Primary Dependencies ({primary_count}):
{primary_deps}

Secondary Dependencies ({secondary_count}):
{secondary_deps}

Identified SPOFs ({spof_count}):
{spofs}

Provide dependency analysis and recommendations.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "hostname": device.hostname,
            "device_type": device.device_type.value,
            "primary_count": len(primary_deps),
            "primary_deps": "\n".join([f"- {d['hostname']} ({d['type']})" for d in context['primary']]),
            "secondary_count": len(secondary_deps),
            "secondary_deps": "\n".join([f"- {d['hostname']} ({d['type']})" for d in context['secondary']]),
            "spof_count": len(spofs),
            "spofs": "\n".join([f"- {d['hostname']} ({d['type']})" for d in context['spofs']]) if spofs else "None identified"
        })

        return {
            "device": device.hostname,
            "primary_dependencies": analysis.primary_dependencies,
            "secondary_dependencies": analysis.secondary_dependencies,
            "single_points_of_failure": analysis.single_points_of_failure,
            "redundancy_level": analysis.redundancy_level,
            "recommendations": analysis.recommendations,
            "dependency_details": context
        }

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Semantic search over network devices

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (device_id, score) tuples
        """
        query_embedding = np.array(self.embeddings.embed_query(query))

        similarities = []
        for device_id, device_embedding in self.device_embeddings.items():
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                device_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((device_id, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def hybrid_rag_query(self, query: str) -> Dict[str, Any]:
        """
        Answer query using hybrid RAG (graph + vector + LLM)

        Args:
            query: User query

        Returns:
            Comprehensive answer with insights
        """
        # Analyze query
        analysis = self.analyze_query(query)

        # Gather context using multiple methods
        graph_context = []
        vector_context = []

        # Vector search for semantic context
        vector_results = self.vector_search(query, top_k=5)
        for device_id, score in vector_results:
            device = self.devices[device_id]
            vector_context.append({
                "hostname": device.hostname,
                "type": device.device_type.value,
                "ip": device.management_ip,
                "description": self.device_descriptions[device_id],
                "relevance_score": score
            })

        # Graph traversal for structural context
        for device_id, score in vector_results[:3]:
            # Get neighbors
            neighbors = list(self.graph.neighbors(device_id))
            if neighbors:
                device = self.devices[device_id]
                neighbor_info = [
                    f"{self.devices[nid].hostname} ({self.devices[nid].device_type.value})"
                    for nid in neighbors[:5]
                ]
                graph_context.append({
                    "device": device.hostname,
                    "neighbors": neighbor_info,
                    "neighbor_count": len(neighbors)
                })

        # Generate answer with LLM
        parser = PydanticOutputParser(pydantic_object=HybridRAGAnswer)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network topology assistant.
Answer questions using both semantic knowledge and structural graph relationships.

{format_instructions}"""),
            ("human", """Question: {query}

Query Analysis:
- Type: {query_type}
- Key Entities: {key_entities}
- Strategy: {strategy}

Vector Search Results (Semantic Matches):
{vector_context}

Graph Context (Structural Relationships):
{graph_context}

Provide comprehensive answer combining both perspectives.""")
        ])

        chain = prompt | self.llm | parser

        answer = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query,
            "query_type": analysis.query_type,
            "key_entities": ", ".join(analysis.key_entities),
            "strategy": analysis.recommended_strategy,
            "vector_context": json.dumps(vector_context, indent=2),
            "graph_context": json.dumps(graph_context, indent=2)
        })

        return {
            "query": query,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "graph_insights": answer.graph_insights,
            "vector_insights": answer.vector_insights,
            "recommendations": answer.recommendations,
            "query_analysis": {
                "type": analysis.query_type,
                "strategy": analysis.recommended_strategy,
                "reasoning": analysis.reasoning
            },
            "sources": {
                "vector_matches": len(vector_results),
                "graph_nodes_explored": len(graph_context)
            }
        }


# ============================================================================
# Examples
# ============================================================================

def example_1_build_topology_graph():
    """Example 1: Build network topology graph"""
    print("=" * 80)
    print("Example 1: Build Network Topology Graph")
    print("=" * 80)

    topo = NetworkTopologyGraph()

    # Add devices
    devices = [
        NetworkDevice("core-r1", DeviceType.ROUTER, "core-router-01", "10.0.0.1",
                     {"model": "Cisco ASR9010", "location": "DC1", "role": "core"},
                     protocols=["BGP", "OSPF", "MPLS"]),
        NetworkDevice("core-r2", DeviceType.ROUTER, "core-router-02", "10.0.0.2",
                     {"model": "Cisco ASR9010", "location": "DC1", "role": "core"},
                     protocols=["BGP", "OSPF", "MPLS"]),
        NetworkDevice("dist-sw1", DeviceType.SWITCH, "dist-switch-01", "10.0.1.1",
                     {"model": "Arista 7280", "location": "DC1", "role": "distribution"},
                     protocols=["OSPF", "LACP"]),
        NetworkDevice("dist-sw2", DeviceType.SWITCH, "dist-switch-02", "10.0.1.2",
                     {"model": "Arista 7280", "location": "DC1", "role": "distribution"},
                     protocols=["OSPF", "LACP"]),
        NetworkDevice("fw1", DeviceType.FIREWALL, "firewall-01", "10.0.2.1",
                     {"model": "Palo Alto PA-5220", "location": "DC1", "role": "edge"}),
        NetworkDevice("web-srv", DeviceType.SERVER, "web-server-01", "10.1.0.10",
                     {"os": "Ubuntu 22.04", "service": "nginx", "location": "DC1"}),
        NetworkDevice("db-srv", DeviceType.SERVER, "db-server-01", "10.1.0.20",
                     {"os": "Ubuntu 22.04", "service": "postgresql", "location": "DC1"}),
    ]

    print("\nAdding devices to topology:")
    for device in devices:
        topo.add_device(device)
        print(f"  ✓ {device.hostname} ({device.device_type.value})")

    # Add connections
    connections = [
        NetworkConnection("core-r1", "core-r2", ConnectionType.REDUNDANT_PAIR,
                        {"protocol": "HSRP", "vlan": "all"}, weight=0.5),
        NetworkConnection("core-r1", "dist-sw1", ConnectionType.L3_LINK,
                        {"interface": "TenGigE0/0/0/0", "subnet": "10.255.1.0/30"}, weight=1.0),
        NetworkConnection("core-r1", "dist-sw2", ConnectionType.L3_LINK,
                        {"interface": "TenGigE0/0/0/1", "subnet": "10.255.2.0/30"}, weight=1.0),
        NetworkConnection("core-r2", "dist-sw1", ConnectionType.L3_LINK,
                        {"interface": "TenGigE0/0/0/0", "subnet": "10.255.3.0/30"}, weight=1.0),
        NetworkConnection("core-r2", "dist-sw2", ConnectionType.L3_LINK,
                        {"interface": "TenGigE0/0/0/1", "subnet": "10.255.4.0/30"}, weight=1.0),
        NetworkConnection("core-r1", "fw1", ConnectionType.L3_LINK,
                        {"interface": "TenGigE0/0/0/2", "subnet": "10.255.5.0/30"}, weight=1.0),
        NetworkConnection("dist-sw1", "web-srv", ConnectionType.L2_ACCESS,
                        {"vlan": "100", "port": "Eth1/10"}, weight=2.0),
        NetworkConnection("dist-sw2", "db-srv", ConnectionType.L2_ACCESS,
                        {"vlan": "200", "port": "Eth1/20"}, weight=2.0),
    ]

    print("\nAdding connections:")
    for conn in connections:
        topo.add_connection(conn)
        src = topo.devices[conn.source_id].hostname
        dst = topo.devices[conn.target_id].hostname
        print(f"  ✓ {src} --[{conn.connection_type.value}]--> {dst}")

    print(f"\nTopology Statistics:")
    print(f"  Devices: {len(topo.devices)}")
    print(f"  Connections: {len(topo.connections)}")
    print(f"  Graph Nodes: {topo.graph.number_of_nodes()}")
    print(f"  Graph Edges: {topo.graph.number_of_edges()}")
    print(f"  Device Embeddings: {len(topo.device_embeddings)}")


def example_2_graph_retrieval():
    """Example 2: Graph-based retrieval and queries"""
    print("\n" + "=" * 80)
    print("Example 2: Graph-Based Retrieval")
    print("=" * 80)

    topo = NetworkTopologyGraph()

    # Build simple topology
    devices = [
        NetworkDevice("r1", DeviceType.ROUTER, "router-01", "10.0.0.1", {"role": "core"}),
        NetworkDevice("sw1", DeviceType.SWITCH, "switch-01", "10.0.1.1", {"role": "access"}),
        NetworkDevice("srv1", DeviceType.SERVER, "server-01", "10.1.0.10", {"service": "web"}),
    ]

    connections = [
        NetworkConnection("r1", "sw1", ConnectionType.L3_LINK, {}),
        NetworkConnection("sw1", "srv1", ConnectionType.L2_ACCESS, {"vlan": "100"}),
    ]

    for device in devices:
        topo.add_device(device)
    for conn in connections:
        topo.add_connection(conn)

    # Vector search
    print("\nVector Search: 'core routing device'")
    results = topo.vector_search("core routing device", top_k=3)

    for i, (device_id, score) in enumerate(results, 1):
        device = topo.devices[device_id]
        print(f"\n{i}. {device.hostname} ({device.device_type.value})")
        print(f"   IP: {device.management_ip}")
        print(f"   Similarity Score: {score:.3f}")
        print(f"   Description: {topo.device_descriptions[device_id]}")


def example_3_impact_analysis():
    """Example 3: Impact analysis for device failure"""
    print("\n" + "=" * 80)
    print("Example 3: Impact Analysis")
    print("=" * 80)

    topo = NetworkTopologyGraph()

    # Build test topology
    devices = [
        NetworkDevice("core-r1", DeviceType.ROUTER, "core-router", "10.0.0.1", {"role": "core"}),
        NetworkDevice("dist-sw1", DeviceType.SWITCH, "dist-switch-01", "10.0.1.1", {"role": "dist"}),
        NetworkDevice("dist-sw2", DeviceType.SWITCH, "dist-switch-02", "10.0.1.2", {"role": "dist"}),
        NetworkDevice("acc-sw1", DeviceType.SWITCH, "access-switch-01", "10.0.2.1", {"role": "access"}),
        NetworkDevice("web1", DeviceType.SERVER, "web-01", "10.1.0.10", {"service": "web"}),
        NetworkDevice("web2", DeviceType.SERVER, "web-02", "10.1.0.11", {"service": "web"}),
    ]

    connections = [
        NetworkConnection("core-r1", "dist-sw1", ConnectionType.L3_LINK, {}),
        NetworkConnection("core-r1", "dist-sw2", ConnectionType.L3_LINK, {}),
        NetworkConnection("dist-sw1", "acc-sw1", ConnectionType.L2_TRUNK, {}),
        NetworkConnection("acc-sw1", "web1", ConnectionType.L2_ACCESS, {}),
        NetworkConnection("acc-sw1", "web2", ConnectionType.L2_ACCESS, {}),
    ]

    for device in devices:
        topo.add_device(device)
    for conn in connections:
        topo.add_connection(conn)

    # Analyze impact
    print("\nAnalyzing impact of: core-router failure")
    impact = topo.analyze_impact("core-r1")

    print(f"\n{'='*80}")
    print(f"IMPACT ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Failed Device: {impact['device']} ({impact['device_type']})")
    print(f"Severity: {impact['severity']}")
    print(f"Blast Radius: {impact['blast_radius']} devices")
    print(f"Downstream Count: {impact['actual_downstream_count']}")

    print(f"\nAffected Services:")
    for service in impact['affected_services']:
        print(f"  - {service}")

    print(f"\nDownstream Devices:")
    for device in impact['downstream_devices'][:5]:
        print(f"  - {device}")

    print(f"\nMitigation Options:")
    for i, option in enumerate(impact['mitigation_options'], 1):
        print(f"  {i}. {option}")

    print(f"\nRecovery Time: {impact['recovery_time_estimate']}")


def example_4_path_finding():
    """Example 4: Path finding and analysis"""
    print("\n" + "=" * 80)
    print("Example 4: Path Finding")
    print("=" * 80)

    topo = NetworkTopologyGraph()

    # Build topology with multiple paths
    devices = [
        NetworkDevice("r1", DeviceType.ROUTER, "router-01", "10.0.0.1", {"location": "DC1"}),
        NetworkDevice("r2", DeviceType.ROUTER, "router-02", "10.0.0.2", {"location": "DC2"}),
        NetworkDevice("sw1", DeviceType.SWITCH, "switch-01", "10.0.1.1", {"location": "DC2"}),
        NetworkDevice("srv1", DeviceType.SERVER, "server-01", "10.1.0.10", {"service": "database"}),
    ]

    connections = [
        NetworkConnection("r1", "r2", ConnectionType.BGP_PEER, {"as_path": "65001 65002"}),
        NetworkConnection("r2", "sw1", ConnectionType.OSPF_NEIGHBOR, {"area": "0"}),
        NetworkConnection("sw1", "srv1", ConnectionType.L2_ACCESS, {"vlan": "100"}),
    ]

    for device in devices:
        topo.add_device(device)
    for conn in connections:
        topo.add_connection(conn)

    # Find path
    print("\nFinding path: router-01 → server-01")
    path_result = topo.find_path("r1", "srv1")

    if path_result.get("path_exists"):
        print(f"\n✓ Path Found!")
        print(f"  Hops: {path_result['hop_count']}")
        print(f"  Path: {' → '.join(path_result['path'])}")
        print(f"  Path Type: {path_result['path_type']}")
        print(f"  Technologies: {', '.join(path_result['technologies'])}")
        print(f"  Latency Estimate: {path_result['latency_estimate']}")

        print(f"\nPath Description:")
        print(f"  {path_result['path_description']}")

        print(f"\nCritical Hops:")
        for hop in path_result['critical_hops']:
            print(f"  - {hop}")

        print(f"\nDetailed Hop Analysis:")
        for hop_detail in path_result['detailed_hops']:
            print(f"  Hop {hop_detail['hop']}: {hop_detail['from']} → {hop_detail['to']}")
            print(f"    Connection: {hop_detail['connection_type']}")
    else:
        print(f"\n✗ No path found: {path_result.get('error')}")


def example_5_hybrid_rag_topology():
    """Example 5: Hybrid RAG for topology queries"""
    print("\n" + "=" * 80)
    print("Example 5: Hybrid RAG Topology Queries")
    print("=" * 80)

    topo = NetworkTopologyGraph()

    # Build comprehensive topology
    devices = [
        NetworkDevice("core-r1", DeviceType.ROUTER, "core-router-01", "10.0.0.1",
                     {"model": "Cisco ASR9010", "role": "core", "redundant": "yes"},
                     protocols=["BGP", "OSPF"]),
        NetworkDevice("core-r2", DeviceType.ROUTER, "core-router-02", "10.0.0.2",
                     {"model": "Cisco ASR9010", "role": "core", "redundant": "yes"},
                     protocols=["BGP", "OSPF"]),
        NetworkDevice("dist-sw1", DeviceType.SWITCH, "dist-switch-01", "10.0.1.1",
                     {"model": "Arista 7280", "role": "distribution"},
                     protocols=["OSPF", "LACP"]),
        NetworkDevice("fw1", DeviceType.FIREWALL, "edge-firewall", "10.0.2.1",
                     {"model": "Palo Alto PA-5220", "role": "edge"}),
        NetworkDevice("web1", DeviceType.SERVER, "web-cluster-01", "10.1.0.10",
                     {"service": "nginx", "cluster": "web-tier"}),
    ]

    connections = [
        NetworkConnection("core-r1", "core-r2", ConnectionType.REDUNDANT_PAIR, {}),
        NetworkConnection("core-r1", "dist-sw1", ConnectionType.L3_LINK, {}),
        NetworkConnection("core-r2", "dist-sw1", ConnectionType.L3_LINK, {}),
        NetworkConnection("core-r1", "fw1", ConnectionType.L3_LINK, {}),
        NetworkConnection("dist-sw1", "web1", ConnectionType.L2_ACCESS, {}),
    ]

    print("\nBuilding topology...")
    for device in devices:
        topo.add_device(device)
        print(f"  ✓ {device.hostname}")

    for conn in connections:
        topo.add_connection(conn)

    # Query using hybrid RAG
    queries = [
        "Which devices provide redundancy in the core?",
        "How does traffic reach the web servers from the firewall?",
        "What would happen if the distribution switch fails?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")

        result = topo.hybrid_rag_query(query)

        print(f"\nAnswer:")
        print(f"  {result['answer']}")

        print(f"\nConfidence: {result['confidence']}")

        print(f"\nGraph Insights:")
        for insight in result['graph_insights']:
            print(f"  - {insight}")

        print(f"\nVector Insights:")
        for insight in result['vector_insights']:
            print(f"  - {insight}")

        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec}")

        print(f"\nQuery Analysis:")
        print(f"  Type: {result['query_analysis']['type']}")
        print(f"  Strategy: {result['query_analysis']['strategy']}")
        print(f"  Reasoning: {result['query_analysis']['reasoning']}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 37: Graph RAG for Network Topology")
    print("Production-ready hybrid RAG with graph traversal and vector search")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Check API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set")
        exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        exit(1)

    # Run all examples
    example_1_build_topology_graph()
    example_2_graph_retrieval()
    example_3_impact_analysis()
    example_4_path_finding()
    example_5_hybrid_rag_topology()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nKey Capabilities:")
    print("  ✓ Network topology as knowledge graph")
    print("  ✓ Graph-based retrieval for structural queries")
    print("  ✓ Vector search for semantic similarity")
    print("  ✓ Hybrid RAG combining both approaches")
    print("  ✓ Impact analysis with blast radius calculation")
    print("  ✓ Path finding with AI-powered analysis")
    print("  ✓ Dependency mapping and SPOF identification")
    print("=" * 80)
