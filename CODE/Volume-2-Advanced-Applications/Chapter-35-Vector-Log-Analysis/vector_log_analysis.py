"""
Chapter 35: Vector-Based Log Analysis
Use embeddings to find similar logs, detect anomalies, cluster patterns
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


# ============================================================================
# Data Models
# ============================================================================

class LogSeverity(str, Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Single log entry"""
    log_id: str
    timestamp: datetime
    severity: LogSeverity
    source: str
    message: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


@dataclass
class LogCluster:
    """Cluster of similar logs"""
    cluster_id: int
    log_ids: List[str]
    representative_message: str
    pattern: str
    count: int
    severity_distribution: Dict[str, int]


@dataclass
class SimilarLog:
    """Similar log result"""
    log_id: str
    message: str
    similarity_score: float
    timestamp: datetime
    source: str


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    log_id: str
    message: str
    anomaly_score: float
    is_anomaly: bool
    explanation: str


# ============================================================================
# Pydantic Models for LLM Outputs
# ============================================================================

class LogPattern(BaseModel):
    """LLM-identified log pattern"""
    pattern_name: str = Field(description="Short name for the pattern (e.g., 'SSH Failed Login')")
    pattern_template: str = Field(description="Template with variables (e.g., 'Failed password for USER from IP')")
    severity: str = Field(description="INFO, WARNING, ERROR, or CRITICAL")
    frequency: str = Field(description="RARE, OCCASIONAL, FREQUENT, or CONSTANT")
    actionable: bool = Field(description="True if requires action")


class ClusterAnalysis(BaseModel):
    """LLM analysis of log cluster"""
    cluster_summary: str = Field(description="What this cluster represents (1-2 sentences)")
    common_pattern: str = Field(description="Common pattern across logs")
    root_cause: str = Field(description="Likely root cause (1 sentence)")
    severity_assessment: str = Field(description="BENIGN, ATTENTION_NEEDED, URGENT, or CRITICAL")
    recommended_action: str = Field(description="What to do about it (1-2 sentences)")


class AnomalyAnalysis(BaseModel):
    """LLM analysis of anomaly"""
    is_genuine_anomaly: bool = Field(description="True if truly anomalous")
    anomaly_type: str = Field(description="NEW_ERROR, RARE_EVENT, SECURITY_CONCERN, or FALSE_POSITIVE")
    explanation: str = Field(description="Why this is anomalous (1-2 sentences)")
    severity: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    recommended_action: str = Field(description="What to investigate (1-2 sentences)")


class LogComparisonResult(BaseModel):
    """LLM comparison of two logs"""
    are_related: bool = Field(description="True if logs are related")
    relationship: str = Field(description="How they're related (e.g., 'same error, different devices')")
    differences: List[str] = Field(description="Key differences (2-3 items)")
    combined_insight: str = Field(description="Insight from comparing both (1-2 sentences)")


# ============================================================================
# Vector Log Analyzer
# ============================================================================

class VectorLogAnalyzer:
    """Analyze logs using vector embeddings"""

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

        self.logs: Dict[str, LogEntry] = {}
        self.embedding_matrix: Optional[np.ndarray] = None

    def add_log(self, log_id: str, timestamp: datetime, severity: LogSeverity,
               source: str, message: str):
        """
        Add log entry and compute embedding

        Args:
            log_id: Unique log identifier
            timestamp: Log timestamp
            severity: Log severity
            source: Log source (device, service)
            message: Log message
        """
        # Compute embedding
        embedding = np.array(self.embeddings.embed_query(message))

        log = LogEntry(
            log_id=log_id,
            timestamp=timestamp,
            severity=severity,
            source=source,
            message=message,
            embedding=embedding
        )

        self.logs[log_id] = log

        # Rebuild embedding matrix
        self._rebuild_embedding_matrix()

    def _rebuild_embedding_matrix(self):
        """Rebuild embedding matrix from all logs"""
        embeddings = [log.embedding for log in self.logs.values() if log.embedding is not None]
        if embeddings:
            self.embedding_matrix = np.vstack(embeddings)

    def find_similar_logs(self, query_log_id: str, top_k: int = 5,
                         min_similarity: float = 0.7) -> List[SimilarLog]:
        """
        Find logs similar to query log

        Args:
            query_log_id: Log to find similar logs for
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar logs
        """
        if query_log_id not in self.logs:
            return []

        query_log = self.logs[query_log_id]
        if query_log.embedding is None:
            return []

        # Compute similarities
        similarities = []
        for log_id, log in self.logs.items():
            if log_id == query_log_id or log.embedding is None:
                continue

            sim = cosine_similarity(
                query_log.embedding.reshape(1, -1),
                log.embedding.reshape(1, -1)
            )[0][0]

            if sim >= min_similarity:
                similarities.append((log_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for log_id, sim in similarities[:top_k]:
            log = self.logs[log_id]
            results.append(SimilarLog(
                log_id=log_id,
                message=log.message,
                similarity_score=float(sim),
                timestamp=log.timestamp,
                source=log.source
            ))

        return results

    def cluster_logs(self, eps: float = 0.3, min_samples: int = 2) -> List[LogCluster]:
        """
        Cluster logs using DBSCAN

        Args:
            eps: DBSCAN epsilon (distance threshold)
            min_samples: Minimum samples per cluster

        Returns:
            List of log clusters
        """
        if self.embedding_matrix is None or len(self.logs) < min_samples:
            return []

        # Run DBSCAN on embeddings
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(self.embedding_matrix)

        # Assign cluster IDs to logs
        log_list = list(self.logs.values())
        for i, log in enumerate(log_list):
            log.cluster_id = int(cluster_labels[i])

        # Build cluster objects
        clusters: Dict[int, List[str]] = {}
        for log in log_list:
            if log.cluster_id is not None and log.cluster_id != -1:  # -1 is noise
                if log.cluster_id not in clusters:
                    clusters[log.cluster_id] = []
                clusters[log.cluster_id].append(log.log_id)

        # Create cluster summaries
        cluster_list = []
        for cluster_id, log_ids in clusters.items():
            # Get logs in cluster
            cluster_logs = [self.logs[lid] for lid in log_ids]

            # Representative message (first log)
            representative = cluster_logs[0].message

            # Severity distribution
            severity_dist = {}
            for log in cluster_logs:
                severity_dist[log.severity.value] = severity_dist.get(log.severity.value, 0) + 1

            cluster_list.append(LogCluster(
                cluster_id=cluster_id,
                log_ids=log_ids,
                representative_message=representative,
                pattern="",  # Will be filled by LLM analysis
                count=len(log_ids),
                severity_distribution=severity_dist
            ))

        return cluster_list

    def analyze_cluster(self, cluster: LogCluster) -> Dict:
        """
        Analyze log cluster with LLM

        Args:
            cluster: Log cluster to analyze

        Returns:
            Cluster analysis
        """
        # Get sample logs from cluster
        sample_logs = [self.logs[lid].message for lid in cluster.log_ids[:5]]

        parser = PydanticOutputParser(pydantic_object=ClusterAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a log analysis expert.
Analyze this cluster of similar logs and identify the pattern.

{format_instructions}"""),
            ("human", """Log Cluster (ID: {cluster_id})
Count: {count} logs
Severity Distribution: {severity_dist}

Sample Logs:
{sample_logs}

Analyze this cluster.""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "cluster_id": cluster.cluster_id,
            "count": cluster.count,
            "severity_dist": str(cluster.severity_distribution),
            "sample_logs": "\n".join([f"- {log}" for log in sample_logs])
        })

        return {
            "cluster_id": cluster.cluster_id,
            "cluster_summary": analysis.cluster_summary,
            "common_pattern": analysis.common_pattern,
            "root_cause": analysis.root_cause,
            "severity_assessment": analysis.severity_assessment,
            "recommended_action": analysis.recommended_action,
            "log_count": cluster.count
        }

    def detect_anomalies(self, percentile: float = 95.0) -> List[AnomalyDetection]:
        """
        Detect anomalous logs using isolation from cluster

        Args:
            percentile: Percentile threshold for anomaly

        Returns:
            List of anomalies
        """
        if len(self.logs) < 10:
            return []

        anomalies = []

        # For each log, compute average similarity to other logs
        for log_id, log in self.logs.items():
            if log.embedding is None:
                continue

            similarities = []
            for other_id, other_log in self.logs.items():
                if other_id != log_id and other_log.embedding is not None:
                    sim = cosine_similarity(
                        log.embedding.reshape(1, -1),
                        other_log.embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

            # Average similarity
            avg_sim = np.mean(similarities) if similarities else 0.0

            # Anomaly score (inverse of similarity)
            anomaly_score = 1.0 - avg_sim

            anomalies.append((log_id, anomaly_score))

        # Sort by anomaly score
        anomalies.sort(key=lambda x: x[1], reverse=True)

        # Get threshold at percentile
        scores = [score for _, score in anomalies]
        threshold = np.percentile(scores, percentile)

        # Build anomaly detections
        results = []
        for log_id, score in anomalies:
            if score >= threshold:
                log = self.logs[log_id]
                results.append(AnomalyDetection(
                    log_id=log_id,
                    message=log.message,
                    anomaly_score=float(score),
                    is_anomaly=True,
                    explanation=f"Isolated from main log patterns (score: {score:.3f})"
                ))

        return results

    def analyze_anomaly(self, anomaly: AnomalyDetection) -> Dict:
        """
        Analyze anomaly with LLM

        Args:
            anomaly: Detected anomaly

        Returns:
            Anomaly analysis
        """
        parser = PydanticOutputParser(pydantic_object=AnomalyAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a log anomaly analyst.
Determine if this is a genuine anomaly and assess severity.

{format_instructions}"""),
            ("human", """Anomalous Log:
Message: {message}
Anomaly Score: {score:.3f}

Is this a genuine anomaly? What should we do about it?""")
        ])

        chain = prompt | self.llm | parser

        analysis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "message": anomaly.message,
            "score": anomaly.anomaly_score
        })

        return {
            "log_id": anomaly.log_id,
            "message": anomaly.message,
            "is_genuine_anomaly": analysis.is_genuine_anomaly,
            "anomaly_type": analysis.anomaly_type,
            "explanation": analysis.explanation,
            "severity": analysis.severity,
            "recommended_action": analysis.recommended_action
        }

    def compare_logs(self, log_id_1: str, log_id_2: str) -> Dict:
        """
        Compare two logs with LLM

        Args:
            log_id_1: First log
            log_id_2: Second log

        Returns:
            Comparison result
        """
        if log_id_1 not in self.logs or log_id_2 not in self.logs:
            return {"error": "Log not found"}

        log1 = self.logs[log_id_1]
        log2 = self.logs[log_id_2]

        # Compute similarity
        if log1.embedding is not None and log2.embedding is not None:
            similarity = cosine_similarity(
                log1.embedding.reshape(1, -1),
                log2.embedding.reshape(1, -1)
            )[0][0]
        else:
            similarity = 0.0

        parser = PydanticOutputParser(pydantic_object=LogComparisonResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are comparing two log entries.
Identify relationships and differences.

{format_instructions}"""),
            ("human", """Log 1:
Source: {source1}
Time: {time1}
Message: {message1}

Log 2:
Source: {source2}
Time: {time2}
Message: {message2}

Vector Similarity: {similarity:.3f}

Compare these logs.""")
        ])

        chain = prompt | self.llm | parser

        comparison = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "source1": log1.source,
            "time1": log1.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "message1": log1.message,
            "source2": log2.source,
            "time2": log2.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "message2": log2.message,
            "similarity": similarity
        })

        return {
            "log_id_1": log_id_1,
            "log_id_2": log_id_2,
            "vector_similarity": float(similarity),
            "are_related": comparison.are_related,
            "relationship": comparison.relationship,
            "differences": comparison.differences,
            "combined_insight": comparison.combined_insight
        }


# ============================================================================
# Examples
# ============================================================================

def example_1_basic_vector_log_analysis():
    """Example 1: Basic vector log analysis"""
    print("=" * 80)
    print("Example 1: Basic Vector Log Analysis")
    print("=" * 80)

    analyzer = VectorLogAnalyzer()

    # Add sample logs
    logs = [
        ("log001", datetime.now(), LogSeverity.ERROR, "router-01", "BGP session down with peer 10.0.0.2"),
        ("log002", datetime.now(), LogSeverity.ERROR, "router-02", "BGP neighbor 10.0.0.1 connection lost"),
        ("log003", datetime.now(), LogSeverity.WARNING, "switch-01", "High CPU usage detected: 85%"),
        ("log004", datetime.now(), LogSeverity.INFO, "firewall-01", "Traffic rate: 1.2 Gbps"),
    ]

    print("\nAdding logs to analyzer...")
    for log_id, timestamp, severity, source, message in logs:
        analyzer.add_log(log_id, timestamp, severity, source, message)
        print(f"  ✓ {log_id}: {message[:50]}...")

    print(f"\nTotal logs: {len(analyzer.logs)}")
    print(f"Embedding matrix shape: {analyzer.embedding_matrix.shape if analyzer.embedding_matrix is not None else 'None'}")


def example_2_find_similar_logs():
    """Example 2: Find similar logs"""
    print("\n" + "=" * 80)
    print("Example 2: Find Similar Logs")
    print("=" * 80)

    analyzer = VectorLogAnalyzer()

    # Add logs
    logs = [
        ("log001", "router-01", "Failed to establish BGP session with 10.0.0.2"),
        ("log002", "router-02", "BGP session failed to peer 10.0.0.1"),
        ("log003", "router-03", "BGP neighbor down: 10.0.0.5"),
        ("log004", "switch-01", "Interface GigE0/1 went down"),
        ("log005", "firewall-01", "Security policy updated"),
    ]

    for log_id, source, message in logs:
        analyzer.add_log(log_id, datetime.now(), LogSeverity.ERROR, source, message)

    # Find similar logs
    query_log = "log001"
    print(f"\nQuery Log: {analyzer.logs[query_log].message}")
    print(f"\nFinding similar logs...\n")

    similar = analyzer.find_similar_logs(query_log, top_k=3, min_similarity=0.6)

    print(f"Found {len(similar)} similar logs:\n")
    for i, sim_log in enumerate(similar, 1):
        print(f"{i}. {sim_log.log_id} (similarity: {sim_log.similarity_score:.3f})")
        print(f"   Source: {sim_log.source}")
        print(f"   Message: {sim_log.message}")
        print()


def example_3_cluster_logs():
    """Example 3: Cluster similar logs"""
    print("\n" + "=" * 80)
    print("Example 3: Cluster Similar Logs")
    print("=" * 80)

    analyzer = VectorLogAnalyzer()

    # Add logs with patterns
    logs = [
        # BGP failures
        ("log001", "router-01", "BGP session down"),
        ("log002", "router-02", "BGP neighbor unreachable"),
        ("log003", "router-03", "BGP peer connection lost"),
        # Interface failures
        ("log004", "switch-01", "Interface down: GigE0/1"),
        ("log005", "switch-02", "Port GigE0/2 link down"),
        ("log006", "switch-03", "Interface GigE0/3 state down"),
        # High CPU
        ("log007", "router-01", "High CPU: 90%"),
        ("log008", "router-04", "CPU utilization critical: 95%"),
    ]

    for log_id, source, message in logs:
        analyzer.add_log(log_id, datetime.now(), LogSeverity.ERROR, source, message)

    print(f"\nAdded {len(logs)} logs")
    print("\nClustering logs...\n")

    clusters = analyzer.cluster_logs(eps=0.4, min_samples=2)

    print(f"Found {len(clusters)} clusters:\n")
    for cluster in clusters:
        print(f"Cluster {cluster.cluster_id}:")
        print(f"  Count: {cluster.count} logs")
        print(f"  Representative: {cluster.representative_message}")
        print(f"  Logs: {', '.join(cluster.log_ids)}")
        print()


def example_4_detect_anomalies():
    """Example 4: Detect anomalous logs"""
    print("\n" + "=" * 80)
    print("Example 4: Detect Anomalous Logs")
    print("=" * 80)

    analyzer = VectorLogAnalyzer()

    # Add normal logs + one anomaly
    logs = [
        ("log001", "router-01", "BGP session established"),
        ("log002", "router-02", "BGP neighbor up"),
        ("log003", "router-03", "BGP peer connected"),
        ("log004", "router-01", "BGP session stable"),
        ("log005", "router-02", "BGP neighbor active"),
        # Anomaly
        ("log006", "router-01", "CRITICAL: Memory corruption detected in kernel space"),
    ]

    for log_id, source, message in logs:
        analyzer.add_log(log_id, datetime.now(), LogSeverity.INFO, source, message)

    print(f"\nAdded {len(logs)} logs")
    print("\nDetecting anomalies...\n")

    anomalies = analyzer.detect_anomalies(percentile=90.0)

    print(f"Found {len(anomalies)} anomalies:\n")
    for anomaly in anomalies:
        print(f"Anomaly: {anomaly.log_id}")
        print(f"  Score: {anomaly.anomaly_score:.3f}")
        print(f"  Message: {anomaly.message}")
        print(f"  Explanation: {anomaly.explanation}")
        print()


def example_5_full_log_analysis_pipeline():
    """Example 5: Complete log analysis pipeline"""
    print("\n" + "=" * 80)
    print("Example 5: Complete Log Analysis Pipeline")
    print("=" * 80)

    analyzer = VectorLogAnalyzer()

    # Add diverse logs
    logs = [
        ("log001", "router-01", "BGP session down with peer 10.0.0.2"),
        ("log002", "router-02", "BGP neighbor connection lost"),
        ("log003", "switch-01", "Interface GigE0/1 down"),
        ("log004", "switch-02", "Port link failure on GigE0/2"),
        ("log005", "firewall-01", "Security policy violation: port 22 from unknown source"),
        ("log006", "router-01", "OSPF adjacency established"),
        ("log007", "router-03", "OSPF neighbor up"),
    ]

    print("\n1. Adding logs...")
    for log_id, source, message in logs:
        analyzer.add_log(log_id, datetime.now(), LogSeverity.ERROR, source, message)
    print(f"   Total logs: {len(analyzer.logs)}")

    print("\n2. Clustering logs...")
    clusters = analyzer.cluster_logs(eps=0.4, min_samples=2)
    print(f"   Clusters found: {len(clusters)}")

    if clusters:
        print(f"\n3. Analyzing first cluster...")
        cluster_analysis = analyzer.analyze_cluster(clusters[0])
        print(f"   Cluster ID: {cluster_analysis['cluster_id']}")
        print(f"   Summary: {cluster_analysis['cluster_summary']}")
        print(f"   Root Cause: {cluster_analysis['root_cause']}")
        print(f"   Severity: {cluster_analysis['severity_assessment']}")
        print(f"   Action: {cluster_analysis['recommended_action']}")

    print(f"\n4. Detecting anomalies...")
    anomalies = analyzer.detect_anomalies(percentile=85.0)
    print(f"   Anomalies found: {len(anomalies)}")

    if anomalies:
        print(f"\n5. Analyzing first anomaly...")
        anomaly_analysis = analyzer.analyze_anomaly(anomalies[0])
        print(f"   Genuine Anomaly: {'✓' if anomaly_analysis['is_genuine_anomaly'] else '✗'}")
        print(f"   Type: {anomaly_analysis['anomaly_type']}")
        print(f"   Severity: {anomaly_analysis['severity']}")
        print(f"   Explanation: {anomaly_analysis['explanation']}")
        print(f"   Action: {anomaly_analysis['recommended_action']}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 35: Vector-Based Log Analysis")
    print("Use embeddings for log clustering and anomaly detection")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_basic_vector_log_analysis()
    example_2_find_similar_logs()
    example_3_cluster_logs()
    example_4_detect_anomalies()
    example_5_full_log_analysis_pipeline()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
