#!/usr/bin/env python3
"""
Semantic Search for Network Documentation

Build a semantic search engine that understands intent, not just keywords.

From: AI for Networking Engineers - Volume 2, Chapter 17
Author: Eduard Dulharu

Usage:
    python semantic_search.py
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


class NetworkDocSearch:
    """
    Semantic search engine for network documentation.

    Understands intent, finds relevant docs even with different terminology.
    """

    def __init__(self, persist_dir="./network_search_db"):
        """Initialize the search engine."""
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index network documentation for search.

        Args:
            documents: List of dicts with 'content' and 'metadata'
        """
        print(f"\nIndexing {len(documents)} documents...")

        # Split documents into chunks
        texts = []
        metadatas = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    **doc.get('metadata', {}),
                    'chunk': i,
                    'total_chunks': len(chunks)
                })

        # Create vector store
        self.vectorstore = Chroma.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_dir
        )

        print(f"✓ Indexed {len(texts)} chunks")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant documentation.

        Args:
            query: Search query (can be natural language)
            k: Number of results to return

        Returns:
            List of (content, score, metadata) tuples
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Search with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [(doc.page_content, score, doc.metadata) for doc, score in results]

    def search_by_category(self, query: str, category: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search within a specific category.

        Args:
            query: Search query
            category: Category to filter by (e.g., 'BGP', 'OSPF', 'VLAN')
            k: Number of results

        Returns:
            List of (content, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("No documents indexed.")

        # Search with metadata filter
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter={"category": category}
        )

        return [(doc.page_content, score) for doc, score in results]


def create_sample_network_docs():
    """Create sample network documentation."""
    return [
        {
            "content": """
BGP Route Filtering Best Practices

Use prefix lists to control BGP advertisements:

1. Prefix Lists
   - More efficient than access lists
   - Can specify exact prefix length or range
   - Easy to read and maintain

Example:
ip prefix-list ALLOW-DEFAULTS seq 5 permit 0.0.0.0/0
ip prefix-list CUSTOMER-ROUTES seq 10 permit 10.0.0.0/8 le 24

2. AS Path Filtering
   - Filter based on AS path
   - Use regular expressions for complex patterns
   - Prepend AS for traffic engineering

Example:
ip as-path access-list 10 permit ^65001_

3. Route Maps
   - Combine multiple match conditions
   - Set attributes (MED, local pref, community)
   - Most flexible but complex

Apply to neighbors:
router bgp 65001
  neighbor 203.0.113.1 prefix-list CUSTOMER-ROUTES out
            """,
            "metadata": {"category": "BGP", "type": "guide"}
        },
        {
            "content": """
OSPF Area Design Guidelines

Choosing the right OSPF area structure:

Area Types:
1. Backbone (Area 0)
   - All areas must connect to Area 0
   - Runs full OSPF (LSA types 1-5)
   - Should be stable and reliable

2. Standard Area
   - Receives all LSAs
   - Full routing table
   - Most flexible

3. Stub Area
   - No external routes (Type 5 LSAs blocked)
   - Reduces memory usage
   - Use default route for external traffic

4. Totally Stubby (Cisco)
   - No external or summary routes
   - Only default route and intra-area routes
   - Smallest routing table

5. NSSA (Not-So-Stubby Area)
   - Like stub but allows external routes from area
   - Type 7 LSAs converted to Type 5 at ABR

Recommendations:
- Limit areas to 50-100 routers
- Use stub areas at edges
- Keep Area 0 simple and reliable
            """,
            "metadata": {"category": "OSPF", "type": "guide"}
        },
        {
            "content": """
VLAN Best Practices and Standards

Enterprise VLAN numbering scheme:

Standard Ranges:
- 1-99: Infrastructure and management
  - VLAN 1: Native VLAN (don't use for data)
  - VLAN 10: Management
  - VLAN 20: Network devices

- 100-199: User VLANs
  - VLAN 100: Corporate users
  - VLAN 110: Guest WiFi
  - VLAN 120: Contractors

- 200-299: Voice
  - VLAN 200: IP phones
  - QoS priority tagging

- 300-399: Servers
  - VLAN 300: Application servers
  - VLAN 310: Database servers
  - VLAN 320: Web servers

- 900-999: Special purpose
  - VLAN 999: Quarantine/blackhole

Trunk Configuration:
- Always set native VLAN to unused VLAN
- Explicitly allow only required VLANs
- Use VTP transparent or off (avoid VTP)

Security:
- Enable DHCP snooping
- Enable ARP inspection
- Enable port security on access ports
- Disable DTP (no switchport mode dynamic)
            """,
            "metadata": {"category": "VLAN", "type": "standards"}
        },
        {
            "content": """
ACL Performance and Optimization

Access Control List best practices:

Ordering Rules:
1. Most specific rules first
2. Most frequently matched rules first
3. Deny rules before permit rules (usually)
4. Explicit deny all at end

Example - Optimized Order:
ip access-list extended OPTIMIZED
  ! Permit critical services first
  permit tcp any host 10.1.1.10 eq 443
  permit tcp any host 10.1.1.10 eq 80
  ! Deny dangerous traffic
  deny ip any host 10.1.1.100
  ! Permit rest of subnet
  permit ip any 10.1.1.0 0.0.0.255
  ! Implicit deny all

Hardware ACLs vs Software:
- TCAM (hardware): Line rate, limited rules
- Software: Flexible, slower, more rules

Performance Tips:
1. Use object groups for similar hosts/services
2. Combine ACLs where possible
3. Remove unused ACL entries
4. Use established keyword for return traffic
5. Consider reflexive ACLs for stateful filtering

Troubleshooting:
- show access-lists (view ACL + hit counts)
- show ip interface (check applied ACLs)
- log keyword adds syslog entries
            """,
            "metadata": {"category": "Security", "type": "guide"}
        },
        {
            "content": """
STP Convergence and Tuning

Spanning Tree Protocol optimization:

Convergence Times:
- 802.1D (STP): 30-50 seconds
- 802.1w (RSTP): 1-2 seconds
- 802.1s (MST): 1-2 seconds

Improving Convergence:
1. Use RSTP/MST instead of PVST+
2. Enable PortFast on access ports
3. Enable BPDUGuard with PortFast
4. Use UplinkFast for access switches
5. Tune timers (hello 2s, max age 20s, forward delay 15s)

Root Bridge Selection:
- Manual selection: spanning-tree vlan X root primary
- Or set priority: spanning-tree vlan X priority 4096
- Primary: 24576, Secondary: 28672

Port Costs (802.1w):
- 10 Mbps: 2,000,000
- 100 Mbps: 200,000
- 1 Gbps: 20,000
- 10 Gbps: 2,000

Protection Features:
- BPDUGuard: Shuts down port receiving BPDU
- BPDUFilter: Stops sending/receiving BPDUs
- RootGuard: Prevents port becoming root port
- LoopGuard: Prevents loops from unidirectional links

Verification:
- show spanning-tree vlan X
- show spanning-tree interface X detail
- show spanning-tree summary
            """,
            "metadata": {"category": "STP", "type": "guide"}
        }
    ]


def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("="*60)
    print("Semantic Search for Network Documentation")
    print("="*60)

    # Create search engine
    search = NetworkDocSearch()

    # Index sample docs
    docs = create_sample_network_docs()
    search.index_documents(docs)

    # Example queries showing semantic understanding
    queries = [
        ("How do I filter BGP routes?", "BGP filtering"),
        ("What's the best way to segment my network?", "VLAN design"),
        ("Spanning tree is taking too long to recover", "STP convergence"),
        ("How do I make ACLs faster?", "ACL performance"),
        ("Should I use stub areas?", "OSPF area types")
    ]

    print("\n" + "="*60)
    print("Semantic Search Examples")
    print("="*60)

    for query, expected in queries:
        print(f"\n{'-'*60}")
        print(f"Query: \"{query}\"")
        print(f"Expected topic: {expected}")
        print(f"{'-'*60}")

        results = search.search(query, k=2)

        for i, (content, score, metadata) in enumerate(results, 1):
            print(f"\nResult {i} (score: {score:.3f}):")
            print(f"  Category: {metadata.get('category', 'N/A')}")
            print(f"  Type: {metadata.get('type', 'N/A')}")
            print(f"  Preview: {content[:150]}...")

    # Demo category filtering
    print("\n\n" + "="*60)
    print("Category-Specific Search")
    print("="*60)

    category_query = "optimization tips"
    category = "Security"

    print(f"\nQuery: \"{category_query}\" in category \"{category}\"")
    results = search.search_by_category(category_query, category, k=1)

    for content, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"Content:\n{content[:300]}...")


def demo_keyword_vs_semantic():
    """Show difference between keyword and semantic search."""
    print("\n\n" + "="*60)
    print("Keyword vs Semantic Search Comparison")
    print("="*60)

    examples = [
        {
            "user_question": "My network is slow during failover",
            "keyword_match": "No exact match for 'slow failover'",
            "semantic_match": "Finds: STP convergence optimization (different words, same intent)"
        },
        {
            "user_question": "How to segment traffic between departments?",
            "keyword_match": "No exact match for 'segment traffic'",
            "semantic_match": "Finds: VLAN best practices (understands segmentation = VLANs)"
        },
        {
            "user_question": "Control what routes we advertise to ISP",
            "keyword_match": "No exact match for 'control routes advertise'",
            "semantic_match": "Finds: BGP route filtering (understands intent)"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. User asks: \"{example['user_question']}\"")
        print(f"   Keyword search: {example['keyword_match']}")
        print(f"   Semantic search: {example['semantic_match']}")


def main():
    """Run all examples."""
    try:
        demo_semantic_search()
        demo_keyword_vs_semantic()

        print("\n\n" + "="*60)
        print("✓ Semantic search demo completed!")
        print("="*60)
        print("\nKey Benefits:")
        print("1. Finds relevant docs even with different terminology")
        print("2. Understands intent, not just keywords")
        print("3. Works across different documentation styles")
        print("4. No need for exact keyword matching")
        print("\nCleanup: rm -rf ./network_search_db")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set OPENAI_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
