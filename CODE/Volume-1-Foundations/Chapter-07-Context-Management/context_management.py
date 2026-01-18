#!/usr/bin/env python3
"""
Context Management for Large Network Configurations

Handle large configs that exceed token limits.

From: AI for Networking Engineers - Volume 1, Chapter 7
Author: Eduard Dulharu

Usage:
    python context_management.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

load_dotenv()


def example_1_chunking():
    """
    Example 1: Chunking Large Configs

    Break large configs into manageable pieces.
    """
    print("\n" + "="*60)
    print("Example 1: Chunking Large Configs")
    print("="*60)

    # Sample large config
    config = """
hostname CORE-SW-01

interface Vlan10
 description Management
 ip address 10.0.10.1 255.255.255.0

interface Vlan20
 description Users
 ip address 10.0.20.1 255.255.255.0

interface GigabitEthernet0/1
 description Uplink_to_Core
 switchport mode trunk

interface GigabitEthernet0/2
 description Access_Port
 switchport mode access
 switchport access vlan 20

""" * 10  # Repeat to make it large

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\ninterface", "\n\n", "\n", " "]
    )

    chunks = splitter.split_text(config)

    print(f"\nOriginal config: {len(config)} characters")
    print(f"Split into {len(chunks)} chunks")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")


def example_2_analyze_chunks():
    """
    Example 2: Analyze Each Chunk Separately

    Process large configs in pieces.
    """
    print("\n" + "="*60)
    print("Example 2: Analyze Chunks Separately")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    config = """
interface GigabitEthernet0/1
 description WAN
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/2
 description LAN
 ip address 10.0.0.1 255.255.255.0
 shutdown

interface GigabitEthernet0/3
 description Management
 ip address 192.168.1.1 255.255.255.0
 no shutdown
"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\ninterface", "\n"]
    )

    chunks = splitter.split_text(config)

    print(f"\nAnalyzing {len(chunks)} interface configs:\n")

    for i, chunk in enumerate(chunks, 1):
        if "interface" in chunk:
            response = llm.invoke(f"Analyze this interface config briefly:\n{chunk}")
            print(f"Chunk {i}:")
            print(f"{response.content[:150]}...")
            print()


def example_3_summarize_and_combine():
    """
    Example 3: Map-Reduce Pattern

    Summarize chunks, then combine summaries.
    """
    print("\n" + "="*60)
    print("Example 3: Map-Reduce Pattern")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Simulated large log file
    logs = [
        "Jan 18 10:15:32 10.1.1.1 %LINK-3-UPDOWN: Interface Gi0/1, changed state to down",
        "Jan 18 10:15:33 10.1.1.1 %LINEPROTO-5-UPDOWN: Line protocol on Gi0/1, changed state to down",
        "Jan 18 10:16:01 10.1.1.2 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 from FULL to DOWN",
        "Jan 18 10:20:15 10.1.1.3 %SEC-6-IPACCESSLOGP: list 101 denied tcp 192.168.1.100(45678) -> 10.1.1.10(22)",
        "Jan 18 10:25:00 10.1.1.4 %SPANTREE-2-ROOTCHANGE: Root Changed for VLAN 10"
    ]

    # Step 1: Summarize each log entry
    print("\nStep 1: Map - Summarize each log:\n")
    summaries = []
    for log in logs:
        response = llm.invoke(f"Summarize this log in 5 words or less:\n{log}")
        summary = response.content.strip()
        summaries.append(summary)
        print(f"  - {summary}")

    # Step 2: Combine summaries
    print("\nStep 2: Reduce - Combine into overall summary:\n")
    combined = "\n".join(summaries)
    final_response = llm.invoke(f"Provide an overall summary of these network events:\n{combined}")
    print(final_response.content)


def example_4_sliding_window():
    """
    Example 4: Sliding Window for Context

    Maintain context across chunks.
    """
    print("\n" + "="*60)
    print("Example 4: Sliding Window Context")
    print("="*60)

    config_sections = [
        "hostname CORE-SW-01",
        "vlan 10\n name Management",
        "vlan 20\n name Users",
        "interface Vlan10\n ip address 10.0.10.1 255.255.255.0",
        "interface Vlan20\n ip address 10.0.20.1 255.255.255.0"
    ]

    print("\nProcessing config with sliding window (window size=2):\n")

    for i in range(len(config_sections) - 1):
        window = config_sections[i:i+2]
        context = "\n".join(window)
        print(f"Window {i+1}:")
        print(f"  {context[:100]}...")
        print()


def main():
    """Run all examples."""
    print("="*60)
    print("Context Management")
    print("="*60)

    try:
        example_1_chunking()
        example_2_analyze_chunks()
        example_3_summarize_and_combine()
        example_4_sliding_window()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Strategies:")
        print("1. Chunking - Split large configs into pieces")
        print("2. Process separately - Analyze each chunk")
        print("3. Map-Reduce - Summarize then combine")
        print("4. Sliding window - Maintain context across chunks")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
