#!/usr/bin/env python3
"""
Context Chunker - Split Large Configs for AI Processing

Handle network configurations larger than model context windows by intelligently
chunking them into processable segments.

From: AI for Networking Engineers - Volume 1, Chapter 7
Author: Eduard Dulharu

Usage:
    from context_chunker import ContextChunker

    chunker = ContextChunker(max_tokens=100000)
    chunks = chunker.chunk_config(large_config)

    for chunk in chunks:
        result = process_chunk(chunk)
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass


@dataclass
class ConfigChunk:
    """A chunk of configuration with metadata."""
    content: str
    chunk_id: int
    start_line: int
    end_line: int
    section_type: Optional[str] = None
    estimated_tokens: int = 0


class ContextChunker:
    """
    Intelligently chunk large network configurations.

    Strategies:
    - Chunk by logical sections (interfaces, routing, etc.)
    - Preserve context boundaries
    - Avoid splitting mid-configuration block
    - Estimate token counts
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        overlap_lines: int = 5
    ):
        """
        Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_lines: Number of overlapping lines between chunks
        """
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines

        # Approximate: 4 characters = 1 token
        self.chars_per_token = 4
        self.max_chars = max_tokens * self.chars_per_token

    def chunk_config(
        self,
        config: str,
        strategy: str = "section"
    ) -> List[ConfigChunk]:
        """
        Chunk configuration using specified strategy.

        Args:
            config: Configuration text
            strategy: Chunking strategy
                - "section": Split by logical sections (interfaces, routing, etc.)
                - "line": Split by line count
                - "size": Split by character/token size

        Returns:
            List of ConfigChunk objects
        """
        if strategy == "section":
            return self._chunk_by_section(config)
        elif strategy == "line":
            return self._chunk_by_lines(config)
        elif strategy == "size":
            return self._chunk_by_size(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _chunk_by_section(self, config: str) -> List[ConfigChunk]:
        """
        Chunk by logical configuration sections.

        Detects:
        - Interface configurations
        - Routing protocol configurations
        - ACL definitions
        - Line configurations
        - Other logical blocks
        """
        lines = config.splitlines()
        chunks = []
        current_chunk = []
        current_section = None
        chunk_id = 0
        start_line = 1

        for i, line in enumerate(lines, start=1):
            # Detect section starts
            section = self._detect_section(line)

            # Check if we should start a new chunk
            if self._should_start_new_chunk(current_chunk, line):
                if current_chunk:
                    # Save current chunk
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(ConfigChunk(
                        content=chunk_content,
                        chunk_id=chunk_id,
                        start_line=start_line,
                        end_line=i - 1,
                        section_type=current_section,
                        estimated_tokens=self._estimate_tokens(chunk_content)
                    ))
                    chunk_id += 1

                    # Start new chunk with overlap
                    overlap = current_chunk[-self.overlap_lines:] if len(current_chunk) > self.overlap_lines else current_chunk
                    current_chunk = overlap
                    start_line = max(1, i - len(overlap))
                else:
                    start_line = i

                current_section = section

            current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(ConfigChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                start_line=start_line,
                end_line=len(lines),
                section_type=current_section,
                estimated_tokens=self._estimate_tokens(chunk_content)
            ))

        return chunks

    def _chunk_by_lines(
        self,
        config: str,
        lines_per_chunk: int = 500
    ) -> List[ConfigChunk]:
        """Chunk by fixed line count with overlap."""
        lines = config.splitlines()
        chunks = []
        chunk_id = 0

        i = 0
        while i < len(lines):
            # Get chunk with overlap
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]

            chunk_content = '\n'.join(chunk_lines)
            chunks.append(ConfigChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                start_line=i + 1,
                end_line=end,
                estimated_tokens=self._estimate_tokens(chunk_content)
            ))

            chunk_id += 1
            i = end - self.overlap_lines  # Overlap for next chunk

        return chunks

    def _chunk_by_size(self, config: str) -> List[ConfigChunk]:
        """Chunk by token/character size."""
        lines = config.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        start_line = 1

        for i, line in enumerate(lines, start=1):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed limit
            if current_size + line_size > self.max_chars and current_chunk:
                # Save current chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append(ConfigChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    start_line=start_line,
                    end_line=i - 1,
                    estimated_tokens=self._estimate_tokens(chunk_content)
                ))
                chunk_id += 1

                # Start new chunk with overlap
                overlap = current_chunk[-self.overlap_lines:]
                current_chunk = overlap
                current_size = sum(len(l) + 1 for l in overlap)
                start_line = i - len(overlap)

            current_chunk.append(line)
            current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(ConfigChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                start_line=start_line,
                end_line=len(lines),
                estimated_tokens=self._estimate_tokens(chunk_content)
            ))

        return chunks

    def _detect_section(self, line: str) -> Optional[str]:
        """Detect configuration section type."""
        line_lower = line.lower().strip()

        # Interface sections
        if line_lower.startswith('interface '):
            return 'interface'

        # Routing protocols
        if any(line_lower.startswith(proto) for proto in
               ['router ospf', 'router eigrp', 'router bgp', 'router rip']):
            return 'routing'

        # ACLs
        if 'access-list' in line_lower or 'ip access-list' in line_lower:
            return 'acl'

        # Line configurations
        if line_lower.startswith('line '):
            return 'line'

        # VLAN
        if line_lower.startswith('vlan '):
            return 'vlan'

        return None

    def _should_start_new_chunk(self, current_chunk: List[str], new_line: str) -> bool:
        """
        Decide if we should start a new chunk.

        Considers:
        - Current chunk size
        - Section boundaries
        - Logical groupings
        """
        if not current_chunk:
            return False

        # Check size limit
        current_content = '\n'.join(current_chunk)
        if len(current_content) + len(new_line) > self.max_chars:
            return True

        # Check for major section boundary
        new_section = self._detect_section(new_line)
        if new_section:
            # Start new chunk at section boundaries if current is large enough
            if len(current_content) > self.max_chars * 0.5:  # 50% of max
                return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // self.chars_per_token

    def get_chunk_summary(self, chunks: List[ConfigChunk]) -> Dict[str, Any]:
        """Get summary statistics for chunks."""
        total_tokens = sum(c.estimated_tokens for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        sections = {}
        for chunk in chunks:
            if chunk.section_type:
                sections[chunk.section_type] = sections.get(chunk.section_type, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_estimated_tokens": total_tokens,
            "avg_tokens_per_chunk": int(avg_tokens),
            "max_tokens_per_chunk": max((c.estimated_tokens for c in chunks), default=0),
            "min_tokens_per_chunk": min((c.estimated_tokens for c in chunks), default=0),
            "sections": sections
        }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Context Chunker Demo
    ========================================
    Split large configs for AI processing
    ========================================
    """)

    # Create sample large config
    sample_config = """
hostname LARGE-ROUTER-01

! === INTERFACE CONFIGURATIONS ===
interface GigabitEthernet0/0
 description WAN Interface
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description LAN Interface
 ip address 10.1.1.1 255.255.255.0
 no shutdown

interface GigabitEthernet0/2
 description DMZ Interface
 ip address 192.168.100.1 255.255.255.0
 no shutdown

! === ROUTING CONFIGURATION ===
router ospf 1
 network 10.1.1.0 0.0.0.255 area 0
 network 192.168.100.0 0.0.0.255 area 0

router bgp 65001
 neighbor 203.0.113.2 remote-as 65002
 network 10.1.1.0 mask 255.255.255.0

! === ACL CONFIGURATIONS ===
ip access-list extended PERMIT_INTERNAL
 permit ip 10.1.1.0 0.0.0.255 any
 deny ip any any log

ip access-list extended BLOCK_DMZ
 deny ip 192.168.100.0 0.0.0.255 10.1.1.0 0.0.0.255
 permit ip any any

! === LINE CONFIGURATIONS ===
line vty 0 4
 transport input ssh
 login local

line console 0
 logging synchronous
"""

    # Test different chunking strategies
    chunker = ContextChunker(max_tokens=200, overlap_lines=2)  # Small for demo

    strategies = ["section", "line", "size"]

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        chunks = chunker.chunk_config(sample_config, strategy=strategy)

        print(f"\nGenerated {len(chunks)} chunks:\n")

        for chunk in chunks:
            print(f"Chunk {chunk.chunk_id}:")
            print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"  Section: {chunk.section_type or 'N/A'}")
            print(f"  Tokens: ~{chunk.estimated_tokens}")
            print(f"  Content preview: {chunk.content[:80].strip()}...")
            print()

        # Show summary
        summary = chunker.get_chunk_summary(chunks)
        print(f"Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    print("\n✅ Demo complete!")
    print("\n💡 Use Cases:")
    print("  - Large switch configs (thousands of interfaces)")
    print("  - Full routing tables")
    print("  - Complete firewall rule sets")
    print("  - Network-wide configurations")
