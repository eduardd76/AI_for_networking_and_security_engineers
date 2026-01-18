#!/usr/bin/env python3
"""
Sliding Window Processor - Process Large Configs Incrementally

Process large configurations using a sliding window approach with context carryover.

From: AI for Networking Engineers - Volume 1, Chapter 7
Author: Eduard Dulharu

Usage:
    from sliding_window import SlidingWindowProcessor

    processor = SlidingWindowProcessor(window_size=50000)
    results = processor.process_config(large_config, analysis_func)
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result from processing a single window."""
    window_id: int
    start_position: int
    end_position: int
    result: Any
    context_summary: Optional[str] = None


class SlidingWindowProcessor:
    """
    Process large texts using sliding window with context preservation.

    The sliding window approach:
    1. Process text in overlapping windows
    2. Carry forward context from previous windows
    3. Merge results intelligently
    4. Handle boundary cases
    """

    def __init__(
        self,
        window_size: int = 50000,
        overlap_size: int = 5000,
        context_size: int = 1000
    ):
        """
        Initialize processor.

        Args:
            window_size: Size of each window (in characters)
            overlap_size: Overlap between windows
            context_size: Size of context to carry forward
        """
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.context_size = context_size

    def process_config(
        self,
        config: str,
        processor_func: Callable[[str, Optional[str]], Any],
        merge_func: Optional[Callable[[List[Any]], Any]] = None
    ) -> List[WindowResult]:
        """
        Process configuration using sliding windows.

        Args:
            config: Configuration text to process
            processor_func: Function to process each window
                Signature: func(window_text: str, context: str) -> result
            merge_func: Optional function to merge results
                Signature: func(results: List[Any]) -> merged_result

        Returns:
            List of WindowResult objects
        """
        if len(config) <= self.window_size:
            # Config fits in one window
            result = processor_func(config, None)
            return [WindowResult(
                window_id=0,
                start_position=0,
                end_position=len(config),
                result=result
            )]

        # Split into windows
        windows = self._create_windows(config)
        results = []
        previous_context = None

        for i, (start, end, window_text) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")

            # Process window with previous context
            result = processor_func(window_text, previous_context)

            # Extract context for next window
            context_summary = self._extract_context(window_text, result)

            results.append(WindowResult(
                window_id=i,
                start_position=start,
                end_position=end,
                result=result,
                context_summary=context_summary
            ))

            # Update context for next iteration
            previous_context = context_summary

        # Optionally merge results
        if merge_func:
            merged = merge_func([r.result for r in results])
            logger.info(f"Merged {len(results)} window results")

        return results

    def _create_windows(self, text: str) -> List[tuple]:
        """
        Create overlapping windows from text.

        Returns:
            List of (start_pos, end_pos, window_text) tuples
        """
        windows = []
        text_length = len(text)
        position = 0

        while position < text_length:
            end_position = min(position + self.window_size, text_length)
            window_text = text[position:end_position]

            windows.append((position, end_position, window_text))

            # Move to next window with overlap
            position = end_position - self.overlap_size

            # Ensure we don't create tiny last window
            if text_length - position < self.overlap_size:
                break

        return windows

    def _extract_context(
        self,
        window_text: str,
        result: Any
    ) -> str:
        """
        Extract context summary from current window to pass to next.

        Args:
            window_text: Text of current window
            result: Processing result from current window

        Returns:
            Context summary string
        """
        # Take last N characters as context
        context = window_text[-self.context_size:] if len(window_text) > self.context_size else window_text

        # Add result summary if available
        if isinstance(result, dict) and 'summary' in result:
            context = f"Previous: {result['summary']}\n\n{context}"

        return context

    def process_with_checkpoints(
        self,
        config: str,
        processor_func: Callable[[str, Optional[str]], Any],
        checkpoint_callback: Optional[Callable[[int, WindowResult], None]] = None
    ) -> List[WindowResult]:
        """
        Process with checkpointing for long-running tasks.

        Args:
            config: Configuration text
            processor_func: Processing function
            checkpoint_callback: Called after each window completes

        Returns:
            List of WindowResult objects
        """
        windows = self._create_windows(config)
        results = []
        previous_context = None

        for i, (start, end, window_text) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)} (checkpoint enabled)")

            try:
                result = processor_func(window_text, previous_context)
                context_summary = self._extract_context(window_text, result)

                window_result = WindowResult(
                    window_id=i,
                    start_position=start,
                    end_position=end,
                    result=result,
                    context_summary=context_summary
                )

                results.append(window_result)
                previous_context = context_summary

                # Checkpoint callback
                if checkpoint_callback:
                    checkpoint_callback(i, window_result)

            except Exception as e:
                logger.error(f"Error processing window {i}: {e}")
                # Continue with next window instead of failing completely
                continue

        return results


class ConfigAnalysisWindows:
    """
    Specialized sliding window processor for configuration analysis.

    Handles network-specific concerns:
    - Preserves configuration context
    - Handles multi-line commands
    - Merges findings intelligently
    """

    def __init__(self):
        self.processor = SlidingWindowProcessor(
            window_size=50000,  # ~12,500 tokens
            overlap_size=5000,   # ~1,250 tokens overlap
            context_size=1000    # ~250 tokens context
        )

    def analyze_security(
        self,
        config: str,
        analysis_func: Callable[[str, Optional[str]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze large config for security issues using sliding windows.

        Args:
            config: Configuration text
            analysis_func: Function that analyzes a config chunk

        Returns:
            Merged security analysis results
        """
        # Process with windows
        results = self.processor.process_config(
            config,
            analysis_func,
            merge_func=self._merge_security_results
        )

        # Merge all results
        return self._merge_security_results([r.result for r in results])

    def _merge_security_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge security findings from multiple windows."""
        merged = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "summary": "Security analysis completed"
        }

        # Collect all findings
        for result in results:
            if isinstance(result, dict):
                for severity in ["critical", "high", "medium", "low"]:
                    if severity in result:
                        merged[severity].extend(result[severity])

        # Deduplicate findings (same issue might appear in overlapping windows)
        for severity in ["critical", "high", "medium", "low"]:
            merged[severity] = self._deduplicate_findings(merged[severity])

        return merged

    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate findings."""
        seen = set()
        unique = []

        for finding in findings:
            # Use issue description as key
            key = finding.get('issue', '')
            if key and key not in seen:
                seen.add(key)
                unique.append(finding)

        return unique


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Sliding Window Processor Demo
    ========================================
    Process large configs incrementally
    ========================================
    """)

    # Create large sample config
    large_config = """
hostname LARGE-ROUTER

! First section with interfaces
""" + "\n".join([
        f"""interface GigabitEthernet0/{i}
 description Interface {i}
 ip address 10.1.{i}.1 255.255.255.0
 no shutdown
"""
        for i in range(100)
    ]) + """

! Security configuration
line vty 0 4
 transport input ssh
 login local

! End of configuration
"""

    # Test 1: Basic sliding window
    print("\nTest 1: Basic Sliding Window Processing")
    print("-" * 60)

    processor = SlidingWindowProcessor(
        window_size=500,  # Small for demo
        overlap_size=100,
        context_size=50
    )

    def simple_analyzer(text: str, context: Optional[str]) -> Dict[str, Any]:
        """Simple analyzer that counts interfaces."""
        interface_count = text.count('interface ')
        return {
            "interface_count": interface_count,
            "context_provided": context is not None,
            "summary": f"Found {interface_count} interfaces"
        }

    results = processor.process_config(large_config, simple_analyzer)

    print(f"Processed {len(results)} windows:")
    for result in results:
        print(f"  Window {result.window_id}: "
              f"{result.start_position}-{result.end_position}, "
              f"Found {result.result['interface_count']} interfaces")

    # Test 2: With checkpointing
    print("\n\nTest 2: Processing with Checkpoints")
    print("-" * 60)

    checkpoints = []

    def checkpoint_callback(window_id: int, result: WindowResult):
        """Save checkpoint."""
        checkpoints.append((window_id, result.result))
        print(f"  ✓ Checkpoint saved for window {window_id}")

    results = processor.process_with_checkpoints(
        large_config,
        simple_analyzer,
        checkpoint_callback
    )

    print(f"\n{len(checkpoints)} checkpoints saved")

    # Test 3: Security analysis with merging
    print("\n\nTest 3: Security Analysis with Result Merging")
    print("-" * 60)

    analyzer = ConfigAnalysisWindows()

    def security_analyzer(text: str, context: Optional[str]) -> Dict[str, Any]:
        """Analyze config chunk for security issues."""
        findings = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        # Check for telnet
        if 'transport input telnet' in text.lower():
            findings["critical"].append({
                "issue": "Telnet enabled on VTY lines",
                "line": "line vty 0 4",
                "fix": "transport input ssh"
            })

        # Check for SSH
        if 'transport input ssh' in text.lower():
            findings["low"].append({
                "issue": "SSH properly configured",
                "line": "line vty 0 4"
            })

        return findings

    merged_results = analyzer.analyze_security(large_config, security_analyzer)

    print(f"Merged security findings:")
    print(f"  Critical: {len(merged_results['critical'])}")
    print(f"  High: {len(merged_results['high'])}")
    print(f"  Medium: {len(merged_results['medium'])}")
    print(f"  Low: {len(merged_results['low'])}")

    if merged_results['critical']:
        print(f"\n  Critical issues found:")
        for finding in merged_results['critical']:
            print(f"    - {finding['issue']}")

    print("\n✅ Demo complete!")
    print("\n💡 Use Cases:")
    print("  - Configs with 10,000+ lines")
    print("  - Full network snapshots")
    print("  - Multi-device configurations")
    print("  - Long-running analysis tasks")
