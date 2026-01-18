#!/usr/bin/env python3
"""
Token Optimizer - Reduce Token Usage and Costs

Optimize prompts and responses to use fewer tokens while maintaining quality.

From: AI for Networking Engineers - Volume 1, Chapter 7
Author: Eduard Dulharu

Usage:
    from token_optimizer import TokenOptimizer

    optimizer = TokenOptimizer()

    # Count tokens
    count = optimizer.count_tokens(text)

    # Optimize config (remove comments, compress)
    optimized = optimizer.optimize_config(config_text)
"""

from typing import Dict, Any, Optional, List
import re


class TokenOptimizer:
    """
    Optimize text for token efficiency.

    Strategies:
    - Remove unnecessary whitespace
    - Strip comments
    - Compress repetitive patterns
    - Abbreviate common terms
    - Remove empty lines
    """

    # Approximate tokens per character (varies by model)
    CHARS_PER_TOKEN = 4

    def __init__(self):
        """Initialize optimizer."""
        # Common networking abbreviations
        self.abbreviations = {
            'GigabitEthernet': 'Gi',
            'FastEthernet': 'Fa',
            'interface': 'int',
            'description': 'desc',
            'ip address': 'ip addr',
            'subnet mask': 'mask',
            'shutdown': 'shut',
            'no shutdown': 'no shut'
        }

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Args:
            text: Text to count

        Returns:
            Estimated token count

        Note: This is an approximation. Actual count varies by model.
        """
        if not text:
            return 0

        # Rough approximation: 4 chars = 1 token
        return len(text) // self.CHARS_PER_TOKEN

    def optimize_config(
        self,
        config: str,
        aggressive: bool = False
    ) -> str:
        """
        Optimize network configuration for token efficiency.

        Args:
            config: Configuration text
            aggressive: If True, use aggressive optimization (may lose readability)

        Returns:
            Optimized configuration
        """
        optimized = config

        # Remove comments
        optimized = self._remove_comments(optimized)

        # Remove empty lines
        optimized = self._remove_empty_lines(optimized)

        # Compress whitespace
        optimized = self._compress_whitespace(optimized)

        if aggressive:
            # Abbreviate common terms
            optimized = self._abbreviate(optimized)

            # Remove unnecessary config lines
            optimized = self._remove_unnecessary_lines(optimized)

        return optimized

    def optimize_prompt(
        self,
        prompt: str,
        preserve_structure: bool = True
    ) -> str:
        """
        Optimize prompt for token efficiency.

        Args:
            prompt: Prompt text
            preserve_structure: Keep formatting for readability

        Returns:
            Optimized prompt
        """
        optimized = prompt

        # Remove excessive newlines
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)

        if not preserve_structure:
            # More aggressive optimization
            optimized = re.sub(r'\n+', '\n', optimized)
            optimized = re.sub(r' {2,}', ' ', optimized)

        # Remove markdown formatting (saves tokens)
        optimized = re.sub(r'\*\*([^*]+)\*\*', r'\1', optimized)  # Bold
        optimized = re.sub(r'\*([^*]+)\*', r'\1', optimized)  # Italic

        return optimized.strip()

    def _remove_comments(self, text: str) -> str:
        """Remove comment lines starting with !"""
        lines = text.splitlines()
        filtered = [line for line in lines if not line.strip().startswith('!')]
        return '\n'.join(filtered)

    def _remove_empty_lines(self, text: str) -> str:
        """Remove empty lines."""
        lines = text.splitlines()
        filtered = [line for line in lines if line.strip()]
        return '\n'.join(filtered)

    def _compress_whitespace(self, text: str) -> str:
        """Compress multiple spaces to single space."""
        lines = text.splitlines()
        compressed = []
        for line in lines:
            # Keep leading whitespace (indentation matters in configs)
            leading = len(line) - len(line.lstrip())
            content = line.strip()
            content = re.sub(r' {2,}', ' ', content)
            compressed.append(' ' * leading + content)
        return '\n'.join(compressed)

    def _abbreviate(self, text: str) -> str:
        """Replace long terms with abbreviations."""
        result = text
        for long_form, short_form in self.abbreviations.items():
            # Case-insensitive replacement
            result = re.sub(
                r'\b' + re.escape(long_form) + r'\b',
                short_form,
                result,
                flags=re.IGNORECASE
            )
        return result

    def _remove_unnecessary_lines(self, text: str) -> str:
        """Remove lines that don't affect functionality."""
        lines = text.splitlines()
        filtered = []

        for line in lines:
            stripped = line.strip()

            # Skip banner messages
            if 'banner' in stripped.lower():
                continue

            # Skip logging buffered (not essential for analysis)
            if 'logging buffered' in stripped.lower():
                continue

            filtered.append(line)

        return '\n'.join(filtered)

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-3-5-sonnet"
    ) -> float:
        """
        Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Estimated cost in USD
        """
        # Pricing per million tokens (January 2026)
        pricing = {
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-5-haiku": {"input": 0.80, "output": 4.0},
            "gpt-4": {"input": 10.0, "output": 30.0},
        }

        if model not in pricing:
            model = "claude-3-5-sonnet"

        rates = pricing[model]

        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost

    def compare_optimization(
        self,
        original: str,
        optimized: str
    ) -> Dict[str, Any]:
        """
        Compare original and optimized versions.

        Args:
            original: Original text
            optimized: Optimized text

        Returns:
            Comparison statistics
        """
        orig_tokens = self.count_tokens(original)
        opt_tokens = self.count_tokens(optimized)
        tokens_saved = orig_tokens - opt_tokens
        savings_pct = (tokens_saved / orig_tokens * 100) if orig_tokens > 0 else 0

        # Calculate cost savings (using Sonnet pricing)
        orig_cost = self.estimate_cost(orig_tokens, orig_tokens // 2)
        opt_cost = self.estimate_cost(opt_tokens, opt_tokens // 2)
        cost_saved = orig_cost - opt_cost

        return {
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "tokens_saved": tokens_saved,
            "savings_percentage": f"{savings_pct:.1f}%",
            "original_cost_estimate": f"${orig_cost:.6f}",
            "optimized_cost_estimate": f"${opt_cost:.6f}",
            "cost_saved": f"${cost_saved:.6f}",
            "original_size_bytes": len(original),
            "optimized_size_bytes": len(optimized)
        }

    def suggest_optimizations(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest potential optimizations.

        Args:
            text: Text to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Check for comments
        comment_lines = sum(1 for line in text.splitlines() if line.strip().startswith('!'))
        if comment_lines > 0:
            tokens_saved = comment_lines * 5  # Approximate
            suggestions.append({
                "optimization": "Remove comments",
                "description": f"Remove {comment_lines} comment lines",
                "tokens_saved": tokens_saved,
                "impact": "low"
            })

        # Check for empty lines
        empty_lines = sum(1 for line in text.splitlines() if not line.strip())
        if empty_lines > 0:
            suggestions.append({
                "optimization": "Remove empty lines",
                "description": f"Remove {empty_lines} empty lines",
                "tokens_saved": empty_lines,
                "impact": "low"
            })

        # Check for long interface names
        long_interfaces = re.findall(r'GigabitEthernet|FastEthernet', text)
        if long_interfaces:
            tokens_saved = len(long_interfaces) * 2
            suggestions.append({
                "optimization": "Abbreviate interface names",
                "description": f"Abbreviate {len(long_interfaces)} long interface names",
                "tokens_saved": tokens_saved,
                "impact": "medium"
            })

        # Check for repetitive patterns
        lines = text.splitlines()
        if len(lines) > len(set(lines)):
            duplicates = len(lines) - len(set(lines))
            suggestions.append({
                "optimization": "Remove duplicate lines",
                "description": f"Found {duplicates} duplicate lines",
                "tokens_saved": duplicates * 10,
                "impact": "high"
            })

        return suggestions


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Token Optimizer Demo
    ========================================
    Reduce tokens and costs
    ========================================
    """)

    optimizer = TokenOptimizer()

    # Sample config with optimization opportunities
    sample_config = """
! This is a configuration file
! Generated on 2026-01-18

hostname ROUTER-01

! Interface configurations
interface GigabitEthernet0/0
 description WAN Interface to ISP
 ip address 203.0.113.1 255.255.255.252
 no shutdown


interface GigabitEthernet0/1
 description LAN Interface
 ip address 10.1.1.1 255.255.255.0
 no shutdown


! Routing configuration
router ospf 1
 network 10.1.1.0 0.0.0.255 area 0


! End of configuration
"""

    # Test 1: Count tokens
    print("\nTest 1: Token Counting")
    print("-" * 60)
    token_count = optimizer.count_tokens(sample_config)
    print(f"Original config: ~{token_count} tokens")
    print(f"Estimated cost: ${optimizer.estimate_cost(token_count, token_count//2):.6f}")

    # Test 2: Basic optimization
    print("\n\nTest 2: Basic Optimization")
    print("-" * 60)
    optimized = optimizer.optimize_config(sample_config, aggressive=False)

    comparison = optimizer.compare_optimization(sample_config, optimized)
    print(f"Original tokens: {comparison['original_tokens']}")
    print(f"Optimized tokens: {comparison['optimized_tokens']}")
    print(f"Tokens saved: {comparison['tokens_saved']} ({comparison['savings_percentage']})")
    print(f"Cost saved: {comparison['cost_saved']}")

    print(f"\nOptimized config:\n{optimized}")

    # Test 3: Aggressive optimization
    print("\n\nTest 3: Aggressive Optimization")
    print("-" * 60)
    aggressive_opt = optimizer.optimize_config(sample_config, aggressive=True)

    comparison2 = optimizer.compare_optimization(sample_config, aggressive_opt)
    print(f"Original tokens: {comparison2['original_tokens']}")
    print(f"Aggressive optimized tokens: {comparison2['optimized_tokens']}")
    print(f"Tokens saved: {comparison2['tokens_saved']} ({comparison2['savings_percentage']})")

    print(f"\nAggressively optimized config:\n{aggressive_opt}")

    # Test 4: Optimization suggestions
    print("\n\nTest 4: Optimization Suggestions")
    print("-" * 60)
    suggestions = optimizer.suggest_optimizations(sample_config)

    print(f"Found {len(suggestions)} optimization opportunities:\n")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['optimization']}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Tokens saved: ~{suggestion['tokens_saved']}")
        print(f"   Impact: {suggestion['impact']}")
        print()

    # Test 5: Prompt optimization
    print("\n\nTest 5: Prompt Optimization")
    print("-" * 60)

    verbose_prompt = """

    Analyze this configuration.


    **Important**: Check for security issues.

    *Please* provide detailed analysis.


    """

    optimized_prompt = optimizer.optimize_prompt(verbose_prompt, preserve_structure=False)

    print(f"Original prompt tokens: {optimizer.count_tokens(verbose_prompt)}")
    print(f"Optimized prompt tokens: {optimizer.count_tokens(optimized_prompt)}")
    print(f"\nOptimized prompt:\n'{optimized_prompt}'")

    print("\n✅ Demo complete!")
    print("\n💡 Optimization Tips:")
    print("  - Remove comments before sending to AI")
    print("  - Use abbreviated interface names")
    print("  - Remove empty lines")
    print("  - Compress whitespace")
    print("  - Typical savings: 20-40% tokens")
