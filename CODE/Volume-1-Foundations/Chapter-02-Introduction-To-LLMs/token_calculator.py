#!/usr/bin/env python3
"""
Token Calculator for Network Engineers
Shows how configs, logs, and queries tokenize across different models.

This is the main project from Chapter 2: Introduction to LLMs.

Usage:
    python token_calculator.py <file>              Analyze a file
    python token_calculator.py interactive         Interactive mode
    python token_calculator.py visualize <text>    Show token breakdown

Author: Eduard Dulharu (Ed Harmoosh)
Company: vExpertAI GmbH
"""

import os
import sys

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, as of January 2026)
# ---------------------------------------------------------------------------

PRICING = {
    "claude-haiku": {"input": 0.25, "output": 1.25, "context": 200_000},
    "claude-sonnet": {"input": 3.00, "output": 15.00, "context": 200_000},
    "claude-opus": {"input": 15.00, "output": 75.00, "context": 200_000},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "context": 128_000},
    "gpt-4o": {"input": 2.50, "output": 10.00, "context": 128_000},
}

# ---------------------------------------------------------------------------
# Token Counting Functions
# ---------------------------------------------------------------------------

def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens using OpenAI's tiktoken library.
    Works offline, no API call needed.
    
    Args:
        text: Text to tokenize
        model: Model name for encoding selection
    
    Returns:
        Token count
    """
    try:
        import tiktoken
    except ImportError:
        print("❌ tiktoken not installed. Run: pip install tiktoken")
        # Fallback: rough estimate of ~4 chars per token
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (GPT-4 default)
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def count_tokens_anthropic(text: str) -> int:
    """
    Count tokens using Anthropic's API.
    More accurate for Claude models but requires API key.
    
    Args:
        text: Text to tokenize
    
    Returns:
        Token count
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback to tiktoken estimate
        return count_tokens_tiktoken(text)
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        response = client.messages.count_tokens(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens
    except Exception:
        # Fallback to tiktoken estimate
        return count_tokens_tiktoken(text)


def get_token_breakdown(text: str) -> list:
    """
    Get individual tokens for visualization.
    
    Args:
        text: Text to tokenize
    
    Returns:
        List of (token_text, token_id) tuples
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(text)
        
        tokens = []
        for token_id in token_ids:
            token_text = encoding.decode([token_id])
            tokens.append((token_text, token_id))
        
        return tokens
    except ImportError:
        return []


# ---------------------------------------------------------------------------
# Cost Calculation
# ---------------------------------------------------------------------------

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> dict:
    """
    Calculate cost for a given model and token counts.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name (from PRICING keys)
    
    Returns:
        Dictionary with cost breakdown
    """
    if model not in PRICING:
        return {"error": f"Unknown model: {model}"}
    
    pricing = PRICING[model]
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "context_limit": pricing["context"],
    }


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def analyze_file(file_path: str, expected_output_tokens: int = 2000):
    """
    Analyze a network config/log file and show tokenization + cost.
    
    Args:
        file_path: Path to file
        expected_output_tokens: Estimated output tokens (for cost calc)
    """
    # Read file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return
    
    print("=" * 80)
    print(f"FILE ANALYSIS: {file_path}")
    print("=" * 80)
    print(f"File size: {len(content):,} characters")
    print(f"File lines: {len(content.splitlines()):,}")
    print()
    
    # Count tokens
    print("TOKEN COUNTS:")
    print("-" * 80)
    
    tiktoken_count = count_tokens_tiktoken(content)
    print(f"Tiktoken (GPT-4):   {tiktoken_count:,} tokens")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        claude_count = count_tokens_anthropic(content)
        print(f"Claude (API):       {claude_count:,} tokens")
    else:
        claude_count = tiktoken_count
        print(f"Claude (estimated): {claude_count:,} tokens")
    
    print()
    
    # Context window check
    print("CONTEXT WINDOW FIT:")
    print("-" * 80)
    
    total_needed = tiktoken_count + expected_output_tokens
    
    for model, info in PRICING.items():
        fits = "✅" if total_needed <= info["context"] else "❌"
        remaining = info["context"] - total_needed
        print(f"{model:15s} ({info['context']:>7,} max): {fits} {remaining:>+8,} tokens")
    
    print()
    
    # Cost estimates
    print("COST ESTIMATES:")
    print("-" * 80)
    print(f"(Assuming {expected_output_tokens:,} output tokens)\n")
    
    for model in PRICING.keys():
        cost = calculate_cost(tiktoken_count, expected_output_tokens, model)
        print(f"{model:15s} → ${cost['total_cost']:.6f}")
    
    print()
    
    # Batch projection
    print("BATCH PROJECTION (1,000 files):")
    print("-" * 80)
    
    for model in ["gpt-4o-mini", "claude-haiku", "claude-sonnet"]:
        cost = calculate_cost(tiktoken_count * 1000, expected_output_tokens * 1000, model)
        print(f"{model:15s} → ${cost['total_cost']:.2f}/month")
    
    print("\n" + "=" * 80)


def visualize_tokens(text: str):
    """
    Show how text is broken into tokens.
    
    Args:
        text: Text to tokenize
    """
    print("=" * 80)
    print("TOKEN VISUALIZATION")
    print("=" * 80)
    print(f"\nInput ({len(text)} chars):")
    print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
    print()
    
    tokens = get_token_breakdown(text)
    
    if not tokens:
        print("❌ tiktoken not installed. Run: pip install tiktoken")
        return
    
    print(f"Total Tokens: {len(tokens)}")
    print("\nToken Breakdown:")
    print("-" * 80)
    
    # Show first 30 tokens
    for i, (token_text, token_id) in enumerate(tokens[:30], 1):
        # Make whitespace visible
        display = token_text.replace('\n', '↵').replace(' ', '·').replace('\t', '→')
        if not display.strip():
            display = repr(token_text)
        print(f"  {i:3d}. '{display}' (ID: {token_id})")
    
    if len(tokens) > 30:
        print(f"  ... and {len(tokens) - 30} more tokens")
    
    print("\n" + "=" * 80)


def interactive_mode():
    """Interactive token calculator."""
    print("=" * 80)
    print("INTERACTIVE TOKEN CALCULATOR")
    print("=" * 80)
    print("Enter text to see tokenization. Commands:")
    print("  'quit' or 'q'  - Exit")
    print("  'file <path>'  - Analyze a file")
    print()
    
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        if text.startswith('file '):
            file_path = text[5:].strip()
            analyze_file(file_path)
            continue
        
        # Count tokens
        tiktoken_count = count_tokens_tiktoken(text)
        
        if os.getenv("ANTHROPIC_API_KEY"):
            claude_count = count_tokens_anthropic(text)
            print(f"  Tiktoken: {tiktoken_count} tokens | Claude: {claude_count} tokens")
        else:
            print(f"  Tokens: {tiktoken_count}")
        
        # Show cost for a typical request
        cost = calculate_cost(tiktoken_count, 500, "claude-sonnet")
        print(f"  Est. cost (Sonnet, 500 output): ${cost['total_cost']:.6f}")
        print()


def show_pricing():
    """Display current pricing table."""
    print("=" * 80)
    print("MODEL PRICING (per 1M tokens, January 2026)")
    print("=" * 80)
    print()
    print(f"{'Model':<15} {'Input':>10} {'Output':>10} {'Context':>12}")
    print("-" * 50)
    
    for model, info in PRICING.items():
        print(f"{model:<15} ${info['input']:>8.2f} ${info['output']:>8.2f} {info['context']:>10,}")
    
    print()
    print("💡 Tips:")
    print("   • Use gpt-4o-mini or claude-haiku for 80% of tasks")
    print("   • Reserve claude-sonnet/opus for complex reasoning")
    print("   • Output tokens cost 3-5x more than input!")
    print()


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = """hostname CORE-RTR-01
!
interface GigabitEthernet0/0
 description WAN Link to ISP
 ip address 203.0.113.1 255.255.255.252
 ip nat outside
 no shutdown
!
interface GigabitEthernet0/1
 description LAN Subnet
 ip address 192.168.1.1 255.255.255.0
 ip nat inside
 no shutdown
!
router bgp 65001
 neighbor 203.0.113.2 remote-as 65000
 network 192.168.1.0 mask 255.255.255.0
!
snmp-server community public RO
snmp-server community private RW
!
line vty 0 4
 password cisco123
 transport input telnet ssh
!
end"""


def demo():
    """Run a quick demo with sample data."""
    print("=" * 80)
    print("TOKEN CALCULATOR DEMO")
    print("=" * 80)
    print()
    
    # Show pricing
    show_pricing()
    
    # Visualize sample config
    print("\n--- Sample Config Tokenization ---\n")
    visualize_tokens(SAMPLE_CONFIG)
    
    # Cost analysis
    print("\n--- Cost Analysis ---\n")
    
    tokens = count_tokens_tiktoken(SAMPLE_CONFIG)
    output_tokens = 2000  # Typical analysis response
    
    print(f"Config: {tokens} tokens")
    print(f"Expected output: {output_tokens} tokens")
    print()
    
    print("Cost per analysis:")
    for model in ["gpt-4o-mini", "claude-haiku", "claude-sonnet"]:
        cost = calculate_cost(tokens, output_tokens, model)
        print(f"  {model:15s}: ${cost['total_cost']:.6f}")
    
    print()
    print("Monthly cost (1,000 configs):")
    for model in ["gpt-4o-mini", "claude-haiku", "claude-sonnet"]:
        cost = calculate_cost(tokens * 1000, output_tokens * 1000, model)
        print(f"  {model:15s}: ${cost['total_cost']:.2f}")
    
    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print()
        print("🔢 Token Calculator for Network Engineers")
        print("   Chapter 2: Introduction to LLMs")
        print()
        print("Usage:")
        print("  python token_calculator.py <file>              Analyze a file")
        print("  python token_calculator.py interactive         Interactive mode")
        print("  python token_calculator.py visualize '<text>'  Show token breakdown")
        print("  python token_calculator.py pricing             Show pricing table")
        print("  python token_calculator.py demo                Run demo with samples")
        print()
        print("Examples:")
        print("  python token_calculator.py router_config.txt")
        print("  python token_calculator.py visualize 'interface GigabitEthernet0/0'")
        print()
        return
    
    command = sys.argv[1]
    
    if command == "interactive":
        interactive_mode()
    elif command == "visualize":
        if len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            visualize_tokens(text)
        else:
            print("Usage: python token_calculator.py visualize '<text>'")
    elif command == "pricing":
        show_pricing()
    elif command == "demo":
        demo()
    else:
        # Assume it's a file path
        file_path = command
        expected_output = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        analyze_file(file_path, expected_output)


if __name__ == "__main__":
    main()
