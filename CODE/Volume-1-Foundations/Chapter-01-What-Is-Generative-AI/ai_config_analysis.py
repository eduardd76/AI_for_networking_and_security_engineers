#!/usr/bin/env python3
"""
Chapter 1: What is Generative AI?
AI-Powered Network Configuration Analysis

This script demonstrates the "Aha Moment" — using AI to analyze
network configurations and diagnose issues automatically.

Run with: python ai_config_analysis.py
Or:       python ai_config_analysis.py --example 1

Author: Eduard Dulharu (Ed Harmoosh)
Company: vExpertAI GmbH
"""

import os
import sys
import argparse

# Load environment variables from .env file (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment

# ---------------------------------------------------------------------------
# AI Client Setup (Simple, no frameworks needed)
# ---------------------------------------------------------------------------

def get_ai_client():
    """
    Create an Anthropic client.
    
    We use the Anthropic SDK directly — no frameworks needed for basic calls.
    This keeps Chapter 1 simple. We'll explore abstractions in later chapters.
    """
    try:
        import anthropic
    except ImportError:
        print("❌ Error: anthropic package not installed")
        print("   Run: pip install anthropic")
        sys.exit(1)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found")
        print("")
        print("   Option 1: Export it directly")
        print("   export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("")
        print("   Option 2: Add to .env file")
        print("   echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env")
        print("")
        print("   Get a key at: https://console.anthropic.com/")
        sys.exit(1)
    
    return anthropic.Anthropic(api_key=api_key)


def ask_ai(prompt: str, temperature: float = 0) -> str:
    """
    Send a prompt to Claude and get a response.
    
    Args:
        prompt: The question or task for the AI
        temperature: 0 = deterministic, higher = more creative
    
    Returns:
        The AI's response as a string
    """
    client = get_ai_client()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text


# ---------------------------------------------------------------------------
# Example 1: Network Topology Analysis
# ---------------------------------------------------------------------------

def example_topology_analysis():
    """
    Analyze a network topology with configuration issues.
    
    Scenario: Users report intermittent connectivity between Site A and Site B.
    BGP is up, OSPF neighbors are up, but 40% of pings fail.
    
    This is a classic "iBGP next-hop" issue that trips up many engineers.
    """
    print("=" * 60)
    print("Example 1: Network Topology Analysis")
    print("=" * 60)
    print()

    topology = """
    Network Topology:

    Site A [R1] <--BGP--> [R2] Site B
              \\          /
               \\ OSPF   /
                \\     /
                 [R3]
                 Core
    """

    r1_config = """
    router bgp 65001
     neighbor 10.0.0.2 remote-as 65001
     network 192.168.1.0 mask 255.255.255.0
    !
    router ospf 1
     network 10.1.0.0 0.0.255.255 area 0
    !
    interface GigabitEthernet0/0
     ip address 192.168.1.1 255.255.255.0
    !
    interface GigabitEthernet0/1
     ip address 10.0.0.1 255.255.255.252
    """

    r2_config = """
    router bgp 65001
     neighbor 10.0.0.1 remote-as 65001
     network 192.168.2.0 mask 255.255.255.0
    !
    router ospf 1
     network 10.2.0.0 0.0.255.255 area 0
    !
    interface GigabitEthernet0/0
     ip address 192.168.2.1 255.255.255.0
    !
    interface GigabitEthernet0/1
     ip address 10.0.0.2 255.255.255.252
    """

    problem = """
    Symptoms reported by users:
    - Site B (192.168.2.0/24) has intermittent connectivity to Site A (192.168.1.0/24)
    - Ping succeeds about 60% of the time
    - Traceroute shows different paths each time
    - BGP session shows as Established
    - OSPF neighbors are all Full

    Question: What is causing the intermittent connectivity?
    """

    prompt = f"""You are a senior network engineer troubleshooting a connectivity issue.

{topology}

R1 Configuration:
{r1_config}

R2 Configuration:
{r2_config}

{problem}

Analyze the configuration and identify the root cause. Provide:
1. Root cause explanation (be specific)
2. Why this causes INTERMITTENT failure (not total failure)
3. Step-by-step fix
4. IOS commands to apply the fix
"""

    print("Scenario: Intermittent connectivity between sites")
    print("Sending to AI for analysis...")
    print()

    response = ask_ai(prompt, temperature=0)

    print("AI Analysis:")
    print("-" * 60)
    print(response)
    print()


# ---------------------------------------------------------------------------
# Example 2: Rule-Based vs AI-Based Comparison
# ---------------------------------------------------------------------------

def example_rule_vs_ai():
    """
    Compare traditional rule-based automation with AI-based analysis.
    
    Traditional approach: Pattern matching with regex
    AI approach: Contextual understanding and correlation
    """
    print("=" * 60)
    print("Example 2: Rule-Based vs AI-Based Automation")
    print("=" * 60)
    print()

    logs = [
        "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired",
        "%OSPF-5-ADJCHG: Process 1, Nbr 10.2.2.2 on Gi0/1 from FULL to DOWN",
        "%LINEPROTO-5-UPDOWN: Line protocol on Interface Gi0/0, changed state to down",
        "Interface GigabitEthernet0/2 flapping detected - 15 changes in 60 seconds",
        "%SYS-5-CONFIG_I: Configured from console by admin on vty0"
    ]

    # Show what traditional automation would do
    print("TRADITIONAL RULE-BASED APPROACH:")
    print("-" * 40)
    print("Pattern: 'BGP.*Down'     → Alert: Critical")
    print("Pattern: 'OSPF.*DOWN'    → Alert: Critical")
    print("Pattern: 'UPDOWN.*down'  → Alert: Warning")
    print("Pattern: 'flapping'      → Alert: Warning")
    print("Pattern: 'CONFIG_I'      → Alert: Info")
    print()
    print("Problems with this approach:")
    print("  • Cannot correlate events")
    print("  • Doesn't understand root cause")
    print("  • Generates separate alerts for related issues")
    print()

    # Now show AI approach
    prompt = f"""Analyze these network syslog messages as a senior network engineer.

Logs (in chronological order):
{chr(10).join(f"  {i+1}. {log}" for i, log in enumerate(logs))}

Provide:
1. Which issues need immediate attention (and why)
2. How these events are likely related to each other
3. Probable root cause
4. Recommended troubleshooting steps (in order)
5. Severity assessment for the overall situation

Think step by step about what these logs tell us together."""

    print("AI-BASED APPROACH:")
    print("-" * 40)
    print("Sending logs to AI for intelligent analysis...")
    print()

    response = ask_ai(prompt, temperature=0)

    print("AI Analysis:")
    print("-" * 60)
    print(response)
    print()


# ---------------------------------------------------------------------------
# Example 3: Auto-Generate Documentation
# ---------------------------------------------------------------------------

def example_generate_docs():
    """
    Automatically generate documentation from network configs.
    
    This shows the "generative" in Generative AI — creating new content
    (documentation) from existing data (config).
    """
    print("=" * 60)
    print("Example 3: Auto-Generate Network Documentation")
    print("=" * 60)
    print()

    config = """
    hostname CORE-SW-01
    !
    vlan 10
     name USERS
    vlan 20
     name SERVERS
    vlan 30
     name VOICE
    vlan 99
     name MANAGEMENT
    !
    interface GigabitEthernet1/0/1
     description Uplink to Core Router
     switchport mode trunk
     switchport trunk allowed vlan 10,20,30,99
    !
    interface GigabitEthernet1/0/10
     description User Access Port - Floor 2
     switchport mode access
     switchport access vlan 10
     spanning-tree portfast
    !
    interface Vlan99
     description Management Interface
     ip address 10.0.99.10 255.255.255.0
    """

    prompt = f"""Generate clear, professional documentation for this switch configuration.

Configuration:
{config}

Create documentation that includes:
1. Device Overview (hostname, apparent role in network)
2. VLAN Design (table format: VLAN ID | Name | Purpose)
3. Interface Summary (what's connected where)
4. Management Access (how to reach this device)
5. Notable Configurations (any best practices or concerns)

Format as clean markdown. Be concise but complete."""

    print("Input: Raw switch configuration")
    print("Output: Professional documentation")
    print()
    print("Generating documentation...")
    print()

    # Slightly higher temperature for more natural writing
    response = ask_ai(prompt, temperature=0.3)

    print("Generated Documentation:")
    print("-" * 60)
    print(response)
    print()


# ---------------------------------------------------------------------------
# Example 4: Security Issue Detection
# ---------------------------------------------------------------------------

def example_security_scan():
    """
    Scan a network config for security issues.
    
    AI can understand security context and best practices,
    finding issues that simple pattern matching would miss.
    """
    print("=" * 60)
    print("Example 4: Security Issue Detection")
    print("=" * 60)
    print()

    config = """
    hostname EDGE-RTR-01
    !
    username admin privilege 15 password cisco123
    !
    line vty 0 4
     transport input telnet
     password cisco
     login
    !
    snmp-server community public RO
    snmp-server community private RW
    !
    interface GigabitEthernet0/0
     description Internet Uplink
     ip address 203.0.113.1 255.255.255.252
     no ip proxy-arp
    !
    interface GigabitEthernet0/1
     description Internal Network
     ip address 192.168.1.1 255.255.255.0
    !
    ip http server
    no ip http secure-server
    !
    access-list 1 permit any
    ip nat inside source list 1 interface GigabitEthernet0/0 overload
    """

    prompt = f"""You are a network security auditor. Analyze this router configuration 
and identify ALL security vulnerabilities.

Configuration:
{config}

For each issue found, provide:
1. Issue name
2. Why it's a security risk
3. Severity (Critical / High / Medium / Low)
4. Specific remediation commands

Be thorough — check authentication, encryption, access control, protocols, 
SNMP, management interfaces, and ACLs."""

    print("Scanning configuration for security issues...")
    print()

    response = ask_ai(prompt, temperature=0)

    print("Security Audit Results:")
    print("-" * 60)
    print(response)
    print()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        ("Topology Analysis", example_topology_analysis),
        ("Rule-Based vs AI", example_rule_vs_ai),
        ("Generate Documentation", example_generate_docs),
        ("Security Scanning", example_security_scan),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        func()
        
        if i < len(examples):
            print()
            try:
                input(f"Press Enter for next example ({i+1}/{len(examples)})...")
            except KeyboardInterrupt:
                print("\n\nExiting.")
                return
            print()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("💡 Key Takeaways:")
    print("   • AI understands network context, not just patterns")
    print("   • Complex troubleshooting in seconds, not hours")
    print("   • Documentation and security audits on autopilot")
    print("   • But always verify before applying to production!")
    print()
    print("→ Next: Chapter 2 - Introduction to LLMs")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Chapter 1: What is Generative AI? — Network Config Analysis Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_config_analysis.py           Run all examples interactively
  python ai_config_analysis.py -e 1      Run only topology analysis
  python ai_config_analysis.py -e 4      Run only security scanning
  python ai_config_analysis.py --list    List all available examples
        """
    )
    
    parser.add_argument(
        "-e", "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run a specific example (1-4)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples"
    )
    
    args = parser.parse_args()
    
    # Print header
    print()
    print("🚀 Chapter 1: What is Generative AI?")
    print("   AI-Powered Network Configuration Analysis")
    print()
    
    if args.list:
        print("Available examples:")
        print("  1. Topology Analysis    — Diagnose BGP/OSPF connectivity issues")
        print("  2. Rule-Based vs AI     — Compare traditional vs AI automation")
        print("  3. Generate Docs        — Auto-create documentation from configs")
        print("  4. Security Scanning    — Find vulnerabilities in router configs")
        print()
        print("Run with: python ai_config_analysis.py --example N")
        return
    
    if args.example:
        examples = {
            1: example_topology_analysis,
            2: example_rule_vs_ai,
            3: example_generate_docs,
            4: example_security_scan,
        }
        examples[args.example]()
    else:
        run_all_examples()


if __name__ == "__main__":
    main()
