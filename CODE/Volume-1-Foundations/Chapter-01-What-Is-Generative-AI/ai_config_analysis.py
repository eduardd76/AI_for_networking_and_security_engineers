"""
Chapter 1: What is Generative AI?
AI-Powered Network Configuration Analysis

This script demonstrates the "Aha Moment" - using an LLM to analyze
a broken network topology automatically.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()

def analyze_network_topology():
    """
    Example 1: Analyze a network topology with configuration issues
    This simulates the lab scenario from Chapter 1.
    """
    print("=" * 60)
    print("Example 1: Network Topology Analysis")
    print("=" * 60)

    # Sample network topology and configs
    topology = """
    Network Topology:

    Site A [R1] <--BGP--> [R2] Site B
              \\          /
               \\ OSPF  /
                \\    /
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

    # Problem description
    problem = """
    Symptoms:
    - Users at Site B (192.168.2.0/24) report intermittent connectivity to Site A (192.168.1.0/24)
    - Ping succeeds 60% of the time
    - Traceroute shows asymmetric paths
    - BGP session is established
    - OSPF neighbors are up

    Question: What is causing the intermittent connectivity?
    """

    # Build comprehensive prompt
    prompt = f"""You are a senior network engineer analyzing a topology issue.

{topology}

R1 Configuration:
{r1_config}

R2 Configuration:
{r2_config}

{problem}

Analyze the configuration and identify the root cause. Provide:
1. Root cause explanation
2. Why it causes intermittent connectivity (not total failure)
3. Step-by-step fix
4. Configuration commands to resolve it
"""

    print("Analyzing topology with AI...")
    print(f"Prompt size: {len(prompt)} characters\n")

    # Call LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    response = llm.invoke(prompt)

    print("AI Analysis Result:")
    print("-" * 60)
    print(response.content)
    print("\n" + "=" * 60 + "\n")


def compare_rule_based_vs_ai():
    """
    Example 2: Compare rule-based vs AI-based automation
    Shows why AI is a paradigm shift
    """
    print("=" * 60)
    print("Example 2: Rule-Based vs AI-Based Automation")
    print("=" * 60)

    # Sample syslog messages
    logs = [
        "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired",
        "%OSPF-5-ADJCHG: Process 1, Nbr 10.2.2.2 on GigabitEthernet0/1 from FULL to DOWN",
        "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/0, changed state to down",
        "Interface GigabitEthernet0/2 flapping detected",
        "%SYS-5-CONFIG_I: Configured from console by admin on vty0"
    ]

    prompt = f"""Analyze these network syslog messages and identify:
1. Which issues are critical and need immediate attention
2. Which issues might be related to each other
3. Recommended troubleshooting steps
4. Estimated severity (Critical, High, Medium, Low)

Logs:
{chr(10).join(logs)}

Provide a structured analysis focusing on actionable insights."""

    print("Traditional rule-based approach:")
    print("- Pattern matching: 'BGP.*Down' → Critical")
    print("- Limited context understanding")
    print("- Cannot correlate events\n")

    print("AI-based approach:")
    print("Sending logs to LLM for intelligent analysis...\n")

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    response = llm.invoke(prompt)

    print("AI Analysis:")
    print("-" * 60)
    print(response.content)
    print("\n" + "=" * 60 + "\n")


def generate_documentation():
    """
    Example 3: Auto-generate documentation from configs
    Demonstrates generative AI capabilities
    """
    print("=" * 60)
    print("Example 3: Auto-Generate Network Documentation")
    print("=" * 60)

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
     description User Access Port
     switchport mode access
     switchport access vlan 10
     spanning-tree portfast
    !
    interface Vlan99
     description Management Interface
     ip address 10.0.99.10 255.255.255.0
    """

    prompt = f"""Generate clear, structured documentation for this switch configuration.
Include:
1. Device Overview (hostname, role)
2. VLAN Design (purpose of each VLAN)
3. Interface Summary (uplinks, access ports)
4. Management Access
5. Best Practices Applied

Format as markdown. Be concise but comprehensive.

Configuration:
{config}
"""

    print("Generating documentation from config...\n")

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    response = llm.invoke(prompt)

    print("Generated Documentation:")
    print("-" * 60)
    print(response.content)
    print("\n" + "=" * 60 + "\n")


def detect_security_issues():
    """
    Example 4: Security issue detection
    Shows AI understanding context and best practices
    """
    print("=" * 60)
    print("Example 4: Security Issue Detection")
    print("=" * 60)

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
     ip nat outside
    !
    interface GigabitEthernet0/1
     description Internal Network
     ip address 192.168.1.1 255.255.255.0
     ip nat inside
    !
    ip nat inside source list 1 interface GigabitEthernet0/0 overload
    !
    access-list 1 permit any
    """

    prompt = f"""As a security-focused network engineer, analyze this router configuration.

Identify ALL security issues with:
1. Issue description
2. Why it's a security risk
3. Severity (Critical, High, Medium, Low)
4. Remediation steps with specific commands

Configuration:
{config}

Be thorough - check passwords, protocols, ACLs, SNMP, etc."""

    print("Scanning configuration for security issues...\n")

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    response = llm.invoke(prompt)

    print("Security Analysis:")
    print("-" * 60)
    print(response.content)
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🚀 Chapter 1: What is Generative AI?")
    print("AI-Powered Network Configuration Analysis\n")

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please set your API key in .env file")
        exit(1)

    try:
        # Run all examples
        analyze_network_topology()
        input("Press Enter to continue to next example...")

        compare_rule_based_vs_ai()
        input("Press Enter to continue to next example...")

        generate_documentation()
        input("Press Enter to continue to next example...")

        detect_security_issues()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- AI can analyze complex network scenarios")
        print("- Goes beyond pattern matching to understand context")
        print("- Can generate documentation and identify security issues")
        print("- This is a paradigm shift from rule-based automation\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your API key is valid and you have internet connectivity")
