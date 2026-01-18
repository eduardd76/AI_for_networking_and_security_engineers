#!/usr/bin/env python3
"""
Working with Network Data

Process real network configs, logs, and command outputs.

From: AI for Networking Engineers - Volume 1, Chapter 9
Author: Eduard Dulharu

Usage:
    python network_data.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def example_1_config_analysis():
    """Analyze network configurations."""
    print("\n" + "="*60)
    print("Example 1: Config Analysis")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    config = """
hostname EDGE-RTR-01

interface GigabitEthernet0/0
 description WAN_Uplink_to_ISP
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description LAN_Internal
 ip address 10.0.0.1 255.255.255.0
 no shutdown

router ospf 1
 network 10.0.0.0 0.0.255.255 area 0

ip route 0.0.0.0 0.0.0.0 203.0.113.2

line vty 0 4
 transport input ssh
"""

    prompt = f"""Analyze this router config and provide:
1. Device role (edge/core/access)
2. Network topology insights
3. Routing protocol in use
4. Security posture

Config:
{config}"""

    response = llm.invoke(prompt)
    print(response.content)


def example_2_log_analysis():
    """Analyze syslog messages."""
    print("\n" + "="*60)
    print("Example 2: Log Analysis")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    logs = """
Jan 18 10:15:32 10.1.1.1 %LINK-3-UPDOWN: Interface Gi0/1, changed state to down
Jan 18 10:15:33 10.1.1.1 %LINEPROTO-5-UPDOWN: Line protocol on Gi0/1, changed state to down
Jan 18 10:16:01 10.1.1.2 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 from FULL to DOWN
Jan 18 10:20:15 10.1.1.3 %SEC-6-IPACCESSLOGP: list 101 denied tcp 192.168.1.100 -> 10.1.1.10(22)
"""

    prompt = f"""Analyze these syslog messages:

{logs}

Provide:
1. Root cause of the issue
2. Affected devices and services
3. Impact assessment
4. Recommended fix"""

    response = llm.invoke(prompt)
    print(response.content)


def example_3_multi_vendor():
    """Handle different vendor formats."""
    print("\n" + "="*60)
    print("Example 3: Multi-Vendor Configs")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    cisco_config = "interface GigabitEthernet0/1\n ip address 10.1.1.1 255.255.255.0"
    juniper_config = "set interfaces ge-0/0/1 unit 0 family inet address 10.1.1.1/24"

    prompt = f"""Compare these configs from different vendors:

Cisco:
{cisco_config}

Juniper:
{juniper_config}

Are they functionally equivalent? Explain."""

    response = llm.invoke(prompt)
    print(response.content)


def example_4_show_command_parsing():
    """Parse show command outputs."""
    print("\n" + "="*60)
    print("Example 4: Show Command Parsing")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    show_int_brief = """
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     10.1.1.1        YES manual up                    up
GigabitEthernet0/1     10.1.2.1        YES manual up                    up
GigabitEthernet0/2     unassigned      YES unset  administratively down down
Vlan1                  unassigned      YES unset  administratively down down
"""

    prompt = f"""Parse this 'show ip interface brief' output:

{show_int_brief}

Provide:
1. Number of active interfaces
2. List of configured IP addresses
3. Any interfaces in error state
4. Recommendations"""

    response = llm.invoke(prompt)
    print(response.content)


def main():
    """Run all examples."""
    print("="*60)
    print("Working with Network Data")
    print("="*60)

    try:
        example_1_config_analysis()
        example_2_log_analysis()
        example_3_multi_vendor()
        example_4_show_command_parsing()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Capabilities:")
        print("1. Analyze configs for insights")
        print("2. Troubleshoot from logs")
        print("3. Handle multi-vendor formats")
        print("4. Parse command outputs")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
