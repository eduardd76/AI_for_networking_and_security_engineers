#!/usr/bin/env python3
"""
API Integration with Netmiko

Combine network automation tools with AI.

From: AI for Networking Engineers - Volume 1, Chapter 10
Author: Eduard Dulharu

Usage:
    python api_integration.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def example_1_simulated_netmiko():
    """Simulate Netmiko integration (without real devices)."""
    print("\n" + "="*60)
    print("Example 1: AI + Netmiko Pattern")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Simulated output (in production, use Netmiko)
    show_run = """
hostname CORE-SW-01

interface GigabitEthernet0/1
 description Uplink
 switchport mode trunk

interface GigabitEthernet0/2
 description Access_Port
 switchport mode access
 switchport access vlan 10
"""

    # Analyze with AI
    prompt = f"""Analyze this switch config:

{show_run}

Generate commands to:
1. Add VLAN 20 for guests
2. Configure Gi0/3 as an access port in VLAN 20
3. Add description "Guest_Network"

Provide only the commands, one per line."""

    response = llm.invoke(prompt)

    print("AI-generated commands:")
    print(response.content)
    print("\n(In production, send these commands via Netmiko)")


def example_2_error_interpretation():
    """Use AI to interpret device errors."""
    print("\n" + "="*60)
    print("Example 2: Error Interpretation")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Simulated error
    error = """
% Invalid input detected at '^' marker.
Switch(config)#interface gigabitethernet 0/1
                    ^
"""

    prompt = f"""I got this error on a Cisco switch:

{error}

What went wrong and how do I fix it?"""

    response = llm.invoke(prompt)
    print(response.content)


def example_3_command_validation():
    """Validate commands before sending to devices."""
    print("\n" + "="*60)
    print("Example 3: Command Validation")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    commands = [
        "interface GigabitEthernet0/1",
        "ip address 10.1.1.1 255.255.255",  # Wrong mask
        "no shutdow",  # Typo
        "exit"
    ]

    prompt = f"""Check these Cisco IOS commands for errors:

{chr(10).join(commands)}

Flag any syntax errors or typos."""

    response = llm.invoke(prompt)
    print(response.content)


def main():
    """Run all examples."""
    print("="*60)
    print("API Integration Patterns")
    print("="*60)

    try:
        example_1_simulated_netmiko()
        example_2_error_interpretation()
        example_3_command_validation()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nIntegration Patterns:")
        print("1. Use AI to generate device commands")
        print("2. Interpret errors from devices")
        print("3. Validate commands before execution")
        print("\nProduction: Combine with Netmiko, NAPALM, Ansible")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
