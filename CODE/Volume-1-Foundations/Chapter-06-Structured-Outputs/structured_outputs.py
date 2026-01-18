#!/usr/bin/env python3
"""
Structured Outputs with Pydantic

Get consistent, validated JSON from AI responses.

From: AI for Networking Engineers - Volume 1, Chapter 6
Author: Eduard Dulharu

Usage:
    python structured_outputs.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

# Load environment variables
load_dotenv()


# Define schemas
class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InterfaceInfo(BaseModel):
    """Interface configuration details."""
    name: str = Field(description="Interface name (e.g., Gi0/1)")
    ip_address: Optional[str] = Field(description="IP address with mask")
    description: Optional[str] = Field(description="Interface description")
    status: str = Field(description="Admin status (up/down)")

    @validator('name')
    def validate_interface_name(cls, v):
        if not any(x in v for x in ['Gigabit', 'FastEthernet', 'Ethernet', 'Vlan', 'Loopback']):
            raise ValueError('Invalid interface name format')
        return v


class ConfigIssue(BaseModel):
    """Security or configuration issue."""
    severity: Severity
    issue: str = Field(description="Description of the issue")
    location: str = Field(description="Where in config")
    recommendation: str = Field(description="How to fix")


class DeviceAnalysis(BaseModel):
    """Complete device analysis."""
    device_type: str = Field(description="Device type (router/switch/firewall)")
    hostname: Optional[str] = Field(description="Device hostname")
    interfaces: List[InterfaceInfo] = Field(description="List of interfaces")
    issues: List[ConfigIssue] = Field(description="Security/config issues")
    summary: str = Field(description="Overall assessment")


def example_1_simple_structured_output():
    """
    Example 1: Simple Structured Output

    Get a Pydantic object instead of text.
    """
    print("\n" + "="*60)
    print("Example 1: Simple Structured Output")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create parser
    parser = PydanticOutputParser(pydantic_object=InterfaceInfo)

    # Create prompt
    template = """Extract interface information from this config snippet:

{config}

{format_instructions}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["config"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    config = "interface GigabitEthernet0/1\n description WAN_UPLINK\n ip address 203.0.113.1 255.255.255.252\n no shutdown"

    # Get structured output
    formatted_prompt = prompt.format(config=config)
    response = llm.invoke(formatted_prompt)
    interface = parser.parse(response.content)

    print(f"\nConfig:\n{config}\n")
    print("Structured output:")
    print(f"  Name: {interface.name}")
    print(f"  IP: {interface.ip_address}")
    print(f"  Description: {interface.description}")
    print(f"  Status: {interface.status}")


def example_2_security_analysis():
    """
    Example 2: Security Issue Detection

    Get structured list of security issues.
    """
    print("\n" + "="*60)
    print("Example 2: Security Issue Detection")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Schema for issues
    class SecurityAnalysis(BaseModel):
        issues: List[ConfigIssue]
        risk_score: int = Field(description="Overall risk score 1-10")

    parser = PydanticOutputParser(pydantic_object=SecurityAnalysis)

    template = """Analyze this network configuration for security issues:

{config}

{format_instructions}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["config"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    config = """
line vty 0 4
 transport input telnet
 password cisco123

snmp-server community public RO
snmp-server community private RW

no service password-encryption
"""

    formatted_prompt = prompt.format(config=config)
    response = llm.invoke(formatted_prompt)
    analysis = parser.parse(response.content)

    print(f"\nConfig:\n{config}\n")
    print(f"Risk Score: {analysis.risk_score}/10\n")
    print(f"Found {len(analysis.issues)} issues:\n")

    for i, issue in enumerate(analysis.issues, 1):
        print(f"{i}. [{issue.severity.value.upper()}] {issue.issue}")
        print(f"   Location: {issue.location}")
        print(f"   Fix: {issue.recommendation}")
        print()


def example_3_device_inventory():
    """
    Example 3: Device Inventory Extraction

    Parse device details into structured format.
    """
    print("\n" + "="*60)
    print("Example 3: Device Inventory")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    class Device(BaseModel):
        hostname: str
        model: str
        ios_version: str
        serial_number: str
        uptime: str

    parser = PydanticOutputParser(pydantic_object=Device)

    template = """Extract device information from this 'show version' output:

{output}

{format_instructions}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["output"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    show_version = """
Cisco IOS Software, C2960 Software (C2960-LANBASEK9-M), Version 15.0(2)SE11
Technical Support: http://www.cisco.com/techsupport
Copyright (c) 1986-2015 by Cisco Systems, Inc.

ROM: Bootstrap program is C2960 boot loader
BOOTLDR: C2960 Boot Loader (C2960-HBOOT-M) Version 12.2(44)SE5

ACCESS-SW-01 uptime is 45 days, 12 hours, 34 minutes
System returned to ROM by power-on

cisco WS-C2960-24TT-L (PowerPC405) processor (revision B0) with 65536K bytes of memory.
Processor board ID FCZ1234A5B6
"""

    formatted_prompt = prompt.format(output=show_version)
    response = llm.invoke(formatted_prompt)
    device = parser.parse(response.content)

    print("\nExtracted device info:")
    print(f"  Hostname: {device.hostname}")
    print(f"  Model: {device.model}")
    print(f"  IOS Version: {device.ios_version}")
    print(f"  Serial: {device.serial_number}")
    print(f"  Uptime: {device.uptime}")


def example_4_routing_table_parsing():
    """
    Example 4: Parse Routing Table

    Extract routing information into structured format.
    """
    print("\n" + "="*60)
    print("Example 4: Routing Table Parsing")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    class Route(BaseModel):
        network: str
        mask: str
        next_hop: str
        protocol: str
        metric: int

    class RoutingTable(BaseModel):
        routes: List[Route]
        total_routes: int

    parser = PydanticOutputParser(pydantic_object=RoutingTable)

    template = """Parse this routing table output:

{output}

{format_instructions}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["output"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    routing_output = """
Gateway of last resort is 10.1.1.254 to network 0.0.0.0

S*    0.0.0.0/0 [1/0] via 10.1.1.254
      10.0.0.0/8 is variably subnetted, 5 subnets
C        10.1.1.0/24 is directly connected, GigabitEthernet0/0
L        10.1.1.1/32 is directly connected, GigabitEthernet0/0
O        10.2.0.0/24 [110/20] via 10.1.1.2
O        10.3.0.0/24 [110/30] via 10.1.1.3
B        192.168.0.0/16 [20/0] via 203.0.113.1
"""

    formatted_prompt = prompt.format(output=routing_output)
    response = llm.invoke(formatted_prompt)
    routing_table = parser.parse(response.content)

    print(f"\nParsed {routing_table.total_routes} routes:\n")
    for route in routing_table.routes:
        print(f"{route.protocol:5s} {route.network:18s} via {route.next_hop:15s} [metric: {route.metric}]")


def main():
    """Run all examples."""
    print("="*60)
    print("Structured Outputs with Pydantic")
    print("="*60)

    try:
        example_1_simple_structured_output()
        example_2_security_analysis()
        example_3_device_inventory()
        example_4_routing_table_parsing()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Benefits:")
        print("1. Consistent output format every time")
        print("2. Automatic validation of data types")
        print("3. Easy to integrate with other code")
        print("4. Type hints for better IDE support")
        print("5. No manual parsing of text responses")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install langchain langchain-anthropic pydantic python-dotenv")


if __name__ == "__main__":
    main()
