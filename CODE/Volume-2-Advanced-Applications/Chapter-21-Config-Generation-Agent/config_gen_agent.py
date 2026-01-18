#!/usr/bin/env python3
"""
Network Configuration Generation Agent

An agent that generates network configurations from high-level requirements.

From: AI for Networking Engineers - Volume 2, Chapter 21
Author: Eduard Dulharu

Usage:
    python config_gen_agent.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define configuration schemas
class InterfaceConfig(BaseModel):
    """Interface configuration."""
    name: str = Field(description="Interface name (e.g., GigabitEthernet0/0)")
    description: str = Field(description="Interface description")
    ip_address: str = Field(description="IP address with mask")
    enabled: bool = Field(default=True, description="Interface enabled/shutdown")


class VLANConfig(BaseModel):
    """VLAN configuration."""
    vlan_id: int = Field(description="VLAN ID (1-4094)")
    name: str = Field(description="VLAN name")
    description: str = Field(description="VLAN purpose/description")


class DeviceConfig(BaseModel):
    """Complete device configuration."""
    hostname: str = Field(description="Device hostname")
    device_type: str = Field(description="Device type (router, switch, firewall)")
    interfaces: List[InterfaceConfig] = Field(description="Interface configurations")
    vlans: List[VLANConfig] = Field(default=[], description="VLAN configurations")
    routing_protocol: str = Field(default="", description="Routing protocol (ospf, bgp, eigrp)")
    management_ip: str = Field(description="Management IP address")


# Define agent state
class ConfigGenState(TypedDict):
    """State for config generation agent."""
    messages: List
    requirements: str
    device_config: dict
    generated_config: str
    validation_results: List[str]


# Define tools for config generation
@tool
def validate_ip_address(ip_address: str) -> str:
    """
    Validate IP address format.

    Args:
        ip_address: IP address to validate (e.g., 10.1.1.1/24)

    Returns:
        Validation result
    """
    import re

    # Simple validation
    pattern = r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$'
    if re.match(pattern, ip_address):
        # Check octets
        ip_part = ip_address.split('/')[0]
        octets = [int(x) for x in ip_part.split('.')]
        if all(0 <= octet <= 255 for octet in octets):
            return f"✓ Valid IP address: {ip_address}"

    return f"✗ Invalid IP address: {ip_address}"


@tool
def check_vlan_range(vlan_id: int) -> str:
    """
    Check if VLAN ID is in valid range.

    Args:
        vlan_id: VLAN ID to check

    Returns:
        Validation result
    """
    if 1 <= vlan_id <= 4094:
        if vlan_id == 1:
            return f"⚠️  VLAN 1 is default VLAN - consider using different VLAN for production"
        elif 1002 <= vlan_id <= 1005:
            return f"⚠️  VLAN {vlan_id} is reserved for Token Ring/FDDI"
        else:
            return f"✓ VLAN {vlan_id} is valid"
    else:
        return f"✗ VLAN {vlan_id} is out of range (1-4094)"


@tool
def generate_interface_config(interface_name: str, ip_address: str, description: str) -> str:
    """
    Generate interface configuration.

    Args:
        interface_name: Interface name
        ip_address: IP address with mask
        description: Interface description

    Returns:
        Configuration snippet
    """
    config = f"""
interface {interface_name}
 description {description}
 ip address {ip_address.replace('/', ' ')}
 no shutdown
"""
    return config.strip()


@tool
def generate_vlan_config(vlan_id: int, vlan_name: str) -> str:
    """
    Generate VLAN configuration.

    Args:
        vlan_id: VLAN ID
        vlan_name: VLAN name

    Returns:
        Configuration snippet
    """
    config = f"""
vlan {vlan_id}
 name {vlan_name}
"""
    return config.strip()


@tool
def generate_security_baseline() -> str:
    """
    Generate security baseline configuration.

    Returns:
        Security configuration snippet
    """
    config = """
! Security Baseline
service password-encryption
no ip http server
no ip http secure-server
ip ssh version 2
ip ssh time-out 60
ip ssh authentication-retries 3

! Disable unused services
no ip bootp server
no ip domain-lookup
no service pad

! Enable logging
logging buffered 16384 informational
logging console critical
logging trap informational
logging facility local5

! NTP configuration
ntp authenticate
ntp update-calendar

! SNMP hardening
no snmp-server
    """
    return config.strip()


# Create config generation agent
class ConfigGenerationAgent:
    """
    Agent that generates network configurations from requirements.

    Uses LangGraph for multi-step config generation and validation.
    """

    def __init__(self):
        """Initialize the agent."""
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0
        )
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the agent workflow graph."""

        def parse_requirements(state: ConfigGenState) -> ConfigGenState:
            """Parse requirements and plan configuration."""
            messages = state["messages"]

            system_msg = SystemMessage(content="""You are a network configuration expert.

Parse the user's requirements and create a structured plan for generating the configuration.

Identify:
1. Device type (router, switch, firewall)
2. Hostname
3. Interfaces needed (with IPs and purposes)
4. VLANs (if switch)
5. Routing protocol (if router)
6. Security requirements

Think step by step and be specific.""")

            response = self.llm.invoke([system_msg] + messages)
            messages.append(response)

            return {**state, "messages": messages}

        def generate_config(state: ConfigGenState) -> ConfigGenState:
            """Generate actual configuration using tools."""
            messages = state["messages"]

            # Bind tools
            llm_with_tools = self.llm.bind_tools([
                validate_ip_address,
                check_vlan_range,
                generate_interface_config,
                generate_vlan_config,
                generate_security_baseline
            ])

            # Ask to generate config
            prompt = HumanMessage(content="""Now generate the actual configuration.

Use the tools to:
1. Validate IP addresses
2. Check VLAN ranges
3. Generate interface configs
4. Generate VLAN configs (if needed)
5. Add security baseline

Call the appropriate tools.""")

            messages.append(prompt)
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            return {**state, "messages": messages}

        def execute_tools(state: ConfigGenState) -> ConfigGenState:
            """Execute config generation tools."""
            messages = state["messages"]
            last_message = messages[-1]
            generated_parts = []

            tool_calls = getattr(last_message, 'tool_calls', [])

            if not tool_calls:
                return state

            # Execute tools
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"  → {tool_name}({tool_args})")

                # Execute tool
                if tool_name == "validate_ip_address":
                    result = validate_ip_address.invoke(tool_args)
                elif tool_name == "check_vlan_range":
                    result = check_vlan_range.invoke(tool_args)
                elif tool_name == "generate_interface_config":
                    result = generate_interface_config.invoke(tool_args)
                    generated_parts.append(result)
                elif tool_name == "generate_vlan_config":
                    result = generate_vlan_config.invoke(tool_args)
                    generated_parts.append(result)
                elif tool_name == "generate_security_baseline":
                    result = generate_security_baseline.invoke(tool_args)
                    generated_parts.append(result)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(result)

            # Add results
            messages.append(HumanMessage(
                content=f"Tool results:\n" + "\n\n".join(tool_results)
            ))

            return {**state, "messages": messages, "generated_config": "\n\n".join(generated_parts)}

        def assemble_config(state: ConfigGenState) -> ConfigGenState:
            """Assemble final configuration."""
            messages = state["messages"]
            generated_config = state.get("generated_config", "")

            prompt = HumanMessage(content="""Based on the generated configuration parts, create a complete, production-ready configuration.

Include:
1. Hostname
2. All interfaces
3. VLANs (if applicable)
4. Routing protocol (if applicable)
5. Security baseline
6. Management settings

Format as a complete Cisco IOS configuration.""")

            messages.append(prompt)
            response = self.llm.invoke(messages)
            messages.append(response)

            final_config = response.content

            return {
                **state,
                "messages": messages,
                "generated_config": final_config
            }

        # Build graph
        workflow = StateGraph(ConfigGenState)

        # Add nodes
        workflow.add_node("parse", parse_requirements)
        workflow.add_node("generate", generate_config)
        workflow.add_node("execute", execute_tools)
        workflow.add_node("assemble", assemble_config)

        # Add edges
        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "generate")

        workflow.add_conditional_edges(
            "generate",
            lambda s: "execute" if s["messages"][-1].tool_calls else "assemble",
            {"execute": "execute", "assemble": "assemble"}
        )

        workflow.add_edge("execute", "assemble")
        workflow.add_edge("assemble", END)

        return workflow.compile()

    def generate(self, requirements: str) -> str:
        """
        Generate configuration from requirements.

        Args:
            requirements: High-level requirements

        Returns:
            Generated configuration
        """
        print(f"\n{'='*60}")
        print("Requirements:")
        print(f"{'='*60}")
        print(requirements)
        print(f"\n{'='*60}")
        print("Generating configuration...")
        print(f"{'='*60}\n")

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=requirements)],
            "requirements": requirements,
            "device_config": {},
            "generated_config": "",
            "validation_results": []
        }

        # Run agent
        final_state = self.graph.invoke(initial_state)

        return final_state["generated_config"]


def main():
    """Demo the config generation agent."""
    print("="*60)
    print("Network Configuration Generation Agent")
    print("="*60)

    try:
        # Create agent
        agent = ConfigGenerationAgent()

        # Test requirements
        requirements = [
            """
            Generate configuration for an access switch:
            - Hostname: ACCESS-SW-01
            - Management VLAN: 10 (10.0.10.0/24, IP: 10.0.10.10)
            - User VLAN: 20 (10.0.20.0/24)
            - Voice VLAN: 30 (10.0.30.0/24)
            - Uplink interface: GigabitEthernet0/1 (trunk to core)
            - Access ports: GigabitEthernet0/2-24 (VLAN 20, voice VLAN 30)
            - Include security baseline
            """,
            """
            Generate configuration for a branch router:
            - Hostname: BRANCH-RTR-01
            - WAN interface: GigabitEthernet0/0 (203.0.113.1/30)
            - LAN interface: GigabitEthernet0/1 (10.1.1.1/24)
            - Enable OSPF area 1 for LAN network
            - Default route pointing to 203.0.113.2
            - Include security hardening
            """
        ]

        for i, req in enumerate(requirements, 1):
            print(f"\n{'#'*60}")
            print(f"Example {i}")
            print(f"{'#'*60}")

            config = agent.generate(req.strip())

            print(f"\n{'='*60}")
            print("Generated Configuration:")
            print(f"{'='*60}")
            print(config)
            print()

        print("\n" + "="*60)
        print("✓ Config generation agent demo completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")
        print("  3. LangGraph installed: pip install langgraph")


if __name__ == "__main__":
    main()
