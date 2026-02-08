#!/usr/bin/env python3
"""
Network Troubleshooting Agent

An autonomous agent that helps diagnose and fix network issues.

From: AI for Networking Engineers - Volume 2, Chapter 20
Author: Eduard Dulharu

Usage:
    python troubleshoot_agent.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()


# Define agent state
class TroubleshootState(TypedDict):
    """State for troubleshooting agent."""
    messages: List
    problem: str
    diagnostics: List[str]
    root_cause: str
    solution: str
    confidence: str


# Define diagnostic tools
@tool
def check_interface_status(interface: str) -> str:
    """
    Check interface status and errors.

    Args:
        interface: Interface name (e.g., GigabitEthernet0/0)

    Returns:
        Interface status output
    """
    # Simulated output - in production, use Netmiko
    mock_outputs = {
        "GigabitEthernet0/0": """
GigabitEthernet0/0 is up, line protocol is down
  Hardware is iGbE, address is 0050.56a1.2345
  Internet address is 10.1.1.1/24
  MTU 1500 bytes, BW 1000000 Kbit/sec
  Full-duplex, 1000Mb/s, media type is RJ45
  Input queue: 0/75/0/0 (size/max/drops/flushes)
  5 minute input rate 0 bits/sec, 0 packets/sec
  5 minute output rate 0 bits/sec, 0 packets/sec
     0 packets input, 0 bytes, 0 no buffer
     0 input errors, 0 CRC, 0 frame, 0 overrun, 0 ignored
     0 output errors, 5 collisions, 2 interface resets
        """,
        "GigabitEthernet0/1": """
GigabitEthernet0/1 is up, line protocol is up
  Hardware is iGbE, address is 0050.56a1.2346
  Internet address is 10.1.2.1/24
  MTU 1500 bytes, BW 1000000 Kbit/sec
  Full-duplex, 1000Mb/s, media type is RJ45
  5 minute input rate 1000 bits/sec, 2 packets/sec
  5 minute output rate 2000 bits/sec, 3 packets/sec
     12345 packets input, 1234567 bytes
     0 input errors, 0 CRC, 0 frame
     23456 packets output, 2345678 bytes
     0 output errors, 0 collisions
        """
    }

    return mock_outputs.get(interface, f"Interface {interface} not found")


@tool
def check_routing_table(prefix: str = "") -> str:
    """
    Check routing table for a specific prefix.

    Args:
        prefix: IP prefix to check (optional)

    Returns:
        Routing table output
    """
    if prefix == "10.2.0.0":
        return """
Routing entry for 10.2.0.0/24
  Known via "ospf 1", distance 110, metric 20
  Redistributing via ospf 1
  Last update from 10.1.1.2 on GigabitEthernet0/0, 00:05:23 ago
  Routing Descriptor Blocks:
  * 10.1.1.2, from 10.2.0.1, via GigabitEthernet0/0
      Route metric is 20, traffic share count is 1
        """
    else:
        return """
Gateway of last resort is 10.1.1.254 to network 0.0.0.0

S*    0.0.0.0/0 [1/0] via 10.1.1.254
      10.0.0.0/8 is variably subnetted, 3 subnets
C        10.1.1.0/24 is directly connected, GigabitEthernet0/0
L        10.1.1.1/32 is directly connected, GigabitEthernet0/0
O        10.2.0.0/24 [110/20] via 10.1.1.2, GigabitEthernet0/0
        """


@tool
def check_bgp_neighbor(neighbor_ip: str) -> str:
    """
    Check BGP neighbor status.

    Args:
        neighbor_ip: BGP neighbor IP address

    Returns:
        BGP neighbor output
    """
    mock_output = f"""
BGP neighbor is {neighbor_ip}, remote AS 65002, external link
  BGP version 4, remote router ID 10.2.0.1
  BGP state = Active
  Last read 00:00:45, last write 00:00:45
  Connections established 0; dropped 15
  Connection state is IDLE

Failed connection attempts = 15

External BGP neighbor not directly connected.
    """
    return mock_output


@tool
def check_ospf_neighbors() -> str:
    """
    Check OSPF neighbor relationships.

    Returns:
        OSPF neighbor output
    """
    return """
Neighbor ID     Pri   State           Dead Time   Address         Interface
10.1.1.2          1   FULL/DR         00:00:35    10.1.1.2        GigabitEthernet0/0
10.1.1.3          1   INIT/DROTHER    00:00:32    10.1.1.3        GigabitEthernet0/1
    """


@tool
def ping_test(target: str) -> str:
    """
    Test connectivity with ping.

    Args:
        target: IP address or hostname to ping

    Returns:
        Ping results
    """
    if "10.1.1.2" in target:
        return """
Sending 5, 100-byte ICMP Echos to 10.1.1.2, timeout is 2 seconds:
!!!!!
Success rate is 100 percent (5/5), round-trip min/avg/max = 1/2/4 ms
        """
    elif "10.2.0.1" in target:
        return """
Sending 5, 100-byte ICMP Echos to 10.2.0.1, timeout is 2 seconds:
.....
Success rate is 0 percent (0/5)
        """
    else:
        return f"Cannot reach {target}"


# Create troubleshooting agent
class TroubleshootingAgent:
    """
    Autonomous troubleshooting agent using LangGraph.

    Can diagnose common network issues and suggest fixes.
    """

    def __init__(self):
        """Initialize the agent."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the agent workflow graph."""

        def analyze_problem(state: TroubleshootState) -> TroubleshootState:
            """Analyze the problem and decide which tools to use."""
            messages = state["messages"]

            # Add system message with tool descriptions
            system_msg = SystemMessage(content="""You are a network troubleshooting expert.

Analyze the problem and use available tools to diagnose the issue.
Available tools: check_interface_status, check_routing_table, check_bgp_neighbor,
check_ospf_neighbors, ping_test

Think step by step:
1. What symptoms are described?
2. What could cause these symptoms?
3. Which diagnostic commands should we run?
4. Call the appropriate tools

Be specific with tool parameters.""")

            # Call LLM with tools
            llm_with_tools = self.llm.bind_tools([
                check_interface_status,
                check_routing_table,
                check_bgp_neighbor,
                check_ospf_neighbors,
                ping_test
            ])

            response = llm_with_tools.invoke([system_msg] + messages)
            messages.append(response)

            return {**state, "messages": messages}

        def execute_tools(state: TroubleshootState) -> TroubleshootState:
            """Execute requested diagnostic tools."""
            messages = state["messages"]
            last_message = messages[-1]
            diagnostics = state.get("diagnostics", [])

            # Get tool calls
            tool_calls = getattr(last_message, 'tool_calls', [])

            if not tool_calls:
                return state

            # Execute each tool
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"\n  → Running: {tool_name}({tool_args})")

                # Call appropriate tool
                if tool_name == "check_interface_status":
                    result = check_interface_status.invoke(tool_args)
                elif tool_name == "check_routing_table":
                    result = check_routing_table.invoke(tool_args)
                elif tool_name == "check_bgp_neighbor":
                    result = check_bgp_neighbor.invoke(tool_args)
                elif tool_name == "check_ospf_neighbors":
                    result = check_ospf_neighbors.invoke(tool_args)
                elif tool_name == "ping_test":
                    result = ping_test.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(f"{tool_name}: {result}")
                diagnostics.append(f"{tool_name}({tool_args}): {result[:100]}...")

            # Add results to messages
            messages.append(HumanMessage(
                content=f"Tool results:\n" + "\n\n".join(tool_results)
            ))

            return {**state, "messages": messages, "diagnostics": diagnostics}

        def diagnose(state: TroubleshootState) -> TroubleshootState:
            """Analyze diagnostic results and provide solution."""
            messages = state["messages"]

            # Ask LLM to provide root cause and solution
            diagnosis_prompt = HumanMessage(content="""Based on the diagnostic results, provide:

1. Root Cause: What is causing the problem?
2. Solution: Step-by-step fix
3. Confidence: High/Medium/Low

Be specific and actionable.""")

            messages.append(diagnosis_prompt)
            response = self.llm.invoke(messages)
            messages.append(response)

            # Extract solution
            solution = response.content

            return {
                **state,
                "messages": messages,
                "solution": solution,
                "confidence": "High"  # Would parse from response in production
            }

        def should_continue(state: TroubleshootState) -> str:
            """Decide if more diagnostics are needed."""
            messages = state["messages"]

            if len(messages) < 2:
                return "analyze"

            last_message = messages[-1]

            # If last message has tool calls, execute them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            # If we've run tools, provide diagnosis
            if state.get("diagnostics"):
                return "diagnose"

            # Otherwise, start analysis
            return "analyze"

        # Build graph
        workflow = StateGraph(TroubleshootState)

        # Add nodes
        workflow.add_node("analyze", analyze_problem)
        workflow.add_node("tools", execute_tools)
        workflow.add_node("diagnose", diagnose)

        # Add edges
        workflow.set_entry_point("analyze")

        workflow.add_conditional_edges(
            "analyze",
            lambda s: "tools" if s["messages"][-1].tool_calls else "diagnose",
            {"tools": "tools", "diagnose": "diagnose"}
        )

        workflow.add_edge("tools", "diagnose")
        workflow.add_edge("diagnose", END)

        return workflow.compile()

    def troubleshoot(self, problem: str) -> str:
        """
        Troubleshoot a network problem.

        Args:
            problem: Description of the issue

        Returns:
            Diagnosis and solution
        """
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")
        print("\nRunning diagnostics...")

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=problem)],
            "problem": problem,
            "diagnostics": [],
            "root_cause": "",
            "solution": "",
            "confidence": ""
        }

        # Run agent
        final_state = self.graph.invoke(initial_state)

        return final_state["solution"]


def main():
    """Demo the troubleshooting agent."""
    print("="*60)
    print("Network Troubleshooting Agent")
    print("="*60)

    try:
        # Create agent
        agent = TroubleshootingAgent()

        # Test problems
        problems = [
            "GigabitEthernet0/0 shows up/down - interface is up but protocol is down",
            "Cannot ping 10.2.0.1 but can ping 10.1.1.2",
            "OSPF neighbor 10.1.1.3 stuck in INIT state"
        ]

        for i, problem in enumerate(problems, 1):
            print(f"\n{'#'*60}")
            print(f"Test Case {i}")
            print(f"{'#'*60}")

            solution = agent.troubleshoot(problem)

            print(f"\n{'='*60}")
            print("Solution:")
            print(f"{'='*60}")
            print(solution)
            print()

        print("\n" + "="*60)
        print("✓ Troubleshooting agent demo completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")
        print("  3. LangGraph installed: pip install langgraph")


if __name__ == "__main__":
    main()
