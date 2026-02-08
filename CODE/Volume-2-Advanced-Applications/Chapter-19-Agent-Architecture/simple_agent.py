#!/usr/bin/env python3
"""
Simple Network Agent with LangGraph

A clean, easy-to-understand agent that can analyze network configs
and answer questions using tools.

From: AI for Networking Engineers - Volume 2, Chapter 19
Author: Eduard Dulharu

Usage:
    python simple_agent.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()


# Define the agent state
class AgentState(TypedDict):
    """State that flows through the agent."""
    messages: list
    config_content: str
    analysis_result: str


# Define tools the agent can use
@tool
def get_config_snippet(section: str) -> str:
    """
    Get a specific section from a network config.

    Args:
        section: Section name like 'interface', 'router bgp', 'line vty'

    Returns:
        The config section text
    """
    # Mock config database
    mock_config = {
        "interface": """
interface GigabitEthernet0/0
 description WAN_LINK
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description LAN
 ip address 10.0.0.1 255.255.255.0
 no shutdown
        """,

        "router bgp": """
router bgp 65001
 neighbor 203.0.113.2 remote-as 174
 neighbor 203.0.113.2 description ISP_PRIMARY
 network 10.0.0.0 mask 255.0.0.0
        """,

        "line vty": """
line vty 0 4
 transport input ssh
 login local
        """,

        "snmp": """
snmp-server community public RO
snmp-server location DataCenter-1
        """
    }

    result = mock_config.get(section.lower(), "Section not found")
    return f"Config section '{section}':\n{result}"


@tool
def check_security_issue(config_snippet: str) -> str:
    """
    Check if a config snippet has security issues.

    Args:
        config_snippet: Config text to check

    Returns:
        Security assessment
    """
    issues = []

    # Simple pattern matching for demo
    if 'telnet' in config_snippet.lower():
        issues.append("CRITICAL: Telnet is insecure, use SSH")

    if 'community public' in config_snippet.lower():
        issues.append("HIGH: Default SNMP community 'public' detected")

    if 'password' in config_snippet.lower() and 'enable secret' not in config_snippet.lower():
        issues.append("MEDIUM: Weak password encryption")

    if not issues:
        return "✓ No obvious security issues found"

    return "Security issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)


# Create simple agent
class NetworkAgent:
    """
    Simple network analysis agent.

    Can retrieve config sections and analyze them for security issues.
    """

    def __init__(self):
        """Initialize the agent."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

        # Create the agent graph
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the agent's workflow graph."""

        # Define workflow steps
        def should_continue(state: AgentState) -> str:
            """Decide if agent should continue or end."""
            messages = state["messages"]

            # If last message is from AI and has no tool calls, we're done
            if isinstance(messages[-1], AIMessage):
                if not hasattr(messages[-1], 'tool_calls') or not messages[-1].tool_calls:
                    return "end"

            return "continue"

        def call_tools(state: AgentState) -> AgentState:
            """Execute tools requested by the agent."""
            messages = state["messages"]
            last_message = messages[-1]

            # Get tool calls from last message
            tool_calls = getattr(last_message, 'tool_calls', [])

            if not tool_calls:
                return state

            # Execute each tool
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Call the appropriate tool
                if tool_name == "get_config_snippet":
                    result = get_config_snippet.invoke(tool_args)
                elif tool_name == "check_security_issue":
                    result = check_security_issue.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(result)

            # Add tool results to messages
            messages.append(HumanMessage(content=f"Tool results: {' | '.join(tool_results)}"))

            return {**state, "messages": messages}

        def call_model(state: AgentState) -> AgentState:
            """Call the LLM with current state."""
            messages = state["messages"]

            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools([
                get_config_snippet,
                check_security_issue
            ])

            # Get response
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            return {**state, "messages": messages}

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)

        # Add edges
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def run(self, question: str) -> str:
        """
        Run the agent with a question.

        Args:
            question: Question to ask the agent

        Returns:
            Agent's final answer
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "config_content": "",
            "analysis_result": ""
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Get final answer
        last_message = final_state["messages"][-1]
        return last_message.content


def main():
    """Demo the network agent."""
    print("="*60)
    print("Simple Network Agent with LangGraph")
    print("="*60)

    try:
        # Create agent
        print("\nInitializing agent...")
        agent = NetworkAgent()

        # Test questions
        questions = [
            "Check the 'line vty' configuration for security issues",
            "Get the BGP configuration and analyze it",
            "What's in the SNMP configuration? Is it secure?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n{'-'*60}")
            print(f"Question {i}: {question}")
            print(f"{'-'*60}")

            answer = agent.run(question)
            print(f"\nAnswer:\n{answer}")

        print("\n" + "="*60)
        print("✓ Agent demo completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")
        print("  3. LangGraph installed: pip install langgraph")


if __name__ == "__main__":
    main()
