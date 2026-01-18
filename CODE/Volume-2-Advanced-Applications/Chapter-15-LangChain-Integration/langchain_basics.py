#!/usr/bin/env python3
"""
LangChain Basics for Network Engineers

Simple, clean examples of using LangChain for network tasks.

From: AI for Networking Engineers - Volume 2, Chapter 15
Author: Eduard Dulharu

Usage:
    python langchain_basics.py
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
load_dotenv()


# Define output schema
class ConfigIssue(BaseModel):
    """A network config issue found by AI."""
    severity: str = Field(description="Severity: critical, high, medium, low")
    issue: str = Field(description="Description of the issue")
    location: str = Field(description="Where in config this appears")
    fix: str = Field(description="How to fix it")


class ConfigAnalysis(BaseModel):
    """Complete config analysis."""
    issues: List[ConfigIssue] = Field(description="List of issues found")
    summary: str = Field(description="Overall assessment")


def example_1_simple_chain():
    """
    Example 1: Simple LLM Chain

    Analyze a config snippet using a basic chain.
    """
    print("\n" + "="*60)
    print("Example 1: Simple Config Analysis Chain")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create prompt template
    template = """You are a network engineer analyzing a device configuration.

Configuration:
{config}

Identify any security issues or misconfigurations.
Be specific and actionable.

Analysis:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["config"]
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Sample config with issues
    config = """
    hostname EDGE-RTR-01

    line vty 0 4
     transport input telnet
     password cisco123

    snmp-server community public RO
    snmp-server community private RW

    interface GigabitEthernet0/0
     ip address 10.1.1.1 255.255.255.0
     no shutdown
    """

    # Run chain
    result = chain.run(config=config)
    print("\n" + result)


def example_2_structured_output():
    """
    Example 2: Structured Output with Pydantic

    Get structured JSON output from config analysis.
    """
    print("\n" + "="*60)
    print("Example 2: Structured Output")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create output parser
    parser = PydanticOutputParser(pydantic_object=ConfigAnalysis)

    # Create prompt with format instructions
    template = """Analyze this network configuration for security issues.

Configuration:
{config}

{format_instructions}

Provide your analysis:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["config"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Sample config
    config = """
    hostname CORE-SW-01

    line vty 0 4
     transport input telnet

    snmp-server community public RO

    interface Vlan1
     ip address 192.168.1.1 255.255.255.0
     shutdown
    """

    # Run chain and parse
    result = chain.run(config=config)
    analysis = parser.parse(result)

    print(f"\nFound {len(analysis.issues)} issues:")
    for i, issue in enumerate(analysis.issues, 1):
        print(f"\n{i}. [{issue.severity.upper()}] {issue.issue}")
        print(f"   Location: {issue.location}")
        print(f"   Fix: {issue.fix}")

    print(f"\nSummary: {analysis.summary}")


def example_3_chat_template():
    """
    Example 3: Chat-style Prompts

    Use chat format for more natural interaction.
    """
    print("\n" + "="*60)
    print("Example 3: Chat-Style Analysis")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior network engineer with 20 years experience. Analyze configurations for best practices and security."),
        ("human", "Please review this BGP configuration:\n\n{config}\n\nWhat are the top 3 improvements I should make?")
    ])

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # BGP config
    config = """
    router bgp 65001
     neighbor 203.0.113.1 remote-as 174
     neighbor 203.0.113.1 description ISP_PRIMARY
     network 10.0.0.0 mask 255.0.0.0
    """

    # Run
    result = chain.run(config=config)
    print("\n" + result)


def example_4_config_generation():
    """
    Example 4: Config Generation

    Generate configs from requirements.
    """
    print("\n" + "="*60)
    print("Example 4: Config Generation")
    print("="*60)

    # Create LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create prompt
    template = """Generate a Cisco IOS configuration for the following requirements:

Requirements:
{requirements}

Provide a complete, production-ready configuration.
Include security best practices.

Configuration:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["requirements"]
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Requirements
    requirements = """
    - Device: Access switch
    - Hostname: ACCESS-SW-02
    - Management VLAN: 10 (10.0.10.0/24)
    - User VLAN: 20 (10.0.20.0/24)
    - Uplink: GigabitEthernet0/1 (trunk)
    - Access ports: GigabitEthernet0/2-24 (VLAN 20)
    """

    # Generate
    result = chain.run(requirements=requirements)
    print("\n" + result)


def main():
    """Run all examples."""
    print("="*60)
    print("LangChain Basics for Network Engineers")
    print("="*60)

    try:
        # Run examples
        example_1_simple_chain()
        example_2_structured_output()
        example_3_chat_template()
        example_4_config_generation()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
