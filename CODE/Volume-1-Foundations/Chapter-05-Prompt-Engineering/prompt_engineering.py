#!/usr/bin/env python3
"""
Prompt Engineering for Network Engineers

Practical prompt engineering techniques for network automation.

From: AI for Networking Engineers - Volume 1, Chapter 5
Author: Eduard Dulharu

Usage:
    python prompt_engineering.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Load environment variables
load_dotenv()


def example_1_basic_vs_detailed():
    """
    Example 1: Basic vs Detailed Prompts

    More detail = better results.
    """
    print("\n" + "="*60)
    print("Example 1: Basic vs Detailed Prompts")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    config = "interface GigabitEthernet0/1\n description WAN\n ip address 10.1.1.1 255.255.255.0"

    # Basic prompt (vague)
    print("\nBasic prompt: 'Analyze this config'")
    response1 = llm.invoke(f"Analyze this config:\n{config}")
    print(f"Result: {response1.content[:150]}...")

    # Detailed prompt (specific)
    detailed_prompt = f"""Analyze this Cisco IOS interface configuration:

{config}

Provide:
1. Interface purpose based on description
2. IP subnet and usable hosts
3. Any missing best practices (shutdown status, security)
4. Recommendations

Be specific and actionable."""

    print("\n\nDetailed prompt with specific requirements:")
    response2 = llm.invoke(detailed_prompt)
    print(f"Result:\n{response2.content}")


def example_2_role_prompting():
    """
    Example 2: Role-Based Prompting

    Give Claude a specific role for better context.
    """
    print("\n" + "="*60)
    print("Example 2: Role-Based Prompting")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    question = "Should I use OSPF or EIGRP for my campus network?"

    # Without role
    print("\nWithout role:")
    response1 = llm.invoke(question)
    print(f"{response1.content[:200]}...")

    # With expert role
    prompt_with_role = f"""You are a senior network architect with 20 years experience designing enterprise networks.

{question}

Consider factors like:
- Vendor interoperability
- Scalability
- Operational complexity
- Industry best practices"""

    print("\n\nWith expert role:")
    response2 = llm.invoke(prompt_with_role)
    print(response2.content)


def example_3_few_shot_prompting():
    """
    Example 3: Few-Shot Prompting

    Show examples of what you want.
    """
    print("\n" + "="*60)
    print("Example 3: Few-Shot Prompting")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    prompt = """Extract device info from syslog messages in this format:

Example 1:
Log: "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up"
Output: {"device": "unknown", "interface": "GigabitEthernet0/1", "event": "up"}

Example 2:
Log: "Jan 18 10:15:32 CORE-SW-01 %LINK-3-UPDOWN: Interface Vlan100, changed state to down"
Output: {"device": "CORE-SW-01", "interface": "Vlan100", "event": "down"}

Now extract from this:
Log: "Jan 18 10:20:05 EDGE-RTR-02 %OSPF-5-ADJCHG: Process 1, Nbr 10.1.1.1 on Gi0/0 from FULL to DOWN"
Output:"""

    response = llm.invoke(prompt)
    print(f"\nPrompt shows examples, then asks for extraction\n")
    print(f"Result:\n{response.content}")


def example_4_chain_of_thought():
    """
    Example 4: Chain of Thought Prompting

    Ask Claude to think step by step.
    """
    print("\n" + "="*60)
    print("Example 4: Chain of Thought")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Without chain of thought
    print("\nWithout chain of thought:")
    question = "A /24 network needs 5 subnets. What subnet mask should I use?"
    response1 = llm.invoke(question)
    print(f"{response1.content[:150]}...")

    # With chain of thought
    print("\n\nWith chain of thought:")
    cot_prompt = f"""{question}

Think step by step:
1. How many subnets do we need?
2. How many bits do we need to borrow?
3. What's the new subnet mask?
4. How many hosts per subnet?
5. Show the subnet ranges"""

    response2 = llm.invoke(cot_prompt)
    print(response2.content)


def example_5_constraints():
    """
    Example 5: Adding Constraints

    Control the format and length of responses.
    """
    print("\n" + "="*60)
    print("Example 5: Adding Constraints")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    question = "Explain how BGP route selection works"

    # Without constraints (verbose)
    print("\nWithout constraints:")
    response1 = llm.invoke(question)
    print(f"Length: {len(response1.content)} characters")
    print(f"Preview: {response1.content[:150]}...")

    # With constraints (concise)
    print("\n\nWith constraints:")
    constrained_prompt = f"""{question}

Requirements:
- Maximum 3 sentences
- Use bullet points
- Focus only on the top 3 criteria
- No additional explanations"""

    response2 = llm.invoke(constrained_prompt)
    print(f"Length: {len(response2.content)} characters")
    print(f"Response:\n{response2.content}")


def example_6_prompt_templates():
    """
    Example 6: Reusable Prompt Templates

    Create templates for common tasks.
    """
    print("\n" + "="*60)
    print("Example 6: Prompt Templates")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Create a template for config analysis
    template = PromptTemplate(
        template="""You are a network security auditor. Analyze this {device_type} configuration:

{config}

Check for:
1. Security issues (weak passwords, insecure protocols)
2. Best practice violations
3. Missing security features

Severity: Mark as CRITICAL, HIGH, MEDIUM, or LOW

Analysis:""",
        input_variables=["device_type", "config"]
    )

    # Test configs
    configs = [
        {
            "device_type": "Cisco router",
            "config": """
line vty 0 4
 transport input telnet
 password cisco123
"""
        },
        {
            "device_type": "Cisco switch",
            "config": """
snmp-server community public RO
snmp-server community private RW
"""
        }
    ]

    print("\nUsing reusable template for multiple configs:\n")

    for i, cfg in enumerate(configs, 1):
        prompt = template.format(**cfg)
        response = llm.invoke(prompt)

        print(f"Config {i} ({cfg['device_type']}):")
        print(response.content)
        print("\n" + "-"*60 + "\n")


def example_7_negative_prompting():
    """
    Example 7: Negative Prompting

    Tell Claude what NOT to do.
    """
    print("\n" + "="*60)
    print("Example 7: Negative Prompting")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    question = "How do I configure OSPF on a Cisco router?"

    # With negative instructions
    prompt = f"""{question}

IMPORTANT - DO NOT:
- Provide full config dumps
- Explain OSPF theory
- Include optional features
- Use verbose explanations

ONLY provide:
- The 5 essential commands
- One-line comment for each command"""

    response = llm.invoke(prompt)

    print(f"\nPrompt tells Claude what NOT to do:")
    print(f"\nResult:\n{response.content}")


def main():
    """Run all examples."""
    print("="*60)
    print("Prompt Engineering for Network Engineers")
    print("="*60)

    try:
        example_1_basic_vs_detailed()
        example_2_role_prompting()
        example_3_few_shot_prompting()
        example_4_chain_of_thought()
        example_5_constraints()
        example_6_prompt_templates()
        example_7_negative_prompting()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Techniques:")
        print("1. Be specific - detailed prompts get better results")
        print("2. Use roles - give Claude expertise context")
        print("3. Show examples - few-shot learning works well")
        print("4. Think step-by-step - chain of thought reasoning")
        print("5. Add constraints - control format and length")
        print("6. Use templates - reuse prompts for consistency")
        print("7. Use negative instructions - tell it what to avoid")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install langchain langchain-anthropic python-dotenv")


if __name__ == "__main__":
    main()
