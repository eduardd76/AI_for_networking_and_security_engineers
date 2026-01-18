#!/usr/bin/env python3
"""
API Basics for Network Engineers

Simple, practical examples of using Claude API with LangChain.

From: AI for Networking Engineers - Volume 1, Chapter 4
Author: Eduard Dulharu

Usage:
    python api_basics.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()


def example_1_simple_call():
    """
    Example 1: Basic API Call

    The simplest way to call Claude using LangChain.
    """
    print("\n" + "="*60)
    print("Example 1: Simple API Call")
    print("="*60)

    # Create LLM instance
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    # Make a simple call
    question = "What is BGP in one sentence?"
    response = llm.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {response.content}")


def example_2_with_context():
    """
    Example 2: API Call with Context

    Pass context along with your question.
    """
    print("\n" + "="*60)
    print("Example 2: API Call with Context")
    print("="*60)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    # Network config context
    config = """
    router bgp 65001
     neighbor 203.0.113.1 remote-as 174
     neighbor 203.0.113.1 shutdown
     network 10.0.0.0 mask 255.0.0.0
    """

    question = f"""Here's a BGP configuration:

{config}

Is there a problem with this config?"""

    response = llm.invoke(question)

    print(f"\nConfig:\n{config}")
    print(f"\nAnalysis:\n{response.content}")


def example_3_streaming():
    """
    Example 3: Streaming Responses

    Stream responses token by token (good for long responses).
    """
    print("\n" + "="*60)
    print("Example 3: Streaming Response")
    print("="*60)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    question = "Explain the 7 layers of the OSI model in simple terms."

    print(f"\nQuestion: {question}\n")
    print("Answer (streaming):")

    # Stream the response
    for chunk in llm.stream(question):
        print(chunk.content, end="", flush=True)

    print("\n")


def example_4_error_handling():
    """
    Example 4: Basic Error Handling

    Handle common API errors gracefully.
    """
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    questions = [
        "What is OSPF?",
        "Explain VLAN trunking",
        "What is STP?"
    ]

    print("\nProcessing multiple questions with error handling:\n")

    for i, question in enumerate(questions, 1):
        try:
            response = llm.invoke(question)
            print(f"{i}. {question}")
            print(f"   ✓ Answer: {response.content[:80]}...")
            print()

        except Exception as e:
            print(f"{i}. {question}")
            print(f"   ✗ Error: {str(e)}")
            print()


def example_5_temperature():
    """
    Example 5: Temperature Settings

    Control randomness in responses.
    """
    print("\n" + "="*60)
    print("Example 5: Temperature Settings")
    print("="*60)

    question = "Suggest a creative name for a network automation tool"

    print(f"\nQuestion: {question}\n")

    # Temperature 0 (deterministic)
    llm_deterministic = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )
    response1 = llm_deterministic.invoke(question)
    print(f"Temperature 0 (deterministic):\n{response1.content}\n")

    # Temperature 1 (creative)
    llm_creative = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=1
    )
    response2 = llm_creative.invoke(question)
    print(f"Temperature 1 (creative):\n{response2.content}\n")


def example_6_cost_tracking():
    """
    Example 6: Track API Costs

    Monitor token usage and costs.
    """
    print("\n" + "="*60)
    print("Example 6: Cost Tracking")
    print("="*60)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    # Pricing (per million tokens)
    pricing = {
        'input': 3.0,   # $3 per million input tokens
        'output': 15.0  # $15 per million output tokens
    }

    questions = [
        "What is OSPF?",
        "Explain BGP route selection",
        "What is VLAN pruning?"
    ]

    print("\nTracking costs for multiple API calls:\n")

    total_input_tokens = 0
    total_output_tokens = 0

    for question in questions:
        response = llm.invoke(question)

        # Get token usage
        input_tokens = len(question.split()) * 1.3  # Rough estimate
        output_tokens = len(response.content.split()) * 1.3

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        call_cost = input_cost + output_cost

        print(f"Q: {question}")
        print(f"   Tokens: ~{int(input_tokens)} in, ~{int(output_tokens)} out")
        print(f"   Cost: ${call_cost:.6f}")
        print()

    # Total cost
    total_input_cost = (total_input_tokens / 1_000_000) * pricing['input']
    total_output_cost = (total_output_tokens / 1_000_000) * pricing['output']
    total_cost = total_input_cost + total_output_cost

    print(f"Total tokens: ~{int(total_input_tokens)} in, ~{int(total_output_tokens)} out")
    print(f"Total cost: ${total_cost:.6f}")


def example_7_model_comparison():
    """
    Example 7: Compare Different Models

    Try different Claude models and see the difference.
    """
    print("\n" + "="*60)
    print("Example 7: Model Comparison")
    print("="*60)

    question = "Explain the difference between TCP and UDP in networking."

    models = [
        ("claude-3-5-haiku-20241022", "Haiku (fast, cheap)"),
        ("claude-3-5-sonnet-20241022", "Sonnet (balanced)")
    ]

    print(f"\nQuestion: {question}\n")

    for model_id, model_name in models:
        llm = ChatAnthropic(model=model_id, temperature=0)

        import time
        start = time.time()
        response = llm.invoke(question)
        duration = time.time() - start

        print(f"{model_name}:")
        print(f"  Time: {duration:.2f}s")
        print(f"  Answer: {response.content[:150]}...")
        print()


def main():
    """Run all examples."""
    print("="*60)
    print("API Basics for Network Engineers")
    print("="*60)

    try:
        example_1_simple_call()
        example_2_with_context()
        example_3_streaming()
        example_4_error_handling()
        example_5_temperature()
        example_6_cost_tracking()
        example_7_model_comparison()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. LangChain makes API calls simple")
        print("2. Always pass context for better answers")
        print("3. Use streaming for long responses")
        print("4. Handle errors gracefully")
        print("5. Temperature controls creativity")
        print("6. Monitor costs to stay in budget")
        print("7. Choose models based on speed/cost/quality needs")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install langchain langchain-anthropic python-dotenv")


if __name__ == "__main__":
    main()
