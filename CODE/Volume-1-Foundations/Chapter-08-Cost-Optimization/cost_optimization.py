#!/usr/bin/env python3
"""
Cost Optimization Techniques

Keep AI costs under control.

From: AI for Networking Engineers - Volume 1, Chapter 8
Author: Eduard Dulharu

Usage:
    python cost_optimization.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import time

load_dotenv()


def example_1_caching():
    """
    Example 1: Cache Responses

    Avoid repeat API calls for same questions.
    """
    print("\n" + "="*60)
    print("Example 1: Caching")
    print("="*60)

    # Enable caching
    set_llm_cache(InMemoryCache())

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    question = "What is OSPF?"

    # First call (hits API)
    print("\nFirst call (no cache):")
    start = time.time()
    response1 = llm.invoke(question)
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    print(f"  Answer: {response1.content[:80]}...")

    # Second call (from cache)
    print("\nSecond call (from cache):")
    start = time.time()
    response2 = llm.invoke(question)
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s (saved {time1-time2:.2f}s)")
    print(f"  Answer: {response2.content[:80]}...")


def example_2_use_cheaper_models():
    """
    Example 2: Use Cheaper Models

    Haiku for simple tasks, Sonnet for complex ones.
    """
    print("\n" + "="*60)
    print("Example 2: Model Selection")
    print("="*60)

    question = "What is BGP?"

    # Pricing per million tokens
    pricing = {
        'haiku': {'input': 0.80, 'output': 4.0},
        'sonnet': {'input': 3.0, 'output': 15.0}
    }

    print(f"\nQuestion: {question}\n")

    # Haiku (cheap)
    llm_haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    response_haiku = llm_haiku.invoke(question)
    est_tokens = len(question.split()) + len(response_haiku.content.split())
    cost_haiku = (est_tokens / 1_000_000) * (pricing['haiku']['input'] + pricing['haiku']['output']) / 2
    print(f"Haiku: ${cost_haiku:.6f} (fast, simple answer)")
    print(f"  {response_haiku.content[:100]}...\n")

    # Sonnet (better)
    llm_sonnet = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    response_sonnet = llm_sonnet.invoke(question)
    est_tokens = len(question.split()) + len(response_sonnet.content.split())
    cost_sonnet = (est_tokens / 1_000_000) * (pricing['sonnet']['input'] + pricing['sonnet']['output']) / 2
    print(f"Sonnet: ${cost_sonnet:.6f} (better, detailed answer)")
    print(f"  {response_sonnet.content[:100]}...")

    print(f"\nSavings with Haiku: ${cost_sonnet - cost_haiku:.6f}")


def example_3_batch_processing():
    """
    Example 3: Batch Processing

    Process multiple items in one call.
    """
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    interfaces = [
        "GigabitEthernet0/1",
        "GigabitEthernet0/2",
        "GigabitEthernet0/3"
    ]

    # Bad: 3 separate calls
    print("\nMethod 1: Separate calls (3x cost)")
    for intf in interfaces:
        # response = llm.invoke(f"Suggest a description for {intf}")
        pass  # Skip to save time
    print("  Would make 3 API calls = 3x cost")

    # Good: 1 batch call
    print("\nMethod 2: Batch call (1x cost)")
    batch_question = f"""Suggest descriptions for these interfaces:
{chr(10).join(interfaces)}

Format: interface_name: description"""

    response = llm.invoke(batch_question)
    print(f"  Makes 1 API call = saves 66% cost")
    print(f"\n{response.content}")


def example_4_prompt_optimization():
    """
    Example 4: Optimize Prompts

    Shorter prompts = lower costs.
    """
    print("\n" + "="*60)
    print("Example 4: Prompt Optimization")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # Verbose prompt (wastes tokens)
    verbose_prompt = """I have a networking question that I would like you to help me with.
I'm working on a project where I need to understand how OSPF works.
Could you please explain to me, in detail, what OSPF is and how it functions?
I would really appreciate your help with this. Thank you!"""

    # Concise prompt (saves tokens)
    concise_prompt = "Explain OSPF briefly."

    print(f"\nVerbose prompt: ~{len(verbose_prompt.split())} words")
    print(f"Concise prompt: ~{len(concise_prompt.split())} words")
    print(f"Tokens saved: ~{len(verbose_prompt.split()) - len(concise_prompt.split())} words\n")

    response = llm.invoke(concise_prompt)
    print(f"Result (concise):\n{response.content[:150]}...")


def main():
    """Run all examples."""
    print("="*60)
    print("Cost Optimization")
    print("="*60)

    try:
        example_1_caching()
        example_2_use_cheaper_models()
        example_3_batch_processing()
        example_4_prompt_optimization()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nCost Saving Strategies:")
        print("1. Cache repeated queries")
        print("2. Use Haiku for simple tasks")
        print("3. Batch process multiple items")
        print("4. Keep prompts concise")
        print("\nTypical savings: 50-70% with these techniques")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
