"""
Chapter 2: Introduction to LLMs
Tokenizer Playground - Understand Tokens and Costs

This script helps you visualize how text is tokenized and estimate costs.
Essential for understanding LLM pricing and context limits.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
import tiktoken  # OpenAI's tokenizer library
from anthropic import Anthropic

# Load environment variables
load_dotenv()


def count_tokens_openai(text: str, model: str = "gpt-4") -> int:
    """Count tokens using OpenAI's tiktoken library"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback to cl100k_base encoding (GPT-4 default)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)


def count_tokens_anthropic(text: str) -> int:
    """Count tokens using Anthropic's API"""
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Anthropic counts tokens via API
        response = client.messages.count_tokens(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens
    except Exception as e:
        # Fallback estimate: ~4 chars per token
        return len(text) // 4


def visualize_tokens(text: str):
    """
    Example 1: Visualize how text is broken into tokens
    """
    print("=" * 60)
    print("Example 1: Token Visualization")
    print("=" * 60)

    # Sample network config
    config = """interface GigabitEthernet0/1
 ip address 192.168.1.1 255.255.255.0
 no shutdown"""

    print(f"Input Text:\n{config}\n")

    # Tokenize with OpenAI encoder
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(config)

    print(f"Total Tokens: {len(tokens)}")
    print(f"\nToken Breakdown:")
    print("-" * 60)

    for i, token_id in enumerate(tokens[:20], 1):  # Show first 20 tokens
        token_text = encoding.decode([token_id])
        # Replace newlines with visible symbol
        token_display = token_text.replace('\n', '↵')
        print(f"Token {i:2d}: '{token_display}' (ID: {token_id})")

    if len(tokens) > 20:
        print(f"... and {len(tokens) - 20} more tokens")

    print("\n" + "=" * 60 + "\n")


def estimate_costs():
    """
    Example 2: Estimate API costs for different models
    """
    print("=" * 60)
    print("Example 2: Cost Estimation")
    print("=" * 60)

    # Sample: Large config file (typical router config)
    config = """hostname CORE-RTR-01
!
interface GigabitEthernet0/0
 description WAN Link to ISP
 ip address 203.0.113.1 255.255.255.252
 ip nat outside
 no shutdown
!
interface GigabitEthernet0/1
 description LAN Subnet
 ip address 192.168.1.1 255.255.255.0
 ip nat inside
 no shutdown
!
router bgp 65001
 neighbor 203.0.113.2 remote-as 65000
 network 192.168.1.0 mask 255.255.255.0
!
ip nat inside source list 1 interface GigabitEthernet0/0 overload
!
access-list 1 permit 192.168.1.0 0.0.0.255
!
end"""

    print(f"Configuration Size: {len(config)} characters\n")

    # Count tokens
    token_count = count_tokens_openai(config, "gpt-4")
    print(f"Token Count (GPT-4 encoding): {token_count} tokens\n")

    # Pricing (as of 2024)
    models = {
        "GPT-4o": {"input": 2.50, "output": 10.00},
        "GPT-4o mini": {"input": 0.15, "output": 0.60},
        "Claude 3.5 Sonnet": {"input": 3.00, "output": 15.00},
        "Claude 3.5 Haiku": {"input": 0.80, "output": 4.00},
    }

    print("Cost Estimates (per 1M tokens):")
    print("-" * 60)

    for model_name, prices in models.items():
        input_cost = (token_count / 1_000_000) * prices["input"]
        # Assume response is ~500 tokens
        output_cost = (500 / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost

        print(f"{model_name:20s}")
        print(f"  Input:  ${input_cost:.6f}")
        print(f"  Output: ${output_cost:.6f}")
        print(f"  Total:  ${total_cost:.6f}\n")

    print("💡 Tip: Use cheaper models (Haiku/mini) for development!")
    print("=" * 60 + "\n")


def context_window_limits():
    """
    Example 3: Understand context window limits
    """
    print("=" * 60)
    print("Example 3: Context Window Limits")
    print("=" * 60)

    # Model context windows
    models = {
        "GPT-4o": 128000,
        "GPT-4o mini": 128000,
        "Claude 3.5 Sonnet": 200000,
        "Claude 3.5 Haiku": 200000,
        "Llama 3.1 8B": 128000,
    }

    print("Model Context Windows:")
    print("-" * 60)
    for model, tokens in models.items():
        print(f"{model:20s}: {tokens:>7,} tokens")

    print("\n")

    # Example scenario
    config_size = 50000  # tokens
    conversation_history = 10000  # tokens
    system_prompt = 500  # tokens

    total_input = config_size + conversation_history + system_prompt
    output_reserve = 4000  # tokens for response

    print("Example Scenario:")
    print("-" * 60)
    print(f"Config file:          {config_size:>7,} tokens")
    print(f"Conversation history: {conversation_history:>7,} tokens")
    print(f"System prompt:        {system_prompt:>7,} tokens")
    print(f"Output reserve:       {output_reserve:>7,} tokens")
    print(f"{'Total Required:':20s} {total_input + output_reserve:>7,} tokens")

    print("\n✅ Fits in:")
    for model, max_tokens in models.items():
        if total_input + output_reserve <= max_tokens:
            remaining = max_tokens - (total_input + output_reserve)
            print(f"  {model:20s} ({remaining:>7,} tokens spare)")

    print("\n❌ Too large for:")
    for model, max_tokens in models.items():
        if total_input + output_reserve > max_tokens:
            shortage = (total_input + output_reserve) - max_tokens
            print(f"  {model:20s} (exceeds by {shortage:>7,} tokens)")

    print("\n" + "=" * 60 + "\n")


def batch_cost_analysis():
    """
    Example 4: Analyze costs for batch processing
    """
    print("=" * 60)
    print("Example 4: Batch Processing Cost Analysis")
    print("=" * 60)

    # Scenario: Analyze 1000 router configs
    num_configs = 1000
    avg_tokens_per_config = 2000
    avg_output_tokens = 500

    print(f"Scenario: Analyze {num_configs} router configurations")
    print(f"Average input: {avg_tokens_per_config} tokens per config")
    print(f"Average output: {avg_output_tokens} tokens per response\n")

    total_input_tokens = num_configs * avg_tokens_per_config
    total_output_tokens = num_configs * avg_output_tokens

    print(f"Total Input Tokens:  {total_input_tokens:>12,}")
    print(f"Total Output Tokens: {total_output_tokens:>12,}\n")

    # Calculate costs
    models = {
        "Claude 3.5 Haiku": {"input": 0.80, "output": 4.00},
        "Claude 3.5 Sonnet": {"input": 3.00, "output": 15.00},
        "GPT-4o mini": {"input": 0.15, "output": 0.60},
    }

    print("Total Cost Comparison:")
    print("-" * 60)

    for model_name, prices in models.items():
        input_cost = (total_input_tokens / 1_000_000) * prices["input"]
        output_cost = (total_output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost

        print(f"{model_name:20s}: ${total_cost:>8.2f}")

    print("\n💰 Cost Savings:")
    sonnet_cost = (total_input_tokens / 1_000_000) * 3.00 + (total_output_tokens / 1_000_000) * 15.00
    haiku_cost = (total_input_tokens / 1_000_000) * 0.80 + (total_output_tokens / 1_000_000) * 4.00
    mini_cost = (total_input_tokens / 1_000_000) * 0.15 + (total_output_tokens / 1_000_000) * 0.60

    savings_haiku = ((sonnet_cost - haiku_cost) / sonnet_cost) * 100
    savings_mini = ((sonnet_cost - mini_cost) / sonnet_cost) * 100

    print(f"  Using Haiku vs Sonnet: {savings_haiku:.1f}% savings")
    print(f"  Using mini vs Sonnet:  {savings_mini:.1f}% savings")

    print("\n" + "=" * 60 + "\n")


def real_world_example():
    """
    Example 5: Real API call with token counting
    """
    print("=" * 60)
    print("Example 5: Real API Call with Token Tracking")
    print("=" * 60)

    config = """interface GigabitEthernet0/1
 description User VLAN
 switchport mode access
 switchport access vlan 10
 spanning-tree portfast"""

    # Count tokens before API call
    input_tokens = count_tokens_openai(config, "gpt-4")
    print(f"Input: {len(config)} chars → {input_tokens} tokens\n")

    prompt = f"Analyze this switch interface config and identify any best practices:\n\n{config}"

    print("Calling API...")
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    response = llm.invoke(prompt)

    # Estimate output tokens
    output_tokens = count_tokens_openai(response.content, "gpt-4")

    print(f"\n✅ Response received")
    print(f"Output: {len(response.content)} chars → ~{output_tokens} tokens\n")

    # Calculate actual cost
    input_cost = (input_tokens / 1_000_000) * 0.80  # Haiku input
    output_cost = (output_tokens / 1_000_000) * 4.00  # Haiku output
    total_cost = input_cost + output_cost

    print(f"Cost Breakdown:")
    print(f"  Input:  {input_tokens:>6} tokens × $0.80/1M = ${input_cost:.6f}")
    print(f"  Output: {output_tokens:>6} tokens × $4.00/1M = ${output_cost:.6f}")
    print(f"  Total: ${total_cost:.6f}")

    print(f"\n📊 Response Preview:")
    print("-" * 60)
    print(response.content[:200] + "...")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔢 Chapter 2: Introduction to LLMs")
    print("Tokenizer Playground & Cost Estimation\n")

    try:
        # Run examples
        visualize_tokens("Sample network config")
        input("Press Enter to continue...")

        estimate_costs()
        input("Press Enter to continue...")

        context_window_limits()
        input("Press Enter to continue...")

        batch_cost_analysis()
        input("Press Enter to continue...")

        # Only run real API call if key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            real_world_example()
        else:
            print("⚠️  Skipping real API call (no ANTHROPIC_API_KEY)")

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Tokens are the basic unit of LLM processing")
        print("- ~4 characters = 1 token (rough estimate)")
        print("- Model choice significantly impacts cost")
        print("- Context windows limit how much text you can send")
        print("- Always estimate costs before batch processing\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Install required packages: pip install tiktoken anthropic")
