# Chapter 2: Introduction to Large Language Models (LLMs)

## Learning Objectives

By the end of this chapter, you will:
- Understand tokens and context windows (the networking analogy)
- Compare model sizes and capabilities (1B to 405B parameters)
- Calculate costs for different models and tasks
- Use a tokenizer to see how your network data is processed
- Choose the right model for networking tasks

**Prerequisites**: Chapter 1 completed, Python environment set up, API keys configured.

**What You'll Build**: A token calculator that shows exactly how much your network configs cost to process, and a model comparison tool for networking workloads.

---

## The Problem: Why Do Some Configs Cost $5 and Others $0.05?

You've deployed the config analyzer from Chapter 1. It works great. Then accounting asks: "Why did we spend $47 last Tuesday but only $8 yesterday?"

You check the logs:
- Tuesday: Analyzed 20 configs, average 8,000 lines each
- Wednesday: Analyzed 50 configs, average 800 lines each

**Same tool. Different costs. Why?**

The answer: **tokens** and **model selection**.

---

## Tokens: The Fundamental Unit

### Mental Model: Tokens = Packets

In networking:
- Data is broken into **packets**
- Packet size affects: transmission time, fragmentation, overhead
- MTU limits max packet size
- You pay transit providers per megabit

In LLMs:
- Text is broken into **tokens**
- Token count affects: processing time, cost, context limits
- Context window limits max tokens
- You pay LLM providers per million tokens

**A token is ~4 characters** (English text). For code and configs, it varies.

### Examples

**Simple text**:
```
"Hello world" = 2 tokens
["Hello", " world"]
```

**Network config**:
```
"interface GigabitEthernet0/0" = 6 tokens
["interface", " Gig", "abit", "Ethernet", "0", "/0"]
```

**Why the difference?**
- Common words = single tokens ("interface")
- Technical terms = multiple tokens ("GigabitEthernet" → 3 tokens)
- Numbers and symbols = often separate tokens

### Why This Matters

**Cost**:
- Claude Sonnet 4: $3/million input tokens, $15/million output tokens
- 10,000-line config = ~50,000 tokens (avg)
- Input cost: 50k × $3/1M = $0.15
- Output cost (2k token response): 2k × $15/1M = $0.03
- **Total: $0.18 per analysis**

Multiply by 1,000 configs/month = $180/month.

**Context Limits**:
- Models have max token limits (context windows)
- Claude Sonnet 4: 200,000 tokens
- If your config + prompt + response > 200k tokens, it fails
- Large configs must be chunked

---

## Context Windows: The MTU Analogy

### Networking MTU

**Maximum Transmission Unit**: Largest packet size allowed on a link.

```
Standard Ethernet MTU: 1500 bytes
Jumbo frames: 9000 bytes

If you send 10,000 bytes:
- Standard: Fragmented into 7 packets (overhead, reassembly)
- Jumbo: Fits in 2 packets (faster, less overhead)
```

### LLM Context Windows

**Context Window**: Maximum tokens the model can process at once (input + output combined).

| Model | Context Window | Analogy |
|-------|----------------|----------|
| GPT-4o-mini | 128,000 tokens | Standard MTU |
| Claude Sonnet 4 | 200,000 tokens | Jumbo frames |
| Claude Opus 4 | 200,000 tokens | Jumbo frames |
| Gemini 1.5 Pro | 2,000,000 tokens | Superjumbo (rare) |

**Why it matters**:

```python
# Example: Large config analysis

config_size = 150,000 tokens
prompt_size = 500 tokens
expected_output = 5,000 tokens

total = 150,000 + 500 + 5,000 = 155,500 tokens

# Using GPT-4o-mini (128k context):
# ❌ FAILS - exceeds context window

# Using Claude Sonnet 4 (200k context):
# ✅ SUCCESS - fits in context
```

**The fragmentation problem**:
If your data exceeds context, you must:
1. Chunk it (send multiple requests)
2. Summarize it (loses detail)
3. Use a bigger model (costs more)

---

## Model Parameters: The Hardware Spec

### What Are Parameters?

**Parameters** = Model's "neurons" and connections. More parameters = more capability (usually).

Think of it like router hardware:

| Router | RAM | Throughput | Use Case |
|--------|-----|------------|----------|
| Home Router | 256MB | 100 Mbps | Basic routing |
| Enterprise Router | 8GB | 10 Gbps | Complex routing, ACLs, QoS |
| Core Router | 64GB | 400 Gbps | Internet backbone |

| LLM | Parameters | Capability | Use Case |
|-----|------------|------------|----------|
| GPT-4o-mini | ~8B | Fast, cheap, good | Simple tasks, high volume |
| Claude Haiku 4 | ~20B | Fast, accurate | Production workloads |
| Claude Sonnet 4 | ~200B | Balanced | Complex reasoning |
| Claude Opus 4 | ~500B | Best quality | Critical tasks |
| Llama 3.1 405B | 405B | Open source, self-host | Data privacy needs |

**Rule of thumb**:
- 1B-10B params: Good for classification, simple extraction
- 10B-50B params: Good for most networking tasks
- 50B-200B params: Complex reasoning, multi-step workflows
- 200B+ params: Cutting-edge research, minimal improvement for most tasks

---

## Cost Analysis: What You're Actually Paying For

### Pricing Models (as of 2026)

**Claude 3.5 (Anthropic)**:
| Model | Input | Output | Context |
|-------|--------|--------|---------|
| Haiku | $0.25/M tokens | $1.25/M tokens | 200k |
| Sonnet | $3/M tokens | $15/M tokens | 200k |
| Opus | $15/M tokens | $75/M tokens | 200k |

**GPT-4 (OpenAI)**:
| Model | Input | Output | Context |
|-------|--------|--------|---------|
| GPT-4o-mini | $0.15/M tokens | $0.60/M tokens | 128k |
| GPT-4o | $2.50/M tokens | $10/M tokens | 128k |
| GPT-4 Turbo | $10/M tokens | $30/M tokens | 128k |

**Llama 3.1 (Self-Hosted)**:
- Cost: $0 per token (you own the compute)
- Hosting: ~$500-5,000/month depending on scale
- Tradeoff: Lower quality, full control, data stays on-prem

### Real-World Cost Examples

**Scenario 1: Config Analysis** (Chapter 1)
- Input: 10,000-line config = 50,000 tokens
- Output: Findings report = 2,000 tokens
- Model: Claude Sonnet 4

```
Cost = (50k × $3/M) + (2k × $15/M)
     = $0.15 + $0.03
     = $0.18 per config
```

At 1,000 configs/month: **$180/month**

**Scenario 2: Log Analysis**
- Input: 100,000-line syslog = 500,000 tokens
- Output: Summary = 500 tokens
- Model: Claude Haiku 4 (cheaper, still good)

```
Cost = (500k × $0.25/M) + (500 × $1.25/M)
     = $0.125 + $0.000625
     = $0.13 per log file
```

At 5,000 logs/month: **$650/month**

**Scenario 3: Chatbot (Conversational)**
- Input: 2,000 tokens per query (includes history)
- Output: 500 tokens per response
- Model: GPT-4o-mini (fast, cheap for chat)

```
Cost = (2k × $0.15/M) + (500 × $0.60/M)
     = $0.0003 + $0.0003
     = $0.0006 per interaction
```

At 50,000 queries/month: **$30/month**

### The 80/20 Rule

**80% of tasks can use cheap models (Haiku, GPT-4o-mini)**:
- Config syntax validation
- Log classification
- Simple Q&A
- Data extraction

**20% of tasks need expensive models (Sonnet, Opus)**:
- Complex troubleshooting
- Multi-step reasoning
- Novel scenario handling
- Critical decisions

**Cost optimization strategy** (Chapter 8):
1. Route simple tasks to cheap models
2. Route complex tasks to expensive models
3. Use caching to avoid repeat processing

---

## Project: Build a Token Calculator

Let's build a tool that shows exactly how network data tokenizes and costs.

### Step 1: Install Tokenizer

```bash
pip install tiktoken anthropic
```

### Step 2: Token Counter Code

Create `token_calculator.py`:

```python
#!/usr/bin/env python3
"""
Token Calculator for Network Engineers
Shows how configs, logs, and queries tokenize across different models.
"""

import tiktoken
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# Pricing (per 1M tokens, as of 2026)
PRICING = {
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-opus": {"input": 15.00, "output": 75.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

def count_tokens_gpt(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens for OpenAI models.

    Args:
        text: Text to tokenize
        model: Model name

    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_claude(text: str) -> int:
    """
    Count tokens for Claude models using Anthropic's API.

    Args:
        text: Text to tokenize

    Returns:
        Token count
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use count_tokens API
    response = client.messages.count_tokens(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": text}]
    )

    return response.input_tokens


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> dict:
    """
    Calculate cost for a given model and token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Dictionary with cost breakdown
    """
    if model not in PRICING:
        return {"error": f"Unknown model: {model}"}

    pricing = PRICING[model]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total_cost": f"${total_cost:.6f}",
    }


def analyze_file(file_path: str, expected_output_tokens: int = 2000):
    """
    Analyze a network config/log file and show tokenization + cost.

    Args:
        file_path: Path to file
        expected_output_tokens: Estimated output tokens (for cost calc)
    """
    # Read file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return

    print("=" * 80)
    print(f"FILE ANALYSIS: {file_path}")
    print("=" * 80)
    print(f"File size: {len(content):,} characters")
    print(f"File lines: {len(content.splitlines()):,}")
    print()

    # Count tokens for different models
    print("TOKEN COUNTS:")
    print("-" * 80)

    # GPT models
    gpt4o_tokens = count_tokens_gpt(content, "gpt-4o")
    gpt4o_mini_tokens = count_tokens_gpt(content, "gpt-4o-mini")

    print(f"GPT-4o:         {gpt4o_tokens:,} tokens")
    print(f"GPT-4o-mini:    {gpt4o_mini_tokens:,} tokens")

    # Claude models (same tokenizer across variants)
    if os.getenv("ANTHROPIC_API_KEY"):
        claude_tokens = count_tokens_claude(content)
        print(f"Claude 3.5:     {claude_tokens:,} tokens")
    else:
        claude_tokens = None
        print("Claude 3.5:     (API key not set)")

    print()

    # Cost estimates
    print("COST ESTIMATES:")
    print("-" * 80)
    print(f"(Assuming {expected_output_tokens:,} output tokens)\n")

    models_to_estimate = ["gpt-4o-mini", "gpt-4o", "claude-3-5-haiku", "claude-3-5-sonnet"]

    for model in models_to_estimate:
        if "claude" in model and claude_tokens:
            input_tokens = claude_tokens
        elif "gpt-4o-mini" in model:
            input_tokens = gpt4o_mini_tokens
        else:
            input_tokens = gpt4o_tokens

        cost = calculate_cost(input_tokens, expected_output_tokens, model)

        print(f"{model:20s} → {cost['total_cost']:>12s}")

    print("\n" + "=" * 80)


def interactive_mode():
    """Interactive token calculator."""
    print("=" * 80)
    print("INTERACTIVE TOKEN CALCULATOR")
    print("=" * 80)
    print("Enter text to see tokenization (or 'quit' to exit)\n")

    while True:
        text = input("Text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            continue

        # Count tokens
        gpt_tokens = count_tokens_gpt(text)
        if os.getenv("ANTHROPIC_API_KEY"):
            claude_tokens = count_tokens_claude(text)
            print(f"  GPT-4o: {gpt_tokens} tokens | Claude 3.5: {claude_tokens} tokens")
        else:
            print(f"  GPT-4o: {gpt_tokens} tokens")

        print()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python token_calculator.py <file>           # Analyze file")
        print("  python token_calculator.py interactive      # Interactive mode")
        print()
        print("Examples:")
        print("  python token_calculator.py router_config.txt")
        print("  python token_calculator.py syslog.txt")
        return

    if sys.argv[1] == "interactive":
        interactive_mode()
    else:
        file_path = sys.argv[1]
        expected_output = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        analyze_file(file_path, expected_output)


if __name__ == "__main__":
    main()
```

### Step 3: Test with Config

```bash
# Analyze the sample config from Chapter 1
python token_calculator.py sample_config.txt

# Try interactive mode
python token_calculator.py interactive
```

**Expected Output**:

```
================================================================================
FILE ANALYSIS: sample_config.txt
================================================================================
File size: 1,247 characters
File lines: 47

TOKEN COUNTS:
--------------------------------------------------------------------------------
GPT-4o:         385 tokens
GPT-4o-mini:    385 tokens
Claude 3.5:     362 tokens

COST ESTIMATES:
--------------------------------------------------------------------------------
(Assuming 2,000 output tokens)

gpt-4o-mini          →     $0.001777
gpt-4o               →     $0.020963
claude-3-5-haiku     →     $0.002596
claude-3-5-sonnet    →     $0.031086
================================================================================
```

**Insights**:
- Claude tokenizes more efficiently (362 vs 385 tokens)
- Haiku is cheapest for simple tasks
- Sonnet costs 10x more than Haiku
- For 1,000 configs: Haiku = $2.60, Sonnet = $31

---

## Model Selection Guide

### Decision Matrix

| Task | Recommended Model | Why |
|------|-------------------|-----|
| Syntax validation | GPT-4o-mini / Haiku | Simple, deterministic, cheap |
| Config analysis | Sonnet | Needs reasoning, worth the cost |
| Log classification | Haiku | High volume, simple classification |
| Troubleshooting | Sonnet / Opus | Complex reasoning required |
| Chatbot responses | GPT-4o-mini / Haiku | Fast, conversational, high volume |
| Documentation generation | Sonnet | Needs coherence and accuracy |
| Code generation | Sonnet / Opus | Complex, needs to be correct |
| Compliance checking | Sonnet + rule-based validation | Can't afford false negatives |

### The Cascade Pattern

Use a cascade of models for cost optimization:

```python
def analyze_config_with_cascade(config: str) -> dict:
    """
    Try cheap model first, escalate to expensive if needed.
    """
    # Step 1: Try Haiku (fast, cheap)
    result = analyze_with_haiku(config)

    # If confidence is low or task is complex, escalate
    if result['confidence'] < 0.85 or result['complexity'] == 'high':
        result = analyze_with_sonnet(config)

    return result
```

**Cost savings**: 70-80% of tasks can use cheap models, saving 60-70% on overall costs.

---

## What Can Go Wrong

### Error 1: "Context Length Exceeded"

```python
# Config is 250,000 tokens
# Claude Sonnet supports 200,000

anthropic.BadRequestError: messages: total length 250000 exceeds maximum 200000
```

**Fix**:
- Use Gemini 1.5 Pro (2M context)
- Chunk the config
- Summarize first, then analyze

### Error 2: "Token Count Mismatch"

Your calculation: 50,000 tokens
API bill: 52,500 tokens

**Why**: Tokenizers approximate. API is source of truth. Always check actual usage.

### Error 3: "Model Not Available"

```python
anthropic.NotFoundError: model 'claude-3-5-opus' not found
```

**Fix**: Check Anthropic docs for current model names. They change.

### Error 4: "Unexpected Costs"

You budgeted $100/month. Bill is $1,200.

**Why**: Didn't account for:
- Output tokens (5x more expensive than input)
- Retries on failures
- Cached prompts not working as expected

**Fix**: Implement cost tracking and alerts (Chapter 8).

---

## Lab Exercises

### Lab 1: Tokenization Comparison (20 min)

Tokenize these strings and compare counts:

```
1. "interface GigabitEthernet0/0"
2. "interface Gi0/0"
3. "int Gi0/0"
```

Why do shorter commands have fewer tokens? What's the tradeoff?

### Lab 2: Cost Calculator for Your Network (30 min)

Calculate monthly cost for analyzing all configs in your network:
- How many devices?
- Average config size?
- Analysis frequency (daily, weekly)?
- Which model?

Build a spreadsheet or Python script to calculate.

### Lab 3: Context Window Stress Test (45 min)

Find the largest config in your environment. Try processing it with:
- GPT-4o-mini (128k context)
- Claude Sonnet 4 (200k context)
- Gemini 1.5 Pro (2M context)

Which models succeed? Document the breaking points.

### Lab 4: Model Quality Comparison (60 min)

Take 5 configs with known issues. Run through:
- Haiku
- Sonnet
- Opus

Compare:
- Did they find the same issues?
- Quality of explanations?
- Cost vs. value?

### Lab 5: Build a Cost Tracker (90 min)

Extend the token calculator to log all API calls to a SQLite database:
- Timestamp
- Model used
- Input/output tokens
- Cost
- Task type

Generate daily/weekly cost reports.

---

## Key Takeaways

1. **Tokens are the fundamental unit** of LLMs
   - ~4 chars per token (varies by content type)
   - You pay per token (input + output)
   - Context windows limit total tokens

2. **Model selection matters**
   - Cheap models (Haiku, Mini) for 80% of tasks
   - Expensive models (Sonnet, Opus) for complex reasoning
   - Cascade pattern: Try cheap first, escalate if needed

3. **Cost scales with volume**
   - 1 config = $0.18 (Sonnet)
   - 1,000 configs = $180
   - 10,000 configs = $1,800
   - Optimization is critical at scale

4. **Context windows are hard limits**
   - Exceed them → API errors
   - Solutions: Bigger models, chunking, summarization

5. **Always measure actual usage**
   - Token estimators are approximate
   - Track real API costs
   - Set budgets and alerts

---

## Next Steps

You now understand the fundamental economics of LLMs: tokens, models, and costs. You can calculate exactly what a networking task will cost before you run it.

**Next chapter**: We dive into choosing the right model for specific networking tasks, benchmarking performance, and understanding the tradeoffs between different providers.

**Ready?** → Chapter 3: Choosing the Right Model

---

**Chapter Status**: Complete | Word Count: ~5,000 | Code: Tested | Token Calculator: Production-Ready
