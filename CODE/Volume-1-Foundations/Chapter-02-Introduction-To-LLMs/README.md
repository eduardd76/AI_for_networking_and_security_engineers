# Chapter 2: Introduction to Large Language Models

> Understand tokens, context windows, and costs — the economics of LLMs.

## What You'll Build

A **token calculator** that:
- Shows exactly how network configs tokenize
- Calculates costs across different models
- Checks context window limits
- Projects monthly costs for batch processing

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Check with `python --version` |
| tiktoken | `pip install tiktoken` (offline tokenization) |
| Anthropic API key | Optional (for accurate Claude counts) |

## Quick Start

```bash
# Navigate to Volume 1
cd CODE/Volume-1-Foundations

# Install tiktoken
pip install tiktoken

# Run the demo
python Chapter-02-Introduction-To-LLMs/token_calculator.py demo

# Analyze a config file
python Chapter-02-Introduction-To-LLMs/token_calculator.py ../Chapter-01-What-Is-Generative-AI/sample_config.txt
```

## Usage

```bash
# Analyze any file
python token_calculator.py router_config.txt

# Interactive mode
python token_calculator.py interactive

# Visualize tokens in text
python token_calculator.py visualize "interface GigabitEthernet0/0"

# Show pricing table
python token_calculator.py pricing

# Run full demo
python token_calculator.py demo
```

## Sample Output

```
================================================================================
FILE ANALYSIS: router_config.txt
================================================================================
File size: 847 characters
File lines: 32

TOKEN COUNTS:
--------------------------------------------------------------------------------
Tiktoken (GPT-4):   245 tokens
Claude (API):       231 tokens

CONTEXT WINDOW FIT:
--------------------------------------------------------------------------------
claude-haiku    (200,000 max): ✅  +197,755 tokens
claude-sonnet   (200,000 max): ✅  +197,755 tokens
gpt-4o-mini     (128,000 max): ✅  +125,755 tokens
gpt-4o          (128,000 max): ✅  +125,755 tokens

COST ESTIMATES:
--------------------------------------------------------------------------------
(Assuming 2,000 output tokens)

gpt-4o-mini     → $0.001237
gpt-4o          → $0.020613
claude-haiku    → $0.002561
claude-sonnet   → $0.030735

BATCH PROJECTION (1,000 files):
--------------------------------------------------------------------------------
gpt-4o-mini     → $1.24/month
claude-haiku    → $2.56/month
claude-sonnet   → $30.74/month
================================================================================
```

## Key Concepts

### Tokens
- **1 token ≈ 4 characters** (English text)
- Technical terms often split into multiple tokens
- `GigabitEthernet` = 3 tokens, `Gi0/0` = 4 tokens

### Context Windows
Think of it like **MTU for LLMs**:
- GPT-4o: 128,000 tokens (standard MTU)
- Claude Sonnet: 200,000 tokens (jumbo frames)
- Gemini 1.5 Pro: 2,000,000 tokens (superjumbo)

Exceed the limit → request fails (like packet fragmentation).

### Pricing (January 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| gpt-4o-mini | $0.15/M | $0.60/M | 128K |
| claude-haiku | $0.25/M | $1.25/M | 200K |
| claude-sonnet | $3.00/M | $15.00/M | 200K |
| claude-opus | $15.00/M | $75.00/M | 200K |

**Rule of thumb:** Output costs 3-5x more than input!

### The 80/20 Rule

- **80% of tasks** → Use cheap models (gpt-4o-mini, claude-haiku)
- **20% of tasks** → Use expensive models (claude-sonnet, opus)

## Files

| File | Purpose |
|------|---------|
| `token_calculator.py` | Main calculator script |
| `README.md` | This file |

## Lab Exercises

From the book chapter:

1. **Tokenization comparison** — Compare `interface GigabitEthernet0/0` vs `int Gi0/0`
2. **Cost calculator** — Calculate monthly cost for your network's configs
3. **Context stress test** — Find your largest config, check which models fit
4. **Model comparison** — Same task with Haiku vs Sonnet, compare quality vs cost

## Troubleshooting

### "tiktoken not installed"
```bash
pip install tiktoken
```

### "ANTHROPIC_API_KEY not set"
Claude token counts will fallback to tiktoken estimate. For exact counts:
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### Tokens seem wrong
Tiktoken uses GPT-4 encoding. Claude may tokenize differently (usually 5-10% fewer tokens).

## Next Steps

→ **Chapter 3:** Choosing the Right Model — benchmarking for network tasks

---

**Full chapter:** See `Volume-1-Foundations/Chapters/Chapter-02-Introduction-To-LLMs.md`
