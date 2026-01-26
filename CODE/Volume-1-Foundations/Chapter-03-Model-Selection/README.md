# Chapter 3: Choosing the Right Model

> Benchmark LLMs on YOUR networking tasks — make decisions based on data, not marketing.

## What You'll Build

A **model benchmarking tool** that:
- Tests multiple models on real networking tasks
- Measures quality, latency, and cost
- Generates comparison reports
- Provides actionable recommendations

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Check with `python --version` |
| API keys | Anthropic and/or OpenAI |
| anthropic package | `pip install anthropic` |
| openai package | `pip install openai` |

## Quick Start

```bash
# Navigate to Chapter 3
cd CODE/Volume-1-Foundations/Chapter-03-Model-Selection

# Install dependencies
pip install anthropic openai python-dotenv

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-api03-...
export OPENAI_API_KEY=sk-proj-...

# List available tasks
python model_benchmark.py --list

# Run a single benchmark
python model_benchmark.py --task security_analysis

# Run all benchmarks (~$0.50 total cost)
python model_benchmark.py --all
```

## Available Benchmark Tasks

| Task | Description |
|------|-------------|
| `security_analysis` | Analyze router config for vulnerabilities |
| `bgp_troubleshooting` | Diagnose BGP session issues |
| `acl_generation` | Generate ACL from requirements |
| `log_classification` | Classify syslog by severity/type |
| `documentation` | Generate docs from config |

## Sample Output

```
🔬 Network AI Model Benchmark Tool
   Chapter 3: Choosing the Right Model
======================================================================

======================================================================
TASK: Config Security Analysis
======================================================================

Testing claude-sonnet-4-20250514... OK (7.2s, $0.0891, 94% quality)
Testing claude-haiku-4-20250514... OK (2.4s, $0.0124, 86% quality)
Testing gpt-4o... OK (4.1s, $0.0712, 91% quality)
Testing gpt-4o-mini... OK (2.1s, $0.0082, 82% quality)

======================================================================
BENCHMARK SUMMARY
======================================================================

Config Security Analysis:
----------------------------------------------------------------------
Model                          Latency         Cost    Quality
----------------------------------------------------------------------
claude-sonnet-4-20250514          7.2s     $0.0891      94.0%
gpt-4o                            4.1s     $0.0712      91.0%
claude-haiku-4-20250514           2.4s     $0.0124      86.0%
gpt-4o-mini                       2.1s     $0.0082      82.0%

======================================================================
RECOMMENDATIONS
======================================================================

💡 Suggestions:
   • Use Claude Sonnet for complex analysis (security, troubleshooting)
   • Use Haiku or GPT-4o-mini for high-volume simple tasks
   • Consider the 80/20 split: 80% cheap models, 20% quality models

💰 Total benchmark cost: $0.1809
📁 Detailed results saved to: benchmark_results.json
```

## The 80/20 Model Strategy

Based on benchmark results, the optimal approach for most teams:

| Traffic | Model | Use For |
|---------|-------|---------|
| 80% | Haiku / GPT-4o-mini | Log parsing, simple Q&A, classification |
| 15% | Sonnet / GPT-4o | Security analysis, troubleshooting, code gen |
| 5% | Opus (when needed) | Complex multi-system issues |

**Cost example** (10,000 requests/month):
- All Sonnet: $930/month
- 80/20 split: $343/month
- **Savings: 63%**

## Adding Custom Tasks

Edit the `TASKS` dictionary in `model_benchmark.py`:

```python
TASKS["my_task"] = {
    "name": "My Custom Task",
    "description": "What this task tests",
    "prompt": "Your actual prompt here...",
    "expected_terms": ["words", "that", "indicate", "good", "response"],
}
```

Then run:
```bash
python model_benchmark.py --task my_task
```

## Files

| File | Purpose |
|------|---------|
| `model_benchmark.py` | Main benchmark tool |
| `benchmark_results.json` | Output with detailed results |
| `README.md` | This file |

## Model Quick Reference

| Model | Best For | Avg Cost/Request |
|-------|----------|------------------|
| Claude Sonnet | Complex reasoning | $0.09 |
| GPT-4o | Fast + quality | $0.07 |
| Claude Haiku | High volume | $0.01 |
| GPT-4o-mini | Budget | $0.008 |

## Troubleshooting

### "API key not set"
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key
export OPENAI_API_KEY=sk-proj-your-key
```

### "Package not installed"
```bash
pip install anthropic openai python-dotenv
```

### Only some models work
That's fine! The tool skips models without valid API keys.

## Next Steps

→ **Chapter 4:** API Basics and Authentication

---

**Full chapter:** See `Volume-1-Foundations/Chapters/Chapter-03-Choosing-The-Right-Model.md`
