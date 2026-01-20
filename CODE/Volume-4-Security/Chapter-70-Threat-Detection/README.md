# Chapter 70: AI-Powered Threat Detection

Production-ready Python code demonstrating dual-AI security analysis with Claude and OpenAI.

## What's Inside

**threat_detection.py** - Dual-AI threat detection with LangChain orchestration

### Features

- **Example 1**: Lateral Movement Detection with Dual-AI Consensus
  - Claude performs deep contextual analysis
  - OpenAI validates with pattern matching
  - Both must agree for high-confidence alerts

- **Example 2**: C2 Beacon Detection using Both Models
  - Statistical analysis identifies periodic traffic
  - Claude analyzes behavioral context
  - OpenAI matches against known C2 frameworks

- **Example 3**: Credential Compromise with Multi-Agent Validation
  - LangChain Sequential Chain orchestration
  - Multi-step reasoning (behavioral → threat intel → verdict)
  - Production pattern for complex decisions

## Installation

### Required Packages

```bash
pip install anthropic openai langchain langchain-anthropic langchain-openai
```

### API Keys

Set environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

Or use Google Colab secrets (automatically detected).

## Usage

### Run All Examples

```bash
python threat_detection.py
```

### Run Individual Examples

```python
from threat_detection import example_1, example_2, example_3

# Run just lateral movement detection
example_1()

# Run just C2 beacon detection
example_2()

# Run just credential compromise detection
example_3()
```

## Google Colab

This code is Colab-compatible. Upload to Colab and set API keys:

```python
# In Colab: Secrets → Add ANTHROPIC_API_KEY and OPENAI_API_KEY
from google.colab import userdata

# Then run normally
!python threat_detection.py
```

## Architecture

### Why Dual-AI?

1. **Reduces False Positives**: Both AIs must agree for alerts
2. **Complementary Strengths**:
   - Claude: Deep reasoning, contextual analysis
   - OpenAI: Pattern matching, threat intelligence
3. **Higher Confidence**: Consensus-based decisions

### When to Use Each AI

- **Claude**: Use for complex reasoning about intent, context, "why is this suspicious?"
- **OpenAI**: Use for classification, pattern matching, "what attack is this?"
- **Both**: Use for critical decisions requiring high confidence

## Output Examples

```
🚨 ALERT: Lateral Movement Detected
Consensus: Both AIs agree - Lateral Movement Detected
Combined Confidence: 92%

📋 Recommended Actions:
  1. Immediately disable john.admin account
  2. Isolate dc01.corp.local from network
  3. Review recent activity from this account
```

## Production Notes

- **Error Handling**: All API calls wrapped with try/except
- **Fail-Secure**: On error, assume threat (alert for review)
- **API Key Management**: Supports environment variables and Colab secrets
- **Cost Management**: Pre-filter with statistical analysis before AI calls

## Related Chapters

- Chapter 72: SIEM Integration (see ../Chapter-72-SIEM/)
- Volume 3: Production AI systems, scaling, monitoring

## Support

For issues or questions about this code:
- Book: "AI for Networking Engineers" by Ed (vExpertAI)
- GitHub: github.com/vexpertai/ai-networking-book
- Twitter/X: @vExpertAI
