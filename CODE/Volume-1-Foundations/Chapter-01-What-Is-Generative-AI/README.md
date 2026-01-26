# Chapter 1: What Is Generative AI?

> Your first "aha moment" — watch AI diagnose network security issues in seconds.

## What You'll Build

A **production-ready config analyzer** that:
- Identifies security vulnerabilities
- Finds best practice violations  
- Suggests optimizations
- Outputs structured JSON results

This is the project from the book chapter. The code here matches the chapter content exactly.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Check with `python --version` |
| Anthropic API key | Get one at [console.anthropic.com](https://console.anthropic.com/) |
| Basic networking | CCNA-level (understand configs) |

## Time & Cost

- ⏱️ **Time:** 5-10 minutes to run
- 💰 **API Cost:** ~$0.03-0.08 per analysis

## Quick Start

```bash
# 1. Navigate to Volume 1
cd CODE/Volume-1-Foundations

# 2. Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Set your API key
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 4. Run the analyzer (creates sample_config.txt automatically)
python Chapter-01-What-Is-Generative-AI/ai_config_analysis.py
```

## Usage Options

```bash
# Analyze the sample config (default)
python ai_config_analysis.py

# Analyze your own config
python ai_config_analysis.py --file /path/to/router.cfg

# Only show critical and high severity issues
python ai_config_analysis.py --severity high

# Save results to custom file
python ai_config_analysis.py --output my_results.json

# Combine options
python ai_config_analysis.py -f myconfig.cfg -s high -o audit.json
```

## Sample Output

```
🔍 AI-Powered Config Analyzer
   Chapter 1: What is Generative AI?
================================================================================
Analyzing: sample_config.txt
This may take 10-20 seconds...

================================================================================
CONFIG ANALYSIS SUMMARY
================================================================================
Total Issues: 8
  🔴 Critical: 3
  🟠 High: 2
  🟡 Medium: 2
  🟢 Low: 1

🔴 CRITICAL ISSUES
--------------------------------------------------------------------------------

1. Weak SNMP Community Strings
   Category: security

   Explanation:
   The configuration uses default 'public' and 'private' SNMP community strings 
   which are widely known. The 'private' string with RW access allows config changes.

   Recommendation:
   no snmp-server community public
   no snmp-server community private
   snmp-server community X92kP!m3Z RO
   snmp-server community Y83nQ!r7W RW

...
```

## The Sample Config

The analyzer creates `sample_config.txt` automatically with intentional issues:

| Issue | Severity | Type |
|-------|----------|------|
| SNMP 'public'/'private' | Critical | Security |
| VTY lines without auth | Critical | Security |
| Cleartext password | Critical | Security |
| Telnet enabled | High | Security |
| No NTP configured | High | Best Practice |
| OSPF area design | Medium | Best Practice |
| Missing descriptions | Medium | Optimization |
| Password encryption off | Low | Best Practice |

## Files

| File | Purpose |
|------|---------|
| `ai_config_analysis.py` | Main analyzer script |
| `sample_config.txt` | Sample Cisco IOS config (auto-created) |
| `analysis_results.json` | Output file with full findings |

## How It Works

1. **Read config** — Load the text file
2. **Build prompt** — Tell Claude what to look for
3. **Call API** — Send to Claude Sonnet 4
4. **Parse JSON** — Extract structured findings
5. **Display results** — Pretty-print by severity

No regex. No rule lists. The AI reasons about the config like an experienced engineer would.

## Lab Exercises

From the book chapter:

1. **Modify the prompt** — Add check for deprecated IOS commands
2. **Analyze your own config** — Use `--file` with a real config (sanitize first!)
3. **Add severity filtering** — ✅ Already implemented with `--severity`
4. **Batch processing** — Modify to process a directory of configs

## Troubleshooting

### "API Key Not Found"
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
# Or add to .env file in this directory
```

### "Rate Limit Exceeded"
Free tier: ~5 requests/minute. Wait and retry.

### "Context Length Exceeded"  
Config too large. Split into sections or use Claude Opus (200K context).

## Next Steps

→ **Chapter 2:** Introduction to LLMs — understand tokens, context, and why this works

---

**Full chapter:** See `Volume-1-Foundations/Chapters/Chapter-01-What-Is-Generative-AI.md`
