# Chapter 1: What Is Generative AI?

## Table of Contents
1. [The 3 AM Wake-Up Call](#the-3-am-wake-up-call)
2. [What Is Generative AI, Really?](#what-is-generative-ai-really)
3. [From Rules to Reasoning: A Networking Perspective](#from-rules-to-reasoning-a-networking-perspective)
4. [How Large Language Models Work](#how-large-language-models-work)
5. [Tokens: The Packets of AI](#tokens-the-packets-of-ai)
6. [The API: Your Interface to Intelligence](#the-api-your-interface-to-intelligence)
7. [Your First API Call](#your-first-api-call)
8. [What Generative AI Can (and Cannot) Do for Network Engineers](#what-generative-ai-can-and-cannot-do-for-network-engineers)
9. [The Landscape: Models, Providers, and Trade-offs](#the-landscape-models-providers-and-trade-offs)
10. [Security and Privacy Considerations](#security-and-privacy-considerations)
11. [Key Takeaways](#key-takeaways)

---

## The 3 AM Wake-Up Call

It's 3:17 AM. Your phone buzzes. PagerDuty. The monitoring system shows BGP session drops across three core routers in your east coast data center. You SSH into the first router and start running `show` commands, scanning through hundreds of lines of output, trying to piece together what changed. Was it a route-map update from last night's maintenance window? A prefix-list that's filtering something it shouldn't? Or something else entirely?

You pull up the config diff from your RANCID backup. There it is -- a junior engineer pushed a route-map change at 11 PM that inadvertently matched all prefixes instead of just the customer's /24. You fix it, verify BGP sessions are re-establishing, and go back to bed. Total time: 47 minutes. Most of that was reading output and tracing the logic.

Now imagine a different scenario. You get the same alert, but this time you paste the syslog messages, the config diff, and the BGP neighbor output into an AI assistant. Within seconds, it tells you: "The route-map ISP-PEER-OUT was modified at 23:04 UTC. The new permit clause at sequence 10 matches all prefixes (no match statement), causing full table advertisement to AS 65002. The previous version matched only prefix-list CUSTOMER-ROUTES. Suggested fix: restore the match clause or revert to the previous route-map version." You verify, apply the fix, and you're back in bed in 12 minutes.

That's what this book is about. Not replacing you. Not automating you out of a job. But giving you a tool that can read configurations, parse logs, and reason about network behavior the way a senior engineer would -- except it does it in seconds, at 3 AM, without needing coffee.

This chapter introduces generative AI from the ground up. No machine learning background required. If you understand how a packet traverses a network, you have all the mental models you need.

---

## What Is Generative AI, Really?

Let's start with a definition that actually makes sense to a network engineer.

**Generative AI** is a category of artificial intelligence that creates new content -- text, code, images, configurations -- based on patterns it learned from enormous amounts of training data. The most relevant form for us is the **Large Language Model (LLM)**, which generates text.

Think of it this way:

- **Traditional automation** (Ansible, Python scripts, Terraform): You write explicit rules. "If interface is down, send alert." The tool does exactly what you tell it, nothing more, nothing less. It's a static route.
- **Generative AI** (Claude, GPT-4, Gemini): You describe what you want in natural language, and the model generates a response. It can handle situations you didn't explicitly program for. It's a routing protocol.

A static route gets you to one destination via one path. A routing protocol understands the topology and adapts. Both are useful. Both have their place. But when you're troubleshooting a complex, multi-factor network issue at 3 AM, you want something that can reason, not just follow a predetermined script.

### What "Generative" Means

The word "generative" is key. Unlike traditional software that retrieves, transforms, or filters existing data, generative AI creates **new** content that didn't exist before:

- You give it a router config and ask for documentation -- it writes a human-readable summary.
- You give it an error log and ask for analysis -- it identifies the root cause and suggests remediation.
- You describe a network requirement in English -- it generates the IOS/NX-OS/JunOS configuration.
- You give it a complex ACL -- it explains what traffic it permits and denies, in plain language.

The model isn't looking up the answer in a database. It's generating it from its understanding of language, networking concepts, and the specific context you provided.

### Why Now?

Network engineers have been scripting for decades. What changed?

Three things converged in the early 2020s:

1. **Scale of training data**: Models like Claude were trained on vast corpora that include networking documentation, RFCs, vendor guides, Stack Overflow answers, configuration examples, and technical books. They have a broad (though imperfect) understanding of networking concepts.

2. **Natural language interface**: Instead of learning yet another domain-specific language or API, you describe what you want in English. The barrier to entry dropped from "knows Python and Jinja2" to "can describe the problem clearly."

3. **API accessibility**: You can call these models from any programming language with a simple HTTP request. No GPU clusters. No ML expertise. Just an API key and a few lines of Python.

For network engineers specifically, the timing is right because:
- Network complexity has outpaced our ability to document and troubleshoot manually
- Multi-vendor environments make it impossible to memorize every CLI syntax
- The industry is already moving toward programmability (YANG, gNMI, APIs) -- LLMs are the next step
- Most networking knowledge is text-based (configs, logs, RFCs) -- exactly what LLMs are best at

---

## From Rules to Reasoning: A Networking Perspective

To understand why generative AI is fundamentally different from traditional automation, consider how you approach a troubleshooting scenario.

### The Traditional Automation Approach

```python
# Traditional: Explicit rules for known problems
def check_interface(device, interface):
    status = get_interface_status(device, interface)
    if status['errors'] > 100:
        alert("High error count on {}/{}".format(device, interface))
    if status['state'] == 'down':
        alert("Interface {}/{} is down".format(device, interface))
    if status['utilization'] > 90:
        alert("High utilization on {}/{}".format(device, interface))
```

This works for known problems. But what about:
- A new type of error you haven't written a rule for?
- A combination of symptoms that together indicate a problem, but individually look fine?
- An issue described in a vendor advisory you haven't read yet?
- A config pattern that's technically valid but operationally dangerous?

### The Generative AI Approach

```python
# Generative AI: Describe the problem, get reasoning
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""Analyze this interface output and identify any issues,
        including subtle problems that might not trigger standard monitoring:

        {interface_output}

        Consider: error rates relative to traffic volume, counter patterns
        that suggest intermittent issues, and configuration mismatches
        between this interface and its CDP neighbor."""
    }]
)
```

The LLM can reason about the output the way an experienced engineer would. It doesn't just check thresholds -- it considers context, patterns, and relationships.

**Networking analogy**: Traditional automation is like a static ACL -- it matches specific, predefined conditions. Generative AI is like a next-generation firewall with deep packet inspection -- it understands the context of what it's looking at.

### Where Each Approach Wins

| Scenario | Traditional Automation | Generative AI |
|----------|----------------------|---------------|
| "Alert if BGP peer is down" | Better (deterministic, fast) | Overkill |
| "Explain why BGP peer is flapping" | Can't do this | Built for this |
| "Deploy standard config to 100 switches" | Better (reliable, repeatable) | Overkill |
| "Review config for security issues" | Limited to known patterns | Can find novel issues |
| "Parse show command into JSON" | Good with TextFSM templates | Good without templates |
| "Diagnose intermittent connectivity issue" | Can't reason about ambiguity | Can correlate symptoms |
| "Generate documentation from configs" | Template-based, rigid | Flexible, readable output |

The takeaway: generative AI is not a replacement for your existing automation. It's a new tool in the toolkit -- one that handles the reasoning and language tasks that traditional automation can't.

---

## How Large Language Models Work

You don't need a PhD in machine learning to use LLMs effectively. But understanding the basics helps you predict when they'll work well and when they won't.

### The Training Phase

An LLM is trained in two stages:

**Stage 1: Pre-training** -- The model reads an enormous corpus of text (books, websites, code, documentation) and learns statistical patterns. "After the sequence 'router bgp', the next token is likely a number (the AS number)." It does this billions of times, adjusting internal parameters to get better at predicting what comes next.

**Networking analogy**: Pre-training is like a junior engineer spending years reading every Cisco, Juniper, and Arista configuration guide, every RFC, every troubleshooting forum, and every network design document ever written. They don't memorize it all perfectly, but they develop an intuition for how things work.

**Stage 2: Fine-tuning and alignment** -- The pre-trained model is then refined using human feedback. Humans rate responses as helpful or unhelpful, safe or unsafe. This teaches the model to give useful, honest answers rather than just predicting the statistically most likely next word.

**Networking analogy**: Fine-tuning is like that junior engineer spending another year working under a senior engineer who says "that answer was good" or "no, that's wrong, here's why." They learn not just what's technically possible, but what's actually useful.

### The Inference Phase

When you send a prompt to Claude, here's what happens:

1. Your text is broken into **tokens** (more on this in the next section)
2. The model processes your tokens through billions of parameters
3. It predicts the most likely next token, generates it, then repeats
4. This continues until it reaches a stopping point or the maximum length

The model generates text one token at a time, but it considers **all** the context you provided (your prompt, system instructions, conversation history) for each token it generates.

**Key insight**: The model doesn't "know" things the way you do. It has learned statistical patterns that are remarkably good at producing useful text. It can generate a valid BGP configuration not because it understands BGP the way you do, but because it has seen thousands of BGP configurations and understands the patterns and relationships between the components.

This distinction matters because it explains both the strengths and weaknesses:
- **Strength**: It can handle any topic it was trained on, in any combination
- **Weakness**: It can generate plausible-sounding but incorrect output (hallucination)
- **Strength**: It understands context and nuance in natural language
- **Weakness**: It doesn't have real-time access to your network (unless you give it data)

---

## Tokens: The Packets of AI

If you're going to use LLM APIs, you need to understand tokens. They're how you get charged, how context limits work, and why your prompt design matters.

### What Is a Token?

A token is a chunk of text -- roughly 3-4 characters or about 0.75 words in English. The model doesn't see characters or words. It sees tokens.

```
"show ip interface brief" → ["show", " ip", " interface", " brief"]  (4 tokens)
"192.168.1.1"             → ["192", ".", "168", ".", "1", ".", "1"]   (7 tokens)
"GigabitEthernet0/0/0"    → ["Gig", "abit", "Ethernet", "0", "/0", "/0"]  (6 tokens)
```

**Networking analogy**: Tokens are to LLMs what packets are to networks.

| Concept | Networking | LLMs |
|---------|-----------|------|
| Data unit | Packet | Token |
| Size limit | MTU (e.g., 1500 bytes) | Context window (e.g., 200K tokens) |
| Overhead | Headers (Ethernet, IP, TCP) | System prompt, formatting |
| Fragmentation | Packet fragmentation when > MTU | Conversation truncation when > context window |
| Cost | Bandwidth (bits/sec) | Price ($/1K tokens) |
| Processing | Router forwarding plane | Model inference compute |

### The Context Window

Every model has a **context window** -- the maximum number of tokens it can process in a single request. This includes your input (prompt) AND the model's output (response).

```
┌─────────────────────────────────────────────────┐
│              Context Window (200K tokens)         │
│                                                   │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐│
│  │System Prompt │  │Your Input│  │Model Response ││
│  │(instructions)│  │(config,  │  │(analysis,    ││
│  │             │  │logs, etc)│  │config, docs) ││
│  └─────────────┘  └──────────┘  └──────────────┘│
│                                                   │
│  ← These all share the same context window →     │
└─────────────────────────────────────────────────┘
```

**Practical implication**: If you paste a 50,000-line router config, that might consume 150K tokens of your context window, leaving less room for the response. Just like network engineers think about MTU and bandwidth, AI engineers think about token budgets.

### Why Tokens Matter for Cost

LLM APIs charge per token. Typical pricing (as of 2025-2026):

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
|-------|---------------------------|----------------------------|
| Claude Haiku 4.5 | $0.80 | $4.00 |
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| Claude Opus 4 | $15.00 | $75.00 |

For networking tasks:
- Analyzing a typical router config (~500 lines): ~2,000 input tokens ≈ $0.006 with Sonnet
- Generating documentation for 100 devices: ~200,000 input tokens ≈ $0.60 with Sonnet
- Using Haiku for simple extraction tasks cuts that cost by ~75%

Understanding tokens helps you make smart decisions about which model to use, how much context to include, and how to structure your prompts efficiently.

---

## The API: Your Interface to Intelligence

As a network engineer, you already understand APIs. You've probably used REST APIs for Meraki, DNA Center, or your monitoring platform. LLM APIs work the same way.

### The Basic Flow

```
┌──────────────┐     HTTPS POST      ┌──────────────┐
│              │  ────────────────▶  │              │
│ Your Script  │   JSON request      │  Claude API  │
│ (Python)     │                     │  (Anthropic) │
│              │  ◀────────────────  │              │
│              │   JSON response     │              │
└──────────────┘                     └──────────────┘
```

**That's it.** You send an HTTPS POST with your prompt in JSON. You get back a JSON response with the generated text. Same pattern as calling any REST API.

### Authentication

Like any API, you need credentials. For Claude, that's an API key:

```python
# Same concept as a Meraki API key or a RESTCONF token
# Store in environment variable, never hardcode
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Networking analogy**: An API key is like TACACS+ credentials for the AI service. It authenticates you, tracks your usage, and controls access. Just like you wouldn't hardcode TACACS credentials in a script, never hardcode API keys.

### The Request Structure

Every Claude API call has a consistent structure:

```python
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

response = client.messages.create(
    model="claude-sonnet-4-20250514",   # Which model to use
    max_tokens=1024,                     # Max response length
    system="You are a network engineer assistant.",  # System prompt
    messages=[                           # The conversation
        {"role": "user", "content": "Your question or task here"}
    ]
)

# The response
print(response.content[0].text)
```

| Parameter | Purpose | Networking Analogy |
|-----------|---------|-------------------|
| `model` | Which AI model to use | Choosing between ISR4451 and ASR1001 |
| `max_tokens` | Maximum response length | Output buffer size |
| `system` | Persistent instructions | A route-map applied to all traffic |
| `messages` | The conversation history | Packet payload |
| `temperature` | Randomness (0=deterministic, 1=creative) | Jitter/entropy |

---

## Your First API Call

Let's make this concrete. Here's a complete, working example that analyzes a piece of network configuration:

```python
from anthropic import Anthropic
import json

# Initialize client (uses ANTHROPIC_API_KEY environment variable)
client = Anthropic()

# A snippet of router configuration to analyze
config_snippet = """
interface GigabitEthernet0/1
 description Uplink to DC-SPINE-01
 ip address 10.0.1.1 255.255.255.252
 ip ospf cost 10
 ip ospf network point-to-point
 no shutdown
!
interface GigabitEthernet0/2
 description Link to WAN-EDGE-02
 ip address 10.0.2.1 255.255.255.252
 shutdown
"""

# Ask Claude to analyze the config
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a senior network engineer reviewing device configurations.",
    messages=[{
        "role": "user",
        "content": f"""Review this configuration snippet and identify:
1. Any operational concerns
2. Configuration best practices being followed
3. Anything that looks misconfigured or suspicious

Configuration:
{config_snippet}"""
    }]
)

print(response.content[0].text)
```

**Expected output** (will vary slightly each time):

> **Operational Concerns:**
> - GigabitEthernet0/2 is administratively shut down. If this is the link to WAN-EDGE-02, connectivity to that device is down. Verify if this is intentional (maintenance) or an oversight.
> - The /30 subnet masks suggest point-to-point links, which is consistent with the OSPF network type on Gi0/1.
>
> **Best Practices Followed:**
> - Interface descriptions are present and meaningful
> - OSPF point-to-point network type is used on a point-to-point link (avoids DR/BDR election overhead)
> - OSPF cost is explicitly set rather than relying on auto-cost
>
> **Potential Issues:**
> - Gi0/2 has no OSPF configuration. If this link should participate in OSPF, the network statement or interface-level OSPF config is missing.
> - Consider adding `ip ospf authentication message-digest` for OSPF security.

This is the power of generative AI for networking. You didn't write rules for "check if OSPF is configured on all interfaces" or "verify interface descriptions exist." You asked for a review, and the model applied its understanding of networking best practices.

---

## What Generative AI Can (and Cannot) Do for Network Engineers

Setting realistic expectations is critical. Hype helps no one. Here's an honest assessment.

### What It Does Well

**Configuration analysis and generation**
- Review configs for security issues, best practice violations, inconsistencies
- Generate configurations from natural language requirements
- Convert between vendor syntaxes (IOS to JunOS, NX-OS to EOS)
- Explain complex configurations in plain language

**Documentation**
- Generate device documentation from running configs
- Create topology diagrams from CDP/LLDP data
- Write runbooks and standard operating procedures
- Summarize change logs and audit trails

**Troubleshooting assistance**
- Correlate symptoms across multiple data sources (logs, show commands, configs)
- Suggest root causes and remediation steps
- Explain error messages and their implications
- Generate diagnostic command sequences

**Learning and knowledge**
- Explain networking concepts at any level of detail
- Compare protocols, technologies, and design approaches
- Interpret RFCs and vendor documentation
- Answer "why" questions about network behavior

**Code and automation**
- Write Python scripts for network tasks (Netmiko, NAPALM, Nornir)
- Generate Ansible playbooks and Jinja2 templates
- Create parsers for non-standard CLI output
- Build API integrations

### What It Does Poorly (or Can't Do)

**Real-time network access**
- An LLM cannot SSH into your devices. It can only analyze data you provide.
- It doesn't know the current state of your network unless you tell it.
- It can't run `show` commands or make config changes directly.

**Guaranteed correctness**
- LLMs can hallucinate -- generate plausible but wrong information.
- Never apply AI-generated configurations to production without review.
- Always verify factual claims against vendor documentation.

**Deterministic behavior**
- The same prompt may produce slightly different outputs each time.
- For tasks requiring exact reproducibility, use temperature=0 (more deterministic, but not perfectly so).
- Critical: don't rely on LLMs for binary pass/fail decisions in automation pipelines without validation.

**Mathematical precision**
- LLMs struggle with exact subnet calculations, especially with unusual masks.
- Use proper tools (ipcalc, Python's ipaddress module) for subnetting.
- The model might say 10.0.0.0/23 contains 510 usable hosts when the exact answer is 510 -- but it might also say 512 or 508. Trust, but verify.

**Proprietary or very recent information**
- The model's training data has a cutoff date. It may not know about:
  - Your organization's custom naming conventions
  - A vendor feature released last month
  - Your specific network topology
  - Internal procedures and policies

This is why **RAG** (Retrieval-Augmented Generation, covered in Chapter 14) matters -- it lets you feed your own documentation into the model's context.

---

## The Landscape: Models, Providers, and Trade-offs

As of 2025-2026, the major LLM providers relevant to network engineers are:

### Anthropic (Claude)

- **Models**: Claude Opus 4 (most capable), Claude Sonnet 4.5 (best balance), Claude Haiku 4.5 (fastest/cheapest)
- **Strengths**: Long context window (200K tokens), strong at following instructions, excellent at code and technical analysis
- **This book uses Claude** for all examples and code

### OpenAI (GPT)

- **Models**: GPT-4o, GPT-4o-mini, o1 (reasoning model)
- **Strengths**: Wide ecosystem, function calling, large marketplace of plugins
- **Note**: The concepts in this book apply to any provider -- the API patterns are similar

### Google (Gemini)

- **Models**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Strengths**: Very large context window (up to 1M tokens), multimodal (can process images of network diagrams)

### Local/Open-Source Models

- **Models**: Llama 3, Mistral, Qwen
- **Strengths**: Run on-premises, no data leaves your network, no per-token cost
- **Weaknesses**: Require GPU infrastructure, generally less capable than frontier models
- **Use case**: Organizations with strict data sovereignty requirements

### How to Choose

For this book, we use Claude because of its strong performance on technical tasks, but the principles apply universally:

| Priority | Recommendation |
|----------|---------------|
| Best quality for complex analysis | Opus 4 or GPT-4o |
| Best balance of quality and cost | Sonnet 4.5 (recommended starting point) |
| Lowest cost for simple tasks | Haiku 4.5 or GPT-4o-mini |
| Data must stay on-premises | Llama 3 or Mistral (self-hosted) |
| Largest context window | Gemini 1.5 Pro (1M tokens) |

**Networking analogy**: Choosing an LLM is like choosing a routing platform. A Catalyst 9300 and an ASR 9000 both forward packets, but they're designed for different scale and feature requirements. Pick the model that fits your use case -- don't use (or pay for) Opus when Haiku will do.

---

## Security and Privacy Considerations

Before you start sending network data to an AI API, you need to think about security. This is critical for network engineers, who routinely work with sensitive infrastructure data.

### What Data Are You Sending?

When you call the Claude API, your prompt -- including any configs, logs, or data you include -- is sent over HTTPS to Anthropic's servers. Ask yourself:

- Does this config contain passwords, SNMP community strings, or TACACS keys?
- Does this data include IP addresses that could reveal your internal topology?
- Are there compliance requirements (PCI-DSS, HIPAA, FedRAMP) that restrict where data can go?

### Practical Guidelines

**Always sanitize sensitive data before sending to the API:**

```python
import re

def sanitize_config(config: str) -> str:
    """Remove sensitive data from configs before sending to AI."""
    sanitized = config
    # Remove passwords
    sanitized = re.sub(
        r'(password|secret|key)\s+\d*\s+\S+',
        r'\1 *** REDACTED ***',
        sanitized, flags=re.IGNORECASE
    )
    # Remove SNMP communities
    sanitized = re.sub(
        r'(snmp-server community)\s+\S+',
        r'\1 REDACTED',
        sanitized, flags=re.IGNORECASE
    )
    # Remove TACACS keys
    sanitized = re.sub(
        r'(tacacs-server key)\s+\S+',
        r'\1 REDACTED',
        sanitized, flags=re.IGNORECASE
    )
    return sanitized
```

**Data handling by providers:**
- Anthropic does not use API data to train models (as of their current policy)
- Check the provider's data retention and processing policies
- For highly sensitive environments, consider self-hosted models

**Network-specific risks:**
- IP addresses in configs can reveal network topology
- ACLs can reveal security policies and what you're protecting
- BGP configurations can reveal peering relationships
- SNMP configs can reveal monitoring infrastructure

**Recommendation**: Start with non-sensitive tasks (documentation generation, learning, code writing) before working toward operational use cases with production data. Develop a data classification policy for what can and cannot be sent to external APIs.

---

## Key Takeaways

### For the Impatient Engineer

1. **Generative AI is a new tool, not a replacement.** It handles the reasoning and language tasks (config review, documentation, troubleshooting analysis) that traditional automation can't. Use both.

2. **It works through an API.** You send a prompt (text) over HTTPS, you get back a response (text). Same pattern as any REST API you've used. No ML expertise required.

3. **Tokens are the unit of everything** -- context, cost, and capacity. Roughly 1 token ≈ 0.75 words. A typical router config is ~2,000 tokens. Budget them like bandwidth.

4. **LLMs are not deterministic.** They can hallucinate. Never apply AI-generated configs to production without human review. Trust, but verify -- like any automation output.

5. **Start with the right model.** Sonnet 4.5 for most tasks. Haiku for simple/bulk work. Opus for complex analysis. Don't pay sports-car prices for a commuter-car workload.

6. **Security first.** Sanitize configs before sending to APIs. Know your data classification policy. Consider self-hosted models for sensitive environments.

### Networking Analogies Cheat Sheet

| AI Concept | Networking Equivalent |
|-----------|----------------------|
| LLM | A very experienced engineer who's read every doc ever written |
| API call | REST API request (like Meraki Dashboard API) |
| Token | Packet |
| Context window | Buffer/MTU (max data per request) |
| System prompt | Route-map (applied to all interactions) |
| Temperature | Jitter (randomness in output) |
| Hallucination | Route leak (plausible but incorrect information) |
| Fine-tuning | Vendor-specific training (specializing a generalist) |
| RAG | DNS (look up knowledge before answering) |
| Pre-training | Years of reading every RFC and config guide |
| Model selection | Choosing the right platform for the job (Cat9300 vs ASR9K) |

### What's Next

In **Chapter 2**, we dive deeper into how Large Language Models work -- the transformer architecture, attention mechanisms, and training process. Understanding the engine helps you drive better.

In **Chapter 3**, we cover how to choose the right model for your specific networking use case, with benchmarks and cost analysis.

But first, open the **Chapter 1 Colab notebook** and make your first API call. There's no substitute for hands-on experience -- you wouldn't learn OSPF without labbing it, and you won't learn AI without running the code.

---

*This chapter established the foundation. Generative AI is a powerful new tool for network engineers -- not magic, not hype, but a practical interface to machine intelligence that's accessible through the same API patterns you already know. The rest of this book shows you how to use it effectively, safely, and in production.*
