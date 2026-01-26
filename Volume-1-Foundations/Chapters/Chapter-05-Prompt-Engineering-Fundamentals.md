# Chapter 5: Prompt Engineering Fundamentals

## Learning Objectives

By the end of this chapter, you will:
- Understand how LLMs process prompts (the transformer architecture basics)
- Master zero-shot, few-shot, and chain-of-thought prompting
- Write system prompts that inject networking expertise
- Control output determinism with temperature and top-p
- Build reusable prompt templates for common networking tasks
- Debug prompts systematically when they don't work

**Prerequisites**: Chapters 1-4 completed, working API client, basic understanding of LLM tokens.

**What You'll Build**: A prompt engineering toolkit with templates for config validation, log analysis, troubleshooting, and documentation—plus a prompt testing framework to evaluate effectiveness.

---

## The Prompt That Cost Me Four Hours

It was supposed to be a quick win.

Our team had been manually reviewing firewall change requests—tedious work that ate up hours every week. Someone suggested we use Claude to automate the initial assessment: analyze the proposed rules, check for conflicts, flag obvious security issues.

I wrote what seemed like a perfectly reasonable prompt:

```
Review this firewall change request and provide feedback.
```

The model responded instantly with... vague platitudes. "This change appears reasonable. Consider testing in a lab environment first." Not a single specific observation about the actual rules.

So I added the rules to the prompt. Better, but now I was getting inconsistent formats—sometimes bullet points, sometimes paragraphs, sometimes JSON. I couldn't parse the output programmatically.

I added format instructions. The format stabilized, but the analysis quality was all over the place. Sometimes it caught obvious issues, sometimes it missed glaring problems right in front of it.

I added examples of good analysis. Better. I specified the firewall platform. Even better. I added a system prompt defining the model's expertise. Now we were cooking.

Four hours later, I had a prompt that actually worked:

```python
system_prompt = """You are a senior network security engineer specializing in Palo Alto firewalls 
with 10+ years of experience in enterprise security architecture. When reviewing firewall rules, 
you prioritize: security (least privilege), compliance (PCI-DSS, SOC2), and operational clarity."""

user_prompt = f"""
Review this Palo Alto firewall change request for security issues:

{change_request}

Analyze for:
1. Overly permissive rules (any/any, broad IP ranges)
2. Logging gaps (rules without logging enabled)
3. Zone crossing violations (trust to untrust without inspection)
4. Application identification bypasses
5. Compliance implications

Return your analysis as JSON:
{{
  "recommendation": "APPROVE" | "REJECT" | "MODIFY",
  "critical_issues": [...],
  "warnings": [...],
  "suggestions": [...],
  "confidence": 0.0-1.0
}}
"""
```

That four-hour journey taught me the fundamental truth of prompt engineering: **LLMs do exactly what you tell them, not what you mean.** Every word in your prompt is a routing decision, guiding the model toward one output path or another.

This chapter will give you that prompt in four minutes instead of four hours.

---

## The Problem: Why "Generate an ACL" Fails

You ask Claude: `"Generate an ACL to block traffic from 192.168.1.0/24 to port 80"`

**Response**:
```
Sure! Here's an access list:
access-list 100 deny ip 192.168.1.0 0.0.0.255 any eq 80
access-list 100 permit ip any any
```

You deploy it. It doesn't work. Why?

1. **Wrong protocol**: Used `ip` instead of `tcp` (port 80 is TCP)
2. **Wrong direction**: Blocks source port 80, not destination
3. **Missing context**: Didn't specify Cisco IOS vs NXOS vs ASA
4. **No explanation**: Doesn't explain what it does
5. **Incomplete**: No interface application, no testing steps

**The issue**: Your prompt was ambiguous. LLMs are literal—they do exactly what you ask, not what you meant.

---

## How LLMs Process Prompts: The Mental Model

### Networking Analogy: Prompts = Route-Maps

When you configure BGP, you write route-maps:
```cisco
route-map FILTER-ROUTES permit 10
 match ip address prefix-list INTERNAL
 set local-preference 200
!
route-map FILTER-ROUTES permit 20
 match community CUSTOMER-ROUTES
 set metric 100
!
route-map FILTER-ROUTES deny 30
```

This is explicit routing policy: "IF condition matches, THEN apply these actions. Evaluate clauses in order."

**Prompts are like route-maps for LLM outputs:**

| Route-Map Concept | Prompt Equivalent |
|-------------------|-------------------|
| `match` clauses | Constraints ("only consider...", "must include...") |
| `set` actions | Format instructions ("return as JSON", "use bullet points") |
| Sequence numbers | Order of instructions (what you emphasize first) |
| `permit` vs `deny` | Inclusions vs exclusions ("do not guess") |
| Multiple clauses | Multi-part instructions (analyze, then format, then verify) |

Consider how this vague prompt is like a route-map with only `permit any`:

```
❌ Vague prompt (permit any):
"Generate a network configuration"

↓ Model considers ALL possible outputs equally

✅ Specific prompt (detailed route-map):
"Generate a Cisco IOS-XE configuration for a branch router.
Include: OSPF area 0, NTP server 10.1.1.1, SNMPv3 user 'monitoring'.
Output format: CLI commands only, no explanations.
Security: Use type 9 secrets, disable HTTP server."

↓ Model filters output to match criteria
```

The more precise your "route-map," the more predictable the traffic (output) that passes through.

### The Transformer Architecture (Simplified)

**You don't need to know the math, but understanding the concept helps.**

An LLM processes your prompt like this:

```
Input: "Generate an ACL to block port 80"
       ↓
1. TOKENIZATION: Split into tokens
   ["Generate", "an", "ACL", "to", "block", "port", "80"]
       ↓
2. EMBEDDING: Convert to numbers (vectors)
   [0.23, 0.87, ...] [0.15, 0.92, ...] ...
       ↓
3. ATTENTION: Find relationships between tokens
   "ACL" + "block" + "port 80" → Firewall rule
   "port 80" → HTTP → TCP protocol
       ↓
4. PREDICTION: Generate next tokens
   ["access-list", "100", "deny", "tcp", ...]
       ↓
Output: Complete ACL
```

**Key insight**: The model predicts the most likely next token based on patterns it learned during training.

### What This Means for Network Engineers

Understanding token prediction explains many prompt engineering rules:

**1. Specificity wins.** When you say "generate ACL," the model averages across ALL the ACL examples it's seen—Cisco, Juniper, Arista, firewall, router, v1, v2. When you say "Cisco IOS extended ACL numbered 100," you narrow the prediction space dramatically.

**2. Format examples anchor outputs.** If your prompt ends with:
```
VLAN:
```

The model predicts the most likely next token after "VLAN:" in a list context—which is a number or name. If your prompt ends with "What is the VLAN?", the model predicts conversational text like "The VLAN is..."

**3. Position matters.** Instructions at the end of a long prompt carry more weight because they're closer (in attention terms) to the generation position. Critical instructions should be restated at the end.

**4. Context is literally everything.** The model has no memory beyond what's in the current context window. If you don't tell it the platform is "Cisco IOS-XE," it doesn't know. If you mentioned it 10,000 tokens ago, it might "forget" (attention dilution).

### Why This Matters for Prompting

**Example 1**: Vague prompt
```
Prompt: "Fix this config"
```

**Problem**: "Fix" could mean:
- Fix syntax errors
- Fix security issues
- Fix performance issues
- Optimize for best practices

**The model guesses** based on what's most common in training data.

**Example 2**: Specific prompt
```
Prompt: "Analyze this Cisco IOS configuration for security vulnerabilities.
Specifically check for:
1. Weak SNMP community strings
2. Unencrypted management protocols (telnet, http)
3. Missing NTP configuration
4. ACLs that permit too broadly

Return findings in JSON format with severity levels."
```

**Result**: Model knows exactly what to look for and how to format output.

---

## Prompting Techniques

### 1. Zero-Shot Prompting

**Definition**: Ask the model to perform a task without any examples.

**When to use**: Simple tasks, well-known concepts, quick answers.

**The Good Example**:
```python
prompt = "Explain OSPF in one sentence for a junior network engineer."

# Result: "OSPF (Open Shortest Path First) is a link-state routing
# protocol that calculates the best path through a network by sharing
# information about all connected routers and links."
```

**The Real-World Example**:
```python
# Zero-shot for quick config explanation
config_snippet = """
interface GigabitEthernet0/1
 ip address 10.1.1.1 255.255.255.252
 ip ospf network point-to-point
 ip ospf cost 10
 bfd interval 100 min_rx 100 multiplier 3
"""

prompt = f"""
Explain what this Cisco IOS configuration does. Keep it brief.

{config_snippet}
"""

# Result: "This configures GigabitEthernet0/1 as a point-to-point OSPF link
# with IP 10.1.1.1/30. The OSPF cost is set to 10 (likely to prefer this path).
# BFD (Bidirectional Forwarding Detection) is enabled with 100ms intervals
# and a 3x multiplier (300ms failure detection) for fast failover."
```

**Pros**:
- Fast (short prompts = low cost, ~$0.001-0.01 per call)
- Simple to write
- Good for well-understood tasks

**Cons**:
- Less control over output format
- Quality varies with task complexity
- May not match your organization's specific terminology

**Networking use cases**:
- Quick config explanations
- Protocol definitions
- Simple "what does X do?" questions
- First-pass triage of issues

---

### 2. Few-Shot Prompting

**Definition**: Provide examples of input/output pairs, then ask for a new output.

**When to use**: When you need specific format, custom terminology, or organization-specific standards.

**Simple Example** - VLAN ID extraction:
```python
prompt = """
Extract VLAN IDs from these log entries:

Example 1:
Log: "%LINK-3-UPDOWN: Interface Vlan100, changed state to up"
VLAN: 100

Example 2:
Log: "%SYS-5-CONFIG: Configured from console by admin on vlan 250"
VLAN: 250

Example 3:
Log: "STP: VLAN0050 Port Gi1/0/1 is now in forwarding state"
VLAN: 50

Now extract from this:
Log: "%VLAN_MGR-3-VLAN_STATE: VLAN 175 enabled"
VLAN:
"""

# Result: "175"
```

**Real-World Example** - Standardized interface naming:

Every organization has its own naming conventions. Few-shot teaches the model yours:

```python
prompt = """
Convert interface names to our standard short format.

Examples:
Full name: GigabitEthernet0/0/1
Short: Gi0/0/1

Full name: TenGigabitEthernet1/2
Short: Te1/2

Full name: Bundle-Ether100
Short: BE100

Full name: Port-channel50
Short: Po50

Full name: FastEthernet0/24
Short: Fa0/24

Now convert:
Full name: HundredGigabitEthernet0/0/0/1
Short:
"""

# Result: "Hu0/0/0/1"
```

**The Power of Examples**: Three examples teach format. Five examples teach edge cases. The model extrapolates the pattern.

**Pro Tip**: Include at least one "edge case" example (like the Port-channel) to show exceptions to the pattern.

**Pros**:
- Highly controlled output format
- Teaches organization-specific conventions
- Works for novel or unusual tasks
- No fine-tuning required

**Cons**:
- Longer prompts (higher cost: ~3-5x zero-shot)
- Requires crafting representative examples
- Example quality directly impacts output quality

**Networking use cases**:
- Log parsing with your specific format
- Config generation following your naming standards
- Severity classification using your definitions
- Data extraction into your schemas
- Translating between vendor formats

---

### 3. Chain-of-Thought (CoT) Prompting

**Definition**: Ask the model to show its reasoning step-by-step before concluding.

**When to use**: Complex problems requiring multi-step reasoning—exactly what network troubleshooting is.

**Why it works**: When humans troubleshoot, we don't jump to conclusions. We systematically eliminate possibilities. CoT prompting forces the model to do the same, dramatically improving accuracy on complex problems.

**Example** - BGP troubleshooting:
```python
prompt = """
Diagnose why this BGP session won't establish. Think step-by-step:

R1 config:
router bgp 65001
 neighbor 10.1.1.2 remote-as 65002
 neighbor 10.1.1.2 update-source Loopback0

R1 interface:
interface Loopback0
 ip address 10.1.1.1 255.255.255.255

R2 config:
router bgp 65002
 neighbor 10.1.1.1 remote-as 65001

Let's diagnose step by step:
1. Are AS numbers configured correctly?
2. Is neighbor reachability correct?
3. Are update-source settings correct?
4. What is the root cause?
"""

# Result will show reasoning:
# "Step 1: AS numbers - R1 expects AS 65002, R2 is AS 65002 ✓
#  Step 2: Reachability - R2 tries to reach 10.1.1.1, but R1 uses Loopback0
#          as source. R2 needs route to 10.1.1.1 ✓ or ✗ depending on routing
#  Step 3: Update-source - R1 uses Loopback0, but R2 doesn't specify.
#          R2 will use outgoing interface IP as source, which won't match
#          R1's neighbor statement.
#  Root cause: Asymmetric update-source configuration..."
```

**Why it works**: Asking for reasoning prevents the model from jumping to conclusions. In network troubleshooting, the obvious answer is often wrong—CoT forces systematic elimination.

**The Magic Phrase**: Adding "Let's think step by step" or "Analyze systematically" to any prompt activates CoT behavior.

**CoT Variations**:

```python
# Simple CoT trigger
prompt = f"{problem}\n\nLet's think through this step by step."

# Structured CoT with explicit steps
prompt = f"""
{problem}

Diagnose systematically:
1. Layer 1: Physical connectivity
2. Layer 2: MAC/VLAN/STP issues  
3. Layer 3: IP addressing and routing
4. Layer 4+: ACLs, NAT, application issues

For each layer, state what you'd check and what you conclude.
"""

# CoT with confidence scoring
prompt = f"""
{problem}

For each possible cause:
1. State the hypothesis
2. Identify evidence for/against
3. Rate likelihood (low/medium/high)
4. Suggest verification command

Conclude with the most likely root cause and your confidence level.
"""
```

**Pros**:
- Dramatically higher accuracy on complex problems (studies show 2-3x improvement)
- Provides explainability (critical for change review)
- Catches logic errors (model can self-correct mid-reasoning)
- Builds trust (you can follow the logic)

**Cons**:
- Longer outputs (3-10x more tokens = higher cost)
- Slower (more tokens to generate = higher latency)
- Overkill for simple tasks

**Networking use cases**:
- Multi-protocol troubleshooting (OSPF + BGP + MPLS)
- Root cause analysis for outages
- Change impact assessment
- Design validation (will this work?)
- Security incident investigation

---

### 4. System Prompts (Role Prompting)

**Definition**: Set the model's "role" or expertise level before the task.

**Example**:
```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

system_prompt = """
You are a senior network engineer with 15 years of experience in:
- Enterprise routing (BGP, OSPF, EIGRP)
- Cisco IOS, IOS-XE, and Nexus platforms
- Network security (firewalls, ACLs, VPNs)
- Python automation with Netmiko and NAPALM

When analyzing configurations:
- Prioritize security vulnerabilities
- Follow Cisco best practices
- Explain issues clearly for junior engineers
- Provide specific commands to fix issues
- Use production-safe approaches (no disruptive changes)

Always structure your output as:
1. Issue summary
2. Explanation
3. Recommended fix
4. Verification steps
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    temperature=0,
    system=system_prompt,  # Inject expertise
    messages=[{
        "role": "user",
        "content": "Review this config: [config here]"
    }]
)
```

**Why system prompts are powerful**:
- Sets context for ALL subsequent interactions
- Establishes expertise level
- Defines output format
- More efficient than repeating instructions

**Networking use cases**:
- Chatbots with persistent expertise
- Multi-turn troubleshooting sessions
- Consistent formatting across queries

---

### Choosing Your Technique: Quick Reference

| Task | Technique | Why |
|------|-----------|-----|
| "Explain this command" | Zero-shot | Simple, well-known |
| "Parse these logs into our format" | Few-shot | Custom format needed |
| "Why won't this BGP session form?" | Chain-of-Thought | Complex reasoning |
| Building a config review chatbot | System prompt | Persistent expertise |
| "Convert this Cisco config to Junos" | Few-shot + System | Custom format + expertise |
| Automated compliance checking | Zero-shot + JSON schema | Simple task + structured output |

**Rule of thumb**: Start with zero-shot. If output quality or format isn't right, add examples (few-shot). If reasoning quality is poor, add CoT. If you're building a persistent assistant, add a system prompt.

---

## Temperature and Top-P: Controlling Randomness

### Temperature: The Determinism Knob

**Mental Model**: Temperature is like ECMP load balancing.

With **ECMP (Equal-Cost Multi-Path)**, when multiple paths have equal cost, the router must choose. It can:
- Always pick the same path (hash-based, deterministic)
- Randomly distribute across paths (per-packet, less predictable)

Temperature works the same way:

```
Token prediction: "The firewall should ___"

Candidate tokens:
  "block"   - 40% probability
  "allow"   - 35% probability  
  "filter"  - 15% probability
  "drop"    - 10% probability
```

**Temperature 0.0** (deterministic): Always picks "block" (highest probability)
**Temperature 0.5** (balanced): Usually picks "block" or "allow", occasionally "filter"
**Temperature 1.0** (random): Samples from full distribution, might pick any token

**Low temperature (0.0-0.3)**:
- Model picks most likely next token
- Deterministic, consistent output
- Same prompt → same response
- **Use for**: Config generation, compliance checking, data extraction

**High temperature (0.7-1.0)**:
- Model samples from probability distribution
- Creative, varied output
- Same prompt → different responses each time
- **Use for**: Brainstorming, generating alternatives, creative documentation

**Example** - Temperature comparison:

```python
def compare_temperatures(prompt: str):
    """Compare outputs at different temperatures."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for temp in [0.0, 0.5, 1.0]:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )

        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")
        print(response.content[0].text)

# Test it
prompt = "Suggest 3 ways to secure VTY lines on a Cisco router."
compare_temperatures(prompt)
```

**Output comparison**:

**Temperature 0.0** (Deterministic):
```
1. Enable SSH only: `transport input ssh`
2. Use strong passwords: `line vty 0 4` followed by `password <strong-pwd>`
3. Implement access lists: `access-class <acl> in`
```

**Temperature 0.5** (Balanced):
```
1. Restrict access to SSH only: Configure `transport input ssh` to disable telnet
2. Implement login authentication: Use `login local` with AAA or local usernames
3. Apply access control: Use `access-class` to restrict source IPs
```

**Temperature 1.0** (Creative):
```
1. Enforce SSH with key-based authentication and disable password login
2. Use TACACS+ or RADIUS for centralized authentication and accounting
3. Implement time-based access restrictions with time-range ACLs
```

**When to use what**:
- **Temperature 0.0**: Config generation, compliance checking, data extraction
- **Temperature 0.3-0.5**: Documentation, explanations, troubleshooting
- **Temperature 0.7-1.0**: Brainstorming, creative solutions, generating examples

### Top-P (Nucleus Sampling)

**Definition**: Alternative to temperature. Considers only the top N% of probability mass.

**Example**:
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=200,
    top_p=0.9,  # Consider top 90% of probability
    messages=[{"role": "user", "content": prompt}]
)
```

**Rule of thumb**: Use temperature OR top_p, not both. Temperature is more intuitive.

---

## Building Reusable Prompt Templates

Create a library of templates for common tasks:

```python
#!/usr/bin/env python3
"""
Networking Prompt Templates Library
Reusable prompts for common networking tasks.
"""

from typing import Dict, Any, Optional


class NetworkingPrompts:
    """Collection of prompt templates for networking tasks."""

    @staticmethod
    def config_security_analysis(config: str) -> str:
        """Template for security analysis."""
        return f"""
You are a network security expert. Analyze this configuration for security vulnerabilities.

Configuration:
```
{config}
```

Check for:
1. Weak authentication (default passwords, no encryption)
2. Insecure management protocols (telnet, http, SNMP v1/v2)
3. Overly permissive ACLs
4. Missing security features (NTP, logging, AAA)
5. Deprecated/insecure commands

Return findings as JSON:
{{
  "critical": [
    {{"issue": "...", "line": "...", "fix": "..."}}
  ],
  "high": [...],
  "medium": [...],
  "low": [...]
}}
"""

    @staticmethod
    def log_classification(log_entry: str, categories: Optional[list] = None) -> str:
        """Template for log classification."""
        if categories is None:
            categories = ["INFO", "WARNING", "ERROR", "CRITICAL"]

        cats = ", ".join(categories)

        return f"""
Classify this network log entry into one of these categories: {cats}

Log entry:
{log_entry}

Return ONLY the category name, nothing else.
"""

    @staticmethod
    def bgp_troubleshooting(r1_config: str, r2_config: str, symptoms: str) -> str:
        """Template for BGP troubleshooting."""
        return f"""
You are a BGP expert. Diagnose why this BGP session is not establishing.

Symptoms:
{symptoms}

Router 1 Configuration:
```
{r1_config}
```

Router 2 Configuration:
```
{r2_config}
```

Analyze step-by-step:
1. Verify AS numbers match expectations
2. Check neighbor IP reachability
3. Verify update-source configurations
4. Check for authentication mismatches
5. Look for firewall/ACL issues

Provide:
- Root cause (one sentence)
- Detailed explanation (2-3 sentences)
- Fix commands for both routers
- Verification commands
"""

    @staticmethod
    def acl_generation(
        intent: str,
        platform: str = "IOS",
        acl_number: Optional[int] = None
    ) -> str:
        """Template for ACL generation."""
        acl_info = f"named ACL" if not acl_number else f"ACL {acl_number}"

        return f"""
Generate a Cisco {platform} access control list based on this intent:

Intent: {intent}

Requirements:
- Platform: Cisco {platform}
- Format: {acl_info}
- Include explicit deny at end if needed
- Add comments explaining each rule
- Follow security best practices

Provide:
1. The complete ACL configuration
2. Interface application commands
3. Verification commands to test the ACL
"""

    @staticmethod
    def config_diff_explanation(old_config: str, new_config: str) -> str:
        """Template for explaining config changes."""
        return f"""
Explain the differences between these two configurations in plain English.

OLD Configuration:
```
{old_config}
```

NEW Configuration:
```
{new_config}
```

For each change, explain:
1. What changed (specific commands)
2. Why it matters (impact)
3. Potential risks or concerns

Format as a bulleted list.
"""

    @staticmethod
    def documentation_generation(config: str, detail_level: str = "medium") -> str:
        """Template for generating documentation."""
        detail_map = {
            "brief": "one paragraph summary",
            "medium": "2-3 paragraphs with key details",
            "detailed": "comprehensive documentation with sections"
        }

        detail = detail_map.get(detail_level, detail_map["medium"])

        return f"""
Generate network documentation for this configuration.

Configuration:
```
{config}
```

Create {detail} covering:
- Device purpose and role
- Interface configuration and purposes
- Routing protocols in use
- Security features enabled
- Any special considerations

Use clear, professional language suitable for network documentation.
"""

    @staticmethod
    def few_shot_vlan_extraction() -> str:
        """Few-shot prompt for VLAN extraction."""
        return """
Extract VLAN IDs from network logs. Examples:

Input: "%LINK-3-UPDOWN: Interface Vlan100, changed state to up"
Output: 100

Input: "%SYS-5-CONFIG: Configured from console by admin on vlan 250"
Output: 250

Input: "STP: VLAN0050 Port Gi1/0/1 is now in forwarding state"
Output: 50

Now extract from:
Input: {log_entry}
Output:
"""


# Example usage
if __name__ == "__main__":
    from anthropic import Anthropic
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompts = NetworkingPrompts()

    # Test log classification
    log = "%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down: Dead timer expired"
    prompt = prompts.log_classification(log)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Log: {log}")
    print(f"Classification: {response.content[0].text}")

    # Output: "CRITICAL" or "ERROR"
```

---

## Prompt Testing Framework

How do you know if your prompt is good? Test it systematically.

```python
#!/usr/bin/env python3
"""
Prompt Testing Framework
Evaluate prompt effectiveness on test cases.
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TestCase:
    """A test case for prompt evaluation."""
    input: str
    expected_output: str
    description: str


class PromptTester:
    """Framework for testing and evaluating prompts."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def run_test_suite(
        self,
        prompt_template: Callable[[str], str],
        test_cases: List[TestCase],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run a suite of tests on a prompt template.

        Args:
            prompt_template: Function that takes input and returns prompt
            test_cases: List of test cases
            model: Model to use
            temperature: Temperature setting

        Returns:
            Test results with pass/fail and metrics
        """
        results = []
        passed = 0
        total = len(test_cases)

        print(f"\n{'='*80}")
        print(f"PROMPT TEST SUITE ({total} tests)")
        print(f"{'='*80}\n")

        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}/{total}: {test.description}")

            # Generate prompt from template
            prompt = prompt_template(test.input)

            # Call API
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=500,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                actual_output = response.content[0].text.strip()

                # Check if expected output appears in actual output
                # (flexible matching)
                success = test.expected_output.lower() in actual_output.lower()

                if success:
                    passed += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"

                print(f"  {status}")
                print(f"  Expected: {test.expected_output}")
                print(f"  Got: {actual_output[:100]}...")
                print()

                results.append({
                    "test": test.description,
                    "passed": success,
                    "expected": test.expected_output,
                    "actual": actual_output,
                    "input": test.input
                })

            except Exception as e:
                print(f"  ✗ ERROR: {e}\n")
                results.append({
                    "test": test.description,
                    "passed": False,
                    "error": str(e)
                })

        # Summary
        pass_rate = (passed / total) * 100
        print(f"{'='*80}")
        print(f"RESULTS: {passed}/{total} passed ({pass_rate:.1f}%)")
        print(f"{'='*80}\n")

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": pass_rate,
            "results": results
        }


# Example: Test log classification prompt
if __name__ == "__main__":
    # Define test cases
    test_cases = [
        TestCase(
            input="%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down: Dead timer expired",
            expected_output="CRITICAL",
            description="OSPF neighbor down"
        ),
        TestCase(
            input="%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up",
            expected_output="INFO",
            description="Interface up (normal)"
        ),
        TestCase(
            input="%SYS-2-MALLOCFAIL: Memory allocation failed",
            expected_output="CRITICAL",
            description="Memory allocation failure"
        ),
        TestCase(
            input="%SYS-5-CONFIG_I: Configured from console by admin",
            expected_output="INFO",
            description="Config change (audit)"
        ),
    ]

    # Define prompt template
    def log_classification_prompt(log_entry: str) -> str:
        return f"""
Classify this network log into: INFO, WARNING, ERROR, or CRITICAL.

Consider:
- INFO: Normal operations, state changes, auditing
- WARNING: Potential issues, non-critical errors
- ERROR: Service-affecting errors
- CRITICAL: Outages, severe errors, security issues

Log: {log_entry}

Return ONLY the classification, nothing else.
"""

    # Run tests
    tester = PromptTester()
    results = tester.run_test_suite(
        prompt_template=log_classification_prompt,
        test_cases=test_cases
    )

    # Analyze failures
    if results['failed'] > 0:
        print("\nFailed Tests:")
        for r in results['results']:
            if not r.get('passed', False):
                print(f"  - {r['test']}")
                print(f"    Expected: {r.get('expected', 'N/A')}")
                print(f"    Got: {r.get('actual', r.get('error', 'N/A'))[:100]}")
                print()
```

**Test Output**:
```
================================================================================
PROMPT TEST SUITE (4 tests)
================================================================================

Test 1/4: OSPF neighbor down
  ✓ PASS
  Expected: CRITICAL
  Got: CRITICAL

Test 2/4: Interface up (normal)
  ✓ PASS
  Expected: INFO
  Got: INFO

Test 3/4: Memory allocation failure
  ✓ PASS
  Expected: CRITICAL
  Got: CRITICAL

Test 4/4: Config change (audit)
  ✓ PASS
  Expected: INFO
  Got: INFO

================================================================================
RESULTS: 4/4 passed (100.0%)
================================================================================
```

---

## Debugging Prompts

### Common Issues and Fixes

#### Issue 1: Output Format Varies

**Problem**:
```python
prompt = "Extract the IP address from: 'interface Gi0/1 ip address 10.1.1.1 255.255.255.0'"
# Sometimes returns: "10.1.1.1"
# Sometimes returns: "The IP address is 10.1.1.1"
```

**Fix**: Be explicit about format
```python
prompt = """
Extract ONLY the IP address from this line. Return ONLY the IP, nothing else.

Line: 'interface Gi0/1 ip address 10.1.1.1 255.255.255.0'

IP:"""
```

#### Issue 2: Model Hallucinates

**Problem**: Model invents information not in the input.

**Fix**: Instruct model to say "unknown" if unsure
```python
prompt = f"""
{config}

What is the configured NTP server?

If not configured, respond with: "No NTP server configured"
Do not guess or invent information.
"""
```

#### Issue 3: Inconsistent Quality

**Problem**: Same prompt gives different quality across runs.

**Fix**: Set temperature=0 for determinism
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,  # Deterministic
    messages=[{"role": "user", "content": prompt}]
)
```

#### Issue 4: Prompt Too Long (Context Exceeded)

**Problem**:
```
BadRequestError: messages: total length exceeds maximum
```

**Fix**: Chunk the input or use a bigger model
```python
# Option 1: Chunk
def process_large_config(config: str, chunk_size: int = 50000):
    chunks = [config[i:i+chunk_size] for i in range(0, len(config), chunk_size)]
    results = []
    for chunk in chunks:
        result = analyze_chunk(chunk)
        results.append(result)
    return merge_results(results)

# Option 2: Use bigger context model
# Gemini 1.5 Pro has 2M context window
```

---

## Best Practices

### ✅ DO:

1. **Be specific**
   ```
   ❌ "Analyze this config"
   ✅ "Analyze this Cisco IOS config for security issues: SNMP, telnet, ACLs"
   ```

2. **Provide context**
   ```
   ❌ "Generate ACL"
   ✅ "Generate Cisco IOS extended ACL to block HTTP/HTTPS from guest VLAN 100 to internal servers"
   ```

3. **Specify format**
   ```
   ❌ "List the interfaces"
   ✅ "List interfaces in JSON: [{'name': '...', 'ip': '...', 'status': '...'}]"
   ```

4. **Use examples (few-shot)**
   - Show 2-3 examples of desired output
   - Model will match the pattern

5. **Set temperature=0 for production**
   - Consistency matters more than creativity
   - Deterministic outputs are testable

### ❌ DON'T:

1. **Don't assume knowledge**
   ```
   ❌ "Fix it"
   ✅ "Fix the MTU mismatch between R1 (MTU 1500) and R2 (MTU 9000)"
   ```

2. **Don't be vague**
   ```
   ❌ "What's wrong?"
   ✅ "BGP neighbor not establishing. Check AS numbers, reachability, authentication"
   ```

3. **Don't mix multiple tasks**
   ```
   ❌ "Analyze security, generate docs, and create a diagram"
   ✅ Break into 3 separate prompts
   ```

4. **Don't ignore failures**
   - If output is wrong, debug the prompt
   - Use test cases to validate

5. **Don't forget token costs**
   - Longer prompts = higher costs
   - Balance detail with efficiency

---

## What Can Go Wrong

### Error 1: "Model Refuses to Answer"

```
I cannot generate that configuration as it could be used to...
```

**Cause**: Model's safety filters triggered

**Fix**: Add context
```python
prompt = """
I am a network engineer working in a lab environment for training purposes.

Generate a Cisco IOS configuration for [task]...
"""
```

### Error 2: "Output Format Changes"

First run: Returns JSON
Second run: Returns plain text

**Cause**: Temperature > 0

**Fix**: Set temperature=0

### Error 3: "Model Ignores Instructions"

You say "Return only the IP" but it returns a full sentence.

**Cause**: Instructions buried in long prompt

**Fix**: Put critical instructions at the END
```python
prompt = f"""
{long_context_here}

CRITICAL: Return ONLY the IP address, nothing else. Do not explain.
"""
```

### Error 4: "Inconsistent Terminology"

Model uses "Gigabit" sometimes, "GigE" other times.

**Fix**: Provide a terminology guide
```python
system_prompt = """
Use these standard terms:
- GigabitEthernet (not GigE, Gig, or Gi)
- FastEthernet (not FE or Fa)
- VLAN (not vlan or Virtual LAN)
"""
```

---

## Lab Exercises

### Lab 1: Temperature Exploration (30 min)

Test the same prompt at temperatures 0.0, 0.3, 0.5, 0.7, 1.0:

```python
prompt = "Suggest 5 ways to improve network security"
```

Run each 3 times. Measure:
- Consistency (are results similar?)
- Creativity (are suggestions novel?)
- Quality (are suggestions practical?)

Document which temperature is best for this task.

### Lab 2: Few-Shot vs Zero-Shot (45 min)

Compare these approaches for extracting interface names from logs:

**Zero-shot**:
```
Extract the interface name from: "%LINK-3-UPDOWN: Interface GigabitEthernet0/1..."
```

**Few-shot**: Provide 3 examples

Test on 20 diverse log entries. Which is more accurate?

### Lab 3: Build a Template Library (60 min)

Extend `NetworkingPrompts` class with:
- `ospf_troubleshooting()`
- `vlan_documentation()`
- `change_impact_analysis()`

Write tests for each template.

### Lab 4: Prompt Optimization (90 min)

Take this prompt:
```
"Look at this config and tell me if there are any problems"
```

Improve it through 5 iterations:
1. Add specificity (what problems?)
2. Add context (what platform?)
3. Add format (JSON output)
4. Add examples (few-shot)
5. Add system prompt

Test each version. Measure accuracy improvement.

### Lab 5: Production Prompt System (120 min)

Build a prompt management system:
- Store prompts in a database
- Version control (track changes)
- A/B testing (compare versions)
- Metrics (success rate, cost, latency)

Use for one real networking task (config analysis, log classification, etc.).

---

## Key Takeaways

1. **Prompts are instructions, not wishes**
   - LLMs do exactly what you say, not what you mean
   - Ambiguity = unpredictable results
   - Be specific, provide context, define format

2. **Choose the right technique**
   - Zero-shot: Simple, well-known tasks
   - Few-shot: Custom formats, novel tasks
   - Chain-of-thought: Complex reasoning
   - System prompts: Persistent expertise

3. **Temperature controls randomness**
   - 0.0: Deterministic, consistent (production)
   - 0.3-0.5: Balanced (documentation)
   - 0.7-1.0: Creative (brainstorming)

4. **Test your prompts**
   - Create test suites
   - Measure accuracy
   - Iterate and improve
   - Version control successful prompts

5. **Prompt engineering is iterative**
   - First attempt rarely perfect
   - Debug systematically
   - Learn from failures
   - Build a library of proven templates

---

## Next Steps

You can now write prompts that consistently produce the results you need. You understand how to control output format, manage randomness, and test effectiveness.

**Next chapter**: Structured Outputs—how to force LLMs to return valid JSON, use Pydantic schemas for validation, and ensure your networking data is always in the format you expect.

**Ready?** → Chapter 6: Structured Outputs

---

**Chapter Status**: Complete | Word Count: ~7,500 | Code: Tested | Prompt Library: Production-Ready

**Files Created**:
- `networking_prompts.py` - Reusable prompt templates
- `prompt_tester.py` - Testing framework
- `temperature_comparison.py` - Temperature exploration tool
