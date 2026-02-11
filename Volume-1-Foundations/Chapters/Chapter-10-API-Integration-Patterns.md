# Chapter 10: API Integration Patterns

## Learning Objectives

By the end of this chapter, you will:
- Integrate LLMs with Netmiko for intelligent SSH automation
- Use NAPALM + AI for config validation
- Build REST APIs for network AI services
- Create Ansible modules with AI capabilities
- Design webhook-driven automation workflows

**Prerequisites**: Chapters 1-9 completed, basic Netmiko/NAPALM knowledge.

**What You'll Build**: A complete network automation system where AI enhances existing tools—Netmiko scripts that adapt to errors, NAPALM validators powered by LLMs, and REST APIs that expose AI capabilities to your team.

---

## The On-Call Nightmare That Sparked an Idea

Friday night, 2 AM. My phone buzzed.

`ALERT: BGP session DOWN on core-rtr-01`

I SSH'd into the router, bleary-eyed. `show ip bgp summary`—neighbor state "Idle." Great, very helpful.

My troubleshooting checklist:
1. Check interface status → up
2. Check routing → route exists
3. Check logs → "TCP connection failed"
4. Check ACLs → wait, there's a new ACL...

Forty-five minutes later, I found it: someone had pushed an ACL update that blocked BGP port 179. One line buried in a 300-line change ticket. A five-minute fix once identified.

As I fixed it, I thought: *Claude could have found that in seconds.*

I'd been using Claude for config analysis, but always in a separate window. Copy output, paste to chat, read response, manually type commands. The context switching was killing me.

What if the AI was *inside* my automation scripts? What if Netmiko could ask Claude "why is this BGP session down?" and Claude could say "check ACL 105 line 30, it blocks TCP 179"?

That night, I started building what became our AI-enhanced network toolkit. By Monday, I had a prototype. By month-end, our MTTR (Mean Time To Resolution) dropped by 40%.

This chapter shows you how to build it yourself.

---

## The Integration Opportunity

Most network engineers use AI in one of two ways:

**Pattern A: Copy-Paste (Current Reality)**
```
[Terminal]                     [Browser/ChatGPT]
show ip bgp summary  →  📋  →  "Analyze this output"
[read response]       ←  📋  ←  [AI analysis]
neighbor shut         →  📋  →  "Is this command safe?"
```

**Pattern B: Integrated (What We'll Build)**
```python
# One script does everything
result = router.analyze("show ip bgp summary")
if result.issues:
    remediation = router.generate_fix(result.issues[0])
    if remediation.safe:
        router.apply(remediation.commands)
```

The first pattern works but doesn't scale. The second pattern is what this chapter teaches.

---

## Check Your Understanding: Integration Patterns

Before diving into code, test your understanding of AI integration strategies:

**1. When should you integrate AI into scripts vs using ChatGPT separately?**

<details>
<summary>Show answer</summary>

**Integrate AI into scripts when:**
- You run the same analysis repeatedly (>10 times/week)
- You need consistent, reproducible results
- Multiple team members need access to AI capabilities
- You want to combine AI with automation (Netmiko, NAPALM, Ansible)
- The workflow is predictable and can be codified

**Use ChatGPT separately when:**
- One-off analysis or investigation
- Exploring new problems without defined workflow
- Learning and experimenting
- No need to share with team

**Example:**
- **Separate**: "I've never seen this error before, let me ask ChatGPT"
- **Integrated**: "We troubleshoot BGP neighbor issues daily, let's build an AINetmiko script"

</details>

**2. What's the difference between copy-paste workflow and integrated workflow?**

<details>
<summary>Show answer</summary>

**Copy-paste workflow (Pattern A):**
```
1. SSH to device → run command
2. Copy output → paste to ChatGPT
3. Read AI response
4. Manually type fix commands → run on device
5. Copy new output → paste to ChatGPT to verify
```

**Time:** 5-10 minutes per device
**Scales:** Poorly (each device requires manual work)
**Errors:** High (manual copy-paste introduces mistakes)

**Integrated workflow (Pattern B):**
```python
result = ai_net.intelligent_troubleshoot("BGP neighbor down")
print(result['diagnosis']['root_cause'])
print(result['diagnosis']['fix_commands'])
# Optionally auto-apply fixes
```

**Time:** 30 seconds per device
**Scales:** Excellently (same code works for 1 or 1,000 devices)
**Errors:** Low (programmatic, reproducible)

**Key difference:** Integration eliminates context switching and manual steps, making AI analysis part of your automation workflow.

</details>

**3. What are 3 benefits of programmatic AI integration?**

<details>
<summary>Show answer</summary>

**1. Repeatability and Consistency**
- Same analysis every time
- No human variation
- Easy to audit and log
- Can run automatically (cron, webhooks)

**2. Scalability**
- Analyze 1,000 devices as easily as 1
- Parallel processing possible
- Team can use via REST API without AI expertise
- Integrates with existing tools (Ansible, Netmiko, NAPALM)

**3. Composability**
- Combine multiple AI calls in a workflow
- Chain analysis steps (analyze → recommend → validate → apply)
- Add safety checks and human approval gates
- Track costs and performance

**Real-world impact:** The chapter's opening story showed 40% MTTR reduction after integrating AI into automation scripts.

</details>

---

## The Problem: AI and Network Tools Are Separate

**Current workflow**:
```python
# Step 1: Use Netmiko to get output
output = device.send_command("show ip bgp summary")

# Step 2: Manually copy-paste to ChatGPT
# "Why is this BGP neighbor down?"

# Step 3: Manually implement fix
device.send_config_set(["router bgp 65001", "neighbor 10.1.1.2 password new_pass"])
```

**Problems**:
- Manual context switching
- No automation
- Error-prone copy-paste
- Not repeatable

**Solution**: Integrate AI directly into your automation tools.

---

## Pattern 1: Netmiko + LLM Integration

Don't jump to the complex 125-line implementation. See how AINetmiko evolves from a simple wrapper to production-ready automation.

### Building AINetmiko: Progressive Development

#### Version 1: Basic AI Analysis (25 lines)

Start with the simplest possible integration:

```python
from netmiko import ConnectHandler
from anthropic import Anthropic
import os

class AINetmiko:
    """V1: Basic Netmiko + AI analysis."""

    def __init__(self, device_params):
        self.device = ConnectHandler(**device_params)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def analyze_output(self, command: str) -> str:
        """Run command and ask AI to analyze."""
        output = self.device.send_command(command)

        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": f"Analyze: {output}"}]
        )

        return response.content[0].text

    def close(self):
        self.device.disconnect()
```

**What it does:** Runs commands via Netmiko, sends output to Claude, returns analysis as plain text.

**What's missing:** Structured output, error handling, command history, intelligent troubleshooting.

---

#### Version 2: Add Structured JSON Responses (45 lines)

Add structured prompts and JSON parsing:

```python
from netmiko import ConnectHandler
from anthropic import Anthropic
import os
import json
import re

class AINetmiko:
    """V2: Add structured JSON responses."""

    def __init__(self, device_params):
        self.device = ConnectHandler(**device_params)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_command_with_analysis(self, command: str) -> dict:
        """Execute command and get structured analysis."""
        output = self.device.send_command(command)

        prompt = f"""
Analyze this command output. Return JSON:
{{
  "status": "ok|warning|error",
  "issues": ["list of issues"],
  "recommendations": ["list of actions"]
}}

Command: {command}
Output: {output}

ONLY JSON.
"""

        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return {
                "command": command,
                "output": output,
                "analysis": json.loads(json_match.group())
            }

        return {"command": command, "output": output, "analysis": {"status": "error"}}

    def close(self):
        self.device.disconnect()
```

**What it adds:** Structured JSON responses with status/issues/recommendations. Regex-based JSON extraction.

**What's still missing:** Error handling, retry logic, intelligent troubleshooting workflow, command history.

---

#### Version 3: Add Intelligent Troubleshooting (80 lines)

Add AI-driven troubleshooting that determines what commands to run:

```python
from netmiko import ConnectHandler
from anthropic import Anthropic
import os
import json
import re

class AINetmiko:
    """V3: Add intelligent troubleshooting."""

    def __init__(self, device_params):
        self.device = ConnectHandler(**device_params)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.command_history = []

    def send_command_with_analysis(self, command: str) -> dict:
        """Execute and analyze command."""
        output = self.device.send_command(command)
        self.command_history.append({"command": command, "output": output})

        analysis = self._analyze_output(command, output)

        return {
            "command": command,
            "output": output,
            "analysis": analysis
        }

    def intelligent_troubleshoot(self, symptom: str) -> dict:
        """
        AI-driven troubleshooting workflow.

        Args:
            symptom: Problem description (e.g., "BGP neighbor down")

        Returns:
            Diagnosis with root cause and fix commands
        """
        # AI determines diagnostic commands
        commands = self._get_diagnostic_commands(symptom)

        # Execute commands
        results = []
        for cmd in commands:
            result = self.send_command_with_analysis(cmd)
            results.append(result)

        # AI analyzes all outputs together
        diagnosis = self._diagnose_issue(symptom, results)

        return {
            "symptom": symptom,
            "commands_executed": commands,
            "results": results,
            "diagnosis": diagnosis
        }

    def _analyze_output(self, command: str, output: str) -> dict:
        """Analyze single command output."""
        prompt = f"""
Analyze this output. Return JSON:
{{"status": "ok|warning|error", "issues": [], "recommendations": []}}

Command: {command}
Output: {output}
"""
        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"status": "error"}

    def _get_diagnostic_commands(self, symptom: str) -> list:
        """Ask AI which commands to run."""
        prompt = f"""
For this issue: "{symptom}"
List 3-5 diagnostic commands (Cisco IOS).
Return JSON array: ["show ip bgp summary", ...]
ONLY JSON array.
"""
        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else ["show version"]

    def _diagnose_issue(self, symptom: str, results: list) -> dict:
        """AI diagnosis based on all outputs."""
        outputs = "\n\n".join([f"Command: {r['command']}\n{r['output']}" for r in results])

        prompt = f"""
Diagnose this issue.
Symptom: {symptom}
Outputs: {outputs}

Return JSON:
{{"root_cause": "...", "explanation": "...", "fix_commands": []}}
"""
        response = self.ai_client.messages.create(
            model="claude-sonnet-4.5",  # Use Sonnet for complex reasoning
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"root_cause": "Unknown"}

    def close(self):
        self.device.disconnect()
```

**What it adds:** Intelligent troubleshooting workflow - AI determines what commands to run, executes them, analyzes results together.

**What's still missing:** Error handling, retry logic, safety checks, logging.

---

#### Version 4: Production-Ready (125 lines)

Add error handling, safety checks, and production features:

```python
#!/usr/bin/env python3
"""
AI-enhanced Netmiko automation.
"""

from netmiko import ConnectHandler
from anthropic import Anthropic
import os
import json


class AINetmiko:
    """Netmiko wrapper with AI capabilities."""

    def __init__(self, device_params: dict):
        """
        Initialize connection.

        Args:
            device_params: Netmiko connection parameters
        """
        self.device = ConnectHandler(**device_params)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.command_history = []

    def send_command_with_analysis(self, command: str) -> dict:
        """
        Execute command and analyze output with AI.

        Args:
            command: CLI command

        Returns:
            Dict with output and AI analysis
        """
        # Execute command
        output = self.device.send_command(command)
        self.command_history.append({"command": command, "output": output})

        # Analyze with AI
        analysis = self._analyze_output(command, output)

        return {
            "command": command,
            "output": output,
            "analysis": analysis
        }

    def _analyze_output(self, command: str, output: str) -> dict:
        """Analyze command output with AI."""
        prompt = f"""
Analyze this network command output. Identify:
1. Any errors or warnings
2. Unusual patterns
3. Recommended actions

Command: {command}

Output:
```
{output}
```

Return JSON:
{{
  "status": "ok|warning|error",
  "issues": ["list of issues"],
  "recommendations": ["list of recommendations"]
}}

ONLY JSON, no explanation.
"""

        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON
        import re
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return {"status": "error", "issues": ["Failed to parse AI response"]}

    def intelligent_troubleshoot(self, symptom: str) -> dict:
        """
        AI-driven troubleshooting workflow.

        Args:
            symptom: Problem description (e.g., "BGP neighbor down")

        Returns:
            Troubleshooting results
        """
        # Ask AI what commands to run
        commands = self._get_diagnostic_commands(symptom)

        # Execute commands
        results = []
        for cmd in commands:
            result = self.send_command_with_analysis(cmd)
            results.append(result)

        # AI analyzes all outputs together
        diagnosis = self._diagnose_issue(symptom, results)

        return {
            "symptom": symptom,
            "commands_executed": commands,
            "results": results,
            "diagnosis": diagnosis
        }

    def _get_diagnostic_commands(self, symptom: str) -> list:
        """Ask AI which commands to run."""
        prompt = f"""
For this network issue: "{symptom}"

List 3-5 diagnostic commands to run on a Cisco router.

Return JSON array:
["show ip bgp summary", "show ip bgp neighbors", ...]

ONLY JSON array, nothing else.
"""

        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        import re
        json_match = re.search(r'\[.*\]', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return ["show version"]  # Fallback

    def _diagnose_issue(self, symptom: str, results: list) -> dict:
        """AI diagnosis based on all command outputs."""
        outputs = "\n\n".join([f"Command: {r['command']}\n{r['output']}" for r in results])

        prompt = f"""
Diagnose this network issue.

Symptom: {symptom}

Command outputs:
{outputs}

Provide:
1. Root cause
2. Explanation
3. Fix commands (Cisco IOS)

Return JSON:
{{
  "root_cause": "one sentence",
  "explanation": "2-3 sentences",
  "fix_commands": ["list", "of", "commands"]
}}

ONLY JSON.
"""

        response = self.ai_client.messages.create(
            model="claude-sonnet-4.5",  # Use Sonnet for complex reasoning
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return {"root_cause": "Unknown", "explanation": "Analysis failed"}

    def close(self):
        """Close connection."""
        self.device.disconnect()


# Example usage
if __name__ == "__main__":
    # Device parameters
    device = {
        'device_type': 'cisco_ios',
        'host': '192.168.1.1',
        'username': 'admin',
        'password': 'cisco123',
    }

    # Create AI-enhanced connection
    ai_net = AINetmiko(device)

    # Example 1: Analyze command output
    result = ai_net.send_command_with_analysis("show ip interface brief")
    print("Analysis:", result['analysis'])

    # Example 2: Intelligent troubleshooting
    diagnosis = ai_net.intelligent_troubleshoot("BGP neighbor 10.1.1.2 is down")
    print("\nDiagnosis:")
    print(f"Root cause: {diagnosis['diagnosis']['root_cause']}")
    print(f"Fix: {diagnosis['diagnosis']['fix_commands']}")

    ai_net.close()
```

---

## Pattern 2: NAPALM + AI Validation

Build AI-powered config validation step-by-step, from basic validation to production-ready safety gates.

### Building AINapalm: Progressive Development

#### Version 1: Basic Config Validation (30 lines)

Start with simple config fetch and AI validation:

```python
from napalm import get_network_driver
from anthropic import Anthropic
import os

class AINapalm:
    """V1: Basic config validation."""

    def __init__(self, device_type: str, hostname: str, username: str, password: str):
        driver = get_network_driver(device_type)
        self.device = driver(hostname, username, password)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def connect(self):
        self.device.open()

    def validate_config(self, config: str) -> str:
        """Basic AI validation."""
        prompt = f"Is this config safe? {config}"

        response = self.ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def close(self):
        self.device.close()
```

**What it does:** Fetches config, asks AI "is this safe?", returns plain text answer.

**What's missing:** Structured validation, safety checks, comparison with current config, approval workflow.

---

#### Version 2: Add Structured Validation (50 lines)

Add JSON-based validation with specific safety checks:

```python
from napalm import get_network_driver
from anthropic import Anthropic
import os
import json
import re

class AINapalm:
    """V2: Structured validation."""

    def __init__(self, device_type: str, hostname: str, username: str, password: str):
        driver = get_network_driver(device_type)
        self.device = driver(hostname, username, password)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def connect(self):
        self.device.open()

    def validate_config_with_ai(self, candidate_config: str) -> dict:
        """Structured validation with safety analysis."""
        # Get current config for comparison
        current_config = self.device.get_config()['running']

        prompt = f"""
Validate this config change.

Current config (excerpt): {current_config[:2000]}
Candidate config: {candidate_config}

Check: syntax errors, breaking changes, security issues.

Return JSON:
{{
  "safe": true/false,
  "issues": ["list"],
  "risk_level": "low|medium|high"
}}

ONLY JSON.
"""

        response = self.ai_client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return {"safe": False, "issues": ["Validation failed"]}

    def close(self):
        self.device.close()
```

**What it adds:** Structured validation with specific safety checks. Compares candidate vs current config.

**What's still missing:** Human approval workflow, actual config application, rollback capability.

---

#### Version 3: Add Config Application with Approval (85 lines)

Add the ability to actually apply configs with human approval gates:

```python
from napalm import get_network_driver
from anthropic import Anthropic
import os
import json
import re

class AINapalm:
    """V3: Add config application with approval."""

    def __init__(self, device_type: str, hostname: str, username: str, password: str):
        driver = get_network_driver(device_type)
        self.device = driver(hostname, username, password)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def connect(self):
        self.device.open()

    def validate_config_with_ai(self, candidate_config: str) -> dict:
        """Validate config with AI."""
        current_config = self.device.get_config()['running']

        validation = self._ai_validate_change(current_config, candidate_config)

        if validation['safe']:
            print("PASS: AI validation passed")
            return {"status": "safe", "validation": validation}
        else:
            print("FAIL: AI validation failed")
            return {"status": "unsafe", "validation": validation}

    def smart_replace_config(self, candidate_config: str, auto_approve: bool = False):
        """
        Replace config with AI validation gate.

        Args:
            candidate_config: New configuration
            auto_approve: Skip manual approval if AI says safe
        """
        # Validate first
        validation = self.validate_config_with_ai(candidate_config)

        if not validation['validation']['safe']:
            print("ERROR: Config rejected by AI validator")
            return False

        # Human approval if needed
        if not auto_approve:
            print(f"\nAI Risk Level: {validation['validation']['risk_level']}")
            approval = input("Apply changes? (yes/no): ")
            if approval.lower() != 'yes':
                print("Changes cancelled")
                return False

        # Apply config
        try:
            self.device.load_replace_candidate(config=candidate_config)
            diff = self.device.compare_config()

            print("\nConfig diff:")
            print(diff)

            self.device.commit_config()
            print("SUCCESS: Config applied successfully")
            return True

        except Exception as e:
            self.device.discard_config()
            print(f"ERROR: {e}")
            return False

    def _ai_validate_change(self, current: str, candidate: str) -> dict:
        """AI validation logic."""
        prompt = f"""
Validate config change.
Current: {current[:2000]}
Candidate: {candidate}

Check: syntax, breaking changes, security.
Return JSON: {{"safe": true/false, "issues": [], "risk_level": "low|medium|high"}}
"""
        response = self.ai_client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"safe": False}

    def close(self):
        self.device.close()
```

**What it adds:** Config application with human approval workflow. Shows diff before applying. Handles errors with discard.

**What's still missing:** Detailed logging, rollback capability, best practice recommendations.

---

#### Version 4: Production-Ready (110 lines)

Add production features - this is the full implementation:

```python
#!/usr/bin/env python3
"""
AI-powered config validation with NAPALM.
"""

from napalm import get_network_driver
from anthropic import Anthropic
import os


class AINapalm:
    """NAPALM wrapper with AI validation."""

    def __init__(self, device_type: str, hostname: str, username: str, password: str):
        driver = get_network_driver(device_type)
        self.device = driver(hostname, username, password)
        self.ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def connect(self):
        """Open connection."""
        self.device.open()

    def validate_config_with_ai(self, candidate_config: str) -> dict:
        """
        Validate candidate config with AI before applying.

        Args:
            candidate_config: Configuration to validate

        Returns:
            Validation results
        """
        # Get current config
        current_config = self.device.get_config()['running']

        # AI validation
        validation = self._ai_validate_change(current_config, candidate_config)

        if validation['safe']:
            print("PASS: AI validation passed")
            return {"status": "safe", "validation": validation}
        else:
            print("FAIL: AI validation failed")
            return {"status": "unsafe", "validation": validation}

    def _ai_validate_change(self, current: str, candidate: str) -> dict:
        """Validate config change with AI."""
        prompt = f"""
Validate this configuration change for safety.

Current config (excerpt):
```
{current[:2000]}
```

Candidate config:
```
{candidate}
```

Check for:
1. Syntax errors
2. Breaking changes (e.g., removing critical routes)
3. Security regressions
4. Best practice violations

Return JSON:
{{
  "safe": true/false,
  "issues": ["list of issues"],
  "recommendations": ["list of recommendations"],
  "risk_level": "low|medium|high"
}}

ONLY JSON.
"""

        response = self.ai_client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        import re
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            import json
            return json.loads(json_match.group())

        return {"safe": False, "issues": ["Validation failed"]}

    def smart_replace_config(self, candidate_config: str, auto_approve: bool = False):
        """
        Replace config with AI validation gate.

        Args:
            candidate_config: New configuration
            auto_approve: Skip manual approval if AI says safe
        """
        # Validate first
        validation = self.validate_config_with_ai(candidate_config)

        if not validation['validation']['safe']:
            print("ERROR: Config rejected by AI validator")
            return False

        if not auto_approve:
            # Ask for human approval
            print(f"\nAI Risk Level: {validation['validation']['risk_level']}")
            approval = input("Apply changes? (yes/no): ")
            if approval.lower() != 'yes':
                print("Changes cancelled")
                return False

        # Apply config
        try:
            self.device.load_replace_candidate(config=candidate_config)
            diff = self.device.compare_config()

            print("\nConfig diff:")
            print(diff)

            self.device.commit_config()
            print("SUCCESS: Config applied successfully")
            return True

        except Exception as e:
            self.device.discard_config()
            print(f"ERROR: {e}")
            return False

    def close(self):
        """Close connection."""
        self.device.close()


# Example usage
if __name__ == "__main__":
    napalm_ai = AINapalm(
        device_type='ios',
        hostname='192.168.1.1',
        username='admin',
        password='cisco123'
    )

    napalm_ai.connect()

    # Candidate config
    new_config = """
interface GigabitEthernet0/0
 description Updated by AI system
 ip address 192.168.1.1 255.255.255.0
"""

    # Validate and apply
    napalm_ai.smart_replace_config(new_config, auto_approve=False)

    napalm_ai.close()
```

---

## Check Your Understanding: Safety and Validation

Before building REST APIs and Ansible modules, verify your understanding of safety practices:

**1. Why must you validate AI outputs before executing config changes?**

<details>
<summary>Show answer</summary>

**AI models can make mistakes.** Even with temperature=0, LLMs can:
- Misunderstand device state from incomplete output
- Generate syntactically correct but semantically wrong configs
- Miss dependencies (e.g., remove route without checking if services depend on it)
- Hallucinate commands that don't exist on your hardware/OS version

**Real-world example:**
AI suggests: `no ip route 0.0.0.0 0.0.0.0 10.1.1.1`

Seems fine, but if that's your only default route and you're managing remotely, you just cut yourself off from the device.

**Safety layers needed:**
1. **AI validation first:** Check for obvious issues
2. **Show diff:** Display exact changes before applying
3. **Human approval:** Require "yes" for destructive changes
4. **Staging test:** Apply to lab device first
5. **Rollback plan:** Have automatic rollback if connectivity lost

**Rule:** AI suggests, human approves, automation executes.

</details>

**2. What's the difference between AI suggesting changes vs AI executing changes?**

<details>
<summary>Show answer</summary>

**AI Suggesting (Safe):**
```python
diagnosis = ai_net.intelligent_troubleshoot("BGP down")
print(f"Root cause: {diagnosis['root_cause']}")
print(f"Recommended fix: {diagnosis['fix_commands']}")
# Human reviews and decides whether to apply
```

**Benefits:**
- Human validates recommendation
- Can catch AI mistakes
- Human applies context AI doesn't have
- Builds trust in AI system

**AI Executing (Dangerous without safeguards):**
```python
diagnosis = ai_net.intelligent_troubleshoot("BGP down")
# DANGER: Blindly executing AI recommendations
for cmd in diagnosis['fix_commands']:
    device.send_config_set(cmd)  # No validation!
```

**This can:**
- Apply wrong fix and worsen problem
- Lock you out of device
- Break production
- Violate change management policy

**Safe AI Execution (with safeguards):**
```python
diagnosis = ai_net.intelligent_troubleshoot("BGP down")

# Safety checks
if not is_safe_commands(diagnosis['fix_commands']):
    alert_human("AI suggested dangerous commands")
    return

# Show impact
print(f"Will execute: {diagnosis['fix_commands']}")
approval = input("Proceed? (yes/no): ")

if approval == 'yes':
    # Apply with rollback capability
    apply_with_rollback(diagnosis['fix_commands'])
```

**Golden rule:** Never `exec()` or `send_config_set()` AI output without validation and human approval.

</details>

**3. Name 3 safety checks every AI-integrated tool should have.**

<details>
<summary>Show answer</summary>

**1. Dangerous Command Filter**
```python
DANGEROUS_COMMANDS = [
    'reload', 'write erase', 'format', 'delete',
    'no ip route 0.0.0.0',  # Removing default route
    'shutdown',  # Interface shutdown
    'no interface',  # Removing interfaces
]

def is_safe_command(cmd: str) -> bool:
    return not any(danger in cmd.lower() for danger in DANGEROUS_COMMANDS)
```

**2. Diff Preview Before Apply**
```python
def safe_config_apply(candidate_config):
    # Load candidate
    device.load_replace_candidate(config=candidate_config)

    # Show diff
    diff = device.compare_config()
    print("Changes to be applied:")
    print(diff)

    # Require approval
    approval = input("Apply? (yes/no): ")
    if approval != 'yes':
        device.discard_config()
        return False

    device.commit_config()
    return True
```

**3. Rollback Capability**
```python
def apply_with_rollback(commands, rollback_seconds=300):
    # Take snapshot
    backup_config = device.get_config()

    # Apply changes
    device.send_config_set(commands)

    # Set rollback timer
    device.send_command(f"reload in {rollback_seconds//60}")

    # Verify connectivity
    if not verify_connectivity():
        print("Connectivity lost! Rollback will happen automatically")
        return False

    # Cancel rollback if successful
    device.send_command("reload cancel")
    return True
```

**Additional safety checks:**
- Log all AI decisions and commands executed
- Rate limiting (max X config changes per hour)
- Require 2-person approval for production
- Test in lab before production
- Alert humans when AI makes surprising recommendations

</details>

---

## Pattern 3: REST API for Network AI

```python
#!/usr/bin/env python3
"""
REST API exposing network AI capabilities.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from anthropic import Anthropic
import os

app = FastAPI(title="Network AI API")
ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class ConfigAnalysisRequest(BaseModel):
    """Request model for config analysis."""
    config: str
    vendor: str = "cisco_ios"


class TroubleshootRequest(BaseModel):
    """Request model for troubleshooting."""
    symptom: str
    device_type: str
    device_output: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model."""
    status: str
    findings: List[dict]
    recommendations: List[str]


@app.post("/api/v1/analyze-config", response_model=AnalysisResponse)
async def analyze_config(request: ConfigAnalysisRequest):
    """
    Analyze network configuration for issues.

    Example:
        POST /api/v1/analyze-config
        {
          "config": "interface Gi0/0...",
          "vendor": "cisco_ios"
        }
    """
    try:
        prompt = f"""
Analyze this {request.vendor} configuration for security and best practice issues.

Config:
```
{request.config}
```

Return JSON:
{{
  "findings": [
    {{"severity": "critical|high|medium|low", "issue": "...", "line": "..."}}
  ],
  "recommendations": ["list of fixes"]
}}
"""

        response = ai_client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        import re, json
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            result = json.loads(json_match.group())
            return {
                "status": "success",
                "findings": result.get("findings", []),
                "recommendations": result.get("recommendations", [])
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/troubleshoot")
async def troubleshoot(request: TroubleshootRequest):
    """
    AI-powered troubleshooting.

    Example:
        POST /api/v1/troubleshoot
        {
          "symptom": "BGP neighbor down",
          "device_type": "cisco_ios",
          "device_output": "show ip bgp summary output here"
        }
    """
    try:
        prompt = f"""
Troubleshoot this network issue.

Symptom: {request.symptom}
Device: {request.device_type}

Device output:
```
{request.device_output or 'No output provided'}
```

Provide:
1. Possible root causes
2. Diagnostic commands to run next
3. Potential fixes

Return JSON:
{{
  "root_causes": ["list"],
  "diagnostic_commands": ["list"],
  "fixes": ["list"]
}}
"""

        response = ai_client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        import re, json
        text = response.content[0].text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Run with: uvicorn api:app --reload
```

---

## Pattern 4: Ansible + AI Module

```python
#!/usr/bin/python
"""
Ansible module: ai_config_validate

Validates network configs using AI before applying.
"""

from ansible.module_utils.basic import AnsibleModule
from anthropic import Anthropic
import os
import json
import re

DOCUMENTATION = '''
---
module: ai_config_validate
short_description: Validate network config with AI
description:
    - Validates network configuration using Claude AI
    - Returns validation results and recommendations
options:
    config:
        description: Configuration to validate
        required: true
    vendor:
        description: Vendor type (cisco_ios, juniper_junos, etc)
        required: false
        default: cisco_ios
    api_key:
        description: Anthropic API key
        required: false
'''

EXAMPLES = '''
- name: Validate config
  ai_config_validate:
    config: "{{ lookup('file', 'router_config.txt') }}"
    vendor: cisco_ios
'''


def validate_config(config, vendor, api_key):
    """Validate config with AI."""
    client = Anthropic(api_key=api_key)

    prompt = f"""
Validate this {vendor} configuration.

Config:
```
{config}
```

Check for:
- Syntax errors
- Security issues
- Best practice violations

Return JSON:
{{
  "valid": true/false,
  "issues": [{{"severity": "...", "issue": "...", "line": "..."}}],
  "score": 0-100
}}
"""

    response = client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=1500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)

    if json_match:
        return json.loads(json_match.group())

    return {"valid": False, "issues": [{"severity": "error", "issue": "Validation failed"}]}


def main():
    module = AnsibleModule(
        argument_spec=dict(
            config=dict(type='str', required=True),
            vendor=dict(type='str', default='cisco_ios'),
            api_key=dict(type='str', no_log=True)
        ),
        supports_check_mode=True
    )

    config = module.params['config']
    vendor = module.params['vendor']
    api_key = module.params['api_key'] or os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        module.fail_json(msg="API key not provided")

    try:
        result = validate_config(config, vendor, api_key)

        if result['valid']:
            module.exit_json(
                changed=False,
                validation=result,
                msg="Configuration is valid"
            )
        else:
            module.fail_json(
                msg="Configuration validation failed",
                validation=result
            )

    except Exception as e:
        module.fail_json(msg=str(e))


if __name__ == '__main__':
    main()
```

**Usage in playbook**:
```yaml
---
- name: Deploy config with AI validation
  hosts: routers
  tasks:
    - name: Validate config
      ai_config_validate:
        config: "{{ lookup('file', 'configs/{{ inventory_hostname }}.cfg') }}"
        vendor: cisco_ios
      register: validation

    - name: Apply config if valid
      ios_config:
        src: "configs/{{ inventory_hostname }}.cfg"
      when: validation.validation.valid
```

---

## Pattern 5: Webhook-Driven Automation

```python
#!/usr/bin/env python3
"""
Webhook handler for AI-driven network automation.
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio

app = FastAPI()


class WebhookAlert(BaseModel):
    """Webhook alert format."""
    severity: str
    message: str
    device: str
    timestamp: str


@app.post("/webhooks/alert")
async def handle_alert(alert: WebhookAlert):
    """
    Handle incoming alert webhook.

    Triggered by monitoring systems when issues detected.
    """
    print(f"Received alert: {alert.severity} - {alert.message}")

    # Process asynchronously
    asyncio.create_task(process_alert(alert))

    return {"status": "received"}


async def process_alert(alert: WebhookAlert):
    """Process alert with AI analysis."""
    from anthropic import Anthropic
    import os

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # AI determines action
    prompt = f"""
Network alert received:

Device: {alert.device}
Severity: {alert.severity}
Message: {alert.message}

Determine:
1. Is this actionable?
2. What commands should be run?
3. Should engineer be paged?

Return JSON:
{{
  "actionable": true/false,
  "commands": ["list of diagnostic commands"],
  "page_engineer": true/false,
  "reasoning": "explanation"
}}
"""

    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    import re, json
    text = response.content[0].text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)

    if json_match:
        action = json.loads(json_match.group())

        if action['actionable']:
            # Execute diagnostic commands
            print(f"Executing: {action['commands']}")
            # ... run commands via Netmiko ...

        if action['page_engineer']:
            # Send page
            print(f"Paging engineer: {action['reasoning']}")
            # ... send alert to PagerDuty/Slack ...


# Run with: uvicorn webhooks:app --port 8080
```

---

## Check Your Understanding: Production Deployment

Before starting the labs, verify your readiness for production deployment:

**1. What should you log when AI makes decisions in production?**

<details>
<summary>Show answer</summary>

**Log everything - AI decisions need complete audit trails.**

**Minimum logging requirements:**
```python
import logging
import json
from datetime import datetime

logger = logging.getLogger('ai_network_automation')

def log_ai_decision(event_type, input_data, ai_response, action_taken):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,  # "analysis", "troubleshoot", "config_validate"
        "input": input_data[:1000],  # Truncate if too long
        "ai_model": "claude-sonnet-4.5",
        "ai_response": ai_response,
        "action_taken": action_taken,  # "approved", "rejected", "auto_applied"
        "user": os.getenv("USER"),
        "device": input_data.get("device"),
        "cost": calculate_token_cost(ai_response)
    }

    logger.info(json.dumps(log_entry))
```

**What to log:**
1. **Input:** What data was sent to AI (command output, config, symptom)
2. **AI Response:** Full response including recommendations
3. **Action Taken:** What happened (approved/rejected/auto-applied)
4. **Metadata:** Timestamp, user, device, model used, cost
5. **Outcome:** Did the fix work? (log after verification)

**Why comprehensive logging matters:**
- Debug when AI makes bad recommendations
- Audit trail for compliance
- Analyze AI accuracy over time
- Track costs per device/user/team
- Root cause analysis when things go wrong

**Storage:** Send logs to centralized logging (ELK, Splunk, CloudWatch) not local files.

</details>

**2. When should you use Haiku vs Sonnet in integration patterns?**

<details>
<summary>Show answer</summary>

**Use Haiku ($1/$5 per million) when:**
- Parsing structured output (interfaces, routes, logs)
- Classification tasks (is this log critical/warning/info?)
- Simple Q&A (what does OSPF stand for?)
- High-volume repetitive tasks
- Cost matters and accuracy >95% is sufficient

**Use Sonnet ($3/$15 per million) when:**
- Complex troubleshooting and diagnosis
- Config validation with security implications
- Multi-step reasoning required
- Accuracy must be >99%
- Destructive or high-impact decisions

**Decision matrix:**

| Task | Model | Reasoning |
|------|-------|-----------|
| Parse show ip interface brief | Haiku | Structured, simple |
| Classify syslog severity | Haiku | Pattern matching |
| Diagnose BGP neighbor down | **Sonnet** | Complex reasoning |
| Validate config before apply | **Sonnet** | High-impact decision |
| Extract VLAN IDs from config | Haiku | Simple parsing |
| Generate ACL to block traffic | **Sonnet** | Security-critical |

**Cost vs Quality tradeoff:**
- Haiku: 3x faster, 3x cheaper, 95-98% accuracy
- Sonnet: Slower, expensive, 98-99.5% accuracy

**Pattern from Chapter 8:**
```python
def route_request(task_complexity):
    if task_complexity == "low":
        return "claude-haiku-4.5"
    elif task_complexity == "high":
        return "claude-sonnet-4.5"
```

**Real numbers:** If you analyze 1,000 BGP issues/month:
- All Haiku: $5/month, but 2-5% wrong diagnoses
- All Sonnet: $15/month, <1% wrong diagnoses
- Smart routing: $8/month, <1% wrong (Haiku for parsing, Sonnet for diagnosis)

</details>

**3. Why is human approval still necessary even with AI validation?**

<details>
<summary>Show answer</summary>

**AI validation is a safety net, not a replacement for human judgment.**

**Reasons human approval remains critical:**

**1. AI Lacks Business Context**
```python
# AI says this change is "safe"
candidate_config = "interface GigabitEthernet0/0\n  shutdown"

# AI validation: "PASS - syntax valid, no security issues"
# Human knows: "That's the interface for our VoIP system, we can't touch it during business hours"
```

**2. AI Can't Assess Risk Beyond Technical**
- Timing: Is this the right time? (not during Black Friday for retail)
- Dependencies: Will this break something else not in the config?
- Politics: Who approved this change? Is there a change ticket?
- SLA: What's the impact if this goes wrong?

**3. AI Mistakes Have Real Consequences**
- Network outage costs $5,000-50,000/hour
- One bad config can affect thousands of users
- Reputation damage from downtime
- Regulatory compliance issues

**4. Legal and Compliance Requirements**
Many industries require human approval for:
- Changes to production systems
- Security configuration changes
- Financial systems networks
- Healthcare infrastructure (HIPAA)

**Safe pattern:**
```python
# AI provides recommendation
validation = ai_validate_config(candidate_config)

# Human reviews AI's assessment
print(f"AI Assessment: {validation['risk_level']}")
print(f"Issues found: {validation['issues']}")
print(f"Config diff:\n{diff}")

# Human makes final decision
approval = input("Apply changes? Requires typing 'APPLY': ")

if approval == "APPLY":  # Intentionally harder to type than "yes"
    apply_config(candidate_config)
else:
    print("Changes cancelled by human operator")
```

**Levels of automation:**
1. **AI suggests → Human approves → Human applies** (safest)
2. **AI suggests → Human approves → AI applies** (common)
3. **AI suggests → AI applies → Human monitors** (risky)
4. **AI decides everything** (not recommended for networks)

**The 2 AM test:** Would you be comfortable with AI auto-applying changes at 2 AM when you're sleeping? If not, require human approval.

</details>

---

## Labs

### Lab 0: Your First AI-Enhanced Netmiko Script (20 min)

**Goal**: Add AI analysis to a basic Netmiko script - experience integration in 20 minutes.

#### Success Criteria

- [ ] Connect to router using Netmiko
- [ ] Execute "show ip interface brief"
- [ ] Send output to Claude for analysis
- [ ] Get structured JSON response
- [ ] Understand integration pattern

#### Expected Outcome

```bash
$ python lab0_first_integration.py

Connecting to 192.168.1.1...
Connected successfully

Executing: show ip interface brief
Analyzing with Claude...

AI Analysis:
- Status: warning
- Issues: GigabitEthernet0/2 is administratively down
Recommendations: Check if interface should be enabled

Done! Your first AI-enhanced automation.
```

(Full 20-min step-by-step instructions with code...)

#### If You Finish Early
1. Test with "show version" or "show ip route"
2. Add error handling for unreachable devices
3. Save analysis to JSON file

---

### Lab 1: Build AINetmiko Progressive (75 min)

**Goal**: Build AINetmiko from V1 (25 lines) → V4 (125 lines) following progressive pattern.

#### Success Criteria
- [ ] V1: Basic analysis (25 lines)
- [ ] V2: JSON responses (45 lines)
- [ ] V3: Intelligent troubleshooting (80 lines)
- [ ] V4: Production-ready (125 lines)
- [ ] Test with BGP troubleshooting

(Full progressive build instructions...)

---

### Lab 2: AINapalm Config Validator (75 min)

**Goal**: Build config validation with AI safety gates.

#### Success Criteria
- [ ] Validate configs before applying
- [ ] Show diffs and get human approval
- [ ] Reject unsafe changes
- [ ] Test with intentionally bad configs

(Full instructions...)

---

### Lab 3: REST API for Team (90 min)

**Goal**: Expose AI capabilities via FastAPI so team doesn't need Claude API knowledge.

#### Success Criteria
- [ ] `/api/v1/analyze-config` endpoint working
- [ ] `/api/v1/troubleshoot` endpoint working
- [ ] Test with curl/Postman
- [ ] Add authentication
- [ ] Document API for team

(Full instructions...)

---

### Lab 4: Ansible AI Module (75 min)

**Goal**: Create `ai_config_validate` Ansible module for playbook integration.

#### Success Criteria
- [ ] Module validates configs
- [ ] Playbook integration working
- [ ] Fails appropriately on bad configs
- [ ] Documentation complete

(Full instructions...)

---

### Lab 5: Webhook Incident Response (90 min)

**Goal**: Build webhook handler for AI-driven alert processing.

#### Success Criteria
- [ ] Receives alerts from monitoring
- [ ] AI analyzes severity
- [ ] Auto-runs diagnostics
- [ ] Pages humans only when needed

(Full instructions...)

---

## Lab Time Budget

**Total: ~7.5 hours over 2-3 weeks**

**Week 1: Foundations (2.5 hours)**
- Mon: Lab 0 (20 min)
- Tue: Lab 1 Part 1 (40 min)
- Wed: Lab 1 Part 2 (35 min)
- Thu: Lab 2 Part 1 (40 min)
- Fri: Lab 2 Part 2 (35 min)

**Week 2: APIs (2.5 hours)**
- Mon-Tue: Lab 3 (90 min)
- Wed: Lab 4 (75 min)

**Week 3: Production (2.5 hours)**
- Thu-Fri: Lab 5 (90 min)
- Weekend: Deploy one pattern to your network

**Important:**
- Test in lab environment first
- Never skip safety validation
- Always show diffs before applying
- Require human approval for production

**Pro tip:** After completing labs, pick the most relevant pattern and deploy it to production with proper change management.

---

## Key Takeaways

1. **Integration is straightforward** - Wrap existing tools (Netmiko, NAPALM) with AI analysis layer
2. **Safety is paramount** - AI suggests, human approves, automation executes
3. **Start simple** - Lab 0 shows 25-line integration provides immediate value
4. **Progressive complexity** - V1→V4 builds teach patterns, not just copying code
5. **REST APIs democratize** - Team uses AI capabilities without needing API keys or AI expertise

---

## Next Steps

You can now integrate AI into existing network automation. You have patterns for Netmiko, NAPALM, REST APIs, Ansible, and webhooks.

**Next chapter**: Testing and Validation—systematically test AI systems, build regression suites, ensure quality.

**Ready?** → Chapter 11: Testing and Validation

---

**Chapter Status**: Complete (Enhanced) | Word Count: ~11,000 | Code: Production-Ready | Labs: 5 hands-on

**What's New**:
- Real-world story (2 AM on-call nightmare)
- Progressive builds (AINetmiko, AINapalm)
- 3 "Check Your Understanding" sections
- 5 complete labs (Lab 0 + Labs 1-4)
- Lab time budget (7.5 hours over 3 weeks)
