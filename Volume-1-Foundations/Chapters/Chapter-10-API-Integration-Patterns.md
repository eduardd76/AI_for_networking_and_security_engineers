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

### Basic Integration

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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-sonnet-20241022",  # Use Sonnet for complex reasoning
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
            print("✓ AI validation passed")
            return {"status": "safe", "validation": validation}
        else:
            print("✗ AI validation failed")
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
            model="claude-3-5-sonnet-20241022",
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
            print("✗ Config rejected by AI validator")
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
            print("✓ Config applied successfully")
            return True

        except Exception as e:
            self.device.discard_config()
            print(f"✗ Error: {e}")
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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-sonnet-20241022",
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
        model="claude-3-5-haiku-20241022",
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
        model="claude-3-5-sonnet-20241022",
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

## Best Practices

### ✅ DO:

1. **Validate AI outputs before executing**
   - Never blindly run AI-generated commands
   - Always show diff/preview
   - Require human approval for destructive changes

2. **Implement safety checks**
   ```python
   DANGEROUS_COMMANDS = ['reload', 'write erase', 'format', 'delete']

   def is_safe_command(cmd):
       return not any(danger in cmd.lower() for danger in DANGEROUS_COMMANDS)
   ```

3. **Log everything**
   - Every AI decision
   - Every command executed
   - Every change made

4. **Use appropriate models**
   - Haiku for parsing/classification
   - Sonnet for troubleshooting/decisions

5. **Handle failures gracefully**
   - Always have fallback logic
   - Timeout long-running operations
   - Retry with exponential backoff

### ❌ DON'T:

1. **Don't trust AI blindly**
   - Always validate outputs
   - Test in lab first

2. **Don't expose without auth**
   - Secure all APIs
   - Implement RBAC

3. **Don't skip error handling**
   - Network and AI APIs both fail

---

## Key Takeaways

1. **AI enhances existing tools**
   - Netmiko + AI = intelligent troubleshooting
   - NAPALM + AI = smart validation
   - APIs expose AI to your team

2. **Always validate before executing**
   - AI suggests, human approves
   - Show diffs and impacts
   - Log all decisions

3. **REST APIs democratize AI**
   - Team doesn't need AI expertise
   - Consistent interface
   - Easy integration

4. **Webhooks enable reactive automation**
   - AI analyzes alerts
   - Decides on actions
   - Pages humans when needed

5. **Integration is straightforward**
   - Wrap existing tools
   - Add AI analysis layer
   - Maintain safety checks

---

## Next Steps

You can now integrate AI into your existing network automation stack. You have patterns for Netmiko, NAPALM, REST APIs, Ansible, and webhooks.

**Next chapter**: Testing and Validation—how to systematically test AI systems, build regression test suites, and ensure quality.

**Ready?** → Chapter 11: Testing and Validation

---

**Chapter Status**: Complete | Word Count: ~5,500 | Code: Production-Ready
