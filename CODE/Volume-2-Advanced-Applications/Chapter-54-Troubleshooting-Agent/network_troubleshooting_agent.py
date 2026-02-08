"""
Chapter 54: AI-Powered Network Troubleshooting Agent
Autonomous Agent for BGP, Routing, and Performance Issues

This agent uses structured reasoning to diagnose network problems,
suggest diagnostic commands, and provide step-by-step solutions.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

load_dotenv()


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DiagnosticStep(BaseModel):
    command: str = Field(description="CLI command to run")
    purpose: str = Field(description="Why run this command")
    expected_output: str = Field(description="What to look for in output")


class TroubleshootingResult(BaseModel):
    problem_category: str = Field(description="Type of problem: BGP, OSPF, Interface, etc")
    severity: Severity
    root_cause_hypothesis: str = Field(description="Most likely root cause")
    diagnostic_steps: List[DiagnosticStep]
    resolution_steps: List[str]
    estimated_time_to_fix: str = Field(description="Estimated time: 5min, 1hour, etc")


def analyze_bgp_issue():
    """
    Example 1: Troubleshoot BGP neighbor down
    """
    print("=" * 60)
    print("Example 1: BGP Neighbor Down Troubleshooting")
    print("=" * 60)

    symptom = """
    Symptom: BGP neighbor 10.1.1.1 is stuck in ACTIVE state

    show ip bgp summary output:
    Neighbor        V    AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
    10.1.1.1        4 65002       0       0        0    0    0 never    Active

    Additional context:
    - This worked yesterday
    - No recent config changes on our side
    - Other BGP neighbors are up
    """

    prompt = f"""You are a senior network troubleshooting expert. Analyze this BGP issue:

{symptom}

Provide structured troubleshooting guidance including:
1. Problem category
2. Severity level
3. Most likely root cause
4. Diagnostic commands to run (with purpose)
5. Step-by-step resolution
6. Estimated time to fix

Be specific and actionable."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=TroubleshootingResult)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing BGP issue...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"Problem Category: {result.problem_category}")
    print(f"Severity: {result.severity.value.upper()}")
    print(f"\nRoot Cause Hypothesis:")
    print(f"  {result.root_cause_hypothesis}")

    print(f"\nDiagnostic Steps:")
    for i, step in enumerate(result.diagnostic_steps, 1):
        print(f"  {i}. {step.command}")
        print(f"     Purpose: {step.purpose}")
        print(f"     Look for: {step.expected_output}")

    print(f"\nResolution Steps:")
    for i, step in enumerate(result.resolution_steps, 1):
        print(f"  {i}. {step}")

    print(f"\nEstimated Time: {result.estimated_time_to_fix}")

    print("\n" + "=" * 60 + "\n")
    return result


def analyze_interface_flapping():
    """
    Example 2: Troubleshoot interface flapping
    """
    print("=" * 60)
    print("Example 2: Interface Flapping Analysis")
    print("=" * 60)

    logs = """
    Jan 18 10:15:32 CORE-SW-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
    Jan 18 10:15:34 CORE-SW-01 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
    Jan 18 10:15:36 CORE-SW-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up
    Jan 18 10:16:01 CORE-SW-01 %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
    Jan 18 10:16:03 CORE-SW-01 %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up

    show interfaces GigabitEthernet0/1:
    GigabitEthernet0/1 is up, line protocol is up
      Hardware is GigabitEthernet, address is 0011.2233.4455
      MTU 1500 bytes, BW 1000000 Kbit/sec, DLY 10 usec,
      reliability 240/255, txload 1/255, rxload 1/255
      Encapsulation ARPA, loopback not set
      Last input 00:00:05, output 00:00:02, output hang never
      Last clearing of "show interface" counters never
      Input queue: 0/75/0/0 (size/max/drops/flushes); Total output drops: 0
      CRC errors: 1542
      Input errors: 1580
    """

    prompt = f"""Analyze this interface flapping issue:

{logs}

Provide structured troubleshooting analysis."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=TroubleshootingResult)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing interface issues...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"Problem: {result.problem_category}")
    print(f"Severity: {result.severity.value.upper()}")
    print(f"\nDiagnosis: {result.root_cause_hypothesis}")

    print(f"\n🔍 Run these diagnostics:")
    for step in result.diagnostic_steps:
        print(f"  • {step.command}")

    print(f"\n🔧 Fix steps:")
    for i, step in enumerate(result.resolution_steps, 1):
        print(f"  {i}. {step}")

    print("\n" + "=" * 60 + "\n")
    return result


def analyze_performance_issue():
    """
    Example 3: Troubleshoot network performance
    """
    print("=" * 60)
    print("Example 3: Network Performance Analysis")
    print("=" * 60)

    problem = """
    User Report: Slow file transfers to server 192.168.10.50

    Symptoms:
    - Transfer speed: 1-2 Mbps (expected 1000 Mbps)
    - Ping latency: Normal (1-2ms)
    - Packet loss: None observed
    - Issue started this morning
    - Only affects this specific server
    - Other servers in same VLAN are fine

    Network path: User PC -> Access Switch -> Core Switch -> Server
    """

    prompt = f"""Troubleshoot this network performance issue:

{problem}

Provide diagnostic steps focusing on:
- Where bottleneck likely is
- What commands to run
- How to isolate the problem
- Quick fixes to try first"""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    parser = PydanticOutputParser(pydantic_object=TroubleshootingResult)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print("\nAnalyzing performance issue...\n")

    response = llm.invoke(full_prompt)
    result = parser.parse(response.content)

    print(f"Problem Type: {result.problem_category}")
    print(f"Severity: {result.severity.value.upper()}")
    print(f"\nLikely Cause:")
    print(f"  {result.root_cause_hypothesis}")

    print(f"\nDiagnostic Commands:")
    for i, step in enumerate(result.diagnostic_steps, 1):
        print(f"\n  Step {i}: {step.command}")
        print(f"  Why: {step.purpose}")
        print(f"  Check: {step.expected_output}")

    print(f"\nResolution:")
    for step in result.resolution_steps:
        print(f"  → {step}")

    print(f"\nTime estimate: {result.estimated_time_to_fix}")

    print("\n" + "=" * 60 + "\n")
    return result


def multi_issue_triage():
    """
    Example 4: Triage multiple concurrent issues
    """
    print("=" * 60)
    print("Example 4: Multi-Issue Triage")
    print("=" * 60)

    issues = [
        {
            "id": 1,
            "description": "BGP neighbor 10.1.1.1 down for 5 minutes",
            "impact": "50% of internet traffic affected"
        },
        {
            "id": 2,
            "description": "Interface GigabitEthernet0/1 flapping every 30 seconds",
            "impact": "Branch office connectivity unstable"
        },
        {
            "id": 3,
            "description": "OSPF neighbor timing out on Vlan100",
            "impact": "Redundant path unavailable"
        },
        {
            "id": 4,
            "description": "High CPU on core switch (85%)",
            "impact": "Slight increase in latency"
        }
    ]

    print("\nConcurrent Issues:")
    for issue in issues:
        print(f"  [{issue['id']}] {issue['description']}")
        print(f"      Impact: {issue['impact']}")

    prompt = """Given these concurrent network issues, provide triage order.

For each issue, assess:
1. Severity (Critical/High/Medium/Low)
2. Business impact
3. Estimated time to fix
4. Dependencies (does fixing one help others?)

Then provide recommended order to address them."""

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    issues_text = "\n".join([
        f"Issue {i['id']}: {i['description']} (Impact: {i['impact']})"
        for i in issues
    ])

    response = llm.invoke(f"{prompt}\n\nIssues:\n{issues_text}")

    print("\nTriage Analysis:")
    print("-" * 60)
    print(response.content)

    print("\n" + "=" * 60 + "\n")


def autonomous_troubleshooting_workflow():
    """
    Example 5: Autonomous troubleshooting workflow
    """
    print("=" * 60)
    print("Example 5: Autonomous Troubleshooting Workflow")
    print("=" * 60)

    workflow = """
AUTONOMOUS TROUBLESHOOTING AGENT WORKFLOW:

Phase 1: INTAKE
├── Collect symptoms from user/monitoring
├── Parse syslog messages
├── Gather show command outputs
└── Identify affected devices/services

Phase 2: CLASSIFICATION
├── Categorize problem type (BGP, OSPF, Interface, Performance, Security)
├── Assess severity (Critical → Low)
├── Estimate business impact
└── Check if known issue (knowledge base lookup)

Phase 3: DIAGNOSIS
├── Generate hypothesis list (ranked by probability)
├── For each hypothesis:
│   ├── Determine diagnostic commands
│   ├── Execute commands (via MCP tools)
│   ├── Analyze output
│   └── Update hypothesis confidence
└── Identify root cause

Phase 4: RESOLUTION
├── Generate resolution steps
├── Check for required approvals (config changes)
├── Execute remediation (if auto-approved)
│   ├── Backup current config
│   ├── Apply fix
│   ├── Verify fix worked
│   └── Rollback if issues
└── Escalate to human if needed

Phase 5: DOCUMENTATION
├── Log all steps taken
├── Update knowledge base
├── Generate incident report
└── Create preventive recommendations

HUMAN-IN-THE-LOOP CHECKPOINTS:
✋ Before running write commands
✋ Before config changes on production
✋ If confidence < 70% on diagnosis
✋ If estimated fix time > 30 minutes
✋ If multiple critical systems affected

AUTONOMOUS CAPABILITIES:
✅ Read-only diagnostics (show commands)
✅ Log analysis and pattern recognition
✅ Configuration comparison
✅ Performance metric analysis
✅ Recommendation generation

REQUIRES APPROVAL:
❌ Configuration changes
❌ Service restarts
❌ Interface shutdowns
❌ Routing changes
❌ ACL modifications
"""

    print(workflow)

    print("\nCode Integration Example:")
    print("-" * 60)
    print("""
from troubleshooting_agent import TroubleshootingAgent

agent = TroubleshootingAgent(
    llm_model="claude-sonnet-4-20250514",
    mcp_tools_enabled=True,
    auto_execute_safe_commands=True,
    require_approval_for_changes=True
)

# Autonomous troubleshooting
result = agent.troubleshoot(
    symptom="BGP neighbor 10.1.1.1 down",
    context={
        "device": "router-01",
        "severity": "critical",
        "business_hours": True
    }
)

print(f"Root Cause: {result.root_cause}")
print(f"Resolution: {result.resolution_steps}")
print(f"Confidence: {result.confidence}%")

if result.requires_human:
    print("⚠️  Escalating to network engineer")
    send_alert(result)
else:
    print("✅ Autonomous resolution successful")
""")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔧 Chapter 54: AI-Powered Network Troubleshooting")
    print("Autonomous Agent for Network Diagnostics\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        exit(1)

    try:
        # Run examples
        analyze_bgp_issue()
        input("Press Enter to continue...")

        analyze_interface_flapping()
        input("Press Enter to continue...")

        analyze_performance_issue()
        input("Press Enter to continue...")

        multi_issue_triage()
        input("Press Enter to continue...")

        autonomous_troubleshooting_workflow()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- AI can diagnose network issues with structured reasoning")
        print("- Provide diagnostic commands with clear purpose")
        print("- Triage multiple issues by severity and impact")
        print("- Human-in-the-loop for config changes")
        print("- Autonomous for read-only diagnostics\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
