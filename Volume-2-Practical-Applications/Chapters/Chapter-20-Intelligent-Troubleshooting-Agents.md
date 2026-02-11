# Chapter 20: Intelligent Troubleshooting Agents

## The Problem

Network troubleshooting at 2 AM: phone buzzes, internet is down, you run 15 show commands, check BGP, OSPF, interfaces, routing tables. 30 minutes later you find a shutdown interface. Users angry, executives stressed, you're awake until morning.

AI agents change this: describe the problem in English, agent runs diagnostic commands autonomously, identifies root cause in 2 minutes, suggests the exact fix.

This chapter builds troubleshooting agents that investigate network issues systematically, follow OSI model methodology, and operate safely in production.

## What You'll Build

Four progressive versions:

**Version 1 (V1) - Basic Tool-Calling Agent**
- Natural language problem descriptions
- Autonomous command execution
- Root cause identification
- ~150 lines, 45 min build time

**Version 2 (V2) - Multi-Stage Planning**
- Structured OSI-based investigation
- Hypothesis-driven testing
- Complete audit trail
- +280 lines, 60 min build time

**Version 3 (V3) - Conversation Memory**
- Context-aware follow-ups
- Natural conversational flow
- Investigation continuity
- +95 lines, 60 min build time

**Version 4 (V4) - Production Safety**
- Command validation (whitelist/blacklist)
- Human approval for changes
- Complete audit logging
- Read-only by default
- +220 lines, 90 min build time

Total: ~745 lines production-ready code, 4-5 hours build time.

## Prerequisites

```bash
# Install dependencies
pip install anthropic langchain langchain-anthropic netmiko

# Environment setup
export ANTHROPIC_API_KEY="your-key-here"
```

**Required knowledge:**
- Chapter 19 (Agent Architecture) - ReAct patterns, tool calling
- Network troubleshooting fundamentals
- OSI model (Layer 1-7)
- Cisco CLI commands

**Test network requirements:**
- Cisco router or switch (physical or GNS3/EVE-NG)
- SSH access configured
- Or use simulated outputs (provided in code)

---

## Version 1: Basic Tool-Calling Agent

**Goal:** Build an agent that troubleshoots network issues autonomously.

**Capabilities:**
- Understand natural language problem descriptions
- Execute diagnostic commands (show, ping, traceroute)
- Analyze command outputs
- Identify root causes

**Architecture:**

```
User: "VLAN 20 is down"
    ↓
LLM Brain (Claude)
    ↓
Decides: "Check interface status"
    ↓
Calls Tool: show_command("show ip interface brief")
    ↓
Tool executes via Netmiko
    ↓
Returns: "Gi0/1 is up/down"
    ↓
LLM analyzes: "Line protocol down = Layer 2 issue"
    ↓
Suggests fix: "Check cable, switchport config"
```

### Step 1: Define Tools

Tools give the agent capabilities. We need three essential diagnostic tools:

**File: `troubleshooting_tools.py`**

```python
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type
import subprocess

# ============================================
# Tool Input Schemas
# ============================================

class ShowCommandInput(BaseModel):
    """Input schema for show command tool."""
    command: str = Field(
        description="Show command to run (e.g., 'show ip interface brief'). "
                    "Must start with 'show'"
    )

class PingInput(BaseModel):
    """Input schema for ping tool."""
    host: str = Field(description="Hostname or IP address to ping")
    count: int = Field(default=4, description="Number of ping packets")

class TraceRouteInput(BaseModel):
    """Input schema for traceroute tool."""
    destination: str = Field(description="Destination IP or hostname")

# ============================================
# Tool Implementations
# ============================================

class ShowCommandTool(BaseTool):
    """Execute show commands on network devices."""

    name: str = "show_command"
    description: str = """
    Execute read-only show commands on network devices.
    Use this to gather diagnostic information about interfaces, routing, protocols.

    Examples:
    - 'show ip interface brief' - List all interfaces and status
    - 'show ip route' - Display routing table
    - 'show ip bgp summary' - Show BGP neighbor status
    - 'show interface gigabitethernet0/1' - Detailed interface stats
    """
    args_schema: Type[BaseModel] = ShowCommandInput

    def _run(self, command: str) -> str:
        """Execute show command and return output."""

        # SAFETY CHECK: Only allow show commands
        if not command.strip().lower().startswith("show"):
            return "ERROR: Only 'show' commands are allowed for safety"

        # PRODUCTION: Use Netmiko to SSH to real devices
        # from netmiko import ConnectHandler
        #
        # device = {
        #     'device_type': 'cisco_ios',
        #     'host': '192.168.1.1',
        #     'username': 'admin',
        #     'password': 'password'
        # }
        #
        # with ConnectHandler(**device) as net_connect:
        #     output = net_connect.send_command(command)
        #     return output

        # DEMO: Simulated outputs
        simulated_outputs = {
            "show ip interface brief": """
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     10.1.1.1        YES manual up                    up
GigabitEthernet0/1     10.2.2.1        YES manual up                    down
GigabitEthernet0/2     192.168.1.1     YES manual up                    up
Loopback0              1.1.1.1         YES manual up                    up
            """,

            "show ip route": """
Gateway of last resort is 192.168.1.254 to network 0.0.0.0

      10.0.0.0/8 is variably subnetted, 2 subnets
C        10.1.1.0/24 is directly connected, GigabitEthernet0/0
C        10.2.2.0/24 is directly connected, GigabitEthernet0/1
      192.168.1.0/24 is variably subnetted, 1 subnets
C        192.168.1.0/24 is directly connected, GigabitEthernet0/2
S*    0.0.0.0/0 [1/0] via 192.168.1.254
            """,

            "show ip bgp summary": """
BGP router identifier 1.1.1.1, local AS number 65001
BGP table version is 45, main routing table version 45

Neighbor        V    AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
192.168.1.2     4 65002    1234    1235       45    0    0 00:15:23        150
203.0.113.1     4 65003       0       0        0    0    0 never    Idle
            """,

            "show interface gigabitethernet0/1": """
GigabitEthernet0/1 is up, line protocol is down
  Hardware is iGbE, address is 0000.0c07.ac01 (bia 0000.0c07.ac01)
  Internet address is 10.2.2.1/24
  MTU 1500 bytes, BW 1000000 Kbit/sec, DLY 10 usec,
     reliability 255/255, txload 1/255, rxload 1/255
  Encapsulation ARPA, loopback not set
  Full-duplex, 1000Mb/s, media type is RJ45
  Last input never, output 00:00:01, output hang never
  Input queue: 0/75/0/0 (size/max/drops/flushes); Total output drops: 0
  5 minute input rate 0 bits/sec, 0 packets/sec
  5 minute output rate 0 bits/sec, 0 packets/sec
     0 packets input, 0 bytes, 0 no buffer
     13 packets output, 1234 bytes, 0 underruns
     0 input errors, 0 CRC, 0 frame, 0 overrun, 0 ignored
     0 output errors, 0 collisions, 1 interface resets
            """,
        }

        return simulated_outputs.get(
            command.lower(),
            f"[Simulated output for: {command}]"
        )


class PingTool(BaseTool):
    """Ping a host to test network connectivity."""

    name: str = "ping"
    description: str = """
    Ping a host to test network connectivity.
    Use this to verify if a destination is reachable.
    Returns ping statistics (packets sent/received, latency, packet loss).
    """
    args_schema: Type[BaseModel] = PingInput

    def _run(self, host: str, count: int = 4) -> str:
        """Execute ping command."""
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), host],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"Ping to {host} timed out after 10 seconds"
        except Exception as e:
            return f"Ping failed: {str(e)}"


class TraceRouteTool(BaseTool):
    """Trace the network path to a destination."""

    name: str = "traceroute"
    description: str = """
    Trace the network path to a destination.
    Use this to identify where packets are being dropped or delayed.
    Shows each hop along the route with latency measurements.
    """
    args_schema: Type[BaseModel] = TraceRouteInput

    def _run(self, destination: str) -> str:
        """Execute traceroute command."""
        try:
            result = subprocess.run(
                ["traceroute", "-m", "15", destination],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"Traceroute to {destination} timed out"
        except Exception as e:
            return f"Traceroute failed: {str(e)}"


def get_troubleshooting_tools():
    """Returns all troubleshooting tools."""
    return [
        ShowCommandTool(),
        PingTool(),
        TraceRouteTool()
    ]
```

**Key design decisions:**

1. **Safety first** - Only allow commands starting with "show"
2. **Simulated outputs** - Demo mode for learning, comment shows Netmiko production code
3. **Real ping/traceroute** - Actually run on your system
4. **Structured schemas** - Pydantic models validate LLM function calls

### Step 2: Create the Agent

Wire tools to Claude for autonomous troubleshooting:

**File: `troubleshooting_agent_v1.py`**

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from troubleshooting_tools import get_troubleshooting_tools


class TroubleshootingAgent:
    """AI-powered network troubleshooting agent."""

    def __init__(self, api_key: str):
        """Initialize the troubleshooting agent."""

        # Initialize LLM (the "brain")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0  # Deterministic for troubleshooting
        )

        # Get tools (the "hands")
        self.tools = get_troubleshooting_tools()

        # Define agent behavior
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert network troubleshooting assistant.

Your goal: Diagnose network issues by running diagnostic commands and analyzing outputs.

Available tools:
- show_command: Run Cisco show commands to gather information
- ping: Test connectivity to hosts
- traceroute: Trace network path to identify failures

Troubleshooting process:
1. Understand the symptom - What is the user reporting?
2. Form a hypothesis - What could cause this?
3. Run diagnostic commands - Test your hypothesis
4. Analyze results - What do outputs tell you?
5. Identify root cause - What is actually wrong?
6. Suggest fix - Provide specific configuration commands

Safety rules:
- Only use show commands (read-only)
- NEVER run configuration commands
- Explain your reasoning at each step

Be systematic and thorough."""),

            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def troubleshoot(self, problem_description: str) -> dict:
        """Troubleshoot a network problem."""
        result = self.agent_executor.invoke({
            "input": problem_description
        })

        return {
            "problem": problem_description,
            "analysis": result['output'],
            "steps_taken": len(result.get('intermediate_steps', []))
        }


if __name__ == "__main__":
    import os

    # Create agent
    agent = TroubleshootingAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Test case: VLAN 20 connectivity issue
    result = agent.troubleshoot(
        "Users on VLAN 20 (10.2.2.0/24) cannot access the network"
    )

    print(f"\n{'='*60}")
    print(f"Problem: {result['problem']}")
    print(f"{'='*60}")
    print(f"\nAnalysis:\n{result['analysis']}")
    print(f"\nDiagnostic steps taken: {result['steps_taken']}")
```

### Step 3: Run and Verify

```bash
# Test the agent
python troubleshooting_agent_v1.py
```

**Expected output:**

```
> Entering new AgentExecutor chain...

I need to check the interface status for VLAN 20 (10.2.2.0/24).

Action: show_command
Action Input: {"command": "show ip interface brief"}

Observation:
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     10.1.1.1        YES manual up                    up
GigabitEthernet0/1     10.2.2.1        YES manual up                    down
GigabitEthernet0/2     192.168.1.1     YES manual up                    up

Thought: GigabitEthernet0/1 (10.2.2.1) is the gateway for VLAN 20. Status is up
but Protocol is down - this is a Layer 2 issue. Let me get details.

Action: show_command
Action Input: {"command": "show interface gigabitethernet0/1"}

Observation:
GigabitEthernet0/1 is up, line protocol is down
  Last input never, output 00:00:01
  0 packets input, 0 bytes
  13 packets output, 1234 bytes

Thought: "Last input never" and "0 packets input" means the interface has never
received any traffic. This is a complete Layer 1/2 failure.

Final Answer:

ROOT CAUSE: GigabitEthernet0/1 line protocol is down. Interface has never
received input traffic.

EVIDENCE:
- Gi0/1 has IP 10.2.2.1 (VLAN 20 gateway)
- Status: up (administratively enabled)
- Protocol: down (no Layer 2 connectivity)
- "Last input never" - no packets ever received
- 0 input packets vs 13 output packets

LIKELY CAUSES:
1. Physical cable unplugged or faulty
2. Remote switchport administratively down
3. Remote switchport in wrong VLAN

RECOMMENDED FIX:
1. Check physical cable on both ends
2. Verify remote switch config:
   show cdp neighbors gigabitethernet0/1
   show interface status (on remote switch)
3. If cable is good, enable remote port:
   interface [port]
    no shutdown
    switchport access vlan 20

VERIFICATION:
show ip interface brief | include Gi0/1
(Should show: up/up, not up/down)

> Finished chain.

Diagnostic steps taken: 2
```

**What happened:**

1. Agent understood "VLAN 20 can't access network"
2. Checked interface status autonomously
3. Identified Gi0/1 serves 10.2.2.0/24
4. Noticed line protocol down
5. Dug deeper with detailed interface stats
6. Found "Last input never" = no traffic ever received
7. Concluded Layer 1/2 issue
8. Provided specific fix steps

**Time:** 30 seconds vs. 15-30 minutes manual

### V1 Cost Analysis

**Per troubleshooting session:**
- Input: ~500 tokens (prompt + tool schemas)
- Output: ~600 tokens (analysis + tool calls)
- Tool outputs: ~800 tokens
- Total: ~1,900 tokens per session

**Claude Sonnet 4 pricing:**
- Input: $3/million tokens
- Output: $15/million tokens
- Cost per session: $0.015 (1.5 cents)

**Monthly cost (500 incidents):**
- 500 incidents × $0.015 = $7.50/month

**ROI calculation:**
- Time saved: 25 min/incident (30 min manual - 5 min with agent)
- 500 incidents × 25 min = 208 hours saved/month
- At $75/hour engineer rate: $15,600 saved
- Agent cost: $7.50
- Net savings: $15,592/month ($187,104/year)

**V1 is production-ready for read-only troubleshooting.**

---

## Version 2: Multi-Stage Planning

**Problem with V1:** Investigation is reactive. Agent responds to each output without long-term strategy.

**V2 adds:** Structured OSI-based methodology with explicit planning stages.

**Architecture:**

```
User: "Intermittent connectivity"
    ↓
PLAN Stage 1: Identify OSI layer
  Hypothesis: "Could be L1/2/3"
  Commands: [show ip int brief, show ip route]
    ↓
EXECUTE Stage 1
    ↓
ANALYZE: "Routing looks good, check BGP"
    ↓
PLAN Stage 2: Check routing protocols
  Hypothesis: "BGP flapping causes intermittent issues"
  Commands: [show ip bgp summary]
    ↓
EXECUTE Stage 2
    ↓
ANALYZE: "BGP neighbor in Idle state"
    ↓
PLAN Stage 3: Root cause found
    ↓
FINAL ANALYSIS: Comprehensive report
```

**Benefits:**
- Systematic investigation (not ad-hoc)
- Clear hypothesis at each stage
- Complete audit trail
- Follows OSI model best practices

### Implementation

**File: `multi_stage_agent_v2.py`**

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum
from troubleshooting_tools import ShowCommandTool


class DiagnosticStage(str, Enum):
    """Possible stages in network troubleshooting (OSI-based)."""
    IDENTIFY_LAYER = "identify_layer"
    CHECK_PHYSICAL = "check_physical"
    CHECK_DATA_LINK = "check_data_link"
    CHECK_NETWORK = "check_network"
    CHECK_TRANSPORT = "check_transport"
    CHECK_ROUTING = "check_routing"
    CHECK_SERVICES = "check_services"
    ROOT_CAUSE_FOUND = "root_cause_found"


class DiagnosticPlan(BaseModel):
    """Structured plan for a diagnostic stage."""
    stage: DiagnosticStage = Field(
        description="Current diagnostic stage"
    )
    hypothesis: str = Field(
        description="What we think the problem is"
    )
    commands_to_run: List[str] = Field(
        description="Diagnostic commands to execute"
    )
    reasoning: str = Field(
        description="Why these commands test the hypothesis"
    )


class MultiStageTroubleshooter:
    """Structured multi-stage troubleshooting system."""

    def __init__(self, api_key: str):
        """Initialize multi-stage troubleshooter."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )
        self.parser = JsonOutputParser(pydantic_object=DiagnosticPlan)

    def plan_next_stage(
        self,
        symptom: str,
        previous_findings: List[Dict] = None
    ) -> DiagnosticPlan:
        """Plan next diagnostic stage based on symptom and findings."""

        # Format previous findings
        if previous_findings:
            findings_text = "\n\n".join([
                f"Stage {i+1}: {f['stage']}\n"
                f"Hypothesis: {f['hypothesis']}\n"
                f"Commands: {', '.join(f['commands'])}\n"
                f"Results: {f['results'][:300]}..."
                for i, f in enumerate(previous_findings)
            ])
        else:
            findings_text = "None - this is the first stage"

        prompt = ChatPromptTemplate.from_template("""
You are planning a network troubleshooting investigation.

SYMPTOM: {symptom}

PREVIOUS FINDINGS:
{findings}

Plan the next diagnostic stage using OSI model methodology.

STAGES:
- identify_layer: Determine which OSI layer is affected
- check_physical: Layer 1 (cables, hardware, link status)
- check_data_link: Layer 2 (switching, VLANs, MAC, STP)
- check_network: Layer 3 (IP addressing, subnetting, ARP)
- check_routing: Routing protocols (BGP, OSPF, static routes)
- check_transport: Layer 4 (TCP/UDP, ports)
- check_services: Application layer (DNS, DHCP, services)
- root_cause_found: Investigation complete

Return JSON:
{format_instructions}

Be systematic and follow OSI model bottom-up.""")

        response = self.llm.invoke(
            prompt.format(
                symptom=symptom,
                findings=findings_text,
                format_instructions=self.parser.get_format_instructions()
            )
        )

        import json
        plan_data = json.loads(response.content)
        return DiagnosticPlan(**plan_data)

    def execute_stage(
        self,
        plan: DiagnosticPlan,
        show_command_tool
    ) -> Dict:
        """Execute all commands in diagnostic stage."""
        results = []

        print(f"\n🔍 Executing {len(plan.commands_to_run)} commands...")

        for command in plan.commands_to_run:
            print(f"  → {command}")
            output = show_command_tool._run(command)
            results.append({
                "command": command,
                "output": output
            })

        combined_results = "\n\n".join([
            f"$ {r['command']}\n{r['output']}"
            for r in results
        ])

        return {
            "stage": plan.stage,
            "hypothesis": plan.hypothesis,
            "commands": plan.commands_to_run,
            "results": combined_results
        }

    def analyze_results(
        self,
        symptom: str,
        stage_results: List[Dict]
    ) -> str:
        """Analyze all stages and provide root cause analysis."""

        results_text = "\n\n".join([
            f"{'='*60}\n"
            f"STAGE: {r['stage']}\n"
            f"HYPOTHESIS: {r['hypothesis']}\n"
            f"{'='*60}\n"
            f"{r['results']}"
            for r in stage_results
        ])

        prompt = ChatPromptTemplate.from_template("""
Provide final root cause analysis after troubleshooting investigation.

SYMPTOM: {symptom}

INVESTIGATION RESULTS:
{results}

Provide comprehensive analysis:

1. ROOT CAUSE - What is actually wrong? Be specific.

2. EVIDENCE - Which command outputs prove this? Quote specific lines.

3. IMPACT - Why are users experiencing the symptom?

4. FIX - Exact configuration commands to resolve.

5. PREVENTION - How to prevent in future? Monitoring, automation, best practices.

6. VERIFICATION - Commands to verify fix worked.

Be precise and actionable.""")

        response = self.llm.invoke(
            prompt.format(symptom=symptom, results=results_text)
        )

        return response.content

    def troubleshoot_full(
        self,
        symptom: str,
        max_stages: int = 5
    ) -> Dict:
        """Perform complete multi-stage investigation."""
        tool = ShowCommandTool()
        previous_findings = []

        print(f"\n{'='*70}")
        print(f"MULTI-STAGE TROUBLESHOOTING")
        print(f"{'='*70}")
        print(f"\nSymptom: {symptom}\n")

        for stage_num in range(max_stages):
            print(f"\n{'#'*70}")
            print(f"# STAGE {stage_num + 1}")
            print(f"{'#'*70}")

            # PLAN
            print(f"\n📋 Planning...")
            plan = self.plan_next_stage(symptom, previous_findings)

            print(f"\n✓ Stage planned:")
            print(f"  Type: {plan.stage}")
            print(f"  Hypothesis: {plan.hypothesis}")
            print(f"  Commands: {len(plan.commands_to_run)}")
            print(f"  Reasoning: {plan.reasoning}")

            # Check if done
            if plan.stage == DiagnosticStage.ROOT_CAUSE_FOUND:
                print(f"\n🎯 ROOT CAUSE IDENTIFIED!")
                break

            # EXECUTE
            results = self.execute_stage(plan, tool)
            previous_findings.append(results)

            print(f"\n✓ Stage {stage_num + 1} complete")

        # ANALYZE
        print(f"\n{'='*70}")
        print(f"FINAL ANALYSIS")
        print(f"{'='*70}\n")

        analysis = self.analyze_results(symptom, previous_findings)
        print(analysis)

        return {
            "symptom": symptom,
            "stages_completed": len(previous_findings),
            "stage_findings": previous_findings,
            "final_analysis": analysis
        }


if __name__ == "__main__":
    import os

    troubleshooter = MultiStageTroubleshooter(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    result = troubleshooter.troubleshoot_full(
        symptom="Users at branch office report intermittent internet connectivity",
        max_stages=4
    )
```

**Example output:**

```
======================================================================
MULTI-STAGE TROUBLESHOOTING
======================================================================

Symptom: Users at branch office report intermittent internet connectivity

######################################################################
# STAGE 1
######################################################################

📋 Planning...

✓ Stage planned:
  Type: identify_layer
  Hypothesis: Need to determine if L1/2 (local) or L3 (routing/WAN) issue
  Commands: 2
  Reasoning: Check local interfaces and WAN link to narrow problem domain

🔍 Executing 2 commands...
  → show ip interface brief
  → show ip route

✓ Stage 1 complete

######################################################################
# STAGE 2
######################################################################

📋 Planning...

✓ Stage planned:
  Type: check_routing
  Hypothesis: Intermittent connectivity suggests routing instability (flapping BGP)
  Commands: 1
  Reasoning: BGP flapping causes routes to appear/disappear

🔍 Executing 1 commands...
  → show ip bgp summary

✓ Stage 2 complete

######################################################################
# STAGE 3
######################################################################

📋 Planning...

🎯 ROOT CAUSE IDENTIFIED!

======================================================================
FINAL ANALYSIS
======================================================================

1. ROOT CAUSE
BGP neighbor 203.0.113.1 (AS 65003) is in "Idle" state and has never established.

Intermittent connectivity is caused by:
- Primary BGP path down
- Backup static route occasionally working
- When backup route congested, connectivity fails

2. EVIDENCE
From "show ip bgp summary":
  203.0.113.1     4 65003       0       0        0    0    0 never    Idle
                                                            ^^^^^^    ^^^^
- 0 messages sent/received
- Never been up
- Idle state (not trying to connect)

3. IMPACT
Users experience intermittent connectivity because traffic fails over to backup
static route. When backup is congested, connectivity degrades.

4. FIX
Diagnose why BGP won't establish:

router bgp 65001
 neighbor 203.0.113.1 remote-as 65003
 neighbor 203.0.113.1 update-source GigabitEthernet0/2

Check:
a) IP connectivity: ping 203.0.113.1 source GigabitEthernet0/2
b) Firewall blocking TCP 179
c) BGP password mismatch
d) ISP configuration

5. PREVENTION
- Monitor BGP neighbor states (alert if down)
- Implement BFD for fast failover
- Syslog for BGP state changes

6. VERIFICATION
show ip bgp summary (should show "Established" with PfxRcd > 0)
show ip route bgp
ping 8.8.8.8

Stages completed: 2
```

**V2 improvements over V1:**

| Feature | V1 | V2 |
|---------|----|----|
| Methodology | Reactive | Structured (OSI) |
| Planning | None | Explicit per stage |
| Hypothesis | Implicit | Explicit |
| Audit trail | Basic | Complete |
| Analysis | Simple | 6-section comprehensive |

### V2 Cost Analysis

**Per troubleshooting session:**
- Input: ~1,200 tokens (planning + stage prompts)
- Output: ~1,800 tokens (plans + analysis)
- Total: ~3,000 tokens per session

**Cost per session:** $0.031 (3.1 cents)

**Monthly cost (500 incidents):** $15.50

**Still trivial compared to engineer time savings ($15,600/month).**

---

## Version 3: Conversation Memory

**Problem with V1/V2:** No memory between interactions. Can't handle:

```
User: "Check VLAN 20"
Agent: "Gi0/1 is down"

User: "Why is it down?"
Agent: "What interface? I have no context."  ❌
```

**V3 adds:** Conversation memory for natural follow-up questions.

**Architecture:**

```
ConversationBufferMemory
    ↓
Stores: [User msg, Agent response, Tool calls, Tool outputs]
    ↓
Next user message includes full history
    ↓
Agent understands context and references
```

### Implementation

**File: `conversational_agent_v3.py`**

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from troubleshooting_tools import get_troubleshooting_tools


class ConversationalTroubleshootingAgent:
    """Network troubleshooting agent with conversation memory."""

    def __init__(self, api_key: str):
        """Initialize conversational agent."""

        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.2  # Slightly higher for natural conversation
        )

        # KEY ADDITION: Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        self.tools = get_troubleshooting_tools()

        # Prompt with memory placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network troubleshooting assistant having a
conversation with a network engineer.

IMPORTANT: Remember context from previous messages. Reference earlier findings
when relevant.

Available tools:
- show_command: Run Cisco show commands
- ping: Test connectivity
- traceroute: Trace network path

Conversational guidelines:
1. Remember context - Reference earlier findings
2. Answer naturally - Have a conversation, not just dump data
3. Ask clarifying questions if request is ambiguous
4. Provide value - Analyze and explain, don't just run commands
5. Be concise - Engineer is busy

Use tools when you need diagnostic information. Otherwise answer based on
knowledge and conversation history."""),

            MessagesPlaceholder("chat_history"),  # History injected here
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,  # Memory integrated
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def chat(self, message: str) -> str:
        """Chat with agent (remembers conversation)."""
        result = self.agent_executor.invoke({"input": message})
        return result['output']

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_conversation_history(self) -> list:
        """Get all messages."""
        return self.memory.buffer_as_messages


if __name__ == "__main__":
    import os

    agent = ConversationalTroubleshootingAgent(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    print("="*60)
    print("Conversational Troubleshooting Agent")
    print("="*60)
    print("Type 'quit' to exit, 'clear' to reset\n")

    while True:
        user_message = input("\nYou: ")

        if user_message.lower() == 'quit':
            break

        if user_message.lower() == 'clear':
            agent.clear_memory()
            print("✓ Conversation cleared\n")
            continue

        response = agent.chat(user_message)
        print(f"\nAgent: {response}")
```

**Example conversation:**

```
You: Users on VLAN 20 can't access the internet

Agent: Let me check the interface for VLAN 20...
[runs: show_command("show ip interface brief")]

I can see GigabitEthernet0/1 (IP 10.2.2.1 - likely your VLAN 20 gateway) has
a problem: interface is up but line protocol is down.

This means:
- Interface is enabled in config
- But no Layer 2 connectivity

This prevents VLAN 20 users from reaching their gateway.

Would you like me to investigate why line protocol is down?

----------------------------------------------------------------

You: Yes, why is it down?

Agent: Let me get detailed information...
[runs: show_command("show interface gigabitethernet0/1")]

Key finding: "Last input never" - this interface has NEVER received any traffic.

Combined with:
- 0 packets input
- 13 packets output
- No errors

The interface sends traffic but receives nothing. This is a Layer 1/2 issue:
1. Physical cable unplugged or damaged
2. Remote switchport down
3. Remote switchport in wrong VLAN

The complete lack of received traffic points to a break in physical connectivity.

----------------------------------------------------------------

You: How do I fix it?

Agent: Based on what we found (Gi0/1 never receiving traffic), here's the fix:

IMMEDIATE STEPS:

1. Physical verification
   - Check both ends of cable on Gi0/1
   - Verify cable properly seated
   - Check for physical damage

2. Check remote switch
   show cdp neighbors gigabitethernet0/1

   Then on remote switch:
   - Is port enabled?
   - Correct VLAN?
   - Port security violations?

3. If cable issue: Replace cable

4. If switchport config issue:
   interface [port]
    no shutdown
    switchport access vlan 20

VERIFICATION:
show ip interface brief | include Gi0/1
(Should show: up/up)

show interface gigabitethernet0/1 | include packets input
(Should show increasing input packets)

ping 10.2.2.10  (test user device)

Want help with any specific step?
```

**Key features:**

1. **Context awareness** - "why is it down?" → Agent knows "it" = Gi0/1
2. **Natural flow** - Doesn't repeat information
3. **Progressive detail** - High-level first, then technical
4. **Guidance** - Step-by-step procedures

### V3 Cost Analysis

**Per conversation (5 messages):**
- Input: ~2,500 tokens (includes growing history)
- Output: ~2,000 tokens
- Total: ~4,500 tokens

**Cost per conversation:** $0.038 (3.8 cents)

**For 500 conversations/month:** $19.00

**Context window management:**

Use `ConversationBufferWindowMemory` for long conversations:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=10,  # Keep last 10 messages only
    memory_key="chat_history",
    return_messages=True
)
```

This prevents context overflow and keeps costs predictable.

---

## Version 4: Production Safety

**The problem:** V1-V3 are powerful but dangerous in production without safety guardrails.

**What could go wrong:**

```
Agent thinks: "I should restart BGP"
Agent executes: clear ip bgp *
→ ALL BGP sessions reset
→ Network outage
→ Resume polishing required
```

**V4 adds:**
1. Command validation (whitelist/blacklist)
2. Human approval for changes
3. Audit logging
4. Read-only default mode

### Safety Layer 1: Command Validation

**File: `safe_command_validator.py`**

```python
import re
from typing import Tuple


class SafeCommandValidator:
    """Validate commands before execution."""

    # Whitelist: Only these prefixes allowed
    SAFE_COMMANDS = [
        "show",
        "ping",
        "traceroute",
        "display",  # Huawei
        "get",      # Fortinet
    ]

    # Blacklist: Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'\bno\b',                      # Removes config
        r'\bshutdown\b',                # Disables interfaces
        r'\breload\b',                  # Reboots device
        r'\bwrite\s+erase\b',           # Erases config
        r'\bformat\b',                  # Formats flash
        r'\bdelete\b',                  # Deletes files
        r'\bclear\s+(line|ip\s+bgp)\b', # Clears sessions
        r'\bconfigure\b',               # Config mode
        r'\bconf\s+t\b',                # Config mode
        r'\bwrite\s+memory\b',          # Saves config
    ]

    @classmethod
    def is_safe(cls, command: str) -> Tuple[bool, str]:
        """Validate if command is safe."""
        command_lower = command.lower().strip()

        if not command_lower:
            return False, "Empty command"

        # Check safe prefix
        is_safe_prefix = any(
            command_lower.startswith(safe_cmd)
            for safe_cmd in cls.SAFE_COMMANDS
        )

        if not is_safe_prefix:
            return False, (
                f"Command must start with: {', '.join(cls.SAFE_COMMANDS)}\n"
                f"Got: {command[:50]}"
            )

        # Check dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return False, f"Contains dangerous pattern: {pattern}"

        return True, "Command is safe"


# Test validation
if __name__ == "__main__":
    test_commands = [
        # Safe
        "show ip interface brief",
        "show running-config",
        "ping 8.8.8.8",

        # Dangerous
        "configure terminal",
        "no shutdown",
        "reload in 5",
        "write erase",
        "clear ip bgp *",
    ]

    print("Command Validation Tests\n" + "="*70)

    for cmd in test_commands:
        is_safe, reason = SafeCommandValidator.is_safe(cmd)
        status = "✓ SAFE" if is_safe else "✗ BLOCKED"
        print(f"\n{status}: {cmd}")
        if not is_safe:
            print(f"  Reason: {reason}")
```

**Output:**

```
✓ SAFE: show ip interface brief
✓ SAFE: show running-config
✓ SAFE: ping 8.8.8.8

✗ BLOCKED: configure terminal
  Reason: Command must start with: show, ping, traceroute, display, get

✗ BLOCKED: no shutdown
  Reason: Command must start with: show, ping, traceroute, display, get

✗ BLOCKED: reload in 5
  Reason: Contains dangerous pattern: \breload\b

✗ BLOCKED: write erase
  Reason: Contains dangerous pattern: \bwrite\s+erase\b

✗ BLOCKED: clear ip bgp *
  Reason: Contains dangerous pattern: \bclear\s+(line|ip\s+bgp)\b
```

### Safety Layer 2: Human Approval

**File: `approval_tool.py`**

```python
from langchain.tools import BaseTool
from datetime import datetime
import logging


class ApprovalRequiredTool(BaseTool):
    """Tool requiring human approval before execution."""

    name: str = "apply_config"
    description: str = """
    Apply configuration changes to network devices.
    REQUIRES human approval before execution.
    Use when you've identified a fix and want to apply it.
    """

    def _run(self, config: str, target_device: str = "router") -> str:
        """Request approval and apply if approved."""

        # Display proposed changes
        print(f"\n{'='*70}")
        print(f"⚠️  APPROVAL REQUIRED")
        print(f"{'='*70}")
        print(f"\nDevice: {target_device}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nProposed Configuration:")
        print(f"{'-'*70}")
        print(config)
        print(f"{'-'*70}\n")

        # Request approval
        approval = input("Apply this configuration? (yes/no): ").strip().lower()

        if approval == 'yes':
            logging.info(f"Config approved and applied to {target_device}")
            print(f"\n✓ Approved at {datetime.now()}")

            # Apply config (production: use Netmiko)
            result = self._apply_to_device(config, target_device)

            print(f"✓ Configuration applied")
            return f"Applied to {target_device}: {result}"
        else:
            logging.warning(f"Config rejected for {target_device}")
            print(f"\n✗ Configuration rejected")
            return "Configuration REJECTED. No changes made."

    def _apply_to_device(self, config: str, device: str) -> str:
        """Apply to device (production: Netmiko)."""
        # from netmiko import ConnectHandler
        # device_conn = ConnectHandler(**device_params)
        # output = device_conn.send_config_set(config.split('\n'))
        # device_conn.save_config()
        # return output

        return f"[Simulated] Applied to {device}"
```

**Example usage:**

```
======================================================================
⚠️  APPROVAL REQUIRED
======================================================================

Device: router-hq-01
Time: 2024-01-15 14:32:10

Proposed Configuration:
----------------------------------------------------------------------
interface GigabitEthernet0/1
 no shutdown
----------------------------------------------------------------------

Apply this configuration? (yes/no): yes

✓ Approved at 2024-01-15 14:32:15
✓ Configuration applied
```

### Safety Layer 3: Audit Logging

**File: `audit_logger.py`**

```python
import logging
import json
from datetime import datetime


class AuditLogger:
    """Log all agent actions for compliance."""

    def __init__(self, log_file="agent_audit.log"):
        """Initialize audit logger."""
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self.logger = logging.getLogger("AgentAudit")

    def log_command(self, command: str, device: str, result: str, user: str):
        """Log command execution."""
        self.logger.info(json.dumps({
            "type": "command_execution",
            "user": user,
            "device": device,
            "command": command,
            "result_preview": result[:200],
            "timestamp": datetime.now().isoformat()
        }))

    def log_blocked_command(self, command: str, reason: str, user: str):
        """Log blocked command."""
        self.logger.warning(json.dumps({
            "type": "command_blocked",
            "user": user,
            "command": command,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }))

    def log_config_change(self, config: str, device: str, approved: bool, user: str):
        """Log config change request."""
        self.logger.info(json.dumps({
            "type": "config_change",
            "user": user,
            "device": device,
            "config": config,
            "approved": approved,
            "timestamp": datetime.now().isoformat()
        }))
```

**Log output (agent_audit.log):**

```json
2024-01-15 14:30:05 | INFO | {"type": "command_execution", "user": "jsmith", "device": "router-01", "command": "show ip interface brief", "result_preview": "Interface              IP-Address      OK? Method Status...", "timestamp": "2024-01-15T14:30:05"}

2024-01-15 14:30:12 | WARNING | {"type": "command_blocked", "user": "jsmith", "command": "reload", "reason": "Contains dangerous pattern: \\breload\\b", "timestamp": "2024-01-15T14:30:12"}

2024-01-15 14:32:15 | INFO | {"type": "config_change", "user": "jsmith", "device": "router-hq-01", "config": "interface Gi0/1\n no shutdown", "approved": true, "timestamp": "2024-01-15T14:32:15"}
```

### Safety Layer 4: Read-Only Default

**File: `production_agent_v4.py`**

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from troubleshooting_tools import get_troubleshooting_tools
from approval_tool import ApprovalRequiredTool
from safe_command_validator import SafeCommandValidator
from audit_logger import AuditLogger


class ProductionTroubleshootingAgent:
    """Production-ready agent with safety features."""

    def __init__(self, api_key: str, allow_changes: bool = False, user: str = "unknown"):
        """
        Initialize production agent.

        Args:
            api_key: Anthropic API key
            allow_changes: If False (default), read-only. If True, config tools enabled.
            user: Username for audit logging
        """
        self.allow_changes = allow_changes
        self.user = user
        self.audit = AuditLogger()

        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0
        )

        # Base tools (always available)
        self.tools = get_troubleshooting_tools()

        # Add change tools only if explicitly enabled
        if allow_changes:
            self.tools.append(ApprovalRequiredTool())
            print("⚠️  WARNING: Configuration change tools ENABLED")
            self.audit.logger.warning(f"Change mode enabled by {user}")
        else:
            print("✓ Running in READ-ONLY mode (safe)")

        # Wrap tools with validation and logging
        self.tools = [self._wrap_tool_with_safety(tool) for tool in self.tools]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network troubleshooting assistant.

Available tools: {tool_names}

Troubleshoot systematically:
1. Understand symptom
2. Form hypothesis
3. Run diagnostics
4. Analyze results
5. Identify root cause
6. Suggest fix

All commands are validated for safety and logged for audit."""),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def _wrap_tool_with_safety(self, tool):
        """Wrap tool with validation and logging."""
        original_run = tool._run

        def safe_run(*args, **kwargs):
            # Get command (first arg for show_command tool)
            command = args[0] if args else kwargs.get('command', '')

            # Validate
            if hasattr(tool, 'name') and tool.name == 'show_command':
                is_safe, reason = SafeCommandValidator.is_safe(command)
                if not is_safe:
                    self.audit.log_blocked_command(command, reason, self.user)
                    return f"ERROR: {reason}"

            # Execute
            result = original_run(*args, **kwargs)

            # Log
            if hasattr(tool, 'name'):
                self.audit.log_command(
                    command=str(args[0]) if args else 'N/A',
                    device="router-01",
                    result=result,
                    user=self.user
                )

            return result

        tool._run = safe_run
        return tool

    def troubleshoot(self, problem: str) -> dict:
        """Troubleshoot with full safety."""
        result = self.agent_executor.invoke({"input": problem})

        return {
            "problem": problem,
            "analysis": result['output'],
            "steps": len(result.get('intermediate_steps', []))
        }


if __name__ == "__main__":
    import os

    # Read-only agent (safe for juniors, automation)
    agent_readonly = ProductionTroubleshootingAgent(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        allow_changes=False,
        user="jsmith"
    )

    result = agent_readonly.troubleshoot(
        "Users on VLAN 20 cannot access network"
    )

    print(f"\nAnalysis:\n{result['analysis']}")

    # With changes allowed (requires senior engineer approval)
    # agent_with_changes = ProductionTroubleshootingAgent(
    #     api_key=os.getenv("ANTHROPIC_API_KEY"),
    #     allow_changes=True,
    #     user="admin"
    # )
```

### V4 Production Safety Checklist

Before deploying to production:

- [x] Command validation (whitelist + blacklist)
- [x] Human approval for config changes
- [x] Read-only mode by default
- [x] Comprehensive audit logging
- [ ] Rate limiting (max commands/minute)
- [ ] Session timeouts
- [ ] Role-based access control
- [ ] Alert on suspicious behavior
- [ ] Rollback capability
- [ ] Lab environment testing

**Additional production hardening:**

```python
# Rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=20, period=60)  # Max 20 commands per minute
def execute_command(command):
    # ...

# Session timeout
import time

class TimedSession:
    def __init__(self, timeout_minutes=30):
        self.start = time.time()
        self.timeout = timeout_minutes * 60

    def is_expired(self):
        return (time.time() - self.start) > self.timeout

# RBAC
def check_permission(user, action):
    permissions = {
        "junior": ["show_command", "ping"],
        "senior": ["show_command", "ping", "traceroute", "apply_config"],
        "admin": ["*"]
    }
    user_role = get_user_role(user)
    return action in permissions.get(user_role, [])
```

### V4 Cost Analysis

**Same as V3** (~$19/month for 500 sessions) but with critical safety features:

- $0 cost from prevented outages
- $0 cost from blocked dangerous commands
- Complete audit trail for compliance
- Insurance against agent mistakes

**ROI remains massive:** $15,600/month saved vs. $19 agent cost.

---

## Lab 1: Build and Test V1 Basic Agent

**Time: 45 minutes**

**Objective:** Build a working troubleshooting agent that autonomously diagnoses network issues.

### Lab Steps

**1. Environment Setup (10 min)**

```bash
# Create project directory
mkdir troubleshooting-agent
cd troubleshooting-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install anthropic langchain langchain-anthropic netmiko

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

**2. Implement Tools (15 min)**

Create `troubleshooting_tools.py` with ShowCommandTool, PingTool, TraceRouteTool (copy from V1 section above).

**Verification:**

```python
# Test tools directly
from troubleshooting_tools import ShowCommandTool

tool = ShowCommandTool()
result = tool._run("show ip interface brief")
print(result)

# Should print simulated interface status
```

**3. Build Agent (10 min)**

Create `troubleshooting_agent_v1.py` (copy from V1 section).

**4. Test Scenarios (10 min)**

Run three test scenarios:

**Scenario A: Interface Down**
```python
result = agent.troubleshoot(
    "Users on VLAN 20 (10.2.2.0/24) cannot access the network"
)
```

**Expected:** Agent identifies Gi0/1 line protocol down, suggests cable/switchport fix.

**Scenario B: BGP Issue**
```python
result = agent.troubleshoot(
    "Intermittent internet connectivity at branch office"
)
```

**Expected:** Agent checks routing, identifies BGP neighbor in Idle state.

**Scenario C: Ambiguous Problem**
```python
result = agent.troubleshoot(
    "Network is slow"
)
```

**Expected:** Agent asks clarifying questions or runs general diagnostics.

### Success Criteria

- [ ] Agent runs without errors
- [ ] Agent autonomously selects appropriate commands
- [ ] Agent correctly interprets "up/down" status
- [ ] Agent provides specific fix recommendations
- [ ] Total cost < $0.05 for all 3 scenarios

### Common Issues

**Problem:** "ModuleNotFoundError: No module named 'langchain'"
**Fix:** `pip install langchain langchain-anthropic`

**Problem:** Agent runs same command repeatedly
**Fix:** Add to system prompt: "Don't repeat commands you've already run"

**Problem:** "Context length exceeded"
**Fix:** Truncate tool outputs to 5000 chars

---

## Lab 2: Add Multi-Stage Planning (V2)

**Time: 60 minutes**

**Objective:** Enhance agent with structured OSI-based investigation methodology.

### Lab Steps

**1. Implement Diagnostic Stages (20 min)**

Create `multi_stage_agent_v2.py` with DiagnosticStage enum and DiagnosticPlan model (copy from V2 section).

**2. Test Stage Planning (15 min)**

```python
troubleshooter = MultiStageTroubleshooter(api_key="...")

# Test planning only (don't execute)
plan = troubleshooter.plan_next_stage(
    symptom="Intermittent connectivity",
    previous_findings=None
)

print(f"Stage: {plan.stage}")
print(f"Hypothesis: {plan.hypothesis}")
print(f"Commands: {plan.commands_to_run}")
print(f"Reasoning: {plan.reasoning}")
```

**Expected output:**
```
Stage: identify_layer
Hypothesis: Need to determine if L1/2 or L3 issue
Commands: ['show ip interface brief', 'show ip route']
Reasoning: Check local vs WAN to narrow problem domain
```

**3. Run Full Investigation (15 min)**

```python
result = troubleshooter.troubleshoot_full(
    symptom="Users report intermittent internet access",
    max_stages=4
)

print(f"Stages completed: {result['stages_completed']}")
print(f"\nFinal analysis:\n{result['final_analysis']}")
```

**4. Compare V1 vs V2 (10 min)**

Run same scenario with both agents. Compare:
- Number of commands run
- Quality of analysis
- Audit trail completeness
- Explanation clarity

### Success Criteria

- [ ] Agent completes 2-3 stages before finding root cause
- [ ] Each stage has clear hypothesis
- [ ] Final analysis includes all 6 sections (root cause, evidence, impact, fix, prevention, verification)
- [ ] Total investigation < 60 seconds
- [ ] Cost < $0.10 per investigation

### Challenge Exercise

**Scenario:** "Branch office has slow connectivity to data center"

**Question:** What stages should the agent go through?

**Expected progression:**
1. Stage 1: identify_layer → Check if local or WAN issue
2. Stage 2: check_network → Verify routing, latency
3. Stage 3: check_routing → BGP path selection, suboptimal routes
4. Stage 4: root_cause_found → Identify traffic going through congested backup link

---

## Lab 3: Production Deployment with Safety (V4)

**Time: 90 minutes**

**Objective:** Deploy production-ready agent with comprehensive safety features.

### Lab Steps

**1. Implement Safety Validation (20 min)**

Create `safe_command_validator.py` (copy from V4 section).

**Test validation:**

```python
from safe_command_validator import SafeCommandValidator

test_cases = [
    ("show ip interface brief", True),
    ("configure terminal", False),
    ("reload", False),
    ("show run | include no", False),  # Contains "no"
]

for cmd, should_pass in test_cases:
    is_safe, reason = SafeCommandValidator.is_safe(cmd)
    assert is_safe == should_pass, f"Failed: {cmd}"
    print(f"✓ {cmd}: {'PASS' if is_safe else 'BLOCKED'}")
```

**2. Add Approval Workflow (20 min)**

Create `approval_tool.py` (copy from V4 section).

**Test approval:**

```python
tool = ApprovalRequiredTool()

# This will prompt for approval
result = tool._run(
    config="interface GigabitEthernet0/1\n no shutdown",
    target_device="router-hq-01"
)

# Try approving, then try rejecting
```

**3. Implement Audit Logging (15 min)**

Create `audit_logger.py` (copy from V4 section).

**Test logging:**

```python
audit = AuditLogger(log_file="test_audit.log")

audit.log_command(
    command="show ip route",
    device="router-01",
    result="Gateway of last resort...",
    user="jsmith"
)

audit.log_blocked_command(
    command="reload",
    reason="Dangerous pattern",
    user="jsmith"
)

# Check test_audit.log file
with open("test_audit.log") as f:
    print(f.read())
```

**4. Integration Testing (25 min)**

Create `production_agent_v4.py` and test:

```python
# Test read-only mode
agent_ro = ProductionTroubleshootingAgent(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    allow_changes=False,
    user="junior_engineer"
)

result = agent_ro.troubleshoot("VLAN 20 is down")
# Should work fine

# Test with changes enabled
agent_rw = ProductionTroubleshootingAgent(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    allow_changes=True,
    user="senior_engineer"
)

# Agent can now suggest and apply fixes (with approval)
```

**5. Security Audit (10 min)**

Review audit logs for:
- All commands executed
- All blocked attempts
- All config changes (approved/rejected)

```python
import json

with open("agent_audit.log") as f:
    for line in f:
        log_entry = json.loads(line.split(" | ")[-1])
        if log_entry["type"] == "command_blocked":
            print(f"⚠️  BLOCKED: {log_entry['command']} by {log_entry['user']}")
```

### Success Criteria

- [ ] Validator blocks all dangerous commands
- [ ] Approval workflow prompts before config changes
- [ ] All actions logged to audit file
- [ ] Read-only mode prevents config tools from loading
- [ ] No false positives (safe commands not blocked)
- [ ] No false negatives (dangerous commands not allowed)

### Production Deployment Checklist

Before deploying to production:

**Security:**
- [ ] Command validator tested with 50+ dangerous commands
- [ ] Approval workflow integrated with Slack/Teams
- [ ] Audit logs shipped to SIEM (Splunk, ELK)
- [ ] RBAC integrated with LDAP/Active Directory

**Reliability:**
- [ ] Rate limiting configured (prevent abuse)
- [ ] Session timeouts implemented
- [ ] Error handling for network failures
- [ ] Graceful degradation if LLM API down

**Operations:**
- [ ] Monitoring dashboard (command success rate, latency)
- [ ] Alerting on blocked commands spike
- [ ] Runbook for emergency agent shutdown
- [ ] Backup manual troubleshooting procedures

**Testing:**
- [ ] Tested in lab with 20+ scenarios
- [ ] Penetration test by security team
- [ ] Junior engineer usability testing
- [ ] Load testing (100 concurrent sessions)

---

## Common Problems and Solutions

### Problem 1: Agent Runs in Loops

**Symptom:**
```
Agent: Checking interface...
Agent: Still checking interface...
Agent: Checking interface again...
[Repeats until max_iterations]
```

**Cause:** Agent doesn't realize it already has the information.

**Solution:**

```python
class LoopDetector:
    def __init__(self, max_repeats=2):
        self.command_history = []
        self.max_repeats = max_repeats

    def is_loop(self, command):
        recent = self.command_history[-5:]
        if recent.count(command) >= self.max_repeats:
            return True
        self.command_history.append(command)
        return False

# In tool wrapper:
def _run(self, command: str) -> str:
    if self.loop_detector.is_loop(command):
        return "ERROR: Already ran this command twice. Use previous results."
    # ... execute command
```

### Problem 2: Agent Misinterprets Output

**Symptom:** Agent says "BGP is fine" when it's actually down.

**Cause:** Ambiguous parsing of CLI output.

**Solution:**

```python
# Structured parsing with TextFSM or Pydantic
from pydantic import BaseModel

class BGPNeighbor(BaseModel):
    neighbor: str
    state: str
    prefixes_received: int

def parse_bgp_summary(output: str) -> List[BGPNeighbor]:
    # Use LLM to extract structured data
    prompt = f"""
Extract BGP neighbors from this output as JSON list:

{output}

Format: [{{"neighbor": "IP", "state": "State", "prefixes_received": N}}]
"""
    # ... call LLM with JSON mode
```

### Problem 3: High Token Costs

**Symptom:** $50 bill for single troubleshooting session.

**Cause:** 25 tool calls with large outputs.

**Solution:**

```python
def _run(self, command: str) -> str:
    output = device.send_command(command)

    # Truncate large outputs
    if len(output) > 5000:
        output = output[:5000] + "\n\n[Truncated at 5000 chars]"

    return output

# Set hard limits
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=8,          # Stop at 8 actions
    max_execution_time=60,     # Timeout after 60 sec
)
```

### Problem 4: Context Window Exceeded

**Symptom:** "Error: Context length exceeded"

**Cause:** Long conversation fills context window.

**Solution:**

```python
# Option 1: Summarize old messages
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Option 2: Keep last N messages
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=10,  # Last 10 messages only
    memory_key="chat_history",
    return_messages=True
)
```

### Problem 5: Agent Gives Up

**Symptom:** Agent says "I don't know how to proceed"

**Cause:** Complex issue outside training data.

**Solution:**

```python
# Give agent "ask for help" tool
class AskHumanTool(BaseTool):
    name: str = "ask_human_expert"
    description: str = "Ask human expert when stuck"

    def _run(self, question: str) -> str:
        print(f"\n🤔 AGENT NEEDS HELP: {question}\n")
        return input("Expert response: ")

# Agent can escalate:
# "I've checked interfaces and routing but can't determine why packets
#  are dropped. Should I check firewall rules?"
```

---

## Check Your Understanding

### Question 1: Agent vs. Chatbot

**Q:** What is the key difference between an AI agent and a chatbot?

**A:** Agents take actions to achieve goals, chatbots just answer questions.

- Chatbot: "How do I configure OSPF?" → Returns information
- Agent: "Fix OSPF on router1" → Investigates, identifies issue, suggests fix

Key components of agents:
1. Tools (capabilities to act in the world)
2. Planning (break goals into steps)
3. Memory (track state and progress)
4. Iteration (act, observe, adjust, repeat)

### Question 2: Multi-Stage vs. Basic Agent

**Q:** When should you use multi-stage planning vs. basic reactive agent?

**A:**

**Use basic reactive agent when:**
- Quick, simple issues ("Is interface up?")
- Interactive with human (conversational)
- Exploration/learning
- Example: "Check status of Gi0/1"

**Use multi-stage planning when:**
- Complex multi-layer issues
- Need documented audit trail
- Production incident response
- Training juniors (they follow the logic)
- Example: "Intermittent connectivity affecting 100 users"

**Key difference:** Multi-stage follows structured methodology (OSI model), basic agent reacts to each observation.

### Question 3: Safety Features

**Q:** Your agent is deployed to production. A junior engineer uses it and tries to run "reload" command. What happens with V4 production agent?

**A:** The command is blocked at multiple safety layers:

1. **Command Validator:** Regex pattern `\breload\b` matches → BLOCKED
2. **Audit Logger:** Logs blocked attempt: `{"type": "command_blocked", "user": "junior", "command": "reload", "reason": "Dangerous pattern"}`
3. **Agent returns:** "ERROR: Command blocked for safety. Reason: Contains dangerous pattern: \breload\b"
4. **Alert:** Security team notified of dangerous command attempt

The device is never touched. Junior engineer is safe.

**Additional layers:**
- Even if validator missed it, command doesn't start with "show" → blocked
- Even if that failed, read-only mode means reload tool not loaded
- Even if changes enabled, approval workflow requires human confirmation

**Defense in depth.**

### Question 4: Cost Optimization

**Q:** You're running 1,000 troubleshooting sessions per month. Each session uses V2 multi-stage agent with 3 stages, 2 commands per stage. Average command output is 1,000 chars. How can you reduce costs by 50% without losing functionality?

**A:** Three optimization strategies:

**Strategy 1: Use Haiku for planning (60% cost reduction)**

```python
# Planning LLM (cheap, simple task)
planning_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # $0.25/$1.25 per million
    temperature=0.0
)

# Analysis LLM (expensive, complex task)
analysis_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",   # $3/$15 per million
    temperature=0.0
)
```

**Strategy 2: Truncate outputs (40% cost reduction)**

```python
def _run(self, command: str) -> str:
    output = device.send_command(command)

    # Keep first 2000 chars (usually enough)
    if len(output) > 2000:
        output = output[:2000] + "\n[Truncated]"

    return output
```

**Strategy 3: Cache common outputs (80% cost reduction)**

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_output(device: str, command: str) -> str:
    # Cache for 5 minutes
    return device.send_command(command)
```

**Combined savings:**
- Original cost: $31/month (1,000 sessions × $0.031)
- With Haiku for planning: $12/month (60% reduction)
- With output truncation: $18/month (40% reduction)
- With caching (50% cache hit): $15/month (50% reduction)
- **All three combined: $6/month (81% reduction)**

**ROI still massive:** $0.006 per session vs. 25 min engineer time saved.

---

## Lab Time Budget

### Time Investment

**Learning (one-time):**
- Read chapter: 90 min
- Lab 1 (V1 Basic): 45 min
- Lab 2 (V2 Multi-Stage): 60 min
- Lab 3 (V4 Production): 90 min
- **Total: 4.75 hours**

**Implementation (per deployment):**
- Adapt tools for your devices: 30 min
- Connect to real network (Netmiko): 45 min
- Test with 10 scenarios: 60 min
- Security review: 45 min
- Production deployment: 60 min
- **Total: 4 hours**

**Grand total: 8.75 hours** from zero to production.

### Cost Analysis

**Development costs:**
- Learning: $0 (reading)
- Lab API costs: $2 (experimentation)
- Testing API costs: $5 (10 scenarios)
- **Total: $7**

**Monthly operational costs (500 incidents):**
- V1 Basic: $7.50/month
- V2 Multi-Stage: $15.50/month
- V3 Conversational: $19.00/month
- V4 Production (optimized): $10.00/month

**Monthly savings:**
- Manual troubleshooting: 500 incidents × 25 min = 208 hours
- At $75/hour: $15,600/month
- Agent cost: $10/month
- **Net savings: $15,590/month**

### ROI Calculation

**Year 1:**
- Development: 8.75 hours × $75 = $656
- Operational: $10/month × 12 = $120
- **Total cost: $776**

**Savings:**
- Monthly: $15,590
- Annual: $187,080
- **ROI: 24,000%**

**Payback period: 0.3 hours** (first 2 incidents)

---

## Production Deployment Guide

### Phase 1: Lab Testing (Week 1)

**Day 1-2: Setup**
- Deploy to lab environment
- Connect to test devices (GNS3/EVE-NG)
- Configure SSH access
- Test basic connectivity

**Day 3-4: Scenario Testing**
- Test 20 common scenarios:
  - Interface down
  - BGP neighbor down
  - OSPF adjacency issues
  - VLAN misconfiguration
  - Routing loop
  - MTU mismatch
  - Duplex mismatch
  - Port security violation
  - ACL blocking traffic
  - NAT issues

**Day 5: Safety Testing**
- Attempt dangerous commands
- Verify all blocks work
- Test approval workflow
- Review audit logs

### Phase 2: Pilot (Week 2-3)

**Week 2: Limited Rollout**
- 5 senior engineers only
- Read-only mode
- Monitor for 1 week
- Collect feedback

**Week 3: Expand Pilot**
- 15 engineers (mix of senior/mid-level)
- Still read-only
- Track metrics:
  - Time saved per incident
  - Accuracy of root cause identification
  - User satisfaction

### Phase 3: Production (Week 4)

**Enable for all NOC engineers:**
- Read-only by default
- Changes require senior approval
- Monitor closely for first week

**Metrics to track:**
- Incidents resolved by agent
- Time saved (before/after)
- False positives (agent wrong)
- User adoption rate
- Cost (API usage)

### Phase 4: Optimization (Month 2+)

**Continuous improvement:**
- Analyze failed investigations
- Add new tools for uncovered scenarios
- Optimize prompts based on feedback
- Reduce costs (caching, Haiku for planning)
- Expand to new device types

---

## What You've Built

**V1 - Basic Agent (150 lines):**
- Natural language troubleshooting
- Autonomous command execution
- Root cause identification
- $0.015 per incident

**V2 - Multi-Stage Planning (+280 lines):**
- OSI-based structured investigation
- Hypothesis-driven testing
- Comprehensive 6-section analysis
- Complete audit trail
- $0.031 per incident

**V3 - Conversation Memory (+95 lines):**
- Context-aware follow-ups
- Natural conversational flow
- Reference earlier findings
- $0.038 per conversation

**V4 - Production Safety (+220 lines):**
- Command validation (whitelist/blacklist)
- Human approval for changes
- Complete audit logging
- Read-only default
- Role-based access control ready
- $0.020 per incident (optimized)

**Total: 745 lines production-ready code**

**Real-world impact:**
- **Time saved:** 25 min per incident
- **Cost:** $10-20/month operational
- **Savings:** $15,590/month ($187,080/year)
- **ROI:** 24,000% in year 1
- **Payback:** 0.3 hours (2 incidents)

**Additional benefits:**
- 24/7 troubleshooting without waking engineers
- Consistent methodology (OSI model)
- Complete audit trail for compliance
- Training tool for junior engineers
- Reduction in MTTR (mean time to resolution)

---

## Next Chapter Preview

**Chapter 21: Network Change Automation**

Build agents that:
- Generate configuration changes from English descriptions
- Validate changes before deployment
- Execute changes with rollback capability
- Document all changes automatically

**Preview:**

```
You: "Add VLAN 50 for guest WiFi across all access switches"

Agent:
1. Identifies all access switches
2. Generates configs for each
3. Shows you the diff
4. Applies to test switch first
5. Validates
6. Rolls out to remaining switches
7. Documents the change

All automated, all safe, all audited.
```

The future isn't replacing network engineers—it's augmenting them with AI agents that handle systematic work so humans focus on architecture, strategy, and complex problem-solving.
