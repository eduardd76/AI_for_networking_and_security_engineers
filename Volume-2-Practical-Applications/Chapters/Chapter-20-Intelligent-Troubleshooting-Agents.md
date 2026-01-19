# Chapter 20: Intelligent Troubleshooting Agents

## Why This Chapter Matters

"Internet is down" ticket comes in at 2 AM. Traditional approach:
1. SSH to edge router
2. Check interface status
3. Check routing table
4. Check BGP neighbors
5. Check upstream links
6. Find root cause after 30 minutes

**AI Agent approach**:
1. Tell agent: "Users can't access internet"
2. Agent automatically runs diagnostics
3. Agent identifies root cause in 2 minutes
4. Agent suggests fix with config commands

This chapter builds autonomous troubleshooting agents that:
- Understand natural language problem descriptions
- Decide which diagnostic commands to run
- Execute commands safely (read-only by default)
- Analyze outputs
- Identify root causes
- Suggest fixes with actual config commands

**This is where AI becomes genuinely useful.**

---

## Section 1: Agent Architecture

### What is an AI Agent?

**Not an agent**: Simple chatbot that answers questions
**Is an agent**: System that takes actions to achieve a goal

**Key components**:
1. **LLM Brain**: Reasoning and decision-making
2. **Tools**: Functions the agent can call (show commands, ping, etc.)
3. **Memory**: Track conversation and actions taken
4. **Planning**: Break complex problems into steps
5. **Safety**: Prevent dangerous commands

**Agent loop**:
```
1. Receive goal: "Troubleshoot connectivity issue"
2. Think: "I should check interface status first"
3. Act: Run show command
4. Observe: Analyze output
5. Think: "Interface is down, I should check logs"
6. Act: Run log command
7. Observe: Found error
8. Think: "Root cause identified"
9. Final answer: "Interface down due to error X, fix with command Y"
```

### Agent Types

**ReAct Agent** (Reason + Act):
- Think → Act → Observe → Think → Act → ...
- Best for troubleshooting

**Plan-and-Execute Agent**:
- Plan all steps first, then execute
- Good for structured processes

**Tool-Calling Agent**:
- LLM directly calls functions
- Most reliable, recommended for production

---

## Section 2: Building a Basic Troubleshooting Agent

### Define Tools

```python
# troubleshooting_tools.py
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type
import subprocess

# Tool input schemas
class ShowCommandInput(BaseModel):
    """Input for show command tool."""
    command: str = Field(
        description="Show command to run (e.g., 'show ip interface brief'). "
                    "Must start with 'show'"
    )

class PingInput(BaseModel):
    """Input for ping tool."""
    host: str = Field(description="Hostname or IP address to ping")
    count: int = Field(default=4, description="Number of ping packets")

class TraceRouteInput(BaseModel):
    """Input for traceroute tool."""
    destination: str = Field(description="Destination IP or hostname")

# Define tools
class ShowCommandTool(BaseTool):
    """Execute show commands on network devices."""
    name: str = "show_command"
    description: str = """
    Execute read-only show commands on network devices.
    Use this to gather diagnostic information.
    Examples: 'show ip interface brief', 'show ip bgp summary', 'show ip route'
    """
    args_schema: Type[BaseModel] = ShowCommandInput

    device_connection = None  # In production: Netmiko connection

    def _run(self, command: str) -> str:
        """Run show command."""
        # Safety check
        if not command.strip().lower().startswith("show"):
            return "ERROR: Only 'show' commands are allowed"

        # In production: use Netmiko
        # output = self.device_connection.send_command(command)
        # return output

        # Simulated output for demo
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
      10.0.0.0/8 is variably subnetted
C        10.1.1.0/24 is directly connected, GigabitEthernet0/0
C        10.2.2.0/24 is directly connected, GigabitEthernet0/1
S*    0.0.0.0/0 [1/0] via 192.168.1.254
            """,
            "show ip bgp summary": """
BGP router identifier 1.1.1.1, local AS number 65001
Neighbor        V    AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
192.168.1.2     4 65002    1234    1235        0    0    0 00:15:23        150
203.0.113.1     4 65003       0       0        0    0    0 never    Idle
            """,
            "show interface gigabitethernet0/1": """
GigabitEthernet0/1 is up, line protocol is down
  Hardware is iGbE, address is 0000.0c07.ac01 (bia 0000.0c07.ac01)
  MTU 1500 bytes, BW 1000000 Kbit/sec
  Last input never, output 00:00:01, output hang never
  Last clearing of "show interface" counters never
  Input queue: 0/75/0/0 (size/max/drops/flushes); Total output drops: 0
  5 minute input rate 0 bits/sec, 0 packets/sec
  5 minute output rate 0 bits/sec, 0 packets/sec
            """,
        }

        return simulated_outputs.get(
            command.lower(),
            f"[Simulated output for: {command}]\nNo specific output configured."
        )

class PingTool(BaseTool):
    """Ping a host to test connectivity."""
    name: str = "ping"
    description: str = """
    Ping a host to test network connectivity.
    Use this to verify if a host is reachable.
    Returns ping statistics.
    """
    args_schema: Type[BaseModel] = PingInput

    def _run(self, host: str, count: int = 4) -> str:
        """Execute ping."""
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), host],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"Ping to {host} timed out"
        except Exception as e:
            return f"Ping failed: {str(e)}"

class TraceRouteTool(BaseTool):
    """Trace route to destination."""
    name: str = "traceroute"
    description: str = """
    Trace the network path to a destination.
    Use this to identify where packets are being dropped.
    """
    args_schema: Type[BaseModel] = TraceRouteInput

    def _run(self, destination: str) -> str:
        """Execute traceroute."""
        try:
            result = subprocess.run(
                ["traceroute", "-m", "15", destination],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except Exception as e:
            return f"Traceroute failed: {str(e)}"


# List of all tools
def get_troubleshooting_tools():
    """Get all troubleshooting tools."""
    return [
        ShowCommandTool(),
        PingTool(),
        TraceRouteTool()
    ]
```

### Create the Agent

```python
# troubleshooting_agent.py
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from troubleshooting_tools import get_troubleshooting_tools

class TroubleshootingAgent:
    """AI-powered network troubleshooting agent."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        # Get tools
        self.tools = get_troubleshooting_tools()

        # Agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert network troubleshooting assistant.

Your goal: Diagnose network issues by running diagnostic commands and analyzing outputs.

Available tools:
- show_command: Run show commands to gather info
- ping: Test connectivity to hosts
- traceroute: Trace network path

Process:
1. Understand the symptom
2. Form a hypothesis
3. Run diagnostic commands to test hypothesis
4. Analyze results
5. Identify root cause
6. Suggest fix with specific commands

Safety rules:
- Only use show commands (read-only)
- Never run config commands
- Explain your reasoning at each step

Be systematic, thorough, and explain your logic."""),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        # Agent executor
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


# Example usage
if __name__ == "__main__":
    agent = TroubleshootingAgent(api_key="your-api-key")

    # Problem 1: Interface down
    result = agent.troubleshoot(
        "Users on VLAN 20 (10.2.2.0/24) cannot access the network"
    )

    print(f"\n{'='*60}")
    print(f"Problem: {result['problem']}")
    print(f"{'='*60}")
    print(f"\nAnalysis:\n{result['analysis']}")
    print(f"\nSteps taken: {result['steps_taken']}")

# Expected agent reasoning:
"""
1. THINK: Users on VLAN 20 can't access network. I should check interface status.
2. ACTION: show_command("show ip interface brief")
3. OBSERVATION: GigabitEthernet0/1 (10.2.2.1) is up/down - line protocol is down!
4. THINK: Interface is up but protocol is down. This suggests Layer 2 issue.
5. ACTION: show_command("show interface gigabitethernet0/1")
6. OBSERVATION: "Last input never" - no traffic received. Possible cable/switchport issue.
7. THINK: Root cause identified - Layer 2 connectivity issue on Gi0/1
8. FINAL ANSWER:
   Root Cause: GigabitEthernet0/1 line protocol is down despite interface being administratively up.
   This indicates a Layer 2 issue - likely:
   - Cable unplugged/faulty
   - Remote switchport disabled
   - Duplex mismatch

   Recommended actions:
   1. Check physical cable connection
   2. Verify remote switchport config: show cdp neighbors detail
   3. Check for errors: show interface gigabitethernet0/1 | include error
   4. If duplex mismatch, configure: interface gi0/1; duplex auto
"""
```

---

## Section 3: Multi-Stage Troubleshooting

### Structured Investigation Process

```python
# multi_stage_agent.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum

class DiagnosticStage(str, Enum):
    IDENTIFY_LAYER = "identify_layer"  # L1/L2/L3?
    CHECK_PHYSICAL = "check_physical"
    CHECK_DATA_LINK = "check_data_link"
    CHECK_NETWORK = "check_network"
    CHECK_ROUTING = "check_routing"
    ROOT_CAUSE_FOUND = "root_cause_found"

class DiagnosticPlan(BaseModel):
    stage: DiagnosticStage = Field(description="Current diagnostic stage")
    hypothesis: str = Field(description="What we think the problem is")
    commands_to_run: List[str] = Field(description="Commands to run in this stage")
    reasoning: str = Field(description="Why these commands")

class MultiStageTroubleshooter:
    """Structured multi-stage troubleshooting."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.parser = JsonOutputParser(pydantic_object=DiagnosticPlan)

    def plan_next_stage(
        self,
        symptom: str,
        previous_findings: List[Dict] = None
    ) -> DiagnosticPlan:
        """Plan the next diagnostic stage."""
        findings_text = "\n".join([
            f"Stage {i+1}: {f['stage']}\nCommands: {f['commands']}\nResults: {f['results'][:200]}..."
            for i, f in enumerate(previous_findings or [])
        ])

        prompt = ChatPromptTemplate.from_template("""
Plan the next troubleshooting stage for this network issue.

Symptom: {symptom}

Previous findings:
{findings}

Based on OSI model troubleshooting (bottom-up or top-down):
1. Identify which layer is likely affected
2. Form a hypothesis
3. Choose diagnostic commands
4. Explain reasoning

Return JSON matching schema:
{format_instructions}

JSON:""")

        response = self.llm.invoke(
            prompt.format(
                symptom=symptom,
                findings=findings_text or "None yet",
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
        """Execute a diagnostic stage."""
        results = []

        for command in plan.commands_to_run:
            output = show_command_tool._run(command)
            results.append({
                "command": command,
                "output": output
            })

        return {
            "stage": plan.stage,
            "hypothesis": plan.hypothesis,
            "commands": plan.commands_to_run,
            "results": "\n\n".join([
                f"$ {r['command']}\n{r['output']}"
                for r in results
            ])
        }

    def analyze_results(
        self,
        symptom: str,
        stage_results: List[Dict]
    ) -> str:
        """Analyze all results and provide root cause analysis."""
        results_text = "\n\n".join([
            f"=== Stage: {r['stage']} ===\nHypothesis: {r['hypothesis']}\n\n{r['results']}"
            for r in stage_results
        ])

        prompt = ChatPromptTemplate.from_template("""
Analyze the troubleshooting results and provide root cause analysis.

Original symptom: {symptom}

All diagnostic results:
{results}

Provide:
1. Root cause (what's actually wrong)
2. Evidence (which outputs prove this)
3. Impact (why users are affected)
4. Fix (exact commands to resolve)
5. Prevention (how to avoid this in future)

Format as a clear, actionable report.""")

        response = self.llm.invoke(
            prompt.format(symptom=symptom, results=results_text)
        )

        return response.content

    def troubleshoot_full(
        self,
        symptom: str,
        max_stages: int = 4
    ) -> Dict:
        """Complete multi-stage troubleshooting."""
        from troubleshooting_tools import ShowCommandTool

        tool = ShowCommandTool()
        previous_findings = []

        print(f"Starting troubleshooting: {symptom}\n")

        for stage_num in range(max_stages):
            print(f"\n{'='*60}")
            print(f"STAGE {stage_num + 1}")
            print(f"{'='*60}")

            # Plan next stage
            plan = self.plan_next_stage(symptom, previous_findings)

            print(f"Hypothesis: {plan.hypothesis}")
            print(f"Stage: {plan.stage}")
            print(f"Commands: {plan.commands_to_run}")
            print(f"Reasoning: {plan.reasoning}\n")

            # Execute stage
            results = self.execute_stage(plan, tool)
            previous_findings.append(results)

            print(f"Results collected.")

            # Check if root cause found
            if plan.stage == DiagnosticStage.ROOT_CAUSE_FOUND:
                print("\nRoot cause identified!")
                break

        # Analyze all results
        print(f"\n{'='*60}")
        print("FINAL ANALYSIS")
        print(f"{'='*60}\n")

        analysis = self.analyze_results(symptom, previous_findings)
        print(analysis)

        return {
            "symptom": symptom,
            "stages": len(previous_findings),
            "findings": previous_findings,
            "analysis": analysis
        }


# Example usage
if __name__ == "__main__":
    troubleshooter = MultiStageTroubleshooter(api_key="your-api-key")

    result = troubleshooter.troubleshoot_full(
        symptom="Users report intermittent internet connectivity from branch office",
        max_stages=3
    )
```

---

## Section 4: Agent with Memory and Context

### Conversational Troubleshooting

```python
# conversational_agent.py
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from troubleshooting_tools import get_troubleshooting_tools

class ConversationalTroubleshootingAgent:
    """Agent that maintains conversation context."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.2
        )

        # Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Tools
        self.tools = get_troubleshooting_tools()

        # Prompt with memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network troubleshooting assistant having a conversation with a network engineer.

Remember the context of previous messages. If they ask a follow-up question, reference earlier findings.

Use tools to gather diagnostic information when needed."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def chat(self, message: str) -> str:
        """Chat with the agent."""
        result = self.agent_executor.invoke({"input": message})
        return result['output']


# Example conversation
if __name__ == "__main__":
    agent = ConversationalTroubleshootingAgent(api_key="your-api-key")

    # Initial problem
    response = agent.chat("Users on VLAN 20 can't access the internet")
    print(f"Agent: {response}\n")
    # Agent runs: show ip interface brief
    # Agent identifies: Gi0/1 is down

    # Follow-up question (agent remembers context!)
    response = agent.chat("What caused that interface to go down?")
    print(f"Agent: {response}\n")
    # Agent runs: show interface gi0/1
    # Agent analyzes: No input traffic, likely L2 issue

    # Another follow-up
    response = agent.chat("How do I fix it?")
    print(f"Agent: {response}\n")
    # Agent provides fix based on all previous analysis
```

---

## Section 5: Production Safety Features

### Prevent Dangerous Commands

```python
# safe_agent.py
from langchain.tools import BaseTool
from typing import List
import re

class SafeCommandValidator:
    """Validate commands before execution."""

    # Allowed command prefixes
    SAFE_COMMANDS = [
        "show",
        "ping",
        "traceroute",
        "display",  # Huawei
        "get",      # Fortinet
    ]

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'\bno\b',
        r'\bshutdown\b',
        r'\breload\b',
        r'\bwrite\s+erase\b',
        r'\bformat\b',
        r'\bdelete\b',
        r'\bclear\s+(line|ip\s+bgp)\b',
        r'\bconfigure\b',
        r'\bconf\s+t\b',
    ]

    @classmethod
    def is_safe(cls, command: str) -> tuple[bool, str]:
        """
        Check if command is safe to run.

        Returns:
            (is_safe: bool, reason: str)
        """
        command_lower = command.lower().strip()

        # Check if starts with safe command
        is_safe_start = any(
            command_lower.startswith(safe)
            for safe in cls.SAFE_COMMANDS
        )

        if not is_safe_start:
            return False, f"Command must start with one of: {', '.join(cls.SAFE_COMMANDS)}"

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower):
                return False, f"Command contains dangerous pattern: {pattern}"

        return True, "Command is safe"


class SafeShowCommandTool(BaseTool):
    """Show command tool with safety validation."""
    name: str = "show_command"
    description: str = "Execute safe read-only commands"

    def _run(self, command: str) -> str:
        """Run command with safety check."""
        is_safe, reason = SafeCommandValidator.is_safe(command)

        if not is_safe:
            return f"ERROR: Command blocked for safety: {reason}"

        # Execute command (your Netmiko code here)
        return "Command output..."


# Usage in agent
"""
Agent tries: "configure terminal"
→ Blocked: "Command must start with one of: show, ping, traceroute..."

Agent tries: "show running-config"
→ Allowed ✓

Agent tries: "show ip route | include no"
→ Blocked: "Command contains dangerous pattern: \\bno\\b"
"""
```

### Human-in-the-Loop for Critical Actions

```python
# approval_agent.py
class ApprovalRequiredTool(BaseTool):
    """Tool that requires human approval."""
    name: str = "apply_config"
    description: str = "Apply configuration changes (requires approval)"

    def _run(self, config: str) -> str:
        """Apply config with approval."""
        print(f"\n{'='*60}")
        print("APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"\nProposed configuration:\n{config}\n")

        approval = input("Apply this config? (yes/no): ")

        if approval.lower() == 'yes':
            # Apply config
            return "Configuration applied successfully"
        else:
            return "Configuration rejected by operator"


# Agent can suggest fixes, but human must approve
"""
Agent: "To fix this issue, apply this config:"
interface GigabitEthernet0/1
 no shutdown

Waiting for approval...
[Human reviews and approves]

Agent: "Configuration applied successfully"
"""
```

---

## What Can Go Wrong

**1. Agent runs in loops**
- Keeps running same commands
- Hits max iterations
- Solution: Add loop detection, limit retries

**2. Agent misinterprets output**
- Command output is ambiguous
- Agent draws wrong conclusion
- Solution: Structured output parsing, validation

**3. Expensive token usage**
- Agent makes 20+ tool calls
- Each call costs money
- Solution: Set max_iterations, use cheaper model for simple tasks

**4. Safety bypass attempts**
- LLM tries clever command injection
- Solution: Strict command validation, whitelist only

**5. Context limit exceeded**
- Long conversation + command outputs
- Exceeds context window
- Solution: Summarize old messages, limit output size

---

## Key Takeaways

1. **Agents automate investigation** - Not just answering, but acting
2. **Tools extend capabilities** - Show commands, ping, traceroute
3. **Multi-stage approach** - Systematic, like human troubleshooting
4. **Memory enables conversation** - Follow-up questions work naturally
5. **Safety is critical** - Validate all commands, require approval for changes

Next chapters: Config generation, log analysis, and more practical applications.
