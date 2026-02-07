# Chapter 20: Intelligent Troubleshooting Agents

## Why This Chapter Matters

Picture this: It's 2 AM. Your phone buzzes with a critical alert—"Internet is down at headquarters." You groan, grab your laptop, and begin the familiar routine:

**Traditional troubleshooting workflow:**
1. SSH to the edge router (find the IP, remember the password)
2. Run `show ip interface brief` to check interface status
3. Run `show ip route` to verify routing table
4. Run `show ip bgp summary` to check BGP neighbors
5. Run `show interface` on suspicious ports
6. Analyze logs, check for errors
7. Finally identify the root cause... **30 minutes later**

By the time you've found the problem, users are angry, executives are stressed, and you're wide awake for the rest of the night.

**AI Agent approach:**
1. Tell the agent: "Users at HQ can't access internet"
2. Agent automatically runs the right diagnostic commands
3. Agent identifies root cause in **2 minutes**
4. Agent suggests the exact fix with configuration commands

This isn't science fiction—it's what you'll build in this chapter.

**Networking analogy**: Think of a troubleshooting agent like an automated runbook on steroids. If your current runbooks say "Step 1: Check BGP neighbors. Step 2: If down, check interface. Step 3: If interface is down, check cable..." — an AI agent does the same thing, but it can *reason* about the output of each step and dynamically decide what to check next, just like an experienced engineer would. It's the difference between a static routing table and a routing protocol.

### What You'll Learn

This chapter teaches you to build **autonomous troubleshooting agents** that:

- **Understand natural language problem descriptions** ("VLAN 20 is down" → knows which commands to run)
- **Decide which diagnostic commands to run** (interface checks → routing checks → BGP checks)
- **Execute commands safely** (read-only by default, with human approval for changes)
- **Analyze command outputs** (parse complex CLI output and extract meaning)
- **Identify root causes** (not just symptoms, but the actual problem)
- **Suggest precise fixes** (exact configuration commands to resolve the issue)

**This is where AI transforms from "interesting" to "indispensable."**

---

## Section 1: Understanding AI Agents

Before we build anything, let's understand what makes something an "agent" versus a simple chatbot.

### What IS an AI Agent?

**❌ Not an agent:** A chatbot that answers questions
- You: "How do I configure OSPF?"
- Chatbot: "Here's the config syntax..."
- (Chatbot just retrieves information)

**✅ Is an agent:** A system that takes actions to achieve a goal
- You: "Users can't reach 10.5.5.0/24"
- Agent: *Thinks* → "I should check routing"
- Agent: *Acts* → Runs `show ip route 10.5.5.0`
- Agent: *Observes* → No route found
- Agent: *Thinks* → "Need to check BGP"
- Agent: *Acts* → Runs `show ip bgp`
- Agent: *Concludes* → "BGP neighbor is down, here's the fix"

The key difference: **Agents take actions in pursuit of a goal.** They don't just answer—they investigate, decide, execute, and iterate until the problem is solved.

### Agent Architecture: The Five Essential Components

Every effective AI agent needs these five components working together:

#### 1. **LLM Brain** (Reasoning and Decision-Making)
The large language model (like GPT-4 or Claude) serves as the "brain" that:
- Understands the problem description
- Decides which actions to take next
- Analyzes command outputs
- Draws conclusions

Think of it as the experienced network engineer's intuition—the part that says "if the interface is up/down, that's a Layer 2 issue."

#### 2. **Tools** (Actions the Agent Can Take)
Tools are functions the agent can call to interact with the world:
- `show_command()` - Run Cisco show commands
- `ping()` - Test connectivity
- `traceroute()` - Trace network path
- `get_logs()` - Retrieve syslog entries

Without tools, the LLM is just thinking. Tools let it **act**.

#### 3. **Memory** (Track Conversation and Actions)
The agent needs to remember:
- What problem you described
- Which commands it already ran
- What it found in previous outputs
- The conversation context

This prevents the agent from running the same command twice or losing track of its investigation.

#### 4. **Planning** (Break Complex Problems into Steps)
A good agent doesn't just randomly try commands. It:
- Breaks the problem into stages ("First check Layer 1, then Layer 2, then Layer 3")
- Forms hypotheses ("I think BGP is down")
- Tests hypotheses systematically
- Adjusts the plan based on findings

#### 5. **Safety** (Prevent Dangerous Commands)
The agent must have guardrails:
- Only allow read-only commands by default (`show`, `ping`, etc.)
- Block dangerous commands (`reload`, `write erase`, `no shutdown`)
- Require human approval for configuration changes
- Validate all commands before execution

**Safety is not optional.** An agent with `configure terminal` access and no guardrails is a disaster waiting to happen.

### The Agent Loop: How It All Works Together

Here's how these components work together in a real troubleshooting scenario:

```
User reports: "Troubleshoot connectivity issue to 10.2.2.0/24"
    ↓
1. THINK (LLM Brain):
   "Users can't reach a specific network. I should first check if the 
    interface for that subnet is up."
    ↓
2. PLAN:
   "Run 'show ip interface brief' to see all interface states"
    ↓
3. ACT (Tool):
   Execute: show_command("show ip interface brief")
    ↓
4. OBSERVE (Tool returns output):
   GigabitEthernet0/1   10.2.2.1   YES manual up   down
    ↓
5. ANALYZE (LLM Brain):
   "Interface is administratively up but line protocol is down.
    This is a Layer 2 issue—likely cable problem or switch port down."
    ↓
6. THINK:
   "I should check interface details to see why line protocol is down"
    ↓
7. ACT (Tool):
   Execute: show_command("show interface gigabitethernet0/1")
    ↓
8. OBSERVE:
   "Last input never, output 00:00:01"
    ↓
9. ANALYZE (LLM Brain):
   "No input traffic ever received. This confirms Layer 2 failure."
    ↓
10. CONCLUDE:
    "Root Cause: GigabitEthernet0/1 line protocol is down.
     Likely causes: Cable unplugged, remote switchport disabled, or duplex mismatch.
     Recommended fix: Check physical cable, verify remote switch config."
```

This loop continues until the agent either:
- Finds the root cause and suggests a fix
- Reaches maximum iterations (safety limit)
- Determines it needs human help

### Agent Types: Choosing the Right Architecture

There are three main types of AI agents, each suited for different use cases:

#### **ReAct Agent** (Reason + Act)
**Pattern:** Think → Act → Observe → Think → Act → Observe → ...

**How it works:**
- Agent reasons about what to do next
- Performs an action
- Observes the result
- Reasons again based on new information
- Continues until goal is achieved

**Best for:**
- **Troubleshooting** (where each step depends on the last)
- **Dynamic investigations** (path is not predetermined)
- **Exploratory tasks** (don't know the full scope upfront)

**Example:**
```
THINK: "Interface is down, I should check why"
ACT:   show interface gi0/1
OBSERVE: "Line protocol is down"
THINK: "No input traffic. Could be cable or switch issue"
ACT:   show cdp neighbors gi0/1
OBSERVE: "No CDP neighbor detected"
THINK: "Definitely a Layer 1/2 issue—cable likely unplugged"
```

#### **Plan-and-Execute Agent**
**Pattern:** Plan all steps → Execute step 1 → Execute step 2 → ... → Done

**How it works:**
- Creates a complete plan upfront
- Executes each step in sequence
- Does not adapt the plan based on intermediate results

**Best for:**
- **Structured processes** (same steps every time)
- **Change procedures** (predefined runbook)
- **Audits and compliance checks** (fixed checklist)

**Example:**
```
PLAN:
  Step 1: Check interface status
  Step 2: Check routing table
  Step 3: Check BGP peers
  Step 4: Generate report

EXECUTE:
  ✓ Step 1: show ip interface brief
  ✓ Step 2: show ip route
  ✓ Step 3: show ip bgp summary
  ✓ Step 4: Create summary
```

**Limitation:** If step 1 reveals the interface is down, a Plan-and-Execute agent will still run steps 2 and 3 (wasting time). A ReAct agent would adapt.

#### **Tool-Calling Agent** (Function-Calling)
**Pattern:** LLM directly calls functions as needed

**How it works:**
- Modern LLMs (GPT-4, Claude) support native "function calling"
- LLM decides which functions to call and with what parameters
- More reliable than parsing text output
- Industry best practice for production systems

**Best for:**
- **Production deployments** (most reliable)
- **Safety-critical systems** (explicit function calls are easier to validate)
- **Integration with existing tools** (clean API integration)

**Example:**
```
LLM decides: "I need to check interface status"
LLM calls: show_command(command="show ip interface brief")
Tool returns: [interface data]
LLM analyzes and decides next action
```

**Why it's better:**
- No parsing ambiguity (function parameters are structured)
- Built-in safety (function schema validates inputs)
- Easier to log and audit
- **This is what we'll use in this chapter**

### Comparing Agent Types

| Feature | ReAct | Plan-and-Execute | Tool-Calling |
|---------|-------|------------------|--------------|
| **Adaptability** | High (adjusts based on results) | Low (fixed plan) | High |
| **Efficiency** | Medium (may take detours) | High (if path is known) | High |
| **Best for troubleshooting?** | Yes ✓ | No | Yes ✓ |
| **Production readiness** | Good | Good | **Excellent** |
| **Setup complexity** | Medium | Low | Low |

### Key Insight: Why Agents Matter for Network Troubleshooting

Traditional scripts and automation follow **predefined logic**:
```python
if ping_fails:
    check_interface()
    if interface_down:
        send_alert()
```

AI agents use **reasoning and adaptation**:
```
Problem: Ping fails
Agent thinks: "Could be interface, routing, or firewall"
Agent investigates dynamically
Agent adapts based on what it finds
Agent concludes with root cause
```

**Scripts execute instructions. Agents pursue goals.**

That's why agents excel at troubleshooting—each network issue is unique, and the investigation path depends on what you discover along the way.

---

## Section 2: Building Your First Troubleshooting Agent

Now that you understand the theory, let's build a real troubleshooting agent. We'll start simple and add complexity in later sections.

### Step 1: Define Tools

Tools are the **actions** your agent can take. We'll create three essential troubleshooting tools:

1. `show_command` - Run Cisco show commands
2. `ping` - Test connectivity
3. `traceroute` - Trace network path

Here's the complete implementation:

```python
# troubleshooting_tools.py
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type
import subprocess

# ============================================
# PART 1: Define Input Schemas
# ============================================
# These schemas tell the LLM what parameters each tool needs

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
# PART 2: Implement Tool Classes
# ============================================

class ShowCommandTool(BaseTool):
    """Execute show commands on network devices."""
    
    name: str = "show_command"
    description: str = """
    Execute read-only show commands on network devices.
    Use this to gather diagnostic information about interfaces, routing, protocols, etc.
    
    Examples:
    - 'show ip interface brief' - List all interfaces and their status
    - 'show ip route' - Display routing table
    - 'show ip bgp summary' - Show BGP neighbor status
    - 'show interface gigabitethernet0/1' - Detailed interface stats
    """
    args_schema: Type[BaseModel] = ShowCommandInput

    device_connection = None  # In production: Netmiko SSH connection object

    def _run(self, command: str) -> str:
        """
        Execute a show command and return output.
        
        In production, this would use Netmiko to SSH to devices:
            output = self.device_connection.send_command(command)
            return output
        
        For this demo, we return simulated outputs.
        """
        
        # SAFETY CHECK: Only allow show commands
        if not command.strip().lower().startswith("show"):
            return "ERROR: Only 'show' commands are allowed for safety"

        # ==========================================
        # PRODUCTION CODE (commented out for demo):
        # ==========================================
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

        # ==========================================
        # SIMULATED OUTPUTS FOR DEMO:
        # ==========================================
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
  Keepalive set (10 sec)
  Full-duplex, 1000Mb/s, media type is RJ45
  output flow-control is unsupported, input flow-control is unsupported
  ARP type: ARPA, ARP Timeout 04:00:00
  Last input never, output 00:00:01, output hang never
  Last clearing of "show interface" counters never
  Input queue: 0/75/0/0 (size/max/drops/flushes); Total output drops: 0
  Queueing strategy: fifo
  Output queue: 0/40 (size/max)
  5 minute input rate 0 bits/sec, 0 packets/sec
  5 minute output rate 0 bits/sec, 0 packets/sec
     0 packets input, 0 bytes, 0 no buffer
     Received 0 broadcasts (0 IP multicasts)
     0 runts, 0 giants, 0 throttles
     0 input errors, 0 CRC, 0 frame, 0 overrun, 0 ignored
     0 watchdog, 0 multicast, 0 pause input
     13 packets output, 1234 bytes, 0 underruns
     0 output errors, 0 collisions, 1 interface resets
     0 unknown protocol drops
     0 babbles, 0 late collision, 0 deferred
     0 lost carrier, 0 no carrier, 0 pause output
     0 output buffer failures, 0 output buffers swapped out
            """,
        }

        # Return simulated output if available, otherwise generic message
        return simulated_outputs.get(
            command.lower(),
            f"[Simulated output for: {command}]\nNo specific output configured for this command."
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
            # Run actual ping command on the system
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
    Use this to identify where in the path packets are being dropped or delayed.
    Shows each hop along the route with latency measurements.
    """
    args_schema: Type[BaseModel] = TraceRouteInput

    def _run(self, destination: str) -> str:
        """Execute traceroute command."""
        try:
            # Run actual traceroute (max 15 hops)
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


# ============================================
# PART 3: Helper Function to Get All Tools
# ============================================

def get_troubleshooting_tools():
    """
    Returns a list of all troubleshooting tools.
    Use this when creating the agent.
    """
    return [
        ShowCommandTool(),
        PingTool(),
        TraceRouteTool()
    ]
```

**Key points about this code:**

1. **Input Schemas** - Tell the LLM exactly what parameters each tool needs
2. **Safety First** - The show command tool ONLY allows commands starting with "show"
3. **Simulated Outputs** - For demo purposes; in production, use Netmiko to SSH to real devices
4. **Real Ping/Traceroute** - These tools actually run on your system
5. **Error Handling** - Timeouts and exceptions are caught and returned as strings

### Step 2: Create the Agent

Now we'll wire up these tools with an LLM to create a functioning agent:

```python
# troubleshooting_agent.py
import os
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from troubleshooting_tools import get_troubleshooting_tools


class TroubleshootingAgent:
    """
    AI-powered network troubleshooting agent.
    
    This agent can:
    - Understand natural language problem descriptions
    - Run diagnostic commands autonomously
    - Analyze command outputs
    - Identify root causes
    - Suggest fixes
    """

    def __init__(self, api_key: str):
        """
        Initialize the troubleshooting agent.
        
        Args:
            api_key: Anthropic API key for Claude
        """
        
        # Initialize the LLM (the "brain")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0  # Deterministic for troubleshooting
        )

        # Get the tools (the "hands")
        self.tools = get_troubleshooting_tools()

        # Define the agent's instructions and behavior
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert network troubleshooting assistant.

Your goal: Diagnose network issues by running diagnostic commands and analyzing outputs.

Available tools:
- show_command: Run Cisco show commands to gather information
- ping: Test connectivity to hosts
- traceroute: Trace network path to identify where failures occur

Your troubleshooting process:
1. **Understand the symptom** - What exactly is the user reporting?
2. **Form a hypothesis** - What could cause this symptom?
3. **Run diagnostic commands** - Use tools to test your hypothesis
4. **Analyze results** - What do the outputs tell you?
5. **Identify root cause** - What is actually wrong (not just the symptom)?
6. **Suggest fix** - Provide specific configuration commands to resolve the issue

Safety rules:
- Only use show commands (read-only operations)
- NEVER run configuration commands
- Explain your reasoning at each step so the engineer understands your logic

Be systematic, thorough, and always explain your thought process."""),
            
            ("human", "{input}"),  # The problem description from the user
            MessagesPlaceholder("agent_scratchpad")  # Space for tool outputs and agent thoughts
        ])

        # Create the agent (combines LLM + tools + prompt)
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create the executor (runs the agent loop)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Print agent's thought process
            max_iterations=10,  # Safety limit: max 10 actions
            handle_parsing_errors=True  # Gracefully handle LLM errors
        )

    def troubleshoot(self, problem_description: str) -> dict:
        """
        Troubleshoot a network problem.
        
        Args:
            problem_description: Natural language description of the problem
            
        Returns:
            Dictionary with problem, analysis, and steps taken
        """
        result = self.agent_executor.invoke({
            "input": problem_description
        })

        return {
            "problem": problem_description,
            "analysis": result['output'],
            "steps_taken": len(result.get('intermediate_steps', []))
        }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Create the agent
    agent = TroubleshootingAgent(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Scenario: Users on VLAN 20 can't access the network
    result = agent.troubleshoot(
        "Users on VLAN 20 (10.2.2.0/24) cannot access the network"
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Problem: {result['problem']}")
    print(f"{'='*60}")
    print(f"\nAnalysis:\n{result['analysis']}")
    print(f"\nDiagnostic steps taken: {result['steps_taken']}")
```

### Step 3: Understanding What Happens When You Run It

Let's trace through exactly what happens when you run this agent:

```
User: "Users on VLAN 20 (10.2.2.0/24) cannot access the network"
```

**Agent's internal reasoning** (what you'll see with `verbose=True`):

```
====================================
STEP 1: Initial Analysis
====================================
Agent thinks:
"Users on VLAN 20 can't access the network. VLAN 20 uses subnet 10.2.2.0/24.
I should first check if the interface serving that VLAN is up."

Agent decides to act:
TOOL: show_command
ARGS: {"command": "show ip interface brief"}

====================================
STEP 2: Analyze Interface Status
====================================
Tool returns:
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     10.1.1.1        YES manual up                    up
GigabitEthernet0/1     10.2.2.1        YES manual up                    down  ← PROBLEM!
GigabitEthernet0/2     192.168.1.1     YES manual up                    up
Loopback0              1.1.1.1         YES manual up                    up

Agent analyzes:
"GigabitEthernet0/1 has IP 10.2.2.1 (the VLAN 20 gateway).
Status is 'up' but Protocol is 'down'.
This means:
  - Interface is administratively enabled (up)
  - But line protocol is down (no Layer 2 connectivity)
  
This is a Layer 2 problem. I need more details."

====================================
STEP 3: Get Detailed Interface Info
====================================
Agent decides to act:
TOOL: show_command
ARGS: {"command": "show interface gigabitethernet0/1"}

Tool returns:
GigabitEthernet0/1 is up, line protocol is down
  ...
  Last input never, output 00:00:01, output hang never  ← KEY INFO
  ...
  0 packets input, 0 bytes  ← NO INPUT TRAFFIC
  13 packets output, 1234 bytes

Agent analyzes:
"Key findings:
1. Last input: NEVER - no packets have EVER been received
2. 0 packets input - confirms no inbound traffic
3. 13 packets output - outbound traffic IS being sent

This definitively proves a Layer 1/2 issue:
  - Cable could be unplugged
  - Remote switch port could be disabled
  - Duplex mismatch (though less likely with 'never' seeing traffic)
  - Wrong VLAN on switch port"

====================================
STEP 4: Final Conclusion
====================================
Agent provides analysis:

ROOT CAUSE IDENTIFIED:
GigabitEthernet0/1 line protocol is down despite being administratively up.
The interface has NEVER received any input traffic.

This indicates a Layer 1 or Layer 2 connectivity failure.

Most likely causes (in order of probability):
1. Physical cable is unplugged or faulty
2. Remote switchport is administratively down
3. Remote switchport is in wrong VLAN
4. Duplex/speed mismatch (less likely given 'never' seen traffic)

RECOMMENDED ACTIONS:
1. Physically check cable connection on both ends
2. Check remote switch configuration:
   - show cdp neighbors detail (to identify connected switch)
   - Verify switchport is enabled: show interface status
   - Verify switchport VLAN: show interface trunk/switchport
3. Check for physical errors:
   - show interface gigabitethernet0/1 | include error
4. If duplex mismatch suspected:
   interface gigabitethernet0/1
    duplex auto
    speed auto

IMPACT:
All users on VLAN 20 (10.2.2.0/24) have no network access because
their default gateway (10.2.2.1) is unreachable.

Steps taken: 2 diagnostic commands
```

### What Makes This Powerful?

1. **Natural language input** - You describe the problem in plain English
2. **Autonomous investigation** - Agent decides which commands to run
3. **Systematic analysis** - Follows OSI model (Layer 2 before Layer 3)
4. **Actionable output** - Specific commands to fix the issue
5. **Fast** - 2 commands, 30 seconds vs. 30 minutes of manual work

### Key Limitations of This Basic Agent

While impressive, this basic agent has limitations we'll address in the next sections:

1. **No structured methodology** - Investigation is somewhat ad-hoc
2. **No conversation** - Can't ask follow-up questions
3. **Limited safety** - Basic command validation only
4. **No planning** - Reacts to each result without long-term strategy

Let's fix these issues in the next sections.

---

## Section 3: Multi-Stage Troubleshooting with Structured Planning

The basic agent we built works, but it's reactive—it responds to each command output without a long-term plan. Real network troubleshooting follows a **structured methodology** based on the OSI model.

Let's build a **multi-stage troubleshooting system** that works like an experienced network engineer:

1. **Plan** the investigation strategy
2. **Execute** diagnostic commands for that stage
3. **Analyze** the results
4. **Decide** the next stage based on findings
5. **Repeat** until root cause found

### Why Multi-Stage Matters

**Problem with reactive agents:**
```
Agent: "Let me check interfaces"
[runs show ip interface brief]
Agent: "Hmm, BGP might be involved"
[runs show ip bgp]
Agent: "Actually, let me check routing"
[runs show ip route]
```
→ Inefficient, jumps around without clear strategy

**Multi-stage approach:**
```
STAGE 1 - Identify OSI Layer:
  Hypothesis: "Problem could be L1, L2, or L3"
  Commands: show ip interface brief
  Finding: "Interface up/down → L2 issue"

STAGE 2 - Investigate Layer 2:
  Hypothesis: "Physical connectivity problem"
  Commands: show interface gi0/1, show cdp neighbors
  Finding: "No CDP neighbor, no input traffic → cable issue"

STAGE 3 - Root Cause Found:
  Conclusion: "Cable unplugged or faulty"
```
→ Systematic, follows proven methodology

### The Multi-Stage Architecture

We'll build a system with:

1. **DiagnosticStage Enum** - Defines possible investigation stages
2. **DiagnosticPlan Model** - Structured plan for each stage
3. **MultiStageTroubleshooter** - Orchestrates the investigation

Here's the complete implementation:

```python
# multi_stage_agent.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum


# ============================================
# PART 1: Define Investigation Stages
# ============================================

class DiagnosticStage(str, Enum):
    """
    Possible stages in network troubleshooting.
    Based on OSI model bottom-up approach.
    """
    IDENTIFY_LAYER = "identify_layer"      # Which OSI layer is affected?
    CHECK_PHYSICAL = "check_physical"      # Layer 1: Cables, hardware
    CHECK_DATA_LINK = "check_data_link"    # Layer 2: Switching, VLANs, MAC
    CHECK_NETWORK = "check_network"        # Layer 3: IP, routing
    CHECK_TRANSPORT = "check_transport"    # Layer 4: TCP/UDP, ports
    CHECK_ROUTING = "check_routing"        # Routing protocols (BGP, OSPF)
    CHECK_SERVICES = "check_services"      # Application layer services
    ROOT_CAUSE_FOUND = "root_cause_found"  # Investigation complete


# ============================================
# PART 2: Define Diagnostic Plan Structure
# ============================================

class DiagnosticPlan(BaseModel):
    """
    Structured plan for a single diagnostic stage.
    
    The LLM generates this plan, which tells us:
    - What stage we're in
    - What we think the problem is
    - Which commands to run
    - Why we're running them
    """
    stage: DiagnosticStage = Field(
        description="Current diagnostic stage in the investigation"
    )
    hypothesis: str = Field(
        description="What we think the problem is at this stage"
    )
    commands_to_run: List[str] = Field(
        description="List of diagnostic commands to execute"
    )
    reasoning: str = Field(
        description="Why these commands will help test the hypothesis"
    )


# ============================================
# PART 3: Multi-Stage Troubleshooter
# ============================================

class MultiStageTroubleshooter:
    """
    Structured multi-stage network troubleshooting system.
    
    This agent follows a disciplined approach:
    1. Plan each stage based on OSI model
    2. Execute diagnostic commands
    3. Analyze results
    4. Plan next stage based on findings
    """

    def __init__(self, api_key: str):
        """Initialize the multi-stage troubleshooter."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.0  # Deterministic for troubleshooting
        )

        # JSON parser for structured outputs
        self.parser = JsonOutputParser(pydantic_object=DiagnosticPlan)

    def plan_next_stage(
        self,
        symptom: str,
        previous_findings: List[Dict] = None
    ) -> DiagnosticPlan:
        """
        Plan the next diagnostic stage based on symptom and previous findings.
        
        Args:
            symptom: Original problem description
            previous_findings: Results from previous stages
            
        Returns:
            DiagnosticPlan with stage, hypothesis, commands, and reasoning
        """
        
        # Format previous findings for context
        if previous_findings:
            findings_text = "\n\n".join([
                f"Stage {i+1}: {f['stage']}\n"
                f"Hypothesis: {f['hypothesis']}\n"
                f"Commands run: {', '.join(f['commands'])}\n"
                f"Key findings: {f['results'][:300]}..."
                for i, f in enumerate(previous_findings)
            ])
        else:
            findings_text = "None yet - this is the first stage"

        # Create prompt for the LLM
        prompt = ChatPromptTemplate.from_template("""
You are planning the next stage of a systematic network troubleshooting investigation.

ORIGINAL SYMPTOM:
{symptom}

PREVIOUS FINDINGS:
{findings}

Your task: Plan the next diagnostic stage using OSI model methodology.

INSTRUCTIONS:
1. Analyze what we know so far
2. Determine which OSI layer to investigate next (or if root cause is found)
3. Form a hypothesis about what might be wrong
4. Choose specific diagnostic commands to test that hypothesis
5. Explain why those commands will help

Available diagnostic stages:
- identify_layer: Determine which OSI layer is affected
- check_physical: Layer 1 - cables, hardware, link status
- check_data_link: Layer 2 - switching, VLANs, MAC addresses, STP
- check_network: Layer 3 - IP addressing, subnetting, ARP
- check_routing: Routing protocols - BGP, OSPF, static routes
- check_transport: Layer 4 - TCP/UDP, port connectivity
- check_services: Application layer - DNS, DHCP, services
- root_cause_found: Root cause identified, ready for fix

Return your plan as JSON matching this schema:
{format_instructions}

Think step-by-step, be systematic, and follow the OSI model.

JSON plan:""")

        # Get LLM's plan
        response = self.llm.invoke(
            prompt.format(
                symptom=symptom,
                findings=findings_text,
                format_instructions=self.parser.get_format_instructions()
            )
        )

        # Parse JSON response into DiagnosticPlan object
        import json
        plan_data = json.loads(response.content)
        return DiagnosticPlan(**plan_data)

    def execute_stage(
        self,
        plan: DiagnosticPlan,
        show_command_tool
    ) -> Dict:
        """
        Execute all commands in a diagnostic stage.
        
        Args:
            plan: The diagnostic plan to execute
            show_command_tool: Tool for running show commands
            
        Returns:
            Dictionary with stage results
        """
        results = []

        print(f"\n🔍 Executing {len(plan.commands_to_run)} commands...")
        
        for command in plan.commands_to_run:
            print(f"  → {command}")
            output = show_command_tool._run(command)
            results.append({
                "command": command,
                "output": output
            })

        # Combine all command outputs
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
        """
        Analyze all stage results and provide comprehensive root cause analysis.
        
        Args:
            symptom: Original problem description
            stage_results: All stage results from the investigation
            
        Returns:
            Detailed root cause analysis with fix recommendations
        """
        
        # Format all results
        results_text = "\n\n".join([
            f"{'='*60}\n"
            f"STAGE: {r['stage']}\n"
            f"HYPOTHESIS: {r['hypothesis']}\n"
            f"{'='*60}\n"
            f"{r['results']}"
            for r in stage_results
        ])

        # Create analysis prompt
        prompt = ChatPromptTemplate.from_template("""
You are providing final root cause analysis after completing a network troubleshooting investigation.

ORIGINAL SYMPTOM:
{symptom}

COMPLETE INVESTIGATION RESULTS:
{results}

Provide a comprehensive root cause analysis with:

1. **ROOT CAUSE**
   What is actually wrong? Be specific and definitive.

2. **EVIDENCE**
   Which command outputs prove this root cause? Quote specific lines.

3. **IMPACT**
   Why are users experiencing the reported symptom?
   How does the root cause lead to the symptom?

4. **FIX**
   Exact configuration commands to resolve the issue.
   Include full command syntax.

5. **PREVENTION**
   How can this be prevented in the future?
   Monitoring, automation, or best practices?

6. **VERIFICATION**
   After applying the fix, which commands should be run to verify it worked?

Format your analysis clearly and actionably. Network engineers will use this to
fix the problem, so be precise and thorough.""")

        # Get comprehensive analysis from LLM
        response = self.llm.invoke(
            prompt.format(
                symptom=symptom,
                results=results_text
            )
        )

        return response.content

    def troubleshoot_full(
        self,
        symptom: str,
        max_stages: int = 5
    ) -> Dict:
        """
        Perform complete multi-stage troubleshooting investigation.
        
        Args:
            symptom: Natural language problem description
            max_stages: Maximum number of stages (safety limit)
            
        Returns:
            Complete investigation results with analysis
        """
        from troubleshooting_tools import ShowCommandTool

        # Initialize show command tool
        tool = ShowCommandTool()
        previous_findings = []

        print(f"\n{'='*70}")
        print(f"MULTI-STAGE TROUBLESHOOTING INVESTIGATION")
        print(f"{'='*70}")
        print(f"\nSymptom: {symptom}\n")

        # Execute stages until root cause found or max stages reached
        for stage_num in range(max_stages):
            print(f"\n{'#'*70}")
            print(f"# STAGE {stage_num + 1}")
            print(f"{'#'*70}")

            # PLAN: Determine next stage
            print(f"\n📋 Planning next stage...")
            plan = self.plan_next_stage(symptom, previous_findings)

            print(f"\n✓ Stage planned:")
            print(f"  Stage type: {plan.stage}")
            print(f"  Hypothesis: {plan.hypothesis}")
            print(f"  Commands to run: {len(plan.commands_to_run)}")
            print(f"  Reasoning: {plan.reasoning}")

            # Check if we've found the root cause
            if plan.stage == DiagnosticStage.ROOT_CAUSE_FOUND:
                print(f"\n🎯 ROOT CAUSE IDENTIFIED!")
                break

            # EXECUTE: Run diagnostic commands
            results = self.execute_stage(plan, tool)
            previous_findings.append(results)

            print(f"\n✓ Stage {stage_num + 1} complete")

        # ANALYZE: Provide final root cause analysis
        print(f"\n{'='*70}")
        print(f"FINAL ROOT CAUSE ANALYSIS")
        print(f"{'='*70}\n")

        analysis = self.analyze_results(symptom, previous_findings)
        print(analysis)

        return {
            "symptom": symptom,
            "stages_completed": len(previous_findings),
            "stage_findings": previous_findings,
            "final_analysis": analysis
        }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Create multi-stage troubleshooter
    troubleshooter = MultiStageTroubleshooter(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Scenario: Intermittent connectivity issues
    result = troubleshooter.troubleshoot_full(
        symptom="Users at branch office report intermittent internet connectivity. "
                "Sometimes it works, sometimes it doesn't.",
        max_stages=4
    )

    # Results are printed during execution
    # The result dict contains all findings for logging/review
```

### Example Output: Multi-Stage Investigation

Here's what the output looks like for a real troubleshooting scenario:

```
======================================================================
MULTI-STAGE TROUBLESHOOTING INVESTIGATION
======================================================================

Symptom: Users at branch office report intermittent internet connectivity

######################################################################
# STAGE 1
######################################################################

📋 Planning next stage...

✓ Stage planned:
  Stage type: identify_layer
  Hypothesis: Need to determine if this is a Layer 1/2 (local) or Layer 3 (routing/WAN) issue
  Commands to run: 2
  Reasoning: Check local interface status and WAN link status to narrow down the problem domain

🔍 Executing 2 commands...
  → show ip interface brief
  → show ip route

✓ Stage 1 complete

######################################################################
# STAGE 2
######################################################################

📋 Planning next stage...

✓ Stage planned:
  Stage type: check_routing
  Hypothesis: Intermittent connectivity suggests routing instability, possibly flapping BGP session
  Commands to run: 2
  Reasoning: BGP flapping would cause routes to appear/disappear, matching the symptom

🔍 Executing 2 commands...
  → show ip bgp summary
  → show ip bgp neighbors 203.0.113.1

✓ Stage 2 complete

######################################################################
# STAGE 3
######################################################################

📋 Planning next stage...

🎯 ROOT CAUSE IDENTIFIED!

======================================================================
FINAL ROOT CAUSE ANALYSIS
======================================================================

1. ROOT CAUSE
BGP neighbor 203.0.113.1 (AS 65003) is in "Idle" state and has never established
a session. This is NOT intermittent flapping—it has never worked.

The "intermittent" connectivity is likely caused by:
- Primary path (BGP) is down
- Backup static route occasionally takes over
- When backup route is congested/unavailable, connectivity fails

2. EVIDENCE
From "show ip bgp summary":
  203.0.113.1     4 65003       0       0        0    0    0 never    Idle
                                                            ^^^^^^    ^^^^
                                                         Never up   Idle state

This neighbor has:
- 0 messages received
- 0 messages sent
- Never been up
- Currently in Idle state (not even trying to connect)

3. IMPACT
Users experience intermittent connectivity because:
- Primary BGP path (intended route) is down
- Traffic fails over to backup static route
- When backup route is congested or the path is suboptimal, connectivity degrades
- This creates the "intermittent" behavior

4. FIX
Step 1: Diagnose why BGP won't establish
!
router bgp 65001
 neighbor 203.0.113.1 description ISP_PRIMARY
 neighbor 203.0.113.1 remote-as 65003
 neighbor 203.0.113.1 update-source GigabitEthernet0/2
 neighbor 203.0.113.1 password bgp_secret_key
!

Possible issues to check:
a) IP connectivity to neighbor:
   ping 203.0.113.1 source GigabitEthernet0/2

b) Access-list blocking TCP 179:
   show access-lists
   (Ensure outbound ACL permits TCP 179 to 203.0.113.1)

c) Firewall blocking BGP:
   Check any firewalls between router and ISP

d) Wrong BGP password:
   Verify password with ISP

e) ISP side not configured:
   Contact ISP to verify their BGP config

Step 2: Once BGP is up, verify routes are received:
show ip bgp summary
show ip bgp
show ip route bgp

5. PREVENTION
- Monitor BGP neighbor states (send alert if neighbor goes down)
- Implement BFD (Bidirectional Forwarding Detection) for fast failover
- Configure syslog for BGP state changes
- Document ISP contact procedures for BGP issues

6. VERIFICATION
After fix is applied:
# Verify BGP session is established
show ip bgp summary
# Should show state "Established" with PfxRcd > 0

# Verify routes are being received
show ip bgp
# Should show routes learned from 203.0.113.1

# Verify routes are in routing table
show ip route bgp
# Should show BGP routes installed

# Test connectivity
ping 8.8.8.8
# Should work reliably now

Stages completed: 2
```

### Why This Approach Is Better

**Compared to the basic agent:**

| Feature | Basic Agent | Multi-Stage Agent |
|---------|-------------|-------------------|
| **Methodology** | Reactive | Structured (OSI model) |
| **Planning** | None | Plans each stage |
| **Explanation** | Limited | Hypothesis + reasoning |
| **Analysis** | Basic | Comprehensive (6 sections) |
| **Audit trail** | Minimal | Complete stage history |
| **Efficiency** | Medium | High (systematic) |

**Real-world benefits:**

1. **Systematic investigation** - Follows proven troubleshooting methodology
2. **Clear logic** - Each stage has explicit hypothesis and reasoning
3. **Complete documentation** - Full audit trail of investigation
4. **Comprehensive fix** - Not just "what" but "how" and "why"
5. **Prevention guidance** - Learn from the incident

### When to Use Multi-Stage vs Basic Agent

**Use Basic Agent when:**
- Quick, simple issues ("Is this interface up?")
- Interactive troubleshooting with human
- Learning/experimentation

**Use Multi-Stage Agent when:**
- Complex, multi-layer issues
- Need documented investigation process
- Production incident response
- Training junior engineers (they can follow the logic)

---

## Section 4: Conversational Troubleshooting Agent with Memory

The multi-stage agent is great for structured investigations, but real troubleshooting is often **conversational**. You might:

- Ask a follow-up question: "What caused that interface to go down?"
- Request more detail: "Can you check if BGP is working?"
- Pivot mid-investigation: "Actually, let's check the firewall instead"

To support this, we need an agent with **memory**—it remembers the conversation and previous findings.

### The Problem: Stateless Agents

Our basic agent has no memory:

```python
agent.troubleshoot("Users can't reach 10.2.2.0/24")
# Agent investigates: "Gi0/1 is down"

agent.troubleshoot("Why is that interface down?")
# Agent has NO IDEA what "that interface" refers to!
# It starts over from scratch
```

### The Solution: Conversation Memory

```python
agent = ConversationalTroubleshootingAgent(api_key="...")

# First message
agent.chat("Users can't reach 10.2.2.0/24")
→ Agent: "Let me check... Gi0/1 is down (line protocol down)"

# Follow-up (agent remembers!)
agent.chat("Why is the line protocol down?")
→ Agent: "Based on my earlier check of Gi0/1, I saw 'Last input never' 
          which indicates no traffic received. Let me get more details..."

# Another follow-up
agent.chat("How do I fix it?")
→ Agent: "Given that Gi0/1 has never received input, the fix is..."
```

The agent maintains **context** across messages, just like chatting with a human engineer.

### Implementation: Conversational Agent with Memory

```python
# conversational_agent.py
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from troubleshooting_tools import get_troubleshooting_tools


class ConversationalTroubleshootingAgent:
    """
    Network troubleshooting agent with conversation memory.
    
    This agent:
    - Remembers all previous messages and findings
    - Understands follow-up questions and context
    - Maintains investigation state across multiple interactions
    - Provides natural conversational experience
    """

    def __init__(self, api_key: str):
        """Initialize conversational agent with memory."""
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.2  # Slightly higher for more natural conversation
        )

        # ==========================================
        # KEY ADDITION: Conversation Memory
        # ==========================================
        # This stores all messages and agent actions
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,  # Return as message objects (not strings)
            output_key="output"    # Which key in the result to remember
        )

        # Get troubleshooting tools
        self.tools = get_troubleshooting_tools()

        # ==========================================
        # Prompt with Memory Placeholder
        # ==========================================
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network troubleshooting assistant having a conversation with a network engineer.

IMPORTANT: Remember the context of previous messages. If they ask a follow-up question, 
reference earlier findings and diagnostic work.

Available tools:
- show_command: Run Cisco show commands
- ping: Test connectivity
- traceroute: Trace network path

Your conversational guidelines:
1. **Remember context**: Reference earlier findings when relevant
2. **Answer naturally**: Have a conversation, not just dump data
3. **Ask clarifying questions** if the request is ambiguous
4. **Provide value**: Don't just run commands—analyze and explain
5. **Be concise**: The engineer is busy; get to the point

Use tools when you need diagnostic information. Otherwise, answer based on your knowledge
and the conversation history."""),
            
            # This is where the conversation history gets injected
            MessagesPlaceholder("chat_history"),
            
            # Current user message
            ("human", "{input}"),
            
            # Space for agent's tool calls and thoughts
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create agent with memory
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        # Agent executor with memory integration
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,  # <-- Memory integrated here
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def chat(self, message: str) -> str:
        """
        Chat with the agent (remembers conversation).
        
        Args:
            message: User message or question
            
        Returns:
            Agent's response
        """
        result = self.agent_executor.invoke({"input": message})
        return result['output']

    def clear_memory(self):
        """Clear conversation history (start fresh)."""
        self.memory.clear()

    def get_conversation_history(self) -> list:
        """Get all messages in the conversation."""
        return self.memory.buffer_as_messages


# ============================================
# Example: Natural Troubleshooting Conversation
# ============================================

if __name__ == "__main__":
    agent = ConversationalTroubleshootingAgent(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("="*60)
    print("Conversational Troubleshooting Agent")
    print("="*60)
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

    while True:
        # Get user input
        user_message = input("\nYou: ")
        
        if user_message.lower() == 'quit':
            break
        
        if user_message.lower() == 'clear':
            agent.clear_memory()
            print("✓ Conversation history cleared\n")
            continue

        # Get agent response
        response = agent.chat(user_message)
        print(f"\nAgent: {response}")
```

### Example Conversation Flow

Here's what a real conversation with this agent looks like:

```
================================================================
Conversational Troubleshooting Agent
================================================================

You: Users on VLAN 20 can't access the internet

Agent: Let me check the interface status for VLAN 20...

[Running: show_command("show ip interface brief")]

I can see that GigabitEthernet0/1 (which has IP 10.2.2.1 - likely your VLAN 20 gateway) 
has a problem: the interface is administratively up, but the line protocol is down.

This means:
- The interface is enabled in configuration
- But there's no Layer 2 connectivity

This is preventing VLAN 20 users from reaching their gateway.

Would you like me to investigate why the line protocol is down?

----------------------------------------------------------------

You: Yes, why is it down?

Agent: Let me get detailed information about that interface...

[Running: show_command("show interface gigabitethernet0/1")]

The key finding here is: "Last input never" - this interface has NEVER received any 
traffic since the counters were last cleared.

Combined with:
- 0 packets input
- 13 packets output
- No input errors or CRC errors

This tells us the interface is sending traffic but receiving absolutely nothing. 
This is a clear Layer 1 or Layer 2 issue, most likely:

1. Physical cable is unplugged or damaged
2. Remote switchport is administratively down
3. Remote switchport is in the wrong VLAN

The lack of ANY received traffic (including keepalives) points to a complete break 
in physical or data-link layer connectivity.

----------------------------------------------------------------

You: How do I fix it?

Agent: Based on what we found (Gi0/1 never receiving traffic), here's how to fix it:

**Immediate Steps:**

1. **Physical verification**
   - Check both ends of the cable connected to Gi0/1
   - Verify the cable is properly seated
   - Look for physical damage to the cable

2. **Check remote switch**
   First, identify the remote switch:
   ```
   show cdp neighbors gigabitethernet0/1
   ```
   
   Then on the remote switch, verify:
   - Is the port enabled? (not in shutdown state)
   - Is it in the correct VLAN?
   - Any port security violations?

3. **Check for interface errors**
   ```
   show interface gigabitethernet0/1 | include error|CRC|collision
   ```

**If it's a cable issue:**
- Replace the cable
- Test with a known-good cable

**If it's a switchport config issue:**
```
! On the remote switch
interface [port]
 no shutdown
 switchport mode access
 switchport access vlan 20
```

**Verification after fix:**
```
show ip interface brief | include GigabitEthernet0/1
```
Should show: up/up (not up/down)

```
show interface gigabitethernet0/1 | include packets input
```
Should show increasing input packet count

**Test user connectivity:**
```
ping 10.2.2.10  ! (ping a user device on VLAN 20)
```

Would you like me to help with any of these steps?

----------------------------------------------------------------

You: What if changing the cable doesn't work?

Agent: Good question. If you've verified the cable is good and still have the 
line protocol down, here's the next level of troubleshooting:

**1. Check for speed/duplex mismatch**
Even though we didn't see errors, mismatches can sometimes cause "Last input never".

On your router:
```
interface gigabitethernet0/1
 duplex auto
 speed auto
```

Check the remote switch's settings match (both should be auto-negotiate).

**2. Verify the remote port is actually connected**
If CDP is enabled but you see no CDP neighbor on Gi0/1, that's a red flag:
```
show cdp neighbors gigabitethernet0/1
```

If no CDP neighbor appears, either:
- The cable isn't connected on the other end
- The remote port is shutdown
- CDP is disabled on one side

**3. Check if port security is blocking**
On the remote switch:
```
show port-security interface [interface]
```

If you see "Security Violation Count" > 0, the port may be in err-disabled:
```
show interface status | include err-disabled
```

To recover:
```
interface [interface]
 shutdown
 no shutdown
```

**4. Try a different port**
If possible, connect to a different port on the remote switch to isolate whether 
it's the port itself that's faulty.

**5. Check STP (Spanning Tree)**
Sometimes STP can block a port. On the remote switch:
```
show spanning-tree interface [interface]
```

Should show "FWD" (forwarding), not "BLK" (blocking).

Since Gi0/1 is showing ZERO received packets (not even spanning-tree BPDUs), I'd 
focus on #1 and #2 first—this really looks like a physical connectivity or remote 
port shutdown issue.

Want me to walk you through any of these checks specifically?
```

### Key Features of Conversational Agent

**1. Context Awareness**
- "why is it down?" → Agent knows "it" refers to Gi0/1
- "How do I fix it?" → Agent remembers the root cause found earlier

**2. Natural Flow**
- Doesn't repeat information already discussed
- References earlier findings naturally
- Asks clarifying questions

**3. Progressive Detail**
- First response: High-level summary
- Follow-up: More technical detail
- Further questions: Advanced troubleshooting

**4. Guidance**
- Not just data dumps
- Explains what to do with the information
- Provides step-by-step procedures

### When to Use Conversational Agent

**Best for:**
- ✅ Interactive troubleshooting sessions
- ✅ Training junior engineers (they can ask questions)
- ✅ Exploratory investigations (don't know exactly what's wrong)
- ✅ Learning mode (explain as you go)

**Not ideal for:**
- ❌ Automated incident response (no human in loop)
- ❌ Batch diagnostics (checking 100 devices)
- ❌ Strict runbook following (multi-stage is better)

---

## Section 5: Production Safety Features

We've built powerful agents, but with power comes risk. Before deploying to production, you MUST implement safety features.

**What could go wrong without safety?**
```
Agent receives: "Fix the BGP issue"
Agent thinks: "I should restart BGP"
Agent executes: clear ip bgp *
→ ALL BGP sessions reset
→ Network outage
→ You're looking for a new job
```

Let's implement layers of safety to prevent disasters.

### Safety Layer 1: Command Validation

**Whitelist only safe commands** and block anything dangerous:

```python
# safe_agent.py
from langchain.tools import BaseTool
from typing import List, Tuple
import re


class SafeCommandValidator:
    """
    Validate commands before execution.
    
    This is your first line of defense against dangerous operations.
    Only allow read-only commands by default.
    """

    # Whitelist: Only these command prefixes are allowed
    SAFE_COMMANDS = [
        "show",           # Cisco show commands
        "ping",           # Connectivity tests
        "traceroute",     # Path tracing
        "display",        # Huawei show equivalent
        "get",            # Fortinet show equivalent
        "describe",       # Some vendors
    ]

    # Blacklist: Dangerous patterns that should NEVER be allowed
    DANGEROUS_PATTERNS = [
        r'\bno\b',                      # Removes config: "no ip address"
        r'\bshutdown\b',                # Disables interfaces
        r'\breload\b',                  # Reboots device
        r'\bwrite\s+erase\b',           # Erases config
        r'\bformat\b',                  # Formats flash/disk
        r'\bdelete\b',                  # Deletes files
        r'\bclear\s+(line|ip\s+bgp)\b', # Clears sessions/connections
        r'\bconfigure\b',               # Enters config mode
        r'\bconf\s+t\b',                # Config mode shortcut
        r'\bexit\b',                    # Might exit critical mode
        r'\bend\b',                     # Might end session
        r'\bwrite\s+(memory|network)\b',# Saves config (could persist bad state)
    ]

    @classmethod
    def is_safe(cls, command: str) -> Tuple[bool, str]:
        """
        Validate if a command is safe to execute.
        
        Args:
            command: The command to validate
            
        Returns:
            (is_safe: bool, reason: str)
            If unsafe, reason explains why it was blocked
        """
        command_lower = command.lower().strip()

        # Empty command
        if not command_lower:
            return False, "Empty command"

        # Check if starts with safe command prefix
        is_safe_prefix = any(
            command_lower.startswith(safe_cmd)
            for safe_cmd in cls.SAFE_COMMANDS
        )

        if not is_safe_prefix:
            return False, (
                f"Command must start with one of: {', '.join(cls.SAFE_COMMANDS)}\n"
                f"Got: {command[:50]}"
            )

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return False, f"Command contains dangerous pattern: {pattern}"

        # Passed all checks
        return True, "Command is safe"


class SafeShowCommandTool(BaseTool):
    """
    Show command tool with safety validation.
    
    This wraps your normal ShowCommandTool with validation.
    """
    name: str = "show_command"
    description: str = "Execute safe read-only show commands (validated for safety)"

    def _run(self, command: str) -> str:
        """Run command only if it passes safety checks."""
        
        # VALIDATE FIRST
        is_safe, reason = SafeCommandValidator.is_safe(command)

        if not is_safe:
            # Log the blocked attempt
            import logging
            logging.warning(f"BLOCKED COMMAND: {command} | Reason: {reason}")
            
            # Return error to the agent
            return f"ERROR: Command blocked for safety.\nReason: {reason}"

        # Command is safe - execute it
        # (Here you'd use your actual Netmiko connection)
        return self._execute_on_device(command)

    def _execute_on_device(self, command: str) -> str:
        """Execute validated command on device."""
        # Your Netmiko code here
        pass


# ============================================
# Example: What Gets Blocked?
# ============================================

if __name__ == "__main__":
    test_commands = [
        # Safe commands (should pass)
        "show ip interface brief",
        "show running-config",
        "show ip bgp summary",
        "ping 8.8.8.8",
        "traceroute google.com",
        
        # Dangerous commands (should block)
        "configure terminal",
        "no shutdown",
        "reload in 5",
        "write erase",
        "clear ip bgp *",
        "show run | include no",  # Contains "no"
        "delete flash:config.txt",
    ]

    print("Command Safety Validation Tests\n")
    print("="*70)
    
    for cmd in test_commands:
        is_safe, reason = SafeCommandValidator.is_safe(cmd)
        status = "✓ SAFE" if is_safe else "✗ BLOCKED"
        print(f"\n{status}: {cmd}")
        if not is_safe:
            print(f"  Reason: {reason}")

"""
Output:
======================================================================
Command Safety Validation Tests

✓ SAFE: show ip interface brief

✓ SAFE: show running-config

✓ SAFE: show ip bgp summary

✓ SAFE: ping 8.8.8.8

✓ SAFE: traceroute google.com

✗ BLOCKED: configure terminal
  Reason: Command must start with one of: show, ping, traceroute, display, get

✗ BLOCKED: no shutdown
  Reason: Command must start with one of: show, ping, traceroute, display, get

✗ BLOCKED: reload in 5
  Reason: Command contains dangerous pattern: \breload\b

✗ BLOCKED: write erase
  Reason: Command contains dangerous pattern: \bwrite\s+erase\b

✗ BLOCKED: clear ip bgp *
  Reason: Command contains dangerous pattern: \bclear\s+(line|ip\s+bgp)\b

✗ BLOCKED: show run | include no
  Reason: Command contains dangerous pattern: \bno\b

✗ BLOCKED: delete flash:config.txt
  Reason: Command contains dangerous pattern: \bdelete\b
"""
```

### Safety Layer 2: Human-in-the-Loop Approval

For operations that change state (even if "safe"), require human approval:

```python
# approval_agent.py
from langchain.tools import BaseTool
from datetime import datetime


class ApprovalRequiredTool(BaseTool):
    """
    Tool that requires explicit human approval before execution.
    
    Use this for any operation that modifies state:
    - Configuration changes
    - Service restarts
    - Route changes
    - Firewall rule modifications
    """
    
    name: str = "apply_config"
    description: str = """
    Apply configuration changes to network devices.
    
    IMPORTANT: This requires human approval before execution.
    Use this when you've identified a fix and want to apply it.
    """

    def _run(self, config: str, target_device: str = "router") -> str:
        """
        Request approval and apply configuration if approved.
        
        Args:
            config: Configuration commands to apply
            target_device: Device to apply config to
            
        Returns:
            Success/failure message
        """
        
        # Display proposed changes
        print(f"\n{'='*70}")
        print(f"⚠️  APPROVAL REQUIRED")
        print(f"{'='*70}")
        print(f"\nTarget Device: {target_device}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nProposed Configuration:")
        print(f"{'-'*70}")
        print(config)
        print(f"{'-'*70}\n")

        # Request approval (in production, this could be Slack/Teams/PagerDuty)
        approval = input("Apply this configuration? (yes/no): ").strip().lower()

        if approval == 'yes':
            # Log the approval
            print(f"\n✓ Approved by operator at {datetime.now()}")
            
            # Apply the configuration
            # (In production: use Netmiko to apply)
            result = self._apply_to_device(config, target_device)
            
            # Log the result
            print(f"\n✓ Configuration applied successfully")
            return f"Configuration applied to {target_device}:\n{result}"
        
        else:
            # Log the rejection
            print(f"\n✗ Configuration rejected by operator")
            return f"Configuration change REJECTED by operator. No changes made."

    def _apply_to_device(self, config: str, device: str) -> str:
        """Apply configuration to device (production: use Netmiko)."""
        
        # Production code:
        # from netmiko import ConnectHandler
        # 
        # device_conn = ConnectHandler(**device_params)
        # output = device_conn.send_config_set(config.split('\n'))
        # device_conn.save_config()
        # return output
        
        # Demo:
        return f"[Simulated] Applied config to {device}"


# ============================================
# Example: Agent Suggests Fix with Approval
# ============================================

"""
Troubleshooting conversation with approval workflow:

You: Users can't reach VLAN 20

Agent: Let me check...
[runs diagnostics]

Agent: Found the issue: GigabitEthernet0/1 is administratively shut down.

The fix is to enable the interface. Would you like me to apply this fix?

You: Yes, fix it

Agent: [Prepares configuration]

======================================================================
⚠️  APPROVAL REQUIRED
======================================================================

Target Device: router-hq-01
Timestamp: 2024-01-15 14:32:10

Proposed Configuration:
----------------------------------------------------------------------
interface GigabitEthernet0/1
 no shutdown
----------------------------------------------------------------------

Apply this configuration? (yes/no): yes

✓ Approved by operator at 2024-01-15 14:32:15
✓ Configuration applied successfully

Agent: Configuration has been applied. Interface Gi0/1 is now enabled.
Please verify with: show ip interface brief | include Gi0/1
"""
```

### Safety Layer 3: Read-Only Mode (Default)

Make read-only the default; require explicit opt-in for changes:

```python
class TroubleshootingAgent:
    """Agent with read-only default mode."""
    
    def __init__(self, api_key: str, allow_changes: bool = False):
        """
        Initialize agent.
        
        Args:
            api_key: API key for LLM
            allow_changes: If False (default), only diagnostic tools available.
                          If True, configuration tools are enabled.
        """
        self.allow_changes = allow_changes
        
        # Base tools (always available)
        self.tools = [
            ShowCommandTool(),
            PingTool(),
            TraceRouteTool()
        ]
        
        # Add change tools only if explicitly enabled
        if allow_changes:
            self.tools.append(ApprovalRequiredTool())
            print("⚠️  WARNING: Configuration change tools are ENABLED")
        else:
            print("✓ Running in READ-ONLY mode (safe)")

# Usage:
# Read-only (safe for juniors, automation, exploration)
agent = TroubleshootingAgent(api_key="...", allow_changes=False)

# With changes allowed (for experienced engineers only)
agent = TroubleshootingAgent(api_key="...", allow_changes=True)
```

### Safety Layer 4: Audit Logging

Log EVERYTHING the agent does:

```python
import logging
import json
from datetime import datetime


class AuditLogger:
    """Log all agent actions for compliance and troubleshooting."""
    
    def __init__(self, log_file="agent_audit.log"):
        """Initialize audit logger."""
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self.logger = logging.getLogger("AgentAudit")

    def log_command(self, command: str, device: str, result: str, user: str):
        """Log a command execution."""
        self.logger.info(json.dumps({
            "type": "command_execution",
            "user": user,
            "device": device,
            "command": command,
            "result_preview": result[:200],
            "timestamp": datetime.now().isoformat()
        }))

    def log_blocked_command(self, command: str, reason: str, user: str):
        """Log a blocked command attempt."""
        self.logger.warning(json.dumps({
            "type": "command_blocked",
            "user": user,
            "command": command,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }))

    def log_config_change(self, config: str, device: str, approved: bool, user: str):
        """Log a configuration change request."""
        self.logger.info(json.dumps({
            "type": "config_change",
            "user": user,
            "device": device,
            "config": config,
            "approved": approved,
            "timestamp": datetime.now().isoformat()
        }))


# Integrate with tools:
class SafeShowCommandTool(BaseTool):
    """Tool with audit logging."""
    
    def __init__(self):
        super().__init__()
        self.audit = AuditLogger()

    def _run(self, command: str) -> str:
        is_safe, reason = SafeCommandValidator.is_safe(command)
        
        if not is_safe:
            # Log blocked attempt
            self.audit.log_blocked_command(
                command=command,
                reason=reason,
                user=self.get_current_user()
            )
            return f"ERROR: {reason}"
        
        # Execute and log
        result = self._execute(command)
        self.audit.log_command(
            command=command,
            device="router-01",
            result=result,
            user=self.get_current_user()
        )
        return result
```

### Safety Checklist for Production

Before deploying your troubleshooting agent to production:

- [ ] ✅ Command validation (whitelist + blacklist)
- [ ] ✅ Human approval for all config changes
- [ ] ✅ Read-only mode by default
- [ ] ✅ Comprehensive audit logging
- [ ] ✅ Rate limiting (max commands per minute)
- [ ] ✅ Session timeouts (auto-disconnect idle agents)
- [ ] ✅ Role-based access control (who can enable change mode?)
- [ ] ✅ Alert on suspicious behavior (100 commands in 1 minute?)
- [ ] ✅ Rollback capability (can you undo agent changes?)
- [ ] ✅ Testing in lab environment first

**Remember: An unsecured agent with network access is more dangerous than no automation at all.**

---

## What Can Go Wrong (And How to Prevent It)

Even with safety features, things can still go wrong. Here are common issues and solutions:

### Problem 1: Agent Runs in Loops

**Symptom:**
```
Agent: Checking interface...
Agent: Still checking interface...
Agent: Checking interface again...
[Repeats until max_iterations hit]
```

**Cause:** Agent doesn't realize it already has the information

**Solution:**
```python
# Add loop detection
class LoopDetector:
    def __init__(self, max_repeats=2):
        self.command_history = []
        self.max_repeats = max_repeats

    def is_loop(self, command):
        recent = self.command_history[-5:]  # Last 5 commands
        if recent.count(command) >= self.max_repeats:
            return True
        self.command_history.append(command)
        return False
```

### Problem 2: Agent Misinterprets Output

**Symptom:** Agent says "BGP is working fine" when it's actually down

**Cause:** Ambiguous output parsing

**Solution:**
```python
# Structured output parsing
from pydantic import BaseModel

class BGPStatus(BaseModel):
    neighbor: str
    state: str
    prefixes_received: int

# Force agent to return structured data
def parse_bgp_output(output: str) -> List[BGPStatus]:
    # Use LLM to extract structured data
    # Or use regex/TextFSM for known formats
    pass
```

### Problem 3: Expensive Token Usage

**Symptom:** $50 bill for troubleshooting a single issue

**Cause:** Agent makes 25 tool calls, each with large outputs

**Solution:**
```python
# Limit output size
def _run(self, command: str) -> str:
    output = device.send_command(command)
    
    # Truncate large outputs
    if len(output) > 5000:
        output = output[:5000] + f"\n\n[Output truncated at 5000 chars]"
    
    return output

# Set hard iteration limit
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=8,  # Hard stop at 8 actions
    max_execution_time=60  # Timeout after 60 seconds
)
```

### Problem 4: Context Window Exceeded

**Symptom:** Error: "Context length exceeded"

**Cause:** Long conversation + large command outputs fill the context window

**Solution:**
```python
# Summarize old messages
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Or use ConversationBufferWindowMemory (keep last N messages)
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=10,  # Keep last 10 messages
    memory_key="chat_history",
    return_messages=True
)
```

### Problem 5: Agent Gets Confused

**Symptom:** Agent gives up: "I don't know how to proceed"

**Cause:** Complex issue outside training data

**Solution:**
```python
# Give the agent an "ask for help" tool
class AskHumanTool(BaseTool):
    name: str = "ask_human_expert"
    description: str = "Ask a human expert when stuck or unsure"

    def _run(self, question: str) -> str:
        print(f"\n🤔 AGENT NEEDS HELP: {question}\n")
        return input("Expert response: ")

# Agent can now escalate:
# "I've checked interfaces and routing, but can't determine why 
#  packets are being dropped. Should I check firewall rules?"
```

---

## Key Takeaways

**What You've Learned:**

1. **Agent != Chatbot** - Agents take actions, not just answer questions
2. **Tools are the hands** - Give agents capabilities (show commands, ping, etc.)
3. **Memory enables conversation** - Context makes follow-ups natural
4. **Multi-stage = systematic** - Structured investigation beats random checking
5. **Safety is non-negotiable** - Validate, approve, log, and limit

**Real-World Impact:**

- **Time saved:** 2 minutes vs. 30 minutes for typical issues
- **Consistency:** Every investigation follows best practices
- **Documentation:** Complete audit trail of what was checked
- **Training:** Junior engineers learn by watching agent's logic
- **Availability:** 24/7 troubleshooting without waking humans

**Next Steps:**

- Chapter 21: Automated configuration generation
- Chapter 22: Log analysis with AI
- Chapter 23: Predictive failure detection
- Chapter 24: Self-healing networks

**The future of network operations isn't replacing engineers—it's augmenting them with AI agents that handle the repetitive, systematic work so humans can focus on architecture, strategy, and complex problem-solving.**

