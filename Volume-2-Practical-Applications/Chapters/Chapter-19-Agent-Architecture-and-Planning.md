# Chapter 19: Agent Architecture and Planning

## Introduction

An AI agent isn't just an LLM with access to functions. A real agent makes decisions, plans sequences of actions, handles failures, and adapts to changing conditions. In network operations, this distinction matters: a simple tool-calling system can run show commands, but an agent can diagnose a complex BGP routing loop by forming hypotheses, gathering evidence, and adjusting its investigation based on what it discovers.

This chapter builds agents from first principles. We'll start with simple tool-calling, add reasoning loops, implement planning, and end with multi-agent systems that coordinate specialized agents for complex operations.

**What you'll build**:
- V1: Tool-calling agent (Claude's native function calling)
- V2: ReAct agent (reasoning + acting interleaved)
- V3: Plan-Execute + Reflexion (planning and learning)
- V4: Multi-agent coordinator (specialized agents working together)

**Real-world scenarios**:
- Diagnose BGP neighbor down (V2 ReAct exploration)
- Deploy VLANs across 50 switches (V3 Plan-Execute)
- Full incident response workflow (V4 multi-agent)

**Prerequisites**: Chapters 4 (API Basics), 10 (API Integration), 15 (LangChain basics)

---

## Agent Design Philosophy

### The Spectrum of Autonomy

```
Simple                                                      Complex
  │                                                            │
  ▼                                                            ▼
Tool      ReAct        Plan-Execute    Reflexion    Multi-Agent
Call      Loop         Workflow        Learning     Coordination
  │         │              │              │              │
  ├─────────┼──────────────┼──────────────┼──────────────┤
  1-2       3-8            5-15           10-30          20-50
 tools     iterations     steps          retries        agent calls

Cost:    $0.05         $0.20          $0.40           $0.80          $1.50
Latency: 2s            8s             15s             30s            45s
Success: 95%           90%            85%             95%            98%
```

**Key insight**: Start simple. Use tool-calling for 80% of tasks. Add complexity only when needed.

### When to Use Each Pattern

**Tool-Calling** (V1):
- Query: "What's the BGP status on router1?"
- Known tools, clear mapping, no exploration needed
- 95% of production queries

**ReAct** (V2):
- Diagnosis: "Why is OSPF neighbor down?"
- Unknown problem space, need to explore
- Multiple hypotheses to test

**Plan-Execute** (V3):
- Deployment: "Configure VLANs 10-20 on all access switches"
- Known structure, sequential steps
- Bulk operations

**Reflexion** (V3):
- Complex automation: "Generate and deploy optimal QoS config"
- First attempt often fails
- Learn from errors

**Multi-Agent** (V4):
- Incident response: "Diagnose, fix, and verify network outage"
- Multiple specialized skills needed
- Coordination required

---

## V1: Tool-Calling Agent (Baseline)

Start with Claude's native tool calling. This is the simplest, fastest, and most reliable pattern.

### Architecture

```
User Query → Claude (with tools) → Tool Selection → Execute → Claude → Answer
                                          ↓
                                    [get_bgp_summary]
                                    [get_interface_status]
                                    [check_routes]
```

**No reasoning loop**. Claude decides which tools to call in one shot, runtime executes them, Claude interprets results.

### Implementation

```python
"""
V1: Tool-Calling Agent - Simplest pattern
File: agents/v1_tool_calling.py
"""
from anthropic import Anthropic
from typing import List, Dict, Callable
import json

class ToolCallingAgent:
    """
    V1: Claude's native tool calling.

    Best for: Simple queries with known tools (80% of use cases).
    """

    def __init__(self, api_key: str, tools: List[Dict], tool_functions: Dict[str, Callable]):
        """
        Args:
            api_key: Anthropic API key
            tools: List of tool definitions in Claude format
            tool_functions: Dict mapping tool names to Python functions
        """
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.tool_functions = tool_functions

    def run(self, user_query: str, verbose: bool = True) -> Dict:
        """
        Execute query using tool calling.

        Returns:
            Dict with answer, tools_used, tokens, latency
        """
        import time
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {user_query}")
            print(f"{'='*60}\n")

        messages = [{"role": "user", "content": user_query}]
        tools_used = []
        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            # Call Claude with tools
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]

                if verbose:
                    print(f"Claude is calling {len(tool_calls)} tool(s):")

                # Execute each tool
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_input = tool_call.input

                    if verbose:
                        print(f"  • {tool_name}({json.dumps(tool_input)})")

                    # Execute the tool
                    result = self._execute_tool(tool_name, tool_input)

                    if verbose:
                        result_preview = str(result)[:100]
                        print(f"    → {result_preview}{'...' if len(str(result)) > 100 else ''}")

                    tools_used.append(tool_name)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": str(result)
                    })

                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude has final answer
                final_text = next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "No response text"
                )

                latency = time.time() - start

                if verbose:
                    print(f"\nFinal Answer:\n{final_text}\n")
                    print(f"Tools used: {tools_used}")
                    print(f"Tokens: {total_input_tokens + total_output_tokens}")
                    print(f"Latency: {latency:.2f}s\n")

                return {
                    "answer": final_text,
                    "tools_used": tools_used,
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "latency_seconds": latency,
                    "success": True
                }

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool function."""
        if tool_name not in self.tool_functions:
            return f"ERROR: Tool {tool_name} not found"

        try:
            func = self.tool_functions[tool_name]
            result = func(**tool_input)
            return str(result)
        except Exception as e:
            return f"ERROR: {str(e)}"


# Network Tools
def get_interface_status(hostname: str, interface: str) -> Dict:
    """Get interface operational status."""
    # Mock data - replace with real Netmiko
    mock_data = {
        "router1": {
            "GigabitEthernet0/0": {"status": "up", "protocol": "up", "ip": "192.168.1.1"},
            "GigabitEthernet0/1": {"status": "administratively down", "protocol": "down", "ip": None},
            "GigabitEthernet0/2": {"status": "up", "protocol": "down", "ip": "192.168.2.1"}
        }
    }

    interface_info = mock_data.get(hostname, {}).get(interface, {"status": "unknown"})
    return {
        "hostname": hostname,
        "interface": interface,
        **interface_info
    }

def get_bgp_summary(hostname: str) -> Dict:
    """Get BGP neighbor summary."""
    return {
        "router_id": "10.0.0.1",
        "local_as": 65001,
        "neighbors": [
            {"ip": "10.0.1.2", "as": 65002, "state": "Established", "prefixes": 150, "uptime": "2d03h"},
            {"ip": "10.0.1.3", "as": 65003, "state": "Idle", "prefixes": 0, "uptime": "never"}
        ]
    }

def get_route_table(hostname: str, prefix: str = None) -> List[Dict]:
    """Get routing table entries."""
    routes = [
        {"prefix": "0.0.0.0/0", "next_hop": "192.168.1.254", "protocol": "static", "metric": 1},
        {"prefix": "10.0.0.0/8", "next_hop": "10.0.1.2", "protocol": "bgp", "metric": 100},
        {"prefix": "172.16.0.0/16", "next_hop": "directly_connected", "protocol": "connected", "metric": 0}
    ]
    if prefix:
        routes = [r for r in routes if r["prefix"] == prefix]
    return routes

def check_recent_changes(hostname: str, hours: int = 24) -> List[Dict]:
    """Check recent configuration changes."""
    return [
        {
            "timestamp": "2024-01-15 14:23:45",
            "user": "admin",
            "command": "interface GigabitEthernet0/1",
            "change": "shutdown"
        }
    ]


# Tool Definitions (Claude format)
TOOL_DEFINITIONS = [
    {
        "name": "get_interface_status",
        "description": "Get operational status of a network interface including IP address, speed, and up/down state",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Router or switch hostname"},
                "interface": {"type": "string", "description": "Interface name (e.g., GigabitEthernet0/0)"}
            },
            "required": ["hostname", "interface"]
        }
    },
    {
        "name": "get_bgp_summary",
        "description": "Get BGP routing protocol summary including all neighbors and their status",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Router hostname"}
            },
            "required": ["hostname"]
        }
    },
    {
        "name": "get_route_table",
        "description": "Get routing table entries, optionally filtered by prefix",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Router hostname"},
                "prefix": {"type": "string", "description": "Optional: Filter by IP prefix (e.g., 10.0.0.0/8)"}
            },
            "required": ["hostname"]
        }
    },
    {
        "name": "check_recent_changes",
        "description": "Check recent configuration changes on a device",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Device hostname"},
                "hours": {"type": "integer", "description": "Look back this many hours (default 24)"}
            },
            "required": ["hostname"]
        }
    }
]


# Example Usage
if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    tool_functions = {
        "get_interface_status": get_interface_status,
        "get_bgp_summary": get_bgp_summary,
        "get_route_table": get_route_table,
        "check_recent_changes": check_recent_changes
    }

    agent = ToolCallingAgent(
        api_key=api_key,
        tools=TOOL_DEFINITIONS,
        tool_functions=tool_functions
    )

    # Test queries
    agent.run("What's the BGP status on router1? Are all neighbors up?")
    agent.run("Check if interface GigabitEthernet0/1 on router1 is operational")
```

### Example Output

```
============================================================
Query: What's the BGP status on router1? Are all neighbors up?
============================================================

Claude is calling 1 tool(s):
  • get_bgp_summary({"hostname": "router1"})
    → {'router_id': '10.0.0.1', 'local_as': 65001, 'neighbors': [{'ip': '10.0.1.2', 'as': 65002, 'stat...

Final Answer:
Based on the BGP summary for router1:

**BGP Status**: Operational
- Router ID: 10.0.0.1
- Local AS: 65001

**Neighbors**:
1. ✓ 10.0.1.2 (AS 65002) - Established, receiving 150 prefixes (uptime: 2d03h)
2. ✗ 10.0.1.3 (AS 65003) - Idle, not established

**Answer**: No, not all neighbors are up. Neighbor 10.0.1.3 is in Idle state and needs investigation.

Tools used: ['get_bgp_summary']
Tokens: 1247
Latency: 2.34s
```

### V1 Characteristics

**Strengths**:
- Fast (single API call + tool execution)
- Cheap (~1,200 tokens per query)
- Reliable (95%+ success rate)
- Perfect for simple queries

**Limitations**:
- Can't explore unknowns
- No reasoning loop
- Struggles with multi-step diagnosis

**When to use**: Information retrieval, status checks, simple troubleshooting (80% of production use cases).

---

## V2: ReAct Agent (Reasoning Loop)

Add a reasoning loop for exploratory tasks. ReAct interleaves **Thought** → **Action** → **Observation** until the problem is solved.

### Architecture

```
User Task → Loop:
              ├─ Thought: "What should I check next?"
              ├─ Action: Call tool
              ├─ Observation: See result
              └─ Repeat or Answer
```

**Key difference from V1**: Agent reasons about what to do next based on observations. Can discover problems through exploration.

### Implementation

```python
"""
V2: ReAct Agent - Reasoning + Acting
File: agents/v2_react.py
"""
from anthropic import Anthropic
from typing import List, Dict, Callable, Optional
import time

class ReActAgent:
    """
    V2: ReAct (Reasoning + Acting) agent.

    Interleaves reasoning steps with tool calls until task is solved.
    Best for: Exploratory diagnosis, unknown problem spaces.
    """

    def __init__(self, api_key: str, tools: Dict[str, Callable], max_iterations: int = 10):
        """
        Args:
            api_key: Anthropic API key
            tools: Dict mapping tool names to callable functions
            max_iterations: Max reasoning loops before giving up
        """
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.max_iterations = max_iterations
        self.conversation_history = []

    def run(self, task: str, verbose: bool = True) -> Dict:
        """
        Execute task using ReAct loop.

        Returns:
            Dict with answer, steps, iterations, tokens, latency
        """
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}\n")

        # Initialize conversation
        self.conversation_history = [{
            "role": "user",
            "content": self._build_initial_prompt(task)
        }]

        steps = []
        total_tokens = 0

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"--- Iteration {iteration + 1} ---\n")

            # Get agent's next thought and action
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=self.conversation_history
            )

            assistant_message = response.content[0].text
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            if verbose:
                print(f"{assistant_message}\n")

            # Parse the response
            parsed = self._parse_response(assistant_message)

            if parsed["type"] == "answer":
                # Agent has reached conclusion
                latency = time.time() - start

                if verbose:
                    print(f"✓ Task Complete")
                    print(f"Steps taken: {iteration + 1}")
                    print(f"Total tokens: {total_tokens}")
                    print(f"Latency: {latency:.2f}s\n")

                return {
                    "answer": parsed["content"],
                    "steps": steps,
                    "iterations": iteration + 1,
                    "total_tokens": total_tokens,
                    "latency_seconds": latency,
                    "success": True
                }

            elif parsed["type"] == "action":
                # Agent wants to use a tool
                tool_name = parsed["tool"]
                tool_args = parsed["args"]

                if verbose:
                    print(f"→ Action: {tool_name}({tool_args})")

                # Execute the tool
                observation = self._execute_tool(tool_name, tool_args)

                if verbose:
                    obs_preview = str(observation)[:200]
                    print(f"→ Observation: {obs_preview}{'...' if len(str(observation)) > 200 else ''}\n")

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

                steps.append({
                    "thought": parsed.get("thought", ""),
                    "action": tool_name,
                    "args": tool_args,
                    "observation": str(observation)
                })

            else:
                # Unexpected response format
                if verbose:
                    print(f"⚠️  Could not parse response")
                break

        # Max iterations reached
        latency = time.time() - start
        return {
            "answer": "Task incomplete - reached max iterations",
            "steps": steps,
            "iterations": self.max_iterations,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "success": False
        }

    def _build_initial_prompt(self, task: str) -> str:
        """Build the initial system prompt for ReAct."""
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])

        return f"""You are a network troubleshooting agent using the ReAct (Reasoning + Acting) framework.

Your task: {task}

Available tools:
{tools_desc}

Format your responses EXACTLY like this:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Args: [arguments as JSON dict, e.g., {{"hostname": "router1"}}]

Or, when you have the final answer:

Thought: [Your final reasoning]
Answer: [Your final answer to the task]

IMPORTANT RULES:
1. Always start with "Thought:" to explain your reasoning
2. Use "Action:" and "Args:" to call tools
3. Use "Answer:" ONLY when you have the complete solution
4. Think step-by-step - don't jump to conclusions
5. Use tools to gather evidence before making claims

Begin!"""

    def _parse_response(self, response: str) -> Dict:
        """Parse agent's response into structured format."""
        lines = response.strip().split("\n")
        result = {"type": "unknown"}

        for line in lines:
            line = line.strip()

            if line.startswith("Thought:"):
                result["thought"] = line.replace("Thought:", "").strip()

            elif line.startswith("Action:"):
                result["type"] = "action"
                result["tool"] = line.replace("Action:", "").strip()

            elif line.startswith("Args:"):
                import json
                args_str = line.replace("Args:", "").strip()
                try:
                    result["args"] = json.loads(args_str)
                except json.JSONDecodeError:
                    result["args"] = {"raw": args_str}

            elif line.startswith("Answer:"):
                result["type"] = "answer"
                result["content"] = line.replace("Answer:", "").strip()

        return result

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool and return the observation."""
        if tool_name not in self.tools:
            return f"ERROR: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        try:
            tool_func = self.tools[tool_name]
            result = tool_func(**args)
            return str(result)
        except Exception as e:
            return f"ERROR executing {tool_name}: {str(e)}"


# Example Usage
if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Reuse tools from V1
    tools = {
        "get_interface_status": get_interface_status,
        "get_bgp_summary": get_bgp_summary,
        "get_route_table": get_route_table,
        "check_recent_changes": check_recent_changes
    }

    agent = ReActAgent(api_key=api_key, tools=tools, max_iterations=8)

    # Complex diagnosis task
    result = agent.run(
        task="Router1 is not receiving any BGP routes from 10.0.1.3. Why?"
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
```

### Example Output

```
============================================================
TASK: Router1 is not receiving any BGP routes from 10.0.1.3. Why?
============================================================

--- Iteration 1 ---

Thought: I need to check the BGP neighbor status first to see the state of the BGP session with 10.0.1.3.
Action: get_bgp_summary
Args: {"hostname": "router1"}

→ Action: get_bgp_summary({'hostname': 'router1'})
→ Observation: {'router_id': '10.0.0.1', 'local_as': 65001, 'neighbors': [{'ip': '10.0.1.2', 'as': 65002, 'state': 'Established', 'prefixes': 150, 'uptime': '2d03h'}, {'ip': '10.0.1.3', 'as': 65003, 'state': 'Idle',...

--- Iteration 2 ---

Thought: The BGP neighbor 10.0.1.3 is in "Idle" state and has never come up (uptime shows "never"). This indicates the TCP session is not establishing. This could be due to interface issues, routing problems, or configuration errors. Let me check if there are recent configuration changes that might have caused this.
Action: check_recent_changes
Args: {"hostname": "router1", "hours": 24}

→ Action: check_recent_changes({'hostname': 'router1', 'hours': 24})
→ Observation: [{'timestamp': '2024-01-15 14:23:45', 'user': 'admin', 'command': 'interface GigabitEthernet0/1', 'change': 'shutdown'}]

--- Iteration 3 ---

Thought: Found it! GigabitEthernet0/1 was administratively shut down recently. This interface is likely the one used to reach BGP neighbor 10.0.1.3. Let me verify the interface status.
Action: get_interface_status
Args: {"hostname": "router1", "interface": "GigabitEthernet0/1"}

→ Action: get_interface_status({'hostname': 'router1', 'interface': 'GigabitEthernet0/1'})
→ Observation: {'hostname': 'router1', 'interface': 'GigabitEthernet0/1', 'status': 'administratively down', 'protocol': 'down', 'ip': None}

--- Iteration 4 ---

Thought: Confirmed. GigabitEthernet0/1 is administratively down, which is why the BGP session with 10.0.1.3 cannot establish. The interface was shut down by admin at 14:23:45 today.
Answer: Router1 is not receiving BGP routes from 10.0.1.3 because the interface GigabitEthernet0/1 (used to reach this neighbor) was administratively shut down by admin on 2024-01-15 at 14:23:45. The BGP session is in "Idle" state and has never established. Solution: Execute "no shutdown" on GigabitEthernet0/1.

✓ Task Complete
Steps taken: 4
Total tokens: 2847
Latency: 8.42s

============================================================
FINAL RESULT
============================================================
Answer: Router1 is not receiving BGP routes from 10.0.1.3 because...
Success: True
```

### V2 vs V1 Comparison

| Aspect | V1 Tool-Calling | V2 ReAct |
|--------|----------------|----------|
| **Decision making** | Single pass | Iterative |
| **Exploration** | None | High |
| **Latency** | 2-3s | 8-12s |
| **Tokens** | 1,200 | 2,800 |
| **Cost** | $0.05 | $0.20 |
| **Success (complex)** | 60% | 90% |
| **Best for** | Known queries | Unknown diagnosis |

**When V2 beats V1**: "Why is X broken?" - Need to explore and form hypotheses.

**When V1 beats V2**: "Show me X" - Known data retrieval.

---

## V3: Plan-Execute + Reflexion

Add two advanced patterns:
1. **Plan-Execute**: Create plan upfront, execute sequentially
2. **Reflexion**: Learn from failures by reflecting

### V3a: Plan-Execute Agent

Best for deployments where you know the structure upfront.

```python
"""
V3a: Plan-and-Execute Agent
File: agents/v3a_plan_execute.py
"""
from anthropic import Anthropic
from typing import List, Dict, Callable
import json
import time

class PlanExecuteAgent:
    """
    V3a: Plan-and-Execute agent.

    Creates complete plan upfront, then executes sequentially.
    Best for: Deployments, migrations, bulk operations.
    """

    def __init__(self, api_key: str, tools: Dict[str, Callable]):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools

    def run(self, task: str, verbose: bool = True) -> Dict:
        """Execute task using plan-then-execute approach."""
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}\n")

        # Phase 1: Planning
        plan, planning_tokens = self._create_plan(task)

        if verbose:
            print("📋 PLAN:")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step['description']}")
                print(f"     Tool: {step['tool']}, Args: {step['args']}")
            print()

        # Phase 2: Execution
        results = []
        for i, step in enumerate(plan, 1):
            if verbose:
                print(f"--- Executing Step {i}/{len(plan)} ---")
                print(f"Description: {step['description']}")

            result = self._execute_step(step)
            results.append(result)

            if result["status"] == "error":
                if verbose:
                    print(f"❌ Step failed: {result['error']}\n")

                latency = time.time() - start
                return {
                    "success": False,
                    "completed_steps": i - 1,
                    "total_steps": len(plan),
                    "results": results,
                    "error": result["error"],
                    "total_tokens": planning_tokens,
                    "latency_seconds": latency
                }

            if verbose:
                output_preview = result['output'][:100]
                print(f"✓ Step completed: {output_preview}{'...' if len(result['output']) > 100 else ''}\n")

        # All steps completed
        latency = time.time() - start

        if verbose:
            print(f"✓ All {len(plan)} steps completed successfully!")
            print(f"Total tokens: {planning_tokens}")
            print(f"Latency: {latency:.2f}s\n")

        return {
            "success": True,
            "completed_steps": len(plan),
            "total_steps": len(plan),
            "results": results,
            "total_tokens": planning_tokens,
            "latency_seconds": latency
        }

    def _create_plan(self, task: str) -> tuple[List[Dict], int]:
        """Create execution plan for the task."""
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])

        prompt = f"""Create a detailed execution plan for this task: {task}

Available tools:
{tools_desc}

Return the plan as a JSON array where each step has:
- description: Clear description of what this step does
- tool: The tool to call
- args: Arguments as a JSON object

Example:
[
  {{
    "description": "Check current VLAN configuration",
    "tool": "get_vlan_config",
    "args": {{"hostname": "switch1"}}
  }},
  {{
    "description": "Create VLAN 100",
    "tool": "create_vlan",
    "args": {{"hostname": "switch1", "vlan_id": 100, "name": "Finance"}}
  }}
]

Return ONLY the JSON array, no other text."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        plan_text = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens

        # Extract JSON from response
        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0].strip()

        plan = json.loads(plan_text)
        return plan, tokens

    def _execute_step(self, step: Dict) -> Dict:
        """Execute a single step from the plan."""
        tool_name = step["tool"]
        args = step["args"]

        if tool_name not in self.tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not found",
                "output": None
            }

        try:
            tool_func = self.tools[tool_name]
            output = tool_func(**args)
            return {
                "status": "success",
                "output": str(output),
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "output": None
            }


# VLAN Management Tools
def get_vlan_config(hostname: str) -> str:
    """Get current VLAN configuration."""
    return f"VLANs on {hostname}: 1 (default), 10 (Engineering), 20 (Sales)"

def create_vlan(hostname: str, vlan_id: int, name: str) -> str:
    """Create a new VLAN."""
    return f"Created VLAN {vlan_id} ({name}) on {hostname}"

def assign_vlan_to_ports(hostname: str, vlan_id: int, ports: List[str]) -> str:
    """Assign VLAN to switch ports."""
    return f"Assigned VLAN {vlan_id} to ports {', '.join(ports)} on {hostname}"

def save_config(hostname: str) -> str:
    """Save running config to startup config."""
    return f"Configuration saved on {hostname}"


# Example Usage
if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    tools = {
        "get_vlan_config": get_vlan_config,
        "create_vlan": create_vlan,
        "assign_vlan_to_ports": assign_vlan_to_ports,
        "save_config": save_config
    }

    agent = PlanExecuteAgent(api_key=api_key, tools=tools)

    result = agent.run(
        task="Create VLAN 100 named 'Finance' on switch1 and assign it to ports Gi0/1-4, then save the config"
    )
```

### Example Output

```
============================================================
TASK: Create VLAN 100 named 'Finance' on switch1 and assign it to ports Gi0/1-4, then save the config
============================================================

📋 PLAN:
  1. Check current VLAN configuration
     Tool: get_vlan_config, Args: {'hostname': 'switch1'}
  2. Create VLAN 100 named Finance
     Tool: create_vlan, Args: {'hostname': 'switch1', 'vlan_id': 100, 'name': 'Finance'}
  3. Assign VLAN 100 to ports Gi0/1-4
     Tool: assign_vlan_to_ports, Args: {'hostname': 'switch1', 'vlan_id': 100, 'ports': ['Gi0/1', 'Gi0/2', 'Gi0/3', 'Gi0/4']}
  4. Save configuration
     Tool: save_config, Args: {'hostname': 'switch1'}

--- Executing Step 1/4 ---
Description: Check current VLAN configuration
✓ Step completed: VLANs on switch1: 1 (default), 10 (Engineering), 20 (Sales)

--- Executing Step 2/4 ---
Description: Create VLAN 100 named Finance
✓ Step completed: Created VLAN 100 (Finance) on switch1

--- Executing Step 3/4 ---
Description: Assign VLAN 100 to ports Gi0/1-4
✓ Step completed: Assigned VLAN 100 to ports Gi0/1, Gi0/2, Gi0/3, Gi0/4 on switch1

--- Executing Step 4/4 ---
Description: Save configuration
✓ Step completed: Configuration saved on switch1

✓ All 4 steps completed successfully!
Total tokens: 1834
Latency: 5.12s
```

### V3b: Reflexion Agent

Learn from failures by reflecting on what went wrong.

```python
"""
V3b: Reflexion Agent - Learning from Failures
File: agents/v3b_reflexion.py
"""
from anthropic import Anthropic
from typing import List, Dict, Callable
import time

class ReflexionAgent:
    """
    V3b: Reflexion agent - learns from failures.

    Try → Fail → Reflect → Retry with new strategy.
    Best for: Complex automation where first attempt often fails.
    """

    def __init__(self, api_key: str, tools: Dict[str, Callable], max_attempts: int = 3):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.max_attempts = max_attempts

    def run(self, task: str, verbose: bool = True) -> Dict:
        """Execute task with reflection on failures."""
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}\n")

        attempt = 0
        reflection_memory = []
        total_tokens = 0
        all_attempts = []

        while attempt < self.max_attempts:
            attempt += 1

            if verbose:
                print(f"--- Attempt {attempt} ---\n")

            # Try to execute the task
            result, tokens = self._attempt_task(task, reflection_memory, verbose)
            total_tokens += tokens
            all_attempts.append(result)

            if result["success"]:
                latency = time.time() - start

                if verbose:
                    print(f"\n✓ Task completed successfully!")
                    print(f"Attempts: {attempt}")
                    print(f"Total tokens: {total_tokens}")
                    print(f"Latency: {latency:.2f}s\n")

                return {
                    "success": True,
                    "answer": result["answer"],
                    "attempts": attempt,
                    "reflections": reflection_memory,
                    "total_tokens": total_tokens,
                    "latency_seconds": latency
                }

            # Task failed - reflect on why
            if verbose:
                print(f"\n❌ Attempt {attempt} failed: {result['error']}\n")

            reflection, reflection_tokens = self._reflect_on_failure(
                task, result["error"], result.get("actions", [])
            )
            total_tokens += reflection_tokens
            reflection_memory.append(reflection)

            if verbose:
                print(f"💭 Reflection: {reflection}\n")

        # Max attempts reached
        latency = time.time() - start

        if verbose:
            print(f"❌ Task failed after {self.max_attempts} attempts")
            print(f"Total tokens: {total_tokens}")
            print(f"Latency: {latency:.2f}s\n")

        return {
            "success": False,
            "error": "Max attempts reached",
            "attempts": self.max_attempts,
            "reflections": reflection_memory,
            "total_tokens": total_tokens,
            "latency_seconds": latency
        }

    def _attempt_task(self, task: str, reflections: List[str], verbose: bool) -> tuple[Dict, int]:
        """Attempt to complete the task."""
        # Build prompt with reflection memory
        reflection_context = ""
        if reflections:
            reflection_context = "\n\nPrevious attempt failures and reflections:\n" + "\n".join([
                f"{i+1}. {r}" for i, r in enumerate(reflections)
            ])

        prompt = f"""Execute this task: {task}{reflection_context}

Think carefully about the approach and execute it. Return your result as:

Success: [true/false]
Answer: [your result if successful]
Error: [what went wrong if failed]
Actions: [list of actions you took]"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        tokens = response.usage.input_tokens + response.usage.output_tokens
        result_text = response.content[0].text

        if verbose:
            print(result_text)

        # Parse result (simplified - in production, use structured output)
        if "Success: true" in result_text or "successful" in result_text.lower():
            return {
                "success": True,
                "answer": result_text,
                "actions": []
            }, tokens
        else:
            return {
                "success": False,
                "error": result_text,
                "actions": []
            }, tokens

    def _reflect_on_failure(self, task: str, error: str, actions: List) -> tuple[str, int]:
        """
        Analyze failure and generate insights for next attempt.

        Returns reflection text and token count.
        """
        prompt = f"""Task: {task}

Error encountered: {error}

Reflect on this failure:
1. Why did this approach fail?
2. What assumption was incorrect?
3. What should we try differently next time?

Keep reflection concise (2-3 sentences)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        tokens = response.usage.input_tokens + response.usage.output_tokens
        reflection = response.content[0].text

        return reflection, tokens


# Example Usage
if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Simulated tools
    tools = {}

    agent = ReflexionAgent(api_key=api_key, tools=tools, max_attempts=3)

    result = agent.run(
        task="Generate QoS configuration for VoIP traffic with 100Mbps guaranteed bandwidth"
    )
```

### V3 Pattern Comparison

| Pattern | Planning | Adapts | Tokens | Best For |
|---------|----------|--------|--------|----------|
| Plan-Execute | Upfront | No | 1,800 | Deployments |
| Reflexion | Iterative | Yes | 4,200 | Complex automation |

**Use Plan-Execute for**: Known workflows (VLAN deployment, device provisioning).

**Use Reflexion for**: Tasks where first attempt often fails (config generation with dependencies, complex troubleshooting).

---

## V4: Multi-Agent System

Coordinate multiple specialized agents for complex operations.

### Architecture

```
                    ┌──────────────────┐
                    │  Supervisor      │
                    │  Agent           │
                    │  (Routes tasks)  │
                    └──────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Diagnosis   │    │  Config      │    │  Execution   │
│  Agent       │    │  Agent       │    │  Agent       │
│  (V2 ReAct)  │    │(V3 Plan-Exec)│    │  (V1 Tool)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Workflow**: Diagnose issue → Generate fix → Execute (with approval) → Verify.

### Implementation

```python
"""
V4: Multi-Agent System - Specialized agents working together
File: agents/v4_multi_agent.py
"""
from anthropic import Anthropic
from typing import Dict
import time

class SupervisorAgent:
    """
    V4: Supervisor coordinating multiple specialized agents.

    Routes tasks to appropriate agent or coordinates multi-agent workflows.
    Best for: Complex operations requiring multiple skills.
    """

    def __init__(self, api_key: str, agents: Dict):
        """
        Args:
            api_key: Anthropic API key
            agents: Dict mapping agent names to agent instances
        """
        self.client = Anthropic(api_key=api_key)
        self.agents = agents

    def run(self, task: str, verbose: bool = True) -> Dict:
        """Route task to appropriate agent(s)."""
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task}")
            print(f"{'='*60}\n")

        # Classify the task
        agent_type, classification_tokens = self._classify_task(task)

        if verbose:
            print(f"📊 Task classified as: {agent_type}\n")

        if agent_type == "complex":
            # Multi-agent coordination needed
            result = self._coordinate_multi_agent(task, verbose)
            result["classification_tokens"] = classification_tokens
        else:
            # Single agent can handle it
            if verbose:
                print(f"→ Routing to {agent_type} agent\n")

            result = self.agents[agent_type].run(task, verbose=verbose)
            result["agent_used"] = agent_type
            result["classification_tokens"] = classification_tokens

        latency = time.time() - start
        result["total_latency"] = latency

        if verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-AGENT SUMMARY")
            print(f"{'='*60}")
            print(f"Total latency: {latency:.2f}s")
            print(f"Total tokens: {result.get('total_tokens', 0) + classification_tokens}")

        return result

    def _classify_task(self, task: str) -> tuple[str, int]:
        """Determine which agent should handle this task."""
        prompt = f"""Classify this network engineering task:

Task: {task}

Available agents:
- diagnosis: Troubleshoot network issues, analyze failures (V2 ReAct)
- config: Generate or modify device configurations (V3 Plan-Execute)
- query: Simple information retrieval (V1 Tool-calling)
- complex: Requires multiple agents working together (full incident response)

Return only the agent name (diagnosis/config/query/complex)."""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Fast, cheap for classification
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        tokens = response.usage.input_tokens + response.usage.output_tokens
        agent_type = response.content[0].text.strip().lower()

        return agent_type, tokens

    def _coordinate_multi_agent(self, task: str, verbose: bool) -> Dict:
        """Coordinate multiple agents for complex task."""
        if verbose:
            print("🔄 Complex task detected - coordinating multiple agents\n")

        total_tokens = 0

        # Step 1: Diagnose the problem
        if verbose:
            print("→ Step 1: Diagnosis Agent\n")

        diagnosis_result = self.agents["diagnosis"].run(
            task=f"Diagnose: {task}",
            verbose=verbose
        )
        total_tokens += diagnosis_result.get("total_tokens", 0)

        if not diagnosis_result.get("success"):
            return {
                "success": False,
                "error": "Diagnosis failed",
                "total_tokens": total_tokens,
                "diagnosis": diagnosis_result
            }

        # Step 2: Generate fix
        if verbose:
            print(f"\n→ Step 2: Config Agent\n")

        config_result = self.agents["config"].run(
            task=f"Generate fix for: {diagnosis_result['answer']}",
            verbose=verbose
        )
        total_tokens += config_result.get("total_tokens", 0)

        if not config_result.get("success"):
            return {
                "success": False,
                "error": "Config generation failed",
                "total_tokens": total_tokens,
                "diagnosis": diagnosis_result,
                "config": config_result
            }

        # Step 3: Execution (requires human approval in production)
        if verbose:
            print(f"\n→ Step 3: Execution (APPROVAL REQUIRED)\n")
            print("⚠️  The following fix was generated:")
            print(config_result.get("answer", "No config generated"))
            print("\n⚠️  In production, human approval would be required here.")
            print("    For this demo, execution is simulated.\n")

        return {
            "success": True,
            "workflow": "multi-agent",
            "diagnosis": diagnosis_result,
            "config": config_result,
            "executed": False,  # Safety: never auto-execute
            "approval_required": True,
            "total_tokens": total_tokens
        }


# Example Usage
if __name__ == "__main__":
    import os
    from v1_tool_calling import ToolCallingAgent, TOOL_DEFINITIONS, tool_functions
    from v2_react import ReActAgent
    from v3a_plan_execute import PlanExecuteAgent

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Initialize specialized agents
    agents = {
        "query": ToolCallingAgent(api_key, TOOL_DEFINITIONS, tool_functions),
        "diagnosis": ReActAgent(api_key, tool_functions),
        "config": PlanExecuteAgent(api_key, tool_functions)
    }

    # Create supervisor
    supervisor = SupervisorAgent(api_key=api_key, agents=agents)

    # Test different task types
    print("\n" + "="*80)
    print("TEST 1: Simple query (should route to V1 Tool-calling)")
    print("="*80)
    supervisor.run("What's the BGP status on router1?")

    print("\n" + "="*80)
    print("TEST 2: Diagnosis (should route to V2 ReAct)")
    print("="*80)
    supervisor.run("Why is router1 not receiving routes from 10.0.1.3?")

    print("\n" + "="*80)
    print("TEST 3: Complex incident (should use multi-agent coordination)")
    print("="*80)
    supervisor.run("Fix the BGP peering issue between router1 and 10.0.1.3")
```

### Example Output

```
============================================================
TASK: Fix the BGP peering issue between router1 and 10.0.1.3
============================================================

📊 Task classified as: complex

🔄 Complex task detected - coordinating multiple agents

→ Step 1: Diagnosis Agent

============================================================
TASK: Diagnose: Fix the BGP peering issue between router1 and 10.0.1.3
============================================================

--- Iteration 1 ---
Thought: I need to check the BGP neighbor status to see the current state...
[... ReAct diagnosis loop ...]

✓ Task Complete
Steps taken: 4
Total tokens: 2847
Latency: 8.42s

→ Step 2: Config Agent

============================================================
TASK: Generate fix for: Interface GigabitEthernet0/1 is administratively down
============================================================

📋 PLAN:
  1. Enable interface GigabitEthernet0/1
     Tool: configure_interface, Args: {'hostname': 'router1', 'interface': 'GigabitEthernet0/1', 'command': 'no shutdown'}
  2. Verify BGP neighbor comes up
     Tool: get_bgp_summary, Args: {'hostname': 'router1'}
  3. Save configuration
     Tool: save_config, Args: {'hostname': 'router1'}
[... Plan execution ...]

→ Step 3: Execution (APPROVAL REQUIRED)

⚠️  The following fix was generated:
1. no shutdown on GigabitEthernet0/1
2. Verify BGP peer establishes
3. Save configuration

⚠️  In production, human approval would be required here.
    For this demo, execution is simulated.

============================================================
MULTI-AGENT SUMMARY
============================================================
Total latency: 15.67s
Total tokens: 4523

Workflow: diagnosis → config → (awaiting approval)
```

### V4 Benefits

**Specialization**: Each agent optimized for its task.
- Diagnosis: V2 ReAct (explores unknowns)
- Config: V3 Plan-Execute (structured workflows)
- Query: V1 Tool-calling (fast, cheap)

**Coordination**: Supervisor routes intelligently.
- Simple query → Direct to V1 (2s latency)
- Complex incident → Multi-agent workflow (15s latency)

**Safety**: Human approval gates for production changes.

---

## Pattern Selection Decision Tree

```
┌─────────────────────────────────────────┐
│ Is it a simple information query?      │
│ ("What is X?" "Show me Y")             │
└──────────┬──────────────────────────────┘
           │
     YES ──┴── Use V1: Tool-Calling Agent
           │    • Latency: 2s
           │    • Cost: $0.05
           │    • Success: 95%
           │
           NO
           │
           ▼
┌─────────────────────────────────────────┐
│ Do you know the structure upfront?     │
│ (Deployment, migration, bulk ops)      │
└──────────┬──────────────────────────────┘
           │
     YES ──┴── Use V3: Plan-Execute Agent
           │    • Latency: 5s
           │    • Cost: $0.15
           │    • Success: 85%
           │
           NO
           │
           ▼
┌─────────────────────────────────────────┐
│ Will first attempt likely fail?        │
│ (Complex config, learning needed)      │
└──────────┬──────────────────────────────┘
           │
     YES ──┴── Use V3: Reflexion Agent
           │    • Latency: 25s
           │    • Cost: $0.60
           │    • Success: 95%
           │
           NO
           │
           ▼
┌─────────────────────────────────────────┐
│ Exploratory diagnosis needed?          │
│ ("Why is X broken?")                   │
└──────────┬──────────────────────────────┘
           │
     YES ──┴── Use V2: ReAct Agent
           │    • Latency: 8s
           │    • Cost: $0.20
           │    • Success: 90%
           │
           NO
           │
           ▼
┌─────────────────────────────────────────┐
│ Requires multiple specialized skills?  │
│ (Full incident response)               │
└──────────┬──────────────────────────────┘
           │
     YES ──┴── Use V4: Multi-Agent System
                • Latency: 15-20s
                • Cost: $0.45
                • Success: 98%
```

---

## Production Considerations

### Error Handling

Every agent needs robust error handling:

```python
class ProductionAgent:
    """Agent with production-grade error handling."""

    def run(self, task: str) -> Dict:
        try:
            result = self._execute_task(task)
            return result

        except APITimeoutError as e:
            return {"success": False, "error": "API timeout", "retry": True}

        except AuthenticationError as e:
            return {"success": False, "error": "Auth failed", "retry": False}

        except RateLimitError as e:
            wait_time = self._get_retry_after(e)
            time.sleep(wait_time)
            return self.run(task)  # Retry after waiting

        except Exception as e:
            # Log unexpected error
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "retry": False}
```

### Safety Guardrails

Network agents MUST have safety controls:

```python
class SafeAgent:
    """Agent with safety guardrails for network operations."""

    def __init__(self, api_key: str, tools: Dict, allowed_operations: List[str]):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.allowed_operations = allowed_operations  # Whitelist
        self.dangerous_patterns = [
            "shutdown",
            "reload",
            "delete",
            "erase",
            "write erase",
            "no "  # Any "no" command (be careful!)
        ]

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute tool with safety checks."""
        # Check 1: Is this operation allowed?
        if tool_name not in self.allowed_operations:
            raise PermissionError(f"Operation '{tool_name}' not allowed")

        # Check 2: Does it contain dangerous commands?
        command = args.get("command", "")
        for pattern in self.dangerous_patterns:
            if pattern in command.lower():
                # Require explicit approval
                if not self._get_human_approval(tool_name, args):
                    raise PermissionError(f"Approval denied for: {pattern}")

        # Check 3: Production environment requires approval
        if args.get("environment") == "production":
            if not self._get_human_approval(tool_name, args):
                raise PermissionError("Human approval required for production")

        # Execute safely
        return self.tools[tool_name](**args)

    def _get_human_approval(self, tool_name: str, args: Dict) -> bool:
        """Get human approval for sensitive operations."""
        print(f"\n⚠️  APPROVAL REQUIRED")
        print(f"Operation: {tool_name}")
        print(f"Arguments: {args}")

        # In production, integrate with ticketing system (ServiceNow, Jira)
        # For demo, use simple input
        response = input("Approve? (yes/no): ")
        return response.lower() == "yes"
```

### Logging and Observability

Log everything for debugging and compliance:

```python
import logging
import json
from datetime import datetime

class ObservableAgent:
    """Agent with comprehensive logging."""

    def __init__(self, api_key: str, tools: Dict):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.logger = logging.getLogger(__name__)
        self.session_id = self._generate_session_id()

    def run(self, task: str) -> Dict:
        """Execute task with full logging."""
        self.logger.info(f"[{self.session_id}] Task started: {task}")

        start_time = datetime.now()

        try:
            result = self._execute_task(task)

            self.logger.info(f"[{self.session_id}] Task completed successfully")
            self._log_metrics(task, result, start_time)

            return result

        except Exception as e:
            self.logger.error(f"[{self.session_id}] Task failed: {e}", exc_info=True)
            raise

    def _log_metrics(self, task: str, result: Dict, start_time: datetime):
        """Log detailed metrics for analysis."""
        metrics = {
            "session_id": self.session_id,
            "task": task,
            "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "tokens_used": result.get("total_tokens", 0),
            "tools_called": len(result.get("tools_used", [])),
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        }

        # Log as JSON for easy parsing
        self.logger.info(f"[{self.session_id}] Metrics: {json.dumps(metrics)}")

        # Send to monitoring system (Prometheus, Datadog, etc.)
        self._send_to_monitoring(metrics)

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _send_to_monitoring(self, metrics: Dict):
        """Send metrics to monitoring system."""
        # Example: Push to Prometheus Pushgateway
        # Or send to Datadog, New Relic, etc.
        pass
```

---

## Lab 1: Build Tool-Calling Agent

**Goal**: Create V1 tool-calling agent for network queries

**Time**: 45 minutes
**Cost**: $0.15 (API calls)

### Setup

1. **Create project**:
```bash
mkdir network-agent && cd network-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install anthropic
```

2. **Set API key**:
```bash
export ANTHROPIC_API_KEY="your_key_here"
```

3. **Create `tools.py`**:
```python
# tools.py - Network query tools

def get_bgp_summary(hostname: str) -> dict:
    """Get BGP neighbor summary."""
    # Mock data - replace with real Netmiko in production
    return {
        "router_id": "10.0.0.1",
        "local_as": 65001,
        "neighbors": [
            {"ip": "10.0.1.2", "as": 65002, "state": "Established", "prefixes": 150},
            {"ip": "10.0.1.3", "as": 65003, "state": "Idle", "prefixes": 0}
        ]
    }

def get_interface_status(hostname: str, interface: str) -> dict:
    """Get interface status."""
    return {
        "interface": interface,
        "status": "up",
        "protocol": "up",
        "ip_address": "192.168.1.1"
    }

# Tool definitions for Claude
TOOLS = [
    {
        "name": "get_bgp_summary",
        "description": "Get BGP neighbor status",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string"}
            },
            "required": ["hostname"]
        }
    },
    {
        "name": "get_interface_status",
        "description": "Get interface status",
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string"},
                "interface": {"type": "string"}
            },
            "required": ["hostname", "interface"]
        }
    }
]
```

### Steps

**Step 1**: Create agent (use V1 code from above)

Save the `ToolCallingAgent` class from V1 section to `agent_v1.py`.

**Step 2**: Test simple query
```bash
python
>>> from agent_v1 import ToolCallingAgent
>>> from tools import TOOLS, get_bgp_summary, get_interface_status
>>> import os
>>>
>>> api_key = os.environ.get("ANTHROPIC_API_KEY")
>>> tool_functions = {
...     "get_bgp_summary": get_bgp_summary,
...     "get_interface_status": get_interface_status
... }
>>>
>>> agent = ToolCallingAgent(api_key, TOOLS, tool_functions)
>>> result = agent.run("What's the BGP status on router1?")
```

**Expected output**:
```
Query: What's the BGP status on router1?

Claude is calling 1 tool(s):
  • get_bgp_summary({"hostname": "router1"})
    → {'router_id': '10.0.0.1', 'local_as': 65001...

Final Answer:
BGP is operational on router1. Router ID 10.0.0.1, AS 65001.
Neighbors: 10.0.1.2 (Established), 10.0.1.3 (Idle - needs investigation).

Tools used: ['get_bgp_summary']
Tokens: 1247
Latency: 2.34s
```

**Step 3**: Test multi-tool query
```python
result = agent.run("Check BGP and interface Gi0/0 status on router1")
```

Should call both tools.

**Step 4**: Measure performance
```python
import time

queries = [
    "What's the BGP status?",
    "Check interface Gi0/0",
    "Show me routing table"
]

for query in queries:
    start = time.time()
    result = agent.run(query, verbose=False)
    print(f"{query}: {time.time() - start:.2f}s, {result['total_tokens']} tokens")
```

### Success Criteria

- [ ] Agent responds to simple queries
- [ ] Tool calls execute successfully
- [ ] Latency < 5s per query
- [ ] Token usage < 2000 per query
- [ ] Error handling works (try invalid tool name)

### Verification Questions

1. **What happens if Claude hallucinates a tool that doesn't exist?**
   <details>
   <summary>Answer</summary>

   The `_execute_tool` method checks if `tool_name` exists in `self.tool_functions`. If not, it returns an error message. Claude sees the error in the tool result and can retry with a valid tool or explain the limitation to the user.

   Test it:
   ```python
   # Modify TOOLS to include a fake tool
   # Claude might try to call it, but execution will fail gracefully
   ```
   </details>

2. **Why use `stop_reason == "tool_use"` instead of checking response content?**
   <details>
   <summary>Answer</summary>

   Claude's API explicitly signals when it wants to use tools via `stop_reason`. This is more reliable than parsing response text. The stop_reason can be:
   - `"tool_use"`: Claude wants to call tools
   - `"end_turn"`: Claude has final answer
   - `"max_tokens"`: Response was truncated

   Checking `stop_reason` is the canonical way to detect tool usage.
   </details>

3. **How would you add Netmiko for real device queries?**
   <details>
   <summary>Answer</summary>

   Replace mock functions with real Netmiko:
   ```python
   from netmiko import ConnectHandler

   def get_bgp_summary(hostname: str) -> dict:
       """Get real BGP summary via Netmiko."""
       device = {
           "device_type": "cisco_ios",
           "host": hostname,
           "username": os.getenv("NET_USER"),
           "password": os.getenv("NET_PASS")
       }

       with ConnectHandler(**device) as conn:
           output = conn.send_command("show ip bgp summary")
           # Parse output into structured dict
           return parse_bgp_output(output)
   ```

   Everything else stays the same—tools abstraction works for both mock and real data.
   </details>

---

## Lab 2: Add ReAct Reasoning Loop

**Goal**: Extend to V2 ReAct for exploratory diagnosis

**Time**: 60 minutes
**Cost**: $0.25

### Setup

Continue from Lab 1 directory.

1. **Add more tools** to `tools.py`:
```python
def check_recent_changes(hostname: str, hours: int = 24) -> list:
    """Check recent config changes."""
    return [
        {
            "timestamp": "2024-01-15 14:23:45",
            "user": "admin",
            "command": "interface GigabitEthernet0/1",
            "change": "shutdown"
        }
    ]

# Add to TOOLS list
TOOLS.append({
    "name": "check_recent_changes",
    "description": "Check recent configuration changes",
    "input_schema": {
        "type": "object",
        "properties": {
            "hostname": {"type": "string"},
            "hours": {"type": "integer", "description": "Hours to look back"}
        },
        "required": ["hostname"]
    }
})
```

2. **Create `agent_v2.py`** (use V2 code from above)

### Steps

**Step 1**: Test ReAct on simple task
```python
from agent_v2 import ReActAgent
from tools import get_bgp_summary, get_interface_status, check_recent_changes

agent = ReActAgent(
    api_key=api_key,
    tools={
        "get_bgp_summary": get_bgp_summary,
        "get_interface_status": get_interface_status,
        "check_recent_changes": check_recent_changes
    },
    max_iterations=6
)

result = agent.run("Why is BGP neighbor 10.0.1.3 down on router1?")
```

**Step 2**: Observe reasoning loop

Agent should:
1. Check BGP summary → See neighbor in Idle state
2. Check recent changes → Find interface shutdown
3. Check interface status → Confirm it's down
4. Provide answer with root cause

**Step 3**: Compare V1 vs V2

```python
# Same task with V1 tool-calling
v1_agent = ToolCallingAgent(api_key, TOOLS, tool_functions)
v1_result = v1_agent.run("Why is BGP neighbor 10.0.1.3 down?", verbose=False)

# V2 ReAct
v2_result = agent.run("Why is BGP neighbor 10.0.1.3 down?", verbose=False)

print(f"V1: {v1_result['total_tokens']} tokens, {v1_result['latency_seconds']:.2f}s")
print(f"V2: {v2_result['total_tokens']} tokens, {v2_result['latency_seconds']:.2f}s")
print(f"V1 answer: {v1_result['answer'][:100]}...")
print(f"V2 answer: {v2_result['answer'][:100]}...")
```

**Expected**:
- V1: Might give incomplete answer (no exploration)
- V2: Full root cause with evidence chain

**Step 4**: Test edge case - max iterations
```python
# Task that requires many steps
result = agent.run(
    "Diagnose connectivity issue between router1 and router2",
    verbose=True
)

# Should hit max_iterations if task is too complex
if not result["success"]:
    print(f"Failed after {result['iterations']} iterations")
```

### Success Criteria

- [ ] Agent completes diagnosis in 3-6 iterations
- [ ] Each iteration shows Thought → Action → Observation
- [ ] Final answer includes root cause + solution
- [ ] Latency 8-15s
- [ ] Tokens 2000-4000

### Verification Questions

1. **Why does ReAct use separate Thought/Action/Observation instead of just calling tools?**
   <details>
   <summary>Answer</summary>

   The explicit reasoning ("Thought") helps the LLM:
   - Form hypotheses before gathering data
   - Chain evidence logically
   - Avoid random tool calling
   - Explain its reasoning to users

   Without "Thought", the agent might call tools without clear purpose. The ReAct format enforces deliberate, explainable decision-making.
   </details>

2. **When would V1 tool-calling be better than V2 ReAct?**
   <details>
   <summary>Answer</summary>

   V1 is better for:
   - **Known queries**: "Show BGP status" - single tool call
   - **Speed requirements**: Production dashboards (2s vs 8s)
   - **Cost constraints**: 1200 tokens vs 2800 tokens
   - **Simple retrieval**: No exploration needed

   V2 is better for:
   - **Unknown problems**: "Why is X broken?"
   - **Multi-step diagnosis**: Need to test hypotheses
   - **Explanation required**: User wants to see reasoning
   </details>

3. **How do you prevent infinite loops in ReAct?**
   <details>
   <summary>Answer</summary>

   Multiple safeguards:

   1. **max_iterations**: Hard limit (default 10)
   2. **Answer detection**: Stop when agent says "Answer:"
   3. **Repeated actions**: Track if agent keeps calling same tool with same args
   4. **Token limit**: API max_tokens prevents runaway generation

   Example addition:
   ```python
   # Track action history
   if (tool_name, str(tool_args)) in recent_actions:
       print("⚠️  Repeated action detected - stopping")
       break
   recent_actions.add((tool_name, str(tool_args)))
   ```
   </details>

---

## Lab 3: Multi-Agent Coordination

**Goal**: Build V4 multi-agent system with supervisor

**Time**: 90 minutes
**Cost**: $0.40

### Setup

Continue from Lab 2.

1. **Create `agent_v3a.py`** (Plan-Execute from V3a section)

2. **Add deployment tools** to `tools.py`:
```python
def create_vlan(hostname: str, vlan_id: int, name: str) -> str:
    """Create VLAN on switch."""
    return f"Created VLAN {vlan_id} ({name}) on {hostname}"

def assign_vlan_to_ports(hostname: str, vlan_id: int, ports: list) -> str:
    """Assign VLAN to ports."""
    return f"Assigned VLAN {vlan_id} to {len(ports)} ports on {hostname}"

def save_config(hostname: str) -> str:
    """Save configuration."""
    return f"Configuration saved on {hostname}"
```

### Steps

**Step 1**: Initialize specialized agents
```python
from agent_v1 import ToolCallingAgent
from agent_v2 import ReActAgent
from agent_v3a import PlanExecuteAgent
from agent_v4 import SupervisorAgent
import os

api_key = os.environ.get("ANTHROPIC_API_KEY")

# Query agent (V1)
query_agent = ToolCallingAgent(api_key, TOOLS, tool_functions)

# Diagnosis agent (V2)
diagnosis_agent = ReActAgent(api_key, tool_functions)

# Config agent (V3)
config_agent = PlanExecuteAgent(api_key, deployment_tools)

# Supervisor (V4)
supervisor = SupervisorAgent(
    api_key=api_key,
    agents={
        "query": query_agent,
        "diagnosis": diagnosis_agent,
        "config": config_agent
    }
)
```

**Step 2**: Test routing
```python
# Simple query - should route to V1
result = supervisor.run("What's the BGP status on router1?")
print(f"Routed to: {result.get('agent_used')}")  # Should be 'query'

# Diagnosis - should route to V2
result = supervisor.run("Why is interface Gi0/1 down?")
print(f"Routed to: {result.get('agent_used')}")  # Should be 'diagnosis'

# Config task - should route to V3
result = supervisor.run("Create VLAN 100 on switch1")
print(f"Routed to: {result.get('agent_used')}")  # Should be 'config'
```

**Step 3**: Test multi-agent workflow
```python
# Complex task requiring multiple agents
result = supervisor.run(
    "Fix the BGP peering issue between router1 and 10.0.1.3"
)

# Should trigger multi-agent coordination:
# 1. Diagnosis agent finds root cause
# 2. Config agent generates fix
# 3. Returns for human approval
```

**Expected output**:
```
Task classified as: complex

Complex task detected - coordinating multiple agents

→ Step 1: Diagnosis Agent
[ReAct loop runs...]
Answer: Interface Gi0/1 is shutdown

→ Step 2: Config Agent
[Plan-Execute runs...]
Generated fix: no shutdown on Gi0/1

→ Step 3: Execution (APPROVAL REQUIRED)
⚠️  Human approval needed for production change
```

**Step 4**: Add approval gate
```python
# Modify SupervisorAgent._coordinate_multi_agent to add real approval:

def _get_approval(self, fix_description: str) -> bool:
    """Get human approval for fix."""
    print(f"\n{'='*60}")
    print("APPROVAL REQUEST")
    print(f"{'='*60}")
    print(fix_description)
    print(f"\n{'='*60}")

    response = input("Approve execution? (yes/no): ")
    return response.lower() == "yes"
```

**Step 5**: Measure multi-agent performance
```python
import time

tasks = [
    ("Simple", "Show BGP status"),
    ("Diagnosis", "Why is OSPF down?"),
    ("Config", "Create VLAN 50"),
    ("Complex", "Fix routing issue on router1")
]

for task_type, task in tasks:
    start = time.time()
    result = supervisor.run(task, verbose=False)
    latency = time.time() - start
    tokens = result.get("total_tokens", 0) + result.get("classification_tokens", 0)

    print(f"{task_type:10} | {latency:5.2f}s | {tokens:5} tokens | Agent: {result.get('agent_used', 'multi-agent')}")
```

### Success Criteria

- [ ] Supervisor correctly classifies tasks
- [ ] Simple queries route to V1 (<3s)
- [ ] Diagnosis routes to V2 (8-12s)
- [ ] Config tasks route to V3 (5-8s)
- [ ] Complex tasks trigger multi-agent (15-25s)
- [ ] Human approval gate works
- [ ] Total tokens < 5000 for complex workflow

### Verification Questions

1. **Why use Haiku for task classification instead of Sonnet?**
   <details>
   <summary>Answer</summary>

   Classification is simple pattern matching:
   - Input: Task description
   - Output: One of 4 categories

   Haiku ($0.25/MTok) is 12x cheaper than Sonnet ($3/MTok) and fast enough (1-2s) for this task. Using Sonnet would waste cost on a simple decision.

   Cost comparison for 1000 classifications:
   - Haiku: ~$0.05 (200 tokens avg)
   - Sonnet: ~$0.60 (200 tokens avg)

   Savings: $0.55 per 1000 tasks = $550/million tasks.
   </details>

2. **What happens if diagnosis fails but config succeeds?**
   <details>
   <summary>Answer</summary>

   The multi-agent coordinator checks success at each step:

   ```python
   if not diagnosis_result.get("success"):
       return {
           "success": False,
           "error": "Diagnosis failed",
           "diagnosis": diagnosis_result
       }
   ```

   If diagnosis fails, config agent never runs. This prevents:
   - Generating fix for unknown problem
   - Wasting tokens on wrong solution
   - Executing incorrect changes

   The workflow is sequential with failure handling at each step.
   </details>

3. **How would you add a verification step after execution?**
   <details>
   <summary>Answer</summary>

   Add a fourth agent (verification) that checks if fix worked:

   ```python
   # Step 4: Verify fix
   if execution_approved and executed:
       verify_result = self.agents["diagnosis"].run(
           task=f"Verify that the original problem is fixed: {original_task}"
       )

       if verify_result.get("success"):
           return {"status": "verified", "fix_successful": True}
       else:
           # Fix didn't work - rollback
           return {"status": "failed", "rollback_needed": True}
   ```

   This creates a closed-loop system: diagnose → fix → verify → confirm or rollback.
   </details>

---

## Check Your Understanding

Test your knowledge of agent architectures:

### Question 1: Pattern Selection

You need to build an agent for these tasks:

**Task A**: "Show me the routing table for 10.0.0.0/8"
**Task B**: "Why are we losing packets to 10.0.0.5?"
**Task C**: "Deploy OSPF configuration to all core routers"
**Task D**: "Generate optimal QoS policy for our VoIP traffic"

Match each task to the best pattern (V1/V2/V3/V4) and explain why.

<details>
<summary>Answer</summary>

**Task A → V1 Tool-Calling**:
- Simple information retrieval
- Single tool: `get_route_table(prefix="10.0.0.0/8")`
- No exploration needed
- Cost: $0.03, Latency: 2s

**Task B → V2 ReAct**:
- Diagnosis requiring exploration
- Unknown problem space
- Need to test hypotheses (MTU? Routing? Interface?)
- ReAct explores: check routes → check interface → check errors → find root cause
- Cost: $0.18, Latency: 9s

**Task C → V3 Plan-Execute**:
- Deployment with known structure
- Can plan all steps upfront:
  1. Backup configs
  2. Generate OSPF config for each router
  3. Deploy sequentially
  4. Verify neighbors
  5. Save configs
- Cost: $0.35, Latency: 12s

**Task D → V3 Reflexion**:
- Complex config generation
- First attempt likely fails (wrong bandwidth allocation, QoS class mapping)
- Need to learn from errors and iterate
- Try → Generate config → Test → Fails → Reflect → Adjust → Retry
- Cost: $0.65, Latency: 28s

**Key insight**: Match task structure to pattern strength.
</details>

---

### Question 2: Cost vs Performance Trade-off

Your production system handles 10,000 queries/day with this distribution:
- 70%: Simple status queries ("Show X")
- 20%: Troubleshooting ("Why is X broken?")
- 8%: Configuration tasks ("Deploy X")
- 2%: Complex incidents ("Fix and verify X")

You can choose:
- **Option A**: Use V2 ReAct for everything (consistent, handles all cases)
- **Option B**: Use V4 Multi-Agent (routes to optimal pattern per task)

Calculate monthly cost for each option and explain which is better.

<details>
<summary>Answer</summary>

**Option A: ReAct for Everything**

ReAct avg cost: $0.20/query

```
Monthly queries: 10,000 * 30 = 300,000
Monthly cost: 300,000 * $0.20 = $60,000
```

**Pros**:
- Simple (one agent to maintain)
- Consistent behavior

**Cons**:
- Overkill for simple queries (70% of traffic)
- Wastes tokens on exploration when answer is direct

**Option B: Multi-Agent with Routing**

```
Simple (70%): 210,000 * $0.05 = $10,500  (V1 Tool-calling)
Troubleshoot (20%): 60,000 * $0.20 = $12,000  (V2 ReAct)
Config (8%): 24,000 * $0.35 = $8,400  (V3 Plan-Execute)
Complex (2%): 6,000 * $0.70 = $4,200  (V4 full workflow)
Classification overhead: 300,000 * $0.01 = $3,000  (Haiku)

Total: $38,100/month
```

**Savings**: $60,000 - $38,100 = **$21,900/month** (37% reduction)

**Better choice: Option B**

Routing adds complexity but saves $263K/year. The 70% simple queries benefit most from V1's efficiency.

**ROI calculation**:
- Engineering time to build multi-agent: 40 hours * $150/hr = $6,000
- Payback period: $6,000 / $21,900/month = **8.2 days**

After 9 days, you're saving money. By end of year, you've saved $257K.

**Key insight**: Most production workloads have skewed distributions. Route simple tasks to simple agents.
</details>

---

### Question 3: Failure Handling

Your ReAct agent (V2) is diagnosing a BGP issue. After 5 iterations:

**Iteration 1-3**: Checked BGP summary, interface status, recent changes
**Iteration 4**: Tried to call `get_routing_policy` but tool doesn't exist → ERROR
**Iteration 5**: Tried to call `get_routing_policy` again → Same ERROR

What went wrong and how do you fix it?

<details>
<summary>Answer</summary>

**What went wrong**:

The agent is stuck in a loop trying to call a non-existent tool. This happens because:
1. LLM hallucinated a tool name based on task context
2. Received error, but didn't learn from it
3. Repeated the same mistake

**Why it repeated**:
- Error message wasn't clear enough ("Tool not found")
- Agent's prompt didn't emphasize learning from errors
- No explicit "don't retry failed tools" instruction

**Fix #1: Better error messages**

```python
def _execute_tool(self, tool_name: str, args: Dict) -> str:
    if tool_name not in self.tools:
        available = ", ".join(self.tools.keys())
        return f"""ERROR: Tool '{tool_name}' does not exist.

Available tools: {available}

Do NOT retry this tool. Choose from available tools only."""

    # ...
```

**Fix #2: Track failed attempts**

```python
def run(self, task: str) -> Dict:
    failed_tools = set()  # Track failures

    # ...

    if parsed["type"] == "action":
        tool_name = parsed["tool"]

        if tool_name in failed_tools:
            observation = f"ERROR: Already tried {tool_name} and it failed. Try different approach."
        else:
            observation = self._execute_tool(tool_name, tool_args)

            if "ERROR" in observation:
                failed_tools.add(tool_name)
```

**Fix #3: Improve system prompt**

```python
prompt = f"""...

CRITICAL: If a tool call fails with ERROR, do NOT retry the same tool.
- Analyze the error
- Choose a different approach
- Use only tools from the available list

..."""
```

**Prevention**:
- Use Claude's native tool calling (validates tools before execution)
- Add tool existence validation in prompt
- Track and penalize repeated failures

**Key insight**: LLM agents need explicit error handling instructions. They don't automatically learn from mistakes without guidance.
</details>

---

### Question 4: Production Safety

You're deploying a V4 multi-agent system to production. The config agent generates this command:

```
interface GigabitEthernet0/0
 shutdown
 no ip address
```

Your safety checks:
1. Whitelist check: `configure_interface` is allowed ✓
2. Production flag: Requires approval ✓
3. Dangerous pattern detection: ...?

Should this be blocked? Why or why not?

<details>
<summary>Answer</summary>

**YES - Should be blocked immediately**

**Why**:

1. **`shutdown` is dangerous**:
   - Takes interface down
   - Breaks connectivity
   - Could isolate device

2. **`no ip address` removes IP**:
   - Loses management access
   - Can't undo remotely
   - Requires console access to fix

3. **Combined = catastrophic**:
   - Device becomes unreachable
   - Manual intervention needed
   - Potential production outage

**Proper safety checks**:

```python
class SafeAgent:
    def __init__(self):
        self.dangerous_patterns = [
            "shutdown",           # Takes interfaces down
            "no ip address",      # Removes IP
            "write erase",        # Erases config
            "reload",             # Reboots device
            "no ",                # Any "no" command (be selective)
            "default interface",  # Resets to default
            "clear",              # Clears sessions/state
        ]

        self.requires_approval = [
            "interface.*shutdown",     # Interface shutdown
            "no ip",                   # IP removal
            "router bgp.*no ",         # BGP changes
            "spanning-tree.*shutdown", # STP changes
        ]

    def check_command(self, command: str) -> tuple[bool, str]:
        """
        Check if command is safe.

        Returns: (is_safe, reason)
        """
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Blocked: Contains dangerous pattern '{pattern}'"

        # Check if approval needed
        for pattern in self.requires_approval:
            if re.search(pattern, command, re.IGNORECASE):
                approved = self._get_human_approval(command)
                return approved, "Human approval required"

        return True, "Safe"
```

**Better approach**:

Instead of generating raw commands, use structured tools:

```python
# Bad: Free-form commands
{"tool": "execute_command", "args": {"command": "shutdown"}}

# Good: Structured, validated tools
{"tool": "disable_interface_safely", "args": {
    "interface": "Gi0/0",
    "backup_first": True,
    "verify_redundancy": True,
    "rollback_timer": 300  # Auto-rollback in 5min if not confirmed
}}
```

**Production safeguards**:
1. **Whitelist** allowed operations
2. **Blacklist** dangerous patterns
3. **Approval gates** for sensitive changes
4. **Rollback timers** (auto-undo if not confirmed)
5. **Change windows** (only allow changes during maintenance)
6. **Dry-run mode** (simulate without executing)

**Key insight**: Never trust LLM-generated commands directly. Always validate, require approval for destructive operations, and implement rollback mechanisms.
</details>

---

## Lab Time Budget

**Total time**: 3 hours 15 minutes
**Total cost**: $0.80 (API calls)

| Lab | Time | Cost | Skills Gained | ROI |
|-----|------|------|---------------|-----|
| Lab 1: Tool-Calling | 45 min | $0.15 | V1 baseline, tool definitions, Claude API | Deploy in 1 hour, handle 80% of queries |
| Lab 2: ReAct Loop | 60 min | $0.25 | V2 reasoning, exploration, diagnosis | Automate troubleshooting ($200/hour engineer time) |
| Lab 3: Multi-Agent | 90 min | $0.40 | V4 coordination, routing, safety | Full incident response ($500/incident) |

**Production value**:

**Scenario**: Network operations team (5 engineers, $150/hr loaded cost)

**Before agents**:
- Simple queries: 10 min/query * 100/day = 16.7 hours/day
- Troubleshooting: 2 hours/incident * 5/day = 10 hours/day
- Config deployment: 4 hours/deployment * 2/day = 8 hours/day
- **Total**: 34.7 hours/day = $5,205/day = **$1.9M/year**

**After agents** (handles 70% of work):
- Agent costs: $38,100/month = $457K/year
- Reduced engineer time: 70% automation = save $1.33M/year
- **Net savings**: $1.33M - $0.46M = **$873K/year**

**ROI**: One-time 3.25-hour investment saves $873K/year = **2,686x return** in first year.

**Additional benefits**:
- 24/7 operation (agents don't sleep)
- Consistent troubleshooting (no variability)
- Instant response (vs 15-min human response time)
- Full audit trail (every decision logged)

**Cost breakdown by pattern**:

| Pattern | Cost/Query | Queries/Day | Monthly Cost | Use Case |
|---------|------------|-------------|--------------|----------|
| V1 Tool | $0.05 | 7,000 | $10,500 | Status checks |
| V2 ReAct | $0.20 | 2,000 | $12,000 | Diagnosis |
| V3 Plan-Execute | $0.35 | 800 | $8,400 | Deployments |
| V3 Reflexion | $0.60 | 100 | $1,800 | Complex config |
| V4 Multi-Agent | $0.70 | 200 | $4,200 | Incidents |
| **Total** | - | **10,100** | **$38,100** | - |

Plus classification overhead: 10,100 * $0.01 = $3,000/month

**Grand total**: $38,100 + $3,000 = **$41,100/month**

---

## Key Takeaways

1. **Start simple**: Use V1 tool-calling for 80% of tasks. Add complexity only when needed.

2. **Pattern selection matters**:
   - V1: Known queries (95% success, $0.05)
   - V2: Unknown diagnosis (90% success, $0.20)
   - V3: Structured workflows (85% success, $0.35)
   - V4: Complex multi-skill (98% success, $0.70)

3. **Cost vs capability**: More complex agents cost more but handle harder tasks. Route intelligently.

4. **Safety is critical**:
   - Whitelist allowed operations
   - Block dangerous commands
   - Require human approval for production
   - Implement rollback mechanisms

5. **Observability**: Log every decision, tool call, and outcome for debugging and compliance.

6. **Progressive enhancement**:
   - Week 1: Deploy V1 for status queries
   - Week 2: Add V2 for troubleshooting
   - Week 3: Add V3 for deployments
   - Week 4: Integrate V4 multi-agent

7. **Real-world performance**:
   - V1: 2s latency, 1.2K tokens
   - V2: 8s latency, 2.8K tokens
   - V3: 12s latency, 3.5K tokens
   - V4: 20s latency, 5.2K tokens

**Production checklist**:
- [ ] Error handling for API failures
- [ ] Safety guardrails for dangerous operations
- [ ] Human approval gates for production changes
- [ ] Comprehensive logging (who/what/when/why)
- [ ] Rate limiting and retry logic
- [ ] Operation whitelisting
- [ ] Dangerous command detection
- [ ] Rollback mechanisms
- [ ] Monitoring and alerting
- [ ] Cost tracking per pattern

**Next chapter**: We'll take these agent patterns and build a complete autonomous troubleshooting system that combines diagnosis, root cause analysis, fix generation, and verification—all with production safety guardrails.

---

## What Can Go Wrong?

**1. Agent runs forever without completing**
- **Cause**: No termination condition, stuck in reasoning loop
- **Fix**: Set `max_iterations`, add explicit "Answer:" detection, track repeated actions

**2. Agent hallucinates tools**
- **Cause**: LLM invents tool names based on task context
- **Fix**: Return clear "Tool not found" with available tools list, use strict parsing

**3. Plan-Execute fails midway**
- **Cause**: Plan didn't account for errors, no rollback
- **Fix**: Add error handling per step, implement transaction rollback, use Reflexion for retry

**4. Tool-calling can't handle complex tasks**
- **Cause**: Task requires reasoning that tool-calling doesn't support
- **Fix**: Route to ReAct for exploration, use multi-agent for multi-step workflows

**5. Reflexion wastes tokens retrying same mistake**
- **Cause**: Agent doesn't learn from reflections, poor reflection prompt
- **Fix**: Improve reflection prompt with "what NOT to try", track failed approaches

**6. Multi-agent makes wrong routing decision**
- **Cause**: Classification prompt is ambiguous, edge cases unclear
- **Fix**: Add examples to classification prompt, log decisions for analysis

**7. Agent executes dangerous commands**
- **Cause**: No safety checks, bypasses guardrails
- **Fix**: Whitelist operations, blacklist patterns, require approval, implement rollback

**8. High cost from using wrong pattern**
- **Cause**: Using ReAct for simple queries, V1 for complex diagnosis
- **Fix**: Implement routing (V4 supervisor), monitor cost per pattern, optimize distribution

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-19/`
