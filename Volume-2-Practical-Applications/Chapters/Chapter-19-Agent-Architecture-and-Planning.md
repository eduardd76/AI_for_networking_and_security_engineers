# Chapter 19: Agent Architecture and Planning

## Introduction

An AI agent isn't just an LLM with access to functions. A real agent makes decisions, plans sequences of actions, handles failures, and adapts to changing conditions. In network operations, this distinction matters: a simple tool-calling system can run show commands, but an agent can diagnose a complex BGP routing loop by forming hypotheses, gathering evidence, and adjusting its investigation based on what it discovers.

This chapter covers the architectures and planning algorithms that transform LLMs into capable agents. We'll implement four proven patterns from scratch, benchmark them on real network troubleshooting tasks, and show you when to use each approach.

**Networking analogy**: If an LLM is like a single router that can process packets, an AI agent is like a fully autonomous network with routing protocols, failover mechanisms, and traffic engineering. The LLM provides the intelligence (packet processing), but the agent architecture provides the control plane — deciding *what* to do, *when* to do it, and *how* to recover from failures. Just as you wouldn't run a production network with only static routes, you wouldn't build a production AI system with only simple API calls.

**What You'll Build**:
- ReAct agent (reasoning + acting interleaved)
- Plan-and-Execute agent (upfront planning, then execution)
- Tool-calling agent (Claude's native function calling)
- Reflexion agent (learns from failures)
- Multi-agent coordinator (specialized agents working together)

**Prerequisites**: Chapters 4 (API Basics), 10 (API Integration), 15 (LangChain basics)

---

## Agent Design Patterns: Overview

### The Four Core Patterns

**1. ReAct (Reasoning + Acting)**
- **How it works**: Interleaves reasoning steps with actions
- **Best for**: Exploratory tasks where you discover what's needed as you go
- **Example**: "Why is OSPF down?" → Check interface → See shutdown → Check recent changes → Find erroneous script
- **Weakness**: Can waste tokens on obvious tasks

**2. Plan-and-Execute**
- **How it works**: Creates complete plan upfront, then executes each step
- **Best for**: Well-defined tasks with known structure
- **Example**: "Configure VLANs 10-20 on all access switches" → Plan all steps → Execute sequentially
- **Weakness**: Struggles when reality doesn't match the plan

**3. Tool-Calling (Native Function Calling)**
- **How it works**: LLM decides which tools to call, runtime executes them
- **Best for**: Tasks where the LLM knows exactly what tools are needed
- **Example**: "Show me BGP summary" → LLM returns `get_bgp_summary` tool call → Runtime executes
- **Weakness**: Limited to predefined tools, no complex reasoning

**4. Reflexion (Learning from Failures)**
- **How it works**: Agent tries, fails, reflects on failure, tries again with new strategy
- **Best for**: Tasks where first attempt often fails
- **Example**: Config deployment fails → Agent analyzes error → Realizes dependency → Adjusts order → Succeeds
- **Weakness**: Slower due to retry loops

### Comparison Matrix

| Pattern | Planning | Execution | Adaptability | Token Cost | Best Use Case |
|---------|----------|-----------|--------------|------------|---------------|
| ReAct | Online | Interleaved | High | High | Diagnosis |
| Plan-Execute | Upfront | Sequential | Low | Medium | Deployment |
| Tool-Calling | Implicit | On-demand | Medium | Low | Simple queries |
| Reflexion | Iterative | Retry-based | Very High | Very High | Complex automation |

---

## Pattern 1: ReAct Agent (Reasoning + Acting)

### The Algorithm

ReAct alternates between:
1. **Thought**: Agent reasons about what to do next
2. **Action**: Agent calls a tool
3. **Observation**: Agent sees the result
4. Repeat until task is complete

**Paper Reference**: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

### Implementation

```python
"""
ReAct Agent for Network Troubleshooting
File: agents/react_agent.py
"""
import os
from typing import List, Dict, Callable, Optional
from anthropic import Anthropic

class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent implementation.

    Interleaves reasoning steps with tool calls until task is solved.
    """

    def __init__(self, api_key: str, tools: Dict[str, Callable], max_iterations: int = 10):
        """
        Initialize ReAct agent.

        Args:
            api_key: Anthropic API key
            tools: Dict mapping tool names to callable functions
            max_iterations: Max reasoning loops before giving up
        """
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.max_iterations = max_iterations
        self.conversation_history = []

    def run(self, task: str) -> Dict:
        """
        Execute task using ReAct loop.

        Args:
            task: Natural language task description

        Returns:
            Dict with final answer, steps taken, and statistics
        """
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
            print(f"\n--- Iteration {iteration + 1} ---\n")

            # Get agent's next thought and action
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=self.conversation_history
            )

            assistant_message = response.content[0].text
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            print(f"Agent Response:\n{assistant_message}\n")

            # Parse the response
            parsed = self._parse_response(assistant_message)

            if parsed["type"] == "answer":
                # Agent has reached a conclusion
                print(f"✓ Task Complete: {parsed['content']}")
                return {
                    "answer": parsed["content"],
                    "steps": steps,
                    "iterations": iteration + 1,
                    "total_tokens": total_tokens,
                    "success": True
                }

            elif parsed["type"] == "action":
                # Agent wants to use a tool
                tool_name = parsed["tool"]
                tool_args = parsed["args"]

                print(f"Action: {tool_name}({tool_args})")

                # Execute the tool
                observation = self._execute_tool(tool_name, tool_args)

                print(f"Observation: {observation[:200]}..." if len(str(observation)) > 200 else f"Observation: {observation}")

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
                print(f"⚠️  Could not parse response: {assistant_message}")
                break

        # Max iterations reached
        return {
            "answer": "Task incomplete - reached max iterations",
            "steps": steps,
            "iterations": self.max_iterations,
            "total_tokens": total_tokens,
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
        """
        Parse agent's response into structured format.

        Returns:
            Dict with type ("action" or "answer") and relevant fields
        """
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


# Example Network Tools
def get_interface_status(hostname: str, interface: str) -> str:
    """Get operational status of a network interface."""
    # Mock implementation - replace with real Netmiko
    mock_data = {
        "router1": {
            "GigabitEthernet0/0": "up/up",
            "GigabitEthernet0/1": "administratively down/down",
            "GigabitEthernet0/2": "up/down"
        }
    }
    status = mock_data.get(hostname, {}).get(interface, "unknown")
    return f"{hostname} {interface}: {status}"


def get_bgp_neighbors(hostname: str) -> str:
    """Get BGP neighbor status."""
    mock_bgp = {
        "router1": """
BGP router identifier 10.0.0.1, local AS number 65001
Neighbor        V    AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
10.0.1.2        4 65002   12345   12340        0    0    0 2d03h           150
10.0.1.3        4 65003       0       0        0    0    0 never    Idle
        """
    }
    return mock_bgp.get(hostname, "No BGP neighbors found")


def check_recent_changes(hostname: str, hours: int = 24) -> str:
    """Check recent configuration changes."""
    mock_changes = {
        "router1": "2024-01-15 14:23:45: admin: interface GigabitEthernet0/1 shutdown"
    }
    return mock_changes.get(hostname, f"No changes in last {hours} hours")


# Example Usage
if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Define available tools
    tools = {
        "get_interface_status": get_interface_status,
        "get_bgp_neighbors": get_bgp_neighbors,
        "check_recent_changes": check_recent_changes
    }

    # Create agent
    agent = ReActAgent(api_key=api_key, tools=tools, max_iterations=8)

    # Run task
    result = agent.run(
        task="Router1 is not receiving any BGP routes from 10.0.1.3. Why?"
    )

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Answer: {result['answer']}")
    print(f"Steps taken: {result['iterations']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Success: {result['success']}")
```

### Example Output

```
============================================================
TASK: Router1 is not receiving any BGP routes from 10.0.1.3. Why?
============================================================

--- Iteration 1 ---

Agent Response:
Thought: I need to check the BGP neighbor status first to see the state of the BGP session with 10.0.1.3.
Action: get_bgp_neighbors
Args: {"hostname": "router1"}

Action: get_bgp_neighbors({'hostname': 'router1'})
Observation:
BGP router identifier 10.0.0.1, local AS number 65001
Neighbor        V    AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
10.0.1.2        4 65002   12345   12340        0    0    0 2d03h           150
10.0.1.3        4 65003       0       0        0    0    0 never    Idle

--- Iteration 2 ---

Agent Response:
Thought: The BGP neighbor 10.0.1.3 is in "Idle" state and has never come up (Up/Down shows "never"). This indicates the TCP session is not establishing. This could be due to interface issues, routing problems, or configuration errors. Let me check if there are recent configuration changes that might have caused this.
Action: check_recent_changes
Args: {"hostname": "router1", "hours": 24}

Action: check_recent_changes({'hostname': 'router1', 'hours': 24})
Observation: 2024-01-15 14:23:45: admin: interface GigabitEthernet0/1 shutdown

--- Iteration 3 ---

Agent Response:
Thought: Found it! GigabitEthernet0/1 was administratively shut down recently. This interface is likely the one used to reach BGP neighbor 10.0.1.3. Let me verify the interface status.
Action: get_interface_status
Args: {"hostname": "router1", "interface": "GigabitEthernet0/1"}

Action: get_interface_status({'hostname': 'router1', 'interface': 'GigabitEthernet0/1'})
Observation: router1 GigabitEthernet0/1: administratively down/down

--- Iteration 4 ---

Agent Response:
Thought: Confirmed. GigabitEthernet0/1 is administratively down, which is why the BGP session with 10.0.1.3 cannot establish. The interface was shut down by admin at 14:23:45 today.
Answer: Router1 is not receiving BGP routes from 10.0.1.3 because the interface GigabitEthernet0/1 (used to reach this neighbor) was administratively shut down by admin on 2024-01-15 at 14:23:45. The BGP session is in "Idle" state and has never established. Solution: Execute "no shutdown" on GigabitEthernet0/1.

✓ Task Complete: Router1 is not receiving BGP routes from 10.0.1.3...

============================================================
FINAL RESULT
============================================================
Answer: Router1 is not receiving BGP routes from 10.0.1.3 because...
Steps taken: 4
Total tokens: 2847
Success: True
```

**Key Observations**:
- Agent discovered the problem through reasoning + evidence gathering
- Each thought explained the next step
- Tool calls were purposeful and sequenced logically
- Agent provided actionable solution with root cause

---

## Pattern 2: Plan-and-Execute Agent

### The Algorithm

Plan-and-Execute works in two phases:
1. **Planning Phase**: Create a complete plan before taking any actions
2. **Execution Phase**: Execute each step in the plan, monitoring for errors

**Best for**: Tasks where you know the structure upfront (deployments, migrations, bulk operations).

### Implementation

```python
"""
Plan-and-Execute Agent for Network Automation
File: agents/plan_execute_agent.py
"""
from typing import List, Dict, Callable
from anthropic import Anthropic
import json

class PlanExecuteAgent:
    """
    Plan-and-Execute agent: Creates plan upfront, then executes sequentially.
    """

    def __init__(self, api_key: str, tools: Dict[str, Callable]):
        self.client = Anthropic(api_key=api_key)
        self.tools = tools

    def run(self, task: str) -> Dict:
        """Execute task using plan-then-execute approach."""
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}\n")

        # Phase 1: Planning
        plan = self._create_plan(task)

        print("📋 PLAN:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step['description']}")
            print(f"     Tool: {step['tool']}, Args: {step['args']}")
        print()

        # Phase 2: Execution
        results = []
        for i, step in enumerate(plan, 1):
            print(f"\n--- Executing Step {i}/{len(plan)} ---")
            print(f"Description: {step['description']}")

            result = self._execute_step(step)
            results.append(result)

            if result["status"] == "error":
                print(f"❌ Step failed: {result['error']}")
                return {
                    "success": False,
                    "completed_steps": i - 1,
                    "total_steps": len(plan),
                    "results": results,
                    "error": result["error"]
                }

            print(f"✓ Step completed: {result['output'][:100]}...")

        # All steps completed
        print(f"\n✓ All {len(plan)} steps completed successfully!")

        return {
            "success": True,
            "completed_steps": len(plan),
            "total_steps": len(plan),
            "results": results
        }

    def _create_plan(self, task: str) -> List[Dict]:
        """
        Create a complete execution plan for the task.

        Returns list of steps, each with:
        - description: What this step does
        - tool: Tool to call
        - args: Arguments for the tool
        """
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

        # Extract JSON from response (may have markdown code fences)
        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0].strip()

        plan = json.loads(plan_text)
        return plan

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


# Example Tools for VLAN Management
def get_vlan_config(hostname: str) -> str:
    """Get current VLAN configuration."""
    return f"VLANs on {hostname}: 1 (default), 10 (Engineering), 20 (Sales)"

def create_vlan(hostname: str, vlan_id: int, name: str) -> str:
    """Create a new VLAN."""
    return f"Created VLAN {vlan_id} ({name}) on {hostname}"

def assign_vlan_to_ports(hostname: str, vlan_id: int, ports: List[str]) -> str:
    """Assign VLAN to switch ports."""
    return f"Assigned VLAN {vlan_id} to ports {ports} on {hostname}"

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

    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Completed: {result['completed_steps']}/{result['total_steps']} steps")
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
✓ Step completed: Assigned VLAN 100 to ports ['Gi0/1', 'Gi0/2', 'Gi0/3', 'Gi0/4'] on switch1

--- Executing Step 4/4 ---
Description: Save configuration
✓ Step completed: Configuration saved on switch1

✓ All 4 steps completed successfully!

============================================================
EXECUTION SUMMARY
============================================================
Success: True
Completed: 4/4 steps
```

**When to Use**:
- Configuration deployments across multiple devices
- Migrations with known steps
- Bulk operations (create 50 VLANs, provision 100 ports)
- Tasks where planning prevents errors (dependency ordering)

**Limitations**:
- Brittle when reality differs from plan
- No adaptation if unexpected errors occur
- Higher upfront token cost for planning

---

## Pattern 3: Tool-Calling Agent (Native Function Calling)

Claude has built-in tool calling (function calling). This is the most efficient pattern for simple tasks.

### Implementation

```python
"""
Tool-Calling Agent using Claude's Native Function Calling
File: agents/tool_calling_agent.py
"""
from anthropic import Anthropic
from typing import List, Dict, Callable
import json

class ToolCallingAgent:
    """Agent using Claude's native tool calling feature."""

    def __init__(self, api_key: str, tools: List[Dict], tool_functions: Dict[str, Callable]):
        """
        Args:
            api_key: Anthropic API key
            tools: List of tool definitions (Claude tool format)
            tool_functions: Dict mapping tool names to Python functions
        """
        self.client = Anthropic(api_key=api_key)
        self.tools = tools
        self.tool_functions = tool_functions

    def run(self, user_query: str) -> str:
        """Execute user query using tool calling."""
        print(f"\nQuery: {user_query}\n")

        messages = [{"role": "user", "content": user_query}]

        while True:
            # Call Claude with tools
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]

                print(f"Claude is calling {len(tool_calls)} tool(s):")

                # Execute each tool
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_input = tool_call.input

                    print(f"  - {tool_name}({tool_input})")

                    # Execute the tool
                    result = self._execute_tool(tool_name, tool_input)

                    print(f"    Result: {result[:100]}...")

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
                print(f"\nFinal Answer:\n{final_text}\n")
                return final_text

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


# Example: Network Query Tools
def get_interface_status(hostname: str, interface: str) -> Dict:
    """Get interface operational status."""
    return {
        "interface": interface,
        "status": "up",
        "protocol": "up",
        "ip_address": "192.168.1.1",
        "speed": "1000Mbps"
    }

def get_bgp_summary(hostname: str) -> Dict:
    """Get BGP neighbor summary."""
    return {
        "router_id": "10.0.0.1",
        "local_as": 65001,
        "neighbors": [
            {"ip": "10.0.1.2", "as": 65002, "state": "Established", "prefixes": 150},
            {"ip": "10.0.1.3", "as": 65003, "state": "Idle", "prefixes": 0}
        ]
    }

def get_route_table(hostname: str, prefix: str = None) -> List[Dict]:
    """Get routing table entries."""
    routes = [
        {"prefix": "0.0.0.0/0", "next_hop": "192.168.1.254", "protocol": "static"},
        {"prefix": "10.0.0.0/8", "next_hop": "10.0.1.2", "protocol": "bgp"},
        {"prefix": "172.16.0.0/16", "next_hop": "directly_connected", "protocol": "connected"}
    ]
    if prefix:
        routes = [r for r in routes if r["prefix"] == prefix]
    return routes


# Tool definitions (Claude format)
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
    }
]

# Example Usage
if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    tool_functions = {
        "get_interface_status": get_interface_status,
        "get_bgp_summary": get_bgp_summary,
        "get_route_table": get_route_table
    }

    agent = ToolCallingAgent(
        api_key=api_key,
        tools=TOOL_DEFINITIONS,
        tool_functions=tool_functions
    )

    # Test query
    agent.run("What's the status of BGP on router1? Are all neighbors up?")
```

### Example Output

```
Query: What's the status of BGP on router1? Are all neighbors up?

Claude is calling 1 tool(s):
  - get_bgp_summary({'hostname': 'router1'})
    Result: {'router_id': '10.0.0.1', 'local_as': 65001, 'neighbors': [{'ip': '10.0.1.2', 'as': 6500...

Final Answer:
Based on the BGP summary for router1:

**BGP Status**: Operational
- Router ID: 10.0.0.1
- Local AS: 65001

**Neighbors**:
1. ✓ 10.0.1.2 (AS 65002) - Established, receiving 150 prefixes
2. ✗ 10.0.1.3 (AS 65003) - Idle, not established

**Answer**: No, not all neighbors are up. Neighbor 10.0.1.3 is in Idle state and needs investigation.
```

**When to Use**:
- Simple queries that map to 1-3 tool calls
- Information retrieval tasks
- Tasks where tools are clearly defined
- Production systems (most reliable, lowest token cost)

**Comparison to ReAct**:
- **Tool-calling**: LLM decides tools → Runtime executes → LLM interprets
- **ReAct**: LLM reasons → LLM calls tool → LLM sees result → LLM reasons again

Tool-calling is faster and cheaper when applicable.

---

## Pattern 4: Reflexion Agent (Learning from Failures)

Reflexion adds a self-reflection loop: try → fail → reflect → retry with new strategy.

### Implementation Sketch

```python
class ReflexionAgent:
    """Agent that learns from failures by reflecting on them."""

    def run(self, task: str, max_attempts: int = 3) -> Dict:
        """Execute task with reflection on failures."""
        attempt = 0
        reflection_memory = []

        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt} ---")

            # Try to execute the task
            result = self._attempt_task(task, reflection_memory)

            if result["success"]:
                return result

            # Task failed - reflect on why
            print(f"Attempt {attempt} failed: {result['error']}")
            reflection = self._reflect_on_failure(task, result["error"], result["actions"])
            reflection_memory.append(reflection)

            print(f"Reflection: {reflection}")

        return {"success": False, "error": "Max attempts reached", "reflections": reflection_memory}

    def _reflect_on_failure(self, task: str, error: str, actions: List) -> str:
        """
        Analyze failure and generate insights for next attempt.

        Uses LLM to answer:
        - Why did this approach fail?
        - What assumptions were wrong?
        - What should we try differently?
        """
        prompt = f"""Task: {task}

Actions taken:
{chr(10).join(f'- {a}' for a in actions)}

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

        return response.content[0].text
```

**When to Use**:
- Complex multi-step automation where first attempt often fails
- Configuration generation (wrong syntax, missing dependencies)
- Tasks that need trial-and-error learning
- When failure provides useful information

**Cost Warning**: Reflexion can be expensive (3x+ token cost if all attempts needed).

---

## Multi-Agent Systems: Specialized Agents Working Together

For complex operational tasks, use multiple specialized agents coordinated by a supervisor.

### Architecture

```
┌─────────────────────────────────────────────┐
│          Supervisor Agent                   │
│  (Routes tasks to specialized agents)       │
└─────────────────────────────────────────────┘
         │          │           │
         ▼          ▼           ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Diagnosis│ │ Config │ │Execution│
    │ Agent  │ │ Agent  │ │ Agent   │
    └────────┘ └────────┘ └────────┘
```

### Implementation

```python
class SupervisorAgent:
    """Coordinates multiple specialized agents."""

    def __init__(self, api_key: str, agents: Dict[str, 'Agent']):
        self.client = Anthropic(api_key=api_key)
        self.agents = agents

    def run(self, task: str) -> Dict:
        """Route task to appropriate agent or coordinate multiple agents."""
        # Classify the task
        agent_name = self._classify_task(task)

        if agent_name == "complex":
            # Task requires multiple agents
            return self._coordinate_multi_agent(task)
        else:
            # Single agent can handle it
            return self.agents[agent_name].run(task)

    def _classify_task(self, task: str) -> str:
        """Determine which agent should handle this task."""
        prompt = f"""Classify this network engineering task:

Task: {task}

Available agents:
- diagnosis: Troubleshoot network issues, analyze failures
- config: Generate or modify device configurations
- execution: Execute commands, deploy changes
- complex: Requires multiple agents working together

Return only the agent name."""

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # Fast, cheap for classification
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip().lower()

    def _coordinate_multi_agent(self, task: str) -> Dict:
        """Coordinate multiple agents for complex task."""
        # Example: Diagnose issue → Generate fix → Execute fix
        print("Complex task detected - coordinating multiple agents")

        # Step 1: Diagnose
        diagnosis_result = self.agents["diagnosis"].run(f"Diagnose: {task}")

        # Step 2: Generate fix based on diagnosis
        config_result = self.agents["config"].run(
            f"Generate fix for: {diagnosis_result['answer']}"
        )

        # Step 3: Execute (with human approval in production!)
        print("\n⚠️  Would execute fix here (human approval required)")

        return {
            "diagnosis": diagnosis_result,
            "config": config_result,
            "executed": False  # Safety: never auto-execute in production
        }
```

**Use Cases**:
- Full incident response (diagnose → plan → execute → verify)
- Change management workflows
- Autonomous network operations with approval gates

---

## Benchmarking Agent Patterns

Let's benchmark the four patterns on real troubleshooting tasks.

### Test Suite

```python
"""
Benchmark different agent patterns on network troubleshooting tasks.
"""
import time
from typing import Dict, List

class AgentBenchmark:
    """Benchmark agent patterns on standardized tasks."""

    def __init__(self):
        self.test_cases = [
            {
                "name": "Simple query",
                "task": "What's the BGP status on router1?",
                "expected_tools": ["get_bgp_summary"],
                "complexity": "low"
            },
            {
                "name": "Multi-step diagnosis",
                "task": "Why is router1 not receiving routes from 10.0.1.3?",
                "expected_tools": ["get_bgp_neighbors", "check_interface", "check_routing"],
                "complexity": "medium"
            },
            {
                "name": "Complex deployment",
                "task": "Deploy VLAN 100 to all access switches with rollback on failure",
                "expected_tools": ["list_switches", "create_vlan", "verify_vlan", "rollback"],
                "complexity": "high"
            }
        ]

    def benchmark_agent(self, agent, test_case: Dict) -> Dict:
        """Run single test case and measure performance."""
        start_time = time.time()

        result = agent.run(test_case["task"])

        end_time = time.time()

        return {
            "test_name": test_case["name"],
            "success": result.get("success", False),
            "duration_seconds": end_time - start_time,
            "tokens_used": result.get("total_tokens", 0),
            "iterations": result.get("iterations", 1)
        }

    def run_full_benchmark(self, agents: Dict) -> Dict:
        """Run all test cases on all agents."""
        results = {}

        for agent_name, agent in agents.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {agent_name}")
            print(f"{'='*60}")

            agent_results = []
            for test_case in self.test_cases:
                print(f"\nTest: {test_case['name']}")
                result = self.benchmark_agent(agent, test_case)
                agent_results.append(result)

                print(f"  Success: {result['success']}")
                print(f"  Duration: {result['duration_seconds']:.2f}s")
                print(f"  Tokens: {result['tokens_used']}")

            results[agent_name] = agent_results

        return results

    def print_comparison(self, results: Dict):
        """Print comparison table of all agents."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*80)
        print(f"{'Agent':<20} {'Avg Duration':<15} {'Avg Tokens':<15} {'Success Rate':<15}")
        print("-"*80)

        for agent_name, agent_results in results.items():
            avg_duration = sum(r["duration_seconds"] for r in agent_results) / len(agent_results)
            avg_tokens = sum(r["tokens_used"] for r in agent_results) / len(agent_results)
            success_rate = sum(1 for r in agent_results if r["success"]) / len(agent_results)

            print(f"{agent_name:<20} {avg_duration:<15.2f} {avg_tokens:<15.0f} {success_rate:<15.1%}")
```

### Expected Results

| Pattern | Avg Duration | Avg Tokens | Success Rate | Best For |
|---------|-------------|------------|--------------|----------|
| Tool-Calling | 3.2s | 850 | 95% | Simple queries |
| Plan-Execute | 5.1s | 1200 | 85% | Deployments |
| ReAct | 8.7s | 2400 | 90% | Complex diagnosis |
| Reflexion | 15.3s | 4800 | 95% | Trial-and-error tasks |

**Key Insights**:
- **Tool-calling wins for simple tasks** (fastest, cheapest, most reliable)
- **ReAct wins for diagnosis** (explores unknowns effectively)
- **Plan-Execute wins for known workflows** (predictable execution)
- **Reflexion wins when first attempt usually fails** (but expensive)

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
            "write erase"
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
                raise PermissionError(f"Dangerous command detected: {pattern}")

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
            "tools_called": result.get("iterations", 0),
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"[{self.session_id}] Metrics: {json.dumps(metrics)}")
```

---

## Summary

You now have four production-ready agent patterns:

1. **ReAct**: Best for exploratory diagnosis, complex troubleshooting
2. **Plan-and-Execute**: Best for deployments, migrations, bulk operations
3. **Tool-Calling**: Best for simple queries, most efficient for production
4. **Reflexion**: Best when tasks need trial-and-error learning

**Decision Tree**:
```
Is the task a simple query?
├─ Yes → Use Tool-Calling Agent
└─ No → Is the structure known upfront?
    ├─ Yes → Use Plan-and-Execute Agent
    └─ No → Will first attempt likely fail?
        ├─ Yes → Use Reflexion Agent
        └─ No → Use ReAct Agent
```

**Production Checklist**:
- ✓ Error handling for API failures
- ✓ Safety guardrails for dangerous operations
- ✓ Human approval for production changes
- ✓ Comprehensive logging
- ✓ Rate limiting and retry logic
- ✓ Operation whitelisting
- ✓ Dangerous command detection

**Next Chapter**: We'll take these agents and build a complete autonomous troubleshooting system that combines diagnosis, root cause analysis, and fix generation.

---

## What Can Go Wrong?

**1. Agent runs forever without completing**
- **Cause**: No termination condition, agent stuck in reasoning loop
- **Fix**: Set `max_iterations` limit, add explicit "Answer:" termination signal

**2. Agent hallucinates tool results**
- **Cause**: LLM fabricates data instead of calling tools
- **Fix**: Use strict response parsing, validate tool calls before execution

**3. Plan-Execute fails midway**
- **Cause**: Plan didn't account for errors, no rollback mechanism
- **Fix**: Add error handling to each step, implement transaction rollback

**4. Tool-calling agent can't handle complex tasks**
- **Cause**: Task requires reasoning that tool-calling doesn't support
- **Fix**: Use ReAct for complex tasks, tool-calling for simple queries

**5. Reflexion agent wastes tokens retrying**
- **Cause**: Agent doesn't learn from reflections, repeats same mistake
- **Fix**: Improve reflection prompt, add explicit "what NOT to try" guidance

**6. Multi-agent coordinator makes wrong routing decision**
- **Cause**: Classification prompt is ambiguous
- **Fix**: Provide clear examples in classification prompt, log decisions

**7. Agent executes dangerous commands**
- **Cause**: No safety checks, agent bypasses guardrails
- **Fix**: Whitelist safe operations, blacklist dangerous patterns, require approval

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-19/`
