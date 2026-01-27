# How to Build an AI Agent for Network Engineers
## By Eduard Dulharu
### Webinar Slides (10-59)

---

## Slide 10: The Four Pillars
**Core Architecture Components**

```
┌─────────────────────────────────────────────────────────┐
│                   AI AGENT ARCHITECTURE                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   PILLAR 1   │  │   PILLAR 2   │  │   PILLAR 3   │   │
│  │  REASONING   │  │    MEMORY    │  │    TOOLS     │   │
│  │   ENGINE     │  │   SYSTEMS    │  │   (SKILLS)   │   │
│  │              │  │              │  │              │   │
│  │  - LLM Core  │  │  - Working   │  │  - SSH/API   │   │
│  │  - Prompts   │  │  - Episodic  │  │  - Show cmds │   │
│  │  - Planning  │  │  - Semantic  │  │  - Config    │   │
│  │  - ReAct     │  │  - Graph     │  │  - Parsing   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│               ┌───────────┴───────────┐                 │
│               │       PILLAR 4        │                 │
│               │       CONTEXT         │                 │
│               │   (Knowledge Base)    │                 │
│               │                       │                 │
│               │   - RAG (Vectors)     │                 │
│               │   - Knowledge Graph   │                 │
│               │   - MCP Servers       │                 │
│               └───────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

**Key Point:** Reasoning (Brain), Memory (Experience), Tools (Hands), Context (Grounding).

---

## Slide 11: Reasoning Patterns: ReAct
**Reason + Act Loop**

**ReAct (Reasoning + Acting)** is the default pattern for troubleshooting.

- **Thought:** Analyze current state
- **Action:** Select a tool to get more info
- **Observation:** Read tool output
- **Repeat:** Loop until problem solved

```
Thought: Users can't reach server. I should check if it's pingable.
Action: ping(192.168.1.100)
Observation: Request timed out.
Thought: Ping failed. I should check the routing table next.
```

---

## Slide 12: Reasoning Patterns: Planning
**Chain of Thought**

**Planning** is for complex, multi-step tasks like network design.

- Agent breaks down high-level goal into sub-tasks
- Executes sub-tasks sequentially
- Adjusts plan if a step fails

```
Goal: Deploy new branch site
Plan:
  1. Allocate IP subnets (IPAM)
  2. Generate switch config
  3. Generate router config
  4. Apply configs
  5. Verify connectivity
```

---

## Slide 13: Reasoning Patterns: Reflection
**Self-Correction**

```
Task: Create ACL to block 10.0.0.0/8

Initial Generation:
  deny ip 10.0.0.0 0.255.255.255 host 192.168.1.100
  permit ip any any

REFLECTION CRITIQUE:
  ❌ SECURITY: "permit ip any any" is too broad.
  ⚠️ STANDARDS: Missing naming convention.
  ⚠️ MISSING: No logging.

Revised Configuration:
  ip access-list extended BLOCK-TO-SERVER-2026
    deny ip 10.0.0.0 0.255.255.255 host 192.168.1.100 log
    permit ip 10.0.0.0 0.255.255.255 192.168.0.0 0.0.255.255
    deny ip any any log
```

**Key Point:** Agent critiques its own work before finalizing. Essential for config generation.

---

## Slide 14: Human-in-the-Loop
**The Safety Net**

```
┌──────────────────────────────────────────────────┐
│                AGENT WORKFLOW                     │
├──────────────────────────────────────────────────┤
│  Phase 1: DIAGNOSIS (Autonomous)                 │
│  Phase 2: SOLUTION GENERATION (Autonomous)       │
│                                                   │
│  ──────────── CHECKPOINT ────────────────        │
│                                                   │
│  Phase 3: HUMAN REVIEW                           │
│    - Review Plan                                 │
│    - Review Config Diff                          │
│    - Approve / Reject / Modify                   │
│                                                   │
│  ──────────── APPROVAL ──────────────────        │
│                                                   │
│  Phase 4: EXECUTION (Autonomous)                 │
│  Phase 5: VERIFICATION (Autonomous)              │
└──────────────────────────────────────────────────┘
```

**Key Point:** Never let an agent write to production without a human checkpoint.

---

## Slide 15: The Cognitive Architecture
**How the Brain Works**

### Profile
Who am I? (Senior Network Engineer)

### Memory
What do I know? (Past tickets, topology)

### Planning
How do I solve this? (Step-by-step)

### Action
What can I do? (Tools)

---

## Slide 16: Why Now?
**The Convergence**

Three trends are converging to make this possible today:

- **LLM Reasoning:** Models like Claude 3.5 Sonnet and GPT-4o can finally reason about complex topologies.
- **Function Calling:** Reliable structured output (JSON) allows models to call APIs deterministically.
- **Context Windows:** 128k+ tokens allow us to feed entire config files and routing tables into context.

---

## Slide 17: Part 1 Summary
**Foundations**

### Agents Reason
They don't just execute scripts; they think.

### Systems Thinking
Treat the network as a complex adaptive system.

### 4 Pillars
Reasoning, Memory, Tools, Context.

### Safety First
Always keep a human in the loop for write actions.

---

## Slide 18: Frameworks: CrewAI vs LangGraph
**Choosing the Right Tool**

### CrewAI
- ✕ High-level abstraction
- ✕ Easy to start
- ✕ Role-based (Manager, Researcher)
- ✕ Harder to control precise flow

### LangGraph
- ✓ Low-level state machine
- ✓ Full control over loops
- ✓ Explicit state management
- ✓ Steeper learning curve

> "For production network agents where safety and precise state transitions are critical, LangGraph is the superior choice."

---

## Slide 19: CrewAI Example
**Role-Based Orchestration**

```python
from crewai import Agent, Task, Crew

# Define Agents
researcher = Agent(
    role='Network Researcher',
    goal='Analyze logs for errors',
    backstory='Expert at parsing syslog...',
    tools=[search_tool]
)
writer = Agent(
    role='Report Writer',
    goal='Summarize findings',
    backstory='Technical writer...'
)

# Define Tasks
task1 = Task(description='Find errors in logs', agent=researcher)
task2 = Task(description='Write RCA report', agent=writer)

# Execute
crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**Key Point:** Great for linear pipelines, but harder to implement complex loops and conditional branching.

---

## Slide 20: LangGraph Example
**State Machine Orchestration**

```python
from langgraph.graph import StateGraph, END

# Define State
class AgentState(TypedDict):
    messages: list
    next_step: str

# Define Nodes
def reason_node(state):
    # LLM logic...
    return {"next_step": "act"}

def act_node(state):
    # Tool execution...
    return {"next_step": "reason"}

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("reason", reason_node)
workflow.add_node("act", act_node)
workflow.set_entry_point("reason")
workflow.add_conditional_edges("reason", should_continue)
workflow.add_edge("act", "reason")
app = workflow.compile()
```

**Key Point:** Explicit control over the loop. You define exactly when to loop, when to stop, and how to handle errors.

---

## Slide 21: State Machine Implementation
**Defining the Phases**

```python
from langgraph.graph import StateGraph

class AgentPhase(Enum):
    OBSERVE = "observe"
    REASON = "reason"
    ACT = "act"
    VERIFY = "verify"
    COMPLETE = "complete"

workflow = StateGraph(AgentState)
workflow.add_node("observe", observe_phase)
workflow.add_node("reason", reason_phase)
workflow.add_edge("observe", "reason")
# ...
```

**Key Point:** We explicitly define the phases of the troubleshooting loop as nodes in the graph.

---

## Slide 22: Agent State Definition
**What the Agent Remembers**

```python
@dataclass
class AgentState:
    user_query: str         # What user asked
    phase: AgentPhase       # Where we are
    evidence: List[str]     # What we've learned
    hypothesis: str         # Current theory
    confidence: float       # How sure we are
    tool_calls: List[ToolCall]  # What we've done
    device_data: Dict       # Device info collected
```

**Key Point:** Just like you remember what you've checked, the agent maintains this state object throughout the execution.

---

## Slide 23: Why LangGraph for NetOps?
**Control is King**

### Cyclic Graphs
Troubleshooting is a loop, not a line. LangGraph handles cycles natively.

### State Persistence
Pause execution, wait for human approval, then resume state.

### Fine-Grained Control
Define exact transition logic (e.g., if ping fails 3 times, escalate).

### Observability
Every step is a node transition, making debugging easier.

---

## Slide 24: Data Architecture
**Garbage In, Garbage Out**

An agent is only as good as its data. You need a **Unified Data Layer**.

- **Source of Truth:** NetBox / Nautobot
- **Operational State:** Prometheus / TIG Stack
- **Logs:** Splunk / ELK
- **Documentation:** Confluence / SharePoint

Don't make the agent scrape CLI for everything. Feed it structured data.

---

## Slide 25: The Context Window Problem
**Why We Need RAG**

```
┌─────────────────────────────────────────────────┐
│             LLM CONTEXT WINDOW                  │
│      (Limited Space - e.g., 128k tokens)        │
├─────────────────────────────────────────────────┤
│  SYSTEM PROMPT (Who you are)                    │
│  ─────────────────────────────────────────────  │
│  USER QUERY (The problem)                       │
│  ─────────────────────────────────────────────  │
│  RETRIEVED CONTEXT (RAG)                        │
│    - Relevant Config Snippets (NOT all configs) │
│    - Relevant Logs (NOT all logs)               │
│    - Relevant Topology (NOT full map)           │
│  ─────────────────────────────────────────────  │
│  HISTORY (Conversation turns)                   │
└─────────────────────────────────────────────────┘
```

**Key Point:** You can't fit the whole network in the prompt. You must retrieve only what matters.

---

## Slide 26: Three-Layer Data Architecture
**The Foundation**

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   LAYER 3    │  │   LAYER 2    │  │   LAYER 1    │
│              │  │              │  │              │
│  KNOWLEDGE   │  │     RAG      │  │     MCP      │
│    GRAPH     │  │   (Vector)   │  │ (Live State) │
│              │  │              │  │              │
│   Topology   │  │Documentation │  │  Standards   │
│ Dependencies │  │Best Practices│  │   Policies   │
│ Relationships│  │Past Incidents│  │Current State │
│              │  │              │  │              │
│    Neo4j     │  │   ChromaDB   │  │  MCP Servers │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Key Point:** Layer 1: Live Context. Layer 2: Docs/History. Layer 3: Topology.

---

