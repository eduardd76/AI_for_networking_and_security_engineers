# Chapter 15: Building AI Agents

## Table of Contents
1. Introduction to AI Agents
2. Agent Fundamentals
3. Agent Architecture Patterns
4. Decision-Making Frameworks
5. Tool Integration and Function Calling
6. Building Multi-Step Workflows
7. Error Handling and Resilience
8. Production Deployment Strategies
9. Real-World Examples and Case Studies
10. Best Practices and Optimization
11. Future Directions

---

## 1. Introduction to AI Agents

### 1.1 What Are AI Agents?

Artificial Intelligence agents represent a paradigm shift in how we design autonomous systems. Unlike traditional software that follows predetermined logic paths, AI agents are systems capable of perceiving their environment, making decisions based on those perceptions, and taking actions to achieve specific goals. They combine language models, reasoning capabilities, external tools, and feedback mechanisms to operate autonomously in complex environments.

An AI agent is fundamentally a loop:

```
Perception → Reasoning → Planning → Action → Feedback
     ↑___________________________________|
```

> **Networking analogy**: If you've worked with event-driven network automation (e.g., StackStorm, Ansible AWX triggered by syslog events), AI agents follow a similar pattern. The difference: instead of hardcoded playbooks for each event, an AI agent uses an LLM to *reason* about what to do. Think of it as replacing static route-maps with a routing protocol -- the system can adapt to situations it wasn't explicitly programmed for.
>
> **Example**: A traditional automation script might say "if interface goes down, send a Slack alert." An AI agent might say "interface Gi0/1 went down. Let me check CDP neighbors to see what's connected, look at the syslog for the root cause, check if there's a redundant path, and then send a targeted alert with all this context."

The power of modern AI agents lies in their ability to:

- **Understand context** through natural language and multi-modal inputs
- **Reason about problems** using chain-of-thought and structured thinking
- **Access external tools** including APIs, databases, computational services, and other systems
- **Learn from feedback** through iterative execution and adjustment
- **Handle complexity** by breaking problems into manageable sub-tasks
- **Operate autonomously** with minimal human intervention

### 1.2 Historical Context

The concept of intelligent agents is not new. It has roots in:

- **Classical AI (1956-1974)**: Early symbolic reasoning systems like ELIZA and SHRDLU
- **Expert Systems (1980-1987)**: Rule-based systems that captured domain expertise
- **Agent-Based Modeling (1990s-2000s)**: Multi-agent systems for simulation
- **Cognitive Architectures (2000s-2010s)**: SOAR, ACT-R, and similar frameworks
- **Deep Learning Era (2010s)**: Reinforcement learning agents with neural networks
- **Large Language Model Era (2020s)**: Emergence of LLM-based agents with tool use

The current renaissance of AI agents is driven by:
- Large language models (LLMs) with in-context learning capabilities
- Function calling APIs that enable deterministic tool integration
- Advances in reasoning and planning techniques
- Growing availability of structured APIs and data sources
- Improved understanding of agent reliability and safety

### 1.3 Why Build AI Agents?

AI agents address critical needs in modern software systems:

**Automation at Scale**: Agents can automate complex workflows that would require significant human effort, from customer service to research to data analysis.

**Flexible Problem-Solving**: Unlike traditional software with hardcoded logic, agents adapt to new problems and situations using reasoning rather than pre-programmed rules.

**Tool Orchestration**: Agents excel at coordinating multiple tools and services—calling the right API at the right time with the right parameters based on context.

**Knowledge Integration**: Agents can combine domain knowledge, real-time information, and reasoning to make better decisions than any single system.

**Scalability**: A single well-designed agent can handle thousands of diverse tasks without modification, whereas traditional software requires branching logic for each case.

### 1.4 Real-World Applications

**Customer Service**: Agents handling support tickets, escalating to humans when needed, and integrating with CRM systems

**Research and Analysis**: Agents conducting literature reviews, analyzing data, and generating insights across multiple sources

**Code Generation and DevOps**: Agents writing code, testing, debugging, and deploying applications

**Business Process Automation**: Agents managing approvals, scheduling, notifications, and cross-system workflows

**Content Creation**: Agents researching, writing, editing, and publishing content across platforms

**Healthcare**: Agents assisting with diagnosis support, patient education, and administrative workflows

### 1.5 The Agent Stack

Building production AI agents requires understanding several layers:

```
┌─────────────────────────────────────────────────┐
│         User Interface & Orchestration           │
│      (Dashboards, APIs, Chat Interfaces)         │
├─────────────────────────────────────────────────┤
│         Agent Control Loop & Planning            │
│   (Decision-making, Action Selection, Routing)   │
├─────────────────────────────────────────────────┤
│         Tool Integration & Execution             │
│  (APIs, Databases, Services, External Systems)  │
├─────────────────────────────────────────────────┤
│            Language Model Foundation             │
│   (Reasoning, Planning, Understanding Context)  │
├─────────────────────────────────────────────────┤
│    Infrastructure & Persistence Layer           │
│  (Logging, State Management, Error Recovery)    │
└─────────────────────────────────────────────────┘
```

Each layer plays a critical role in agent reliability and effectiveness.

---

## 2. Agent Fundamentals

### 2.1 Core Components

Every AI agent, regardless of its specific design, consists of several essential components:

#### 2.1.1 The Model Core

The language model is the reasoning engine of the agent. It processes context and generates decisions. Key considerations:

- **Model Selection**: Larger models (e.g., GPT-4) provide better reasoning but higher latency and cost. Smaller models (e.g., Claude Instant) are faster and cheaper but less capable.
- **Temperature and Sampling**: Lower temperature (0.0-0.3) produces deterministic, consistent behavior; higher temperature (0.7-1.0) produces more creative, varied outputs.
- **Context Window**: Larger context windows allow agents to maintain longer histories and process more information simultaneously.
- **Fine-tuning**: Custom training on agent-specific tasks can improve performance and reduce hallucination.

#### 2.1.2 Memory Systems

Agents must maintain different types of memory:

**Short-Term Memory**: The current conversation or task context, maintained in the prompt or message history. Typically limited by context window size.

```
Current Task Context
├── User Input
├── Previous Interactions (recent)
├── Relevant Retrieved Information
└── Current State
```

**Long-Term Memory**: Persistent storage of important facts, learned patterns, and historical context. Stored in databases and retrieved when relevant.

```
Long-Term Storage
├── Agent Profiles & Preferences
├── Completed Tasks & Outcomes
├── Learned Patterns & Insights
├── Domain Knowledge
└── Interaction History
```

**Working Memory**: Intermediate computations, parsed tool responses, and reasoning traces maintained during task execution.

#### 2.1.3 Tool and Capability Definition

Tools are the agent's interface to the external world. Each tool must be clearly defined:

```python
{
    "name": "search_knowledge_base",
    "description": "Searches the company knowledge base for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 10
            }
        },
        "required": ["query"]
    }
}
```

Tool definitions should be:
- **Precise**: Clear names and descriptions that guide the model
- **Complete**: All required and optional parameters documented
- **Constrained**: Sensible defaults and validation rules
- **Safe**: Input validation and rate limiting

#### 2.1.4 Planning and Reasoning Engine

This component determines how the agent approaches problems:

- **Decomposition**: Breaking complex tasks into sub-tasks
- **Sequencing**: Ordering tasks logically
- **Validation**: Checking intermediate results
- **Adaptation**: Adjusting plans based on outcomes

#### 2.1.5 Execution Environment

The runtime context where the agent operates:

- **State Management**: Tracking agent progress and decisions
- **Error Handling**: Gracefully managing failures
- **Logging and Monitoring**: Observability for debugging and improvement
- **Resource Management**: Managing timeouts, rate limits, and costs

### 2.2 Agent Lifecycle

Understanding the lifecycle helps design robust systems:

```
┌──────────────┐
│ Initialization│  Agent is created, loaded with context
└──────┬───────┘
       │
┌──────▼───────────┐
│ Input Reception   │  User query, trigger, or scheduled task
└──────┬───────────┘
       │
┌──────▼──────────────┐
│ Context Assembly    │  Gather relevant history, facts, knowledge
└──────┬──────────────┘
       │
┌──────▼─────────────┐
│ Initial Reasoning  │  Model analyzes task, determines approach
└──────┬─────────────┘
       │
┌──────▼──────────┐
│ Plan Generation │  Create sequence of actions
└──────┬──────────┘
       │
┌──────▼──────────────────┐
│ Tool Selection & Params │  Identify which tool to use, generate parameters
└──────┬──────────────────┘
       │
┌──────▼──────────────┐
│ Tool Execution      │  Call external system, handle responses/errors
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ Result Integration  │  Parse response, update agent state
└──────┬──────────────┘
       │
       │  More steps needed?
       ├─────────────┬──────────────┐
       │ Yes         │ No           │
       │             │              │
┌──────▼─────────────┐  ┌──────────▼────────┐
│ Continue Loop      │  │ Response Generation│
│ Return to Planning │  │ Format for output │
└──────┬─────────────┘  └──────────┬────────┘
       │                          │
       │                    ┌─────▼──────────┐
       └───────────────────▶│ Output & Logging│
                            └─────────────────┘
```

### 2.3 Types of Agents

Different agent architectures suit different problems:

#### 2.3.1 Reactive Agents

**Characteristics**: No internal state, responds immediately to inputs

**Advantages**:
- Stateless, horizontally scalable
- Predictable behavior
- Fast response times

**Disadvantages**:
- Limited memory
- Cannot maintain context across interactions
- Struggles with multi-step tasks

**Best For**: Immediate response systems, API wrappers, simple classifications

```python
def reactive_agent(user_input, tools):
    context = assemble_context(user_input)
    response = model.generate(context, tools)
    tool = parse_tool_call(response)
    result = execute_tool(tool)
    return format_response(result)
```

#### 2.3.2 Deliberative Agents

**Characteristics**: Maintains state, creates explicit plans

**Advantages**:
- Can handle complex, multi-step tasks
- Transparent reasoning (plan is explicit)
- Better error recovery

**Disadvantages**:
- Requires state management
- Slower decision-making
- More complex to implement

**Best For**: Complex workflows, long-running tasks, situations requiring transparency

```python
def deliberative_agent(user_input, tools):
    state = load_or_create_state()
    plan = create_plan(user_input, state)
    
    for step in plan:
        if should_execute(step, state):
            result = execute_step(step, tools)
            update_state(state, step, result)
            
            if is_error(result):
                plan = revise_plan(plan, result, state)
    
    return generate_response(state)
```

#### 2.3.3 Learning Agents

**Characteristics**: Improves performance through experience

**Advantages**:
- Improves over time
- Adapts to new domains
- Learns from feedback

**Disadvantages**:
- Requires feedback loops
- Slower initial performance
- Complex to tune and monitor

**Best For**: Long-running systems, domains with feedback signals, continuous improvement scenarios

#### 2.3.4 Hybrid Agents

**Characteristics**: Combines reactive and deliberative elements

Most production systems use hybrid approaches: reactive for simple cases, deliberative planning for complex scenarios.

### 2.4 Agent States and Transitions

Agents move through various states during execution:

```
INITIAL
   │
   ├─→ READY (awaiting input)
   │
   ├─→ PLANNING (analyzing task, creating plan)
   │
   ├─→ EXECUTING (running tools)
   │   │
   │   ├─→ SUCCESS (task completed)
   │   │
   │   ├─→ ERROR (recoverable error)
   │   │   └─→ RETRYING
   │   │
   │   └─→ BLOCKED (needs human input)
   │       └─→ WAITING
   │
   └─→ COMPLETE
```

State machine design is crucial for reliability.

---

## 3. Agent Architecture Patterns

### 3.1 Agentic Workflow Patterns

Different patterns solve different problems. Understanding these enables choosing the right architecture for your task.

#### 3.1.1 Simple Loop Pattern

The most basic agent pattern: perceive, act, repeat.

```
User Input
    │
    ▼
┌─────────────────────┐
│  Generate Response  │
│  (with tool calls)  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Execute Tools     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Add Results to      │
│ Conversation        │
└─────────────────────┘
    │
    ├─→ Continue needed? ───→ Loop back to Generate Response
    │
    └─→ Done? ──────────────→ Return Final Response
```

**Use Cases**: Customer support, Q&A systems, simple automation

**Implementation**:

```python
class SimpleAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.conversation = []
    
    def run(self, user_input, max_iterations=10):
        self.conversation.append({
            "role": "user",
            "content": user_input
        })
        
        for _ in range(max_iterations):
            response = self.model.generate(
                messages=self.conversation,
                tools=self.tools
            )
            
            self.conversation.append({
                "role": "assistant",
                "content": response.content
            })
            
            if not response.tool_calls:
                return response.content
            
            for tool_call in response.tool_calls:
                result = self.execute_tool(tool_call)
                self.conversation.append({
                    "role": "user",
                    "content": f"Tool result: {result}"
                })
        
        return "Max iterations reached"
    
    def execute_tool(self, tool_call):
        tool = self.tools[tool_call.name]
        return tool(**tool_call.arguments)
```

#### 3.1.2 Planning and Execution Pattern

Explicit planning phase followed by execution.

```
User Input
    │
    ▼
┌──────────────────┐
│  Analyze & Plan  │
│  (create steps)  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  For each step:  │
│  ├─ Verify ready │
│  ├─ Execute      │
│  └─ Check result │
└──────────────────┘
    │
    ├─→ Any failures? ──→ Revise plan
    │
    └─→ All done? ──────→ Synthesize response
```

**Use Cases**: Complex workflows, research tasks, code generation

**Implementation**:

```python
class PlanningAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
    
    def run(self, user_input):
        # Phase 1: Create plan
        plan = self.create_plan(user_input)
        print(f"Plan: {plan}")
        
        # Phase 2: Execute plan
        results = {}
        for step_id, step in enumerate(plan.steps):
            result = self.execute_step(step)
            results[step_id] = result
            
            if step.requires_verification and not self.verify_result(result):
                plan = self.revise_plan(plan, step_id, result)
        
        # Phase 3: Synthesize
        return self.synthesize(plan, results)
    
    def create_plan(self, user_input):
        response = self.model.generate(
            prompt=f"""Given this task: {user_input}
            
Create a detailed step-by-step plan. Each step should specify:
1. What needs to be done
2. What tool(s) to use
3. What success looks like
4. Dependencies on other steps

Format as a structured plan.""",
            temperature=0.3  # Deterministic for planning
        )
        return self.parse_plan(response)
    
    def execute_step(self, step):
        # Implement tool selection and execution
        pass
    
    def revise_plan(self, plan, failed_step_id, error):
        # Regenerate plan with error context
        pass
    
    def synthesize(self, plan, results):
        # Combine results into final response
        pass
```

#### 3.1.3 Hierarchical/Tree Search Pattern

Breaking down tasks into hierarchies, with branches for different approaches.

```
                    User Task
                        │
        ┌───────────────┼───────────────┐
        │               │               │
      Approach A     Approach B     Approach C
        │               │               │
    ┌──┴──┐        ┌──┴──┐        ┌──┴──┐
    │     │        │     │        │     │
    S1    S2       S1    S2       S1    S2
    │     │        │     │        │     │
    R1    R2       R1    R2       R1    R2
```

**Use Cases**: Multi-path decision making, A/B testing approaches, creative tasks

**Implementation**:

```python
class HierarchicalAgent:
    def run(self, task, depth=0, max_depth=3):
        if depth >= max_depth:
            return self.execute_simple(task)
        
        # Generate multiple approaches
        approaches = self.generate_approaches(task)
        
        results = {}
        for approach in approaches:
            # Recursively apply agent
            result = self.run(approach, depth + 1, max_depth)
            results[approach.name] = result
        
        # Evaluate and select best
        return self.select_best(results)
```

#### 3.1.4 Supervised Chain Pattern

Sequential execution where each step depends on the previous.

```
Step 1 Input
    │
    ▼
Step 1 Execution (Tool A)
    │
    ▼
Step 2 Input (uses Step 1 output)
    │
    ▼
Step 2 Execution (Tool B)
    │
    ▼
Step 3 Input (uses Step 1 & 2 output)
    │
    ▼
Step 3 Execution (Tool C)
    │
    ▼
Final Output
```

**Use Cases**: Data pipelines, ETL workflows, sequential processes

### 3.1.5 Evaluation and Refinement Pattern

Generate, evaluate, refine cycle.

```
Initial Generation
    │
    ▼
Evaluate Against Criteria
    │
    ├─→ Meets criteria? ──→ Return
    │
    └─→ Doesn't meet? ──→ Generate Refinement
                              │
                              ▼
                        Evaluate Again
                              │
                              ├─→ Better? Loop back
                              │
                              └─→ No improvement? Return best attempt
```

**Use Cases**: Content generation, code review, iterative refinement

**Implementation**:

```python
class RefinementAgent:
    def run(self, task, max_iterations=3):
        current = self.generate(task)
        
        for iteration in range(max_iterations):
            score = self.evaluate(current, task)
            
            if score >= self.success_threshold:
                return current
            
            feedback = self.generate_feedback(current, score)
            current = self.refine(current, feedback)
        
        return current
```

### 3.2 State Management Patterns

#### 3.2.1 Explicit State Objects

Maintain clear state structure:

```python
@dataclass
class AgentState:
    task_id: str
    user_input: str
    status: str  # "planning", "executing", "complete"
    current_step: int
    plan: List[str]
    results: Dict[int, str]
    errors: List[str]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self):
        return asdict(self)
    
    def save(self):
        # Persist to database
        pass
    
    @classmethod
    def load(cls, task_id):
        # Load from database
        pass
```

#### 3.2.2 Message-Based State

State maintained through message history:

```python
class MessageBasedAgent:
    def __init__(self):
        self.messages = []
    
    def add_state(self, role, content, metadata=None):
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
    
    def get_state(self):
        return self.messages
    
    def get_context(self, max_tokens=4000):
        # Summarize messages to fit context window
        return compress_messages(self.messages, max_tokens)
```

#### 3.2.3 Event-Driven State

State updated through discrete events:

```python
class EventDrivenAgent:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.state = {}
    
    def process_event(self, event):
        handler = self.event_handlers[event.type]
        new_state = handler(self.state, event)
        self.state = new_state
        self.event_bus.publish(event)
```

### 3.3 Scaling Patterns

#### 3.3.1 Single Agent, Multiple Tasks

One agent instance handles multiple tasks sequentially or in rotation.

```python
class TaskQueue:
    def __init__(self, agent):
        self.agent = agent
        self.queue = asyncio.Queue()
    
    async def process_tasks(self):
        while True:
            task = await self.queue.get()
            result = await self.agent.run(task)
            await self.queue.task_done()
```

#### 3.3.2 Multi-Agent Systems

Multiple specialized agents collaborate on complex tasks.

```python
class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "analysis": AnalysisAgent(),
            "synthesis": SynthesisAgent()
        }
    
    async def run(self, task):
        research_results = await self.agents["research"].run(task)
        analysis = await self.agents["analysis"].run(research_results)
        return await self.agents["synthesis"].run(analysis)
```

#### 3.3.3 Agent Hierarchies

Agents managing other agents.

```python
class ManagerAgent:
    def __init__(self):
        self.workers = [WorkerAgent() for _ in range(5)]
    
    async def run(self, tasks):
        results = await asyncio.gather(
            *[worker.run(task) for worker, task in zip(self.workers, tasks)]
        )
        return self.aggregate_results(results)
```

---

## 4. Decision-Making Frameworks

### 4.1 Decision Trees and Rules

Explicit rule-based decision making.

```python
class DecisionTree:
    def __init__(self):
        self.root = None
    
    def decide(self, context):
        return self.traverse(self.root, context)
    
    def traverse(self, node, context):
        if node.is_leaf:
            return node.action
        
        if node.condition(context):
            return self.traverse(node.yes_child, context)
        else:
            return self.traverse(node.no_child, context)
```

Example:

```python
if user_is_premium:
    if response_time_critical:
        use_fast_model()
    else:
        use_accurate_model()
else:
    if request_is_complex:
        escalate_to_human()
    else:
        use_light_model()
```

### 4.2 Multi-Criteria Decision Analysis

Evaluating options against multiple criteria.

```python
class MCDAFramework:
    def __init__(self, criteria, weights):
        self.criteria = criteria  # List of evaluation functions
        self.weights = weights    # Weight for each criterion
    
    def decide(self, options):
        scores = {}
        
        for option in options:
            total_score = 0
            for criterion, weight in zip(self.criteria, self.weights):
                criterion_score = criterion(option)
                total_score += criterion_score * weight
            scores[option] = total_score
        
        return max(options, key=lambda o: scores[o])
```

Example:

```python
criteria = [
    lambda tool: tool.accuracy,      # weight: 0.4
    lambda tool: 1 / tool.latency,   # weight: 0.3
    lambda tool: 1 / tool.cost,      # weight: 0.2
    lambda tool: tool.reliability    # weight: 0.1
]
weights = [0.4, 0.3, 0.2, 0.1]

best_tool = mcda.decide(available_tools)
```

### 4.3 Uncertainty and Confidence-Based Decisions

Making decisions despite uncertainty.

```python
class UncertaintyAwareDecision:
    def decide(self, options_with_confidence):
        # options_with_confidence: [(option, confidence), ...]
        
        high_confidence = [
            o for o, c in options_with_confidence 
            if c > 0.8
        ]
        
        if high_confidence:
            return random.choice(high_confidence)
        elif any(c > 0.5 for _, c in options_with_confidence):
            return max(options_with_confidence, key=lambda x: x[1])[0]
        else:
            return self.escalate_to_human(options_with_confidence)
```

### 4.4 Bayesian Decision Making

Using probability to guide decisions.

```python
class BayesianDecider:
    def __init__(self, prior_beliefs):
        self.beliefs = prior_beliefs  # P(option)
    
    def update_beliefs(self, evidence):
        # Bayes' theorem: P(option|evidence) = P(evidence|option) * P(option) / P(evidence)
        for option in self.beliefs:
            likelihood = self.calculate_likelihood(evidence, option)
            self.beliefs[option] = likelihood * self.beliefs[option]
        
        # Normalize
        total = sum(self.beliefs.values())
        self.beliefs = {o: p/total for o, p in self.beliefs.items()}
    
    def decide(self):
        return max(self.beliefs, key=self.beliefs.get)
```

### 4.5 Reinforcement Learning-Based Decisions

Learning decision policies from experience.

```python
class ReinforcementLearner:
    def __init__(self):
        self.q_values = {}  # Q(state, action)
        self.learning_rate = 0.1
        self.discount_factor = 0.95
    
    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            # Exploration: try random action
            return random.choice(self.get_actions(state))
        else:
            # Exploitation: choose best known action
            actions = self.get_actions(state)
            return max(actions, 
                      key=lambda a: self.q_values.get((state, a), 0))
    
    def learn(self, state, action, reward, next_state):
        current_q = self.q_values.get((state, action), 0)
        max_next_q = max(
            [self.q_values.get((next_state, a), 0) 
             for a in self.get_actions(next_state)],
            default=0
        )
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_values[(state, action)] = new_q
```

---

## 5. Tool Integration and Function Calling

### 5.1 Designing Tool Interfaces

Tools must be designed for easy agent use.

#### 5.1.1 Tool Specification Format

```python
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web for information on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10
                },
                "language": {
                    "type": "string",
                    "enum": ["en", "es", "fr", "de"],
                    "default": "en"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_document",
        "description": "Read the contents of a document",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Unique identifier for the document"
                },
                "section": {
                    "type": "string",
                    "description": "Specific section to read (optional)"
                }
            },
            "required": ["document_id"]
        }
    }
]
```

#### 5.1.2 Good Tool Design Principles

**Clarity**: Tool names and descriptions should be unambiguous

```python
# Bad
{"name": "get_data", "description": "Gets data"}

# Good
{"name": "search_customer_database", 
 "description": "Search for customers by name, email, or ID"}
```

**Composability**: Tools should work well together

```python
# Tools that complement each other
- search_documents
- read_document
- extract_text
- analyze_text
```

**Determinism**: Same inputs should produce same outputs (for reliability)

```python
# Good: deterministic
def get_customer(customer_id: str) -> dict

# Bad: non-deterministic
def get_random_product() -> dict
```

**Idempotence**: Running a tool multiple times shouldn't cause problems

```python
# Good: idempotent
def set_user_preference(user_id, pref_name, pref_value)

# Bad: not idempotent
def increment_user_counter(user_id)  # Different result each call
```

**Feedback**: Tools should return clear success/failure indicators

```python
# Bad
def create_order(items):
    # What if creation fails?
    return None

# Good
def create_order(items):
    try:
        order_id = db.create_order(items)
        return {
            "success": True,
            "order_id": order_id,
            "status": "created"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
```

### 5.2 Tool Execution Framework

```python
class ToolExecutor:
    def __init__(self):
        self.tools = {}
        self.execution_history = []
    
    def register_tool(self, definition, handler):
        self.tools[definition["name"]] = {
            "definition": definition,
            "handler": handler
        }
    
    def execute(self, tool_name, parameters):
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool_info = self.tools[tool_name]
        
        # Validate parameters
        validation_error = self.validate_parameters(
            parameters, 
            tool_info["definition"]["parameters"]
        )
        if validation_error:
            return {"error": f"Invalid parameters: {validation_error}"}
        
        # Execute with error handling
        try:
            result = tool_info["handler"](**parameters)
            execution_record = {
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "status": "success",
                "timestamp": datetime.now()
            }
        except Exception as e:
            result = {"error": str(e)}
            execution_record = {
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "status": "error",
                "timestamp": datetime.now(),
                "exception_type": type(e).__name__
            }
        
        self.execution_history.append(execution_record)
        return result
    
    def validate_parameters(self, params, param_spec):
        # Check required parameters
        required = param_spec.get("required", [])
        for req_param in required:
            if req_param not in params:
                return f"Missing required parameter: {req_param}"
        
        # Validate parameter types
        for param_name, param_value in params.items():
            if param_name in param_spec["properties"]:
                expected_type = param_spec["properties"][param_name]["type"]
                if not self.check_type(param_value, expected_type):
                    return f"Parameter {param_name} has wrong type"
        
        return None
    
    def check_type(self, value, expected_type):
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return isinstance(value, type_map.get(expected_type))
```

### 5.3 Handling Tool Failures

```python
class ResilientToolExecutor(ToolExecutor):
    def __init__(self, max_retries=3, backoff_factor=2):
        super().__init__()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def execute_with_retry(self, tool_name, parameters):
        delay = 1
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.execute(tool_name, parameters)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.backoff_factor
        
        return {
            "error": f"Failed after {self.max_retries} attempts",
            "last_error": str(last_error)
        }
    
    def execute_with_fallback(self, primary_tool, fallback_tool, parameters):
        """Try primary tool, fall back if it fails"""
        result = self.execute(primary_tool, parameters)
        
        if "error" in result:
            return self.execute(fallback_tool, parameters)
        
        return result
```

### 5.4 Tool Use Patterns

#### 5.4.1 Sequential Tool Use

```python
def research_topic(topic):
    # Step 1: Search for information
    search_results = executor.execute("web_search", {"query": topic})
    
    # Step 2: Read top results
    documents = []
    for result in search_results[:3]:
        doc = executor.execute("read_document", {"url": result["url"]})
        documents.append(doc)
    
    # Step 3: Analyze information
    analysis = model.analyze(documents)
    
    return analysis
```

#### 5.4.2 Parallel Tool Use

```python
async def analyze_multi_source(queries):
    # Execute multiple searches in parallel
    search_tasks = [
        executor.execute_async("web_search", {"query": q})
        for q in queries
    ]
    
    results = await asyncio.gather(*search_tasks)
    return aggregate_results(results)
```

#### 5.4.3 Conditional Tool Use

```python
def decide_and_act(situation):
    if situation["urgency"] == "high":
        return executor.execute("create_alert", situation)
    elif situation["type"] == "question":
        return executor.execute("search_knowledge_base", situation)
    else:
        return executor.execute("log_event", situation)
```

### 5.5 Function Calling with LLMs

Most modern LLMs support structured function calling. Example with Claude API:

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

def process_tool_call(tool_name, tool_input):
    if tool_name == "get_weather":
        return f"The weather in {tool_input['location']} is sunny"

messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"}
]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # Check if we're done
    if response.stop_reason == "end_turn":
        break
    
    # Process tool calls
    if response.stop_reason == "tool_use":
        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": response.content})
        
        # Process each tool call
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = process_tool_call(
                    content_block.name,
                    content_block.input
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results to messages
        messages.append({
            "role": "user",
            "content": tool_results
        })
```

---

## 6. Building Multi-Step Workflows

### 6.1 Workflow Definition

Workflows are sequences of steps with dependencies, branching, and error handling.

```python
@dataclass
class WorkflowStep:
    id: str
    name: str
    action: Callable
    dependencies: List[str]  # IDs of steps this depends on
    retry_policy: Optional['RetryPolicy'] = None
    timeout: Optional[int] = None
    skip_on_error: bool = False
    required: bool = True

class Workflow:
    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_log = []
    
    def add_step(self, step: WorkflowStep):
        self.steps[step.id] = step
    
    def execute(self):
        completed = set()
        results = {}
        
        while len(completed) < len(self.steps):
            # Find executable steps (all dependencies completed)
            executable = [
                step for step in self.steps.values()
                if step.id not in completed 
                and all(dep in completed for dep in step.dependencies)
            ]
            
            if not executable:
                raise Exception("Circular dependency or missing steps")
            
            for step in executable:
                try:
                    # Get inputs from dependencies
                    step_inputs = {
                        dep: results[dep] for dep in step.dependencies
                    }
                    
                    # Execute step
                    result = self._execute_step(step, step_inputs)
                    results[step.id] = result
                    completed.add(step.id)
                    
                except Exception as e:
                    if step.required:
                        raise
                    elif step.skip_on_error:
                        completed.add(step.id)
                        results[step.id] = None
        
        return results
    
    def _execute_step(self, step: WorkflowStep, inputs):
        try:
            if step.timeout:
                # Execute with timeout
                return execute_with_timeout(
                    step.action,
                    inputs,
                    step.timeout
                )
            else:
                return step.action(**inputs)
        except Exception as e:
            if step.retry_policy:
                return self._retry_step(step, inputs, e)
            raise
    
    def _retry_step(self, step, inputs, initial_error):
        for attempt in range(step.retry_policy.max_attempts):
            try:
                return step.action(**inputs)
            except Exception as e:
                if attempt == step.retry_policy.max_attempts - 1:
                    raise
                time.sleep(step.retry_policy.backoff ** attempt)
```

### 6.2 DAG-Based Workflows

Directed Acyclic Graphs for complex dependencies:

```python
import networkx as nx

class DAGWorkflow:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.results = {}
    
    def add_step(self, step_id, action, dependencies=None):
        self.graph.add_node(step_id, action=action)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, step_id)
    
    def validate(self):
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Workflow contains cycles")
    
    def execute(self):
        self.validate()
        
        # Topological sort to get execution order
        execution_order = list(nx.topological_sort(self.graph))
        
        for step_id in execution_order:
            node = self.graph.nodes[step_id]
            action = node["action"]
            
            # Get results from dependencies
            dependencies = list(self.graph.predecessors(step_id))
            inputs = {dep: self.results[dep] for dep in dependencies}
            
            # Execute
            self.results[step_id] = action(**inputs)
        
        return self.results
```

### 6.3 Conditional Workflows

Branching based on conditions:

```python
class ConditionalWorkflow:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step_id, action, condition=None):
        self.steps.append({
            "id": step_id,
            "action": action,
            "condition": condition or (lambda x: True)
        })
    
    def execute(self, context):
        results = {}
        
        for step in self.steps:
            # Check condition
            if step["condition"](context, results):
                result = step["action"](context, results)
                results[step["id"]] = result
        
        return results

# Example
workflow = ConditionalWorkflow()
workflow.add_step(
    "check_eligibility",
    lambda ctx, res: check_user_eligibility(ctx["user"]),
    condition=lambda ctx, res: True
)
workflow.add_step(
    "generate_discount",
    lambda ctx, res: generate_discount_code(),
    condition=lambda ctx, res: res.get("check_eligibility")
)
workflow.add_step(
    "send_notification",
    lambda ctx, res: send_email(ctx["user"], res.get("generate_discount")),
    condition=lambda ctx, res: res.get("generate_discount") is not None
)
```

### 6.4 Human-in-the-Loop Workflows

Workflows that pause for human approval:

```python
class HumanApprovalWorkflow:
    def __init__(self, approval_channel):
        self.approval_channel = approval_channel
    
    def execute_with_approval(self, action, approval_message, timeout=3600):
        # Execute the action
        result = action()
        
        # Request approval
        approval_id = self.request_approval(approval_message, result)
        
        # Wait for approval
        approved = self.wait_for_approval(approval_id, timeout)
        
        if approved:
            return {"status": "approved", "result": result}
        else:
            return {"status": "rejected", "result": result}
    
    def request_approval(self, message, result):
        # Send to approval channel (Slack, email, etc.)
        approval_id = str(uuid.uuid4())
        self.approval_channel.send({
            "id": approval_id,
            "message": message,
            "result": result,
            "callback_url": f"/approve/{approval_id}"
        })
        return approval_id
    
    def wait_for_approval(self, approval_id, timeout):
        # Poll or use callback
        start = time.time()
        while time.time() - start < timeout:
            response = self.approval_channel.check_response(approval_id)
            if response:
                return response["approved"]
            time.sleep(5)
        return False
```

### 6.5 Async Workflows

Parallel execution of independent steps:

```python
import asyncio

class AsyncWorkflow:
    def __init__(self):
        self.steps = {}
    
    def add_step(self, step_id, async_action, dependencies=None):
        self.steps[step_id] = {
            "action": async_action,
            "dependencies": dependencies or []
        }
    
    async def execute(self):
        completed = set()
        results = {}
        
        while len(completed) < len(self.steps):
            # Find executable steps
            executable = [
                (step_id, step) for step_id, step in self.steps.items()
                if step_id not in completed 
                and all(dep in completed for dep in step["dependencies"])
            ]
            
            # Execute all executable steps in parallel
            tasks = []
            for step_id, step in executable:
                inputs = {dep: results[dep] for dep in step["dependencies"]}
                task = asyncio.create_task(
                    step["action"](**inputs),
                    name=step_id
                )
                tasks.append((step_id, task))
            
            # Wait for all to complete
            for step_id, task in tasks:
                results[step_id] = await task
                completed.add(step_id)
        
        return results
```

---

## 7. Error Handling and Resilience

### 7.1 Error Classification

Different errors require different handling:

```python
from enum import Enum

class ErrorType(Enum):
    TRANSIENT = "transient"      # Temporary, retry likely to succeed
    PERMANENT = "permanent"      # Won't succeed, don't retry
    RESOURCE = "resource"        # Resource exhausted, wait and retry
    USER = "user"                # User input invalid, ask for correction
    EXTERNAL = "external"        # External service issue
    INTERNAL = "internal"        # Internal system error

def classify_error(exception):
    if isinstance(exception, TimeoutError):
        return ErrorType.TRANSIENT
    elif isinstance(exception, ValueError):
        return ErrorType.USER
    elif isinstance(exception, RateLimitError):
        return ErrorType.RESOURCE
    elif isinstance(exception, KeyError):
        return ErrorType.PERMANENT
    else:
        return ErrorType.INTERNAL
```

### 7.2 Retry Strategies

```python
class RetryPolicy:
    def __init__(self, max_attempts=3, backoff_type="exponential", 
                 initial_delay=1, max_delay=60):
        self.max_attempts = max_attempts
        self.backoff_type = backoff_type
        self.initial_delay = initial_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt):
        if self.backoff_type == "exponential":
            delay = self.initial_delay * (2 ** attempt)
        elif self.backoff_type == "linear":
            delay = self.initial_delay * (attempt + 1)
        else:
            delay = self.initial_delay
        
        return min(delay, self.max_delay)

class RetryExecutor:
    def __init__(self, policy=None):
        self.policy = policy or RetryPolicy()
    
    def execute(self, fn, args=None, kwargs=None):
        args = args or ()
        kwargs = kwargs or {}
        
        last_error = None
        for attempt in range(self.policy.max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_type = classify_error(e)
                
                if error_type == ErrorType.PERMANENT:
                    raise
                
                if attempt < self.policy.max_attempts - 1:
                    delay = self.policy.get_delay(attempt)
                    time.sleep(delay)
        
        raise Exception(f"Failed after {self.policy.max_attempts} attempts") from last_error
```

### 7.3 Circuit Breaker Pattern

Preventing cascading failures:

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def call(self, fn, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = fn(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
breaker = CircuitBreaker()
try:
    result = breaker.call(external_api_call)
except Exception as e:
    # Fallback
    result = get_cached_result()
```

### 7.4 Graceful Degradation

Providing reduced functionality when systems fail:

```python
class GracefulDegradation:
    def __init__(self):
        self.primary_service = ServiceA()
        self.fallback_service = ServiceB()
        self.cache = {}
    
    def get_data(self, key):
        # Try primary
        try:
            return self.primary_service.get(key)
        except Exception as e:
            print(f"Primary service failed: {e}")
        
        # Try fallback
        try:
            return self.fallback_service.get(key)
        except Exception as e:
            print(f"Fallback service failed: {e}")
        
        # Use cache
        if key in self.cache:
            print(f"Using cached value for {key}")
            return self.cache[key]
        
        # Return default
        return self.get_default_value(key)
    
    def get_default_value(self, key):
        # Return safe default value
        return None
```

### 7.5 Error Context and Logging

```python
from dataclasses import dataclass
import logging

@dataclass
class ErrorContext:
    error_type: ErrorType
    message: str
    original_exception: Exception
    step_id: str
    attempt: int
    timestamp: datetime
    agent_state: dict
    
    def to_dict(self):
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "timestamp": self.timestamp.isoformat(),
            "agent_state": self.agent_state
        }

class ErrorLogger:
    def __init__(self):
        self.logger = logging.getLogger("agent_errors")
        self.error_history = []
    
    def log_error(self, context: ErrorContext):
        self.logger.error(
            f"Agent error in {context.step_id}: {context.message}",
            extra=context.to_dict()
        )
        self.error_history.append(context)
    
    def get_error_summary(self, step_id=None):
        errors = self.error_history
        if step_id:
            errors = [e for e in errors if e.step_id == step_id]
        
        return {
            "total_errors": len(errors),
            "by_type": self._count_by_type(errors),
            "by_step": self._count_by_step(errors)
        }
    
    def _count_by_type(self, errors):
        from collections import Counter
        return Counter(e.error_type.value for e in errors)
    
    def _count_by_step(self, errors):
        from collections import Counter
        return Counter(e.step_id for e in errors)
```

---

## 8. Production Deployment Strategies

### 8.1 Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│            Load Balancer / API Gateway           │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴────┬─────────┬─────────┐
    │          │         │         │
┌───▼──┐  ┌───▼──┐  ┌──▼───┐  ┌──▼───┐
│Agent │  │Agent │  │Agent │  │Agent │
│Inst1 │  │Inst2 │  │Inst3 │  │Inst4 │
└───┬──┘  └───┬──┘  └──┬───┘  └──┬───┘
    │         │        │        │
    └────┬────┴────┬───┴────┬───┘
         │         │        │
    ┌────▼──────┐  │  ┌─────▼──────┐
    │Task Queue │  │  │State Store  │
    │(Redis)    │  │  │(PostgreSQL) │
    └───────────┘  │  └─────────────┘
                   │
            ┌──────▼────────┐
            │Cache Layer    │
            │(Redis/Memcache)
            └───────────────┘
```

### 8.2 Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run agent
CMD ["python", "-m", "uvicorn", "agent_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.3 Configuration Management

```python
from pydantic import BaseSettings

class AgentConfig(BaseSettings):
    # Model settings
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # Retry settings
    max_retries: int = 3
    retry_delay: int = 1
    
    # Timeout settings
    tool_timeout: int = 30
    workflow_timeout: int = 300
    
    # Resource limits
    max_parallel_tasks: int = 10
    max_memory_mb: int = 1024
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
config = AgentConfig()
agent = Agent(config=config)
```

### 8.4 Monitoring and Observability

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class AgentMetrics:
    def __init__(self):
        self.task_count = Counter(
            'agent_tasks_total',
            'Total tasks processed',
            ['status']
        )
        self.task_duration = Histogram(
            'agent_task_duration_seconds',
            'Time to process task'
        )
        self.tool_calls = Counter(
            'agent_tool_calls_total',
            'Total tool calls',
            ['tool_name', 'status']
        )
        self.errors = Counter(
            'agent_errors_total',
            'Total errors',
            ['error_type']
        )
        self.concurrent_tasks = Gauge(
            'agent_concurrent_tasks',
            'Currently running tasks'
        )
    
    def record_task(self, status):
        self.task_count.labels(status=status).inc()
    
    def record_duration(self, duration):
        self.task_duration.observe(duration)
    
    def record_tool_call(self, tool_name, status):
        self.tool_calls.labels(tool_name=tool_name, status=status).inc()
    
    def record_error(self, error_type):
        self.errors.labels(error_type=error_type).inc()

class MonitoredAgent:
    def __init__(self, agent, metrics):
        self.agent = agent
        self.metrics = metrics
    
    def run_task(self, task):
        self.metrics.concurrent_tasks.inc()
        start = time.time()
        
        try:
            result = self.agent.run(task)
            self.metrics.record_task('success')
            return result
        except Exception as e:
            self.metrics.record_error(type(e).__name__)
            self.metrics.record_task('error')
            raise
        finally:
            duration = time.time() - start
            self.metrics.record_duration(duration)
            self.metrics.concurrent_tasks.dec()
```

### 8.5 Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: pylint agent/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: docker build -t agent:${{ github.sha }} .
      - run: docker tag agent:${{ github.sha }} agent:latest
      - run: docker push gcr.io/project/agent:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: |
          gcloud run deploy agent \
            --image gcr.io/project/agent:${{ github.sha }} \
            --region us-central1 \
            --memory 2Gi \
            --timeout 3600
```

### 8.6 Scaling Considerations

**Horizontal Scaling**: Running multiple agent instances

```python
class LoadBalancedAgent:
    def __init__(self, num_instances=4):
        self.instances = [Agent() for _ in range(num_instances)]
        self.next_instance = 0
    
    def run(self, task):
        instance = self.instances[self.next_instance]
        self.next_instance = (self.next_instance + 1) % len(self.instances)
        return instance.run(task)
```

**Vertical Scaling**: Increasing resource limits

```python
# Increase memory and CPU for complex tasks
def run_complex_task(task):
    if task.complexity > HIGH_THRESHOLD:
        return agent.run_with_resources(
            task,
            memory_gb=4,
            cpu_cores=4,
            timeout=600
        )
    else:
        return agent.run(task)
```

**Caching**: Reducing redundant computation

```python
class CachedAgent:
    def __init__(self, agent):
        self.agent = agent
        self.cache = {}
    
    def run(self, task):
        cache_key = hash(task.content)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.agent.run(task)
        self.cache[cache_key] = result
        return result
```

---

## 9. Real-World Examples and Case Studies

### 9.1 Customer Support Agent

```python
class CustomerSupportAgent:
    def __init__(self):
        self.model = Claude()
        self.tools = {
            "search_kb": SearchKnowledgeBase(),
            "get_order": GetOrderStatus(),
            "get_policy": GetPolicy(),
            "create_ticket": CreateSupportTicket()
        }
    
    def run(self, user_message):
        system_prompt = """You are a helpful customer support agent. 
Help the customer with their issue. You have access to:
- Knowledge base search
- Order lookup
- Return/shipping policy
- Ticket creation for complex issues

Follow this approach:
1. Understand the customer's issue
2. Search knowledge base for solutions
3. If found, provide the answer
4. If not, offer additional help (order lookup, etc.)
5. If still unresolved, create a support ticket

Always be empathetic and professional."""
        
        return self.agentic_loop(
            user_message,
            system_prompt,
            max_iterations=5
        )
    
    def agentic_loop(self, initial_input, system_prompt, max_iterations):
        messages = [
            {"role": "user", "content": initial_input}
        ]
        
        for iteration in range(max_iterations):
            response = self.model.generate(
                messages=messages,
                system=system_prompt,
                tools=self.tools.keys()
            )
            
            messages.append({
                "role": "assistant",
                "content": response.text
            })
            
            if not response.tool_calls:
                return response.text
            
            tool_results = []
            for tool_call in response.tool_calls:
                tool = self.tools[tool_call.name]
                result = tool.execute(**tool_call.arguments)
                tool_results.append({
                    "tool": tool_call.name,
                    "result": result
                })
            
            messages.append({
                "role": "user",
                "content": f"Tool results: {tool_results}"
            })
        
        return "Unable to resolve. Creating support ticket."
```

### 9.2 Research and Analysis Agent

A detailed example is provided in the Colab notebook.

### 9.3 Code Generation and Testing Agent

```python
class CodeGenerationAgent:
    def __init__(self):
        self.model = Claude()
        self.tools = {
            "write_code": WriteCode(),
            "test_code": TestCode(),
            "run_tests": RunTests(),
            "fix_issue": FixIssue()
        }
    
    def generate_solution(self, requirements):
        plan = self.create_plan(requirements)
        code = self.generate_code(plan)
        
        for iteration in range(3):
            test_results = self.test_code(code)
            
            if test_results["all_pass"]:
                return code
            
            code = self.fix_failing_tests(code, test_results)
        
        return code
```

---

## 10. Best Practices and Optimization

### 10.1 Agent Design Principles

**Single Responsibility**: Each agent should have a clear, focused purpose

```python
# Good
class EmailAgent:
    """Handles email composition and sending"""

class SchedulingAgent:
    """Handles calendar and meeting scheduling"""

# Bad
class MegaAgent:
    """Does everything"""
```

**Clear Tool Definitions**: Make it easy for the model to understand tool use

```python
# Good
{
    "name": "calculate_loan_payment",
    "description": "Calculate monthly loan payment amount",
    "parameters": {
        "principal_amount": "Loan amount in dollars",
        "interest_rate": "Annual interest rate as decimal",
        "term_months": "Loan term in months"
    }
}

# Bad
{
    "name": "calculate",
    "description": "Do some math"
}
```

**Fail Fast**: Detect and report errors early

```python
def validate_inputs(user_input):
    if not user_input:
        raise ValueError("Input cannot be empty")
    
    if len(user_input) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input exceeds {MAX_INPUT_LENGTH} characters")
    
    return user_input
```

### 10.2 Performance Optimization

**Prompt Optimization**: Reducing token usage

```python
# Bad: Long, verbose system prompt
system = """You are a helpful assistant. You should be respectful and kind...
[1000 more words]"""

# Good: Concise, focused
system = """Answer questions clearly and concisely."""
```

**Caching and Memoization**:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_hash):
    return result
```

**Parallel Execution**:

```python
async def parallel_operations(tasks):
    return await asyncio.gather(*tasks)
```

**Streaming**: Getting results incrementally

```python
def stream_response(task):
    for chunk in model.stream(task):
        yield chunk
        # Stream to user immediately
```

### 10.3 Cost Optimization

**Token Management**:

```python
def estimate_cost(prompt_tokens, completion_tokens, model="claude-3-5-sonnet"):
    rates = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015}
    }
    
    rate = rates.get(model, rates["gpt-4"])
    prompt_cost = (prompt_tokens / 1000) * rate["prompt"]
    completion_cost = (completion_tokens / 1000) * rate["completion"]
    
    return prompt_cost + completion_cost
```

**Batch Processing**:

```python
def batch_process_tasks(tasks, batch_size=100):
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        results = process_batch(batch)
        yield results
```

### 10.4 Reliability Improvements

**Input Validation**:

```python
def validate_agent_input(user_input):
    # Check length
    if len(user_input) > MAX_LENGTH:
        return None, "Input too long"
    
    # Check for injection attacks
    if contains_injection_pattern(user_input):
        return None, "Invalid input format"
    
    # Check for PII if needed
    if contains_pii(user_input):
        return None, "Cannot process personal information"
    
    return user_input, None
```

**Output Validation**:

```python
def validate_agent_output(output):
    # Check for harmful content
    if is_harmful(output):
        return None, "Output contains harmful content"
    
    # Check coherence
    if not is_coherent(output):
        return None, "Output is incoherent"
    
    return output, None
```

---

## 11. Future Directions

### 11.1 Emerging Capabilities

**Multimodal Agents**: Processing images, audio, and video

```python
class MultimodalAgent:
    def understand_image(self, image_url):
        return self.model.analyze_image(image_url)
    
    def process_audio(self, audio_url):
        transcript = self.model.transcribe(audio_url)
        return self.understand(transcript)
```

**Long-Context Agents**: Leveraging larger context windows

```python
class LongContextAgent:
    def analyze_entire_document(self, document):
        # Can now fit entire documents in context
        analysis = self.model.analyze(document)
        return analysis
```

**Tool Learning**: Agents discovering new tools

```python
class ToolLearningAgent:
    def discover_tool(self, api_documentation):
        # Parse docs, understand capability
        tool_spec = self.infer_tool_spec(api_documentation)
        self.register_tool(tool_spec)
```

### 11.2 Multi-Agent Collaboration

**Agent Communication**:

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
    
    async def run_collaboration(self, task):
        # Agents communicate through message queue
        results = await asyncio.gather(
            *[agent.run(task, self.message_queue) for agent in self.agents.values()]
        )
        return self.synthesize_results(results)
```

**Agent Competition and Debate**:

```python
class DebateFramework:
    def run_debate(self, topic):
        pro_agent = Agent("pro")
        con_agent = Agent("con")
        judge = Agent("judge")
        
        pro_argument = pro_agent.argue(topic)
        con_argument = con_agent.argue(topic, pro_argument)
        
        decision = judge.decide(pro_argument, con_argument)
        return decision
```

### 11.3 Continuous Learning

**Online Learning**: Agents improving from user feedback

```python
class LearningAgent:
    def process_feedback(self, task, output, feedback):
        # Learn from feedback
        self.update_preferences(feedback)
        
        # Improve future performance
        self.fine_tune_model(task, output, feedback)
```

### 11.4 Ethics and Safety

**Alignment**: Ensuring agents align with human values

```python
class AlignedAgent:
    def __init__(self):
        self.values = load_human_values()
        self.safety_checks = SafetyChecker()
    
    def run(self, task):
        # Check alignment
        if not self.is_aligned(task):
            return self.refuse_task(task)
        
        return self.execute(task)
```

**Interpretability**: Understanding agent decisions

```python
class InterpretableAgent:
    def run(self, task):
        result = self.execute(task)
        explanation = self.explain_decision(task, result)
        
        return {
            "result": result,
            "reasoning": explanation,
            "confidence": self.get_confidence()
        }
```

---

## Conclusion

Building AI agents represents a frontier in software engineering. As language models become more capable and reliable, agents will become increasingly central to how we design intelligent systems. The patterns, practices, and frameworks discussed in this chapter provide a foundation for building production-ready agents that are reliable, scalable, and maintainable.

The key to successful agent development is balancing capability with safety, ambition with reliability, and automation with appropriate human oversight. As the field evolves, staying informed about new architectures and best practices will be essential for building the next generation of intelligent systems.

---

## References and Further Reading

- Wooldridge, M. (2009). "An Introduction to MultiAgent Systems" (2nd ed.)
- Russell, S., & Norvig, P. (2020). "Artificial Intelligence: A Modern Approach" (4th ed.)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
- LangChain Documentation: https://python.langchain.com/
- AutoGPT and Autonomous Agents Research
- Recent papers on prompt engineering and chain-of-thought reasoning
