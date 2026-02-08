# Chapter 15: Building AI Agents - Autonomous Network Operations

## Preface: From Reactive to Autonomous Operations

You have built:
- **Chapter 13**: Documentation that generates itself
- **Chapter 14**: A search system that understands intent

Now comes the final transformation: **a system that makes decisions and takes action.**

This is the agent layer. An agent perceives its environment, reasons about what must be done, makes decisions, executes actions, and verifies outcomes—without human intervention (though with human oversight).

The question is not whether agents are necessary. The question is: **can your organization afford NOT to have them?**

When an experienced network operator performs their job, they execute a repeatable process:
1. Notice something (monitoring alert, user report)
2. Understand the situation (check configs, read logs, search docs)
3. Reason about solutions (what are the options?)
4. Make a decision (which option is best?)
5. Execute action (apply config change, restart service)
6. Verify results (did it work?)

Agents automate this process. The network engineer becomes a supervisor of agents, not a executor of tasks.

---

## Part 1: Agent Theory and Fundamentals

### 1.1 What is an Agent? From Theory to Practice

**Definition**: An agent is a system that maintains internal state, receives observations from its environment, applies reasoning to those observations, chooses actions based on that reasoning, and learns from the outcomes.

This is more precise than "it's like a chatbot." Key components:

1. **Internal State** - What the agent knows/remembers (goal, history, context)
2. **Observations** - What it perceives (device status, logs, configs)
3. **Reasoning** - How it thinks (Claude's reasoning)
4. **Action Selection** - What it decides to do (tool choice)
5. **Execution** - How it does it (tool calling)
6. **Learning** - How it improves (feedback loop)

### 1.2 The Mathematical Foundation: Markov Decision Process (MDP)

Network operations can be modeled as a Markov Decision Process:

```
State Space (S):
├─ Device states: {up, down, degraded}
├─ Configuration states: {correct, incorrect}
├─ Service states: {running, stopped, hanging}
└─ Network states: {converged, unconverged}

Action Space (A):
├─ Observe actions: {check_status, get_config, analyze_logs}
├─ Intervention actions: {apply_config, restart_service}
└─ Communication actions: {notify_team, escalate}

Transitions:
S_t + A_t → S_t+1

Example:
State: router-02 BGP neighbors down
Action: fix_bgp_config
Result: State → BGP neighbors up

Reward:
R(S_t, A_t, S_t+1) = benefit of action
- Success fixing issue: +100
- Unsuccessful fix: -50
- Action took too long: -10
```

**Why this matters**: Understanding agents as MDPs clarifies that agent decision-making is fundamentally about **finding optimal action sequences in complex state spaces**.

The agent's job: "Given current state S and goal G, choose action A that maximizes probability of reaching goal state."

### 1.3 The Agent Loop: Detailed Theory

The core agent algorithm is fundamentally simple but has deep implications:

```python
def agent_loop(goal, max_iterations=10):
    """
    The core agent algorithm
    """
    
    state = initial_state()  # OBSERVE initial situation
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # OBSERVE: Perceive current situation
        observations = observe_environment()
        state.update(observations)
        
        # CHECK: Is goal achieved?
        if state.goal_achieved():
            return {'status': 'success', 'iterations': iteration}
        
        # THINK: Reason about what to do
        decision = reason_about_options(goal, state, history)
        
        # CHECK: Should we escalate instead?
        if decision.requires_human_input:
            return {'status': 'escalated', 'reason': decision.reason}
        
        # ACT: Execute decision
        action_result = execute_action(decision)
        
        # VERIFY: Did it work?
        verification = verify_action(action_result)
        
        # LEARN: Remember what happened
        history.append({
            'iteration': iteration,
            'observations': observations,
            'decision': decision,
            'action_result': action_result,
            'verification': verification
        })
        
        if not verification.success:
            state.record_failure(action_result)
    
    return {'status': 'max_iterations_reached'}
```

**Key insight**: The loop is **data-driven**. Each iteration:
1. Gathers new data (OBSERVE)
2. Updates understanding of state
3. Makes decision based on current state
4. Executes
5. Verifies and learns

This is fundamentally different from a pre-programmed script, which follows a fixed sequence regardless of outcomes.

### 1.4 Agent vs Script: A Detailed Comparison

To understand when agents are appropriate, compare them systematically:

```
SCRIPT APPROACH:
if bgp_neighbors_down:
    check_bgp_config()
    if config_wrong:
        fix_bgp_config()
    elif adjacency_down:
        check_links()
        if link_down:
            escalate_to_human()

Problem 1: Explosion of branches
- N conditions → 2^N possible paths
- 5 conditions → 32 possible paths (hard to maintain)
- 10 conditions → 1024 possible paths (impossible)

Problem 2: No reasoning
- Script can't understand "why" a condition exists
- Can't adapt to novel situations
- No learning from experience

Problem 3: Brittleness
- Small change in conditions breaks assumptions
- Requires code modification to adapt


AGENT APPROACH:
goal = "fix_bgp_connectivity"

while not goal_achieved:
    observations = check_current_state()
    
    decision = claude.reason(
        "Given observations, what's the best next action?"
    )
    
    execute(decision)
    verify()

Advantage 1: Flexibility
- Agent reasons about novel situations
- No predetermined path needed
- Handles unexpected scenarios

Advantage 2: Learning
- Agent learns from what works
- Improves over time
- Captures context in prompts

Advantage 3: Scalability
- Works with any number of conditions
- Conditions added without code changes
- Team can update without engineering
```

**Detailed Trade-off Analysis**:

| Aspect | Script | Agent |
|--------|--------|-------|
| Predictability | Very high (follows fixed path) | Medium (follows reasoning) |
| Maintainability | Hard (explosion of branches) | Easier (reasoning is central) |
| Novelty handling | Poor (breaks on new situations) | Good (reasons through unknowns) |
| Speed | Fast (no thinking required) | Slower (includes reasoning time) |
| Implementation time | Medium (lots of conditions) | Higher (prompt engineering) |
| Debugging when wrong | Easy (trace through if-else) | Harder (why did it think that?) |
| Cost | Low (no LLM calls) | High (LLM calls per iteration) |
| Team skill needed | Mid (programming) | Higher (prompt eng + programming) |

**Decision rule:**
- **Use scripts** if: Problem is well-specified, small number of scenarios, cost-sensitive
- **Use agents** if: Problem is complex, novel situations occur, human oversight available

Most enterprise networks need **both**: scripts for simple, repetitive tasks; agents for complex troubleshooting.

---

## Part 2: The Agent Loop in Detail - How Agents Actually Work

### 2.1 The OBSERVE Phase: Perception and State Gathering

Before an agent can think, it must understand its situation. This is the observation phase:

```python
def observe_environment(agent_context):
    """
    The OBSERVE phase: gather all relevant information
    """
    
    observations = {}
    
    # Get device state
    observations['device_status'] = {
        'router_core_01': check_device_alive('router_core_01'),
        'router_core_02': check_device_alive('router_core_02'),
    }
    
    # Get current configurations
    observations['configs'] = {
        'router_core_01': get_device_config('router_core_01'),
        'router_core_02': get_device_config('router_core_02'),
    }
    
    # Get metrics
    observations['metrics'] = {
        'cpu': get_device_cpu('router_core_01'),
        'memory': get_device_memory('router_core_01'),
        'bgp_neighbors': get_bgp_neighbors('router_core_01'),
    }
    
    # Get logs (recent errors/warnings)
    observations['logs'] = get_recent_logs('router_core_01', last_n=50)
    
    return observations
```

**Theory**: Observation quality determines reasoning quality. Incomplete observations lead to poor decisions.

**Constraint**: You cannot observe more than your tools allow. If you don't have a "get_logs" tool, agent can't analyze logs.

**Design principle**: Include all observation tools that might be relevant. Better to have unused tools than to need one and not have it.

### 2.2 The THINK Phase: Reasoning and Decision-Making

Now the agent must reason about observations:

```python
def think_about_situation(goal, observations, history, rag_system):
    """
    The THINK phase: reason about what to do
    
    This is where Claude's reasoning capability shines.
    """
    
    # Build context for Claude
    context = f"""
    GOAL: {goal}
    
    CURRENT OBSERVATIONS:
    {json.dumps(observations, indent=2)}
    
    PREVIOUS ACTIONS (what we've already tried):
    {json.dumps(history, indent=2)}
    
    AVAILABLE TOOLS:
    {list_available_tools()}
    
    RELEVANT DOCUMENTATION:
    {rag_system.search("How to troubleshoot BGP issues")}
    """
    
    # Ask Claude to reason
    reasoning_prompt = f"""
    {context}
    
    Based on the above information:
    1. What is the root cause of the problem?
    2. What have we already ruled out?
    3. What is the best next action?
    4. Why is that the best choice?
    
    Think carefully. Consider multiple possibilities before deciding.
    """
    
    decision = claude.think(reasoning_prompt)
    
    return {
        'reasoning': decision['thinking'],
        'root_cause': decision['root_cause'],
        'next_action': decision['next_action'],
        'confidence': decision.get('confidence', 'medium')
    }
```

**Key concept**: This is where the agent demonstrates intelligence. The quality of reasoning depends on:

1. **Observation completeness** - Do we have all needed information?
2. **Context richness** - Do we explain the situation well?
3. **Documentation quality** - What does RAG return?
4. **Prompt quality** - How clearly do we ask for reasoning?

### 2.3 The ACT Phase: Taking Action

Once the agent decides what to do, it must execute:

```python
def act_on_decision(decision, safety_constraints):
    """
    The ACT phase: execute the chosen action
    
    But with safety checks!
    """
    
    tool_name = decision['tool']
    tool_args = decision['args']
    
    # Safety Check 1: Is this tool available?
    if tool_name not in AVAILABLE_TOOLS:
        return {
            'status': 'error',
            'reason': f'Tool {tool_name} not available',
            'escalate': True
        }
    
    tool = AVAILABLE_TOOLS[tool_name]
    
    # Safety Check 2: Does this tool require approval?
    if tool.requires_approval:
        approval = request_approval(tool, tool_args)
        if not approval:
            return {
                'status': 'rejected',
                'reason': 'Human rejected this action',
                'escalate': True
            }
    
    # Safety Check 3: Rate limiting
    if tool.is_write_operation:
        if check_rate_limit_exceeded(tool):
            return {
                'status': 'rate_limited',
                'reason': 'Too many write operations in this time window',
                'escalate': True
            }
    
    # Safety Check 4: Backup before write operations
    if tool.is_write_operation:
        backup = create_backup()
    
    # Execute the action
    try:
        result = tool.execute(**tool_args)
        return {
            'status': 'success',
            'result': result,
            'backup': backup if tool.is_write_operation else None
        }
    except Exception as e:
        # Automatic rollback on failure
        if tool.is_write_operation:
            rollback(backup)
        
        return {
            'status': 'failed',
            'error': str(e),
            'rolled_back': True,
            'escalate': True
        }
```

**Theory**: Action execution must be **safe by default**. Safety mechanisms should be:
- Automatic (no human has to remember)
- Layered (multiple checks)
- Verifiable (we can prove they ran)

### 2.4 The VERIFY Phase: Confirming Success

After acting, the agent must verify the action worked:

```python
def verify_action(action_result, original_goal):
    """
    The VERIFY phase: confirm the action achieved its purpose
    
    Verification is critical - don't assume success.
    """
    
    if action_result['status'] != 'success':
        return {
            'verification_passed': False,
            'reason': 'Action failed'
        }
    
    # Wait for system to stabilize
    time.sleep(5)
    
    # Gather new observations
    new_observations = observe_environment()
    
    # Check if goal is achieved
    if achieved_goal(new_observations, original_goal):
        return {
            'verification_passed': True,
            'goal_achieved': True,
            'observations': new_observations
        }
    
    # Check if we made progress (even if goal not fully achieved)
    if made_progress(new_observations, original_goal):
        return {
            'verification_passed': True,
            'goal_achieved': False,
            'progress_made': True,
            'next_step_needed': True,
            'observations': new_observations
        }
    
    # If nothing changed, action didn't work
    return {
        'verification_passed': False,
        'goal_achieved': False,
        'progress_made': False,
        'observations': new_observations,
        'recommendation': 'Try different approach'
    }
```

**Critical principle**: Never assume an action succeeded. Verify it with fresh observations.

---

## Part 3: Prompt Engineering for Agents - The Art and Science

### 3.1 The Prompt Structure: Setting Up Claude to Think Like an Agent

The quality of agent reasoning depends entirely on the prompt. This is not obvious but absolutely critical.

```python
def build_agent_prompt(goal, observations, history, tools, rag_docs):
    """
    Build a well-structured prompt for the agent to reason
    
    The prompt structure matters a lot.
    """
    
    prompt = f"""
You are a network operations agent. Your job is to help diagnose and fix network issues.

# CONTEXT
## Goal
{goal}

## Current Observations
{format_observations(observations)}

## What We've Already Tried
{format_history(history)}

## Available Tools
{format_tools(tools)}

## Relevant Documentation
{format_rag_docs(rag_docs)}

# DECISION FRAMEWORK
Think through these questions in order:

1. **Understanding the Problem**
   - What is the current state?
   - What is the desired state?
   - What's the gap?

2. **Root Cause Analysis**
   - What are the possible root causes?
   - Which is most likely?
   - What evidence supports this?

3. **Consider Your Options**
   - What are all possible next actions?
   - What would each action accomplish?
   - What are the risks?

4. **Choose the Best Action**
   - Of all options, which is best?
   - Why is it the best?
   - What's the confidence level?

5. **Plan the Action**
   - What tool will you use?
   - What arguments to that tool?
   - What's the expected outcome?

# OUTPUT FORMAT
Return your reasoning in this exact format:

ROOT_CAUSE: [Your diagnosis]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ACTION: [Tool name]
ARGS: [Arguments as JSON]
REASONING: [Explain your choice]
RISK_LEVEL: [HIGH/MEDIUM/LOW]
ALTERNATIVE_ACTIONS: [Other options considered]
"""
    
    return prompt
```

**Key design principles:**

1. **Structure the prompt** - Organize information clearly
2. **Give decision framework** - Tell Claude how to think
3. **Request output format** - Make parsing unambiguous
4. **Include context** - Observations, history, tools, docs
5. **Ask for reasoning** - Explain the thought process

### 3.2 System Prompt vs User Prompt

The system prompt sets the agent's "personality" and constraints:

```python
SYSTEM_PROMPT = """
You are a careful, deliberate network operations agent. 

YOUR CONSTRAINTS:
- Never recommend an action you don't fully understand
- Always verify your assumptions
- Always check documentation before acting
- Escalate when uncertain
- Prioritize safety over speed

YOUR CAPABILITIES:
- You can call tools to observe network state
- You can read and understand network documentation
- You can reason about complex technical problems
- You can learn from your history of actions

YOUR LIMITATIONS:
- You cannot make changes without approval (for HIGH risk)
- You cannot predict the future
- You cannot understand unstated goals
- You cannot learn across sessions
"""

def run_agent(goal, observations, history):
    """Run agent with system and user prompts"""
    
    # System prompt: sets personality
    response = claude.messages.create(
        system=SYSTEM_PROMPT,
        
        # User prompt: specific task
        user=build_agent_prompt(goal, observations, history, ...),
        
        model="claude-sonnet-4-20250514",
        max_tokens=2000
    )
    
    return response
```

**Why this matters**: System prompt controls agent behavior across all sessions. If you want cautious agents, make that clear in system prompt.

### 3.3 Few-Shot Prompting: Teaching by Example

Agents learn from examples. Including examples in the prompt improves reasoning:

```python
def build_prompt_with_examples():
    """
    Few-shot prompting: include examples of good reasoning
    """
    
    prompt = """
You are a network agent. Here are examples of good decisions:

EXAMPLE 1: Interface Down
Observations: Interface Gi0/1 is down on router-core-01
Tools available: check_interface_config, check_physical_link, check_logs

Good reasoning:
ROOT_CAUSE: Physical link problem
ACTION: check_physical_link(interface=Gi0/1, device=router-core-01)
REASONING: Interface is down - could be physical or configuration. 
Physical link check is safer (read-only) than config change. Start there.

EXAMPLE 2: BGP Routes Not Advertising
Observations: BGP is running, neighbors are up, but routes not advertising
Tools available: search_docs, get_bgp_config, check_routing_table

Good reasoning:
ROOT_CAUSE: Redistribute policy missing (most common)
ACTION: search_documentation(query="BGP route redistribution requirements")
REASONING: Before making changes, search docs for proper procedure.
Documentation should clarify what redistribute statement is needed.

Now, for this situation:
Observations: {observations}
...
"""
    
    return prompt
```

**Why this matters**: Examples help Claude understand the style of reasoning you want. Good examples → better decisions.

### 3.4 Chain-of-Thought Prompting: Making Claude Show Its Work

Explicitly ask Claude to show its reasoning step-by-step:

```python
def chain_of_thought_prompt():
    """
    Chain-of-thought: ask for step-by-step reasoning
    
    This improves accuracy significantly.
    """
    
    prompt = """
Diagnose the issue step-by-step.

STEP 1: What is the current state?
[First, describe what you observe in detail]

STEP 2: What is the desired state?
[Second, describe the goal clearly]

STEP 3: What's the gap?
[Third, identify what's wrong]

STEP 4: What could cause this gap?
[List all possible root causes]

STEP 5: Which root cause is most likely?
[Eliminate unlikely causes, narrow down]

STEP 6: What action would test this hypothesis?
[What tool would help confirm?]

STEP 7: What's the expected outcome?
[If I do this action, what should I see?]

STEP 8: Decision
[Given all above, what's the best action?]
"""
    
    return prompt
```

**Why this matters**: Forcing explicit step-by-step reasoning reduces errors and makes reasoning auditable.

---

## Part 4: Tool Design for Agents - The Agent's Interface to the World

### 4.1 Designing Tools: From Theory to Implementation

A tool is not just a function. It's an **interface between agent and world**. Good tool design is critical.

```python
@dataclass
class Tool:
    """
    Complete tool specification
    """
    # Identity
    name: str
    description: str  # What the tool does
    
    # Execution
    execute_func: Callable  # The actual function
    
    # Safety
    is_write_operation: bool  # Does it modify state?
    is_read_only: bool  # Just observing?
    requires_approval: bool  # Need human OK?
    
    # Constraints
    rate_limit: Optional[int]  # Max calls per hour?
    risk_level: str  # LOW/MEDIUM/HIGH
    
    # Documentation for agent
    parameters: Dict[str, str]  # Parameter descriptions
    return_description: str  # What does it return?
    preconditions: List[str]  # What must be true first?
    postconditions: List[str]  # What happens after?
    side_effects: List[str]  # What else might change?
    
    # Safety mechanisms
    rollback_supported: bool  # Can we undo?
    
    def __str__(self) -> str:
        """Format for agent to understand"""
        return f"""
Tool: {self.name}
Purpose: {self.description}
Parameters: {self.parameters}
Returns: {self.return_description}
Risk Level: {self.risk_level}
Requires Approval: {self.requires_approval}
Can be rolled back: {self.rollback_supported}
"""
```

**Key design principle**: Tools should be **self-documenting** so agents understand them correctly.

### 4.2 Tool Ontology: Organizing Tools Hierarchically

As you add more tools, organize them in a taxonomy:

```
READ TOOLS (Safe, no approval needed)
├── Observation Tools
│   ├─ get_device_status
│   ├─ get_device_config
│   ├─ get_interface_status
│   └─ get_routing_table
├── Analysis Tools
│   ├─ check_bgp_status
│   ├─ check_ospf_status
│   └─ analyze_routing_table
└── Documentation Tools
    ├─ search_documentation (RAG)
    └─ get_best_practices

WRITE TOOLS (Require approval)
├── Low-Risk Write Tools
│   ├─ enable_debug_logging
│   ├─ disable_debug_logging
│   └─ clear_statistics
├── Medium-Risk Write Tools
│   ├─ apply_interface_config
│   ├─ change_ospf_cost
│   └─ adjust_bgp_timers
└── High-Risk Write Tools
    ├─ change_bgp_asn
    ├─ delete_route
    └─ shutdown_interface

COMMUNICATION TOOLS (No approval needed)
├─ send_slack_notification
├─ send_email
└─ create_ticket
```

**Why this matters**: When agent has 50+ tools, it must understand relationships between them.

### 4.3 Tool Error Handling: Building Resilient Tools

Tools should handle errors gracefully:

```python
def create_tool(name, execute_func):
    """
    Wrap a function as a tool with error handling
    """
    
    def robust_execute(**kwargs):
        """
        Execute tool with proper error handling
        """
        
        # Validate inputs
        try:
            validate_inputs(kwargs)
        except ValueError as e:
            return {
                'status': 'error',
                'error_type': 'invalid_input',
                'message': str(e),
                'user_message': 'The arguments you provided are invalid. Here\'s why...'
            }
        
        # Execute with timeout
        try:
            result = timeout_call(execute_func, kwargs, timeout=30)
        except TimeoutError:
            return {
                'status': 'error',
                'error_type': 'timeout',
                'message': f'Tool {name} took too long (>30s)',
                'user_message': 'This operation timed out. The device might be slow.'
            }
        except ConnectionError as e:
            return {
                'status': 'error',
                'error_type': 'connection_failed',
                'message': str(e),
                'user_message': 'Could not reach the device. It might be down.'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'unexpected',
                'message': str(e),
                'user_message': 'An unexpected error occurred. Please investigate.'
            }
        
        # Validate output
        try:
            validate_output(result)
        except ValueError as e:
            return {
                'status': 'error',
                'error_type': 'invalid_output',
                'message': str(e),
                'user_message': 'The device returned unexpected data.'
            }
        
        # Success
        return {
            'status': 'success',
            'data': result
        }
    
    return Tool(
        name=name,
        execute_func=robust_execute,
        ...
    )
```

**Key principle**: Tools should fail gracefully, returning meaningful error messages that agents can understand and react to.

---

## Part 5: State Management in Agents

### 5.1 Agent Context: Maintaining State Across Iterations

An agent must maintain state to avoid repeating mistakes:

```python
class AgentContext:
    """
    Maintain agent state across iterations
    """
    
    def __init__(self, goal):
        self.goal = goal
        self.observations = {}  # Current observations
        self.history = []  # What we've done
        self.failed_actions = []  # What didn't work
        self.successful_actions = []  # What did work
        self.current_hypothesis = None  # Our diagnosis
        self.iteration = 0
        self.max_iterations = 10
    
    def record_observation(self, obs_type, data):
        """Record what we observe"""
        self.observations[obs_type] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def record_action(self, action, result, success):
        """Record what we tried and if it worked"""
        self.history.append({
            'iteration': self.iteration,
            'action': action,
            'result': result,
            'success': success,
            'timestamp': datetime.now()
        })
        
        if success:
            self.successful_actions.append(action)
        else:
            self.failed_actions.append(action)
    
    def should_escalate(self) -> bool:
        """
        Decide if we should give up and ask for human help
        """
        
        # Escalate if we've hit iteration limit
        if self.iteration >= self.max_iterations:
            return True
        
        # Escalate if we've tried the same thing twice unsuccessfully
        if self.has_tried_twice_unsuccessfully():
            return True
        
        # Escalate if we're stuck (no progress in last 3 iterations)
        if self.no_progress_last_n_iterations(3):
            return True
        
        # Otherwise, keep trying
        return False
    
    def get_summary_for_prompt(self):
        """
        Summarize context for Claude
        """
        return f"""
GOAL: {self.goal}

CURRENT OBSERVATIONS:
{self.format_observations()}

WHAT WE'VE TRIED:
{self.format_history()}

WHAT WORKED:
{self.successful_actions}

WHAT DIDN'T WORK:
{self.failed_actions}

CURRENT HYPOTHESIS:
{self.current_hypothesis}

ITERATION: {self.iteration}/{self.max_iterations}
"""
```

**Critical principle**: An agent without state is like a human with amnesia. Each iteration would start from scratch.

### 5.2 Memory and Learning: Building Agent Knowledge

Agents should improve through experience:

```python
class AgentMemory:
    """
    Remember what we learn for future episodes
    """
    
    def __init__(self):
        self.successful_patterns = []  # Techniques that worked
        self.failure_patterns = []  # Mistakes to avoid
        self.domain_knowledge = {}  # What we learned about this network
    
    def record_success(self, problem, solution, outcome):
        """Learn from successes"""
        self.successful_patterns.append({
            'problem': problem,
            'solution': solution,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
    
    def record_failure(self, attempted_solution, reason):
        """Learn from failures"""
        self.failure_patterns.append({
            'solution': attempted_solution,
            'why_it_failed': reason,
            'timestamp': datetime.now()
        })
    
    def get_relevant_memories(self, current_problem):
        """
        When facing a new problem, look for similar past problems
        """
        relevant = []
        for pattern in self.successful_patterns:
            if self.is_similar(current_problem, pattern['problem']):
                relevant.append(pattern)
        return relevant
```

**Important note**: Agents need persistent memory to improve. Without it, every episode is treated as a new problem.

---

## Part 6: Advanced Agent Decision-Making

### 6.1 Decision Trees in Agent Context

Agents must navigate decision spaces. Understanding decision theory helps:

```python
def agent_decision_analysis(observations, available_actions):
    """
    Analyze decisions systematically
    
    Agents should think in terms of decision trees.
    """
    
    decision_tree = {
        'problem': 'BGP routes not advertising',
        
        'decisions': [
            {
                'decision': 'Is BGP running?',
                'outcomes': {
                    'yes': 'Move to next decision',
                    'no': 'Action: start BGP'
                }
            },
            {
                'decision': 'Are BGP neighbors up?',
                'outcomes': {
                    'yes': 'Move to next decision',
                    'no': 'Action: fix BGP neighbor config'
                }
            },
            {
                'decision': 'Is redistribute configured?',
                'outcomes': {
                    'yes': 'Routes should be advertising',
                    'no': 'Action: add redistribute statement'
                }
            }
        ]
    }
    
    # Agent traverses this tree by observation
```

**Key insight**: Good agent prompts should guide Claude to think in decision trees.

### 6.2 Confidence Scoring: Knowing When You're Uncertain

Agents must express certainty:

```python
def score_confidence(decision):
    """
    Rate confidence in decision
    
    HIGH: >80% confident this is right
    MEDIUM: 50-80% confident
    LOW: <50% confident
    """
    
    factors = {
        'observation_quality': 0.8,  # How complete are observations?
        'past_success_rate': 0.7,  # Similar problems always resolved this way?
        'risk_level': 0.6,  # How risky is this action?
        'documentation_clarity': 0.9,  # How clear is the documentation?
    }
    
    overall = sum(factors.values()) / len(factors)
    
    if overall > 0.8:
        return 'HIGH'
    elif overall > 0.5:
        return 'MEDIUM'
    else:
        return 'LOW'
    
    # Rule: If confidence is LOW, escalate to human
```

### 6.3 Exploration vs Exploitation

Should the agent try new approaches or stick with what works?

```python
def choose_action_strategy(history):
    """
    Balance trying proven solutions vs exploring new ones
    
    This is the exploration-exploitation trade-off
    """
    
    # Count successes
    success_rate = count_successes(history) / len(history)
    
    if success_rate > 0.8:
        # We're doing well - mostly exploit
        strategy = 'mostly_exploit_some_exploration'
    elif success_rate > 0.5:
        # Mixed success - balance both
        strategy = 'balanced'
    else:
        # Not working - explore more
        strategy = 'mostly_explore_some_exploitation'
    
    if strategy == 'mostly_exploit_some_exploration':
        # 90% of time: choose proven solutions
        # 10% of time: try novel approaches
        
    return strategy
```

---

## Part 7: Real-World Implementation Patterns

### 7.1 The Diagnostic Agent Pattern

A common pattern for network troubleshooting:

```python
class DiagnosticAgent:
    """
    Systematically diagnose network issues
    
    Pattern: OBSERVE → HYPOTHESIS → TEST → CONFIRM → ACT
    """
    
    def diagnose(self, symptom):
        """
        Given a symptom, find root cause
        """
        
        # Phase 1: OBSERVE
        observations = self.gather_observations(symptom)
        
        # Phase 2: HYPOTHESIS GENERATION
        hypotheses = self.generate_hypotheses(observations)
        # Examples: "Link failure", "Configuration error", "Policy issue"
        
        # Phase 3: HYPOTHESIS TESTING
        for hypothesis in sorted(hypotheses, key=lambda h: h['likelihood']):
            test_result = self.test_hypothesis(hypothesis, observations)
            
            if test_result['confirmed']:
                root_cause = hypothesis
                break
        
        # Phase 4: SOLUTION
        solution = self.find_solution(root_cause)
        
        # Phase 5: EXECUTION (with approval)
        self.execute_solution(solution)
        
        # Phase 6: VERIFICATION
        self.verify_solution_worked()
        
        return root_cause
```

### 7.2 The Patrol Agent Pattern

Continuously monitoring and proactive fixing:

```python
class PatrolAgent:
    """
    Continuously monitor, detect issues, fix proactively
    
    Pattern: Infinite loop of observation and correction
    """
    
    def patrol(self):
        """Run continuously"""
        
        while True:
            # Observe
            state = self.observe_network()
            
            # Detect issues
            issues = self.detect_issues(state)
            
            # Fix automatically
            for issue in issues:
                if issue['severity'] == 'LOW':
                    # Auto-fix low severity
                    self.fix_automatically(issue)
                elif issue['severity'] == 'MEDIUM':
                    # Alert and propose fix
                    self.alert_and_propose(issue)
                else:
                    # High severity: escalate to human immediately
                    self.escalate_to_human(issue)
            
            # Sleep before next patrol
            time.sleep(60)
```

### 7.3 The Responder Agent Pattern

Respond to human requests intelligently:

```python
class ResponderAgent:
    """
    Answer human questions about network state
    
    Pattern: QUERY → UNDERSTAND → SEARCH → SYNTHESIZE → ANSWER
    """
    
    def answer_question(self, question):
        """
        Answer a network question
        """
        
        # UNDERSTAND: What is being asked?
        intent = claude.understand_intent(question)
        
        # SEARCH: Find relevant documentation and data
        docs = self.rag_system.search(intent)
        observations = self.gather_observations(intent)
        
        # SYNTHESIZE: Combine docs and observations
        context = docs + observations
        
        # ANSWER: Generate response
        answer = claude.synthesize_answer(question, context)
        
        return answer
```

---

## Part 8: Testing and Validation

### 8.1 Agent Testing: How to Know If It Works

```python
def test_agent(agent, test_cases):
    """
    Test an agent against known problems
    """
    
    results = []
    
    for test_case in test_cases:
        # Run agent
        result = agent.run(
            goal=test_case['goal'],
            initial_observations=test_case['observations']
        )
        
        # Check if it succeeded
        success = result['goal_achieved']
        
        # Check quality metrics
        iterations = result['iterations_used']
        cost = result['api_cost']
        
        results.append({
            'test': test_case['name'],
            'success': success,
            'iterations': iterations,
            'cost': cost
        })
    
    # Summarize
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_iterations = sum(r['iterations'] for r in results) / len(results)
    total_cost = sum(r['cost'] for r in results)
    
    return {
        'success_rate': success_rate,
        'avg_iterations': avg_iterations,
        'total_cost': total_cost,
        'results': results
    }
```

### 8.2 Benchmark Test Cases

Every agent should be tested against standard scenarios:

```python
BENCHMARK_TEST_CASES = [
    {
        'name': 'Simple: Interface Down',
        'goal': 'Fix interface down on router-core-01',
        'observations': {
            'interface_status': 'down',
            'interface_error': 'physical link down'
        },
        'expected_solution': 'Check physical link'
    },
    {
        'name': 'Medium: BGP Neighbors Down',
        'goal': 'Fix BGP neighbors not responding',
        'observations': {
            'bgp_neighbors': 'down',
            'bgp_running': True,
            'interfaces': 'up'
        },
        'expected_solution': 'Check BGP config or neighbors down'
    },
    {
        'name': 'Complex: Intermittent Issues',
        'goal': 'Find intermittent connectivity issue',
        'observations': {
            'status': 'intermittent',
            'no_constant_pattern': True
        },
        'expected_solution': 'Detailed analysis required'
    }
]
```

---

## Part 9: Complete Working Example

### A Full Agent Implementation

```python
class NetworkOperationsAgent:
    """
    Complete, working agent for network operations
    """
    
    def __init__(self, tools, rag_system):
        self.tools = tools
        self.rag = rag_system
        self.context = AgentContext(goal=None)
        self.memory = AgentMemory()
    
    def run(self, goal, initial_observations):
        """
        Execute the full agent loop
        """
        
        self.context.goal = goal
        self.context.record_observation('initial', initial_observations)
        
        while not self.context.should_escalate():
            self.context.iteration += 1
            
            # OBSERVE
            observations = self.observe()
            self.context.record_observation(f'iteration_{self.context.iteration}', observations)
            
            # THINK
            decision = self.think()
            
            # ACT
            action_result = self.act(decision)
            
            # VERIFY
            verification = self.verify(action_result)
            
            # RECORD
            self.context.record_action(decision, action_result, verification['success'])
            
            # SUCCESS?
            if verification['goal_achieved']:
                return self.success_result()
        
        return self.escalation_result()
    
    def think(self):
        """Ask Claude to reason"""
        
        prompt = self.build_thinking_prompt()
        
        response = claude.think(
            system="You are a careful network agent. Think step by step.",
            user=prompt
        )
        
        return self.parse_decision(response)
    
    def build_thinking_prompt(self):
        """Build a good thinking prompt"""
        
        relevant_memories = self.memory.get_relevant_memories(
            self.context.goal
        )
        
        prompt = f"""
GOAL: {self.context.goal}

CONTEXT:
{self.context.get_summary_for_prompt()}

RELEVANT PAST SUCCESSES:
{self.format_memories(relevant_memories)}

Based on all above, what's the best next action?

Think carefully through:
1. Root cause
2. Options
3. Best choice
4. Why

Then provide: ACTION, ARGS, REASONING, CONFIDENCE
"""
        
        return prompt
    
    def observe(self):
        """Gather observations"""
        
        observations = {
            'device_status': self.tools['check_device_status'](),
            'configs': self.tools['get_configs'](),
            'metrics': self.tools['get_metrics'](),
        }
        
        return observations
    
    def act(self, decision):
        """Execute decision"""
        
        tool = decision['tool']
        args = decision['args']
        
        # Safety check
        if self.tools[tool].requires_approval:
            approval = request_approval(tool, args)
            if not approval:
                return {'status': 'rejected'}
        
        # Execute
        result = self.tools[tool].execute(**args)
        
        return result
    
    def verify(self, action_result):
        """Verify action worked"""
        
        if action_result['status'] != 'success':
            return {'success': False}
        
        # Wait for convergence
        time.sleep(5)
        
        # Check if goal achieved
        observations = self.observe()
        goal_achieved = self.check_goal(observations)
        
        return {'success': True, 'goal_achieved': goal_achieved}
    
    def success_result(self):
        """Format successful result"""
        
        return {
            'status': 'success',
            'goal': self.context.goal,
            'iterations': self.context.iteration,
            'actions_taken': self.context.history,
            'api_cost': self.calculate_cost()
        }
```

---

## Summary: Key Theoretical Concepts

1. **Agents as Decision Makers**: Agents solve problems through reasoning + action, not pre-programmed scripts

2. **The Agent Loop**: OBSERVE → THINK → ACT → VERIFY repeats until goal achieved

3. **Prompt Engineering**: Quality of reasoning depends on prompt quality

4. **Tool Design**: Tools must be safe, well-documented, error-handling

5. **State and Context**: Agents need memory to avoid repeating mistakes

6. **Safety First**: All write operations require approval + verification + rollback capability

7. **Escalation**: Know when to give up and ask for human help

8. **Testing and Validation**: Benchmark agents against known scenarios

---

## Next Steps: Building Your First Agent

1. **Define your goal** - What problem does the agent solve?
2. **Build tools** - What capabilities does it need?
3. **Design prompts** - How should it reason?
4. **Test thoroughly** - Benchmark against known scenarios
5. **Deploy safely** - Start with read-only, graduate to write operations
6. **Monitor and improve** - Track what works and doesn't

---

## Chapter 15 Complete ✓

*Agents represent the future of network operations. Understanding their theory and practice is essential.*
