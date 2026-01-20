# Single-Agent vs Multi-Agent: Decision Guide

When should you use a multi-agent system versus a simple single-agent approach? This guide helps you make the right architectural decision.

## Quick Decision Tree

```
Is the task complex with multiple domains?
│
├─ No → Use Single Agent
│
└─ Yes → Can subtasks be parallelized?
          │
          ├─ No → Use Single Agent with Tools
          │
          └─ Yes → Do subtasks require different expertise?
                   │
                   ├─ No → Use Single Agent with Parallel Tools
                   │
                   └─ Yes → Use Multi-Agent System
```

## Single Agent Architecture

### When to Use

✓ **Simple, focused tasks**
- Single domain expertise needed
- Linear workflow
- Quick execution required
- Limited complexity

✓ **Examples**:
- "Check interface status"
- "Generate VLAN configuration"
- "Validate BGP config"
- "Parse syslog messages"

### Example: Single Agent

```python
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

@tool
def check_interface(interface: str) -> str:
    """Check interface status."""
    return "Interface status: up/up"

class SimpleAgent:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    def execute(self, task: str):
        llm_with_tools = self.llm.bind_tools([check_interface])
        response = llm_with_tools.invoke([HumanMessage(content=task)])
        return response.content

# Usage
agent = SimpleAgent()
result = agent.execute("Check GigabitEthernet0/0 status")
```

**Pros**:
- Simple to implement and maintain
- Low latency (single LLM call)
- Predictable behavior
- Lower cost (fewer API calls)

**Cons**:
- Limited to single domain
- Cannot parallelize work
- No specialization
- Harder to scale complexity

### Complexity Threshold

Use single agent when:
- Task completable in < 5 steps
- Single domain of expertise
- No parallelization benefits
- Cost/latency critical

## Multi-Agent Architecture

### When to Use

✓ **Complex, multi-domain tasks**
- Multiple expertise areas needed
- Parallel execution possible
- Different perspectives required
- Coordinated workflows

✓ **Examples**:
- "Complete network audit" (security + performance + config)
- "Diagnose and fix production incident" (diagnosis → analysis → remediation)
- "Change impact assessment" (security + performance + compliance)
- "Infrastructure optimization" (multiple systems in parallel)

### Example: Multi-Agent

```python
class SupervisorAgent:
    def __init__(self):
        self.diagnosis_agent = DiagnosisAgent()
        self.security_agent = SecurityAgent()
        self.performance_agent = PerformanceAgent()

    def execute(self, request: str):
        # Parallel execution
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.diagnosis_agent.diagnose, request): "diagnosis",
                executor.submit(self.security_agent.analyze, request): "security",
                executor.submit(self.performance_agent.analyze, request): "performance"
            }

            results = {}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        # Synthesize results
        return self.synthesize(results)

# Usage
supervisor = SupervisorAgent()
result = supervisor.execute("Analyze core router performance issues")
```

**Pros**:
- Specialized expertise per agent
- Parallel execution (faster)
- Modular and maintainable
- Scales to complex problems

**Cons**:
- Higher implementation complexity
- More API calls (higher cost)
- Coordination overhead
- Harder to debug

### Complexity Threshold

Use multi-agent when:
- Task requires > 5 steps
- Multiple domains involved
- Parallelization provides 2x+ speedup
- Need for specialized expertise

## Detailed Comparison

### Performance

| Aspect | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| **Latency** | 1-2 seconds | 3-5 seconds (sequential)<br>1-2 seconds (parallel) |
| **API Calls** | 1-3 calls | 5-15 calls |
| **Cost** | $0.01-0.05 | $0.10-0.50 |
| **Throughput** | 1 task at a time | N tasks in parallel |
| **Scalability** | Limited | High |

### Use Case Matrix

| Use Case | Complexity | Best Choice | Why |
|----------|-----------|-------------|-----|
| **Interface diagnostics** | Low | Single Agent | Simple, focused task |
| **Config generation** | Low-Medium | Single Agent | Linear workflow |
| **Security audit** | Medium | Single Agent | Single domain |
| **Complete infrastructure audit** | High | Multi-Agent | Multiple domains, parallelizable |
| **Incident response** | High | Multi-Agent | Multiple phases, different expertise |
| **Change validation** | Medium-High | Multi-Agent | Multiple perspectives needed |
| **Performance optimization** | Medium | Single Agent | Focused domain |
| **Compliance + Security + Performance** | High | Multi-Agent | Independent analyses |

### Code Complexity

**Single Agent**: ~50-150 lines
```python
# Simple agent with tools
class Agent:
    - __init__()
    - execute()
    - 2-3 tool functions

# Total: ~100 lines
```

**Multi-Agent**: ~500-1000 lines
```python
# Multiple specialist agents
class DiagnosisAgent: ~150 lines
class SecurityAgent: ~150 lines
class PerformanceAgent: ~150 lines
class SupervisorAgent: ~200 lines
# Support code: ~200 lines

# Total: ~800 lines
```

## Real-World Scenarios

### Scenario 1: Quick Interface Check

**Task**: "Is GigabitEthernet0/0 up?"

**Recommendation**: Single Agent

**Rationale**:
- Simple yes/no question
- Single tool call needed
- No parallelization benefit
- Low latency critical

**Implementation**:
```python
agent = SimpleAgent()
result = agent.execute("Check if GigabitEthernet0/0 is up")
# Response time: ~1 second
# Cost: ~$0.01
```

### Scenario 2: Production Incident

**Task**: "Core router down, users cannot access internet"

**Recommendation**: Multi-Agent

**Rationale**:
- Multiple investigation paths
- Different expertise needed (diagnosis, security, performance)
- Parallel diagnostics faster
- Complex problem requiring coordination

**Implementation**:
```python
supervisor = SupervisorAgent()

# Phase 1: Parallel initial assessment (2 seconds)
phase1 = supervisor.execute_parallel([
    DiagnosisTask("Identify root cause"),
    SecurityTask("Check for security breach")
])

# Phase 2: Detailed analysis (3 seconds)
phase2 = supervisor.execute_sequential([
    PerformanceTask("Deep performance analysis", context=phase1)
])

# Phase 3: Remediation (2 seconds)
phase3 = supervisor.execute_sequential([
    ConfigTask("Generate fix configuration", context={**phase1, **phase2})
])

# Total time: ~7 seconds (vs 15+ seconds sequential)
# Cost: ~$0.30
```

### Scenario 3: Daily Health Check

**Task**: "Generate daily health report for all devices"

**Recommendation**: Multi-Agent (Parallel)

**Rationale**:
- Multiple devices (parallelizable)
- Different checks (health, security, performance)
- Not time-critical
- Comprehensive analysis needed

**Implementation**:
```python
supervisor = SupervisorAgent()

devices = ["router-1", "router-2", "switch-1", "firewall-1"]

# Parallel execution across all devices
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(
            supervisor.execute_parallel,
            [
                SecurityTask(f"Audit {device}"),
                PerformanceTask(f"Check {device} health")
            ]
        ): device
        for device in devices
    }

    results = {device: future.result()
               for future, device in futures.items()}

# 4 devices × 2 checks = 8 tasks
# Sequential time: ~16 seconds
# Parallel time: ~4 seconds (4x speedup)
```

### Scenario 4: Configuration Generation

**Task**: "Generate standard branch router configuration"

**Recommendation**: Single Agent

**Rationale**:
- Well-defined template
- Single domain (configuration)
- Linear workflow
- No parallelization benefit

**Implementation**:
```python
config_agent = ConfigAgent()
result = config_agent.generate("""
Generate branch router config:
- WAN: GigabitEthernet0/0 (DHCP)
- LAN: GigabitEthernet0/1 (192.168.1.1/24)
- OSPF area 1
- Security baseline
""")
# Response time: ~2 seconds
# Cost: ~$0.02
```

## Migration Path

### Start Simple, Grow Complex

```
Phase 1: Single Agent
├─ Implement basic functionality
├─ Validate approach
└─ Low complexity, fast iteration

Phase 2: Single Agent + Multiple Tools
├─ Add specialized tools
├─ Expand capabilities
└─ Still manageable complexity

Phase 3: Multi-Agent (Domain Split)
├─ Split by domain (diagnosis, security, config)
├─ Add supervisor coordination
└─ Handle complex workflows

Phase 4: Multi-Agent (Advanced)
├─ Add parallel execution
├─ Implement caching
├─ Add monitoring
└─ Production-grade system
```

### Example Migration

**Version 1: Simple Agent**
```python
class NetworkAgent:
    def analyze(self, problem: str):
        return self.llm.invoke(problem)

# Works for simple queries
```

**Version 2: Agent + Tools**
```python
class NetworkAgent:
    def analyze(self, problem: str):
        llm_with_tools = self.llm.bind_tools([
            check_interface,
            check_routing,
            check_health
        ])
        return llm_with_tools.invoke(problem)

# Handles more complex scenarios
```

**Version 3: Multi-Agent**
```python
class SupervisorAgent:
    def __init__(self):
        self.diagnosis = DiagnosisAgent()
        self.security = SecurityAgent()
        self.performance = PerformanceAgent()

    def analyze(self, problem: str):
        # Coordinate specialist agents
        results = self.execute_parallel([...])
        return self.synthesize(results)

# Production-ready for complex operations
```

## Cost Analysis

### Single Agent Economics

**Simple query**: "Check interface status"
- API calls: 2 (initial + tool execution)
- Tokens: ~500 input + 200 output
- Cost: ~$0.01
- Time: ~1 second

**Daily usage (100 queries)**:
- Cost: ~$1.00
- Time: ~100 seconds

### Multi-Agent Economics

**Complex analysis**: "Complete infrastructure audit"
- API calls: 12 (supervisor + 3 agents + synthesis)
- Tokens: ~5000 input + 2000 output
- Cost: ~$0.30
- Time: ~4 seconds (parallel) or ~12 seconds (sequential)

**Daily usage (20 audits)**:
- Cost: ~$6.00
- Time: ~80 seconds (parallel) or ~240 seconds (sequential)
- Savings: 160 seconds/day from parallelization

### ROI Calculation

**When does multi-agent pay off?**

Break-even occurs when:
```
Time Saved × Hourly Cost > Additional Development Cost
```

Example:
- Development time: 40 hours @ $100/hr = $4,000
- Time saved per day: 160 seconds = 2.67 minutes
- Automation runs: 250 days/year
- Annual time saved: 667 minutes = 11.1 hours
- Annual value (@ $100/hr): $1,110

**ROI**: Break-even in 3.6 years

**But consider**:
- Reduced errors
- Faster incident response
- Better insights
- Scalability benefits

**Realistic ROI**: 6-12 months for high-frequency operations

## Best Practices

### When to Choose Single Agent

1. **Proof of Concept**
   - Validate approach quickly
   - Minimal investment
   - Fast iteration

2. **Well-Defined Tasks**
   - Clear inputs/outputs
   - Single domain
   - Predictable workflow

3. **Cost-Sensitive Applications**
   - High volume, low complexity
   - Budget constraints
   - Simple queries

4. **Low-Latency Requirements**
   - Real-time responses needed
   - < 2 second SLA
   - Interactive applications

### When to Choose Multi-Agent

1. **Production Operations**
   - Complex workflows
   - Multiple domains
   - Quality over speed

2. **Incident Response**
   - Parallel investigation
   - Different perspectives
   - Coordinated remediation

3. **Comprehensive Analysis**
   - Security + Performance + Config
   - Multiple systems
   - Deep insights needed

4. **Scalability Required**
   - Growing complexity
   - Parallel processing
   - Modular architecture

## Decision Checklist

Before implementing multi-agent, ask:

- [ ] Does the task span multiple domains?
- [ ] Can subtasks be parallelized?
- [ ] Is 2x+ speedup achievable?
- [ ] Is the added complexity justified?
- [ ] Will the system grow more complex?
- [ ] Are specialized agents needed?
- [ ] Can you afford higher API costs?
- [ ] Do you have time for proper implementation?

**If 5+ checkmarks**: Multi-agent is likely the right choice

**If 2-4 checkmarks**: Consider single agent with enhanced tools

**If 0-1 checkmarks**: Single agent is sufficient

## Summary

| Criteria | Single Agent | Multi-Agent |
|----------|--------------|-------------|
| **Complexity** | Low-Medium | High |
| **Cost** | Low ($0.01-0.05) | Medium-High ($0.10-0.50) |
| **Speed** | Fast (1-2s) | Medium (parallel) to Slow (sequential) |
| **Maintenance** | Easy | Moderate |
| **Scalability** | Limited | High |
| **Flexibility** | Low | High |
| **Development Time** | Hours | Days-Weeks |
| **Best For** | Simple, focused tasks | Complex, multi-domain operations |

## Conclusion

Start with a **single agent** for most tasks. Migrate to **multi-agent** when:
- Complexity demands specialization
- Parallel execution provides significant benefits
- Multiple domains must be coordinated
- Production quality requires modular architecture

The multi-agent system in this chapter provides a production-ready foundation for complex network operations, but remember: **use the simplest solution that meets your requirements**.
