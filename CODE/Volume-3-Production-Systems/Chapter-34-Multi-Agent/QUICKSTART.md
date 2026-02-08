# Quick Start Guide: Multi-Agent Orchestration

Get started with multi-agent network operations in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- Anthropic API key ([get one here](https://console.anthropic.com/))
- Basic understanding of network operations

## Installation

### Step 1: Clone or Navigate to Directory
```bash
cd Chapter-34-Multi-Agent
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
# ANTHROPIC_API_KEY=sk-ant-...
```

## Run Your First Example

### Basic Test (Single Agent)
```bash
python -c "
from multi_agent import DiagnosisAgent

agent = DiagnosisAgent()
result = agent.diagnose('Interface GigabitEthernet0/0 is flapping')

print(f'Status: {result.status}')
print(f'Confidence: {result.confidence:.0%}')
print(f'Findings: {result.findings[0]}')
"
```

### Full Demo (All Examples)
```bash
python multi_agent.py
```

This runs all 5 examples:
1. Specialist agents working independently
2. Supervisor orchestration pattern
3. Parallel vs sequential execution comparison
4. Complex multi-agent workflow
5. Agent performance metrics

**Runtime**: ~2-3 minutes (with API calls)

## Understanding the Output

### Example 1: Specialist Agents
```
DiagnosisAgent: Troubleshooting interface issue
Status: success
Confidence: 85%
Findings:
  - Layer 1/2 issue detected on GigabitEthernet0/0
  - Frame errors and collisions indicate duplex mismatch
```

### Example 2: Supervisor Orchestration
```
Supervisor creating execution plan...
→ Executing diagnosis: Diagnose connectivity issues
  ✓ Completed in 2.34s
→ Executing performance: Analyze router performance
  ✓ Completed in 1.89s

FINAL SYNTHESIS
The connectivity issues stem from...
```

### Example 3: Parallel Execution
```
Sequential total time: 8.45s
Parallel total time: 3.12s
Speedup: 2.71x
```

## Common Issues

### Issue: API Key Not Found
```
Error: API key not configured
```

**Solution**:
1. Create `.env` file in project directory
2. Add `ANTHROPIC_API_KEY=your_key_here`
3. Restart your Python session

### Issue: Module Not Found
```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: Rate Limit Errors
```
Error: Rate limit exceeded
```

**Solution**:
- Wait 60 seconds and retry
- Reduce parallel execution (MAX_PARALLEL_AGENTS=2 in .env)
- Upgrade API tier if needed

## Next Steps

### 1. Customize for Your Network

Edit the mock data in `multi_agent.py`:

```python
# Replace mock_stats with your actual network data
@tool
def get_interface_statistics(interface: str) -> str:
    # Connect to your device using Netmiko
    from netmiko import ConnectHandler

    device = {
        'device_type': 'cisco_ios',
        'host': 'your_router_ip',
        'username': 'admin',
        'password': 'password',
    }

    connection = ConnectHandler(**device)
    output = connection.send_command(f'show interface {interface}')
    connection.disconnect()

    return output
```

### 2. Add Custom Agents

```python
class BackupAgent:
    """Backup and restore specialist."""

    def __init__(self):
        self.name = "BackupAgent"
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    def backup_config(self, device: str) -> AgentResponse:
        # Your backup logic
        pass
```

### 3. Integrate with Your Tools

Replace simulated tools with real integrations:
- **Netmiko**: Device command execution
- **NAPALM**: Multi-vendor network automation
- **Netbox**: IPAM and inventory
- **Prometheus**: Metrics collection
- **ELK Stack**: Log analysis

### 4. Production Deployment

See Chapter 51 for:
- Docker containerization
- Kubernetes deployment
- Monitoring and alerting
- CI/CD pipelines
- Security hardening

## Example Workflows

### Workflow 1: Troubleshooting
```python
from multi_agent import SupervisorAgent

supervisor = SupervisorAgent()

# User reports connectivity issue
problem = "Cannot reach 10.2.0.0/24 network from 10.1.0.0/24"

# Supervisor coordinates diagnosis
tasks = [
    AgentTask("diagnosis", "troubleshoot", problem),
    AgentTask("performance", "analyze", "Check routing and interfaces")
]

results = supervisor.execute_parallel(tasks)
solution = supervisor.synthesize_results(state)
print(solution)
```

### Workflow 2: Security Audit
```python
# Audit all network devices
tasks = [
    AgentTask("security", "audit", "Core routers"),
    AgentTask("security", "audit", "Access switches"),
    AgentTask("security", "audit", "Firewalls")
]

results = supervisor.execute_parallel(tasks)

# Generate compliance report
for agent_name, result in results.items():
    print(f"\n{agent_name}:")
    print(f"  Risk Score: {result.confidence}")
    print(f"  Vulnerabilities: {len(result.findings)}")
```

### Workflow 3: Change Automation
```python
# Generate and validate configuration change
tasks = [
    AgentTask("config", "generate", "Add VLAN 100 to all access switches"),
    AgentTask("security", "validate", "Check security compliance"),
    AgentTask("diagnosis", "simulate", "Predict impact")
]

# Sequential execution (each depends on previous)
results = supervisor.execute_sequential(tasks)

if all(r.status == "success" for r in results.values()):
    print("✓ Safe to deploy")
else:
    print("⚠ Issues found, review required")
```

## Architecture at a Glance

```
Request: "Network performance issues"
    ↓
Supervisor: Analyzes request, creates plan
    ↓
    ├── DiagnosisAgent: "Check for hardware issues"
    ├── SecurityAgent: "Rule out security breach"
    └── PerformanceAgent: "Identify bottlenecks"
    ↓
Supervisor: Synthesizes results
    ↓
Response: "Root cause: CPU saturation. Fix: Upgrade hardware"
```

## Key Concepts

### 1. Agent Specialization
Each agent is an expert in one domain:
- **Diagnosis**: Root cause analysis
- **Config**: Configuration management
- **Security**: Vulnerability assessment
- **Performance**: Optimization

### 2. Orchestration Patterns

**Sequential**: A → B → C
- Use when: Tasks depend on each other
- Example: Diagnose → Analyze → Fix

**Parallel**: A + B + C (simultaneously)
- Use when: Tasks are independent
- Example: Security audit + Performance check

**Hybrid**: (A + B) → C → (D + E)
- Use when: Mix of dependencies
- Example: Initial assessment → Analysis → Multiple fixes

### 3. Confidence Scoring
Agents return confidence (0-1):
- **0.9+**: Auto-remediate
- **0.7-0.9**: Human review recommended
- **<0.7**: Human intervention required

## Performance Tips

1. **Use parallel execution** for independent tasks (2-3x speedup)
2. **Cache agent responses** for repeated queries
3. **Set appropriate timeouts** (default: 300s)
4. **Limit concurrent agents** (default: 4) based on resources
5. **Use streaming** for real-time updates (advanced)

## Resource Usage

**Typical API usage per example**:
- Example 1: ~2-3 API calls
- Example 2: ~5-7 API calls
- Example 3: ~6-8 API calls
- Example 4: ~10-12 API calls
- Example 5: ~8-10 API calls

**Total for full demo**: ~30-40 API calls (~$0.50-1.00 USD)

## Get Help

- **Documentation**: See README.md for detailed architecture
- **Examples**: All 5 examples include inline comments
- **Issues**: Check common issues section above
- **Community**: Network automation forums, LangChain Discord

## What's Next?

After mastering multi-agent orchestration:

1. **Chapter 48**: Monitoring and observability
2. **Chapter 51**: Scaling and production deployment
3. **Chapter 61**: Real-world case studies

Happy orchestrating!
