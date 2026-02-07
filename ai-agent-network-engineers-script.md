# How to Build an AI Agent for Network Engineers
## 60-Minute Video Script (Split for TikTok/Short-Form Platforms)

---

# FULL SCRIPT OVERVIEW

**Total Runtime:** 60 minutes  
**Format:** 12 segments × ~5 minutes each  
**Target Audience:** Network engineers, IT professionals, DevOps engineers  
**Tone:** Educational, practical, conversational

---

# SEGMENT 1: THE HOOK (5 min)
**TikTok Title:** "Network Engineers, AI is About to Change EVERYTHING 🔥"

### Script:

**[0:00-0:30 - Pattern Interrupt]**

"What if I told you that you could automate 80% of your daily network tasks... without writing a single line of traditional code?

I'm talking about AI agents. Not chatbots. Not simple scripts. Actual intelligent agents that can troubleshoot your network, configure devices, and even predict failures before they happen.

And the best part? You don't need to be a data scientist to build one."

**[0:30-2:00 - The Problem]**

"Let's be real for a second. As network engineers, we're drowning in:
- Alert fatigue from monitoring tools
- Repetitive configuration changes
- Endless troubleshooting sessions at 3 AM
- Documentation that's never up to date
- Ticket queues that never end

We spend more time on mundane tasks than actual engineering. Sound familiar?"

**[2:00-4:00 - The Promise]**

"Here's what an AI agent can do for you:

1. **Intelligent Troubleshooting** - Feed it an alert, it investigates across multiple devices, correlates data, and gives you the root cause

2. **Automated Configuration** - Describe what you want in plain English, it generates and validates the config

3. **Proactive Monitoring** - It learns your network's normal behavior and flags anomalies before they become outages

4. **Documentation on Autopilot** - Every change, every incident, automatically documented

This isn't science fiction. This is what we're building today."

**[4:00-5:00 - Transition]**

"In this series, I'm going to show you exactly how to build your own AI agent for network engineering. Step by step. No fluff. Just practical, hands-on guidance.

Let's dive in."

---

# SEGMENT 2: UNDERSTANDING AI AGENTS (5 min)
**TikTok Title:** "AI Agents Explained in 5 Minutes (Network Engineer Edition)"

### Script:

**[0:00-1:30 - What is an AI Agent?]**

"Before we build anything, let's understand what we're actually building.

An AI agent is NOT just ChatGPT with a fancy wrapper. Here's the key difference:

- A **chatbot** answers questions
- An **AI agent** takes actions

Think of it like this: ChatGPT is like a really smart consultant who gives advice. An AI agent is like hiring an assistant who actually does the work.

An agent has three core components:
1. **Brain** - The LLM (Large Language Model) that thinks and reasons
2. **Memory** - Context and history it can reference
3. **Tools** - Actual capabilities to interact with the real world"

**[1:30-3:00 - The Agent Loop]**

"Here's how an AI agent actually works - it's called the Agent Loop:

1. **Observe** - Receive input (an alert, a request, a question)
2. **Think** - Reason about what needs to be done
3. **Act** - Use tools to take action
4. **Observe** - See the results
5. **Repeat** - Continue until the task is complete

This is fundamentally different from traditional automation which is:
IF this THEN that.

An agent is:
GIVEN this situation, FIGURE OUT the best approach, EXECUTE it, ADAPT if needed.

That's the magic. It can handle situations it's never seen before."

**[3:00-5:00 - Network Engineering Context]**

"For network engineers, this means:

Your agent can SSH into a device, run show commands, analyze the output, decide what to check next, run more commands, and arrive at a diagnosis - just like you would.

But faster. And it doesn't get tired. And it can check 50 devices simultaneously.

The tools we'll give our agent:
- SSH/NETCONF/RESTCONF access to devices
- API calls to monitoring systems
- Access to documentation and runbooks
- Ability to create tickets and send notifications

By the end of this series, you'll have an agent that can do all of this."

---

# SEGMENT 3: ARCHITECTURE OVERVIEW (5 min)
**TikTok Title:** "The Blueprint: AI Agent Architecture for Networks 🏗️"

### Script:

**[0:00-2:00 - High-Level Architecture]**

"Let me show you the architecture we're going to build.

*[Show diagram]*

At the center is your AI Agent - we'll use an LLM like Claude or GPT-4 as the brain.

Connected to it:

**Input Sources:**
- Monitoring alerts (Prometheus, Datadog, PRTG)
- Chat interfaces (Slack, Teams)
- Tickets (ServiceNow, Jira)

**Tools/Capabilities:**
- Network device access (Netmiko, NAPALM, Nornir)
- API integrations (REST, GraphQL)
- Database queries
- File operations

**Memory/Knowledge:**
- Vector database for documentation
- Conversation history
- Network inventory and topology

**Output:**
- Actions on devices
- Responses to users
- Ticket updates
- Documentation"

**[2:00-3:30 - Technology Choices]**

"Here's my recommended stack - and I'll explain why:

**LLM:** Claude 3.5 Sonnet or GPT-4
- Why: Best balance of reasoning ability and tool use
- Alternative: Open source models like Llama 3 for air-gapped environments

**Agent Framework:** LangChain or CrewAI
- Why: Handles the agent loop, memory, and tool integration
- Alternative: Build from scratch (we'll cover both)

**Network Automation:** Nornir + NAPALM
- Why: Python-native, flexible, multi-vendor
- Already familiar to most network engineers

**Vector Database:** ChromaDB or Pinecone
- Why: Store and search documentation semantically

**Interface:** Slack bot or Web UI
- Start simple, iterate from there"

**[3:30-5:00 - Security Considerations]**

"Now, I know what you're thinking: 'You want me to give an AI access to my production network?!'

Valid concern. Here's how we handle security:

1. **Read-only first** - Start with show commands only
2. **Approval workflows** - Agent proposes, human approves
3. **Sandboxing** - Test in lab before production
4. **Audit logging** - Every action is logged
5. **Scope limiting** - Restrict which devices and commands

We'll build all of this in. Safety first."

---

# SEGMENT 4: SETTING UP YOUR ENVIRONMENT (5 min)
**TikTok Title:** "Set Up Your AI Agent Dev Environment in 5 Minutes ⚡"

### Script:

**[0:00-1:00 - Prerequisites]**

"Alright, let's get hands-on. Here's what you need:

- Python 3.10 or higher
- An API key for your LLM (OpenAI or Anthropic)
- A network lab (EVE-NG, GNS3, or even just one test device)
- Basic Python knowledge (if you can write a for loop, you're good)

Optional but helpful:
- Docker for containerization
- Git for version control"

**[1:00-3:00 - Installation]**

"Let's set up our environment. I'll go through this step by step.

```bash
# Create a virtual environment
python -m venv ai-network-agent
source ai-network-agent/bin/activate

# Install core packages
pip install langchain langchain-anthropic
pip install netmiko napalm nornir
pip install chromadb
pip install python-dotenv

# Create our project structure
mkdir -p network_agent/{tools,prompts,memory}
touch network_agent/__init__.py
```

Now create a `.env` file for your API keys:

```
ANTHROPIC_API_KEY=your-key-here
# Never commit this file!
```"

**[3:00-5:00 - First Test]**

"Let's verify everything works with a simple test:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model='claude-sonnet-4-20250514')
response = llm.invoke('What is BGP in one sentence?')
print(response.content)
```

If you see a response about Border Gateway Protocol, you're ready to go!

In the next segment, we'll build our first tool - the ability to SSH into a device and run commands."

---

# SEGMENT 5: BUILDING YOUR FIRST TOOL (5 min)
**TikTok Title:** "Give Your AI Agent SSH Superpowers 🔐"

### Script:

**[0:00-1:30 - Tool Concept]**

"Tools are how your agent interacts with the real world. Think of them as functions the AI can call.

A tool has:
- A **name** the agent uses to call it
- A **description** so the agent knows when to use it
- **Parameters** it accepts
- **Logic** that executes

Let's build a tool that lets our agent SSH into a network device and run show commands."

**[1:30-4:00 - The Code]**

"Here's our SSH tool:

```python
from langchain.tools import tool
from netmiko import ConnectHandler

@tool
def run_show_command(device_ip: str, command: str) -> str:
    '''
    SSH into a network device and run a show command.
    Use this to gather information from network devices.
    
    Args:
        device_ip: IP address of the device
        command: The show command to run (e.g., 'show interfaces')
    
    Returns:
        The command output from the device
    '''
    device = {
        'device_type': 'cisco_ios',
        'ip': device_ip,
        'username': 'admin',
        'password': 'admin123',  # Use env vars in production!
    }
    
    try:
        with ConnectHandler(**device) as conn:
            output = conn.send_command(command)
            return output
    except Exception as e:
        return f'Error connecting to {device_ip}: {str(e)}'
```

The decorator `@tool` and that docstring are crucial - that's how the agent learns what this tool does and when to use it."

**[4:00-5:00 - Testing]**

"Let's test it standalone first:

```python
result = run_show_command.invoke({
    'device_ip': '192.168.1.1',
    'command': 'show ip interface brief'
})
print(result)
```

You should see your interface table. 

Now we have a building block. In the next segment, we'll connect this tool to our agent so it can decide when to use it."

---

# SEGMENT 6: CREATING THE AGENT (5 min)
**TikTok Title:** "Your First Network AI Agent in 20 Lines of Code 🤖"

### Script:

**[0:00-2:00 - Agent Setup]**

"Now for the exciting part - let's create an agent that can use our tool.

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Our LLM
llm = ChatAnthropic(model='claude-sonnet-4-20250514')

# Our tools
tools = [run_show_command]

# The prompt that defines our agent's personality
prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a network engineering assistant. 
    You help troubleshoot and gather information from network devices.
    Always explain what you're doing and why.
    Be concise but thorough.'''),
    ('human', '{input}'),
    ('placeholder', '{agent_scratchpad}')
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```"

**[2:00-4:00 - Running the Agent]**

"Let's give it a task:

```python
result = executor.invoke({
    'input': 'Check the interface status on device 192.168.1.1'
})
print(result['output'])
```

Watch what happens in verbose mode - you'll see the agent:
1. Understand the request
2. Decide to use the SSH tool
3. Choose the right command
4. Execute it
5. Interpret the results
6. Respond in plain English

This is the agent loop in action!"

**[4:00-5:00 - The Magic Moment]**

"Here's what makes this powerful: try asking it something slightly different:

'Is interface GigabitEthernet0/1 up on 192.168.1.1?'

The agent figures out it needs to:
1. Run 'show ip interface brief'
2. Parse the output
3. Find that specific interface
4. Report the status

You didn't program any of that logic. The agent reasons through it.

That's the difference between automation and intelligence."

---

# SEGMENT 7: ADDING MEMORY (5 min)
**TikTok Title:** "Give Your AI Agent a Memory (It Changes Everything) 🧠"

### Script:

**[0:00-1:30 - Why Memory Matters]**

"Right now, every conversation with our agent starts fresh. It has no memory.

Ask it: 'What device did I just check?'
It has no idea.

For real troubleshooting, you need context:
- What have we tried?
- What did we find?
- What's the history?

Let's add memory."

**[1:30-3:30 - Implementation]**

"LangChain makes this easy:

```python
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# Update our prompt to include history
prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a network engineering assistant...'''),
    ('placeholder', '{chat_history}'),
    ('human', '{input}'),
    ('placeholder', '{agent_scratchpad}')
])

# Update executor
executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True
)
```

Now conversations flow naturally:

You: 'Check interfaces on 192.168.1.1'
Agent: *runs command, reports status*

You: 'What about the routing table?'
Agent: *knows you mean 192.168.1.1, runs show ip route*

You: 'Compare that to 192.168.1.2'
Agent: *understands context, checks the second device*"

**[3:30-5:00 - Long-term Memory]**

"For long-term memory, we'll add a vector database in a later segment. This lets the agent remember:
- Past incidents and solutions
- Your network documentation
- Runbooks and procedures

Imagine: 'How did we fix that BGP issue last month?'
And the agent actually remembers."

---

# SEGMENT 8: MULTI-DEVICE OPERATIONS (5 min)
**TikTok Title:** "Make Your AI Agent Check 100 Devices at Once 🚀"

### Script:

**[0:00-2:00 - Scaling Up]**

"One device is nice. But you probably have hundreds. Let's scale.

We'll use Nornir - it's built for multi-device operations:

```python
from nornir import InitNornir
from nornir_netmiko import netmiko_send_command

@tool  
def check_multiple_devices(command: str, device_filter: str = None) -> str:
    '''
    Run a command across multiple network devices.
    
    Args:
        command: Show command to run
        device_filter: Optional filter (e.g., 'role:spine' or 'site:dc1')
    '''
    nr = InitNornir(config_file='nornir_config.yaml')
    
    if device_filter:
        key, value = device_filter.split(':')
        nr = nr.filter(**{key: value})
    
    results = nr.run(task=netmiko_send_command, command_string=command)
    
    output = []
    for host, result in results.items():
        output.append(f'=== {host} ===\n{result.result}')
    
    return '\n'.join(output)
```"

**[2:00-4:00 - Inventory Setup]**

"Your inventory file (`hosts.yaml`):

```yaml
spine1:
  hostname: 192.168.1.1
  role: spine
  site: dc1

spine2:
  hostname: 192.168.1.2
  role: spine
  site: dc1

leaf1:
  hostname: 192.168.1.10
  role: leaf
  site: dc1
```

Now the agent can:

'Check BGP status on all spine switches'
→ Filters to role:spine, runs show ip bgp summary

'Show interface errors across the DC1 site'
→ Filters to site:dc1, runs show interfaces | include errors"

**[4:00-5:00 - Parallel Execution]**

"The beauty of Nornir: it runs in parallel by default. 

100 devices? Still takes seconds.

Your agent can now say: 'I checked all 47 leaf switches. 3 have elevated error counters: leaf12, leaf28, leaf41.'

That's power."

---

# SEGMENT 9: RAG - KNOWLEDGE BASE (5 min)
**TikTok Title:** "Your AI Agent Can Read Your Documentation 📚"

### Script:

**[0:00-2:00 - The Problem with LLMs]**

"LLMs are smart, but they don't know YOUR network. They don't know:
- Your naming conventions
- Your topology
- Your runbooks
- Your past incidents

RAG (Retrieval Augmented Generation) fixes this. We give the agent access to your documentation."

**[2:00-4:00 - Building the Knowledge Base]**

"Here's how:

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import AnthropicEmbeddings

# Load your docs
loader = DirectoryLoader('./network_docs/', glob='**/*.md')
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    chunks,
    AnthropicEmbeddings(),
    persist_directory='./chroma_db'
)

# Create retriever tool
@tool
def search_documentation(query: str) -> str:
    '''Search network documentation for relevant information.'''
    docs = vectorstore.similarity_search(query, k=3)
    return '\n\n'.join([d.page_content for d in docs])
```

Add your runbooks, network diagrams (as text), incident reports, everything."

**[4:00-5:00 - The Result]**

"Now your agent can answer:

'What's the escalation path for a core switch failure?'
→ Retrieves from your runbook

'What subnet is VLAN 100?'
→ Retrieves from your documentation

'How did we handle the BGP flap in January?'
→ Retrieves from incident reports

Your documentation becomes active, searchable, and useful."

---

# SEGMENT 10: ALERT HANDLING (5 min)
**TikTok Title:** "AI Agent That Handles Alerts For You 🚨"

### Script:

**[0:00-2:00 - Alert Overload]**

"You get an alert: 'Interface down on router-core-01'

What do you do?
1. SSH in
2. Check interface status
3. Check logs
4. Check the other end
5. Check for related alerts
6. Decide if it's a real problem

What if your agent did steps 1-5 automatically?

Let's build an alert handler."

**[2:00-4:00 - The Alert Agent]**

"```python
alert_prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a network alert analyst.
    When you receive an alert:
    1. Investigate the affected device
    2. Gather relevant information
    3. Check related devices if needed
    4. Correlate with recent changes
    5. Provide a summary with severity assessment
    
    Always explain your reasoning.'''),
    ('human', 'ALERT: {alert_text}'),
    ('placeholder', '{agent_scratchpad}')
])

# Tools include: SSH, documentation search, log search
alert_tools = [run_show_command, search_documentation, check_logs]

alert_agent = create_tool_calling_agent(llm, alert_tools, alert_prompt)
alert_executor = AgentExecutor(agent=alert_agent, tools=alert_tools)
```

Webhook from your monitoring:

```python
@app.route('/alert', methods=['POST'])
def handle_alert():
    alert = request.json
    result = alert_executor.invoke({'alert_text': str(alert)})
    # Send result to Slack/Teams
    send_notification(result['output'])
    return 'OK'
```"

**[4:00-5:00 - What This Looks Like]**

"Alert comes in: 'High CPU on spine-01'

Agent automatically:
- SSHs to spine-01
- Checks CPU processes
- Checks BGP neighbor count
- Checks recent config changes
- Searches docs for known issues

Sends you: 'CPU spike caused by BGP reconvergence after leaf-12 reboot 5 minutes ago. 47 routes being recalculated. Expected to stabilize in ~10 minutes. No action needed - monitoring.'

That's the dream. That's what we're building."

---

# SEGMENT 11: SLACK INTEGRATION (5 min)
**TikTok Title:** "Chat With Your Network on Slack 💬"

### Script:

**[0:00-2:00 - Why Slack?]**

"Command line is great for development. But for daily use, you want to chat with your agent where you already work.

Slack, Teams, whatever your team uses.

Let's integrate with Slack."

**[2:00-4:00 - Implementation]**

"Using Slack Bolt framework:

```python
from slack_bolt import App

app = App(token=os.environ['SLACK_BOT_TOKEN'])

@app.event('app_mention')
def handle_mention(event, say):
    user_message = event['text']
    thread_ts = event.get('thread_ts', event['ts'])
    
    # Run through our agent
    result = executor.invoke({'input': user_message})
    
    # Reply in thread
    say(text=result['output'], thread_ts=thread_ts)

@app.event('message')
def handle_dm(event, say):
    if event.get('channel_type') == 'im':
        result = executor.invoke({'input': event['text']})
        say(text=result['output'])

if __name__ == '__main__':
    app.start(port=3000)
```"

**[4:00-5:00 - Usage]**

"Now in Slack:

@NetworkAgent check BGP on all spine switches

'Checking BGP status across 4 spine switches...

✅ spine-01: 12 neighbors, all established
✅ spine-02: 12 neighbors, all established  
⚠️ spine-03: 11 neighbors, leaf-07 is down
✅ spine-04: 12 neighbors, all established

Recommendation: Investigate connectivity to leaf-07'

Your whole team can interact with it. Knowledge sharing happens automatically. And you have a record of every troubleshooting session."

---

# SEGMENT 12: PRODUCTION & NEXT STEPS (5 min)
**TikTok Title:** "Ship It! Taking Your AI Agent to Production 🚢"

### Script:

**[0:00-2:00 - Production Checklist]**

"You've built something powerful. Before production:

**Security:**
- ✅ Credentials in environment variables/vault
- ✅ Read-only mode initially
- ✅ Approval workflow for write operations
- ✅ Audit logging for every action
- ✅ Network segmentation for the agent

**Reliability:**
- ✅ Error handling and retries
- ✅ Timeout limits
- ✅ Rate limiting to prevent runaway
- ✅ Health checks and monitoring

**Observability:**
- ✅ Log all agent decisions
- ✅ Track token usage and costs
- ✅ Monitor response times
- ✅ Feedback loop for improvement"

**[2:00-4:00 - Deployment]**

"Simple Docker deployment:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ['python', 'main.py']
```

Run it:
```bash
docker build -t network-agent .
docker run -d --env-file .env network-agent
```

For high availability, use Kubernetes or your platform of choice."

**[4:00-5:00 - What's Next]**

"This is just the beginning. Where to go from here:

1. **More tools** - Config generation, change validation, capacity planning

2. **Multi-agent systems** - Specialized agents that collaborate

3. **Learning from feedback** - Agent improves over time

4. **Predictive capabilities** - Anticipate failures before they happen

The future of network engineering isn't about replacing engineers. It's about augmenting them with AI that handles the routine so you can focus on the interesting problems.

Start small. Build incrementally. Ship something this week.

And if you want to go deeper, check out the full resources at vexpertai.com.

Now go build something amazing. 🚀"

---

# APPENDIX: SHORT-FORM CUTS

## TikTok/Reels Optimization

Each segment is ~5 minutes. For 60-second cuts, use these hooks:

### Segment 1 (60s cut):
- 0:00-0:05: "Network engineers, AI agents are about to change everything"
- 0:05-0:30: Problem list (alert fatigue, config changes, 3 AM)
- 0:30-0:55: "Here's what an AI agent can do" (top 3 capabilities)
- 0:55-1:00: "Link in bio for the full tutorial"

### Segment 6 (60s cut):
- 0:00-0:05: "20 lines of Python to create an AI network agent"
- 0:05-0:40: Show the code scrolling with voiceover
- 0:40-0:55: Demo the agent running a command
- 0:55-1:00: "Full tutorial on [platform]"

### Segment 10 (60s cut):
- 0:00-0:05: "What if your AI handled alerts for you?"
- 0:05-0:20: Show alert coming in
- 0:20-0:50: Show agent investigating automatically
- 0:50-1:00: Show final summary notification

## Hashtags
#NetworkEngineering #AIAgent #NetDevOps #Automation #Python #LangChain #NetworkAutomation #DevOps #TechTok #LearnOnTikTok #CodingTok #AI #MachineLearning

## Posting Schedule Recommendation
- **Day 1:** Segment 1 (The Hook) - Full + 60s cut
- **Day 2:** Segment 6 (First Agent) - 60s cut teaser
- **Day 3:** Segment 2 (Understanding Agents) - Full
- **Day 4:** Segment 10 (Alert Handling) - 60s cut
- **Day 5:** Segment 5 (First Tool) - Full
- Continue alternating full segments with 60s cuts of upcoming content
