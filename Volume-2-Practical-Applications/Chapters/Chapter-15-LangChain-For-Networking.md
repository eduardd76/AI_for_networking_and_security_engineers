# Chapter 15: LangChain for Networking

## Why This Chapter Matters

You've learned the fundamentals. Now let's build real systems faster.

LangChain is a framework for building LLM applications. Instead of writing hundreds of lines of boilerplate for prompts, chains, and memory, you use pre-built components.

**The networking equivalent**: LangChain is like Ansible vs raw SSH commands. You *could* SSH to 100 routers and run commands manually. Or you could use Ansible playbooks.

**What you'll build**:
- Config analyzer with conversation memory
- Multi-step troubleshooting chain
- Network documentation Q&A system
- Automated config generation with validation

No more reinventing the wheel. Let's use LangChain to build production systems fast.

---

## Section 1: LangChain Fundamentals

### What Problem Does LangChain Solve?

**Without LangChain**:
```python
# You write this every time
def analyze_config(config):
    # Build prompt
    prompt = f"Analyze this config: {config}"

    # Call API
    response = client.messages.create(...)

    # Parse response
    result = parse_json(response)

    # Store conversation
    conversation_history.append(...)

    # Handle errors
    if error:
        retry(...)

    # Return result
    return result

# 50+ lines of boilerplate, repeated everywhere
```

**With LangChain**:
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a network config analyzer."),
    ("human", "Analyze this config: {config}")
])

chain = prompt | llm
result = chain.invoke({"config": config})

# 7 lines. Done.
```

### Core LangChain Concepts

**1. LLMs**: The AI model interface
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",
    max_tokens=1000,
    temperature=0.0
)

response = llm.invoke("What is BGP?")
print(response.content)
```

**2. Prompts**: Reusable templates
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Analyze this {device_type} config for {issue_type} issues:\n\n{config}"
)

prompt = template.invoke({
    "device_type": "Cisco IOS",
    "issue_type": "security",
    "config": "..."
})
```

**3. Chains**: Multi-step workflows
```python
from langchain_core.runnables import RunnableSequence

# Chain: prompt → LLM → parser
chain = prompt | llm | parser
result = chain.invoke({"config": "..."})
```

**4. Memory**: Conversation context
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "What's wrong with this BGP config?"},
    {"output": "Missing BGP authentication"}
)
```

### Installation

```bash
pip install langchain langchain-anthropic langchain-community
pip install langchain-core chromadb  # For RAG (next chapter)
```

---

## Section 2: Building a Config Analyzer with LangChain

### Simple Config Analyzer

```python
# config_analyzer_lc.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Define output structure
class SecurityIssue(BaseModel):
    severity: str = Field(description="low, medium, high, or critical")
    issue: str = Field(description="Description of the security issue")
    location: str = Field(description="Where in the config this appears")
    recommendation: str = Field(description="How to fix it")

class ConfigAnalysis(BaseModel):
    summary: str = Field(description="Overall security assessment")
    issues: List[SecurityIssue] = Field(description="List of security issues found")
    compliant: bool = Field(description="Whether config meets security standards")

class NetworkConfigAnalyzer:
    """LangChain-powered config analyzer."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        # JSON output parser
        self.parser = JsonOutputParser(pydantic_object=ConfigAnalysis)

        # Prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network security expert.
Analyze configurations for security issues.
Return your analysis in JSON format matching this schema:
{format_instructions}"""),
            ("human", "Analyze this configuration:\n\n{config}")
        ])

        # Build the chain: prompt → LLM → parser
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, config: str) -> ConfigAnalysis:
        """Analyze a network configuration."""
        result = self.chain.invoke({
            "config": config,
            "format_instructions": self.parser.get_format_instructions()
        })

        return ConfigAnalysis(**result)


# Example usage
if __name__ == "__main__":
    analyzer = NetworkConfigAnalyzer(api_key="your-api-key")

    config = """
hostname router-01
!
enable secret weak123
!
interface GigabitEthernet0/0
 ip address 10.1.1.1 255.255.255.0
 no shutdown
!
snmp-server community public RW
!
line vty 0 4
 transport input telnet
    """

    analysis = analyzer.analyze(config)

    print(f"Summary: {analysis.summary}")
    print(f"Compliant: {analysis.compliant}")
    print(f"\nIssues found: {len(analysis.issues)}")

    for issue in analysis.issues:
        print(f"\n[{issue.severity.upper()}] {issue.issue}")
        print(f"Location: {issue.location}")
        print(f"Fix: {issue.recommendation}")
```

**Output**:
```
Summary: Configuration has multiple critical security issues
Compliant: False

Issues found: 4

[CRITICAL] Weak enable password
Location: enable secret weak123
Fix: Use strong password with minimum 12 characters

[CRITICAL] SNMP community with write access
Location: snmp-server community public RW
Fix: Remove RW access, use RO with strong community string

[HIGH] Telnet enabled
Location: line vty 0 4
Fix: Use SSH instead of telnet

[MEDIUM] No ACL on VTY lines
Location: line vty 0 4
Fix: Add access-class with ACL
```

**Why this is better**:
- Structured output automatically parsed
- Reusable chain
- Type-safe with Pydantic
- 50 lines vs 150+ manually

---

## Section 3: Adding Conversation Memory

### Config Analyzer with Chat History

```python
# config_analyzer_with_memory.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ConversationalConfigAnalyzer:
    """Config analyzer that remembers conversation context."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.3
        )

        # Memory to store conversation
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )

        # Prompt with history placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network engineer assistant.
You help analyze configurations and answer follow-up questions.
Remember the context of previous messages in this conversation."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Create conversation chain
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )

    def chat(self, message: str) -> str:
        """Send a message and get response."""
        response = self.chain.invoke({"input": message})
        return response['response']

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


# Example usage
if __name__ == "__main__":
    analyzer = ConversationalConfigAnalyzer(api_key="your-api-key")

    # First message: analyze config
    config = """
interface GigabitEthernet0/0
 ip address 10.1.1.1 255.255.255.0
 ip access-group 100 in
 no shutdown
!
access-list 100 permit ip any any
    """

    response = analyzer.chat(f"Analyze this config:\n\n{config}")
    print("Assistant:", response)
    # Output: "This config has a security issue. ACL 100 permits all traffic..."

    # Follow-up question (uses memory!)
    response = analyzer.chat("What ACL should I use instead?")
    print("\nAssistant:", response)
    # Output: "For the interface you showed, you should use a more restrictive ACL..."

    # Another follow-up
    response = analyzer.chat("Show me an example")
    print("\nAssistant:", response)
    # Output: "Here's an example ACL for GigabitEthernet0/0..."

    # Clear memory for new conversation
    analyzer.clear_memory()
```

**Key benefit**: The analyzer remembers context. You can ask follow-up questions without repeating the entire config.

---

## Section 4: Multi-Step Troubleshooting Chain

### Sequential Chain for Troubleshooting

```python
# troubleshooting_chain.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class NetworkTroubleshooter:
    """Multi-step troubleshooting using LangChain chains."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        # Step 1: Identify problem category
        self.categorize_prompt = ChatPromptTemplate.from_template("""
Given this network issue description, categorize it.

Issue: {symptom}

Return ONE of: routing, switching, security, performance, connectivity

Category:""")

        # Step 2: Suggest diagnostic commands
        self.commands_prompt = ChatPromptTemplate.from_template("""
For a {category} issue with this symptom:
{symptom}

List 3-5 diagnostic commands to run (Cisco IOS format).
One command per line, no explanations.
""")

        # Step 3: Analyze command outputs
        self.analyze_prompt = ChatPromptTemplate.from_template("""
Issue category: {category}
Symptom: {symptom}
Commands run: {commands}

Command outputs:
{outputs}

Provide:
1. Root cause analysis
2. Step-by-step fix
3. Prevention recommendations
""")

        self.parser = StrOutputParser()

    def troubleshoot(
        self,
        symptom: str,
        command_outputs: dict = None
    ) -> dict:
        """
        Multi-step troubleshooting process.

        Args:
            symptom: Description of the problem
            command_outputs: Optional dict of {command: output}

        Returns:
            dict with category, commands, and analysis
        """
        # Step 1: Categorize
        category_chain = self.categorize_prompt | self.llm | self.parser
        category = category_chain.invoke({"symptom": symptom}).strip()

        print(f"Issue category: {category}")

        # Step 2: Get diagnostic commands
        commands_chain = self.commands_prompt | self.llm | self.parser
        commands = commands_chain.invoke({
            "category": category,
            "symptom": symptom
        })

        print(f"\nDiagnostic commands:\n{commands}")

        # Step 3: Analyze (if outputs provided)
        analysis = None
        if command_outputs:
            outputs_text = "\n\n".join([
                f"Command: {cmd}\n{output}"
                for cmd, output in command_outputs.items()
            ])

            analyze_chain = self.analyze_prompt | self.llm | self.parser
            analysis = analyze_chain.invoke({
                "category": category,
                "symptom": symptom,
                "commands": commands,
                "outputs": outputs_text
            })

            print(f"\nAnalysis:\n{analysis}")

        return {
            "category": category,
            "commands": commands,
            "analysis": analysis
        }


# Example usage
if __name__ == "__main__":
    troubleshooter = NetworkTroubleshooter(api_key="your-api-key")

    # Describe the problem
    symptom = "Users on VLAN 10 cannot reach the internet, but VLAN 20 works fine"

    # Get diagnostic commands
    result = troubleshooter.troubleshoot(symptom)

    # Simulate running commands and getting output
    command_outputs = {
        "show ip route": """
Gateway of last resort is 0.0.0.0 to network 0.0.0.0
      10.0.0.0/8 is variably subnetted, 2 subnets
C        10.10.0.0/24 is directly connected, Vlan10
C        10.20.0.0/24 is directly connected, Vlan20
S*    0.0.0.0/0 [1/0] via 192.168.1.1
        """,
        "show ip interface brief": """
Interface              IP-Address      OK? Method Status                Protocol
Vlan10                 10.10.0.1       YES manual up                    down
Vlan20                 10.20.0.1       YES manual up                    up
        """,
        "show vlan": """
VLAN Name                             Status    Ports
---- -------------------------------- --------- -------------------------------
10   VLAN0010                         active    Gi0/1, Gi0/2
20   VLAN0020                         active    Gi0/3, Gi0/4
        """
    }

    # Analyze with command outputs
    result = troubleshooter.troubleshoot(symptom, command_outputs)
```

**Output**:
```
Issue category: connectivity

Diagnostic commands:
show ip interface brief
show ip route
show vlan
show interface vlan 10
show ip arp vlan 10

Analysis:
Root cause: Interface VLAN 10 is administratively down
The "show ip interface brief" output shows Vlan10 status as "up"
but protocol as "down", indicating Layer 2 issue.

Fix:
1. Enter interface configuration: interface vlan 10
2. Ensure interface is up: no shutdown
3. Verify SVI has correct IP: show ip interface vlan 10
4. Check trunk ports carry VLAN 10: show interfaces trunk
5. Verify routing: show ip route | include 10.10.0.0

Prevention:
- Document all VLANs and their expected states
- Implement config management to prevent accidental shutdowns
- Add monitoring alerts for interface status changes
```

---

## Section 5: Automated Config Generation

### Template-Based Config Generator

```python
# config_generator_lc.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List

class NetworkConfigGenerator:
    """Generate network configs using LangChain."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.parser = StrOutputParser()

    def generate_interface_config(
        self,
        device_type: str,
        interface_name: str,
        ip_address: str,
        subnet_mask: str,
        description: str = "",
        vlan: int = None
    ) -> str:
        """Generate interface configuration."""
        prompt = ChatPromptTemplate.from_template("""
Generate a {device_type} interface configuration.

Requirements:
- Interface: {interface_name}
- IP Address: {ip_address}
- Subnet Mask: {subnet_mask}
- Description: {description}
- VLAN: {vlan}

Include:
1. Interface configuration
2. IP address
3. Description
4. Enable interface (no shutdown)
5. Basic security (if applicable)

Return ONLY the configuration commands, no explanations.
""")

        chain = prompt | self.llm | self.parser

        config = chain.invoke({
            "device_type": device_type,
            "interface_name": interface_name,
            "ip_address": ip_address,
            "subnet_mask": subnet_mask,
            "description": description or "Managed interface",
            "vlan": vlan or "N/A"
        })

        return config.strip()

    def generate_vlan_config(
        self,
        device_type: str,
        vlan_id: int,
        vlan_name: str,
        interfaces: List[str]
    ) -> str:
        """Generate VLAN configuration."""
        prompt = ChatPromptTemplate.from_template("""
Generate {device_type} VLAN configuration.

Requirements:
- VLAN ID: {vlan_id}
- VLAN Name: {vlan_name}
- Assign to interfaces: {interfaces}

Include:
1. VLAN creation
2. VLAN naming
3. Interface assignments
4. Switchport mode access

Return ONLY configuration commands.
""")

        chain = prompt | self.llm | self.parser

        config = chain.invoke({
            "device_type": device_type,
            "vlan_id": vlan_id,
            "vlan_name": vlan_name,
            "interfaces": ", ".join(interfaces)
        })

        return config.strip()

    def generate_acl(
        self,
        device_type: str,
        acl_name: str,
        purpose: str,
        rules: List[Dict[str, str]]
    ) -> str:
        """Generate ACL configuration."""
        rules_text = "\n".join([
            f"- {rule['action']} {rule['protocol']} from {rule['source']} to {rule['destination']}"
            for rule in rules
        ])

        prompt = ChatPromptTemplate.from_template("""
Generate {device_type} ACL configuration.

ACL Name/Number: {acl_name}
Purpose: {purpose}

Rules:
{rules}

Include:
1. ACL definition
2. All rules in order
3. Implicit deny at end (if not already included)

Return ONLY configuration commands.
""")

        chain = prompt | self.llm | self.parser

        config = chain.invoke({
            "device_type": device_type,
            "acl_name": acl_name,
            "purpose": purpose,
            "rules": rules_text
        })

        return config.strip()


# Example usage
if __name__ == "__main__":
    generator = NetworkConfigGenerator(api_key="your-api-key")

    # Generate interface config
    interface_config = generator.generate_interface_config(
        device_type="Cisco IOS",
        interface_name="GigabitEthernet0/1",
        ip_address="192.168.10.1",
        subnet_mask="255.255.255.0",
        description="Management Interface"
    )

    print("=== Interface Config ===")
    print(interface_config)

    # Generate VLAN config
    vlan_config = generator.generate_vlan_config(
        device_type="Cisco IOS",
        vlan_id=100,
        vlan_name="SERVERS",
        interfaces=["GigabitEthernet0/10", "GigabitEthernet0/11"]
    )

    print("\n=== VLAN Config ===")
    print(vlan_config)

    # Generate ACL
    acl_config = generator.generate_acl(
        device_type="Cisco IOS",
        acl_name="100",
        purpose="Restrict access to management subnet",
        rules=[
            {
                "action": "permit",
                "protocol": "tcp",
                "source": "10.0.0.0/24",
                "destination": "192.168.10.0/24 port 22"
            },
            {
                "action": "permit",
                "protocol": "icmp",
                "source": "10.0.0.0/24",
                "destination": "192.168.10.0/24"
            },
            {
                "action": "deny",
                "protocol": "ip",
                "source": "any",
                "destination": "192.168.10.0/24"
            }
        ]
    )

    print("\n=== ACL Config ===")
    print(acl_config)
```

---

## Section 6: Config Validation Chain

### Validate Before Applying

```python
# config_validator_lc.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class ValidationIssue(BaseModel):
    severity: str = Field(description="critical, high, medium, low, or info")
    issue: str = Field(description="What's wrong")
    line: str = Field(description="Problematic configuration line")
    recommendation: str = Field(description="How to fix")

class ValidationResult(BaseModel):
    valid: bool = Field(description="Whether config is safe to apply")
    issues: List[ValidationIssue] = Field(description="List of issues found")
    summary: str = Field(description="Overall assessment")

class ConfigValidator:
    """Validate configs before applying them."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.parser = JsonOutputParser(pydantic_object=ValidationResult)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a network config validator.
Analyze configurations for syntax errors, security issues, and best practices.
Mark config as valid=false if there are any critical or high severity issues.

Return JSON matching this schema:
{format_instructions}"""),
            ("human", """Validate this {device_type} configuration:

{config}

Check for:
1. Syntax errors
2. Security issues
3. Best practice violations
4. Potential outages
""")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def validate(self, config: str, device_type: str = "Cisco IOS") -> ValidationResult:
        """Validate a configuration."""
        result = self.chain.invoke({
            "config": config,
            "device_type": device_type,
            "format_instructions": self.parser.get_format_instructions()
        })

        return ValidationResult(**result)


# Example usage with approval workflow
if __name__ == "__main__":
    validator = ConfigValidator(api_key="your-api-key")

    new_config = """
interface GigabitEthernet0/1
 ip address 192.168.1.1 255.255.255.0
 no shutdown

interface GigabitEthernet0/2
 ip address 192.168.1.2 255.255.255.0
 no shutdown
    """

    result = validator.validate(new_config)

    print(f"Valid: {result.valid}")
    print(f"Summary: {result.summary}\n")

    if not result.valid:
        print("⚠️  ISSUES FOUND - DO NOT APPLY")
        for issue in result.issues:
            print(f"\n[{issue.severity.upper()}] {issue.issue}")
            print(f"Line: {issue.line}")
            print(f"Fix: {issue.recommendation}")
    else:
        print("✅ Configuration is safe to apply")
```

---

## Section 7: Custom LangChain Tools

### Building Network-Specific Tools

```python
# network_tools.py
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Type
from pydantic import BaseModel, Field
import subprocess

# Tool input schemas
class PingInput(BaseModel):
    host: str = Field(description="IP address or hostname to ping")
    count: int = Field(default=4, description="Number of ping packets")

class ShowCommandInput(BaseModel):
    device: str = Field(description="Device hostname or IP")
    command: str = Field(description="Show command to run")

# Define tools
class PingTool(BaseTool):
    name: str = "ping"
    description: str = "Ping a host to check connectivity. Use this to verify network reachability."
    args_schema: Type[BaseModel] = PingInput

    def _run(self, host: str, count: int = 4) -> str:
        """Ping a host."""
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), host],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception as e:
            return f"Ping failed: {str(e)}"

class ShowCommandTool(BaseTool):
    name: str = "show_command"
    description: str = "Run a read-only show command on a network device. Only 'show' commands allowed."
    args_schema: Type[BaseModel] = ShowCommandInput

    def _run(self, device: str, command: str) -> str:
        """Run a show command (simulated)."""
        # In production, use Netmiko here
        if not command.startswith("show"):
            return "Error: Only 'show' commands are allowed"

        # Simulated output
        return f"[Simulated output from {device}]\nCommand: {command}\n..."

# Create agent with tools
def create_network_agent(api_key: str):
    """Create a network troubleshooting agent with tools."""
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0.0
    )

    tools = [PingTool(), ShowCommandTool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a network troubleshooting assistant.
You have access to tools to diagnose network issues.

Use the ping tool to check connectivity.
Use the show_command tool to gather device information.

Always explain what you're doing and why."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

# Example usage
if __name__ == "__main__":
    agent = create_network_agent(api_key="your-api-key")

    result = agent.invoke({
        "input": "Check if 192.168.1.1 is reachable and get its interface status"
    })

    print(result['output'])
```

---

## What Can Go Wrong

**1. Over-abstraction**
- You add LangChain when a simple API call would work
- More dependencies = more failure points
- Use LangChain when complexity justifies it

**2. Version incompatibility**
- LangChain updates frequently
- Breaking changes between versions
- Pin versions: `langchain==0.1.0`

**3. Memory leaks**
- ConversationBufferMemory grows unbounded
- Long conversations exceed context limits
- Use ConversationBufferWindowMemory (keeps last N messages)

**4. Cost from verbose chains**
- Complex chains make multiple API calls
- Each step costs money
- Monitor token usage

**5. Error handling gaps**
- LangChain abstracts errors
- Harder to debug API failures
- Add try/except around chain invocations

---

## Lab: Build a Complete Network Assistant

Combine everything into a production system:

```python
# network_assistant.py - Complete LangChain system
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.output_parsers import JsonOutputParser
from config_validator_lc import ConfigValidator
from config_generator_lc import NetworkConfigGenerator

class NetworkAssistant:
    """Complete network engineering assistant."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.3
        )

        # Memory keeps last 10 messages
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="history"
        )

        # Main conversation chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert network engineer assistant.
You can:
- Analyze configurations
- Generate new configs
- Troubleshoot issues
- Answer networking questions

Always provide clear explanations and working configurations."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt
        )

        # Specialized components
        self.validator = ConfigValidator(api_key)
        self.generator = NetworkConfigGenerator(api_key)

    def chat(self, message: str) -> str:
        """General conversation."""
        response = self.chain.invoke({"input": message})
        return response['response']

    def generate_and_validate_config(self, **kwargs) -> dict:
        """Generate config and validate it."""
        # Generate
        config = self.generator.generate_interface_config(**kwargs)

        # Validate
        validation = self.validator.validate(config)

        return {
            "config": config,
            "valid": validation.valid,
            "issues": validation.issues,
            "summary": validation.summary
        }

# Example
if __name__ == "__main__":
    assistant = NetworkAssistant(api_key="your-api-key")

    # Chat
    print(assistant.chat("What's the difference between OSPF and BGP?"))

    # Generate and validate
    result = assistant.generate_and_validate_config(
        device_type="Cisco IOS",
        interface_name="GigabitEthernet0/1",
        ip_address="10.1.1.1",
        subnet_mask="255.255.255.0"
    )

    if result['valid']:
        print(f"✅ Valid config:\n{result['config']}")
    else:
        print(f"⚠️  Issues: {result['summary']}")
```

---

## Key Takeaways

1. **LangChain accelerates development** - Less boilerplate, more features
2. **Chains compose complex workflows** - Multi-step processes made simple
3. **Memory enables conversations** - Context across multiple interactions
4. **Output parsers ensure structure** - Type-safe responses with Pydantic
5. **Tools extend capabilities** - Integrate with existing network tools
6. **Validation before execution** - AI suggests, humans approve

LangChain is your framework for building production AI networking systems fast.

Next chapter: RAG (Retrieval Augmented Generation) for network documentation.
