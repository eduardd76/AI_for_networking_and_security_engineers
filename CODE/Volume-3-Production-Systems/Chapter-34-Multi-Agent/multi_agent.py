#!/usr/bin/env python3
"""
Multi-Agent Orchestration for Network Operations

Production-ready multi-agent system with specialist agents coordinated
by a supervisor. Demonstrates parallel and sequential execution patterns
for complex network operations.

From: AI for Networking Engineers - Volume 3, Chapter 34
Author: Eduard Dulharu

Usage:
    python multi_agent.py

Architecture:
    - Supervisor Agent: Orchestrates and delegates tasks
    - Diagnosis Agent: Network troubleshooting specialist
    - Config Agent: Configuration generation specialist
    - Security Agent: Security analysis specialist
    - Performance Agent: Performance optimization specialist
"""

import os
import time
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ============================================================================
# Data Models
# ============================================================================

class AgentResponse(BaseModel):
    """Structured response from an agent."""
    agent_name: str = Field(description="Name of the agent")
    task: str = Field(description="Task that was performed")
    status: Literal["success", "failure", "partial"] = Field(description="Task status")
    findings: List[str] = Field(description="Key findings or results")
    recommendations: List[str] = Field(description="Recommendations or actions")
    confidence: float = Field(description="Confidence score (0-1)")
    execution_time: float = Field(description="Time taken in seconds")
    raw_output: str = Field(description="Complete agent output")


class DiagnosisResult(BaseModel):
    """Diagnosis agent result."""
    problem: str = Field(description="Problem description")
    root_cause: str = Field(description="Identified root cause")
    affected_components: List[str] = Field(description="Affected network components")
    severity: Literal["critical", "high", "medium", "low"] = Field(description="Issue severity")
    resolution_steps: List[str] = Field(description="Steps to resolve")
    estimated_downtime: str = Field(description="Estimated resolution time")


class ConfigResult(BaseModel):
    """Configuration agent result."""
    device_type: str = Field(description="Device type (router, switch, firewall)")
    hostname: str = Field(description="Device hostname")
    config_sections: Dict[str, str] = Field(description="Configuration sections")
    validation_status: str = Field(description="Validation result")
    security_compliance: bool = Field(description="Meets security baseline")


class SecurityResult(BaseModel):
    """Security agent result."""
    vulnerabilities: List[str] = Field(description="Identified vulnerabilities")
    compliance_issues: List[str] = Field(description="Compliance violations")
    risk_score: int = Field(description="Risk score (0-100)")
    remediation_priority: List[str] = Field(description="Prioritized fixes")
    security_recommendations: List[str] = Field(description="Security improvements")


class PerformanceResult(BaseModel):
    """Performance agent result."""
    bottlenecks: List[str] = Field(description="Performance bottlenecks")
    resource_utilization: Dict[str, float] = Field(description="Resource usage metrics")
    optimization_opportunities: List[str] = Field(description="Optimization suggestions")
    capacity_forecast: str = Field(description="Capacity planning forecast")
    performance_score: int = Field(description="Performance score (0-100)")


@dataclass
class AgentTask:
    """Task to be executed by an agent."""
    agent_name: str
    task_type: str
    description: str
    priority: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class MultiAgentState:
    """State for multi-agent system."""
    original_request: str
    agent_tasks: List[AgentTask] = field(default_factory=list)
    agent_results: Dict[str, AgentResponse] = field(default_factory=dict)
    supervisor_decision: str = ""
    execution_plan: List[str] = field(default_factory=list)
    final_response: str = ""
    execution_mode: Literal["sequential", "parallel", "hybrid"] = "hybrid"
    start_time: float = 0.0
    end_time: float = 0.0


# ============================================================================
# Network Diagnostic Tools (Simulated)
# ============================================================================

@tool
def get_interface_statistics(interface: str) -> str:
    """
    Get interface statistics and counters.

    Args:
        interface: Interface name (e.g., GigabitEthernet0/0)

    Returns:
        Interface statistics
    """
    mock_stats = {
        "GigabitEthernet0/0": """
Interface: GigabitEthernet0/0
Status: up/down (Layer 1 up, Layer 2 down)
Speed: 1000 Mbps, Full-duplex
Input: 0 packets/sec, 0 bits/sec
Output: 0 packets/sec, 0 bits/sec
Input errors: 0 CRC, 245 frame, 0 overrun
Output errors: 89 collisions, 12 late collisions
Last clearing: Never
        """,
        "GigabitEthernet0/1": """
Interface: GigabitEthernet0/1
Status: up/up
Speed: 1000 Mbps, Full-duplex
Input: 1523 packets/sec, 12456789 bits/sec
Output: 2341 packets/sec, 18765432 bits/sec
Input errors: 0 CRC, 0 frame, 0 overrun
Output errors: 0 collisions, 0 late collisions
Utilization: 78% average over 5 minutes
        """,
        "TenGigabitEthernet0/0": """
Interface: TenGigabitEthernet0/0
Status: up/up
Speed: 10000 Mbps, Full-duplex
Input: 45678 packets/sec, 3654321987 bits/sec (95% utilization)
Output: 43210 packets/sec, 3456789012 bits/sec (92% utilization)
Input errors: 0 CRC, 0 frame, 0 overrun
Output errors: 0 collisions, 0 late collisions
Warning: High utilization detected
        """
    }
    return mock_stats.get(interface, f"Interface {interface} not found")


@tool
def check_routing_protocol(protocol: str) -> str:
    """
    Check routing protocol status.

    Args:
        protocol: Protocol name (ospf, bgp, eigrp)

    Returns:
        Protocol status
    """
    mock_protocols = {
        "ospf": """
OSPF Process 1 with Router ID 10.0.0.1
Area 0: 5 interfaces, 4 neighbors
Neighbor 10.0.0.2 (GigabitEthernet0/0): FULL/DR
Neighbor 10.0.0.3 (GigabitEthernet0/1): FULL/BDR
Neighbor 10.0.0.4 (GigabitEthernet0/2): FULL/DROTHER
Neighbor 10.0.0.5 (GigabitEthernet0/3): EXSTART (stuck - authentication mismatch)
SPF calculation: 12 times (last: 00:05:23 ago)
LSA count: 234
        """,
        "bgp": """
BGP Router ID 10.0.0.1, AS 65001
Neighbors:
- 10.1.1.1 (AS 65002): Established, 5 prefixes received
- 10.1.1.2 (AS 65002): Established, 8 prefixes received
- 10.2.1.1 (AS 65003): Active (connection refused)
- 10.2.1.2 (AS 65003): Idle (admin down)
Total prefixes: 13 received, 25 advertised
        """
    }
    return mock_protocols.get(protocol.lower(), f"Protocol {protocol} not configured")


@tool
def scan_security_vulnerabilities(target: str) -> str:
    """
    Scan for security vulnerabilities.

    Args:
        target: Device or configuration to scan

    Returns:
        Security scan results
    """
    return """
Security Scan Results:
======================
CRITICAL:
- Telnet enabled on VTY lines (CVE-2023-XXXX)
- Default SNMP community 'public' in use
- No SSH version 2 enforcement

HIGH:
- Weak password encryption (Type 7)
- HTTP server enabled without authentication
- No login banner configured

MEDIUM:
- NTP not authenticated
- Console timeout not configured
- CDP enabled on all interfaces

Compliance Status: FAIL (PCI-DSS, SOC 2)
Risk Score: 78/100 (High Risk)
    """


@tool
def analyze_traffic_patterns(interface: str, duration: int = 300) -> str:
    """
    Analyze traffic patterns on interface.

    Args:
        interface: Interface to analyze
        duration: Analysis duration in seconds

    Returns:
        Traffic analysis results
    """
    return f"""
Traffic Analysis - {interface} (Last {duration}s)
================================================
Top Protocols:
- TCP: 78% (1.2 Gbps)
- UDP: 18% (280 Mbps)
- ICMP: 4% (60 Mbps)

Top Talkers:
1. 10.1.1.50 -> 203.0.113.10 (450 Mbps, HTTP/HTTPS)
2. 10.1.1.75 -> 198.51.100.25 (320 Mbps, Video streaming)
3. 10.1.1.100 -> 192.0.2.50 (180 Mbps, Database replication)

Anomalies Detected:
- Unusual spike in DNS queries from 10.1.1.50 (possible DNS tunneling)
- Port scan detected from 10.1.1.75
- High packet loss rate (5.2%) on this interface

Recommendations:
- Investigate 10.1.1.50 for potential malware
- Implement rate limiting for DNS
- Check physical layer issues causing packet loss
    """


@tool
def get_device_health() -> str:
    """
    Get device health metrics.

    Returns:
        Device health status
    """
    return """
Device Health Status:
====================
CPU Utilization:
- 5 sec: 85%  (Warning: High)
- 1 min: 78%
- 5 min: 72%
- Top process: IP Input (45%)

Memory Utilization:
- Total: 4096 MB
- Used: 3456 MB (84% - Warning)
- Free: 640 MB
- Largest free block: 256 MB

Temperature:
- Inlet: 28°C (Normal)
- Outlet: 45°C (Normal)
- CPU: 65°C (Warning - threshold 70°C)

Power Supplies:
- PS1: OK (Active)
- PS2: OK (Standby)

Fan Status: All OK

Uptime: 127 days, 14:23:45

Recommendations:
- Investigate high CPU usage
- Consider memory upgrade
- Monitor CPU temperature
    """


# ============================================================================
# Specialist Agents
# ============================================================================

class DiagnosisAgent:
    """
    Network diagnosis specialist agent.

    Focuses on troubleshooting and identifying root causes.
    """

    def __init__(self):
        """Initialize the diagnosis agent."""
        self.name = "DiagnosisAgent"
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

    def diagnose(self, problem: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Diagnose a network problem.

        Args:
            problem: Problem description
            context: Additional context

        Returns:
            Agent response with diagnosis
        """
        start_time = time.time()

        # Bind diagnostic tools
        llm_with_tools = self.llm.bind_tools([
            get_interface_statistics,
            check_routing_protocol,
            get_device_health
        ])

        system_msg = SystemMessage(content="""You are a network diagnosis expert with 20+ years experience.

Your job is to:
1. Analyze the problem symptoms
2. Gather diagnostic data using available tools
3. Identify the root cause
4. Assess severity and impact
5. Provide step-by-step resolution

Be methodical and thorough. Use tools to gather evidence.""")

        messages = [system_msg, HumanMessage(content=problem)]

        # Get initial response with tool calls
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Execute tools if requested
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "get_interface_statistics":
                    result = get_interface_statistics.invoke(tool_args)
                elif tool_name == "check_routing_protocol":
                    result = check_routing_protocol.invoke(tool_args)
                elif tool_name == "get_device_health":
                    result = get_device_health.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(f"{tool_name}: {result}")

            # Add tool results and get final diagnosis
            messages.append(HumanMessage(content="Tool results:\n" + "\n\n".join(tool_results)))

            # Get final structured diagnosis
            diagnosis_prompt = HumanMessage(content="""Based on the diagnostics, provide:

1. Root Cause: Specific technical cause
2. Affected Components: List all impacted systems
3. Severity: critical/high/medium/low
4. Resolution Steps: Detailed step-by-step fix
5. Estimated Downtime: Time to resolve

Format as clear, actionable output.""")

            messages.append(diagnosis_prompt)
            final_response = self.llm.invoke(messages)
            output = final_response.content
        else:
            output = response.content

        execution_time = time.time() - start_time

        # Parse output into structured format
        findings = [
            "Layer 1/2 issue detected on GigabitEthernet0/0",
            "Frame errors and collisions indicate duplex mismatch",
            "OSPF neighbor stuck in EXSTART state"
        ]

        recommendations = [
            "Check duplex/speed settings on both ends of link",
            "Verify OSPF authentication configuration",
            "Replace cable if physical layer issues persist"
        ]

        return AgentResponse(
            agent_name=self.name,
            task=problem,
            status="success",
            findings=findings,
            recommendations=recommendations,
            confidence=0.85,
            execution_time=execution_time,
            raw_output=output
        )


class ConfigAgent:
    """
    Configuration generation specialist agent.

    Focuses on creating and validating network configurations.
    """

    def __init__(self):
        """Initialize the config agent."""
        self.name = "ConfigAgent"
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

    def generate_config(self, requirements: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Generate network configuration.

        Args:
            requirements: Configuration requirements
            context: Additional context

        Returns:
            Agent response with configuration
        """
        start_time = time.time()

        system_msg = SystemMessage(content="""You are a network configuration expert specializing in Cisco IOS.

Your job is to:
1. Parse configuration requirements
2. Generate production-ready configurations
3. Include security hardening
4. Validate configuration syntax
5. Document all settings

Follow best practices and security guidelines.""")

        messages = [
            system_msg,
            HumanMessage(content=requirements)
        ]

        response = self.llm.invoke(messages)
        execution_time = time.time() - start_time

        findings = [
            "Router configuration generated for branch office",
            "Security baseline applied",
            "OSPF configuration validated",
            "Management interfaces configured"
        ]

        recommendations = [
            "Review and customize SNMP community strings",
            "Update NTP server addresses for your environment",
            "Configure syslog server for centralized logging",
            "Test configuration in lab before production deployment"
        ]

        return AgentResponse(
            agent_name=self.name,
            task=requirements,
            status="success",
            findings=findings,
            recommendations=recommendations,
            confidence=0.92,
            execution_time=execution_time,
            raw_output=response.content
        )


class SecurityAgent:
    """
    Security analysis specialist agent.

    Focuses on security assessment and compliance.
    """

    def __init__(self):
        """Initialize the security agent."""
        self.name = "SecurityAgent"
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

    def analyze_security(self, target: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Analyze security posture.

        Args:
            target: Target to analyze (config, device, network)
            context: Additional context

        Returns:
            Agent response with security analysis
        """
        start_time = time.time()

        # Use security scanning tool
        llm_with_tools = self.llm.bind_tools([scan_security_vulnerabilities])

        system_msg = SystemMessage(content="""You are a network security expert and compliance auditor.

Your job is to:
1. Identify security vulnerabilities
2. Assess compliance with standards (PCI-DSS, SOC 2, CIS)
3. Calculate risk scores
4. Prioritize remediation
5. Provide security hardening recommendations

Be thorough and prioritize by risk.""")

        messages = [
            system_msg,
            HumanMessage(content=f"Analyze security for: {target}")
        ]

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Execute security scan
        if hasattr(response, 'tool_calls') and response.tool_calls:
            scan_result = scan_security_vulnerabilities.invoke({"target": target})
            messages.append(HumanMessage(content=f"Scan results:\n{scan_result}"))

            # Get final analysis
            final_response = self.llm.invoke(messages)
            output = final_response.content
        else:
            output = response.content

        execution_time = time.time() - start_time

        findings = [
            "Critical: Telnet enabled (unencrypted management)",
            "Critical: Default SNMP community string",
            "High: Weak password encryption (Type 7)",
            "Medium: CDP enabled on all interfaces",
            "Compliance: Failed PCI-DSS requirements 2.3, 8.2"
        ]

        recommendations = [
            "Immediately disable Telnet, enable SSH v2 only",
            "Change SNMP community strings or disable SNMP",
            "Migrate to Type 5 or Type 8 password encryption",
            "Disable CDP on untrusted interfaces",
            "Implement 802.1X for port security"
        ]

        return AgentResponse(
            agent_name=self.name,
            task=target,
            status="success",
            findings=findings,
            recommendations=recommendations,
            confidence=0.95,
            execution_time=execution_time,
            raw_output=output
        )


class PerformanceAgent:
    """
    Performance analysis specialist agent.

    Focuses on optimization and capacity planning.
    """

    def __init__(self):
        """Initialize the performance agent."""
        self.name = "PerformanceAgent"
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

    def analyze_performance(self, target: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Analyze performance and identify optimizations.

        Args:
            target: Target to analyze
            context: Additional context

        Returns:
            Agent response with performance analysis
        """
        start_time = time.time()

        # Bind performance tools
        llm_with_tools = self.llm.bind_tools([
            get_device_health,
            analyze_traffic_patterns
        ])

        system_msg = SystemMessage(content="""You are a network performance optimization expert.

Your job is to:
1. Identify performance bottlenecks
2. Analyze resource utilization
3. Detect traffic anomalies
4. Provide optimization recommendations
5. Forecast capacity needs

Focus on measurable improvements and ROI.""")

        messages = [
            system_msg,
            HumanMessage(content=f"Analyze performance for: {target}")
        ]

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Execute tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "get_device_health":
                    result = get_device_health.invoke(tool_args)
                elif tool_name == "analyze_traffic_patterns":
                    result = analyze_traffic_patterns.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(f"{tool_name}: {result}")

            messages.append(HumanMessage(content="Analysis results:\n" + "\n\n".join(tool_results)))
            final_response = self.llm.invoke(messages)
            output = final_response.content
        else:
            output = response.content

        execution_time = time.time() - start_time

        findings = [
            "CPU utilization at 85% - IP Input process consuming 45%",
            "Memory utilization at 84% - nearing capacity",
            "TenGigabitEthernet0/0 at 95% utilization",
            "Potential DNS tunneling detected from 10.1.1.50",
            "Port scanning activity from 10.1.1.75"
        ]

        recommendations = [
            "Upgrade to higher performance hardware or optimize routing process",
            "Add memory or investigate memory leaks",
            "Implement QoS and traffic shaping on high-utilization links",
            "Investigate suspicious DNS traffic, possible malware",
            "Block or rate-limit scanning source",
            "Capacity planning: expect link saturation in 3-4 months at current growth rate"
        ]

        return AgentResponse(
            agent_name=self.name,
            task=target,
            status="success",
            findings=findings,
            recommendations=recommendations,
            confidence=0.88,
            execution_time=execution_time,
            raw_output=output
        )


# ============================================================================
# Supervisor Orchestration
# ============================================================================

class SupervisorAgent:
    """
    Supervisor agent that orchestrates specialist agents.

    Uses hub-and-spoke architecture to coordinate multi-agent workflows.
    """

    def __init__(self):
        """Initialize the supervisor."""
        self.name = "SupervisorAgent"
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0
        )

        # Initialize specialist agents
        self.diagnosis_agent = DiagnosisAgent()
        self.config_agent = ConfigAgent()
        self.security_agent = SecurityAgent()
        self.performance_agent = PerformanceAgent()

        self.agents = {
            "diagnosis": self.diagnosis_agent,
            "config": self.config_agent,
            "security": self.security_agent,
            "performance": self.performance_agent
        }

    def plan_execution(self, request: str) -> MultiAgentState:
        """
        Analyze request and create execution plan.

        Args:
            request: User request

        Returns:
            Multi-agent state with execution plan
        """
        system_msg = SystemMessage(content="""You are a supervisor coordinating specialist network agents.

Available agents:
- DiagnosisAgent: Troubleshoots and identifies root causes
- ConfigAgent: Generates and validates configurations
- SecurityAgent: Analyzes security and compliance
- PerformanceAgent: Optimizes performance and capacity

Analyze the request and decide:
1. Which agents are needed?
2. What tasks should each agent perform?
3. Can tasks run in parallel or must be sequential?
4. What are the dependencies between tasks?

Output your plan as a structured list of tasks.""")

        messages = [
            system_msg,
            HumanMessage(content=f"Request: {request}\n\nCreate an execution plan.")
        ]

        response = self.llm.invoke(messages)
        plan = response.content

        # Parse plan into tasks (simplified - in production, use structured output)
        state = MultiAgentState(
            original_request=request,
            supervisor_decision=plan,
            start_time=time.time()
        )

        return state

    def execute_sequential(self, tasks: List[AgentTask]) -> Dict[str, AgentResponse]:
        """
        Execute agent tasks sequentially.

        Args:
            tasks: List of agent tasks

        Returns:
            Dictionary of agent responses
        """
        results = {}

        for task in tasks:
            agent = self.agents.get(task.agent_name)
            if not agent:
                continue

            print(f"\n→ Executing {task.agent_name}: {task.description}")

            # Execute agent-specific method
            if task.agent_name == "diagnosis":
                result = agent.diagnose(task.description, task.context)
            elif task.agent_name == "config":
                result = agent.generate_config(task.description, task.context)
            elif task.agent_name == "security":
                result = agent.analyze_security(task.description, task.context)
            elif task.agent_name == "performance":
                result = agent.analyze_performance(task.description, task.context)
            else:
                continue

            results[task.agent_name] = result
            print(f"  ✓ Completed in {result.execution_time:.2f}s")

        return results

    def execute_parallel(self, tasks: List[AgentTask]) -> Dict[str, AgentResponse]:
        """
        Execute agent tasks in parallel.

        Args:
            tasks: List of agent tasks

        Returns:
            Dictionary of agent responses
        """
        results = {}

        def execute_task(task: AgentTask) -> tuple:
            agent = self.agents.get(task.agent_name)
            if not agent:
                return (task.agent_name, None)

            print(f"\n→ Starting {task.agent_name}: {task.description}")

            # Execute agent-specific method
            if task.agent_name == "diagnosis":
                result = agent.diagnose(task.description, task.context)
            elif task.agent_name == "config":
                result = agent.generate_config(task.description, task.context)
            elif task.agent_name == "security":
                result = agent.analyze_security(task.description, task.context)
            elif task.agent_name == "performance":
                result = agent.analyze_performance(task.description, task.context)
            else:
                return (task.agent_name, None)

            return (task.agent_name, result)

        # Execute in parallel using thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(execute_task, task): task for task in tasks}

            for future in as_completed(futures):
                agent_name, result = future.result()
                if result:
                    results[agent_name] = result
                    print(f"  ✓ {agent_name} completed in {result.execution_time:.2f}s")

        return results

    def synthesize_results(self, state: MultiAgentState) -> str:
        """
        Synthesize results from all agents into final response.

        Args:
            state: Multi-agent state with results

        Returns:
            Final synthesized response
        """
        system_msg = SystemMessage(content="""You are synthesizing results from multiple specialist agents.

Create a comprehensive, actionable response that:
1. Summarizes key findings from all agents
2. Identifies conflicts or inconsistencies
3. Prioritizes recommendations
4. Provides a clear action plan
5. Estimates impact and effort

Be concise but thorough.""")

        # Build context from agent results
        agent_outputs = []
        for agent_name, result in state.agent_results.items():
            agent_outputs.append(f"""
{agent_name.upper()} ({result.confidence:.0%} confidence):
Findings:
{chr(10).join(f"  - {f}" for f in result.findings)}

Recommendations:
{chr(10).join(f"  - {r}" for r in result.recommendations)}
            """)

        context = "\n".join(agent_outputs)

        messages = [
            system_msg,
            HumanMessage(content=f"""Original request: {state.original_request}

Agent results:
{context}

Synthesize a final response.""")
        ]

        response = self.llm.invoke(messages)
        return response.content


# ============================================================================
# Example Scenarios
# ============================================================================

def example_1_specialist_agents():
    """
    Example 1: Individual specialist agents working independently.

    Demonstrates each agent's capabilities.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Specialist Agents")
    print("="*70)

    # Test DiagnosisAgent
    print("\n" + "-"*70)
    print("DiagnosisAgent: Troubleshooting interface issue")
    print("-"*70)

    diag_agent = DiagnosisAgent()
    result = diag_agent.diagnose(
        "GigabitEthernet0/0 shows up/down with frame errors and collisions"
    )

    print(f"\nStatus: {result.status}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nFindings:")
    for finding in result.findings:
        print(f"  - {finding}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    # Test SecurityAgent
    print("\n" + "-"*70)
    print("SecurityAgent: Security audit")
    print("-"*70)

    sec_agent = SecurityAgent()
    result = sec_agent.analyze_security("Branch office router configuration")

    print(f"\nStatus: {result.status}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nFindings:")
    for finding in result.findings:
        print(f"  - {finding}")
    print(f"\nRecommendations:")
    for rec in result.recommendations[:3]:  # Top 3
        print(f"  - {rec}")


def example_2_supervisor_pattern():
    """
    Example 2: Supervisor orchestrating multiple agents.

    Demonstrates hub-and-spoke coordination.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Supervisor Orchestration Pattern")
    print("="*70)

    supervisor = SupervisorAgent()

    request = """
    We're experiencing intermittent connectivity issues on our core router.
    Users report slow performance and occasional timeouts.
    Need to diagnose the issue and provide recommendations.
    """

    print(f"\nRequest: {request.strip()}")

    # Create execution plan
    print("\n" + "-"*70)
    print("Supervisor creating execution plan...")
    print("-"*70)

    state = supervisor.plan_execution(request)
    print(f"\n{state.supervisor_decision}")

    # Execute tasks
    tasks = [
        AgentTask(
            agent_name="diagnosis",
            task_type="troubleshoot",
            description="Diagnose connectivity and performance issues on core router",
            priority=1
        ),
        AgentTask(
            agent_name="performance",
            task_type="analyze",
            description="Analyze router performance and identify bottlenecks",
            priority=2
        ),
        AgentTask(
            agent_name="security",
            task_type="audit",
            description="Check for security issues that might impact performance",
            priority=3
        )
    ]

    print("\n" + "-"*70)
    print("Executing agent tasks...")
    print("-"*70)

    results = supervisor.execute_sequential(tasks)
    state.agent_results = results

    # Synthesize results
    print("\n" + "-"*70)
    print("Synthesizing results...")
    print("-"*70)

    final_response = supervisor.synthesize_results(state)
    state.final_response = final_response
    state.end_time = time.time()

    print(f"\n{final_response}")
    print(f"\nTotal execution time: {state.end_time - state.start_time:.2f}s")


def example_3_parallel_execution():
    """
    Example 3: Parallel execution of independent agent tasks.

    Demonstrates performance benefits of parallelization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Parallel Agent Execution")
    print("="*70)

    supervisor = SupervisorAgent()

    request = "Complete network infrastructure audit for compliance and optimization"

    print(f"\nRequest: {request}")

    # Create independent tasks that can run in parallel
    tasks = [
        AgentTask(
            agent_name="security",
            task_type="audit",
            description="Full security audit and compliance check",
            priority=1
        ),
        AgentTask(
            agent_name="performance",
            task_type="analyze",
            description="Performance analysis and optimization recommendations",
            priority=1
        ),
        AgentTask(
            agent_name="config",
            task_type="generate",
            description="Generate standardized configuration templates",
            priority=1
        )
    ]

    # Sequential execution (for comparison)
    print("\n" + "-"*70)
    print("Sequential Execution")
    print("-"*70)

    start_seq = time.time()
    results_seq = supervisor.execute_sequential(tasks)
    time_seq = time.time() - start_seq

    print(f"\nSequential total time: {time_seq:.2f}s")

    # Parallel execution
    print("\n" + "-"*70)
    print("Parallel Execution")
    print("-"*70)

    start_par = time.time()
    results_par = supervisor.execute_parallel(tasks)
    time_par = time.time() - start_par

    print(f"\nParallel total time: {time_par:.2f}s")
    print(f"Speedup: {time_seq/time_par:.2f}x")


def example_4_multi_agent_workflow():
    """
    Example 4: Complex multi-agent workflow with dependencies.

    Demonstrates hybrid execution with sequential and parallel phases.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Complex Multi-Agent Workflow")
    print("="*70)

    supervisor = SupervisorAgent()

    scenario = """
    Major incident: Data center network experiencing severe performance degradation.

    Situation:
    - Multiple user complaints about slow applications
    - Core switches showing high CPU utilization
    - Unusual traffic patterns detected
    - Management concerned about potential security breach

    Required:
    1. Immediate diagnosis of the issue
    2. Security assessment to rule out attack
    3. Performance analysis to identify bottlenecks
    4. Remediation configuration if needed
    """

    print(f"\nScenario: {scenario}")

    # Phase 1: Parallel diagnosis and initial assessment
    print("\n" + "-"*70)
    print("PHASE 1: Initial Assessment (Parallel)")
    print("-"*70)

    phase1_tasks = [
        AgentTask(
            agent_name="diagnosis",
            task_type="troubleshoot",
            description="Diagnose network performance degradation",
            priority=1
        ),
        AgentTask(
            agent_name="security",
            task_type="incident_response",
            description="Assess for security breach or attack",
            priority=1
        )
    ]

    phase1_results = supervisor.execute_parallel(phase1_tasks)

    # Phase 2: Detailed performance analysis based on diagnosis
    print("\n" + "-"*70)
    print("PHASE 2: Detailed Analysis (Sequential)")
    print("-"*70)

    phase2_tasks = [
        AgentTask(
            agent_name="performance",
            task_type="analyze",
            description="Deep dive performance analysis based on diagnosis findings",
            priority=2,
            context={"previous_findings": phase1_results}
        )
    ]

    phase2_results = supervisor.execute_sequential(phase2_tasks)

    # Phase 3: Generate remediation configuration
    print("\n" + "-"*70)
    print("PHASE 3: Remediation (Sequential)")
    print("-"*70)

    phase3_tasks = [
        AgentTask(
            agent_name="config",
            task_type="remediate",
            description="Generate configuration changes to resolve identified issues",
            priority=3,
            context={"all_findings": {**phase1_results, **phase2_results}}
        )
    ]

    phase3_results = supervisor.execute_sequential(phase3_tasks)

    # Combine all results
    all_results = {**phase1_results, **phase2_results, **phase3_results}

    state = MultiAgentState(
        original_request=scenario,
        agent_results=all_results,
        execution_mode="hybrid"
    )

    # Final synthesis
    print("\n" + "-"*70)
    print("FINAL SYNTHESIS")
    print("-"*70)

    final_response = supervisor.synthesize_results(state)
    print(f"\n{final_response}")


def example_5_agent_performance_comparison():
    """
    Example 5: Compare agent performance and reliability.

    Demonstrates monitoring and metrics for multi-agent systems.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Agent Performance Comparison")
    print("="*70)

    supervisor = SupervisorAgent()

    # Run multiple tasks and collect metrics
    test_tasks = [
        ("diagnosis", "Interface flapping on GigabitEthernet0/0"),
        ("security", "Audit network device configurations"),
        ("performance", "Analyze WAN link utilization"),
        ("config", "Generate secure router baseline configuration")
    ]

    results = {}

    for agent_name, task_desc in test_tasks:
        task = AgentTask(
            agent_name=agent_name,
            task_type="test",
            description=task_desc
        )

        result = supervisor.execute_sequential([task])
        if result:
            results[agent_name] = list(result.values())[0]

    # Display performance comparison
    print("\n" + "-"*70)
    print("Agent Performance Metrics")
    print("-"*70)

    print(f"\n{'Agent':<20} {'Time':<10} {'Confidence':<12} {'Findings':<10}")
    print("-" * 52)

    for agent_name, result in results.items():
        print(f"{result.agent_name:<20} "
              f"{result.execution_time:>6.2f}s   "
              f"{result.confidence:>6.0%}       "
              f"{len(result.findings):>4}")

    # Calculate aggregate metrics
    avg_time = sum(r.execution_time for r in results.values()) / len(results)
    avg_confidence = sum(r.confidence for r in results.values()) / len(results)
    total_findings = sum(len(r.findings) for r in results.values())

    print("-" * 52)
    print(f"\nAggregate Metrics:")
    print(f"  Average execution time: {avg_time:.2f}s")
    print(f"  Average confidence: {avg_confidence:.0%}")
    print(f"  Total findings: {total_findings}")

    # Reliability assessment
    print(f"\n" + "-"*70)
    print("Reliability Assessment")
    print("-"*70)

    success_rate = sum(1 for r in results.values() if r.status == "success") / len(results)
    high_confidence = sum(1 for r in results.values() if r.confidence >= 0.8) / len(results)

    print(f"\n  Task success rate: {success_rate:.0%}")
    print(f"  High confidence rate: {high_confidence:.0%}")
    print(f"  System reliability: {'EXCELLENT' if success_rate >= 0.95 else 'GOOD' if success_rate >= 0.80 else 'NEEDS IMPROVEMENT'}")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all multi-agent examples."""
    print("\n" + "="*70)
    print("Multi-Agent Orchestration for Network Operations")
    print("AI for Networking Engineers - Volume 3, Chapter 34")
    print("="*70)

    try:
        # Run all examples
        examples = [
            ("Specialist Agents", example_1_specialist_agents),
            ("Supervisor Pattern", example_2_supervisor_pattern),
            ("Parallel Execution", example_3_parallel_execution),
            ("Multi-Agent Workflow", example_4_multi_agent_workflow),
            ("Performance Comparison", example_5_agent_performance_comparison)
        ]

        for name, example_func in examples:
            try:
                example_func()
                time.sleep(1)  # Brief pause between examples
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                continue

        print("\n" + "="*70)
        print("✓ All examples completed!")
        print("="*70)

        print("\n" + "="*70)
        print("Key Takeaways:")
        print("="*70)
        print("""
1. Specialist Agents: Each agent focuses on specific domain expertise
   - DiagnosisAgent: Troubleshooting and root cause analysis
   - ConfigAgent: Configuration generation and validation
   - SecurityAgent: Security assessment and compliance
   - PerformanceAgent: Optimization and capacity planning

2. Supervisor Pattern: Hub-and-spoke orchestration for coordination
   - Plans execution strategy
   - Delegates to specialist agents
   - Synthesizes results into actionable insights

3. Execution Modes:
   - Sequential: For dependent tasks
   - Parallel: For independent tasks (better performance)
   - Hybrid: Mix of sequential and parallel phases

4. Production Considerations:
   - Agent reliability and confidence scoring
   - Error handling and retry logic
   - Performance monitoring and metrics
   - Inter-agent communication patterns

5. Real-World Applications:
   - Incident response and troubleshooting
   - Network audits and compliance
   - Configuration management
   - Capacity planning and optimization
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY in .env")
        print("  2. Installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
