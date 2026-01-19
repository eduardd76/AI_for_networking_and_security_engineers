# Chapter 24: Log Analysis with AI

## Why This Chapter Matters

10,000 syslog messages per hour. An outage happened. Which log line caused it?

**Traditional approach**: grep, awk, sed scripts for hours
**AI approach**: "What caused the BGP session to flap?" → Answer in 30 seconds with exact log lines

This chapter builds AI-powered log analysis that:
- Processes massive log volumes
- Identifies anomalies automatically
- Correlates events across devices
- Extracts root causes from noise
- Generates human-readable summaries

**Real impact**: Mean time to resolution (MTTR) drops from hours to minutes.

---

## Section 1: The Log Analysis Challenge

### Why Logs Are Hard

**Volume**: 10,000-100,000+ messages per hour
**Variety**: Syslog, SNMP traps, NetFlow, API logs
**Verbosity**: 99% noise, 1% signal
**Timing**: Correlated events across multiple devices
**Formats**: Every vendor uses different formats

**Traditional tools**:
- `grep`: Finds keywords, but you need to know what to search for
- Splunk/ELK: Great for known patterns, poor for unknown issues
- Regex: Brittle, breaks with format changes

**AI advantage**: Understands semantics, not just patterns.

---

## Section 2: Simple Log Analyzer

### Basic Log Summarization

```python
# log_analyzer_simple.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

class SimpleLogAnalyzer:
    """Analyze network logs with AI."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

    def analyze_logs(self, logs: str, question: str = None) -> str:
        """
        Analyze logs and answer questions.

        Args:
            logs: Raw log text
            question: Optional specific question about logs

        Returns:
            Analysis or answer
        """
        if question:
            prompt = ChatPromptTemplate.from_template("""
Analyze these network logs and answer the question.

Logs:
{logs}

Question: {question}

Answer:""")
            message = prompt.format(logs=logs, question=question)
        else:
            prompt = ChatPromptTemplate.from_template("""
Analyze these network logs and provide:
1. Summary of events
2. Any errors or warnings
3. Root cause if there's an issue
4. Timeline of key events

Logs:
{logs}

Analysis:""")
            message = prompt.format(logs=logs)

        parser = StrOutputParser()
        chain = prompt | self.llm | parser

        return chain.invoke({"logs": logs, "question": question} if question else {"logs": logs})

    def find_root_cause(self, logs: str) -> str:
        """Find root cause of issues in logs."""
        return self.analyze_logs(logs, "What is the root cause of the problem in these logs?")

    def get_timeline(self, logs: str) -> str:
        """Extract timeline of events."""
        return self.analyze_logs(logs, "Create a timeline of the key events in these logs.")


# Example usage
if __name__ == "__main__":
    analyzer = SimpleLogAnalyzer(api_key="your-api-key")

    # Sample logs
    logs = """
2024-01-17 14:32:15 router-core-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
2024-01-17 14:32:16 router-core-01: %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
2024-01-17 14:32:17 router-core-01: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
2024-01-17 14:32:45 router-core-01: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
2024-01-17 14:32:46 router-core-01: %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up
2024-01-17 14:32:55 router-core-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Up
2024-01-17 14:35:20 router-core-01: %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
2024-01-17 14:35:21 router-core-01: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
2024-01-17 14:35:22 router-core-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
    """

    # Analyze
    print("=== Root Cause Analysis ===")
    print(analyzer.find_root_cause(logs))

    print("\n=== Timeline ===")
    print(analyzer.get_timeline(logs))

    print("\n=== Custom Question ===")
    print(analyzer.analyze_logs(logs, "How many times did the BGP session flap?"))
```

**Output**:
```
=== Root Cause Analysis ===
Root Cause: GigabitEthernet0/1 is experiencing interface flapping, causing BGP neighbor 192.168.1.2 to go down repeatedly.

The interface went down at 14:32:17, came back up at 14:32:45, then went down again at 14:35:21. This indicates:
- Physical layer issue (cable, SFP, or remote port problem)
- Possible power issues
- Environmental problems (EMI, temperature)

The BGP session is a victim of the underlying interface instability, not the root cause itself.

Recommended actions:
1. Check interface error counters: show interface gi0/1
2. Check cable and SFP module
3. Check remote switch port
4. Review environmental sensors

=== Timeline ===
14:32:15 - BGP neighbor 192.168.1.2 goes down (interface flap)
14:32:16 - Line protocol on Gi0/1 goes down
14:32:17 - Interface Gi0/1 goes down
14:32:45 - Interface Gi0/1 comes back up (28 seconds outage)
14:32:46 - Line protocol on Gi0/1 comes back up
14:32:55 - BGP neighbor 192.168.1.2 reestablishes
14:35:20 - Line protocol on Gi0/1 goes down again
14:35:21 - Interface Gi0/1 goes down again
14:35:22 - BGP neighbor 192.168.1.2 goes down again

Pattern: Interface flapping every ~3 minutes, indicating intermittent physical connectivity.

=== Custom Question ===
The BGP session flapped 2 times during this log period (went down at 14:32:15 and 14:35:22).
```

---

## Section 3: Structured Log Extraction

### Extract Structured Data from Logs

```python
# structured_log_extractor.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class LogEvent(BaseModel):
    """Structured log event."""
    timestamp: str = Field(description="Event timestamp")
    severity: str = Field(description="Severity level (critical, error, warning, info)")
    device: str = Field(description="Device hostname")
    interface: Optional[str] = Field(description="Interface name if applicable")
    protocol: Optional[str] = Field(description="Protocol (BGP, OSPF, etc.)")
    event_type: str = Field(description="Type of event")
    description: str = Field(description="Human-readable description")

class LogAnalysis(BaseModel):
    """Complete log analysis."""
    summary: str = Field(description="Overall summary")
    events: List[LogEvent] = Field(description="List of key events")
    root_cause: str = Field(description="Root cause analysis")
    affected_services: List[str] = Field(description="Impacted services/protocols")
    recommended_actions: List[str] = Field(description="What to do next")

class StructuredLogExtractor:
    """Extract structured data from logs."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

        self.parser = JsonOutputParser(pydantic_object=LogAnalysis)

    def extract(self, logs: str) -> LogAnalysis:
        """Extract structured analysis from logs."""
        prompt = ChatPromptTemplate.from_template("""
Analyze these network logs and extract structured information.

Logs:
{logs}

Return JSON matching this schema:
{format_instructions}

JSON:""")

        chain = prompt | self.llm | self.parser

        result = chain.invoke({
            "logs": logs,
            "format_instructions": self.parser.get_format_instructions()
        })

        return LogAnalysis(**result)


# Example usage
if __name__ == "__main__":
    extractor = StructuredLogExtractor(api_key="your-api-key")

    logs = """
2024-01-17 14:32:15 router-core-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
2024-01-17 14:32:17 router-core-01: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
2024-01-17 14:32:45 router-core-01: %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
2024-01-17 14:32:55 router-core-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Up
    """

    analysis = extractor.extract(logs)

    print(f"Summary: {analysis.summary}\n")
    print(f"Root Cause: {analysis.root_cause}\n")
    print(f"Affected Services: {', '.join(analysis.affected_services)}\n")

    print("Events:")
    for event in analysis.events:
        print(f"  [{event.timestamp}] {event.severity.upper()}: {event.description}")

    print("\nRecommended Actions:")
    for i, action in enumerate(analysis.recommended_actions, 1):
        print(f"  {i}. {action}")
```

---

## Section 4: Anomaly Detection

### Find Unusual Patterns

```python
# log_anomaly_detector.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from collections import Counter
import re

class Anomaly(BaseModel):
    """Detected anomaly."""
    timestamp: str = Field(description="When anomaly occurred")
    type: str = Field(description="Type of anomaly")
    severity: str = Field(description="critical, high, medium, low")
    description: str = Field(description="What's anomalous")
    baseline: str = Field(description="Normal behavior")
    actual: str = Field(description="Observed behavior")
    impact: str = Field(description="Potential impact")

class LogAnomalyDetector:
    """Detect anomalies in logs."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # Use Haiku for cost efficiency
            api_key=api_key,
            temperature=0.0
        )

    def baseline_analysis(self, logs: str) -> dict:
        """Analyze logs to understand baseline behavior."""
        # Simple statistical baseline
        lines = logs.strip().split('\n')

        # Count message types
        message_types = Counter()
        for line in lines:
            # Extract message type (e.g., %BGP-5-ADJCHANGE)
            match = re.search(r'%([A-Z]+)-\d+-([A-Z]+)', line)
            if match:
                msg_type = f"{match.group(1)}-{match.group(2)}"
                message_types[msg_type] += 1

        # Count severity levels
        severities = Counter()
        for line in lines:
            match = re.search(r'%-\d+-', line)
            if match:
                severity = match.group(0)
                severities[severity] += 1

        return {
            "total_messages": len(lines),
            "message_types": dict(message_types),
            "severities": dict(severities),
            "messages_per_minute": len(lines) / 60  # Assume 1 hour of logs
        }

    def detect_anomalies(
        self,
        current_logs: str,
        baseline_logs: str = None
    ) -> List[Anomaly]:
        """Detect anomalies by comparing to baseline."""
        current_stats = self.baseline_analysis(current_logs)

        if baseline_logs:
            baseline_stats = self.baseline_analysis(baseline_logs)
        else:
            # Use current logs as baseline (look for internal anomalies)
            baseline_stats = current_stats

        # Use LLM to identify anomalies
        prompt = ChatPromptTemplate.from_template("""
Analyze these logs for anomalies.

Current log statistics:
{current_stats}

Baseline statistics:
{baseline_stats}

Current logs:
{current_logs}

Identify anomalies:
- Message rate spikes
- New error types
- Repeated failures
- Unusual patterns

Return JSON array of anomalies matching this schema for each:
{{
  "timestamp": "when",
  "type": "type of anomaly",
  "severity": "critical/high/medium/low",
  "description": "what's anomalous",
  "baseline": "normal behavior",
  "actual": "observed behavior",
  "impact": "potential impact"
}}

JSON array:""")

        response = self.llm.invoke(
            prompt.format(
                current_stats=current_stats,
                baseline_stats=baseline_stats,
                current_logs=current_logs
            )
        )

        # Parse JSON
        import json
        try:
            anomalies_data = json.loads(response.content)
            return [Anomaly(**a) for a in anomalies_data]
        except:
            return []


# Example usage
if __name__ == "__main__":
    detector = LogAnomalyDetector(api_key="your-api-key")

    # Baseline (normal) logs
    baseline = """
2024-01-17 10:00:00 router-01: %SYS-5-CONFIG_I: Configured from console
2024-01-17 10:15:00 router-01: %SYS-5-CONFIG_I: Configured from console
    """

    # Current logs with anomalies
    current = """
2024-01-17 14:00:00 router-01: %BGP-3-NOTIFICATION: sent to neighbor 192.168.1.2
2024-01-17 14:00:01 router-01: %BGP-3-NOTIFICATION: sent to neighbor 192.168.1.2
2024-01-17 14:00:02 router-01: %BGP-3-NOTIFICATION: sent to neighbor 192.168.1.2
2024-01-17 14:00:03 router-01: %BGP-3-NOTIFICATION: sent to neighbor 192.168.1.2
2024-01-17 14:00:04 router-01: %BGP-3-NOTIFICATION: sent to neighbor 192.168.1.2
2024-01-17 14:00:15 router-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down BGP Notification sent
2024-01-17 14:00:20 router-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.3 Down BGP Notification sent
2024-01-17 14:00:25 router-01: %BGP-5-ADJCHANGE: neighbor 192.168.1.4 Down BGP Notification sent
    """

    anomalies = detector.detect_anomalies(current, baseline)

    print(f"Detected {len(anomalies)} anomalies:\n")

    for anomaly in anomalies:
        print(f"[{anomaly.severity.upper()}] {anomaly.type}")
        print(f"  Time: {anomaly.timestamp}")
        print(f"  Description: {anomaly.description}")
        print(f"  Baseline: {anomaly.baseline}")
        print(f"  Actual: {anomaly.actual}")
        print(f"  Impact: {anomaly.impact}\n")

# Expected output:
"""
Detected 2 anomalies:

[CRITICAL] BGP Notification Storm
  Time: 2024-01-17 14:00:00
  Description: Rapid repeated BGP notifications to same neighbor
  Baseline: No BGP notifications in baseline period
  Actual: 5 notifications in 5 seconds to 192.168.1.2
  Impact: BGP session instability, potential route flapping

[HIGH] Multiple BGP Session Failures
  Time: 2024-01-17 14:00:15-25
  Description: Three BGP neighbors went down in 10 seconds
  Baseline: No BGP session changes
  Actual: 3 neighbors (192.168.1.2, .3, .4) all went down
  Impact: Possible local router issue affecting all peerings
"""
```

---

## Section 5: Multi-Device Log Correlation

### Correlate Events Across Devices

```python
# log_correlator.py
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DeviceLogs:
    """Logs from a single device."""
    device_name: str
    logs: str

class LogCorrelator:
    """Correlate logs across multiple devices."""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.0
        )

    def correlate(self, device_logs: List[DeviceLogs]) -> str:
        """
        Correlate logs from multiple devices.

        Args:
            device_logs: List of DeviceLogs objects

        Returns:
            Correlation analysis
        """
        # Format logs for LLM
        formatted_logs = []
        for dl in device_logs:
            formatted_logs.append(f"=== {dl.device_name} ===\n{dl.logs}")

        all_logs = "\n\n".join(formatted_logs)

        prompt = ChatPromptTemplate.from_template("""
Analyze logs from multiple network devices and correlate events.

Logs from {num_devices} devices:
{logs}

Provide:
1. Timeline of correlated events (what happened across all devices)
2. Root cause analysis (which device/event triggered the cascade)
3. Impact radius (which devices were affected and how)
4. Recovery sequence (what to fix first)

Analysis:""")

        response = self.llm.invoke(
            prompt.format(num_devices=len(device_logs), logs=all_logs)
        )

        return response.content


# Example usage
if __name__ == "__main__":
    correlator = LogCorrelator(api_key="your-api-key")

    # Logs from 3 devices showing cascade failure
    logs = [
        DeviceLogs(
            device_name="router-core-01",
            logs="""
2024-01-17 14:32:15 %LINEPROTO-5-UPDOWN: Line protocol on Interface Gi0/1, changed state to down
2024-01-17 14:32:16 %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
2024-01-17 14:32:17 %OSPF-5-ADJCHG: Process 1, Nbr 192.168.1.3 on Gi0/2 from FULL to DOWN
            """
        ),
        DeviceLogs(
            device_name="router-edge-01",
            logs="""
2024-01-17 14:32:16 %BGP-5-ADJCHANGE: neighbor 192.168.1.1 Down Peer closed the session
2024-01-17 14:32:30 %ROUTING-4-NORTETO: No route to host 10.0.0.0/8
            """
        ),
        DeviceLogs(
            device_name="switch-dist-01",
            logs="""
2024-01-17 14:32:14 %LINK-3-UPDOWN: Interface Gi1/0/24, changed state to down
2024-01-17 14:32:15 %LINEPROTO-5-UPDOWN: Line protocol on Interface Gi1/0/24, changed state to down
            """
        )
    ]

    analysis = correlator.correlate(logs)
    print(analysis)

# Expected output:
"""
Timeline of Correlated Events:

14:32:14 - switch-dist-01: Interface Gi1/0/24 goes down
14:32:15 - switch-dist-01: Line protocol on Gi1/0/24 goes down
14:32:15 - router-core-01: Gi0/1 line protocol down (connected to switch Gi1/0/24)
14:32:16 - router-core-01: BGP neighbor 192.168.1.2 down (over Gi0/1)
14:32:16 - router-edge-01: BGP session to 192.168.1.1 closed
14:32:17 - router-core-01: OSPF neighbor down
14:32:30 - router-edge-01: Lost routes to 10.0.0.0/8

Root Cause Analysis:
The failure originated at switch-dist-01 interface Gi1/0/24 going down. This is the uplink to router-core-01's Gi0/1 interface. The link failure cascaded:
1. Core router lost connectivity on Gi0/1
2. BGP sessions over that link failed
3. OSPF adjacency lost
4. Edge router lost BGP session to core
5. Edge router lost routes to internal networks

Impact Radius:
- Primary: switch-dist-01 (link failure)
- Secondary: router-core-01 (lost connectivity)
- Tertiary: router-edge-01 and all downstream users (lost routes)

Recovery Sequence:
1. Fix switch-dist-01 Gi1/0/24 (check cable, port, or remote device)
2. Verify router-core-01 Gi0/1 comes back up
3. Wait for BGP sessions to reestablish (may take 3-5 minutes)
4. Verify OSPF adjacencies reform
5. Verify edge router regains routes to 10.0.0.0/8
6. Test end-to-end connectivity
"""
```

---

## What Can Go Wrong

**1. Token limits with large logs**
- 100K log lines exceed context window
- LLM can't process
- Solution: Chunk logs, summarize, or filter first

**2. Hallucination with ambiguous logs**
- AI invents explanations for unclear logs
- Looks confident but wrong
- Solution: Cross-reference with actual device state

**3. Cost at scale**
- Analyzing every log message is expensive
- $100s per day at high volume
- Solution: Pre-filter with traditional tools, AI for anomalies only

**4. Latency for real-time**
- API calls take 1-2 seconds
- Too slow for real-time alerting
- Solution: Use AI for post-incident analysis, not real-time

**5. Missing context**
- Logs don't include device configs
- AI can't fully diagnose
- Solution: Provide context (device role, expected neighbors, etc.)

---

## Key Takeaways

1. **AI excels at log summarization** - Turns noise into signal
2. **Structured extraction** enables automation downstream
3. **Anomaly detection** finds issues you didn't know to look for
4. **Correlation** reveals cascade failures across devices
5. **Combine traditional + AI** - Filter with grep, analyze with AI

Next: Network automation, config generation, and advanced applications.
