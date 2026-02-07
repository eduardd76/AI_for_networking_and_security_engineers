# Chapter 24: Log Analysis with AI

## From Collection to Intelligence

Chapter 23 covered collecting and classifying logs. This chapter goes deeper — using AI to *analyze* logs, extract patterns, detect threats, and generate actionable intelligence. The goal is to turn raw syslog noise into insights that help you keep the network running.

**Networking analogy**: If Chapter 23 was building the syslog infrastructure (like setting up your NMS), this chapter is building the dashboards and alerting rules — except instead of static thresholds, AI understands context and can reason about what log patterns mean.

---

## Use Case 1: Intelligent Log Summarization

Network teams often need to understand "what happened overnight" or "what's the current state of the network." Rather than reading thousands of log lines, let AI summarize.

```python
import os
import json
import re
from typing import Dict, List
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY environment variable

def summarize_network_logs(
    logs: List[str],
    time_period: str = "last 24 hours"
) -> str:
    """Generate an executive summary of network log activity.

    Like a NOC shift handoff report — what happened, what's
    still broken, what needs attention on the next shift.
    """

    log_text = "\n".join(logs[:200])  # Limit to avoid token overflow

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze these network device logs from the {time_period}
and create a NOC shift handoff summary.

Include these sections:
1. **Critical Events**: Any outages, failures, or security incidents
2. **Notable Changes**: Interface state changes, routing changes, config changes
3. **Trending Issues**: Patterns that suggest developing problems
4. **Action Items**: What the next team should investigate or monitor
5. **Statistics**: Count of events by severity and category

Logs:
{log_text}

Write the summary as a network engineer would — concise, technical, actionable."""
        }]
    )

    return response.content[0].text


# Example
sample_logs = [
    "Jan 15 02:15:33 router-core-01: %BGP-5-ADJCHANGE: neighbor 203.0.113.2 Down BGP Notification sent",
    "Jan 15 02:15:34 router-core-01: %BGP-3-NOTIFICATION: sent to neighbor 203.0.113.2 4/0 (hold time expired)",
    "Jan 15 02:17:45 router-core-01: %BGP-5-ADJCHANGE: neighbor 203.0.113.2 Up",
    "Jan 15 03:00:01 switch-dist-01: %LINK-3-UPDOWN: Interface Gi1/0/24, changed state to down",
    "Jan 15 03:00:02 switch-dist-01: %LINK-3-UPDOWN: Interface Gi1/0/24, changed state to up",
    "Jan 15 03:00:45 switch-dist-01: %LINK-3-UPDOWN: Interface Gi1/0/24, changed state to down",
    "Jan 15 03:00:46 switch-dist-01: %LINK-3-UPDOWN: Interface Gi1/0/24, changed state to up",
    "Jan 15 04:30:00 router-core-02: %SYS-5-CONFIG_I: Configured from console by admin on vty0",
    "Jan 15 05:00:15 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99/443 dst inside:10.0.1.50/8080 by access-group OUTSIDE_IN",
    "Jan 15 05:00:16 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99/443 dst inside:10.0.1.50/8080 by access-group OUTSIDE_IN",
    "Jan 15 05:00:17 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99/443 dst inside:10.0.1.51/8080 by access-group OUTSIDE_IN",
    "Jan 15 06:12:00 switch-access-01: %AUTHMGR-7-FAILOVER: Failing over from method RADIUS to method LOCAL for client on Gi1/0/5",
    "Jan 15 06:12:01 switch-access-01: %RADIUS-4-RADIUS_DEAD: RADIUS server 10.0.0.50:1812 is not responding",
]

summary = summarize_network_logs(sample_logs)
print(summary)
```

### Expected Summary Output

```
## NOC Shift Handoff — Jan 15, 02:00-07:00

### Critical Events
- **BGP session flap** on router-core-01: Neighbor 203.0.113.2 went down at 02:15
  due to hold timer expiration, recovered at 02:17 (2 min outage)
- **RADIUS server failure**: 10.0.0.50 stopped responding at 06:12. 802.1X
  falling back to local auth on switch-access-01

### Notable Changes
- Config change on router-core-02 at 04:30 by 'admin' via VTY (verify intended)
- Interface Gi1/0/24 on switch-dist-01 flapping (4 state changes in < 1 min)

### Trending Issues
- Firewall OUTSIDE_IN denying traffic from 198.51.100.99 to multiple internal
  hosts on port 8080 — could be misconfigured app or scanning activity
- Interface flapping on switch-dist-01 suggests a bad cable or optic

### Action Items
1. Verify RADIUS server 10.0.0.50 is operational
2. Check cable/optic on switch-dist-01 Gi1/0/24
3. Confirm config change on router-core-02 was authorized
4. Investigate traffic from 198.51.100.99 — legitimate or malicious?

### Statistics
- Critical: 2 | High: 3 | Medium: 4 | Low/Info: 4
```

---

## Use Case 2: Root Cause Analysis

When an outage happens, the first question is always "why?" AI can analyze the log timeline and propose root causes.

```python
def analyze_root_cause(
    logs: List[str],
    incident_description: str
) -> Dict:
    """Perform root cause analysis on an incident using log data.

    Like having a senior engineer review the logs and draw a
    timeline on a whiteboard — but in seconds.
    """

    log_text = "\n".join(logs)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Perform a root cause analysis for this network incident.

Incident: {incident_description}

Related logs (in chronological order):
{log_text}

Provide your analysis as JSON:
{{
    "timeline": [
        {{"time": "HH:MM:SS", "event": "description", "significance": "why this matters"}}
    ],
    "root_cause": "your assessment of the root cause",
    "contributing_factors": ["list of factors that made this worse"],
    "impact": "what was affected and for how long",
    "recommended_fixes": [
        {{"action": "what to do", "priority": "immediate/short-term/long-term", "prevents": "what this prevents"}}
    ],
    "similar_past_patterns": "any patterns that suggest this is recurring"
}}

Think like a senior network engineer doing a post-mortem."""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {"root_cause": text, "recommended_fixes": []}


# Example: BGP session flap RCA
bgp_logs = [
    "Jan 15 02:14:00 router-core-01: %SYS-5-PROGNOSIS: CPU utilization is 85%",
    "Jan 15 02:14:30 router-core-01: %SYS-5-PROGNOSIS: CPU utilization is 92%",
    "Jan 15 02:15:00 router-core-01: %SYS-5-PROGNOSIS: CPU utilization is 98%",
    "Jan 15 02:15:33 router-core-01: %BGP-5-ADJCHANGE: neighbor 203.0.113.2 Down BGP Notification sent",
    "Jan 15 02:15:34 router-core-01: %BGP-3-NOTIFICATION: sent to neighbor 203.0.113.2 4/0 (hold time expired)",
    "Jan 15 02:16:00 router-core-01: %SYS-5-PROGNOSIS: CPU utilization is 45%",
    "Jan 15 02:17:45 router-core-01: %BGP-5-ADJCHANGE: neighbor 203.0.113.2 Up",
]

rca = analyze_root_cause(
    bgp_logs,
    "BGP session with ISP-A (203.0.113.2) dropped for 2 minutes"
)

print(f"Root Cause: {rca['root_cause']}")
print(f"\nRecommended Fixes:")
for fix in rca.get('recommended_fixes', []):
    print(f"  [{fix['priority']}] {fix['action']}")
```

---

## Use Case 3: Security Log Analysis

Security teams need to quickly identify threats from firewall, IDS/IPS, and authentication logs. AI can triage security events faster than manual review.

```python
def analyze_security_logs(
    logs: List[str],
    context: str = "enterprise network"
) -> Dict:
    """Analyze security-related logs for threats and anomalies.

    Like having a SOC analyst review the logs — classifies
    events by threat level and recommends response actions.
    """

    log_text = "\n".join(logs[:100])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze these security logs from a {context} and identify
potential security threats or concerns.

Logs:
{log_text}

For each finding, provide:
1. **Threat Category**: brute_force, scanning, data_exfiltration, unauthorized_access,
   policy_violation, malware_c2, reconnaissance, or benign
2. **Confidence**: high, medium, or low
3. **Evidence**: which log entries support this finding
4. **MITRE ATT&CK Technique**: if applicable (e.g., T1110 Brute Force)
5. **Recommended Response**: what to do next

Return as a JSON array of findings. Focus on actionable threats, not noise."""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return json.loads(match.group(0))
        return []


# Example: Analyze firewall and auth logs
security_logs = [
    "Jan 15 05:00:15 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99 dst inside:10.0.1.50/22",
    "Jan 15 05:00:16 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99 dst inside:10.0.1.51/22",
    "Jan 15 05:00:17 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99 dst inside:10.0.1.52/22",
    "Jan 15 05:00:18 firewall-01: %ASA-4-106023: Deny tcp src outside:198.51.100.99 dst inside:10.0.1.53/22",
    "Jan 15 05:01:00 router-core-01: %SEC-6-IPACCESSLOGP: list VTY_ACCESS denied tcp 10.0.5.200(54321) -> 10.0.0.1(22), 15 packets",
    "Jan 15 05:01:30 router-core-01: %SEC-6-IPACCESSLOGP: list VTY_ACCESS denied tcp 10.0.5.200(54322) -> 10.0.0.2(22), 12 packets",
    "Jan 15 05:02:00 switch-dist-01: %AUTHMGR-5-FAIL: Authorization failed for client (MAC 00:11:22:33:44:55) on Gi1/0/10",
    "Jan 15 05:02:05 switch-dist-01: %AUTHMGR-5-FAIL: Authorization failed for client (MAC 00:11:22:33:44:55) on Gi1/0/10",
    "Jan 15 05:02:10 switch-dist-01: %AUTHMGR-5-FAIL: Authorization failed for client (MAC 00:11:22:33:44:55) on Gi1/0/10",
]

findings = analyze_security_logs(security_logs)
for finding in findings:
    print(f"\n[{finding.get('confidence', '?')} confidence] "
          f"{finding.get('threat_category', 'unknown')}")
    print(f"  Evidence: {finding.get('evidence', 'N/A')}")
    print(f"  Response: {finding.get('recommended_response', 'Investigate')}")
```

---

## Use Case 4: Trend Detection

Instead of looking at individual log entries, AI can identify trends over time — things like gradual memory degradation, increasing error rates, or slowly rising CPU utilization.

```python
def detect_trends(
    log_summaries: List[Dict],
    lookback_days: int = 7
) -> List[Dict]:
    """Detect concerning trends across multiple days of log data.

    Like looking at MRTG/Cacti graphs for long-term trends, but
    for log patterns instead of interface utilization.
    """

    summary_text = json.dumps(log_summaries, indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze these daily log summaries from the past {lookback_days} days
and identify concerning trends.

Daily summaries:
{summary_text}

Look for:
1. Increasing error rates (e.g., more BGP flaps each day)
2. Gradual resource degradation (e.g., rising memory warnings)
3. New patterns that weren't present earlier (e.g., new source IPs in deny logs)
4. Recurring issues at specific times (e.g., daily at 2 AM)
5. Correlated changes across devices (e.g., multiple devices showing same issue)

For each trend, assess:
- severity: critical/warning/informational
- trend_direction: increasing/decreasing/periodic/new
- prediction: what will happen if this continues
- recommended_action: what to do about it

Return as a JSON array of trend objects."""
        }]
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', text)
        return json.loads(match.group(0)) if match else []


# Example: Weekly trend analysis
daily_summaries = [
    {"date": "Jan 09", "bgp_flaps": 0, "memory_warnings": 2, "auth_failures": 5, "interface_errors": 3},
    {"date": "Jan 10", "bgp_flaps": 1, "memory_warnings": 3, "auth_failures": 4, "interface_errors": 5},
    {"date": "Jan 11", "bgp_flaps": 0, "memory_warnings": 5, "auth_failures": 6, "interface_errors": 4},
    {"date": "Jan 12", "bgp_flaps": 2, "memory_warnings": 8, "auth_failures": 15, "interface_errors": 3},
    {"date": "Jan 13", "bgp_flaps": 1, "memory_warnings": 12, "auth_failures": 22, "interface_errors": 6},
    {"date": "Jan 14", "bgp_flaps": 3, "memory_warnings": 18, "auth_failures": 45, "interface_errors": 5},
    {"date": "Jan 15", "bgp_flaps": 5, "memory_warnings": 25, "auth_failures": 78, "interface_errors": 4},
]

trends = detect_trends(daily_summaries)
for trend in trends:
    print(f"[{trend.get('severity', '?')}] {trend.get('description', 'Unknown trend')}")
    print(f"  Direction: {trend.get('trend_direction', '?')}")
    print(f"  Prediction: {trend.get('prediction', 'N/A')}")
    print(f"  Action: {trend.get('recommended_action', 'Monitor')}")
    print()
```

---

## Building a Complete Log Analysis System

Here's how all the pieces fit together:

```python
class NetworkLogAnalyzer:
    """Complete AI-powered log analysis system.

    Combines summarization, root cause analysis, security analysis,
    and trend detection into a unified system.

    Think of this as your AI-powered NOC assistant — it watches
    the logs, identifies what matters, and tells you what to do.
    """

    def __init__(self):
        self.daily_summaries = []

    def morning_briefing(self, overnight_logs: List[str]) -> str:
        """Generate a morning briefing from overnight logs.

        The NOC equivalent of 'what happened while I was sleeping?'
        """
        return summarize_network_logs(overnight_logs, "overnight (00:00-08:00)")

    def investigate_incident(
        self,
        incident: str,
        related_logs: List[str]
    ) -> Dict:
        """Investigate a specific incident with root cause analysis."""
        return analyze_root_cause(related_logs, incident)

    def security_review(self, security_logs: List[str]) -> list:
        """Review security logs for threats."""
        return analyze_security_logs(security_logs)

    def weekly_trends(self) -> list:
        """Analyze trends from accumulated daily summaries."""
        if len(self.daily_summaries) < 3:
            return [{"note": "Need at least 3 days of data for trend analysis"}]
        return detect_trends(self.daily_summaries)

    def record_daily_summary(self, summary: Dict):
        """Record daily stats for trend analysis."""
        self.daily_summaries.append(summary)
        # Keep last 30 days
        self.daily_summaries = self.daily_summaries[-30:]
```

---

## Cost Considerations

| Analysis Type | Model | Tokens per Call | Cost per Call |
|---------------|-------|-----------------|---------------|
| Log summarization | Claude Sonnet | ~2,000 | ~$0.01 |
| Root cause analysis | Claude Sonnet | ~3,000 | ~$0.02 |
| Security analysis | Claude Sonnet | ~2,500 | ~$0.015 |
| Trend detection | Claude Sonnet | ~2,000 | ~$0.01 |
| Log classification (batch) | Claude Haiku | ~1,000 | ~$0.001 |

**Daily cost estimate** (500-device network):
- 4 log summaries/day: ~$0.04
- 2-3 RCA investigations: ~$0.06
- 1 security review: ~$0.015
- Classification (100 batches): ~$0.10
- **Total: ~$0.20-0.50/day**

That's less than $15/month for AI-powered network intelligence.

---

## Best Practices

### 1. Start Simple
Don't try to analyze every log with AI on day one. Start with:
- Morning briefing (summarize overnight logs)
- On-demand RCA (when incidents happen)
- Weekly security review

### 2. Build Feedback Loops
Track which AI findings were accurate:
- Was the root cause correct?
- Were security findings confirmed?
- Were trend predictions validated?

Use this feedback to refine your prompts over time.

### 3. Combine AI with Traditional Tools
AI doesn't replace your syslog server, SIEM, or NMS. It adds an intelligence layer on top:
- Syslog-ng collects and stores logs (fast, reliable)
- Grafana/Kibana visualizes patterns (visual, interactive)
- Claude analyzes and explains (intelligent, contextual)

### 4. Sanitize Before Sending
Always review what you're sending to external APIs:
- Mask internal IPs if required by security policy
- Strip passwords and community strings
- Aggregate rather than send raw logs when possible

---

## Key Takeaways

1. **Log summarization** turns thousands of entries into a shift handoff report — saving engineers from reading raw logs.

2. **Root cause analysis** correlates log timelines to identify why incidents happened, not just what happened.

3. **Security analysis** triages threats with MITRE ATT&CK mapping and confidence scoring.

4. **Trend detection** identifies developing problems before they become outages.

5. **Cost is minimal** — AI-powered log intelligence costs less than $15/month for most networks.

---

## What's Next

In **Chapter 25**, we'll use AI for anomaly detection — building systems that learn what "normal" looks like for your network and automatically flag deviations.
