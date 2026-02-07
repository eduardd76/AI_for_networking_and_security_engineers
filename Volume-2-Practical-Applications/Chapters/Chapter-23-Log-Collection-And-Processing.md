# Chapter 23: Log Collection and Processing

## Why Logs Matter (More Than You Think)

Every network device generates logs — syslog messages, SNMP traps, authentication events, routing protocol state changes. These logs are the black box recorder of your network. When something goes wrong at 3 AM, logs are how you figure out what happened.

The problem? A mid-sized network generates millions of log entries per day. No human can read them all. Most teams either ignore logs until there's an incident (reactive), or build brittle regex-based alerting rules that either miss real issues or cry wolf constantly.

**Networking analogy**: Think of your current logging like running `debug all` on a router — you get everything, but you can't make sense of any of it. What you need is targeted logging with intelligent filtering, like `debug ip bgp updates` — focused on what matters.

AI changes this equation. Instead of writing regex rules for every possible log pattern, you can use LLMs to understand log *meaning* and classify entries by severity, category, and required action.

---

## Log Collection Architecture

Before AI can analyze logs, you need to collect them reliably. Here's the standard architecture:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Network    │    │   Log        │    │   AI         │
│   Devices    │───▶│   Collector  │───▶│   Analysis   │
│              │    │   (Syslog)   │    │   Pipeline   │
└──────────────┘    └──────────────┘    └──────────────┘
  Routers            Syslog-ng          Claude API
  Switches           Fluentd            Classification
  Firewalls          Logstash           Correlation
  APs                Vector/Graylog     Alerting
```

### Collecting Syslog Messages

```python
import socket
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

class NetworkLogCollector:
    """Collect and parse syslog messages from network devices.

    In production, you'd use syslog-ng, rsyslog, or Fluentd.
    This simplified collector demonstrates the concepts for
    understanding what happens before AI analysis.
    """

    # Syslog severity levels (RFC 5424) — network engineers know these
    SEVERITY_MAP = {
        0: "emergency",    # System unusable (like a chassis failure)
        1: "alert",        # Immediate action needed
        2: "critical",     # Critical condition (e.g., hardware error)
        3: "error",        # Error condition (e.g., interface down)
        4: "warning",      # Warning (e.g., high CPU)
        5: "notice",       # Normal but significant (e.g., neighbor up)
        6: "informational", # Informational (e.g., config change)
        7: "debug"         # Debug messages
    }

    # Syslog facility codes relevant to networking
    FACILITY_MAP = {
        0: "kernel",
        4: "auth",
        10: "security",
        16: "local0",  # Often used for network devices
        23: "local7"   # Cisco default
    }

    def __init__(self, listen_port: int = 514):
        self.port = listen_port
        self.log_buffer = []

    def parse_syslog_message(self, raw_message: str) -> Dict:
        """Parse a raw syslog message into structured data.

        Handles common formats from Cisco IOS, NX-OS, and Junos.

        Example input:
        <189>: Jan 15 10:23:45 EST: %LINEPROTO-5-UPDOWN:
        Line protocol on Interface GigabitEthernet0/1, changed state to up
        """
        parsed = {
            "raw": raw_message,
            "timestamp": datetime.now().isoformat(),
            "hostname": "unknown",
            "severity": "informational",
            "facility": "local7",
            "message": raw_message
        }

        # Extract priority value <PRI>
        pri_match = re.match(r'<(\d+)>', raw_message)
        if pri_match:
            pri = int(pri_match.group(1))
            severity_num = pri % 8
            facility_num = pri // 8
            parsed["severity"] = self.SEVERITY_MAP.get(severity_num, "unknown")
            parsed["facility"] = self.FACILITY_MAP.get(facility_num, f"local{facility_num}")

        # Extract Cisco IOS-style mnemonic (%FACILITY-SEV-MNEMONIC)
        mnemonic_match = re.search(r'%(\w+)-(\d)-(\w+)', raw_message)
        if mnemonic_match:
            parsed["cisco_facility"] = mnemonic_match.group(1)
            parsed["cisco_severity"] = int(mnemonic_match.group(2))
            parsed["cisco_mnemonic"] = mnemonic_match.group(3)

        # Extract hostname (if present after timestamp)
        host_match = re.search(
            r'(?::\s+)?(\w[\w.-]+)\s+(?:%|\w)', raw_message
        )
        if host_match:
            parsed["hostname"] = host_match.group(1)

        return parsed

    def collect_sample_logs(self) -> List[Dict]:
        """Return sample network logs for demonstration.

        These represent real-world log patterns from a production network.
        """
        sample_logs = [
            "<189>: Jan 15 10:23:45 router-core-01: %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to up",
            "<187>: Jan 15 10:23:46 router-core-01: %BGP-3-NOTIFICATION: received from neighbor 203.0.113.2 4/0 (hold time expired)",
            "<189>: Jan 15 10:23:47 switch-dist-01: %LINK-3-UPDOWN: Interface GigabitEthernet1/0/24, changed state to down",
            "<190>: Jan 15 10:24:01 firewall-01: %ASA-6-302013: Built inbound TCP connection for inside:10.0.1.100/443 to outside:198.51.100.50/54321",
            "<186>: Jan 15 10:24:15 router-core-01: %SYS-2-MALLOCFAIL: Memory allocation of 65536 bytes failed",
            "<189>: Jan 15 10:24:30 switch-access-01: %AUTHMGR-5-SUCCESS: Authorization succeeded for client (10.0.10.100) on Interface Gi1/0/5",
            "<188>: Jan 15 10:24:45 router-core-02: %OSPF-4-ERRRCV: OSPF received invalid packet: mismatch area ID, from backbone area must be virtual-link but not found from 10.0.1.2, GigabitEthernet0/1",
            "<189>: Jan 15 10:25:00 switch-dist-01: %STORM_CONTROL-3-FILTERED: A packet storm was detected on Gi1/0/15",
            "<190>: Jan 15 10:25:15 router-core-01: %SEC-6-IPACCESSLOGP: list MGMT_ACCESS denied tcp 192.168.1.50(22) -> 10.0.0.1(22), 3 packets",
            "<187>: Jan 15 10:25:30 router-core-01: %TRACKING-5-STATE: 1 ip sla 1 reachability Down -> Up",
        ]

        return [self.parse_syslog_message(log) for log in sample_logs]
```

---

## AI-Powered Log Classification

Now for the interesting part — using Claude to classify and understand log messages. Instead of writing hundreds of regex rules, you describe what you want in natural language.

**Networking analogy**: Traditional log parsing is like static ACLs — you need an explicit rule for every pattern. AI-powered classification is like a next-gen firewall with application awareness — it understands the *intent* of the log message, not just its syntax.

```python
import os
import json
import re
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY environment variable

def classify_log_batch(logs: List[Dict], batch_size: int = 20) -> List[Dict]:
    """Classify a batch of network log entries using Claude.

    Sends logs in batches to minimize API calls while keeping
    context manageable. Returns each log enriched with:
    - category: routing, security, hardware, interface, auth, config
    - urgency: critical, high, medium, low, informational
    - action_needed: what to do about it (or 'monitor')
    - related_device: which device is primarily affected
    """

    classified = []

    for i in range(0, len(logs), batch_size):
        batch = logs[i:i + batch_size]
        log_text = "\n".join([
            f"[{j+1}] {log['raw']}" for j, log in enumerate(batch)
        ])

        response = client.messages.create(
            model="claude-haiku-4-20250514",  # Haiku is fast + cheap for classification
            max_tokens=2000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Classify these network device log entries. For each entry,
provide a JSON object with these fields:
- id: the entry number
- category: one of [routing, security, hardware, interface, auth, config, system]
- urgency: one of [critical, high, medium, low, informational]
- summary: one-sentence plain English explanation
- action_needed: recommended action or "monitor" if no action needed
- related_protocol: the networking protocol involved (BGP, OSPF, STP, etc.) or "none"

Log entries:
{log_text}

Return ONLY a JSON array of classification objects."""
            }]
        )

        text = response.content[0].text.strip()
        try:
            classifications = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting from code fences
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                classifications = json.loads(match.group(1))
            else:
                start = text.find('[')
                end = text.rfind(']')
                classifications = json.loads(text[start:end+1])

        # Merge classifications back into log entries
        for cls in classifications:
            idx = cls.get('id', 0) - 1
            if 0 <= idx < len(batch):
                batch[idx].update(cls)
                classified.append(batch[idx])

    return classified


# Example usage
collector = NetworkLogCollector()
logs = collector.collect_sample_logs()
classified = classify_log_batch(logs)

for log in classified:
    urgency = log.get('urgency', 'unknown')
    category = log.get('category', 'unknown')
    summary = log.get('summary', log['raw'][:80])
    print(f"[{urgency:>13}] [{category:>10}] {summary}")
```

### Expected Output

```
[     critical] [    system] Memory allocation failure on router-core-01 — potential out-of-memory condition
[         high] [   routing] BGP session with 203.0.113.2 dropped due to hold timer expiration
[         high] [ interface] Interface Gi1/0/24 on switch-dist-01 went down
[         high] [ interface] Broadcast storm detected on switch-dist-01 port Gi1/0/15
[       medium] [   routing] OSPF area ID mismatch received on router-core-02 Gi0/1
[       medium] [  security] Management access denied from unauthorized source 192.168.1.50
[          low] [ interface] Interface Gi0/1 on router-core-01 came back up
[          low] [   routing] IP SLA tracking object recovered — primary path restored
[informational] [  security] Firewall built new inbound TCP connection (normal traffic)
[informational] [      auth] 802.1X authentication succeeded for client on switch-access-01
```

---

## Log Correlation: Connecting the Dots

Individual log messages tell you *what* happened. Correlation tells you *why*. When you see a BGP hold timer expiration followed by an interface down event followed by IP SLA state change, those aren't three separate events — they're one story: a link failure caused a cascade.

**Networking analogy**: Individual logs are like individual SNMP traps. Log correlation is like an NMS (Network Management System) that groups related events into a single incident — "Interface down" + "BGP session reset" + "route withdrawn" = "WAN link failure."

```python
def correlate_log_events(
    classified_logs: List[Dict],
    time_window_seconds: int = 300
) -> List[Dict]:
    """Group related log events into correlated incidents.

    Looks for patterns like:
    - Interface down → BGP session drop → route change
    - Auth failures from same source → potential brute force
    - Multiple device errors → shared infrastructure issue

    Returns a list of 'incidents' — grouped, correlated events.
    """

    log_text = "\n".join([
        f"[{log.get('timestamp', 'N/A')}] [{log.get('hostname', '?')}] "
        f"[{log.get('category', '?')}] {log.get('summary', log['raw'][:100])}"
        for log in classified_logs
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # Sonnet for complex reasoning
        max_tokens=2000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Analyze these network log events and identify correlated incidents.
Group related events that are likely caused by the same root issue.

For each incident, provide:
- incident_id: sequential number
- title: short description of the incident
- root_cause: your best assessment of what caused it
- affected_devices: list of affected device hostnames
- severity: critical/high/medium/low
- related_log_indices: which log entries belong to this incident
- recommended_actions: list of steps to investigate or resolve

Log events:
{log_text}

Return ONLY a JSON array of incident objects."""
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


# Example: Correlate the classified logs
incidents = correlate_log_events(classified)

for incident in incidents:
    print(f"\n{'='*60}")
    print(f"Incident: {incident.get('title', 'Unknown')}")
    print(f"Severity: {incident.get('severity', 'Unknown')}")
    print(f"Root Cause: {incident.get('root_cause', 'Under investigation')}")
    print(f"Affected: {', '.join(incident.get('affected_devices', []))}")
    print(f"Actions: ")
    for action in incident.get('recommended_actions', []):
        print(f"  - {action}")
```

---

## Building a Log Processing Pipeline

Here's a complete pipeline that collects, classifies, correlates, and alerts on network logs:

```python
from datetime import datetime, timedelta

class NetworkLogPipeline:
    """End-to-end log processing pipeline with AI analysis.

    Flow:
    1. Collect raw syslog messages
    2. Parse into structured format
    3. Classify with AI (category, urgency)
    4. Correlate related events into incidents
    5. Alert on high-severity incidents

    Networking analogy: This is like building an NMS (SolarWinds,
    LibreNMS) but with AI understanding instead of static thresholds.
    """

    def __init__(self):
        self.collector = NetworkLogCollector()
        self.log_buffer = []
        self.incidents = []
        self.alert_thresholds = {
            "critical": 0,   # Alert immediately
            "high": 3,       # Alert after 3 occurrences in window
            "medium": 10,    # Alert after 10 occurrences
            "low": 0         # Don't alert, just log
        }

    def process_logs(self, raw_logs: List[str]) -> Dict:
        """Process a batch of raw log messages through the full pipeline."""

        print(f"\n{'='*50}")
        print(f"Log Pipeline Run: {datetime.now()}")
        print(f"Processing {len(raw_logs)} log entries")
        print(f"{'='*50}")

        # Step 1: Parse raw logs
        parsed = [self.collector.parse_syslog_message(log) for log in raw_logs]
        print(f"  Parsed: {len(parsed)} entries")

        # Step 2: Classify with AI
        classified = classify_log_batch(parsed)
        print(f"  Classified: {len(classified)} entries")

        # Step 3: Summarize by category and urgency
        summary = self._summarize(classified)

        # Step 4: Correlate into incidents
        high_priority = [
            log for log in classified
            if log.get('urgency') in ('critical', 'high', 'medium')
        ]

        if high_priority:
            incidents = correlate_log_events(high_priority)
            self.incidents.extend(incidents)
            print(f"  Incidents detected: {len(incidents)}")
        else:
            incidents = []
            print(f"  No incidents detected")

        # Step 5: Check alert thresholds
        alerts = self._check_alerts(classified)

        return {
            "total_logs": len(raw_logs),
            "summary": summary,
            "incidents": incidents,
            "alerts": alerts,
            "processed_at": datetime.now().isoformat()
        }

    def _summarize(self, classified: List[Dict]) -> Dict:
        """Summarize classified logs by category and urgency."""
        summary = {}
        for log in classified:
            cat = log.get('category', 'unknown')
            urg = log.get('urgency', 'unknown')
            key = f"{cat}/{urg}"
            summary[key] = summary.get(key, 0) + 1
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    def _check_alerts(self, classified: List[Dict]) -> List[Dict]:
        """Check if any logs exceed alerting thresholds."""
        alerts = []
        for log in classified:
            urgency = log.get('urgency', 'low')
            threshold = self.alert_thresholds.get(urgency, 999)

            if urgency == 'critical':
                alerts.append({
                    "type": "CRITICAL",
                    "message": log.get('summary', log['raw'][:100]),
                    "device": log.get('hostname', 'unknown'),
                    "action": log.get('action_needed', 'Investigate immediately')
                })

        return alerts


# Run the pipeline
pipeline = NetworkLogPipeline()

# Sample logs (in production, these come from syslog listener)
raw_logs = [
    "<187>: Jan 15 10:23:46 router-core-01: %BGP-3-NOTIFICATION: received from neighbor 203.0.113.2 4/0 (hold time expired)",
    "<189>: Jan 15 10:23:47 switch-dist-01: %LINK-3-UPDOWN: Interface GigabitEthernet1/0/24, changed state to down",
    "<186>: Jan 15 10:24:15 router-core-01: %SYS-2-MALLOCFAIL: Memory allocation of 65536 bytes failed",
    "<189>: Jan 15 10:25:00 switch-dist-01: %STORM_CONTROL-3-FILTERED: A packet storm was detected on Gi1/0/15",
    "<190>: Jan 15 10:25:15 router-core-01: %SEC-6-IPACCESSLOGP: list MGMT_ACCESS denied tcp 192.168.1.50(22) -> 10.0.0.1(22), 3 packets",
]

result = pipeline.process_logs(raw_logs)
```

---

## Log Storage Strategies

### What to Store and For How Long

| Log Type | Retention | Storage | Reason |
|----------|-----------|---------|--------|
| Security events | 1+ year | Searchable DB | Compliance (PCI-DSS, SOX) |
| Routing changes | 90 days | Time-series DB | Troubleshooting history |
| Interface events | 30 days | Time-series DB | Pattern detection |
| Auth successes | 7 days | Compressed files | Audit trail |
| Debug/info | 24 hours | Circular buffer | Real-time only |

### Cost-Effective AI Processing

Not every log needs AI analysis. Use a tiered approach:

```python
def should_analyze_with_ai(log: Dict) -> bool:
    """Determine if a log entry warrants AI analysis.

    Most logs are routine (interface up/down, auth success).
    Only send unusual or high-severity logs to the AI — this
    keeps costs manageable.

    Think of this as your 'rate-limit' for AI processing,
    like CoPP (Control Plane Policing) for your AI budget.
    """
    severity = log.get('cisco_severity', 6)

    # Always analyze: severity 0-3 (emergency through error)
    if severity <= 3:
        return True

    # Always analyze: security-related facilities
    security_facilities = {'SEC', 'AUTHMGR', 'SSH', 'RADIUS', 'TACACS'}
    if log.get('cisco_facility') in security_facilities:
        return True

    # Skip: routine informational/debug messages
    if severity >= 6:
        return False

    # Default: analyze warnings and notices
    return True
```

---

## Best Practices for Network Log Processing

### 1. Normalize Before Analysis
Different vendors format logs differently. Normalize them into a common format before sending to AI:

```python
def normalize_log(raw: str, vendor: str = "cisco_ios") -> Dict:
    """Normalize vendor-specific logs into a common format.

    Like translating between OSPF and EIGRP — different protocols,
    same underlying information. Normalize once, analyze consistently.
    """
    normalized = {
        "raw": raw,
        "vendor": vendor,
        "timestamp": None,
        "hostname": None,
        "severity": None,
        "message": None
    }

    if vendor == "cisco_ios":
        # Format: %FACILITY-SEV-MNEMONIC: message
        match = re.search(r'(\S+):\s+%(\w+)-(\d)-(\w+):\s+(.*)', raw)
        if match:
            normalized["hostname"] = match.group(1)
            normalized["facility"] = match.group(2)
            normalized["severity"] = int(match.group(3))
            normalized["mnemonic"] = match.group(4)
            normalized["message"] = match.group(5)

    elif vendor == "junos":
        # Format: hostname process[pid]: FACILITY_MNEMONIC: message
        match = re.search(r'(\S+)\s+\w+\[\d+\]:\s+(\w+):\s+(.*)', raw)
        if match:
            normalized["hostname"] = match.group(1)
            normalized["mnemonic"] = match.group(2)
            normalized["message"] = match.group(3)

    elif vendor == "paloalto":
        # CSV format logs
        fields = raw.split(',')
        if len(fields) > 5:
            normalized["hostname"] = fields[1] if len(fields) > 1 else None
            normalized["message"] = raw

    return normalized
```

### 2. Rate-Limit AI Calls
Don't send every log to Claude. Use local pre-filtering to keep costs down:
- Filter by severity (only errors and above)
- Deduplicate repeating messages (e.g., interface flapping)
- Batch logs in 5-minute windows for correlation efficiency

### 3. Protect Sensitive Data
Before sending logs to any external API, sanitize sensitive information:
- Remove or mask IP addresses of internal hosts if required by policy
- Strip usernames from authentication logs
- Never send cryptographic keys, passwords, or community strings

---

## Key Takeaways

1. **Structured collection** comes first — normalize syslog from all vendors into a common format before AI analysis.

2. **AI classification** replaces hundreds of regex rules with a single prompt that understands log *meaning*, not just patterns.

3. **Log correlation** groups related events into incidents — turning noise into actionable intelligence.

4. **Tiered processing** keeps costs down — only send unusual logs to the AI, like CoPP for your AI budget.

5. **Pipeline architecture** connects collection, classification, correlation, and alerting into a continuous workflow.

---

## What's Next

In **Chapter 24**, we'll go deeper into AI-powered log analysis — building models that learn your network's normal behavior and flag deviations that matter.
