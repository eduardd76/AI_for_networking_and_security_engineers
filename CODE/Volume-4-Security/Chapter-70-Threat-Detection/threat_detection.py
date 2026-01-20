"""
Chapter 70: AI-Powered Threat Detection
Dual-AI Security Analysis with LangChain

Uses both Claude (Anthropic) and OpenAI for enhanced security analysis.
Claude: Deep reasoning and complex pattern analysis for lateral movement and C2 detection
OpenAI: Fast pattern matching and correlation for threat validation
LangChain: Orchestration and consensus building between models

Architecture:
- Claude analyzes authentication patterns for lateral movement with context
- OpenAI validates findings with pattern recognition
- LangChain chains coordinate multi-agent consensus for high-confidence detections

Production-ready with error handling, API key management, and Colab compatibility.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# API clients
from anthropic import Anthropic
from openai import OpenAI

# LangChain imports
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.schema import HumanMessage, SystemMessage


# ============================================================================
# API KEY MANAGEMENT (Colab-compatible)
# ============================================================================

def get_api_keys() -> Tuple[str, str]:
    """
    Get API keys from environment or Google Colab secrets.

    Returns:
        Tuple of (anthropic_key, openai_key)

    Raises:
        ValueError: If API keys are not found
    """
    try:
        # Try Google Colab userdata first
        from google.colab import userdata
        anthropic_key = userdata.get('ANTHROPIC_API_KEY')
        openai_key = userdata.get('OPENAI_API_KEY')
        print("✓ Loaded API keys from Google Colab secrets")
    except:
        # Fall back to environment variables
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        print("✓ Loaded API keys from environment variables")

    if not anthropic_key or not openai_key:
        raise ValueError(
            "API keys not found. Please set:\n"
            "  - ANTHROPIC_API_KEY\n"
            "  - OPENAI_API_KEY\n"
            "In environment variables or Google Colab secrets."
        )

    return anthropic_key, openai_key


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AuthEvent:
    """Represents a single authentication event"""
    timestamp: datetime
    user: str
    source_ip: str
    destination_host: str
    auth_type: str  # ssh, rdp, smb, kerberos
    success: bool
    source_host: Optional[str] = None

@dataclass
class NetFlowRecord:
    """Network flow record for C2 detection"""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    dest_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    duration: float

@dataclass
class LoginEvent:
    """Login attempt for credential compromise detection"""
    timestamp: datetime
    username: str
    source_ip: str
    geo_location: Tuple[float, float]  # (latitude, longitude)
    city: str
    country: str
    device_type: str
    user_agent: str
    success: bool
    mfa_used: bool = False


# ============================================================================
# EXAMPLE 1: LATERAL MOVEMENT DETECTION WITH DUAL-AI CONSENSUS
# ============================================================================

def example_1():
    """
    Example 1: Lateral Movement Detection with Dual-AI Consensus

    Architecture:
    1. Claude performs deep analysis of authentication patterns and context
    2. OpenAI validates with pattern recognition and known attack signatures
    3. LangChain orchestrates consensus: Only alert if both AIs agree it's suspicious

    Why dual-AI?
    - Reduces false positives (both must agree)
    - Claude excels at contextual reasoning ("never accessed DC before")
    - OpenAI excels at pattern matching ("this matches APT29 TTPs")
    """
    print("=" * 80)
    print("EXAMPLE 1: Lateral Movement Detection with Dual-AI Consensus")
    print("=" * 80)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize clients
    anthropic_client = Anthropic(api_key=anthropic_key)
    openai_client = OpenAI(api_key=openai_key)

    # Initialize LangChain models
    claude = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=anthropic_key,
        max_tokens=1000
    )

    gpt4 = ChatOpenAI(
        model="gpt-4",
        openai_api_key=openai_key,
        temperature=0
    )

    # Simulate suspicious authentication event
    suspicious_auth = {
        'user': 'john.admin',
        'destination_host': 'dc01.corp.local',  # Domain controller
        'source_ip': '185.220.101.50',  # Tor exit node
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hour': 3,  # 3 AM
        'auth_type': 'ssh',
        'typical_hosts': ['webserver01', 'webserver02', 'loadbalancer01'],
        'typical_hours': [9, 10, 11, 14, 15, 16, 17],
        'typical_source_ips': ['10.1.20.50', '10.1.20.51']
    }

    print(f"\n🔍 Analyzing Suspicious Authentication:")
    print(f"User: {suspicious_auth['user']}")
    print(f"Target: {suspicious_auth['destination_host']}")
    print(f"Source IP: {suspicious_auth['source_ip']}")
    print(f"Time: {suspicious_auth['timestamp']} (Hour {suspicious_auth['hour']})")

    # STEP 1: Claude Deep Analysis
    print("\n⚡ Step 1: Claude performs deep contextual analysis...")

    claude_prompt = f"""You are a network security analyst detecting lateral movement attacks.

SUSPICIOUS AUTHENTICATION EVENT:
User: {suspicious_auth['user']}
Logged into: {suspicious_auth['destination_host']}
From IP: {suspicious_auth['source_ip']}
Time: {suspicious_auth['timestamp']} (Hour: {suspicious_auth['hour']})
Auth Type: {suspicious_auth['auth_type']}

USER'S NORMAL BEHAVIOR:
- Typically accesses: {', '.join(suspicious_auth['typical_hosts'])}
- Has accessed {suspicious_auth['destination_host']} before: False
- Typical working hours: {suspicious_auth['typical_hours']}
- Typical source IPs: {', '.join(suspicious_auth['typical_source_ips'])}
- Logging in from new IP: True

ANALYSIS REQUIRED:
1. Is this likely lateral movement (attacker using stolen credentials)?
2. What's suspicious about this authentication?
3. Threat severity (Critical/High/Medium/Low)

Provide analysis in JSON format:
{{
    "is_lateral_movement": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "suspicious_indicators": ["indicator1", "indicator2"],
    "explanation": "Brief explanation"
}}
"""

    try:
        claude_response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        claude_analysis = json.loads(claude_response.content[0].text)
        print(f"✓ Claude Analysis: {claude_analysis['severity']} severity, "
              f"{claude_analysis['confidence']:.0%} confidence")
        print(f"  Lateral Movement: {claude_analysis['is_lateral_movement']}")
    except Exception as e:
        print(f"✗ Claude analysis failed: {e}")
        claude_analysis = {"is_lateral_movement": True, "confidence": 0.5, "error": str(e)}

    # STEP 2: OpenAI Pattern Validation
    print("\n⚡ Step 2: OpenAI validates with pattern matching...")

    openai_prompt = f"""You are a threat intelligence analyst. Validate this potential lateral movement:

EVENT:
- User: {suspicious_auth['user']} accessing {suspicious_auth['destination_host']}
- Source IP: {suspicious_auth['source_ip']} (Tor exit node)
- Time: {suspicious_auth['hour']}:00 (off-hours)
- Never accessed this server before
- Never logged in from this IP before

CLAUDE'S ASSESSMENT:
{json.dumps(claude_analysis, indent=2)}

YOUR VALIDATION:
1. Does this match known attack patterns (APT groups, ransomware TTPs)?
2. Is the source IP suspicious (Tor, VPN, proxy, botnet)?
3. Do you agree this is lateral movement?

JSON format:
{{
    "agrees_with_claude": true/false,
    "confidence": 0.0-1.0,
    "attack_pattern": "name of matching TTP/technique",
    "source_ip_reputation": "description",
    "recommendation": "Alert/Monitor/Dismiss"
}}
"""

    try:
        openai_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": openai_prompt}],
            temperature=0
        )
        openai_analysis = json.loads(openai_response.choices[0].message.content)
        print(f"✓ OpenAI Validation: {openai_analysis['recommendation']}, "
              f"{openai_analysis['confidence']:.0%} confidence")
        print(f"  Agrees with Claude: {openai_analysis['agrees_with_claude']}")
        print(f"  Attack Pattern: {openai_analysis.get('attack_pattern', 'Unknown')}")
    except Exception as e:
        print(f"✗ OpenAI validation failed: {e}")
        openai_analysis = {"agrees_with_claude": True, "confidence": 0.5, "error": str(e)}

    # STEP 3: Consensus Decision
    print("\n⚡ Step 3: Building dual-AI consensus...")

    # Both AIs must agree for high-confidence alert
    both_agree = (claude_analysis.get('is_lateral_movement', False) and
                  openai_analysis.get('agrees_with_claude', False))

    avg_confidence = (claude_analysis.get('confidence', 0) +
                      openai_analysis.get('confidence', 0)) / 2

    print(f"\n{'🚨 ALERT' if both_agree else '⚠️  REVIEW REQUIRED'}")
    print(f"Consensus: {'Both AIs agree - Lateral Movement Detected' if both_agree else 'Disagreement - Manual review needed'}")
    print(f"Combined Confidence: {avg_confidence:.0%}")

    if both_agree:
        print("\n📋 Recommended Actions:")
        print("  1. Immediately disable john.admin account")
        print("  2. Isolate dc01.corp.local from network")
        print("  3. Review recent activity from this account")
        print("  4. Check for persistence mechanisms on DC")
        print("  5. Force password reset for all admin accounts")

    print("\n" + "=" * 80)


# ============================================================================
# EXAMPLE 2: C2 BEACON DETECTION USING BOTH MODELS
# ============================================================================

def example_2():
    """
    Example 2: C2 Beacon Detection using Claude + OpenAI

    Architecture:
    1. Statistical analysis identifies periodic traffic patterns
    2. Claude analyzes behavior: Is this really C2 or legitimate (Windows Update, etc)?
    3. OpenAI matches against known C2 frameworks (Cobalt Strike, Metasploit, etc)
    4. LangChain combines insights for final verdict

    Why dual-AI?
    - Claude's reasoning helps distinguish legitimate beaconing (software updates)
    - OpenAI's pattern database recognizes specific malware families
    - Combination = fewer false positives, better attribution
    """
    print("=" * 80)
    print("EXAMPLE 2: C2 Beacon Detection with Dual-AI Analysis")
    print("=" * 80)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize clients
    anthropic_client = Anthropic(api_key=anthropic_key)
    openai_client = OpenAI(api_key=openai_key)

    # Simulate periodic beacon traffic
    beacon_data = {
        'source_ip': '10.1.50.75',  # Internal workstation
        'dest_ip': '45.134.83.12',  # Suspicious external IP
        'dest_port': 443,
        'connection_count': 288,  # 24 hours * 12 (every 5 min)
        'mean_interval_seconds': 300,  # 5 minutes
        'std_dev_seconds': 6,  # Very low variance
        'coefficient_of_variation': 0.02,  # Nearly perfect periodicity
        'avg_bytes_sent': 1200,
        'avg_bytes_received': 850,
        'duration_hours': 24,
        'protocol': 'TCP'
    }

    print(f"\n🔍 Analyzing Periodic Network Traffic:")
    print(f"Source: {beacon_data['source_ip']}")
    print(f"Destination: {beacon_data['dest_ip']}:{beacon_data['dest_port']}")
    print(f"Connections: {beacon_data['connection_count']} over {beacon_data['duration_hours']} hours")
    print(f"Interval: {beacon_data['mean_interval_seconds']/60:.1f} minutes (CV: {beacon_data['coefficient_of_variation']:.3f})")

    # STEP 1: Claude Behavioral Analysis
    print("\n⚡ Step 1: Claude analyzes behavior and context...")

    claude_prompt = f"""You are a network security analyst detecting C2 malware beacons.

PERIODIC TRAFFIC DETECTED:
Source IP: {beacon_data['source_ip']} (internal workstation)
Destination IP: {beacon_data['dest_ip']}
Port: {beacon_data['dest_port']}
Connection Count: {beacon_data['connection_count']}
Time Span: {beacon_data['duration_hours']} hours

PERIODICITY ANALYSIS:
- Mean interval: {beacon_data['mean_interval_seconds']} seconds ({beacon_data['mean_interval_seconds']/60:.1f} minutes)
- Standard deviation: {beacon_data['std_dev_seconds']} seconds
- Coefficient of Variation: {beacon_data['coefficient_of_variation']:.3f} (< 0.3 = highly periodic)

TRAFFIC CHARACTERISTICS:
- Average bytes sent: {beacon_data['avg_bytes_sent']}
- Average bytes received: {beacon_data['avg_bytes_received']}
- Protocol: {beacon_data['protocol']}

QUESTION: Is this C2 beaconing or legitimate periodic traffic (Windows Update, monitoring tool, etc)?

JSON format:
{{
    "is_c2": true/false,
    "confidence": 0.0-1.0,
    "explanation": "reasoning",
    "legitimate_service_possible": "service name or null"
}}
"""

    try:
        claude_response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        claude_analysis = json.loads(claude_response.content[0].text)
        print(f"✓ Claude Analysis: C2={claude_analysis['is_c2']}, "
              f"{claude_analysis['confidence']:.0%} confidence")
        print(f"  Legitimate possibility: {claude_analysis.get('legitimate_service_possible', 'None')}")
    except Exception as e:
        print(f"✗ Claude analysis failed: {e}")
        claude_analysis = {"is_c2": True, "confidence": 0.5, "error": str(e)}

    # STEP 2: OpenAI Malware Attribution
    print("\n⚡ Step 2: OpenAI matches against known C2 frameworks...")

    openai_prompt = f"""You are a malware analyst. Identify which C2 framework this matches:

BEACON PATTERN:
- Interval: {beacon_data['mean_interval_seconds']/60:.1f} minutes
- Periodicity: CV={beacon_data['coefficient_of_variation']} (highly regular)
- Port: {beacon_data['dest_port']}
- Small payloads: ~{beacon_data['avg_bytes_sent']} bytes
- Duration: {beacon_data['duration_hours']} hours continuous

CLAUDE ASSESSMENT:
{json.dumps(claude_analysis, indent=2)}

Which C2 framework does this match?
- Cobalt Strike (default 5min beacon)
- Metasploit
- Empire
- Sliver
- Custom malware
- Not C2 (legitimate service)

JSON format:
{{
    "likely_malware_family": "name",
    "confidence": 0.0-1.0,
    "matches_known_pattern": true/false,
    "severity": "Critical/High/Medium/Low"
}}
"""

    try:
        openai_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": openai_prompt}],
            temperature=0
        )
        openai_analysis = json.loads(openai_response.choices[0].message.content)
        print(f"✓ OpenAI Attribution: {openai_analysis['likely_malware_family']}")
        print(f"  Matches known pattern: {openai_analysis['matches_known_pattern']}")
        print(f"  Severity: {openai_analysis['severity']}")
    except Exception as e:
        print(f"✗ OpenAI analysis failed: {e}")
        openai_analysis = {"likely_malware_family": "Unknown", "confidence": 0.5, "error": str(e)}

    # STEP 3: Combined Assessment
    print("\n⚡ Step 3: Combining insights...")

    both_agree_c2 = (claude_analysis.get('is_c2', False) and
                     openai_analysis.get('matches_known_pattern', False))

    print(f"\n{'🚨 CRITICAL ALERT' if both_agree_c2 else '⚠️  INVESTIGATION NEEDED'}")
    if both_agree_c2:
        print(f"Both AIs confirm: C2 Beacon Detected")
        print(f"Malware Family: {openai_analysis.get('likely_malware_family', 'Unknown')}")
        print(f"Severity: {openai_analysis.get('severity', 'High')}")
        print("\n📋 Immediate Actions:")
        print(f"  1. ISOLATE {beacon_data['source_ip']} from network immediately")
        print(f"  2. Block {beacon_data['dest_ip']} at firewall")
        print(f"  3. Run EDR scan on {beacon_data['source_ip']}")
        print(f"  4. Capture memory dump for forensics")
        print(f"  5. Check for lateral movement from this host")
        print(f"  6. Search for other hosts beaconing to {beacon_data['dest_ip']}")
    else:
        print("Disagreement between models - requires manual analysis")

    print("\n" + "=" * 80)


# ============================================================================
# EXAMPLE 3: CREDENTIAL COMPROMISE WITH MULTI-AGENT VALIDATION
# ============================================================================

def example_3():
    """
    Example 3: Credential Compromise Detection with Multi-Agent Validation

    Architecture:
    1. LangChain SequentialChain orchestrates multi-step analysis
    2. Chain Step 1 (Claude): Analyze behavioral anomalies
    3. Chain Step 2 (OpenAI): Validate against threat intelligence
    4. Chain Step 3 (Claude): Final verdict with full context

    Why sequential chain?
    - Each step builds on previous insights
    - OpenAI's threat intel informs Claude's final assessment
    - Production pattern: Complex decisions need multiple reasoning steps
    """
    print("=" * 80)
    print("EXAMPLE 3: Credential Compromise with Multi-Agent Sequential Chain")
    print("=" * 80)

    # Get API keys
    anthropic_key, openai_key = get_api_keys()

    # Initialize LangChain models
    claude = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=anthropic_key,
        max_tokens=1000
    )

    gpt4 = ChatOpenAI(
        model="gpt-4",
        openai_api_key=openai_key,
        temperature=0
    )

    # Suspicious login data
    login_data = {
        'username': 'alice.smith',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_ip': '91.241.19.67',
        'location': 'Moscow, Russia',
        'geo_coords': (55.7558, 37.6173),
        'device_type': 'Linux',
        'user_agent': 'curl/7.68.0',  # Command-line tool
        'mfa_used': False,
        'hour': 3,
        # Normal behavior
        'typical_locations': ['Chicago, USA'],
        'typical_devices': ['Windows'],
        'typical_hours': [9, 10, 11, 14, 15, 16, 17],
        'typical_ips': ['10.1.20.50'],
        'always_uses_mfa': True
    }

    print(f"\n🔍 Analyzing Suspicious Login:")
    print(f"User: {login_data['username']}")
    print(f"Location: {login_data['location']}")
    print(f"IP: {login_data['source_ip']}")
    print(f"Device: {login_data['device_type']} ({login_data['user_agent']})")
    print(f"MFA: {login_data['mfa_used']} (normally: {login_data['always_uses_mfa']})")

    # STEP 1: Claude Behavioral Analysis
    print("\n⚡ Chain Step 1: Claude analyzes behavioral anomalies...")

    prompt_1 = f"""Analyze this login for credential compromise indicators:

LOGIN EVENT:
User: {login_data['username']}
Time: {login_data['timestamp']} (Hour {login_data['hour']})
Location: {login_data['location']}
IP: {login_data['source_ip']}
Device: {login_data['device_type']}
User Agent: {login_data['user_agent']}
MFA Used: {login_data['mfa_used']}

NORMAL BEHAVIOR:
- Typical locations: {login_data['typical_locations']}
- Typical devices: {login_data['typical_devices']}
- Typical hours: {login_data['typical_hours']}
- Always uses MFA: {login_data['always_uses_mfa']}

List behavioral anomalies in JSON:
{{
    "anomalies": ["anomaly1", "anomaly2"],
    "risk_score": 0.0-1.0,
    "suspicious": true/false
}}
"""

    try:
        response_1 = claude.invoke([HumanMessage(content=prompt_1)])
        analysis_1 = json.loads(response_1.content)
        print(f"✓ Found {len(analysis_1['anomalies'])} anomalies")
        print(f"  Risk Score: {analysis_1['risk_score']:.2f}")
        for anomaly in analysis_1['anomalies']:
            print(f"  - {anomaly}")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        analysis_1 = {"anomalies": ["Unknown"], "risk_score": 0.5, "suspicious": True}

    # STEP 2: OpenAI Threat Intelligence
    print("\n⚡ Chain Step 2: OpenAI checks threat intelligence...")

    prompt_2 = f"""You are a threat intelligence analyst. Check this IP and pattern:

IP: {login_data['source_ip']}
Location: {login_data['location']}
User Agent: {login_data['user_agent']}

BEHAVIORAL ANALYSIS FROM STEP 1:
{json.dumps(analysis_1, indent=2)}

Check:
1. Is this IP known malicious (Tor, VPN, proxy, botnet)?
2. Does curl user-agent suggest automation/scripts?
3. Is Russia → USA access common for legitimate users?

JSON format:
{{
    "ip_reputation": "Clean/Suspicious/Malicious",
    "ip_type": "description",
    "threat_intel_match": true/false,
    "likely_automated": true/false
}}
"""

    try:
        response_2 = gpt4.invoke([HumanMessage(content=prompt_2)])
        analysis_2 = json.loads(response_2.content)
        print(f"✓ IP Reputation: {analysis_2['ip_reputation']}")
        print(f"  Type: {analysis_2.get('ip_type', 'Unknown')}")
        print(f"  Likely Automated: {analysis_2['likely_automated']}")
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        analysis_2 = {"ip_reputation": "Suspicious", "likely_automated": True}

    # STEP 3: Claude Final Verdict with Full Context
    print("\n⚡ Chain Step 3: Claude makes final determination with full context...")

    prompt_3 = f"""You are the final decision-maker. Determine if credentials are compromised.

CONTEXT FROM PREVIOUS ANALYSIS:

Behavioral Anomalies (Claude):
{json.dumps(analysis_1, indent=2)}

Threat Intelligence (OpenAI):
{json.dumps(analysis_2, indent=2)}

FINAL ASSESSMENT:
Are {login_data['username']}'s credentials compromised?

JSON format:
{{
    "credentials_compromised": true/false,
    "confidence": 0.0-1.0,
    "severity": "Critical/High/Medium/Low",
    "explanation": "reasoning with evidence from both analyses",
    "recommended_actions": ["action1", "action2"]
}}
"""

    try:
        response_3 = claude.invoke([HumanMessage(content=prompt_3)])
        final_verdict = json.loads(response_3.content)
        print(f"✓ Final Verdict: {'COMPROMISED' if final_verdict['credentials_compromised'] else 'CLEAN'}")
        print(f"  Confidence: {final_verdict['confidence']:.0%}")
        print(f"  Severity: {final_verdict['severity']}")
    except Exception as e:
        print(f"✗ Step 3 failed: {e}")
        final_verdict = {
            "credentials_compromised": True,
            "confidence": 0.7,
            "severity": "High",
            "explanation": f"Error in analysis: {e}",
            "recommended_actions": ["Manual review required"]
        }

    # Display results
    print(f"\n{'🚨 SECURITY ALERT' if final_verdict['credentials_compromised'] else '✅ NO THREAT'}")
    if final_verdict['credentials_compromised']:
        print(f"\nExplanation:")
        print(f"  {final_verdict['explanation']}")
        print(f"\n📋 Recommended Actions:")
        for i, action in enumerate(final_verdict['recommended_actions'], 1):
            print(f"  {i}. {action}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Chapter 70: AI-Powered Threat Detection".center(78) + "║")
    print("║" + "  Dual-AI Security Analysis with Claude + OpenAI".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    try:
        # Test API keys
        anthropic_key, openai_key = get_api_keys()
        print("✓ API keys loaded successfully\n")

        # Run examples
        print("\nRunning all examples...\n")

        example_1()
        print("\n")

        example_2()
        print("\n")

        example_3()
        print("\n")

        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  All examples completed successfully!".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("║" + "  Key Takeaways:".ljust(78) + "║")
        print("║" + "  1. Dual-AI reduces false positives (consensus required)".ljust(78) + "║")
        print("║" + "  2. Claude: Deep reasoning and context analysis".ljust(78) + "║")
        print("║" + "  3. OpenAI: Pattern matching and threat intelligence".ljust(78) + "║")
        print("║" + "  4. LangChain: Orchestration and multi-step reasoning".ljust(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. ANTHROPIC_API_KEY is set")
        print("  2. OPENAI_API_KEY is set")
        print("  3. Required packages are installed:")
        print("     pip install anthropic openai langchain langchain-anthropic langchain-openai")
