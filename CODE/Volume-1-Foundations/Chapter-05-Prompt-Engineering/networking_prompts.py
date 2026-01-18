#!/usr/bin/env python3
"""
Networking Prompt Templates Library

Reusable prompts for common networking tasks with AI.

From: AI for Networking Engineers - Volume 1, Chapter 5
Author: Eduard Dulharu

This library provides battle-tested prompt templates for:
- Configuration security analysis
- Log classification and analysis
- BGP troubleshooting
- ACL generation
- Config diff explanation
- Documentation generation
- VLAN/IP extraction

Usage:
    from networking_prompts import NetworkingPrompts

    prompts = NetworkingPrompts()
    prompt = prompts.config_security_analysis(config_text)
    response = client.call(prompt)
"""

from typing import Optional, List


class NetworkingPrompts:
    """
    Collection of prompt templates for networking tasks.

    All methods return formatted prompts ready to send to LLMs.
    Prompts are designed for deterministic output (use temperature=0).
    """

    @staticmethod
    def config_security_analysis(config: str, platform: str = "Cisco IOS") -> str:
        """
        Analyze network device configuration for security vulnerabilities.

        Args:
            config: Device configuration text
            platform: Device platform (e.g., "Cisco IOS", "Juniper Junos")

        Returns:
            Formatted prompt for security analysis
        """
        return f"""You are a network security expert. Analyze this {platform} configuration for security vulnerabilities.

Configuration:
```
{config}
```

Check for these security issues:
1. **Authentication & Authorization**:
   - Weak or default passwords
   - Missing AAA configuration
   - No TACACS+/RADIUS for centralized auth
   - Weak SNMP community strings (public/private)

2. **Management Protocols**:
   - Unencrypted protocols (telnet, http, SNMP v1/v2c)
   - Missing SSH configuration
   - VTY lines without access control

3. **Access Control**:
   - Overly permissive ACLs (permit ip any any)
   - Missing ACLs on management interfaces
   - No control-plane policing

4. **System Security**:
   - Missing NTP configuration (time sync for logs)
   - Missing logging configuration
   - CDP/LLDP enabled on untrusted interfaces
   - IP source routing enabled

5. **Deprecated/Insecure Features**:
   - Old encryption algorithms
   - Deprecated commands
   - Insecure VPN configurations

Return findings as JSON:
{{
  "critical": [
    {{"issue": "Description of critical issue", "line": "config line or 'N/A'", "fix": "Specific commands to fix", "impact": "Why this matters"}}
  ],
  "high": [
    {{"issue": "...", "line": "...", "fix": "...", "impact": "..."}}
  ],
  "medium": [
    {{"issue": "...", "line": "...", "fix": "...", "impact": "..."}}
  ],
  "low": [
    {{"issue": "...", "line": "...", "fix": "...", "impact": "..."}}
  ],
  "summary": "One-sentence overall security assessment"
}}

If configuration is secure, return empty arrays for each severity level."""

    @staticmethod
    def log_classification(
        log_entry: str,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Classify network log entry by severity.

        Args:
            log_entry: Single log line to classify
            categories: List of categories (default: INFO, WARNING, ERROR, CRITICAL)

        Returns:
            Formatted prompt for log classification
        """
        if categories is None:
            categories = ["INFO", "WARNING", "ERROR", "CRITICAL"]

        cats = ", ".join(categories)

        return f"""Classify this network log entry into ONE of these categories: {cats}

Guidelines:
- **INFO**: Normal operations, state changes, successful authentications, config changes, auditing
- **WARNING**: Potential issues, high utilization, approaching thresholds, non-critical errors, retries
- **ERROR**: Service-affecting errors, interface downs, routing failures, failed authentications
- **CRITICAL**: Outages, complete failures, security breaches, hardware failures, memory exhaustion

Log entry:
{log_entry}

Return ONLY the category name (one word), nothing else."""

    @staticmethod
    def bgp_troubleshooting(
        r1_config: str,
        r2_config: str,
        symptoms: str
    ) -> str:
        """
        Diagnose BGP session issues between two routers.

        Args:
            r1_config: Router 1 BGP configuration
            r2_config: Router 2 BGP configuration
            symptoms: Description of the problem

        Returns:
            Formatted prompt for BGP troubleshooting
        """
        return f"""You are a BGP expert. Diagnose why this BGP session is not establishing.

Symptoms:
{symptoms}

Router 1 Configuration:
```
{r1_config}
```

Router 2 Configuration:
```
{r2_config}
```

Analyze step-by-step:
1. **AS Numbers**: Verify each router expects the correct remote AS
2. **Neighbor IPs**: Check if neighbor IPs are reachable
3. **Update-Source**: Verify update-source configuration matches
4. **Authentication**: Check for MD5 auth mismatches
5. **TTL/Multihop**: For eBGP over non-directly-connected links
6. **ACLs/Firewalls**: Check for blocking TCP 179

Provide response in this format:
```
ROOT CAUSE: [One sentence summary]

EXPLANATION:
[2-3 sentences explaining the issue in detail]

FIX FOR ROUTER 1:
[Specific commands to run on R1]

FIX FOR ROUTER 2:
[Specific commands to run on R2]

VERIFICATION:
[Commands to verify BGP session is up]
```"""

    @staticmethod
    def acl_generation(
        intent: str,
        platform: str = "Cisco IOS",
        acl_number: Optional[int] = None,
        acl_name: Optional[str] = None
    ) -> str:
        """
        Generate access control list from plain English description.

        Args:
            intent: Description of what ACL should do
            platform: Device platform (default: "Cisco IOS")
            acl_number: ACL number (for numbered ACLs)
            acl_name: ACL name (for named ACLs)

        Returns:
            Formatted prompt for ACL generation
        """
        if acl_number:
            acl_info = f"numbered ACL {acl_number}"
        elif acl_name:
            acl_info = f"named ACL '{acl_name}'"
        else:
            acl_info = "named ACL (choose appropriate name)"

        return f"""Generate a {platform} access control list based on this intent:

Intent: {intent}

Requirements:
- Platform: {platform}
- Format: {acl_info}
- Follow security best practices (principle of least privilege)
- Add comments explaining each rule
- Include explicit deny at end if appropriate
- Consider the order of rules (most specific first)

Provide:
1. **Complete ACL Configuration**:
   [Full ACL config with inline comments]

2. **Interface Application**:
   [Commands to apply ACL to interface, both inbound and outbound examples]

3. **Verification Commands**:
   [Commands to verify ACL is working correctly]

4. **Security Notes**:
   [Any security considerations or warnings]"""

    @staticmethod
    def config_diff_explanation(old_config: str, new_config: str) -> str:
        """
        Explain differences between two configurations.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            Formatted prompt for config diff explanation
        """
        return f"""Explain the differences between these two network configurations in plain English for a network operations team.

OLD Configuration:
```
{old_config}
```

NEW Configuration:
```
{new_config}
```

For each change, explain:
1. **What changed** (specific commands added/removed/modified)
2. **Why it matters** (functional impact on network)
3. **Potential risks** (what could go wrong)
4. **Validation needed** (how to verify change works)

Format as a bulleted list:
- Change 1: [description]
  - Impact: [what this affects]
  - Risk: [potential issues]
  - Verify: [validation steps]

- Change 2: ...

If configurations are identical, return: "No changes detected."
"""

    @staticmethod
    def documentation_generation(
        config: str,
        detail_level: str = "medium",
        include_diagram: bool = False
    ) -> str:
        """
        Generate documentation from network device configuration.

        Args:
            config: Device configuration
            detail_level: "brief", "medium", or "detailed"
            include_diagram: Whether to include ASCII network diagram

        Returns:
            Formatted prompt for documentation generation
        """
        detail_map = {
            "brief": "one paragraph summary suitable for executive overview",
            "medium": "2-3 paragraphs with key details for network engineers",
            "detailed": "comprehensive documentation with sections, suitable for runbook"
        }

        detail = detail_map.get(detail_level, detail_map["medium"])

        diagram_instruction = ""
        if include_diagram:
            diagram_instruction = "\n\n5. **Network Diagram**: Create ASCII diagram showing topology and connections"

        return f"""Generate professional network documentation for this device configuration.

Configuration:
```
{config}
```

Create {detail} covering:
1. **Device Purpose**: Role in network (core, distribution, access, edge, etc.)
2. **Interfaces**: Key interfaces, their purposes, IP addressing
3. **Routing**: Routing protocols in use, key routes, neighbors
4. **Security**: ACLs, authentication, encryption features
5. **Special Considerations**: Any unique configurations or gotchas{diagram_instruction}

Use clear, professional language suitable for:
- Network operations team (day-to-day reference)
- New team members (onboarding documentation)
- Audit/compliance review

Format with markdown headers and bullet points."""

    @staticmethod
    def few_shot_vlan_extraction() -> str:
        """
        Extract VLAN IDs from network logs using few-shot learning.

        Returns:
            Formatted few-shot prompt for VLAN extraction
        """
        return """Extract VLAN IDs from network logs. Follow these examples:

Example 1:
Input: "%LINK-3-UPDOWN: Interface Vlan100, changed state to up"
Output: 100

Example 2:
Input: "%SYS-5-CONFIG: Configured from console by admin on vlan 250"
Output: 250

Example 3:
Input: "STP: VLAN0050 Port Gi1/0/1 is now in forwarding state"
Output: 50

Example 4:
Input: "%VLAN_MGR-3-VLAN_STATE: VLAN 175 enabled"
Output: 175

Now extract from this log:
Input: {log_entry}
Output:"""

    @staticmethod
    def ip_address_extraction(text: str, format: str = "list") -> str:
        """
        Extract IP addresses from text (config, logs, etc.).

        Args:
            text: Text containing IP addresses
            format: Output format ("list", "json", "table")

        Returns:
            Formatted prompt for IP extraction
        """
        format_instructions = {
            "list": "Return IP addresses as a simple list, one per line",
            "json": "Return as JSON array: [\"10.1.1.1\", \"10.1.1.2\", ...]",
            "table": "Return as markdown table with columns: IP Address, Subnet Mask, Context"
        }

        instruction = format_instructions.get(format, format_instructions["list"])

        return f"""Extract all IP addresses from this text.

Text:
{text}

Instructions:
- Find all IPv4 addresses
- Include subnet masks if present
- {instruction}
- Return ONLY the formatted output, no explanation"""

    @staticmethod
    def troubleshooting_guide(
        problem_description: str,
        symptoms: List[str],
        logs: Optional[str] = None
    ) -> str:
        """
        Generate troubleshooting steps for a network issue.

        Args:
            problem_description: Brief description of the problem
            symptoms: List of observed symptoms
            logs: Optional relevant log entries

        Returns:
            Formatted prompt for troubleshooting guide generation
        """
        symptoms_text = "\n".join(f"- {s}" for s in symptoms)

        logs_section = ""
        if logs:
            logs_section = f"\n\nRelevant Logs:\n```\n{logs}\n```"

        return f"""Generate a troubleshooting guide for this network issue.

Problem: {problem_description}

Symptoms:
{symptoms_text}{logs_section}

Provide a step-by-step troubleshooting guide:

1. **Initial Assessment** (2-3 quick checks to narrow down the issue)
2. **Data Collection** (show commands to run, what to look for)
3. **Analysis** (how to interpret the data)
4. **Likely Causes** (ranked by probability)
5. **Resolution Steps** (specific commands to fix each likely cause)
6. **Verification** (how to confirm issue is resolved)
7. **Prevention** (how to prevent recurrence)

Format each step with:
- Clear action items
- Specific commands to run
- Expected output
- Decision points (if X, then Y; else Z)

Make it suitable for:
- Junior engineers (explain why, not just what)
- High-pressure situations (be concise, actionable)
- Documentation (can be saved as runbook)"""

    @staticmethod
    def change_impact_analysis(
        proposed_change: str,
        current_config: str,
        network_context: Optional[str] = None
    ) -> str:
        """
        Analyze impact of proposed configuration change.

        Args:
            proposed_change: Description of planned change
            current_config: Current device configuration
            network_context: Optional description of network topology/role

        Returns:
            Formatted prompt for change impact analysis
        """
        context_section = ""
        if network_context:
            context_section = f"\n\nNetwork Context:\n{network_context}"

        return f"""Analyze the impact of this proposed configuration change.

Proposed Change:
{proposed_change}

Current Configuration:
```
{current_config}
```{context_section}

Provide impact analysis:

1. **Affected Services/Features**:
   - What will be directly impacted
   - What dependencies exist

2. **Risk Assessment**:
   - High/Medium/Low risk rating with justification
   - Potential failure modes
   - Blast radius (how many users/services affected)

3. **Prerequisites**:
   - Changes needed before this change
   - Dependencies that must be in place

4. **Implementation Steps**:
   - Step-by-step commands
   - Order of operations
   - Estimated time per step

5. **Rollback Plan**:
   - How to revert if something goes wrong
   - Specific rollback commands
   - Rollback validation

6. **Validation Plan**:
   - Tests to confirm change worked
   - Monitoring to watch during/after change
   - Success criteria

7. **Maintenance Window**:
   - Recommended change window (hours, day of week)
   - Expected downtime (if any)
   - Communication requirements"""


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Try to import API client if available
    try:
        from Chapter_04_API_Basics_Authentication.resilient_api_client import ResilientAPIClient
        from dotenv import load_dotenv

        load_dotenv()
        client = ResilientAPIClient(max_retries=2)
        api_available = True
    except (ImportError, ValueError):
        client = None
        api_available = False
        print("⚠️  API client not available. Showing prompt templates only.\n")

    print("""
    ========================================
    Networking Prompts Library Demo
    ========================================
    """)

    prompts = NetworkingPrompts()

    # Example 1: Log Classification
    print("\n" + "="*60)
    print("Example 1: Log Classification")
    print("="*60)

    test_log = "%OSPF-5-ADJCHG: Neighbor 10.1.1.2 Down: Dead timer expired"
    prompt = prompts.log_classification(test_log)

    print(f"\nLog: {test_log}")
    print(f"\nPrompt (truncated):\n{prompt[:200]}...\n")

    if api_available and client:
        result = client.call(prompt, max_tokens=50, temperature=0)
        if result:
            print(f"Classification: {result['text']}")
            print(f"Cost: ${result['cost']:.6f}")

    # Example 2: ACL Generation
    print("\n" + "="*60)
    print("Example 2: ACL Generation")
    print("="*60)

    intent = "Block all traffic from VLAN 100 (192.168.100.0/24) to the management subnet (10.0.0.0/24)"
    prompt = prompts.acl_generation(
        intent=intent,
        platform="Cisco IOS",
        acl_name="BLOCK_GUEST_TO_MGMT"
    )

    print(f"\nIntent: {intent}")
    print(f"\nPrompt length: {len(prompt)} characters")

    if api_available and client:
        print("\n[Calling API...this may take a few seconds]\n")
        result = client.call(prompt, max_tokens=500, temperature=0)
        if result:
            print("Generated ACL:")
            print("-" * 60)
            print(result['text'][:400])  # Show first 400 chars
            print("...(truncated)")
            print(f"\nTokens: {result['input_tokens']} in + {result['output_tokens']} out")
            print(f"Cost: ${result['cost']:.6f}")

    # Example 3: VLAN Extraction (Few-Shot)
    print("\n" + "="*60)
    print("Example 3: VLAN Extraction (Few-Shot Learning)")
    print("="*60)

    test_log2 = "%VLAN_MGR-3-VLAN_STATE: VLAN 175 enabled"
    prompt_template = prompts.few_shot_vlan_extraction()
    prompt = prompt_template.format(log_entry=test_log2)

    print(f"\nLog: {test_log2}")

    if api_available and client:
        result = client.call(prompt, max_tokens=10, temperature=0)
        if result:
            print(f"Extracted VLAN: {result['text'].strip()}")
            print(f"Cost: ${result['cost']:.6f}")

    # Show all available templates
    print("\n" + "="*60)
    print("Available Prompt Templates")
    print("="*60)

    templates = [
        ("config_security_analysis", "Security vulnerability scanning"),
        ("log_classification", "Classify log severity"),
        ("bgp_troubleshooting", "Diagnose BGP issues"),
        ("acl_generation", "Generate ACLs from intent"),
        ("config_diff_explanation", "Explain config changes"),
        ("documentation_generation", "Generate device docs"),
        ("few_shot_vlan_extraction", "Extract VLANs from logs"),
        ("ip_address_extraction", "Extract IPs from text"),
        ("troubleshooting_guide", "Generate troubleshooting steps"),
        ("change_impact_analysis", "Analyze change impact")
    ]

    for name, description in templates:
        print(f"  • {name:30s} - {description}")

    print("\n" + "="*60)
    print("💡 Usage:")
    print("="*60)
    print("""
    from networking_prompts import NetworkingPrompts

    prompts = NetworkingPrompts()
    prompt = prompts.config_security_analysis(config_text)
    response = client.call(prompt)
    """)

    if api_available and client:
        print("\n✅ Demo complete!")
        client.print_metrics()
    else:
        print("\n✅ Demo complete (templates only - no API calls)")
        print("\n💡 To test with real API calls:")
        print("   1. Set ANTHROPIC_API_KEY in .env")
        print("   2. Install: pip install anthropic python-dotenv")
        print("   3. Run again: python networking_prompts.py")
