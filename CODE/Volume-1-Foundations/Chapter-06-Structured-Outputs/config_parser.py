#!/usr/bin/env python3
"""
Configuration Parser with AI and Structured Outputs

Parse network device configurations using AI and validate with Pydantic schemas.

From: AI for Networking Engineers - Volume 1, Chapter 6
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from config_parser import AIConfigParser

    parser = AIConfigParser()
    device = parser.parse_device_config(config_text)

    print(f"Device: {device.hostname}")
    for interface in device.interfaces:
        print(f"  {interface.name}: {interface.ip_address}")
"""

import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from json_validator import JSONValidator
from pydantic_schemas import (
    DeviceConfig,
    InterfaceConfig,
    SecurityAnalysisResult,
    SecurityFinding
)

try:
    from Chapter_04_API_Basics_Authentication.resilient_api_client import ResilientAPIClient
    from dotenv import load_dotenv
    load_dotenv()
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIConfigParser:
    """
    Parse network configurations using AI with structured output validation.

    Combines:
    - AI for intelligent parsing
    - JSON extraction for structured data
    - Pydantic validation for type safety
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize parser.

        Args:
            api_key: Anthropic API key (optional if in env)
        """
        if not API_AVAILABLE:
            raise ImportError(
                "API client not available. Install requirements: "
                "pip install anthropic python-dotenv"
            )

        self.client = ResilientAPIClient(api_key=api_key)
        self.validator = JSONValidator()

    def parse_device_config(
        self,
        config: str,
        platform: str = "Cisco IOS"
    ) -> DeviceConfig:
        """
        Parse device configuration into structured data.

        Args:
            config: Device configuration text
            platform: Device platform (e.g., "Cisco IOS")

        Returns:
            DeviceConfig object with validated data

        Raises:
            ValueError: If parsing or validation fails
        """
        prompt = f"""Extract device information from this {platform} configuration.

Configuration:
```
{config}
```

Extract the following information and return as JSON:
{{
  "hostname": "device hostname",
  "mgmt_ip": "management IP address or 'unknown'",
  "device_type": "router|switch|firewall|load_balancer|access_point",
  "model": "device model if mentioned",
  "ios_version": "software version if mentioned",
  "location": "location if mentioned in description",
  "interfaces": [
    {{
      "name": "interface name (e.g., GigabitEthernet0/1)",
      "ip_address": "IP address or null if not configured",
      "subnet_mask": "subnet mask or null",
      "status": "up|down|admin_down (determine from 'no shutdown' etc)",
      "description": "interface description if present",
      "vlan": VLAN number if configured (integer)
    }}
  ]
}}

Rules:
- Extract ALL interfaces mentioned
- Infer status from 'shutdown' command (shutdown=down, no shutdown=up)
- Use null for missing values
- Return ONLY valid JSON, no explanation"""

        # Call AI
        result = self.client.call(prompt, temperature=0, max_tokens=2000)

        if not result:
            raise ValueError("AI API call failed")

        # Extract and validate JSON
        try:
            json_data = self.validator.extract_json(result['text'])
        except Exception as e:
            raise ValueError(f"Failed to extract JSON: {e}")

        # Validate with Pydantic
        try:
            device = DeviceConfig(**json_data)
            return device
        except Exception as e:
            raise ValueError(f"Validation failed: {e}")

    def parse_interface_config(
        self,
        config_snippet: str
    ) -> InterfaceConfig:
        """
        Parse a single interface configuration.

        Args:
            config_snippet: Interface configuration snippet

        Returns:
            InterfaceConfig object
        """
        prompt = f"""Extract interface information from this configuration:

```
{config_snippet}
```

Return as JSON:
{{
  "name": "interface name",
  "ip_address": "IP address or null",
  "subnet_mask": "subnet mask or null",
  "status": "up|down|admin_down",
  "description": "description or null",
  "vlan": VLAN ID (integer) or null
}}

Return ONLY JSON."""

        result = self.client.call(prompt, temperature=0, max_tokens=300)

        if not result:
            raise ValueError("AI API call failed")

        json_data = self.validator.extract_json(result['text'])
        return InterfaceConfig(**json_data)

    def security_analysis(
        self,
        config: str,
        platform: str = "Cisco IOS"
    ) -> SecurityAnalysisResult:
        """
        Analyze configuration for security issues.

        Args:
            config: Device configuration
            platform: Device platform

        Returns:
            SecurityAnalysisResult with findings
        """
        prompt = f"""Analyze this {platform} configuration for security vulnerabilities.

Configuration:
```
{config}
```

Return findings as JSON:
{{
  "device_hostname": "hostname from config",
  "critical_findings": [
    {{
      "issue": "description",
      "severity": "critical",
      "line": "config line with issue",
      "fix": "how to fix",
      "impact": "security impact"
    }}
  ],
  "high_findings": [...],
  "medium_findings": [...],
  "low_findings": [...],
  "summary": "one sentence overall assessment"
}}

Check for:
- Telnet, HTTP (should use SSH, HTTPS)
- Weak SNMP (v1/v2c with default community strings)
- Missing AAA
- No NTP
- Overly permissive ACLs

Return ONLY JSON."""

        result = self.client.call(prompt, temperature=0, max_tokens=2000)

        if not result:
            raise ValueError("AI API call failed")

        json_data = self.validator.extract_json(result['text'])
        return SecurityAnalysisResult(**json_data)


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    AI Config Parser Demo
    ========================================
    Parse network configs with AI + validation
    ========================================
    """)

    if not API_AVAILABLE:
        print("❌ Error: API client not available")
        print("\n💡 Install requirements:")
        print("   pip install anthropic python-dotenv")
        print("   Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    # Sample config
    sample_config = """
hostname CORE-RTR-01

interface GigabitEthernet0/0
 description Uplink to ISP
 ip address 203.0.113.1 255.255.255.252
 no shutdown

interface GigabitEthernet0/1
 description LAN Interface
 ip address 10.1.1.1 255.255.255.0
 no shutdown

interface GigabitEthernet0/2
 description Unused
 shutdown

line vty 0 4
 transport input telnet
 password cisco123

snmp-server community public RO
"""

    try:
        parser = AIConfigParser()

        # Test 1: Parse device config
        print("\nTest 1: Parse Device Configuration")
        print("-" * 60)

        print("[Calling AI API...]\n")
        device = parser.parse_device_config(sample_config)

        print(f"✓ Parsed device: {device.hostname}")
        print(f"  Type: {device.device_type}")
        print(f"  Interfaces: {len(device.interfaces)}")

        for iface in device.interfaces:
            ip_info = f"{iface.ip_address}/{iface.subnet_mask}" if iface.ip_address else "No IP"
            print(f"    - {iface.name}: {ip_info} ({iface.status})")
            if iface.description:
                print(f"       Description: {iface.description}")

        # Test 2: Security analysis
        print("\n\nTest 2: Security Analysis")
        print("-" * 60)

        print("[Calling AI API...]\n")
        analysis = parser.security_analysis(sample_config)

        print(f"✓ Analysis complete for: {analysis.device_hostname}")
        print(f"  Total findings: {analysis.total_findings()}")
        print(f"  Summary: {analysis.summary}")

        if analysis.critical_findings:
            print(f"\n  ⚠️  CRITICAL Issues:")
            for finding in analysis.critical_findings:
                print(f"    - {finding.issue}")
                print(f"      Line: {finding.line}")
                print(f"      Fix: {finding.fix}")
                print()

        if analysis.high_findings:
            print(f"  ⚠️  HIGH Issues:")
            for finding in analysis.high_findings:
                print(f"    - {finding.issue}")
                print()

        # Show cost
        metrics = parser.client.get_metrics()
        print(f"\n{'='*60}")
        print(f"API Usage:")
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Total cost: {metrics['total_cost']}")
        print(f"{'='*60}")

        print("\n✅ Demo complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
