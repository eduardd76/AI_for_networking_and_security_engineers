#!/usr/bin/env python3
"""
AI-Powered Network Config Analyzer
Identifies security issues, best practice violations, and optimization opportunities.

This is the main project from Chapter 1: What is Generative AI?

Usage:
    python ai_config_analysis.py                    # Analyze sample_config.txt
    python ai_config_analysis.py --file myconfig.cfg  # Analyze custom file
    python ai_config_analysis.py --severity high    # Only show high+ severity

Author: Eduard Dulharu (Ed Harmoosh)
Company: vExpertAI GmbH
"""

import os
import sys
import json
import re
import argparse

# Load environment variables from .env file (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment


def get_ai_client():
    """
    Create an Anthropic client.
    
    We use the Anthropic SDK directly — no frameworks needed for basic calls.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Error: anthropic package not installed")
        print("   Run: pip install anthropic")
        sys.exit(1)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found")
        print("")
        print("   Option 1: Export it directly")
        print("   export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("")
        print("   Option 2: Add to .env file")
        print("   echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env")
        print("")
        print("   Get a key at: https://console.anthropic.com/")
        sys.exit(1)
    
    return Anthropic(api_key=api_key)


def analyze_config(config_file: str) -> dict:
    """
    Analyze network configuration using Claude.
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Dictionary with findings categorized by severity
    """
    # Read config file
    try:
        with open(config_file, 'r') as f:
            config_text = f.read()
    except FileNotFoundError:
        return {"error": f"Config file not found: {config_file}"}
    
    client = get_ai_client()
    
    # Craft the prompt
    prompt = f"""You are a senior network security engineer conducting a configuration review.

Analyze this Cisco IOS configuration and identify:

1. **Critical Security Issues**: Vulnerabilities that could lead to compromise
2. **Warnings**: Best practice violations that should be addressed
3. **Optimizations**: Opportunities to improve performance or maintainability

For each finding, provide:
- **Category**: security | best-practice | optimization
- **Severity**: critical | high | medium | low
- **Issue**: One-line description
- **Explanation**: Why this matters (2-3 sentences)
- **Recommendation**: Specific fix with example commands

Configuration:
```
{config_text}
```

Return your analysis as valid JSON in this exact format:
{{
  "findings": [
    {{
      "category": "security",
      "severity": "critical",
      "issue": "Weak SNMP community string",
      "explanation": "The 'public' community string is widely known and allows read access to device information.",
      "recommendation": "Change to a strong, unique community string: snmp-server community X92kP!m3Z RO"
    }}
  ],
  "summary": {{
    "total_issues": 5,
    "critical": 2,
    "high": 1,
    "medium": 1,
    "low": 1
  }}
}}
"""
    
    # Call Claude API
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0,  # Deterministic output for consistency
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract response text
        response_text = response.content[0].text
        
        # Parse JSON (Claude should return valid JSON)
        try:
            # Find JSON in response (may have surrounding text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"error": "No JSON found in response", "raw": response_text}
        
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON in response", "raw": response_text}
        
        return result
    
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}


def format_findings(analysis: dict, min_severity: str = None) -> None:
    """
    Pretty-print the analysis results.
    
    Args:
        analysis: Dictionary from analyze_config()
        min_severity: Minimum severity to show (critical, high, medium, low)
    """
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        if "raw" in analysis:
            print(f"\nRaw response:\n{analysis['raw']}")
        return
    
    severity_order = ['critical', 'high', 'medium', 'low']
    severity_icons = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    
    # Filter by minimum severity if specified
    if min_severity:
        min_idx = severity_order.index(min_severity)
        show_severities = severity_order[:min_idx + 1]
    else:
        show_severities = severity_order
    
    # Print summary
    summary = analysis.get('summary', {})
    print("=" * 80)
    print("CONFIG ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Issues: {summary.get('total_issues', 0)}")
    print(f"  🔴 Critical: {summary.get('critical', 0)}")
    print(f"  🟠 High: {summary.get('high', 0)}")
    print(f"  🟡 Medium: {summary.get('medium', 0)}")
    print(f"  🟢 Low: {summary.get('low', 0)}")
    print()
    
    if min_severity:
        print(f"(Showing {min_severity} and above)")
        print()
    
    # Print findings by severity
    for severity in severity_order:
        if severity not in show_severities:
            continue
            
        findings = [f for f in analysis.get('findings', [])
                   if f.get('severity') == severity]
        
        if not findings:
            continue
        
        print(f"\n{severity_icons[severity]} {severity.upper()} ISSUES")
        print("-" * 80)
        
        for i, finding in enumerate(findings, 1):
            print(f"\n{i}. {finding.get('issue', 'Unknown issue')}")
            print(f"   Category: {finding.get('category', 'unknown')}")
            print(f"\n   Explanation:")
            print(f"   {finding.get('explanation', 'No explanation')}")
            print(f"\n   Recommendation:")
            print(f"   {finding.get('recommendation', 'No recommendation')}")
    
    print("\n" + "=" * 80)


def create_sample_config():
    """Create sample config file if it doesn't exist."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_config.txt")
    
    if os.path.exists(sample_path):
        return sample_path
    
    sample_config = """version 15.2
service timestamps debug datetime msec
service timestamps log datetime msec
no service password-encryption
!
hostname Branch-RTR-01
!
boot-start-marker
boot-end-marker
!
enable secret 5 $1$mERr$Vh5xqO5S7K8X1.L4K2kNz1
!
no aaa new-model
!
ip cef
!
interface GigabitEthernet0/0
 description WAN-Uplink
 ip address 203.0.113.5 255.255.255.252
 duplex auto
 speed auto
!
interface GigabitEthernet0/1
 description LAN-Access
 ip address 192.168.100.1 255.255.255.0
 duplex auto
 speed auto
!
interface Vlan10
 description Guest-Network
 ip address 10.10.10.1 255.255.255.0
!
router ospf 1
 network 192.168.100.0 0.0.0.255 area 0
 network 10.10.10.0 0.0.0.255 area 1
!
ip route 0.0.0.0 0.0.0.0 203.0.113.6
!
snmp-server community public RO
snmp-server community private RW
!
line vty 0 4
 password cisco123
 transport input telnet ssh
line vty 5 15
 no login
!
end
"""
    
    with open(sample_path, 'w') as f:
        f.write(sample_config)
    
    print(f"Created sample config: {sample_path}")
    return sample_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Network Config Analyzer - Chapter 1 Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_config_analysis.py                      Analyze sample_config.txt
  python ai_config_analysis.py --file router.cfg   Analyze custom file
  python ai_config_analysis.py --severity high     Only show high and critical
  python ai_config_analysis.py --output results.json   Save to custom file
        """
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Config file to analyze (default: sample_config.txt)"
    )
    
    parser.add_argument(
        "-s", "--severity",
        choices=['critical', 'high', 'medium', 'low'],
        help="Minimum severity to display"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="analysis_results.json",
        help="Output JSON file (default: analysis_results.json)"
    )
    
    args = parser.parse_args()
    
    print()
    print("🔍 AI-Powered Config Analyzer")
    print("   Chapter 1: What is Generative AI?")
    print("=" * 80)
    
    # Determine config file
    if args.file:
        config_file = args.file
    else:
        config_file = create_sample_config()
    
    print(f"Analyzing: {config_file}")
    print("This may take 10-20 seconds...\n")
    
    # Run analysis
    analysis = analyze_config(config_file)
    
    # Display results
    format_findings(analysis, min_severity=args.severity)
    
    # Save results
    if "error" not in analysis:
        output_file = args.output
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✅ Full results saved to: {output_file}")
    
    print()
    print("💡 Next Steps:")
    print("   • Try analyzing your own configs (sanitize sensitive data first)")
    print("   • Modify the prompt to add custom checks")
    print("   • See Chapter 2 for how LLMs work under the hood")
    print()


if __name__ == "__main__":
    main()
