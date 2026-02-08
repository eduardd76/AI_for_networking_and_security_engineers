#!/usr/bin/env python3
"""
Ethics and Responsible AI

Safe, responsible AI use in production networks.

From: AI for Networking Engineers - Volume 1, Chapter 12
Author: Eduard Dulharu

Usage:
    python ethics.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from typing import List, Dict

load_dotenv()


def example_1_command_guardrails():
    """Prevent dangerous commands."""
    print("\n" + "="*60)
    print("Example 1: Command Guardrails")
    print("="*60)

    # Dangerous command patterns to block
    dangerous_patterns = [
        "reload",
        "write erase",
        "format",
        "delete flash:",
        "no service"
    ]

    test_commands = [
        "show ip interface brief",  # Safe
        "reload in 5",  # Dangerous
        "configure terminal",  # Safe
        "write erase",  # Dangerous
        "interface Gi0/1",  # Safe
    ]

    print("\nChecking commands against guardrails:\n")

    for cmd in test_commands:
        dangerous = any(pattern in cmd.lower() for pattern in dangerous_patterns)

        if dangerous:
            print(f"✗ BLOCKED: {cmd}")
            print(f"   Reason: Dangerous operation")
        else:
            print(f"✓ ALLOWED: {cmd}")
        print()


def example_2_approval_workflow():
    """Require human approval for changes."""
    print("\n" + "="*60)
    print("Example 2: Approval Workflow")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    # AI generates commands
    response = llm.invoke("Generate commands to shutdown interface Gi0/1")
    commands = response.content

    print("AI-generated commands:")
    print(commands)

    print("\n" + "-"*60)
    print("Approval required before execution:")
    print("  [ ] Network Engineer")
    print("  [ ] Team Lead")
    print("  [ ] Change Management")
    print("\n(In production, integrate with ticketing system)")


def example_3_audit_logging():
    """Log all AI interactions."""
    print("\n" + "="*60)
    print("Example 3: Audit Logging")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    question = "How do I configure OSPF?"
    response = llm.invoke(question)

    # Audit log entry
    audit_log = {
        "timestamp": "2026-01-18 10:30:45",
        "user": "john.doe@company.com",
        "action": "query",
        "question": question,
        "response_length": len(response.content),
        "model": "claude-sonnet-4-20250514",
        "approved": False,
        "executed": False
    }

    print("\nAudit log entry:")
    for key, value in audit_log.items():
        print(f"  {key:20s}: {value}")

    print("\n(In production, store in secure database)")


def example_4_read_only_mode():
    """Default to read-only operations."""
    print("\n" + "="*60)
    print("Example 4: Read-Only by Default")
    print("="*60)

    # Safe commands (read-only)
    safe_commands = [
        "show running-config",
        "show ip interface brief",
        "show version",
        "show ip route",
        "show vlan brief"
    ]

    # Requires elevated permissions
    write_commands = [
        "configure terminal",
        "reload",
        "write memory"
    ]

    print("\nDefault mode: READ-ONLY")
    print("\nAllowed commands:")
    for cmd in safe_commands:
        print(f"  ✓ {cmd}")

    print("\nRequires elevated access:")
    for cmd in write_commands:
        print(f"  🔒 {cmd}")


def example_5_change_verification():
    """Verify changes before and after."""
    print("\n" + "="*60)
    print("Example 5: Change Verification")
    print("="*60)

    print("\nChange workflow:")
    print("\n1. Pre-check:")
    print("   - Capture current config")
    print("   - Document current state")
    print("   - Create rollback plan")

    print("\n2. Execute change:")
    print("   - Apply AI-generated commands")
    print("   - Monitor for errors")
    print("   - Log all outputs")

    print("\n3. Post-check:")
    print("   - Verify expected changes")
    print("   - Check for side effects")
    print("   - Confirm services operational")

    print("\n4. Rollback if needed:")
    print("   - Restore previous config")
    print("   - Document failure")
    print("   - Update AI model")


def main():
    """Run all examples."""
    print("="*60)
    print("Ethics and Responsible AI")
    print("="*60)

    try:
        example_1_command_guardrails()
        example_2_approval_workflow()
        example_3_audit_logging()
        example_4_read_only_mode()
        example_5_change_verification()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nBest Practices:")
        print("1. Guardrails - Block dangerous commands")
        print("2. Approval - Human review before execution")
        print("3. Audit logs - Track all AI interactions")
        print("4. Read-only default - Require explicit permissions")
        print("5. Verification - Check before and after changes")
        print("\n🔒 Safety first in production networks!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
