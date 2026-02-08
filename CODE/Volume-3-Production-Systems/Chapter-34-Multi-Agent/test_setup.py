#!/usr/bin/env python3
"""
Test Setup Script for Multi-Agent Orchestration

Verifies that all dependencies are installed and API keys are configured.

Usage:
    python test_setup.py
"""

import sys
import os


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")

    required_packages = [
        ("anthropic", "Anthropic SDK"),
        ("langchain", "LangChain"),
        ("langchain_anthropic", "LangChain Anthropic"),
        ("langgraph", "LangGraph"),
        ("pydantic", "Pydantic"),
        ("dotenv", "Python-dotenv")
    ]

    failed = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            failed.append(name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    print("\n✓ All required packages installed")
    return True


def test_api_key():
    """Test that API key is configured."""
    print("\nTesting API key configuration...")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("  ✗ ANTHROPIC_API_KEY not found in environment")
        print("\nCreate a .env file with:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        return False

    if not api_key.startswith("sk-ant-"):
        print("  ⚠ API key format looks unusual (should start with sk-ant-)")
        return False

    print(f"  ✓ API key configured ({api_key[:20]}...)")
    return True


def test_basic_functionality():
    """Test basic agent functionality."""
    print("\nTesting basic agent functionality...")

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.schema import HumanMessage

        # Create a simple LLM instance
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            max_tokens=100
        )

        # Test simple message
        response = llm.invoke([HumanMessage(content="Say 'test successful' and nothing else")])

        print(f"  ✓ API connection successful")
        print(f"  ✓ Response received: {response.content[:50]}...")
        return True

    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        return False


def test_multi_agent_imports():
    """Test that multi_agent.py can be imported."""
    print("\nTesting multi-agent module...")

    try:
        from multi_agent import (
            DiagnosisAgent,
            ConfigAgent,
            SecurityAgent,
            PerformanceAgent,
            SupervisorAgent,
            AgentResponse
        )

        print("  ✓ DiagnosisAgent")
        print("  ✓ ConfigAgent")
        print("  ✓ SecurityAgent")
        print("  ✓ PerformanceAgent")
        print("  ✓ SupervisorAgent")
        print("  ✓ AgentResponse model")

        return True

    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_pydantic_models():
    """Test Pydantic models."""
    print("\nTesting data models...")

    try:
        from multi_agent import AgentResponse, AgentTask

        # Test AgentResponse
        response = AgentResponse(
            agent_name="TestAgent",
            task="Test task",
            status="success",
            findings=["Finding 1", "Finding 2"],
            recommendations=["Rec 1"],
            confidence=0.95,
            execution_time=1.5,
            raw_output="Test output"
        )

        print(f"  ✓ AgentResponse model")
        print(f"    - Agent: {response.agent_name}")
        print(f"    - Status: {response.status}")
        print(f"    - Confidence: {response.confidence:.0%}")

        # Test AgentTask
        task = AgentTask(
            agent_name="diagnosis",
            task_type="troubleshoot",
            description="Test troubleshooting"
        )

        print(f"  ✓ AgentTask dataclass")
        print(f"    - Agent: {task.agent_name}")
        print(f"    - Type: {task.task_type}")

        return True

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_tools():
    """Test diagnostic tools."""
    print("\nTesting diagnostic tools...")

    try:
        from multi_agent import (
            get_interface_statistics,
            check_routing_protocol,
            get_device_health
        )

        # Test interface stats
        result = get_interface_statistics.invoke({"interface": "GigabitEthernet0/0"})
        print(f"  ✓ get_interface_statistics")

        # Test routing protocol
        result = check_routing_protocol.invoke({"protocol": "ospf"})
        print(f"  ✓ check_routing_protocol")

        # Test device health
        result = get_device_health.invoke({})
        print(f"  ✓ get_device_health")

        return True

    except Exception as e:
        print(f"  ✗ Tool test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Multi-Agent Orchestration - Setup Verification")
    print("="*70)

    results = []

    # Run tests
    results.append(("Package imports", test_imports()))
    results.append(("API key", test_api_key()))

    # Only run API test if previous tests passed
    if all(r[1] for r in results):
        results.append(("API connection", test_basic_functionality()))

    results.append(("Module imports", test_multi_agent_imports()))
    results.append(("Data models", test_pydantic_models()))
    results.append(("Diagnostic tools", test_tools()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")

    print("="*70)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    if passed == total:
        print(f"\n✓ All tests passed ({passed}/{total})")
        print("\nYou're ready to run the examples!")
        print("Run: python multi_agent.py")
        return 0
    else:
        print(f"\n⚠ Some tests failed ({passed}/{total} passed)")
        print("\nPlease fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
