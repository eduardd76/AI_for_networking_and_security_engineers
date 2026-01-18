#!/usr/bin/env python3
"""
Testing and Validation

Ensure AI outputs are accurate and reliable.

From: AI for Networking Engineers - Volume 1, Chapter 11
Author: Eduard Dulharu

Usage:
    python testing.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def example_1_test_accuracy():
    """Test if AI provides correct answers."""
    print("\n" + "="*60)
    print("Example 1: Accuracy Testing")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Known correct answers
    test_cases = [
        {
            "question": "What is the default administrative distance for OSPF?",
            "expected": "110",
            "category": "routing"
        },
        {
            "question": "What TCP port does SSH use?",
            "expected": "22",
            "category": "security"
        },
        {
            "question": "What is the default VLAN on Cisco switches?",
            "expected": "1",
            "category": "switching"
        }
    ]

    print("\nTesting known facts:\n")

    passed = 0
    for i, test in enumerate(test_cases, 1):
        response = llm.invoke(test['question'])
        answer = response.content

        # Check if expected answer is in response
        correct = test['expected'] in answer

        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"{i}. [{test['category']}] {test['question']}")
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {answer[:100]}...")
        print(f"   {status}\n")

        if correct:
            passed += 1

    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.0f}%)")


def example_2_consistency_testing():
    """Test if AI gives consistent answers."""
    print("\n" + "="*60)
    print("Example 2: Consistency Testing")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    question = "What are the OSPF area types?"

    print(f"\nQuestion: {question}\n")
    print("Running 3 times to check consistency:\n")

    responses = []
    for i in range(3):
        response = llm.invoke(question)
        responses.append(response.content)
        print(f"Run {i+1}: {response.content[:100]}...")

    # Check if all responses mention the same key terms
    key_terms = ["standard", "stub", "totally stubby", "NSSA"]
    print(f"\nChecking for key terms: {', '.join(key_terms)}")

    consistent = all(
        all(term.lower() in resp.lower() for term in key_terms)
        for resp in responses
    )

    print(f"\nConsistency: {'✓ PASS' if consistent else '✗ FAIL'}")


def example_3_output_validation():
    """Validate structured outputs."""
    print("\n" + "="*60)
    print("Example 3: Output Validation")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    prompt = """Generate a Cisco IOS config for VLAN 10 named "Users".

Provide ONLY the commands, one per line, no explanations."""

    response = llm.invoke(prompt)

    print("Generated commands:")
    print(response.content)

    # Validate output
    commands = response.content.strip().split('\n')

    validations = {
        "Has vlan command": any("vlan" in cmd.lower() for cmd in commands),
        "Has name command": any("name" in cmd.lower() for cmd in commands),
        "No explanatory text": not any(len(cmd.split()) > 5 for cmd in commands),
        "Proper format": all(not cmd.startswith('#') for cmd in commands)
    }

    print("\nValidation:")
    for check, passed in validations.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")


def example_4_regression_testing():
    """Ensure changes don't break existing functionality."""
    print("\n" + "="*60)
    print("Example 4: Regression Testing")
    print("="*60)

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    # Baseline tests (run before changes)
    baseline_tests = [
        "What is BGP?",
        "Explain VLAN trunking",
        "What is STP?"
    ]

    print("\nRunning regression tests (ensure basic functionality works):\n")

    for i, test in enumerate(baseline_tests, 1):
        response = llm.invoke(test)
        # In production, compare against stored baseline responses
        passed = len(response.content) > 50  # Simple check

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{i}. {test}")
        print(f"   {status} (response length: {len(response.content)} chars)\n")


def main():
    """Run all examples."""
    print("="*60)
    print("Testing and Validation")
    print("="*60)

    try:
        example_1_test_accuracy()
        example_2_consistency_testing()
        example_3_output_validation()
        example_4_regression_testing()

        print("\n" + "="*60)
        print("✓ All tests completed!")
        print("="*60)
        print("\nTesting Strategies:")
        print("1. Accuracy - Verify correct answers")
        print("2. Consistency - Same question, same answer")
        print("3. Validation - Check output format")
        print("4. Regression - Ensure no breaking changes")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have ANTHROPIC_API_KEY in .env")


if __name__ == "__main__":
    main()
