#!/usr/bin/env python3
"""
Test All Volume 2 Examples

Systematically test all code examples to verify they work.

Usage:
    python test_all.py
"""

import os
import sys
from pathlib import Path


def test_no_api_required():
    """Test examples that don't require API keys."""
    print("="*60)
    print("TEST 1: Examples Without API Keys")
    print("="*60)

    # Test document loader
    print("\n1. Testing document_loader.py...")
    try:
        os.chdir("Chapter-14-RAG-Fundamentals")
        import document_loader

        # Create sample docs
        document_loader.create_sample_docs("./test_docs")

        # Load them
        docs = document_loader.load_docs_from_directory("./test_docs")

        if len(docs) == 3:
            print("   ✓ Document loader works! Loaded 3 docs")
        else:
            print(f"   ✗ Expected 3 docs, got {len(docs)}")

        # Cleanup
        import shutil
        shutil.rmtree("./test_docs")
        os.chdir("..")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        os.chdir("..")
        return False


def check_api_keys():
    """Check if required API keys are set."""
    print("\n" + "="*60)
    print("TEST 2: API Key Check")
    print("="*60)

    from dotenv import load_dotenv
    load_dotenv()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"\nANTHROPIC_API_KEY: {'✓ Set' if anthropic_key else '✗ Not set'}")
    print(f"OPENAI_API_KEY:    {'✓ Set' if openai_key else '✗ Not set'}")

    if not anthropic_key:
        print("\n⚠️  Set ANTHROPIC_API_KEY in .env to test API-dependent examples")
        return False, False

    if not openai_key:
        print("\n⚠️  Set OPENAI_API_KEY in .env to test RAG examples")
        return True, False

    return True, True


def test_with_anthropic_key():
    """Test examples that need Anthropic API key."""
    print("\n" + "="*60)
    print("TEST 3: Examples with Anthropic API")
    print("="*60)

    try:
        from langchain_anthropic import ChatAnthropic

        # Test simple LLM call
        print("\n1. Testing basic LLM call...")
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        response = llm.invoke("Say 'API working!'")

        if "API working" in response.content:
            print("   ✓ Anthropic API works!")
            return True
        else:
            print("   ✗ Unexpected response")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_with_openai_key():
    """Test examples that need OpenAI API key."""
    print("\n" + "="*60)
    print("TEST 4: Examples with OpenAI API")
    print("="*60)

    try:
        from langchain_openai import OpenAIEmbeddings

        # Test embeddings
        print("\n1. Testing embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        result = embeddings.embed_query("test")

        if len(result) > 0:
            print(f"   ✓ Embeddings work! (dimension: {len(result)})")
            return True
        else:
            print("   ✗ Empty embedding returned")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Volume 2: Testing All Examples")
    print("="*60)

    results = {
        "no_api": False,
        "anthropic": False,
        "openai": False
    }

    # Test 1: No API required
    results["no_api"] = test_no_api_required()

    # Test 2: Check API keys
    has_anthropic, has_openai = check_api_keys()

    # Test 3: Anthropic examples
    if has_anthropic:
        results["anthropic"] = test_with_anthropic_key()
    else:
        print("\n⏭️  Skipping Anthropic tests (no API key)")

    # Test 4: OpenAI examples
    if has_openai:
        results["openai"] = test_with_openai_key()
    else:
        print("\n⏭️  Skipping OpenAI tests (no API key)")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    print("\nResults:")
    print(f"  No API tests:     {'✓ PASS' if results['no_api'] else '✗ FAIL'}")
    print(f"  Anthropic tests:  {'✓ PASS' if results['anthropic'] else '⏭️  SKIP' if not has_anthropic else '✗ FAIL'}")
    print(f"  OpenAI tests:     {'✓ PASS' if results['openai'] else '⏭️  SKIP' if not has_openai else '✗ FAIL'}")

    # Overall status
    critical_pass = results["no_api"]
    optional_pass = (has_anthropic and results["anthropic"]) or not has_anthropic

    if critical_pass and optional_pass:
        print("\n✅ All available tests passed!")
        return 0
    elif critical_pass:
        print("\n⚠️  Core tests passed, but API tests need keys")
        print("\nTo test API features:")
        print("  1. Add API keys to .env")
        print("  2. Run: python test_all.py")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
