#!/usr/bin/env python3
"""
Model Benchmarking Tool for Network Engineers
Compare multiple LLMs on your specific networking tasks.

This is the main project from Chapter 3: Choosing the Right Model.

Usage:
    python model_benchmark.py --task security_analysis
    python model_benchmark.py --all
    python model_benchmark.py --list

Author: Eduard Dulharu (Ed Harmoosh)
Company: vExpertAI GmbH
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import API clients
ANTHROPIC_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, January 2026)
# ---------------------------------------------------------------------------

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


# ---------------------------------------------------------------------------
# API Call Functions
# ---------------------------------------------------------------------------

def call_claude(model: str, prompt: str) -> dict:
    """Call Claude API and measure performance."""
    if not ANTHROPIC_AVAILABLE:
        return {"error": "anthropic package not installed (pip install anthropic)"}
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    
    client = Anthropic(api_key=api_key)
    start = time.time()
    
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    latency = time.time() - start
    
    return {
        "response": response.content[0].text,
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def call_openai(model: str, prompt: str) -> dict:
    """Call OpenAI API and measure performance."""
    if not OPENAI_AVAILABLE:
        return {"error": "openai package not installed (pip install openai)"}
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    client = OpenAI(api_key=api_key)
    start = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000
    )
    
    latency = time.time() - start
    
    return {
        "response": response.choices[0].message.content,
        "latency": latency,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost for a model call."""
    if model not in PRICING:
        return 0.0
    pricing = PRICING[model]
    return (input_tokens / 1_000_000) * pricing["input"] + \
           (output_tokens / 1_000_000) * pricing["output"]


# ---------------------------------------------------------------------------
# Benchmark Tasks
# ---------------------------------------------------------------------------

TASKS = {
    "security_analysis": {
        "name": "Config Security Analysis",
        "description": "Analyze a router config for security vulnerabilities",
        "prompt": """Analyze this Cisco IOS configuration for security issues:

hostname Branch-RTR-01
!
interface GigabitEthernet0/1
 description LAN
 ip address 192.168.1.1 255.255.255.0
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
no service password-encryption

List all security issues found with severity (critical/high/medium/low) and remediation.""",
        "expected_terms": ["snmp", "community", "public", "private", "telnet", 
                          "password", "vty", "no login", "encryption"],
    },
    
    "bgp_troubleshooting": {
        "name": "BGP Troubleshooting",
        "description": "Diagnose why a BGP session won't establish",
        "prompt": """Router R1 cannot establish BGP with R2. Diagnose the issue:

R1 config:
router bgp 65001
 neighbor 10.1.1.2 remote-as 65002

R2 config:
router bgp 65003
 neighbor 10.1.1.1 remote-as 65001

The BGP session stays in Active state. What's wrong and how do we fix it?""",
        "expected_terms": ["AS", "mismatch", "65002", "65003", "remote-as", "neighbor"],
    },
    
    "acl_generation": {
        "name": "ACL Generation",
        "description": "Generate an ACL from natural language requirements",
        "prompt": """Generate a Cisco extended ACL named BLOCK-WEB to:
1. Block HTTP (port 80) and HTTPS (port 443) from 192.168.100.0/24 to any destination
2. Block DNS (port 53 UDP and TCP) from the same subnet
3. Allow all other traffic from that subnet
4. Implicitly deny everything else

Use standard Cisco IOS extended ACL syntax with remarks.""",
        "expected_terms": ["access-list", "deny", "tcp", "udp", "192.168.100.0", 
                          "80", "443", "53", "permit", "remark"],
    },
    
    "log_classification": {
        "name": "Log Classification",
        "description": "Classify syslog messages by severity and type",
        "prompt": """Classify each of these syslog messages by:
- Severity: critical, high, medium, low, info
- Category: security, hardware, routing, interface, system

Messages:
1. %SEC-6-IPACCESSLOGP: list 101 denied tcp 10.1.1.100(43521) -> 192.168.1.1(22), 1 packet
2. %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
3. %BGP-5-ADJCHANGE: neighbor 10.0.0.2 Down BGP Notification sent
4. %SYS-5-CONFIG_I: Configured from console by admin on vty0 (10.1.1.50)
5. %PLATFORM-2-PEM_FAULT: Power supply 1 has failed

Return as a table with columns: Message#, Severity, Category, Brief Explanation.""",
        "expected_terms": ["critical", "high", "security", "routing", "interface", 
                          "hardware", "power", "BGP", "denied"],
    },
    
    "documentation": {
        "name": "Documentation Generation",
        "description": "Generate network documentation from config",
        "prompt": """Generate concise network documentation from this config:

hostname DIST-SW-01
!
vlan 10
 name USERS
vlan 20
 name SERVERS
vlan 30
 name MANAGEMENT
vlan 99
 name NATIVE
!
interface GigabitEthernet0/1
 description Uplink-to-Core-SW
 switchport trunk encapsulation dot1q
 switchport mode trunk
 switchport trunk native vlan 99
 switchport trunk allowed vlan 10,20,30
!
interface GigabitEthernet0/2
 description Server-Rack-A
 switchport mode access
 switchport access vlan 20
!
interface Vlan30
 ip address 10.30.1.1 255.255.255.0

Document: device purpose, VLAN design, key interfaces, and any notable configurations.""",
        "expected_terms": ["distribution", "switch", "VLAN", "trunk", "uplink", 
                          "server", "management", "native"],
    },
}


# ---------------------------------------------------------------------------
# Benchmark Functions
# ---------------------------------------------------------------------------

def score_response(response: str, expected_terms: list) -> float:
    """Simple quality scoring based on expected terms."""
    response_lower = response.lower()
    found = sum(1 for term in expected_terms if term.lower() in response_lower)
    return found / len(expected_terms)


def run_benchmark(task_key: str, verbose: bool = False) -> dict:
    """Run a benchmark task across all available models."""
    task = TASKS[task_key]
    results = []
    
    print(f"\n{'='*70}")
    print(f"TASK: {task['name']}")
    print(f"{'='*70}")
    
    if verbose:
        print(f"\nDescription: {task['description']}")
    
    models = [
        ("claude", "claude-sonnet-4-20250514"),
        ("claude", "claude-haiku-4-20250514"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
    ]
    
    for provider, model in models:
        print(f"\nTesting {model}...", end=" ", flush=True)
        
        try:
            if provider == "claude":
                result = call_claude(model, task["prompt"])
            else:
                result = call_openai(model, task["prompt"])
            
            if "error" in result:
                print(f"SKIP ({result['error']})")
                continue
            
            cost = calculate_cost(
                result["input_tokens"],
                result["output_tokens"],
                model
            )
            quality = score_response(result["response"], task["expected_terms"])
            
            results.append({
                "model": model,
                "latency": round(result["latency"], 2),
                "cost": round(cost, 6),
                "quality": round(quality * 100, 1),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "response_preview": result["response"][:300] + "..." if len(result["response"]) > 300 else result["response"],
            })
            
            print(f"OK ({result['latency']:.1f}s, ${cost:.4f}, {quality*100:.0f}% quality)")
            
        except Exception as e:
            print(f"ERROR ({str(e)[:50]})")
            results.append({
                "model": model,
                "error": str(e),
            })
    
    return {"task": task["name"], "task_key": task_key, "results": results}


def print_summary(all_results: list):
    """Print a summary table of all benchmark results."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    for task_result in all_results:
        print(f"\n{task_result['task']}:")
        print("-" * 70)
        print(f"{'Model':<30} {'Latency':>10} {'Cost':>12} {'Quality':>10}")
        print("-" * 70)
        
        valid_results = [r for r in task_result['results'] if 'error' not in r]
        for r in sorted(valid_results, key=lambda x: -x['quality']):
            print(f"{r['model']:<30} {r['latency']:>8.1f}s ${r['cost']:>10.4f} {r['quality']:>8.1f}%")
        
        # Show errors
        error_results = [r for r in task_result['results'] if 'error' in r]
        for r in error_results:
            print(f"{r['model']:<30} {'ERROR':>10} {r['error'][:30]}")


def print_recommendations(all_results: list):
    """Print model recommendations based on results."""
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Aggregate scores
    model_scores = {}
    for task_result in all_results:
        for r in task_result['results']:
            if 'error' not in r:
                if r['model'] not in model_scores:
                    model_scores[r['model']] = {'quality': [], 'cost': [], 'latency': []}
                model_scores[r['model']]['quality'].append(r['quality'])
                model_scores[r['model']]['cost'].append(r['cost'])
                model_scores[r['model']]['latency'].append(r['latency'])
    
    print("\nOverall Performance:")
    print("-" * 70)
    print(f"{'Model':<30} {'Avg Quality':>12} {'Avg Cost':>12} {'Avg Latency':>12}")
    print("-" * 70)
    
    for model, scores in model_scores.items():
        avg_q = sum(scores['quality']) / len(scores['quality'])
        avg_c = sum(scores['cost']) / len(scores['cost'])
        avg_l = sum(scores['latency']) / len(scores['latency'])
        print(f"{model:<30} {avg_q:>10.1f}% ${avg_c:>10.4f} {avg_l:>10.1f}s")
    
    print("\n💡 Suggestions:")
    print("   • Use Claude Sonnet for complex analysis (security, troubleshooting)")
    print("   • Use Haiku or GPT-4o-mini for high-volume simple tasks")
    print("   • Consider the 80/20 split: 80% cheap models, 20% quality models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Model Benchmarking Tool - Chapter 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python model_benchmark.py --list                 List available tasks
    python model_benchmark.py --task security_analysis   Run one task
    python model_benchmark.py --all                  Run all tasks
    python model_benchmark.py --all --verbose        Run all with details
        """
    )
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Run specific task")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    print("\n🔬 Network AI Model Benchmark Tool")
    print("   Chapter 3: Choosing the Right Model")
    print("=" * 70)
    
    # List tasks
    if args.list:
        print("\nAvailable benchmark tasks:")
        for key, task in TASKS.items():
            print(f"  {key:<25} {task['name']}")
            print(f"  {'':<25} {task['description']}")
            print()
        return
    
    # Check if any task specified
    if not args.task and not args.all:
        print("\nUsage:")
        print("  python model_benchmark.py --list          List tasks")
        print("  python model_benchmark.py --task <name>   Run one task")
        print("  python model_benchmark.py --all           Run all tasks")
        print("\nSet API keys first:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export OPENAI_API_KEY=sk-proj-...")
        return
    
    # Run benchmarks
    tasks_to_run = list(TASKS.keys()) if args.all else [args.task]
    all_results = []
    
    total_cost = 0
    for task_key in tasks_to_run:
        result = run_benchmark(task_key, verbose=args.verbose)
        all_results.append(result)
        for r in result['results']:
            if 'cost' in r:
                total_cost += r['cost']
    
    # Print results
    print_summary(all_results)
    print_recommendations(all_results)
    
    print(f"\n💰 Total benchmark cost: ${total_cost:.4f}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_cost": total_cost,
        "results": all_results
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"📁 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
