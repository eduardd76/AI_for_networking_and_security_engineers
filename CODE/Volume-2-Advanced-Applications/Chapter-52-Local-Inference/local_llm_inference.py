"""
Chapter 52: Local LLM Inference on Edge Devices
Run Llama, Mistral, and other models locally without cloud APIs

This script demonstrates running LLMs locally using Ollama for
network operations that require on-premises inference.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import time
import subprocess
from typing import Dict, List, Optional
import psutil
import requests
from dataclasses import dataclass

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class ModelSpecs:
    name: str
    vram_gb: float
    tokens_per_sec: int
    quality: str
    quantization: str


# Model specifications (approximate)
MODELS = {
    "llama3:8b-q4": ModelSpecs(
        name="Llama 3 8B (Q4)",
        vram_gb=6,
        tokens_per_sec=45,
        quality="Medium",
        quantization="Q4_K_M"
    ),
    "llama3:8b-q8": ModelSpecs(
        name="Llama 3 8B (Q8)",
        vram_gb=9,
        tokens_per_sec=35,
        quality="High",
        quantization="Q8_0"
    ),
    "mistral:7b-q4": ModelSpecs(
        name="Mistral 7B (Q4)",
        vram_gb=5,
        tokens_per_sec=50,
        quality="Medium",
        quantization="Q4_K_M"
    ),
    "llama3:70b-q4": ModelSpecs(
        name="Llama 3 70B (Q4)",
        vram_gb=40,
        tokens_per_sec=12,
        quality="Very High",
        quantization="Q4_K_M"
    ),
    "mixtral:8x7b-q4": ModelSpecs(
        name="Mixtral 8x7B (Q4)",
        vram_gb=26,
        tokens_per_sec=25,
        quality="High",
        quantization="Q4_K_M"
    )
}


def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_hardware_resources() -> Dict:
    """
    Example 1: Check available hardware resources
    """
    print("=" * 60)
    print("Example 1: Hardware Resource Check")
    print("=" * 60)

    resources = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "gpu_available": GPU_AVAILABLE
    }

    print(f"\nCPU: {resources['cpu_count']} cores @ {resources['cpu_percent']}% usage")
    print(f"RAM: {resources['ram_available_gb']}GB / {resources['ram_total_gb']}GB available")

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            resources["gpus"] = []

            for gpu in gpus:
                gpu_info = {
                    "name": gpu.name,
                    "vram_total_gb": round(gpu.memoryTotal / 1024, 2),
                    "vram_free_gb": round(gpu.memoryFree / 1024, 2),
                    "temperature": gpu.temperature,
                    "load": gpu.load * 100
                }
                resources["gpus"].append(gpu_info)

                print(f"\nGPU: {gpu_info['name']}")
                print(f"  VRAM: {gpu_info['vram_free_gb']}GB / {gpu_info['vram_total_gb']}GB free")
                print(f"  Temp: {gpu_info['temperature']}°C")
                print(f"  Load: {gpu_info['load']:.1f}%")
        except Exception as e:
            print(f"\nGPU detection error: {e}")
            resources["gpus"] = []
    else:
        print("\nGPU: Not detected (install py3nvml or GPUtil)")
        resources["gpus"] = []

    print("\n" + "=" * 60 + "\n")
    return resources


def model_selection_guide(available_vram_gb: float):
    """
    Example 2: Model selection based on available hardware
    """
    print("=" * 60)
    print("Example 2: Model Selection Guide")
    print("=" * 60)

    print(f"\nAvailable VRAM: {available_vram_gb}GB\n")

    print("Recommended Models:")
    print("-" * 60)

    compatible_models = []

    for model_id, specs in MODELS.items():
        fits = specs.vram_gb <= available_vram_gb
        status = "✅ Fits" if fits else "❌ Requires more VRAM"

        print(f"\n{specs.name}")
        print(f"  Required VRAM: {specs.vram_gb}GB")
        print(f"  Speed: ~{specs.tokens_per_sec} tokens/sec")
        print(f"  Quality: {specs.quality}")
        print(f"  Quantization: {specs.quantization}")
        print(f"  Status: {status}")

        if fits:
            compatible_models.append(model_id)

    if compatible_models:
        print(f"\n💡 Recommended: {MODELS[compatible_models[0]].name}")
    else:
        print("\n⚠️  No models fit in available VRAM. Consider:")
        print("  - Upgrading GPU")
        print("  - Using smaller quantization (Q2/Q3)")
        print("  - Running on CPU (much slower)")

    print("\n" + "=" * 60 + "\n")
    return compatible_models


def call_ollama_api(model: str, prompt: str) -> Dict:
    """
    Example 3: Call Ollama API for local inference
    """
    print("=" * 60)
    print("Example 3: Local Inference with Ollama")
    print("=" * 60)

    if not check_ollama_installed():
        print("\n❌ Ollama not installed!")
        print("\nInstall Ollama:")
        print("  macOS/Linux: curl https://ollama.ai/install.sh | sh")
        print("  Windows: https://ollama.ai/download")
        print("\nThen run: ollama pull llama3:8b-q4")
        return {}

    print(f"\nModel: {model}")
    print(f"Prompt: {prompt[:100]}...\n")

    try:
        # Call Ollama local API
        start_time = time.time()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return {}

        result = response.json()
        inference_time = time.time() - start_time

        print("Response:")
        print("-" * 60)
        print(result["response"])
        print("-" * 60)

        tokens_generated = result.get("eval_count", 0)
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

        print(f"\nMetrics:")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Tokens generated: {tokens_generated}")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

        print("\n" + "=" * 60 + "\n")

        return {
            "response": result["response"],
            "inference_time": inference_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_per_sec
        }

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("Start Ollama: ollama serve")
        return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def network_config_analysis_local():
    """
    Example 4: Analyze network config using local LLM
    """
    print("=" * 60)
    print("Example 4: Network Config Analysis (Local)")
    print("=" * 60)

    config = """
    interface GigabitEthernet0/1
     description WAN Uplink
     ip address 203.0.113.1 255.255.255.252
     no shutdown
    !
    router bgp 65001
     neighbor 203.0.113.2 remote-as 65000
     network 192.168.1.0 mask 255.255.255.0
    !
    line vty 0 4
     transport input telnet
     password cisco123
    """

    prompt = f"""Analyze this network configuration for security issues:

{config}

List any security problems and recommend fixes. Be concise."""

    # Use Llama 3 8B (most commonly available)
    result = call_ollama_api("llama3:8b-q4", prompt)

    if result:
        print("💡 Analysis complete!")
        print(f"   Time: {result['inference_time']:.1f}s")
        print(f"   Speed: {result['tokens_per_sec']:.0f} tokens/sec")

    return result


def batch_log_analysis():
    """
    Example 5: Batch analyze syslogs using local LLM
    """
    print("=" * 60)
    print("Example 5: Batch Syslog Analysis")
    print("=" * 60)

    logs = [
        "%BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down - Hold timer expired",
        "%LINEPROTO-5-UPDOWN: Line protocol on GigabitEthernet0/1, changed state to down",
        "%SYS-5-CONFIG_I: Configured from console by admin on vty0",
        "%SEC-6-IPACCESSLOGP: list 100 denied tcp 192.168.1.50 -> 10.0.0.1"
    ]

    print(f"\nAnalyzing {len(logs)} syslog messages...\n")

    results = []
    total_time = 0

    for i, log in enumerate(logs, 1):
        prompt = f"Classify this syslog as CRITICAL, HIGH, MEDIUM, or LOW severity. Reply with only the severity level:\n\n{log}"

        print(f"{i}. {log[:60]}...")

        result = call_ollama_api("llama3:8b-q4", prompt)

        if result:
            severity = result["response"].strip().split()[0]
            results.append({
                "log": log,
                "severity": severity,
                "time": result["inference_time"]
            })
            total_time += result["inference_time"]
            print(f"   → {severity} ({result['inference_time']:.1f}s)\n")

    print("=" * 60)
    print(f"Batch complete: {len(results)}/{len(logs)} analyzed")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time: {total_time/len(results):.1f}s per log")
    print("=" * 60 + "\n")

    return results


def quantization_comparison():
    """
    Example 6: Compare quantization levels
    """
    print("=" * 60)
    print("Example 6: Quantization Comparison")
    print("=" * 60)

    print("""
Quantization reduces model size and VRAM usage by reducing precision:

FP32 (Full Precision):
  - Size: 100%
  - VRAM: 100%
  - Quality: 100%
  - Speed: Baseline

FP16 (Half Precision):
  - Size: 50%
  - VRAM: 50%
  - Quality: 99%
  - Speed: 1.5-2x faster

Q8 (8-bit Quantization):
  - Size: 25%
  - VRAM: 25%
  - Quality: 95-98%
  - Speed: 2-3x faster

Q4 (4-bit Quantization):
  - Size: 12.5%
  - VRAM: 12.5%
  - Quality: 85-95%
  - Speed: 3-4x faster

Q2 (2-bit Quantization):
  - Size: 6.25%
  - VRAM: 6.25%
  - Quality: 70-85%
  - Speed: 4-5x faster

Recommendations:
- For network operations: Q4 is the sweet spot
- For classification tasks: Q4 or even Q2 works well
- For critical analysis: Use Q8 or FP16
- For large context: Q4 allows running bigger models

Example: Llama 3 70B
  - FP32: 280GB VRAM (impossible on consumer GPU)
  - Q8: 70GB (needs A100 80GB or 2x 4090)
  - Q4: 35GB (fits on single A100 40GB or 2x 3090)
  - Q2: 17.5GB (fits on single 4090)
""")

    print("=" * 60 + "\n")


def production_deployment_guide():
    """
    Example 7: Production deployment considerations
    """
    print("=" * 60)
    print("Example 7: Production Deployment Guide")
    print("=" * 60)

    guide = """
PRODUCTION DEPLOYMENT OPTIONS:

Option 1: Single GPU Server
├── Hardware: Single NVIDIA A100 (40GB)
├── Models: Llama 3 70B Q4, Mixtral 8x7B Q4
├── Use Case: Central AI server for network team
├── Cost: $10-20K (one-time) + $300/mo power
└── Pros: Simple, centralized, no cloud costs

Option 2: Multi-GPU Workstation
├── Hardware: 2x RTX 4090 (48GB total)
├── Models: Llama 3 70B Q4, Multiple smaller models
├── Use Case: Team workstation with GPU pooling
├── Cost: $5-8K (one-time) + $200/mo power
└── Pros: Developer-friendly, good performance

Option 3: Consumer GPU (Budget)
├── Hardware: Single RTX 3060 (12GB)
├── Models: Llama 3 8B, Mistral 7B (Q4/Q8)
├── Use Case: Individual use, testing, development
├── Cost: $300-500 (one-time) + $50/mo power
└── Pros: Affordable, sufficient for most tasks

Option 4: CPU-Only (No GPU)
├── Hardware: High-core-count CPU (32+ cores)
├── Models: Smaller models only (7B-13B)
├── Use Case: Air-gapped environments, extreme budget
├── Cost: Included with server
└── Cons: 10-50x slower than GPU

WHEN TO USE LOCAL vs CLOUD:

Use Local LLMs When:
✅ Data cannot leave premises (compliance, security)
✅ High volume of requests (100K+/day)
✅ Low latency required (< 100ms)
✅ Predictable, constant workload
✅ Long-term deployment (> 1 year ROI)

Use Cloud APIs When:
✅ Variable workload (spikes and valleys)
✅ Low volume (< 10K requests/day)
✅ Need multiple models frequently
✅ Want zero infrastructure management
✅ Short-term projects (< 6 months)

COST COMPARISON (100K requests/day):

Cloud (OpenAI GPT-4):
  - Cost: $5,000-10,000/month
  - Break-even: Never (always expensive at scale)

Local (Llama 3 70B on A100):
  - Hardware: $15,000 (one-time)
  - Power: $300/month
  - Break-even: 3-6 months
  - Year 1 Total: $18,600
  - Year 2+ Total: $3,600/year

For network operations with 100K+ AI calls/day,
local inference pays for itself in 3-6 months.
"""

    print(guide)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\n💻 Chapter 52: Local LLM Inference")
    print("Run Models On-Premises Without Cloud APIs\n")

    try:
        # Run examples
        resources = check_hardware_resources()
        input("Press Enter to continue...")

        # Get VRAM (or estimate if no GPU)
        vram_available = 12  # Default estimate
        if resources.get("gpus"):
            vram_available = resources["gpus"][0]["vram_free_gb"]

        compatible_models = model_selection_guide(vram_available)
        input("Press Enter to continue...")

        # Only run inference examples if Ollama is installed
        if check_ollama_installed():
            network_config_analysis_local()
            input("Press Enter to continue...")

            print("\n⚠️  Batch analysis will take several minutes...")
            proceed = input("Run batch analysis? (y/n): ")
            if proceed.lower() == 'y':
                batch_log_analysis()
        else:
            print("\n💡 Install Ollama to run inference examples:")
            print("   https://ollama.ai\n")

        quantization_comparison()
        input("Press Enter to continue...")

        production_deployment_guide()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Local inference is cost-effective at scale")
        print("- Q4 quantization is the sweet spot for network ops")
        print("- 12GB VRAM handles most practical use cases")
        print("- Ollama makes local deployment simple")
        print("- Break-even vs cloud is 3-6 months at 100K req/day\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
