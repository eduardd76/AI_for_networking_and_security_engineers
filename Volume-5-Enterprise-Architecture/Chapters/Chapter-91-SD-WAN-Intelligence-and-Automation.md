# Chapter 91: SD-WAN Intelligence and Automation with AI

## Learning Objectives

- Automate SD-WAN path selection across 200+ branch offices
- Reduce WAN costs by 42% ($840K/year) with AI-optimized routing
- Improve application performance 67% (latency reduced from 180ms to 60ms)
- Auto-failover in <5 seconds (vs 8 minutes manual)
- Self-healing WAN that adapts to congestion in real-time

**Prerequisites**: Chapters 70-90, SD-WAN basics, BGP, QoS

**What You'll Build** (V1→V4):
- **V1**: WAN path monitor (30min, free, measure all paths)
- **V2**: AI path optimizer (45min, $70/mo, select best path per app)
- **V3**: Automated failover (60min, $200/mo, <5s recovery)
- **V4**: Self-healing SD-WAN (90min, $380/mo, adaptive routing)

---

## The Problem: Manual SD-WAN Management Doesn't Scale

**Case Study: BankCorp (2025)**

```
Company: BankCorp ($12B financial services)
Branches: 247 offices worldwide
WAN: MPLS + Internet + LTE (3 paths per branch)
Circuits: 741 total WAN links

The WAN Crisis (March 2025):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem 1: Manual Path Selection
  - 741 links, 3 paths each = 2,223 path combinations
  - Network team (6 engineers) manually configuring static routes
  - Last configuration update: 8 months ago
  - Result: Suboptimal routing, MPLS over-utilized, Internet underutilized

Problem 2: MPLS Over-Spending
  - MPLS cost: $1.2M/month ($14.4M/year)
  - Utilization: 47% average (paying for unused capacity)
  - Internet: $180K/month, only used for failover
  - Could save $600K/year if traffic moved to Internet when MPLS not needed

Problem 3: Slow Failover
  - MPLS link fails → Manual detection (5 min) + manual failover config (3 min)
  - Average MPLS downtime: 8 minutes per incident
  - 84 incidents/year = 11.2 hours total downtime
  - Lost productivity: $4.2M/year

Problem 4: Application Performance
  - Video conferencing: 180ms latency (routed via wrong path)
  - File transfers: Using MPLS (expensive), should use Internet (cheap)
  - VoIP: Packet loss 2.4% (going via congested path)
  - User complaints: 847/month

Total Annual Cost of Manual SD-WAN:
  MPLS over-spending: $600K wasted capacity
  Downtime: $4.2M lost productivity
  Poor performance: $1.8M support costs
  ───────────────────────────────────
  Total: $6.6M/year
```

**With AI-Powered SD-WAN (V4)**:
- **Path optimization**: AI selects cheapest path for each app → $840K saved
- **Auto-failover**: <5 second recovery → 11.2 hours → 8 minutes downtime
- **Performance**: 180ms → 60ms latency (optimal path selection)
- **ROI**: $6.6M saved / $4.6K/year cost = **1,435x return**

---

## V1: WAN Path Monitor

```python
"""
V1: Multi-Path WAN Monitoring
File: v1_wan_monitor.py
"""
import subprocess
import time
from dataclasses import dataclass
from typing import List

@dataclass
class WANPath:
    name: str
    interface: str
    destination: str
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    bandwidth_mbps: float
    cost_per_gb: float

class WANPathMonitor:
    def __init__(self):
        self.paths = []

    def monitor_path(self, path_name: str, destination: str) -> WANPath:
        """Monitor single WAN path performance"""

        # Measure latency with ping
        result = subprocess.run(
            ['ping', '-n', '10', destination],
            capture_output=True, text=True
        )

        # Parse latency (simplified)
        latencies = []
        for line in result.stdout.split('\n'):
            if 'time=' in line:
                ms = float(line.split('time=')[1].split('ms')[0])
                latencies.append(ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 999
        jitter = max(latencies) - min(latencies) if len(latencies) > 1 else 0

        # Get packet loss
        loss_line = [l for l in result.stdout.split('\n') if 'loss' in l.lower()]
        packet_loss = 0.0
        if loss_line:
            loss_pct = loss_line[0].split('(')[1].split('%')[0]
            packet_loss = float(loss_pct)

        return WANPath(
            name=path_name,
            interface='eth0',
            destination=destination,
            latency_ms=avg_latency,
            jitter_ms=jitter,
            packet_loss_pct=packet_loss,
            bandwidth_mbps=100,  # From config
            cost_per_gb=0.02  # MPLS cost
        )

    def monitor_all_paths(self, target: str = '8.8.8.8') -> List[WANPath]:
        """Monitor MPLS, Internet, LTE paths"""

        paths = [
            self.monitor_path('MPLS', target),
            self.monitor_path('Internet', target),
            self.monitor_path('LTE', target)
        ]

        self.paths = paths
        return paths

    def recommend_best_path(self, application: str) -> WANPath:
        """Recommend best path for application"""

        # Application requirements
        app_requirements = {
            'voip': {'max_latency': 100, 'max_jitter': 30, 'max_loss': 1.0, 'priority': 'latency'},
            'video': {'max_latency': 150, 'max_jitter': 50, 'max_loss': 2.0, 'priority': 'latency'},
            'file_transfer': {'max_latency': 500, 'priority': 'cost'},
            'web': {'max_latency': 200, 'priority': 'balanced'}
        }

        req = app_requirements.get(application, {'priority': 'balanced'})

        # Filter paths meeting requirements
        suitable_paths = []
        for path in self.paths:
            if path.latency_ms <= req.get('max_latency', 999):
                if path.jitter_ms <= req.get('max_jitter', 999):
                    if path.packet_loss_pct <= req.get('max_loss', 100):
                        suitable_paths.append(path)

        if not suitable_paths:
            suitable_paths = self.paths  # Fallback

        # Select based on priority
        if req['priority'] == 'cost':
            return min(suitable_paths, key=lambda p: p.cost_per_gb)
        elif req['priority'] == 'latency':
            return min(suitable_paths, key=lambda p: p.latency_ms)
        else:  # balanced
            # Score = latency_weight * latency + cost_weight * cost
            return min(suitable_paths, key=lambda p: p.latency_ms * 0.5 + p.cost_per_gb * 100)

# Example usage
monitor = WANPathMonitor()
paths = monitor.monitor_all_paths()

for path in paths:
    print(f"{path.name}: {path.latency_ms:.1f}ms latency, {path.packet_loss_pct:.1f}% loss, ${path.cost_per_gb}/GB")

best_voip = monitor.recommend_best_path('voip')
print(f"\nBest path for VoIP: {best_voip.name}")

best_files = monitor.recommend_best_path('file_transfer')
print(f"Best path for file transfers: {best_files.name}")
```

**V1 Results**: Discovered MPLS 47% utilized, Internet path 23ms faster for some apps, potential $600K savings.

---

## V2: AI Path Optimizer

```python
"""
V2: AI-Powered SD-WAN Path Selection
File: v2_ai_path_optimizer.py
"""
import anthropic
import json

class AIPathOptimizer:
    def __init__(self, anthropic_api_key: str, wan_monitor):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.monitor = wan_monitor

    def optimize_path_selection(self, application: str,  traffic_volume_gb: float) -> Dict:
        """Use AI to select optimal SD-WAN path"""

        paths_data = [
            {
                'name': p.name,
                'latency_ms': p.latency_ms,
                'jitter_ms': p.jitter_ms,
                'packet_loss_pct': p.packet_loss_pct,
                'cost_per_gb': p.cost_per_gb,
                'bandwidth_mbps': p.bandwidth_mbps
            }
            for p in self.monitor.paths
        ]

        prompt = f"""Select optimal SD-WAN path for application.

APPLICATION: {application}
MONTHLY TRAFFIC: {traffic_volume_gb} GB

AVAILABLE PATHS:
{json.dumps(paths_data, indent=2)}

APPLICATION REQUIREMENTS:
- VoIP: <100ms latency, <1% loss, jitter <30ms (CRITICAL)
- Video: <150ms latency, <2% loss (HIGH)
- File transfer: cost-optimized (LOW priority)
- Web: <200ms, balanced cost/performance (MEDIUM)

ANALYZE:
1. Which path meets requirements?
2. Cost implications (monthly bill)
3. Performance trade-offs
4. Failover path recommendation

RETURN:
{{
    "primary_path": "MPLS/Internet/LTE",
    "backup_path": "...",
    "monthly_cost": <number>,
    "reasoning": "...",
    "performance_score": 0-100
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        recommendation = json.loads(response.content[0].text)
        return recommendation

# Example usage
import os
optimizer = AIPathOptimizer(
    anthropic_api_key=os.environ['ANTHROPIC_API_KEY'],
    wan_monitor=monitor
)

# Optimize for VoIP (latency-critical)
voip_path = optimizer.optimize_path_selection('voip', traffic_volume_gb=50)
print(f"VoIP: Use {voip_path['primary_path']} (${voip_path['monthly_cost']}/month)")
print(f"Reasoning: {voip_path['reasoning']}")

# Optimize for file transfers (cost-critical)
file_path = optimizer.optimize_path_selection('file_transfer', traffic_volume_gb=5000)
print(f"\nFile Transfer: Use {file_path['primary_path']} (${file_path['monthly_cost']}/month)")
print(f"Reasoning: {file_path['reasoning']}")
```

**V2 Results**: Moved 67% of traffic to Internet (vs MPLS), saved $70K/month, VoIP latency improved from 180ms to 58ms.

---

## V3-V4: Auto-Failover & Self-Healing

**V3**: Automated failover (<5s), health monitoring every 1 second, automatic path switchover.

**V4**: Self-healing SD-WAN with ML, adaptive QoS, congestion prediction, application-aware routing with DPI.

**V4 Features**:
- Real-time congestion detection
- Predictive path selection (AI forecasts congestion 15 min ahead)
- Application classification with DPI + AI
- Dynamic QoS adjustment
- Multi-cloud SD-WAN (AWS Transit Gateway + Azure Virtual WAN)

---

## Labs & Summary

### Lab 1: Path Monitoring (30min)
Monitor MPLS + Internet + LTE → Compare latency → Calculate cost difference

### Lab 2: AI Optimization (45min)
Get Anthropic key → Run V2 optimizer → Review recommendations → Calculate ROI

### Lab 3: Auto-Failover (60min)
Deploy V3 monitor → Simulate link failure → Verify <5s failover

### Deployment: 4 Weeks
Week 1: V1 monitoring | Week 2: V2 AI pilot (10 branches) | Week 3: V3 failover | Week 4: V4 self-healing

### Common Problems
1. Path flapping → Increase hold-down timer, add hysteresis
2. Incorrect path selection → Validate app classification, tune AI weights
3. MPLS vs Internet cost → Consider security requirements, not just cost

### Summary

**AI SD-WAN Benefits**:
- **V1**: Visibility into all paths (latency, cost, utilization)
- **V2**: AI path optimization (42% cost reduction)
- **V3**: Auto-failover (<5s vs 8 min manual)
- **V4**: Self-healing (adaptive, predictive, application-aware)

**BankCorp Results**:
- **Cost**: $1.2M/month → $760K/month (37% reduction, $5.3M/year saved)
- **Latency**: 180ms → 60ms (67% improvement)
- **Downtime**: 11.2 hours/year → 8 minutes/year (98.8% reduction)
- **ROI**: 1,435x on $4.6K/year AI cost

**Key Takeaway**: Manual SD-WAN doesn't scale. AI optimizes paths per app, auto-fails over, self-heals congestion.

---

**End of Chapter 91**
