#!/usr/bin/env python3
"""
Cost Analyzer - Analyze and Forecast AI API Costs

Track, analyze, and forecast API costs. Identify cost optimization opportunities.

From: AI for Networking Engineers - Volume 1, Chapter 8
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from cost_analyzer import CostAnalyzer

    analyzer = CostAnalyzer()
    analysis = analyzer.analyze_usage("usage_log.jsonl")
    print(f"Monthly forecast: {analysis['monthly_forecast']}")
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics


@dataclass
class CostBreakdown:
    """Cost breakdown by category."""
    model: str
    requests: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    avg_cost_per_request: float


class CostAnalyzer:
    """
    Analyze API usage costs and provide optimization recommendations.

    Features:
    - Cost breakdown by model, task, time period
    - Monthly forecasting
    - Optimization suggestions
    - Cost trend analysis
    """

    # Model pricing (per million tokens, January 2026)
    PRICING = {
        'claude-3-5-sonnet': {'input': 3.0, 'output': 15.0},
        'claude-3-5-haiku': {'input': 0.80, 'output': 4.0},
        'claude-opus-3-5': {'input': 15.0, 'output': 75.0},
        'gpt-4': {'input': 10.0, 'output': 30.0},
        'gpt-4-turbo': {'input': 5.0, 'output': 15.0},
    }

    def analyze_usage(
        self,
        log_file: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze usage from log file.

        Args:
            log_file: Path to usage log (JSONL format)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Comprehensive cost analysis
        """
        log_path = Path(log_file)
        if not log_path.exists():
            return {"error": f"Log file not found: {log_file}"}

        # Parse log entries
        entries = self._parse_log_file(log_path, start_date, end_date)

        if not entries:
            return {"error": "No entries found in specified date range"}

        # Calculate metrics
        total_cost = sum(e['cost'] for e in entries)
        total_requests = len(entries)
        total_tokens = sum(e['total_tokens'] for e in entries)

        # Breakdown by model
        by_model = self._breakdown_by_model(entries)

        # Breakdown by task
        by_task = self._breakdown_by_task(entries)

        # Daily costs
        daily_costs = self._calculate_daily_costs(entries)

        # Forecasting
        forecast = self._forecast_monthly_cost(daily_costs)

        # Optimization opportunities
        optimizations = self._suggest_optimizations(entries, by_model)

        return {
            "period": self._get_period_description(entries),
            "summary": {
                "total_cost": f"${total_cost:.4f}",
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "avg_cost_per_request": f"${total_cost/total_requests:.6f}",
                "avg_tokens_per_request": int(total_tokens/total_requests)
            },
            "by_model": by_model,
            "by_task": by_task,
            "daily_costs": daily_costs,
            "forecast": forecast,
            "optimizations": optimizations
        }

    def _parse_log_file(
        self,
        log_path: Path,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Parse and filter log entries."""
        entries = []

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Parse timestamp
                    timestamp = datetime.fromisoformat(entry['timestamp'])

                    # Apply date filters
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue

                    entries.append(entry)

                except (json.JSONDecodeError, KeyError) as e:
                    continue

        return entries

    def _breakdown_by_model(self, entries: List[Dict[str, Any]]) -> List[CostBreakdown]:
        """Break down costs by model."""
        by_model = {}

        for entry in entries:
            model = entry.get('model', 'unknown')

            if model not in by_model:
                by_model[model] = {
                    'requests': 0,
                    'cost': 0.0,
                    'input_tokens': 0,
                    'output_tokens': 0
                }

            by_model[model]['requests'] += 1
            by_model[model]['cost'] += entry.get('cost', 0.0)
            by_model[model]['input_tokens'] += entry.get('input_tokens', 0)
            by_model[model]['output_tokens'] += entry.get('output_tokens', 0)

        # Convert to CostBreakdown objects
        breakdowns = []
        for model, data in by_model.items():
            avg_cost = data['cost'] / data['requests'] if data['requests'] > 0 else 0
            breakdowns.append(CostBreakdown(
                model=model,
                requests=data['requests'],
                total_cost=data['cost'],
                input_tokens=data['input_tokens'],
                output_tokens=data['output_tokens'],
                avg_cost_per_request=avg_cost
            ))

        # Sort by cost descending
        breakdowns.sort(key=lambda x: x.total_cost, reverse=True)

        return [
            {
                "model": b.model,
                "requests": b.requests,
                "total_cost": f"${b.total_cost:.4f}",
                "input_tokens": b.input_tokens,
                "output_tokens": b.output_tokens,
                "avg_cost": f"${b.avg_cost_per_request:.6f}"
            }
            for b in breakdowns
        ]

    def _breakdown_by_task(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Break down costs by task type."""
        by_task = {}

        for entry in entries:
            metadata = entry.get('metadata', {})
            task = metadata.get('task', 'unknown') if isinstance(metadata, dict) else 'unknown'

            if task not in by_task:
                by_task[task] = {'requests': 0, 'cost': 0.0}

            by_task[task]['requests'] += 1
            by_task[task]['cost'] += entry.get('cost', 0.0)

        # Convert to list and sort
        task_list = [
            {
                "task": task,
                "requests": data['requests'],
                "total_cost": f"${data['cost']:.4f}",
                "avg_cost": f"${data['cost']/data['requests']:.6f}"
            }
            for task, data in by_task.items()
        ]

        task_list.sort(key=lambda x: float(x['total_cost'].replace('$', '')), reverse=True)
        return task_list

    def _calculate_daily_costs(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate costs by day."""
        daily = {}

        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            date_key = timestamp.date().isoformat()

            if date_key not in daily:
                daily[date_key] = {'cost': 0.0, 'requests': 0}

            daily[date_key]['cost'] += entry.get('cost', 0.0)
            daily[date_key]['requests'] += 1

        # Convert to sorted list
        daily_list = [
            {
                "date": date,
                "cost": f"${data['cost']:.4f}",
                "requests": data['requests']
            }
            for date, data in sorted(daily.items())
        ]

        return daily_list

    def _forecast_monthly_cost(self, daily_costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Forecast monthly cost based on recent usage."""
        if not daily_costs:
            return {"error": "No data for forecasting"}

        # Extract costs (remove $ sign)
        costs = [float(d['cost'].replace('$', '')) for d in daily_costs]

        # Calculate statistics
        avg_daily = statistics.mean(costs)
        median_daily = statistics.median(costs)

        if len(costs) > 1:
            stdev_daily = statistics.stdev(costs)
        else:
            stdev_daily = 0

        # Forecast
        monthly_avg = avg_daily * 30
        monthly_median = median_daily * 30
        monthly_high = (avg_daily + stdev_daily) * 30
        monthly_low = max(0, (avg_daily - stdev_daily) * 30)

        return {
            "sample_days": len(costs),
            "avg_daily_cost": f"${avg_daily:.2f}",
            "median_daily_cost": f"${median_daily:.2f}",
            "monthly_forecast_avg": f"${monthly_avg:.2f}",
            "monthly_forecast_median": f"${monthly_median:.2f}",
            "monthly_forecast_range": f"${monthly_low:.2f} - ${monthly_high:.2f}"
        }

    def _suggest_optimizations(
        self,
        entries: List[Dict[str, Any]],
        by_model: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest cost optimization opportunities."""
        suggestions = []

        # Check if using expensive models for simple tasks
        if by_model:
            most_used = by_model[0]
            model_name = most_used['model']

            if 'opus' in model_name.lower():
                sonnet_cost = float(most_used['total_cost'].replace('$', '')) * 0.2  # 5x cheaper
                savings = float(most_used['total_cost'].replace('$', '')) - sonnet_cost
                suggestions.append({
                    "optimization": "Switch from Opus to Sonnet",
                    "description": f"Using Opus for {most_used['requests']} requests",
                    "potential_savings": f"${savings:.2f}/month",
                    "action": "Use Sonnet for routine tasks, Opus only for complex analysis"
                })

            elif 'sonnet' in model_name.lower():
                haiku_cost = float(most_used['total_cost'].replace('$', '')) * 0.27  # ~4x cheaper
                savings = float(most_used['total_cost'].replace('$', '')) - haiku_cost
                suggestions.append({
                    "optimization": "Use Haiku for simple tasks",
                    "description": f"Some Sonnet requests may not need advanced reasoning",
                    "potential_savings": f"${savings * 0.3:.2f}/month (30% of requests)",
                    "action": "Use Haiku for log classification, simple parsing"
                })

        # Check for caching opportunities
        total_requests = len(entries)
        if total_requests > 100:
            cache_savings = sum(e.get('cost', 0) for e in entries) * 0.4  # Assume 40% cache hit rate
            suggestions.append({
                "optimization": "Implement response caching",
                "description": f"Processing {total_requests} requests without caching",
                "potential_savings": f"${cache_savings:.2f}/month (40% hit rate)",
                "action": "Cache deterministic responses (temperature=0)"
            })

        # Check for batch processing opportunities
        if total_requests > 50:
            suggestions.append({
                "optimization": "Use batch processing",
                "description": "Processing items individually",
                "potential_savings": "20-30% (reduced overhead)",
                "action": "Batch similar requests together"
            })

        return suggestions

    def _get_period_description(self, entries: List[Dict[str, Any]]) -> str:
        """Get human-readable period description."""
        if not entries:
            return "No data"

        timestamps = [datetime.fromisoformat(e['timestamp']) for e in entries]
        start = min(timestamps)
        end = max(timestamps)

        days = (end - start).days + 1

        return f"{start.date()} to {end.date()} ({days} days)"


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Cost Analyzer Demo
    ========================================
    Analyze and forecast API costs
    ========================================
    """)

    # Create sample usage log
    sample_log_file = "sample_usage.jsonl"

    # Generate sample data
    print(f"\nGenerating sample usage data...")
    with open(sample_log_file, 'w') as f:
        for i in range(30):  # 30 days
            date = datetime.now() - timedelta(days=29-i)

            # Generate 5-10 requests per day
            import random
            num_requests = random.randint(5, 10)

            for j in range(num_requests):
                entry = {
                    "timestamp": (date + timedelta(hours=j)).isoformat(),
                    "model": random.choice(['claude-3-5-sonnet', 'claude-3-5-haiku']),
                    "input_tokens": random.randint(100, 500),
                    "output_tokens": random.randint(50, 200),
                    "cost": random.uniform(0.001, 0.01),
                    "metadata": {"task": random.choice(['config_analysis', 'log_classification', 'troubleshooting'])}
                }
                f.write(json.dumps(entry) + '\n')

    # Analyze
    analyzer = CostAnalyzer()
    analysis = analyzer.analyze_usage(sample_log_file)

    # Display results
    print(f"\n{'='*60}")
    print("COST ANALYSIS")
    print(f"{'='*60}")

    print(f"\nPeriod: {analysis['period']}")

    print(f"\nSummary:")
    for key, value in analysis['summary'].items():
        print(f"  {key:25s}: {value}")

    print(f"\nCosts by Model:")
    for model_data in analysis['by_model']:
        print(f"  {model_data['model']:20s}: {model_data['total_cost']:>10s} ({model_data['requests']} requests)")

    print(f"\nCosts by Task:")
    for task_data in analysis['by_task'][:3]:
        print(f"  {task_data['task']:20s}: {task_data['total_cost']:>10s}")

    print(f"\nMonthly Forecast:")
    for key, value in analysis['forecast'].items():
        print(f"  {key:25s}: {value}")

    print(f"\n{'='*60}")
    print("OPTIMIZATION OPPORTUNITIES")
    print(f"{'='*60}")

    for i, opt in enumerate(analysis['optimizations'], 1):
        print(f"\n{i}. {opt['optimization']}")
        print(f"   {opt['description']}")
        print(f"   Potential savings: {opt['potential_savings']}")
        print(f"   Action: {opt['action']}")

    # Clean up
    Path(sample_log_file).unlink()

    print("\n✅ Demo complete!")
    print("\n💡 Cost Optimization Strategy:")
    print("  1. Use Haiku for simple tasks (4x cheaper than Sonnet)")
    print("  2. Implement caching (50-70% savings)")
    print("  3. Batch similar requests")
    print("  4. Monitor daily costs")
    print("  5. Set budget alerts")
