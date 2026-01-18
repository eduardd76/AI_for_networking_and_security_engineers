#!/usr/bin/env python3
"""
API usage tracker with cost monitoring and analysis.

From: AI for Networking Engineers - Volume 1, Chapter 4
Author: Eduard Dulharu (Ed Harmoosh)

Track API usage, costs, and generate reports for budgeting and optimization.

Usage:
    from usage_tracker import UsageTracker

    tracker = UsageTracker()

    # Log a request
    tracker.log_request(
        model="claude-3-5-sonnet",
        input_tokens=100,
        output_tokens=200,
        cost=0.003,
        latency=2.5,
        success=True,
        task="config_analysis"
    )

    # Get summary
    summary = tracker.get_summary(hours=24)
    print(summary)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class UsageTracker:
    """Track API usage and costs for monitoring and budgeting."""

    def __init__(self, log_file: str = "api_usage.jsonl"):
        """
        Initialize usage tracker.

        Args:
            log_file: Path to log file (JSONL format)
        """
        self.log_file = Path(log_file)

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"UsageTracker initialized (log_file={self.log_file})")

    def log_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency: float,
        success: bool,
        task: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an API request.

        Args:
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Cost in USD
            latency: Response time in seconds
            success: Whether request succeeded
            task: Description of task
            metadata: Additional metadata
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "latency": latency,
            "success": success,
            "task": task,
            "metadata": metadata or {}
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            logger.debug(f"Logged request: {task} ({input_tokens + output_tokens} tokens, ${cost:.6f})")

        except Exception as e:
            logger.error(f"Failed to log request: {e}")

    def get_summary(
        self,
        hours: Optional[int] = None,
        days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get usage summary for a time period.

        Args:
            hours: Look back N hours (default: None)
            days: Look back N days (default: None)
            start_date: Start date (default: None)
            end_date: End date (default: now)

        Returns:
            Summary statistics including:
            - time_period
            - total_requests
            - successful_requests
            - failed_requests
            - total_cost
            - total_tokens
            - avg_cost_per_request
            - avg_latency
            - requests_by_model
            - requests_by_task
        """
        if not self.log_file.exists():
            return {"error": "No usage data"}

        # Determine time range
        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            time_desc = f"Last {hours} hours"
        elif days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            time_desc = f"Last {days} days"
        elif start_date:
            cutoff = start_date
            end_date = end_date or datetime.utcnow()
            time_desc = f"{start_date.date()} to {end_date.date()}"
        else:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            time_desc = "Last 24 hours"

        # Process log file
        total_cost = 0.0
        total_requests = 0
        total_tokens = 0
        successful = 0
        failed = 0
        total_latency = 0.0

        requests_by_model = {}
        requests_by_task = {}
        cost_by_model = {}
        cost_by_task = {}

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        timestamp = datetime.fromisoformat(entry['timestamp'])

                        # Check if within time range
                        if timestamp < cutoff:
                            continue
                        if end_date and timestamp > end_date:
                            continue

                        # Aggregate stats
                        total_requests += 1
                        total_cost += entry['cost']
                        total_tokens += entry['total_tokens']
                        total_latency += entry['latency']

                        if entry['success']:
                            successful += 1
                        else:
                            failed += 1

                        # By model
                        model = entry['model']
                        requests_by_model[model] = requests_by_model.get(model, 0) + 1
                        cost_by_model[model] = cost_by_model.get(model, 0.0) + entry['cost']

                        # By task
                        task = entry['task']
                        requests_by_task[task] = requests_by_task.get(task, 0) + 1
                        cost_by_task[task] = cost_by_task.get(task, 0.0) + entry['cost']

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping invalid log entry: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return {"error": str(e)}

        # Calculate averages
        avg_cost = total_cost / max(total_requests, 1)
        avg_latency = total_latency / max(successful, 1)
        success_rate = (successful / max(total_requests, 1)) * 100

        # Sort by cost descending
        requests_by_model_sorted = dict(
            sorted(requests_by_model.items(), key=lambda x: cost_by_model[x[0]], reverse=True)
        )
        requests_by_task_sorted = dict(
            sorted(requests_by_task.items(), key=lambda x: cost_by_task[x[0]], reverse=True)
        )

        return {
            "time_period": time_desc,
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": f"{success_rate:.1f}%",
            "total_cost": f"${total_cost:.4f}",
            "total_tokens": total_tokens,
            "avg_cost_per_request": f"${avg_cost:.6f}",
            "avg_latency": f"{avg_latency:.2f}s",
            "requests_by_model": requests_by_model_sorted,
            "cost_by_model": {k: f"${v:.4f}" for k, v in cost_by_model.items()},
            "requests_by_task": requests_by_task_sorted,
            "cost_by_task": {k: f"${v:.4f}" for k, v in cost_by_task.items()}
        }

    def get_hourly_breakdown(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get hourly breakdown of usage.

        Args:
            hours: Number of hours to analyze

        Returns:
            List of dicts with hourly statistics
        """
        if not self.log_file.exists():
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Initialize hourly buckets
        hourly_stats = {}
        for i in range(hours):
            hour_start = datetime.utcnow() - timedelta(hours=hours - i)
            hour_key = hour_start.strftime("%Y-%m-%d %H:00")
            hourly_stats[hour_key] = {
                "hour": hour_key,
                "requests": 0,
                "cost": 0.0,
                "tokens": 0
            }

        # Process log file
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        timestamp = datetime.fromisoformat(entry['timestamp'])

                        if timestamp < cutoff:
                            continue

                        # Bucket into hour
                        hour_key = timestamp.strftime("%Y-%m-%d %H:00")

                        if hour_key in hourly_stats:
                            hourly_stats[hour_key]["requests"] += 1
                            hourly_stats[hour_key]["cost"] += entry['cost']
                            hourly_stats[hour_key]["tokens"] += entry['total_tokens']

                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []

        # Convert to list and format
        result = []
        for hour_data in hourly_stats.values():
            result.append({
                "hour": hour_data["hour"],
                "requests": hour_data["requests"],
                "cost": f"${hour_data['cost']:.4f}",
                "tokens": hour_data["tokens"]
            })

        return result

    def estimate_monthly_cost(self, days_sample: int = 7) -> Dict[str, Any]:
        """
        Estimate monthly cost based on recent usage.

        Args:
            days_sample: Number of days to use for estimation

        Returns:
            Dict with current_rate, estimated_monthly, estimated_annual
        """
        summary = self.get_summary(days=days_sample)

        if "error" in summary:
            return summary

        # Parse cost
        total_cost_str = summary["total_cost"]
        total_cost = float(total_cost_str.replace("$", ""))

        # Calculate daily average
        daily_avg = total_cost / days_sample

        # Project to month and year
        monthly_estimate = daily_avg * 30
        annual_estimate = daily_avg * 365

        return {
            "sample_period": f"{days_sample} days",
            "sample_cost": f"${total_cost:.2f}",
            "daily_avg": f"${daily_avg:.2f}",
            "estimated_monthly": f"${monthly_estimate:.2f}",
            "estimated_annual": f"${annual_estimate:.2f}"
        }

    def check_budget(self, daily_limit: float) -> Dict[str, Any]:
        """
        Check if current usage is within daily budget.

        Args:
            daily_limit: Daily budget limit in USD

        Returns:
            Dict with status, usage, remaining
        """
        summary = self.get_summary(hours=24)

        if "error" in summary:
            return summary

        # Parse cost
        today_cost_str = summary["total_cost"]
        today_cost = float(today_cost_str.replace("$", ""))

        remaining = daily_limit - today_cost
        utilization = (today_cost / daily_limit) * 100

        status = "OK"
        if utilization >= 100:
            status = "EXCEEDED"
        elif utilization >= 80:
            status = "WARNING"

        return {
            "status": status,
            "daily_limit": f"${daily_limit:.2f}",
            "today_cost": f"${today_cost:.2f}",
            "remaining": f"${remaining:.2f}",
            "utilization": f"{utilization:.1f}%"
        }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Usage Tracker Demo
    ========================================
    Testing API usage tracking and analysis
    ========================================
    """)

    # Initialize tracker
    tracker = UsageTracker(log_file="./metrics/demo_usage.jsonl")

    # Log some sample requests
    print("\nLogging sample requests...\n")

    sample_requests = [
        {"model": "claude-3-5-sonnet", "input": 100, "output": 200, "cost": 0.003300, "latency": 2.5, "task": "config_analysis"},
        {"model": "claude-3-5-sonnet", "input": 150, "output": 250, "cost": 0.004200, "latency": 3.1, "task": "config_analysis"},
        {"model": "claude-3-5-haiku", "input": 80, "output": 120, "cost": 0.000544, "latency": 1.2, "task": "log_classification"},
        {"model": "claude-3-5-haiku", "input": 90, "output": 130, "cost": 0.000592, "latency": 1.4, "task": "log_classification"},
        {"model": "claude-3-5-sonnet", "input": 200, "output": 400, "cost": 0.006600, "latency": 4.2, "task": "troubleshooting"},
    ]

    for req in sample_requests:
        tracker.log_request(
            model=req["model"],
            input_tokens=req["input"],
            output_tokens=req["output"],
            cost=req["cost"],
            latency=req["latency"],
            success=True,
            task=req["task"]
        )
        print(f"  ✓ Logged: {req['task']} ({req['input'] + req['output']} tokens, ${req['cost']:.6f})")

    # Get summary
    print("\n" + "="*60)
    print("USAGE SUMMARY (Last 24 Hours)")
    print("="*60)

    summary = tracker.get_summary(hours=24)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"  {k:30s}: {v}")
        else:
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:30s}: {value}")

    # Monthly estimate
    print("\n" + "="*60)
    print("MONTHLY COST ESTIMATE")
    print("="*60)

    estimate = tracker.estimate_monthly_cost(days_sample=1)
    for key, value in estimate.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key:30s}: {value}")

    # Budget check
    print("\n" + "="*60)
    print("BUDGET CHECK")
    print("="*60)

    budget_status = tracker.check_budget(daily_limit=10.0)
    for key, value in budget_status.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key:30s}: {value}")

    # Hourly breakdown
    print("\n" + "="*60)
    print("HOURLY BREAKDOWN (Last 6 Hours)")
    print("="*60)

    hourly = tracker.get_hourly_breakdown(hours=6)
    print(f"\n{'Hour':<20s} {'Requests':>10s} {'Cost':>12s} {'Tokens':>12s}")
    print("-" * 60)
    for hour_data in hourly[-6:]:  # Last 6 hours
        print(
            f"{hour_data['hour']:<20s} "
            f"{hour_data['requests']:>10d} "
            f"{hour_data['cost']:>12s} "
            f"{hour_data['tokens']:>12d}"
        )

    print("\n✅ Demo complete!")
    print(f"\n💡 Check {tracker.log_file} for detailed logs")
