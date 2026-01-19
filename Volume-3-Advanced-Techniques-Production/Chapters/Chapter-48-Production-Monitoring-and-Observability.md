# Chapter 48: Production Monitoring & Observability

## Introduction

Your AI system works great in development. You deploy to production. Two weeks later, costs are 3x projections. API latency spikes to 15 seconds during peak hours. Error rates hit 8%, but no one noticed until users complained. The finance team wants to know which department is spending $5,000/month on AI tokens.

**You have no visibility**.

Production AI systems need the same observability as any critical infrastructure: metrics, logs, traces, alerts, dashboards. You need to know what's happening in real-time, track costs per user/department/application, detect anomalies before they become incidents, and optimize based on data.

This chapter shows you how to instrument AI systems for production observability using standard tools (Prometheus, Grafana, ELK stack) and build dashboards that tell you exactly what's happening with your AI operations.

**What You'll Build**:
- Instrumentation for LLM API calls (latency, tokens, errors, costs)
- Prometheus metrics and Grafana dashboards
- Cost attribution system (track spending by user/department/app)
- Alerting rules (error rates, latency SLOs, cost anomalies)
- Performance optimization based on monitoring data
- Complete observability stack for AI operations

**Prerequisites**: Chapter 8 (Cost Optimization), Chapter 19 (Agent Architecture), Chapter 51 (Scaling - for context on production scale)

---

## The Four Pillars of AI Observability

### 1. Metrics (What's Happening)

**Key metrics for AI systems**:
- **Request volume**: Requests per second/minute/hour
- **Latency**: P50, P95, P99 response times
- **Token usage**: Input tokens, output tokens, total tokens per request
- **Cost**: $ spent per request, per hour, per day
- **Error rate**: % of requests that fail
- **Model performance**: Accuracy, hallucination rate (if measurable)

### 2. Logs (What Happened)

**What to log**:
- Every LLM API call (request, response, tokens, latency)
- Errors and exceptions (with full context)
- User actions and outcomes
- Cost attribution metadata (user, department, application)

### 3. Traces (How It Happened)

**Distributed tracing for multi-agent systems**:
- End-to-end request flow
- Which agents were invoked
- Time spent in each component
- Dependencies and bottlenecks

### 4. Alerts (When Something's Wrong)

**Alert on**:
- Error rate > 5% for 5 minutes
- P95 latency > 10 seconds
- Daily cost > $1,000
- Token usage anomaly (50% increase vs baseline)

---

## Instrumentation: Capturing Metrics

First, instrument your LLM client to capture every metric.

### Implementation: Instrumented LLM Client

```python
"""
Instrumented LLM Client with Metrics
File: observability/instrumented_client.py
"""
import time
import json
from anthropic import Anthropic
from typing import Dict, Optional, List
from prometheus_client import Counter, Histogram, Gauge
import logging

# Prometheus metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status', 'user', 'application']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM API request duration',
    ['model', 'user', 'application'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'token_type', 'user', 'application']
)

llm_cost_total = Counter(
    'llm_cost_dollars_total',
    'Total cost in dollars',
    ['model', 'user', 'application']
)

llm_errors_total = Counter(
    'llm_errors_total',
    'Total LLM errors',
    ['model', 'error_type', 'user', 'application']
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstrumentedAnthropicClient:
    """
    Anthropic client with full observability instrumentation.

    Captures metrics, logs, and provides cost tracking.
    """

    # Pricing per 1M tokens (update these based on current pricing)
    PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0}
    }

    def __init__(self, api_key: str, user_id: str = 'unknown', application: str = 'default'):
        """
        Args:
            api_key: Anthropic API key
            user_id: User ID for cost attribution
            application: Application name for tracking
        """
        self.client = Anthropic(api_key=api_key)
        self.user_id = user_id
        self.application = application

    def create_message(self, model: str, max_tokens: int, messages: List[Dict], **kwargs) -> Dict:
        """
        Create message with full instrumentation.

        Returns:
            Dict with response and metrics
        """
        start_time = time.time()

        # Labels for metrics
        labels = {
            'model': model,
            'user': self.user_id,
            'application': self.application
        }

        try:
            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs
            )

            # Calculate metrics
            duration = time.time() - start_time
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost = self._calculate_cost(model, input_tokens, output_tokens)

            # Record metrics
            llm_requests_total.labels(status='success', **labels).inc()
            llm_request_duration.labels(**labels).observe(duration)
            llm_tokens_total.labels(token_type='input', **labels).inc(input_tokens)
            llm_tokens_total.labels(token_type='output', **labels).inc(output_tokens)
            llm_cost_total.labels(**labels).inc(cost)

            # Log the request
            logger.info(
                f"LLM Request Success",
                extra={
                    'user': self.user_id,
                    'application': self.application,
                    'model': model,
                    'duration_seconds': duration,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'cost_dollars': cost,
                    'status': 'success'
                }
            )

            return {
                'response': response,
                'metrics': {
                    'duration_seconds': duration,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'cost_dollars': cost
                }
            }

        except Exception as e:
            # Record error
            duration = time.time() - start_time
            error_type = type(e).__name__

            llm_requests_total.labels(status='error', **labels).inc()
            llm_request_duration.labels(**labels).observe(duration)
            llm_errors_total.labels(error_type=error_type, **labels).inc()

            # Log the error
            logger.error(
                f"LLM Request Failed: {e}",
                extra={
                    'user': self.user_id,
                    'application': self.application,
                    'model': model,
                    'duration_seconds': duration,
                    'error_type': error_type,
                    'error_message': str(e),
                    'status': 'error'
                },
                exc_info=True
            )

            raise

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in dollars for this request."""
        if model not in self.PRICING:
            logger.warning(f"Unknown model pricing: {model}, using Sonnet rates")
            model = 'claude-3-5-sonnet-20241022'

        pricing = self.PRICING[model]

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost


# Example Usage
if __name__ == "__main__":
    import os

    # Create instrumented client
    client = InstrumentedAnthropicClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        user_id='john.doe',
        application='network-troubleshooting'
    )

    # Make a request (metrics automatically captured)
    result = client.create_message(
        model='claude-3-5-sonnet-20241022',
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "Explain BGP path selection in 2 sentences"}
        ]
    )

    print(f"\nResponse: {result['response'].content[0].text}")
    print(f"\nMetrics:")
    print(f"  Duration: {result['metrics']['duration_seconds']:.3f}s")
    print(f"  Tokens: {result['metrics']['total_tokens']}")
    print(f"  Cost: ${result['metrics']['cost_dollars']:.6f}")
```

### Exposing Metrics to Prometheus

```python
"""
Prometheus Metrics Exporter
File: observability/metrics_exporter.py
"""
from prometheus_client import start_http_server, REGISTRY
import time

def start_metrics_server(port: int = 9090):
    """
    Start HTTP server to expose Prometheus metrics.

    Args:
        port: Port to listen on (default: 9090)
    """
    start_http_server(port)
    print(f"✓ Prometheus metrics available at http://localhost:{port}/metrics")

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMetrics server stopped")


if __name__ == "__main__":
    # Start metrics server
    print("Starting Prometheus metrics exporter...")
    start_metrics_server(port=9090)
```

Now Prometheus can scrape metrics from `http://localhost:9090/metrics`.

---

## Cost Attribution System

Track spending by user, department, and application.

### Implementation: Cost Tracking Database

```python
"""
Cost Attribution System
File: observability/cost_tracker.py
"""
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class CostTracker:
    """
    Track and attribute AI costs to users, departments, and applications.
    """

    def __init__(self, db_path: str = 'ai_costs.db'):
        """
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                user_id TEXT NOT NULL,
                department TEXT,
                application TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost_dollars REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_timestamp
            ON requests(user_id, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_department_timestamp
            ON requests(department, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_application_timestamp
            ON requests(application, timestamp)
        """)

        conn.commit()
        conn.close()

    def record_request(self, user_id: str, department: str, application: str,
                      model: str, input_tokens: int, output_tokens: int,
                      cost_dollars: float, duration_seconds: float, status: str):
        """Record a request for cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO requests
            (timestamp, user_id, department, application, model,
             input_tokens, output_tokens, total_tokens, cost_dollars,
             duration_seconds, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            user_id,
            department,
            application,
            model,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            cost_dollars,
            duration_seconds,
            status
        ))

        conn.commit()
        conn.close()

    def get_costs_by_user(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get costs broken down by user."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                user_id,
                COUNT(*) as request_count,
                SUM(total_tokens) as total_tokens,
                SUM(cost_dollars) as total_cost,
                AVG(duration_seconds) as avg_duration,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY user_id
            ORDER BY total_cost DESC
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        return df

    def get_costs_by_department(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get costs broken down by department."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                department,
                COUNT(*) as request_count,
                SUM(total_tokens) as total_tokens,
                SUM(cost_dollars) as total_cost,
                AVG(cost_dollars) as avg_cost_per_request
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY department
            ORDER BY total_cost DESC
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        return df

    def get_costs_by_application(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get costs broken down by application."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                application,
                COUNT(*) as request_count,
                SUM(total_tokens) as total_tokens,
                SUM(cost_dollars) as total_cost,
                AVG(duration_seconds) as avg_duration
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY application
            ORDER BY total_cost DESC
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        return df

    def get_daily_costs(self, days: int = 30) -> pd.DataFrame:
        """Get daily cost trends."""
        conn = sqlite3.connect(self.db_path)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = """
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as request_count,
                SUM(cost_dollars) as total_cost,
                SUM(total_tokens) as total_tokens
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        return df

    def generate_chargeback_report(self, month: int, year: int) -> Dict:
        """
        Generate monthly chargeback report.

        Args:
            month: Month (1-12)
            year: Year (e.g., 2024)

        Returns:
            Dict with department costs for chargeback
        """
        import calendar

        # Get first and last day of month
        _, last_day = calendar.monthrange(year, month)
        start_date = datetime(year, month, 1)
        end_date = datetime(year, month, last_day, 23, 59, 59)

        # Get costs by department
        department_costs = self.get_costs_by_department(start_date, end_date)

        return {
            'month': month,
            'year': year,
            'departments': department_costs.to_dict('records'),
            'total_cost': department_costs['total_cost'].sum(),
            'total_requests': int(department_costs['request_count'].sum())
        }

    def print_cost_summary(self, days: int = 7):
        """Print cost summary for last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"\n{'='*70}")
        print(f"COST SUMMARY - Last {days} Days")
        print(f"{'='*70}")

        # By user
        print(f"\nBy User:")
        user_costs = self.get_costs_by_user(start_date, end_date)
        print(user_costs.to_string(index=False))

        # By department
        print(f"\nBy Department:")
        dept_costs = self.get_costs_by_department(start_date, end_date)
        print(dept_costs.to_string(index=False))

        # By application
        print(f"\nBy Application:")
        app_costs = self.get_costs_by_application(start_date, end_date)
        print(app_costs.to_string(index=False))

        # Total
        total_cost = user_costs['total_cost'].sum()
        print(f"\nTotal Cost: ${total_cost:.2f}")


# Example Usage
if __name__ == "__main__":
    tracker = CostTracker()

    # Simulate some requests
    tracker.record_request(
        user_id='john.doe',
        department='IT',
        application='network-troubleshooting',
        model='claude-3-5-sonnet-20241022',
        input_tokens=1000,
        output_tokens=500,
        cost_dollars=0.012,
        duration_seconds=2.3,
        status='success'
    )

    tracker.record_request(
        user_id='jane.smith',
        department='Operations',
        application='config-generator',
        model='claude-3-haiku-20240307',
        input_tokens=500,
        output_tokens=300,
        cost_dollars=0.0005,
        duration_seconds=0.8,
        status='success'
    )

    # Print summary
    tracker.print_cost_summary(days=7)

    # Generate chargeback report
    chargeback = tracker.generate_chargeback_report(month=1, year=2024)
    print(f"\nChargeback Report for {chargeback['month']}/{chargeback['year']}:")
    print(f"Total: ${chargeback['total_cost']:.2f}")
```

---

## Grafana Dashboards

Visualize metrics in Grafana for real-time monitoring.

### Dashboard Configuration

```yaml
# Grafana Dashboard for AI Operations
# File: observability/grafana_dashboard.json
{
  "dashboard": {
    "title": "AI Operations Dashboard",
    "panels": [
      {
        "title": "Request Rate (req/min)",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m]) * 60",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate (%)",
        "targets": [
          {
            "expr": "(rate(llm_requests_total{status=\"error\"}[5m]) / rate(llm_requests_total[5m])) * 100",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph",
        "alert": {
          "conditions": [
            {
              "evaluator": { "type": "gt", "params": [5] },
              "query": { "model": "A", "timeRange": "5m" }
            }
          ],
          "message": "Error rate > 5%"
        }
      },
      {
        "title": "Token Usage (per minute)",
        "targets": [
          {
            "expr": "rate(llm_tokens_total[1m]) * 60",
            "legendFormat": "{{token_type}} - {{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cost Rate ($/hour)",
        "targets": [
          {
            "expr": "rate(llm_cost_dollars_total[1h]) * 3600",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Top Users by Cost",
        "targets": [
          {
            "expr": "topk(10, sum by (user) (llm_cost_dollars_total))",
            "legendFormat": "{{user}}"
          }
        ],
        "type": "table"
      },
      {
        "title": "Top Applications by Request Volume",
        "targets": [
          {
            "expr": "topk(10, sum by (application) (llm_requests_total))",
            "legendFormat": "{{application}}"
          }
        ],
        "type": "piechart"
      }
    ]
  }
}
```

### Key Metrics to Track

**Operational Health**:
- Request rate: Monitor traffic patterns, detect anomalies
- Latency: P50, P95, P99 to catch performance degradation
- Error rate: Alert on spikes (>5% for 5 minutes)
- Throughput: Requests per second by model

**Cost Metrics**:
- Cost per hour/day/month
- Cost per user/department/application
- Token usage trends
- Cost anomalies (50% increase vs baseline)

**User Experience**:
- Latency distribution (are users experiencing slow responses?)
- Success rate by user (is someone having problems?)
- Peak usage times (capacity planning)

---

## Alerting Rules

Define alerts for critical conditions.

### Prometheus Alerting Rules

```yaml
# Prometheus Alert Rules
# File: observability/alert_rules.yml
groups:
  - name: ai_operations
    interval: 30s
    rules:
      # Error rate alert
      - alert: HighLLMErrorRate
        expr: |
          (
            rate(llm_requests_total{status="error"}[5m])
            /
            rate(llm_requests_total[5m])
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High LLM error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for model {{ $labels.model }}"

      # Latency alert
      - alert: HighLLMLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM latency detected"
          description: "P95 latency is {{ $value }}s for model {{ $labels.model }}"

      # Cost anomaly alert
      - alert: LLMCostAnomaly
        expr: |
          (
            rate(llm_cost_dollars_total[1h])
            /
            avg_over_time(rate(llm_cost_dollars_total[1h])[7d:1h])
          ) > 1.5
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "LLM cost anomaly detected"
          description: "Cost rate is 50% above baseline"

      # No requests alert (service down?)
      - alert: LLMServiceDown
        expr: |
          rate(llm_requests_total[5m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "LLM service appears down"
          description: "No requests in last 10 minutes"

      # Daily budget alert
      - alert: DailyBudgetExceeded
        expr: |
          sum(increase(llm_cost_dollars_total[1d])) > 1000
        labels:
          severity: critical
        annotations:
          summary: "Daily AI budget exceeded"
          description: "Daily cost is ${{ $value }}, budget is $1000"

      # Token usage spike
      - alert: TokenUsageSpike
        expr: |
          (
            rate(llm_tokens_total[10m])
            /
            avg_over_time(rate(llm_tokens_total[10m])[1h])
          ) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Token usage spike detected"
          description: "Token usage is 2x normal for application {{ $labels.application }}"
```

### Alert Integration

```python
"""
Alert Handler
File: observability/alert_handler.py
"""
import requests
from typing import Dict

class AlertHandler:
    """Handle alerts from monitoring system."""

    def __init__(self, slack_webhook: str = None, pagerduty_key: str = None):
        """
        Args:
            slack_webhook: Slack webhook URL for notifications
            pagerduty_key: PagerDuty API key for critical alerts
        """
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

    def send_slack_alert(self, alert: Dict):
        """Send alert to Slack."""
        if not self.slack_webhook:
            return

        message = {
            "text": f"🚨 {alert['summary']}",
            "attachments": [
                {
                    "color": "danger" if alert['severity'] == 'critical' else "warning",
                    "fields": [
                        {"title": "Severity", "value": alert['severity'], "short": True},
                        {"title": "Description", "value": alert['description'], "short": False}
                    ]
                }
            ]
        }

        try:
            response = requests.post(self.slack_webhook, json=message)
            response.raise_for_status()
            print(f"✓ Slack alert sent: {alert['summary']}")
        except Exception as e:
            print(f"✗ Failed to send Slack alert: {e}")

    def send_pagerduty_alert(self, alert: Dict):
        """Send critical alert to PagerDuty."""
        if not self.pagerduty_key or alert['severity'] != 'critical':
            return

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert['summary'],
                "severity": "critical",
                "source": "ai-operations-monitoring",
                "custom_details": alert
            }
        }

        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()
            print(f"✓ PagerDuty alert sent: {alert['summary']}")
        except Exception as e:
            print(f"✗ Failed to send PagerDuty alert: {e}")

    def handle_alert(self, alert: Dict):
        """Route alert to appropriate channels based on severity."""
        print(f"\n[ALERT] {alert['severity'].upper()}: {alert['summary']}")

        # Always send to Slack
        self.send_slack_alert(alert)

        # Critical alerts also go to PagerDuty
        if alert['severity'] == 'critical':
            self.send_pagerduty_alert(alert)


# Example Usage
if __name__ == "__main__":
    handler = AlertHandler(
        slack_webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        pagerduty_key="YOUR_PAGERDUTY_KEY"
    )

    # Simulate alert
    alert = {
        'summary': 'High LLM error rate detected',
        'severity': 'critical',
        'description': 'Error rate is 8.5% for model claude-3-5-sonnet-20241022',
        'timestamp': '2024-01-15T14:30:00Z'
    }

    handler.handle_alert(alert)
```

---

## Performance Optimization Based on Monitoring

Use monitoring data to optimize performance and cost.

### Implementation: Optimization Analyzer

```python
"""
Performance Optimization Analyzer
File: observability/optimization_analyzer.py
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class OptimizationAnalyzer:
    """Analyze monitoring data to find optimization opportunities."""

    def __init__(self, cost_tracker):
        """
        Args:
            cost_tracker: CostTracker instance
        """
        self.cost_tracker = cost_tracker

    def analyze_model_usage(self, days: int = 7) -> Dict:
        """
        Analyze model usage to find cost optimization opportunities.

        Returns:
            Dict with recommendations
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = self.cost_tracker.db_path
        import sqlite3
        conn = sqlite3.connect(conn)

        # Get model usage statistics
        query = """
            SELECT
                model,
                COUNT(*) as request_count,
                AVG(duration_seconds) as avg_duration,
                AVG(cost_dollars) as avg_cost,
                SUM(cost_dollars) as total_cost,
                AVG(total_tokens) as avg_tokens
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY model
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        recommendations = []

        # Check if using expensive models for simple tasks
        if len(df) > 0:
            # Find requests with low token usage but expensive model
            expensive_models = df[df['avg_tokens'] < 1000]
            if not expensive_models.empty:
                for _, row in expensive_models.iterrows():
                    if 'opus' in row['model'].lower() or 'sonnet' in row['model'].lower():
                        potential_savings = row['total_cost'] * 0.8  # 80% savings with Haiku
                        recommendations.append({
                            'type': 'model_downgrade',
                            'current_model': row['model'],
                            'recommended_model': 'claude-3-haiku',
                            'reason': f'Average tokens ({row["avg_tokens"]:.0f}) suggests Haiku would work',
                            'potential_savings': potential_savings,
                            'affected_requests': int(row['request_count'])
                        })

        return {
            'period_days': days,
            'recommendations': recommendations,
            'total_potential_savings': sum(r['potential_savings'] for r in recommendations)
        }

    def analyze_prompt_efficiency(self, days: int = 7) -> Dict:
        """Analyze prompt token usage to find efficiency opportunities."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        conn = sqlite3.connect(self.cost_tracker.db_path)

        # Find requests with high input token counts
        query = """
            SELECT
                application,
                user_id,
                AVG(input_tokens) as avg_input_tokens,
                AVG(output_tokens) as avg_output_tokens,
                COUNT(*) as request_count,
                SUM(cost_dollars) as total_cost
            FROM requests
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY application, user_id
            HAVING AVG(input_tokens) > 3000
            ORDER BY total_cost DESC
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        recommendations = []

        for _, row in df.iterrows():
            # Large prompts suggest context could be optimized
            if row['avg_input_tokens'] > 3000:
                potential_savings = (row['total_cost'] * 0.4)  # 40% reduction possible
                recommendations.append({
                    'type': 'prompt_optimization',
                    'application': row['application'],
                    'user': row['user_id'],
                    'current_avg_tokens': int(row['avg_input_tokens']),
                    'reason': 'Large prompts suggest context could be reduced via RAG or summarization',
                    'potential_savings': potential_savings,
                    'affected_requests': int(row['request_count'])
                })

        return {
            'period_days': days,
            'recommendations': recommendations,
            'total_potential_savings': sum(r['potential_savings'] for r in recommendations)
        }

    def generate_optimization_report(self, days: int = 7):
        """Generate complete optimization report."""
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION OPPORTUNITIES - Last {days} Days")
        print(f"{'='*70}")

        # Model usage analysis
        model_analysis = self.analyze_model_usage(days)
        print(f"\nModel Usage Recommendations:")
        if model_analysis['recommendations']:
            for i, rec in enumerate(model_analysis['recommendations'], 1):
                print(f"\n{i}. {rec['type'].upper()}")
                print(f"   Current: {rec['current_model']}")
                print(f"   Recommended: {rec['recommended_model']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Potential Savings: ${rec['potential_savings']:.2f}")
                print(f"   Affected Requests: {rec['affected_requests']}")
        else:
            print("  ✓ No model optimization opportunities found")

        # Prompt efficiency analysis
        prompt_analysis = self.analyze_prompt_efficiency(days)
        print(f"\nPrompt Efficiency Recommendations:")
        if prompt_analysis['recommendations']:
            for i, rec in enumerate(prompt_analysis['recommendations'], 1):
                print(f"\n{i}. {rec['type'].upper()}")
                print(f"   Application: {rec['application']}")
                print(f"   User: {rec['user']}")
                print(f"   Current Avg Tokens: {rec['current_avg_tokens']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Potential Savings: ${rec['potential_savings']:.2f}")
        else:
            print("  ✓ No prompt optimization opportunities found")

        # Total potential savings
        total_savings = (
            model_analysis['total_potential_savings'] +
            prompt_analysis['total_potential_savings']
        )

        print(f"\n{'='*70}")
        print(f"TOTAL POTENTIAL SAVINGS: ${total_savings:.2f}/week")
        print(f"Annual Projection: ${total_savings * 52:.2f}")
        print(f"{'='*70}")


# Example Usage
if __name__ == "__main__":
    from cost_tracker import CostTracker

    tracker = CostTracker()
    analyzer = OptimizationAnalyzer(tracker)

    # Generate optimization report
    analyzer.generate_optimization_report(days=7)
```

---

## Complete Observability Stack

Putting it all together: full production monitoring setup.

### Docker Compose Setup

```yaml
# Docker Compose for Complete Monitoring Stack
# File: observability/docker-compose.yml
version: '3.8'

services:
  # Prometheus - Metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana - Visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana_dashboard.json:/etc/grafana/provisioning/dashboards/ai_ops.json
    depends_on:
      - prometheus

  # Alertmanager - Alert routing
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'

volumes:
  prometheus_data:
  grafana_data:
```

### Start the Stack

```bash
# Start monitoring stack
cd observability
docker-compose up -d

# Access Grafana
open http://localhost:3000
# Login: admin / admin

# Access Prometheus
open http://localhost:9091
```

---

## Summary

You now have complete observability for AI operations:

1. **Instrumentation**: Every LLM call captured with metrics (latency, tokens, cost, errors)
2. **Cost Attribution**: Track spending by user, department, application
3. **Dashboards**: Real-time Grafana dashboards for operational visibility
4. **Alerting**: Automated alerts for errors, latency, cost anomalies
5. **Optimization**: Data-driven recommendations to reduce costs

**Production Benefits**:
- **Visibility**: Know exactly what's happening with AI systems in real-time
- **Cost Control**: Track and control AI spending with chargeback
- **Reliability**: Detect and respond to issues before users complain
- **Optimization**: Data-driven decisions to reduce costs 30-50%

**Real-World Impact**:
- Finance team gets monthly chargeback reports
- Ops team gets paged when error rate spikes
- Engineering team optimizes based on actual usage data
- Management sees ROI with concrete metrics

**Next Chapter**: We'll show how to scale these systems from 10 devices to 10,000 devices with batch processing, queues, and distributed architectures.

---

## What Can Go Wrong?

**1. Metrics collection adds too much latency**
- **Cause**: Synchronous metric recording blocks request
- **Fix**: Record metrics asynchronously in background thread

**2. Database fills up with cost tracking data**
- **Cause**: Recording every request forever
- **Fix**: Implement data retention policy (keep 90 days, aggregate older data)

**3. Alert fatigue (too many alerts)**
- **Cause**: Thresholds too sensitive, alerting on normal variance
- **Fix**: Tune thresholds based on baselines, require sustained conditions (5+ minutes)

**4. Grafana dashboards are slow**
- **Cause**: Queries too complex, too much historical data
- **Fix**: Add aggregation, use recording rules in Prometheus for expensive queries

**5. Cost attribution is inaccurate**
- **Cause**: User/department metadata missing or wrong
- **Fix**: Require metadata on every request, validate at instrumentation layer

**6. Monitoring system itself costs too much**
- **Cause**: Over-collection (recording every detail)
- **Fix**: Sample high-frequency metrics, aggregate low-value data

**7. No one looks at dashboards (observability theater)**
- **Cause**: Dashboards don't answer important questions
- **Fix**: Design dashboards around actual operational needs, actionable metrics only

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-48/`
