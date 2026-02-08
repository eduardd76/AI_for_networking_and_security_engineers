"""
Chapter 48: Production Monitoring & Observability
Real-time monitoring and alerting for AI-powered network operations

This module provides production-grade monitoring infrastructure including
metrics collection, structured logging, cost tracking, alerting, and
performance dashboards for LLM-based network automation systems.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
import hashlib


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class SLOStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    VIOLATED = "violated"


@dataclass
class MetricPoint:
    """Time-series metric data point."""
    name: str
    value: float
    unit: str
    timestamp: str
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class LogRecord:
    """Structured JSON log entry."""
    timestamp: str
    level: str
    service: str
    operation: str
    message: str
    trace_id: str
    span_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)


@dataclass
class Alert:
    """Alert notification."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: str
    triggered_at: str
    resolved_at: Optional[str] = None
    status: str = "active"


@dataclass
class CostAttribution:
    """Cost breakdown by dimension."""
    dimension: str  # team, project, environment, user
    value: str  # team-network, proj-bgp-analyzer, prod, admin
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str


@dataclass
class SLODefinition:
    """Service Level Objective definition."""
    name: str
    description: str
    target_percent: float
    metric_name: str
    threshold: float
    comparison: str  # 'less_than', 'greater_than'
    window_minutes: int


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    request_count: int
    error_count: int
    total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    avg_tokens_per_request: float
    cost_per_request: float


class ProductionMonitor:
    """
    Production-grade monitoring system.

    Features:
    - Real-time metrics collection (requests, tokens, costs, latency)
    - Structured JSON logging with trace correlation
    - SLO tracking and violation detection
    - Multi-dimensional cost attribution
    - Alert management with deduplication
    - Performance dashboards
    - Rolling time-series data
    """

    def __init__(
        self,
        service_name: str = "network-ai-service",
        retention_hours: int = 24
    ):
        self.service_name = service_name
        self.retention_hours = retention_hours
        self.lock = Lock()

        # Storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.logs: deque = deque(maxlen=50000)
        self.alerts: Dict[str, Alert] = {}
        self.cost_attributions: List[CostAttribution] = []

        # Counters
        self.request_counter = 0
        self.error_counter = 0
        self.total_cost = 0.0
        self.total_tokens = 0

        # Latency tracking
        self.latencies: deque = deque(maxlen=1000)

        # Model pricing (per 1M tokens)
        self.model_pricing = {
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'claude-opus-4-20250115': {'input': 15.00, 'output': 75.00},
            'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku': {'input': 0.25, 'output': 1.25},
            'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.00},
        }

        # SLO definitions
        self.slos = [
            SLODefinition(
                name="request_success_rate",
                description="99% of requests should succeed",
                target_percent=99.0,
                metric_name="success_rate",
                threshold=99.0,
                comparison="greater_than",
                window_minutes=5
            ),
            SLODefinition(
                name="p99_latency",
                description="99th percentile latency under 2 seconds",
                target_percent=99.0,
                metric_name="latency_p99",
                threshold=2000.0,
                comparison="less_than",
                window_minutes=5
            ),
            SLODefinition(
                name="error_rate",
                description="Error rate below 1%",
                target_percent=99.0,
                metric_name="error_rate",
                threshold=1.0,
                comparison="less_than",
                window_minutes=5
            ),
        ]

        # Alert rules
        self.alert_rules = {
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 5.0,
                'severity': AlertSeverity.CRITICAL,
                'description': 'Error rate above 5%'
            },
            'high_latency': {
                'metric': 'latency_p99',
                'threshold': 3000.0,
                'severity': AlertSeverity.WARNING,
                'description': 'P99 latency above 3 seconds'
            },
            'cost_spike': {
                'metric': 'cost_per_hour',
                'threshold': 50.0,
                'severity': AlertSeverity.WARNING,
                'description': 'Cost exceeding $50/hour'
            },
            'low_success_rate': {
                'metric': 'success_rate',
                'threshold': 95.0,
                'severity': AlertSeverity.CRITICAL,
                'description': 'Success rate below 95%'
            },
        }

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metric_type: Type of metric
            labels: Additional labels for filtering
        """
        with self.lock:
            metric = MetricPoint(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now().isoformat(),
                labels=labels or {},
                metric_type=metric_type
            )

            self.metrics[name].append(metric)
            self._cleanup_old_metrics()

    def log(
        self,
        level: str,
        operation: str,
        message: str,
        trace_id: str,
        span_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        **metadata
    ):
        """
        Write structured log entry.

        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR)
            operation: Operation being performed
            message: Log message
            trace_id: Distributed trace ID
            span_id: Span ID within trace
            duration_ms: Operation duration
            error: Error message if applicable
            **metadata: Additional context
        """
        with self.lock:
            log_record = LogRecord(
                timestamp=datetime.now().isoformat(),
                level=level,
                service=self.service_name,
                operation=operation,
                message=message,
                trace_id=trace_id,
                span_id=span_id,
                duration_ms=duration_ms,
                error=error,
                metadata=metadata
            )

            self.logs.append(log_record)
            self._cleanup_old_logs()

    def track_request(
        self,
        trace_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        team: str = "default",
        project: str = "default",
        environment: str = "production"
    ):
        """
        Track complete request metrics.

        Args:
            trace_id: Request trace ID
            model: LLM model used
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Request latency
            success: Whether request succeeded
            team: Team attribution
            project: Project attribution
            environment: Environment (prod, staging, dev)
        """
        with self.lock:
            self.request_counter += 1

            if not success:
                self.error_counter += 1

            # Calculate cost
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            self.total_cost += cost
            self.total_tokens += (input_tokens + output_tokens)

            # Track latency
            self.latencies.append(latency_ms)

            # Record metrics
            self.record_metric("requests_total", self.request_counter, "requests", MetricType.COUNTER)
            self.record_metric("tokens_total", self.total_tokens, "tokens", MetricType.COUNTER)
            self.record_metric("cost_total", self.total_cost, "USD", MetricType.COUNTER)
            self.record_metric("latency", latency_ms, "ms", MetricType.HISTOGRAM)

            # Cost attribution
            attribution = CostAttribution(
                dimension="team",
                value=team,
                requests=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                timestamp=datetime.now().isoformat()
            )
            self.cost_attributions.append(attribution)

            # Log request
            self.log(
                level="INFO" if success else "ERROR",
                operation="llm_request",
                message=f"Request completed: {model}",
                trace_id=trace_id,
                duration_ms=latency_ms,
                error=None if success else "Request failed",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                team=team,
                project=project,
                environment=environment
            )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for request."""
        if model not in self.model_pricing:
            model = 'gpt-3.5-turbo'

        pricing = self.model_pricing[model]
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost

    def get_performance_metrics(self, window_minutes: int = 5) -> PerformanceMetrics:
        """
        Get aggregated performance metrics.

        Args:
            window_minutes: Time window for metrics

        Returns:
            Performance metrics
        """
        with self.lock:
            # Get recent data
            cutoff = datetime.now() - timedelta(minutes=window_minutes)

            # Recent requests (approximation based on counters)
            request_count = self.request_counter
            error_count = self.error_counter

            # Calculate rates
            success_rate = 0.0
            if request_count > 0:
                success_rate = ((request_count - error_count) / request_count) * 100

            # Latency percentiles
            if self.latencies:
                sorted_latencies = sorted(self.latencies)
                p50_idx = int(len(sorted_latencies) * 0.50)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)

                p50 = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
                p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
                p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
                avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            else:
                p50 = p95 = p99 = avg_latency = 0.0

            # Cost metrics
            avg_tokens = self.total_tokens / request_count if request_count > 0 else 0
            cost_per_req = self.total_cost / request_count if request_count > 0 else 0

            return PerformanceMetrics(
                request_count=request_count,
                error_count=error_count,
                total_latency_ms=avg_latency * request_count,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                success_rate=success_rate,
                avg_tokens_per_request=avg_tokens,
                cost_per_request=cost_per_req
            )

    def check_slos(self) -> Dict[str, SLOStatus]:
        """
        Check SLO compliance.

        Returns:
            SLO status for each defined SLO
        """
        results = {}
        metrics = self.get_performance_metrics()

        for slo in self.slos:
            # Get metric value
            metric_value = None

            if slo.metric_name == "success_rate":
                metric_value = metrics.success_rate
            elif slo.metric_name == "latency_p99":
                metric_value = metrics.p99_latency_ms
            elif slo.metric_name == "error_rate":
                error_rate = (metrics.error_count / max(1, metrics.request_count)) * 100
                metric_value = error_rate

            if metric_value is None:
                continue

            # Check threshold
            if slo.comparison == "greater_than":
                if metric_value >= slo.threshold:
                    status = SLOStatus.HEALTHY
                elif metric_value >= (slo.threshold * 0.95):
                    status = SLOStatus.DEGRADED
                else:
                    status = SLOStatus.VIOLATED
            else:  # less_than
                if metric_value <= slo.threshold:
                    status = SLOStatus.HEALTHY
                elif metric_value <= (slo.threshold * 1.05):
                    status = SLOStatus.DEGRADED
                else:
                    status = SLOStatus.VIOLATED

            results[slo.name] = status

        return results

    def check_alerts(self) -> List[Alert]:
        """
        Check alert rules and generate alerts.

        Returns:
            List of active alerts
        """
        new_alerts = []
        metrics = self.get_performance_metrics()

        for rule_name, rule in self.alert_rules.items():
            # Get metric value
            metric_value = None

            if rule['metric'] == 'error_rate':
                metric_value = (metrics.error_count / max(1, metrics.request_count)) * 100
            elif rule['metric'] == 'latency_p99':
                metric_value = metrics.p99_latency_ms
            elif rule['metric'] == 'cost_per_hour':
                # Extrapolate from current spend
                metric_value = self.total_cost * 60  # Rough hourly rate
            elif rule['metric'] == 'success_rate':
                metric_value = metrics.success_rate

            if metric_value is None:
                continue

            # Check threshold
            triggered = False
            if rule['metric'] == 'success_rate':
                triggered = metric_value < rule['threshold']
            else:
                triggered = metric_value > rule['threshold']

            if triggered:
                # Check for duplicate alert
                alert_id = hashlib.md5(
                    f"{rule_name}-{rule['metric']}".encode()
                ).hexdigest()[:12]

                if alert_id in self.alerts and self.alerts[alert_id].status == "active":
                    continue  # Skip duplicate

                alert = Alert(
                    id=alert_id,
                    severity=rule['severity'],
                    title=f"{rule_name.replace('_', ' ').title()}",
                    description=rule['description'],
                    metric_name=rule['metric'],
                    current_value=metric_value,
                    threshold=rule['threshold'],
                    timestamp=datetime.now().isoformat(),
                    triggered_at=datetime.now().isoformat()
                )

                self.alerts[alert_id] = alert
                new_alerts.append(alert)
            else:
                # Resolve alert if it exists
                alert_id = hashlib.md5(
                    f"{rule_name}-{rule['metric']}".encode()
                ).hexdigest()[:12]

                if alert_id in self.alerts and self.alerts[alert_id].status == "active":
                    self.alerts[alert_id].status = "resolved"
                    self.alerts[alert_id].resolved_at = datetime.now().isoformat()

        return new_alerts

    def get_cost_attribution(self, dimension: str = "team") -> Dict[str, float]:
        """
        Get cost breakdown by dimension.

        Args:
            dimension: Dimension to group by (team, project, etc.)

        Returns:
            Cost breakdown
        """
        breakdown = defaultdict(float)

        for attribution in self.cost_attributions:
            if attribution.dimension == dimension:
                breakdown[attribution.value] += attribution.cost_usd

        return dict(breakdown)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.

        Returns:
            Dashboard data including all metrics and status
        """
        metrics = self.get_performance_metrics()
        slo_status = self.check_slos()
        active_alerts = [a for a in self.alerts.values() if a.status == "active"]

        return {
            'timestamp': datetime.now().isoformat(),
            'service': self.service_name,
            'metrics': {
                'requests': metrics.request_count,
                'errors': metrics.error_count,
                'success_rate': round(metrics.success_rate, 2),
                'p50_latency_ms': round(metrics.p50_latency_ms, 2),
                'p95_latency_ms': round(metrics.p95_latency_ms, 2),
                'p99_latency_ms': round(metrics.p99_latency_ms, 2),
                'total_tokens': self.total_tokens,
                'avg_tokens_per_request': round(metrics.avg_tokens_per_request, 2),
                'total_cost_usd': round(self.total_cost, 4),
                'cost_per_request': round(metrics.cost_per_request, 6)
            },
            'slo_status': {name: status.value for name, status in slo_status.items()},
            'active_alerts': len(active_alerts),
            'cost_attribution': self.get_cost_attribution('team')
        }

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        for metric_name in list(self.metrics.keys()):
            metrics = self.metrics[metric_name]

            # Remove old entries
            while metrics and datetime.fromisoformat(metrics[0].timestamp) < cutoff:
                metrics.popleft()

    def _cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        while self.logs and datetime.fromisoformat(self.logs[0].timestamp) < cutoff:
            self.logs.popleft()


# ============================================================================
# EXAMPLE FUNCTIONS
# ============================================================================


def example_1_metrics_collection():
    """
    Example 1: Real-time metrics collection

    Track requests, tokens, costs, and latency in production.
    """
    print("=" * 70)
    print("Example 1: Metrics Collection")
    print("=" * 70)

    monitor = ProductionMonitor(service_name="network-ai-prod")

    print("\n📊 Simulating 50K requests/day workload...\n")

    # Simulate production traffic (50K requests/day = ~35 req/min)
    models = ['claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001', 'gpt-4-turbo', 'gpt-3.5-turbo']
    teams = ['network-ops', 'security', 'noc', 'automation']
    projects = ['bgp-analyzer', 'log-parser', 'config-gen', 'troubleshooter']

    for i in range(100):
        trace_id = f"trace-{i:04d}"
        model = models[i % len(models)]
        team = teams[i % len(teams)]
        project = projects[i % len(projects)]

        # Vary token usage
        if model.startswith('gpt-4'):
            input_tokens = 500
            output_tokens = 800
        else:
            input_tokens = 300
            output_tokens = 500

        # Vary latency
        latency = 800 + (i % 10) * 100

        # 3% error rate
        success = (i % 33) != 0

        monitor.track_request(
            trace_id=trace_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            success=success,
            team=team,
            project=project
        )

    # Display metrics
    metrics = monitor.get_performance_metrics()

    print("Production Metrics (last 5 minutes):")
    print(f"  Total Requests:       {metrics.request_count:,}")
    print(f"  Errors:               {metrics.error_count}")
    print(f"  Success Rate:         {metrics.success_rate:.2f}%")
    print(f"\nLatency:")
    print(f"  P50:                  {metrics.p50_latency_ms:.0f}ms")
    print(f"  P95:                  {metrics.p95_latency_ms:.0f}ms")
    print(f"  P99:                  {metrics.p99_latency_ms:.0f}ms")
    print(f"\nTokens:")
    print(f"  Total:                {monitor.total_tokens:,}")
    print(f"  Avg per Request:      {metrics.avg_tokens_per_request:.0f}")
    print(f"\nCost:")
    print(f"  Total:                ${monitor.total_cost:.4f}")
    print(f"  Per Request:          ${metrics.cost_per_request:.6f}")
    print(f"  Projected Daily:      ${monitor.total_cost * 500:.2f}")
    print(f"  Projected Monthly:    ${monitor.total_cost * 15000:.2f}")

    print("\n" + "=" * 70 + "\n")
    return monitor


def example_2_structured_logging():
    """
    Example 2: Structured JSON logging

    JSON logs with trace correlation for debugging and analysis.
    """
    print("=" * 70)
    print("Example 2: Structured Logging")
    print("=" * 70)

    monitor = ProductionMonitor(service_name="network-ai-prod")

    print("\n📝 Generating structured logs...\n")

    # Simulate a request with detailed logging
    trace_id = "trace-bgp-001"

    # Request start
    monitor.log(
        level="INFO",
        operation="bgp_analysis",
        message="Starting BGP route analysis",
        trace_id=trace_id,
        span_id="span-001",
        router="router-core-01",
        prefix_count=150000,
        user="ops-team"
    )

    # Data retrieval
    monitor.log(
        level="DEBUG",
        operation="fetch_routes",
        message="Fetching BGP routes from device",
        trace_id=trace_id,
        span_id="span-002",
        duration_ms=245,
        device="router-core-01",
        method="netconf"
    )

    # LLM processing
    monitor.log(
        level="INFO",
        operation="llm_analysis",
        message="Analyzing routes with Claude",
        trace_id=trace_id,
        span_id="span-003",
        duration_ms=1234,
        model="claude-sonnet-4-20250514",
        input_tokens=2500,
        output_tokens=800
    )

    # Error case
    monitor.log(
        level="ERROR",
        operation="validation",
        message="Invalid BGP community detected",
        trace_id=trace_id,
        span_id="span-004",
        error="InvalidCommunityError: 65000:999999",
        community="65000:999999",
        expected_format="asn:value"
    )

    # Request complete
    monitor.log(
        level="INFO",
        operation="bgp_analysis",
        message="BGP analysis completed",
        trace_id=trace_id,
        span_id="span-001",
        duration_ms=1650,
        routes_analyzed=150000,
        issues_found=3
    )

    # Display logs
    print("Structured Logs (JSON format):\n")

    recent_logs = list(monitor.logs)[-5:]
    for log in recent_logs:
        print(log.to_json())
        print()

    print("\n" + "=" * 70 + "\n")


def example_3_cost_tracking():
    """
    Example 3: Multi-dimensional cost tracking

    Track costs by team, project, and environment for chargebacks.
    """
    print("=" * 70)
    print("Example 3: Cost Attribution System")
    print("=" * 70)

    monitor = ProductionMonitor(service_name="network-ai-prod")

    print("\n💰 Simulating production workload with cost tracking...\n")

    # Simulate realistic production traffic
    workload = [
        # Network Ops team - Heavy users
        ('network-ops', 'bgp-analyzer', 'claude-sonnet-4-20250514', 40, 500, 800),
        ('network-ops', 'config-validator', 'claude-haiku-4-5-20251001', 30, 300, 400),
        ('network-ops', 'troubleshooter', 'gpt-4-turbo', 20, 600, 1000),

        # Security team - Moderate users
        ('security', 'threat-detector', 'claude-sonnet-4-20250514', 25, 400, 600),
        ('security', 'compliance-checker', 'claude-haiku-4-5-20251001', 15, 250, 350),

        # NOC team - Light users
        ('noc', 'log-parser', 'gpt-3.5-turbo', 35, 200, 300),
        ('noc', 'alert-triage', 'claude-haiku-4-5-20251001', 20, 180, 250),

        # Automation team - API heavy
        ('automation', 'config-generator', 'claude-sonnet-4-20250514', 50, 450, 700),
        ('automation', 'test-generator', 'gpt-3.5-turbo', 40, 220, 320),
    ]

    for team, project, model, requests, input_tokens, output_tokens in workload:
        for i in range(requests):
            monitor.track_request(
                trace_id=f"trace-{team}-{i}",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=900,
                success=True,
                team=team,
                project=project,
                environment='production'
            )

    # Display cost attribution
    cost_by_team = monitor.get_cost_attribution('team')

    print("Cost Attribution by Team:\n")

    total_cost = sum(cost_by_team.values())

    for team, cost in sorted(cost_by_team.items(), key=lambda x: x[1], reverse=True):
        percent = (cost / total_cost) * 100
        daily_cost = cost * 500  # Extrapolate to daily
        monthly_cost = daily_cost * 30

        print(f"  {team:20s}")
        print(f"    Current:  ${cost:.4f}")
        print(f"    Daily:    ${daily_cost:.2f}")
        print(f"    Monthly:  ${monthly_cost:.2f} ({percent:.1f}%)")
        print()

    print(f"Total Cost:")
    print(f"  Current:  ${total_cost:.4f}")
    print(f"  Daily:    ${total_cost * 500:.2f}")
    print(f"  Monthly:  ${total_cost * 15000:.2f}")

    print("\n" + "=" * 70 + "\n")


def example_4_alerting_rules():
    """
    Example 4: Alerting and SLO violations

    Automatic alerts when metrics exceed thresholds.
    """
    print("=" * 70)
    print("Example 4: Alerting & SLO Monitoring")
    print("=" * 70)

    monitor = ProductionMonitor(service_name="network-ai-prod")

    print("\n🚨 Testing alert rules...\n")

    # Scenario 1: Normal operation
    print("Scenario 1: Normal Operation")
    for i in range(20):
        monitor.track_request(
            trace_id=f"trace-{i}",
            model='claude-haiku-4-5-20251001',
            input_tokens=300,
            output_tokens=400,
            latency_ms=800,
            success=True,
            team='network-ops'
        )

    slo_status = monitor.check_slos()
    alerts = monitor.check_alerts()

    print(f"  SLO Status:")
    for slo_name, status in slo_status.items():
        emoji = "✅" if status == SLOStatus.HEALTHY else "⚠️" if status == SLOStatus.DEGRADED else "❌"
        print(f"    {emoji} {slo_name}: {status.value}")

    print(f"  Active Alerts: {len(alerts)}")
    print()

    # Scenario 2: High error rate
    print("Scenario 2: High Error Rate (Triggering Alert)")
    for i in range(20):
        monitor.track_request(
            trace_id=f"trace-error-{i}",
            model='gpt-4-turbo',
            input_tokens=500,
            output_tokens=800,
            latency_ms=1000,
            success=(i % 3) != 0,  # 33% error rate
            team='security'
        )

    slo_status = monitor.check_slos()
    alerts = monitor.check_alerts()

    print(f"  SLO Status:")
    for slo_name, status in slo_status.items():
        emoji = "✅" if status == SLOStatus.HEALTHY else "⚠️" if status == SLOStatus.DEGRADED else "❌"
        print(f"    {emoji} {slo_name}: {status.value}")

    print(f"\n  Active Alerts: {len(alerts)}")
    for alert in alerts:
        severity_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        print(f"    {severity_emoji[alert.severity.value]} [{alert.severity.value.upper()}] {alert.title}")
        print(f"       {alert.description}")
        print(f"       Current: {alert.current_value:.2f}, Threshold: {alert.threshold:.2f}")
        print()

    # Scenario 3: High latency
    print("Scenario 3: High Latency")
    for i in range(20):
        monitor.track_request(
            trace_id=f"trace-slow-{i}",
            model='gpt-4',
            input_tokens=1000,
            output_tokens=2000,
            latency_ms=3500,  # Very slow
            success=True,
            team='automation'
        )

    alerts = monitor.check_alerts()
    new_alerts = [a for a in alerts if 'latency' in a.metric_name]

    if new_alerts:
        print(f"  New Alerts Triggered: {len(new_alerts)}")
        for alert in new_alerts:
            print(f"    ⚠️  {alert.title}: {alert.current_value:.0f}ms (threshold: {alert.threshold:.0f}ms)")
    print()

    print("=" * 70 + "\n")


def example_5_performance_dashboard():
    """
    Example 5: Complete production dashboard

    Real-time dashboard with all metrics, SLOs, alerts, and costs.
    """
    print("=" * 70)
    print("Example 5: Production Performance Dashboard")
    print("=" * 70)

    monitor = ProductionMonitor(service_name="network-ai-prod")

    print("\n🚀 Simulating 24-hour production workload...\n")

    # Simulate realistic production traffic patterns
    models = ['claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001', 'gpt-4-turbo', 'gpt-3.5-turbo']
    teams = ['network-ops', 'security', 'noc', 'automation']

    # Peak hours: more traffic, some errors, varied latency
    for i in range(200):
        model = models[i % len(models)]
        team = teams[i % len(teams)]

        # Token usage varies by model
        if 'gpt-4' in model:
            input_tokens = 600
            output_tokens = 1000
        elif 'sonnet' in model:
            input_tokens = 400
            output_tokens = 700
        else:
            input_tokens = 250
            output_tokens = 400

        # Latency varies
        base_latency = 900
        if 'gpt-4' in model:
            base_latency = 1500
        latency = base_latency + (i % 15) * 50

        # Error rate: 2%
        success = (i % 50) != 0

        monitor.track_request(
            trace_id=f"trace-prod-{i}",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            success=success,
            team=team,
            project='production'
        )

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()

    # Display dashboard
    print("=" * 70)
    print("📊 PRODUCTION DASHBOARD")
    print("=" * 70)
    print(f"\nService: {dashboard['service']}")
    print(f"Timestamp: {dashboard['timestamp'][:19]}")
    print()

    print("━" * 70)
    print("TRAFFIC METRICS")
    print("━" * 70)
    m = dashboard['metrics']
    print(f"  Requests:             {m['requests']:,}")
    print(f"  Errors:               {m['errors']}")
    print(f"  Success Rate:         {m['success_rate']:.2f}%")
    print()

    print("━" * 70)
    print("LATENCY (ms)")
    print("━" * 70)
    print(f"  P50:                  {m['p50_latency_ms']:.0f}ms")
    print(f"  P95:                  {m['p95_latency_ms']:.0f}ms")
    print(f"  P99:                  {m['p99_latency_ms']:.0f}ms")
    print()

    print("━" * 70)
    print("TOKEN USAGE")
    print("━" * 70)
    print(f"  Total Tokens:         {m['total_tokens']:,}")
    print(f"  Avg per Request:      {m['avg_tokens_per_request']:.0f}")
    print()

    print("━" * 70)
    print("COST ANALYSIS")
    print("━" * 70)
    print(f"  Total Cost:           ${m['total_cost_usd']:.4f}")
    print(f"  Cost per Request:     ${m['cost_per_request']:.6f}")
    print(f"  Projected Daily:      ${m['total_cost_usd'] * 250:.2f}")
    print(f"  Projected Monthly:    ${m['total_cost_usd'] * 7500:.2f}")
    print()

    print("  Cost by Team:")
    for team, cost in sorted(dashboard['cost_attribution'].items(), key=lambda x: x[1], reverse=True):
        percent = (cost / m['total_cost_usd']) * 100
        print(f"    {team:20s} ${cost:.4f} ({percent:.1f}%)")
    print()

    print("━" * 70)
    print("SLO COMPLIANCE")
    print("━" * 70)
    for slo_name, status in dashboard['slo_status'].items():
        emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
        print(f"  {emoji} {slo_name:25s} {status.upper()}")
    print()

    print("━" * 70)
    print("ACTIVE ALERTS")
    print("━" * 70)
    if dashboard['active_alerts'] > 0:
        print(f"  🚨 {dashboard['active_alerts']} active alert(s)")
        active = [a for a in monitor.alerts.values() if a.status == "active"]
        for alert in active[:3]:
            print(f"     [{alert.severity.value.upper()}] {alert.title}")
            print(f"     {alert.description}")
    else:
        print("  ✅ No active alerts")
    print()

    print("=" * 70)
    print("\n💡 Dashboard Features:")
    print("  • Real-time metrics from 200 production requests")
    print("  • Multi-dimensional cost attribution (team, project)")
    print("  • SLO tracking (success rate, latency, error rate)")
    print("  • Automatic alerting on threshold violations")
    print("  • Cost projection to monthly estimates")
    print()
    print("Typical Production Scale (50K req/day):")
    print(f"  Daily Cost:    ${m['total_cost_usd'] * 250:.2f}")
    print(f"  Monthly Cost:  ${m['total_cost_usd'] * 7500:.2f} (${m['total_cost_usd'] * 7500:.0f})")
    print()

    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Chapter 48: Production Monitoring & Observability")
    print("Real-time monitoring for AI-powered network operations")
    print("=" * 70 + "\n")

    try:
        # Example 1: Metrics collection
        monitor = example_1_metrics_collection()
        input("Press Enter to continue...")

        # Example 2: Structured logging
        example_2_structured_logging()
        input("Press Enter to continue...")

        # Example 3: Cost tracking
        example_3_cost_tracking()
        input("Press Enter to continue...")

        # Example 4: Alerting
        example_4_alerting_rules()
        input("Press Enter to continue...")

        # Example 5: Performance dashboard
        example_5_performance_dashboard()

        # Summary
        print("=" * 70)
        print("✅ All Examples Completed!")
        print("=" * 70)
        print("\n💡 Key Takeaways:")
        print("  • Metrics collection tracks requests, tokens, costs, latency")
        print("  • Structured JSON logs enable debugging and trace correlation")
        print("  • Cost attribution enables chargebacks by team/project")
        print("  • SLO monitoring detects degradation before user impact")
        print("  • Automated alerting reduces MTTR (Mean Time To Resolution)")
        print("  • Production dashboards provide real-time visibility")
        print()
        print("📊 Production Scale Examples:")
        print("  • 50K requests/day = ~35 req/min")
        print("  • Monthly cost: $2,500-3,500 (typical network AI workload)")
        print("  • P99 latency target: <2 seconds")
        print("  • Success rate SLO: >99%")
        print()
        print("🔧 Integration Points:")
        print("  • Export metrics to Prometheus/Grafana")
        print("  • Send logs to ELK/Splunk/Datadog")
        print("  • Alert to PagerDuty/Slack/OpsGenie")
        print("  • Cost data to CloudHealth/Kubecost")
        print()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
