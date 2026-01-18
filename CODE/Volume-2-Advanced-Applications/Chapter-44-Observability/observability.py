"""
Chapter 44: Observability & Distributed Tracing
Monitor and debug AI-powered network operations

This module demonstrates production observability including distributed
tracing, metrics collection, and log aggregation for LLM systems.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class SpanStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class LogEntry:
    """Structured log event."""
    id: str
    timestamp: str
    level: LogLevel
    event: str
    trace_id: str
    details: Dict


@dataclass
class TraceSpan:
    """Distributed trace span."""
    id: str
    name: str
    trace_id: str
    parent_span_id: Optional[str] = None
    start_time_ns: int = 0
    duration_ns: int = 0
    status: SpanStatus = SpanStatus.SUCCESS
    attributes: Dict = field(default_factory=dict)
    children: List['TraceSpan'] = field(default_factory=list)


@dataclass
class Metric:
    """Real-time metric."""
    name: str
    value: float
    unit: str
    timestamp: str


class ObservabilityCollector:
    """
    Collect and analyze observability data.

    Features:
    - Distributed tracing with parent/child spans
    - Structured logging with trace correlation
    - Real-time metrics (request rate, error rate, latency, cost)
    - Slow request analysis
    - Span breakdown by operation type
    """

    def __init__(self):
        self.logs: List[LogEntry] = []
        self.spans: Dict[str, TraceSpan] = {}
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)

        # Counters
        self.request_count = 0
        self.error_count = 0
        self.total_cost_usd = 0.0

    def create_trace(self, trace_id: str) -> str:
        """Initialize a new trace."""
        self.traces[trace_id] = []
        return trace_id

    def start_span(
        self,
        trace_id: str,
        span_name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict] = None
    ) -> str:
        """Start a new trace span."""
        span_id = f"{trace_id}-{span_name}-{int(time.time_ns())}"

        span = TraceSpan(
            id=span_id,
            name=span_name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            start_time_ns=int(time.time_ns()),
            attributes=attributes or {}
        )

        self.spans[span_id] = span
        self.traces[trace_id].append(span)

        # Link to parent
        if parent_span_id and parent_span_id in self.spans:
            parent = self.spans[parent_span_id]
            parent.children.append(span)

        return span_id

    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.SUCCESS
    ):
        """End a trace span."""
        if span_id not in self.spans:
            return

        span = self.spans[span_id]
        span.duration_ns = int(time.time_ns()) - span.start_time_ns
        span.status = status

    def log_event(
        self,
        trace_id: str,
        event_name: str,
        level: LogLevel = LogLevel.INFO,
        details: Optional[Dict] = None
    ):
        """Log structured event."""
        log_id = f"{trace_id}-{int(time.time_ns())}"

        log = LogEntry(
            id=log_id,
            timestamp=datetime.now().isoformat(),
            level=level,
            event=event_name,
            trace_id=trace_id,
            details=details or {}
        )

        self.logs.append(log)

    def record_metric(self, metric_name: str, value: float, unit: str = ""):
        """Record a metric value."""
        metric = Metric(
            name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat()
        )

        self.metrics[metric_name].append(metric)

    def get_current_metrics(self) -> Dict[str, Metric]:
        """Get latest metric values."""
        current = {}

        # Request rate (req/min)
        now = datetime.now()
        one_min_ago = now - timedelta(minutes=1)

        recent_spans = [
            s for s in self.spans.values()
            if s.parent_span_id is None and
            datetime.fromisoformat(s.attributes.get('start_time', '1970-01-01')) > one_min_ago
        ]

        current['request_rate'] = Metric(
            name='request_rate',
            value=len(recent_spans),
            unit='req/min',
            timestamp=now.isoformat()
        )

        # Error rate
        total_requests = max(1, len([s for s in self.spans.values() if s.parent_span_id is None]))
        error_rate = (self.error_count / total_requests) * 100

        current['error_rate'] = Metric(
            name='error_rate',
            value=error_rate,
            unit='%',
            timestamp=now.isoformat()
        )

        # P99 Latency
        root_spans = [s for s in self.spans.values() if s.parent_span_id is None]

        if root_spans:
            latencies = sorted([s.duration_ns / 1_000_000 for s in root_spans])
            p99_idx = int(len(latencies) * 0.99)
            p99_latency = latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1]
        else:
            p99_latency = 0.0

        current['p99_latency'] = Metric(
            name='p99_latency',
            value=p99_latency,
            unit='ms',
            timestamp=now.isoformat()
        )

        # Cost per hour
        current['cost_per_hour'] = Metric(
            name='cost_per_hour',
            value=self.total_cost_usd,
            unit='USD',
            timestamp=now.isoformat()
        )

        return current

    def get_trace(self, trace_id: str) -> Optional[List[TraceSpan]]:
        """Retrieve full trace."""
        return self.traces.get(trace_id)

    def get_recent_logs(self, limit: int = 50) -> List[LogEntry]:
        """Get most recent logs."""
        return sorted(self.logs[-limit:], key=lambda x: x.timestamp, reverse=True)

    def analyze_slow_requests(self, threshold_ms: int = 1000) -> List[Tuple[str, float]]:
        """Identify slow requests."""
        slow_spans = []

        for span in self.spans.values():
            if span.parent_span_id is None:  # Root spans only
                duration_ms = span.duration_ns / 1_000_000

                if duration_ms > threshold_ms:
                    slow_spans.append((span.trace_id, duration_ms))

        return sorted(slow_spans, key=lambda x: x[1], reverse=True)

    def get_span_breakdown(self, trace_id: str) -> Dict[str, Dict]:
        """Breakdown of time spent in each span type."""
        spans = self.traces.get(trace_id, [])
        breakdown = defaultdict(lambda: {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0})

        for span in spans:
            duration_ms = span.duration_ns / 1_000_000
            breakdown[span.name]['count'] += 1
            breakdown[span.name]['total_ms'] += duration_ms

        # Calculate averages
        for span_type, stats in breakdown.items():
            stats['avg_ms'] = stats['total_ms'] / stats['count']

        return dict(breakdown)

    def simulate_llm_request(
        self,
        model: str = "gpt-4",
        input_tokens: int = 150,
        output_tokens: int = 300,
        is_error: bool = False
    ) -> str:
        """Simulate a complete LLM request with tracing."""
        trace_id = f"trace-{int(time.time_ns())}"
        self.create_trace(trace_id)
        self.request_count += 1

        # Root span
        root_span = self.start_span(
            trace_id,
            'llm_request',
            attributes={
                'start_time': datetime.now().isoformat(),
                'model': model
            }
        )

        self.log_event(
            trace_id,
            'llm_request_started',
            details={'model': model, 'input_tokens': input_tokens}
        )

        # Child span 1: Prompt processing
        prompt_span = self.start_span(
            trace_id,
            'prompt_processing',
            parent_span_id=root_span
        )
        time.sleep(0.02)
        self.end_span(prompt_span)

        # Child span 2: LLM inference
        inference_span = self.start_span(
            trace_id,
            'llm_inference',
            parent_span_id=root_span,
            attributes={'vendor': 'openai'}
        )
        time.sleep(0.05)

        if is_error:
            self.end_span(inference_span, SpanStatus.ERROR)
            self.error_count += 1
        else:
            self.end_span(inference_span)

        # Child span 3: Response formatting
        response_span = self.start_span(
            trace_id,
            'response_formatting',
            parent_span_id=root_span
        )
        time.sleep(0.01)
        self.end_span(response_span)

        # End root span
        self.end_span(root_span)

        # Calculate cost
        input_cost = (input_tokens / 1000) * 0.03
        output_cost = (output_tokens / 1000) * 0.06
        request_cost = input_cost + output_cost
        self.total_cost_usd += request_cost

        # Log completion
        level = LogLevel.ERROR if is_error else LogLevel.INFO
        event = 'llm_request_failed' if is_error else 'llm_request_completed'

        self.log_event(
            trace_id,
            event,
            level=level,
            details={
                'model': model,
                'latency_ms': self.spans[root_span].duration_ns / 1_000_000,
                'tokens': input_tokens + output_tokens,
                'cost_usd': request_cost,
                'error': 'RateLimitError' if is_error else None
            }
        )

        return trace_id


def example_1_basic_tracing():
    """
    Example 1: Basic distributed tracing
    """
    print("=" * 60)
    print("Example 1: Distributed Tracing Basics")
    print("=" * 60)

    collector = ObservabilityCollector()

    # Simulate a simple request
    trace_id = "trace-001"
    collector.create_trace(trace_id)

    print("\n🔍 Creating trace with nested spans...\n")

    # Root span
    root = collector.start_span(
        trace_id,
        "process_config_request",
        attributes={'user': 'admin', 'device': 'router-01'}
    )
    time.sleep(0.05)

    # Child spans
    parse_span = collector.start_span(trace_id, "parse_config", parent_span_id=root)
    time.sleep(0.02)
    collector.end_span(parse_span)

    analyze_span = collector.start_span(trace_id, "analyze_config", parent_span_id=root)
    time.sleep(0.03)
    collector.end_span(analyze_span)

    collector.end_span(root)

    # Display trace
    trace = collector.get_trace(trace_id)

    print(f"Trace ID: {trace_id}\n")

    for span in trace:
        indent = "  " if span.parent_span_id else ""
        duration = span.duration_ns / 1_000_000

        print(f"{indent}Span: {span.name}")
        print(f"{indent}  Duration: {duration:.2f}ms")
        print(f"{indent}  Status: {span.status.value}")
        print()

    print("=" * 60 + "\n")


def example_2_collect_metrics():
    """
    Example 2: Real-time metrics collection
    """
    print("=" * 60)
    print("Example 2: Real-Time Metrics Dashboard")
    print("=" * 60)

    collector = ObservabilityCollector()

    print("\n📊 Simulating 20 LLM requests...\n")

    # Simulate requests
    for i in range(20):
        is_error = (i % 10 == 9)  # 10% error rate
        collector.simulate_llm_request(
            model="gpt-4",
            input_tokens=150,
            output_tokens=300,
            is_error=is_error
        )

    time.sleep(0.1)

    # Get metrics
    metrics = collector.get_current_metrics()

    print("Current Metrics:")
    print(f"  Request Rate: {metrics['request_rate'].value:.0f} {metrics['request_rate'].unit}")
    print(f"  Error Rate: {metrics['error_rate'].value:.1f}{metrics['error_rate'].unit}")
    print(f"  P99 Latency: {metrics['p99_latency'].value:.0f}{metrics['p99_latency'].unit}")
    print(f"  Cost/Hour: ${metrics['cost_per_hour'].value:.4f}")

    print("\n" + "=" * 60 + "\n")


def example_3_structured_logging():
    """
    Example 3: Structured logging with trace correlation
    """
    print("=" * 60)
    print("Example 3: Structured Logging")
    print("=" * 60)

    collector = ObservabilityCollector()

    # Create trace with logs
    trace_id = collector.simulate_llm_request(model="claude-3-sonnet")

    print(f"\n📝 Logs for trace {trace_id[:12]}...\n")

    # Get logs for this trace
    trace_logs = [log for log in collector.logs if log.trace_id == trace_id]

    for log in trace_logs:
        emoji = {
            'debug': '🔵',
            'info': '🟢',
            'warn': '🟡',
            'error': '🔴'
        }.get(log.level.value, '⚪')

        print(f"{emoji} [{log.level.value.upper()}] {log.event}")

        if log.details:
            for key, value in log.details.items():
                if value is not None:
                    print(f"     {key}: {value}")

        print()

    print("=" * 60 + "\n")


def example_4_slow_request_analysis():
    """
    Example 4: Identify and analyze slow requests
    """
    print("=" * 60)
    print("Example 4: Slow Request Analysis")
    print("=" * 60)

    collector = ObservabilityCollector()

    # Simulate mix of fast and slow requests
    print("\n⏱️  Simulating requests with varying latency...\n")

    for i in range(10):
        # Artificially slow down some requests
        if i in [3, 7]:
            time.sleep(0.05)  # Slow request

        collector.simulate_llm_request(model="gpt-4")

    # Analyze slow requests
    slow_requests = collector.analyze_slow_requests(threshold_ms=50)

    print(f"Found {len(slow_requests)} slow requests (>50ms):\n")

    for trace_id, latency in slow_requests[:5]:
        print(f"  Trace: {trace_id[:20]}...")
        print(f"  Latency: {latency:.2f}ms")

        # Get span breakdown
        breakdown = collector.get_span_breakdown(trace_id)

        print("  Time Breakdown:")
        for span_name, stats in breakdown.items():
            print(f"    {span_name}: {stats['avg_ms']:.2f}ms")

        print()

    print("=" * 60 + "\n")


def example_5_production_dashboard():
    """
    Example 5: Complete production monitoring dashboard
    """
    print("=" * 60)
    print("Example 5: Production Monitoring Dashboard")
    print("=" * 60)

    collector = ObservabilityCollector()

    print("\n🚀 Simulating production traffic...\n")

    # Simulate 1 minute of production traffic
    for i in range(50):
        # 5% error rate
        is_error = (i % 20 == 0)

        # Vary models
        models = ['gpt-4', 'gpt-4', 'gpt-3.5-turbo', 'claude-3-sonnet']
        model = models[i % len(models)]

        collector.simulate_llm_request(
            model=model,
            input_tokens=150,
            output_tokens=300,
            is_error=is_error
        )

    print("📊 Production Dashboard\n")
    print("-" * 60)

    # Metrics
    metrics = collector.get_current_metrics()

    print("\nKey Metrics:")
    print(f"  ├─ Total Requests: {collector.request_count}")
    print(f"  ├─ Request Rate: {metrics['request_rate'].value:.0f} req/min")
    print(f"  ├─ Error Rate: {metrics['error_rate'].value:.1f}%")
    print(f"  ├─ P99 Latency: {metrics['p99_latency'].value:.0f}ms")
    print(f"  └─ Total Cost: ${metrics['cost_per_hour'].value:.4f}")

    # Error logs
    error_logs = [log for log in collector.logs if log.level == LogLevel.ERROR]

    print(f"\n⚠️  Recent Errors ({len(error_logs)}):")
    for log in error_logs[-3:]:
        print(f"  • {log.event}")
        if 'error' in log.details:
            print(f"    Error: {log.details['error']}")

    # Slow requests
    slow = collector.analyze_slow_requests(threshold_ms=100)

    print(f"\n🐌 Slow Requests ({len(slow)}):")
    for trace_id, latency in slow[:3]:
        print(f"  • Trace {trace_id[:16]}... ({latency:.0f}ms)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n📡 Chapter 44: Observability & Distributed Tracing")
    print("Monitor and Debug AI Systems in Production\n")

    try:
        example_1_basic_tracing()
        input("Press Enter to continue...")

        example_2_collect_metrics()
        input("Press Enter to continue...")

        example_3_structured_logging()
        input("Press Enter to continue...")

        example_4_slow_request_analysis()
        input("Press Enter to continue...")

        example_5_production_dashboard()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Distributed tracing shows request flow across services")
        print("- Structured logging enables correlation and analysis")
        print("- Real-time metrics catch issues before users do")
        print("- Span breakdown identifies performance bottlenecks")
        print("- Production observability is essential for reliability\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
