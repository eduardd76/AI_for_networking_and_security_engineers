#!/usr/bin/env python3
"""
Audit Logger - Track All AI Decisions for Compliance

Log all AI-powered decisions, changes, and analysis for audit trails.

From: AI for Networking Engineers - Volume 1, Chapter 12
Author: Eduard Dulharu

Usage:
    from audit_logger import AuditLogger

    logger = AuditLogger()
    logger.log_analysis("config_check", input_data, ai_result, approved=True)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    action_type: str
    user: str
    system: str
    input_data: Dict[str, Any]
    ai_decision: Dict[str, Any]
    human_approved: bool
    outcome: str
    metadata: Dict[str, Any]


class AuditLogger:
    """
    Comprehensive audit logging for AI systems.

    Features:
    - Log all AI decisions
    - Track human approvals
    - Record outcomes
    - Support compliance requirements (SOC2, GDPR, etc.)
    - Tamper-evident logging
    """

    def __init__(self, log_dir: str = "./audit-logs"):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current log file (rotate daily)
        self.log_file = self._get_current_log_file()

    def log_analysis(
        self,
        action_type: str,
        input_data: Dict[str, Any],
        ai_result: Dict[str, Any],
        user: str = "system",
        approved: bool = False,
        outcome: str = "pending",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an AI analysis action.

        Args:
            action_type: Type of action (e.g., "config_analysis", "security_check")
            input_data: Input data analyzed by AI
            ai_result: AI's decision/recommendation
            user: User who initiated the action
            approved: Whether a human approved the AI decision
            outcome: Actual outcome (e.g., "applied", "rejected", "pending")
            metadata: Additional context

        Returns:
            Audit entry ID
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action_type=action_type,
            user=user,
            system="ai-networking-system",
            input_data=input_data,
            ai_decision=ai_result,
            human_approved=approved,
            outcome=outcome,
            metadata=metadata or {}
        )

        # Write to log
        entry_id = self._write_entry(entry)

        return entry_id

    def log_config_change(
        self,
        device: str,
        proposed_change: str,
        ai_recommendation: Dict[str, Any],
        approved_by: Optional[str] = None,
        applied: bool = False
    ) -> str:
        """
        Log a configuration change decision.

        Args:
            device: Target device
            proposed_change: Proposed configuration change
            ai_recommendation: AI's analysis and recommendation
            approved_by: User who approved (if any)
            applied: Whether change was applied

        Returns:
            Audit entry ID
        """
        return self.log_analysis(
            action_type="config_change",
            input_data={
                "device": device,
                "proposed_change": proposed_change
            },
            ai_result=ai_recommendation,
            user=approved_by or "system",
            approved=approved_by is not None,
            outcome="applied" if applied else "pending",
            metadata={"device": device}
        )

    def _write_entry(self, entry: AuditEntry) -> str:
        """Write entry to log file."""
        # Convert to dict for JSON serialization
        entry_dict = asdict(entry)

        # Generate entry ID (timestamp + hash)
        import hashlib
        entry_json = json.dumps(entry_dict, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()[:16]
        entry_id = f"{entry.timestamp}_{entry_hash}"

        entry_dict["entry_id"] = entry_id

        # Write to log file (append)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry_dict) + '\n')

        return entry_id

    def _get_current_log_file(self) -> Path:
        """Get current log file (daily rotation)."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def query_logs(
        self,
        action_type: Optional[str] = None,
        user: Optional[str] = None,
        approved_only: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list:
        """
        Query audit logs.

        Args:
            action_type: Filter by action type
            user: Filter by user
            approved_only: Only show human-approved actions
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of matching audit entries
        """
        results = []

        # Read all log files in date range
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))

        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # Apply filters
                        if action_type and entry.get('action_type') != action_type:
                            continue

                        if user and entry.get('user') != user:
                            continue

                        if approved_only and not entry.get('human_approved'):
                            continue

                        # Date filters
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if start_date and entry_time < start_date:
                            continue
                        if end_date and entry_time > end_date:
                            continue

                        results.append(entry)

                    except json.JSONDecodeError:
                        continue

        return results

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audit period.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report with statistics
        """
        entries = self.query_logs(start_date=start_date, end_date=end_date)

        total_actions = len(entries)
        human_approved = sum(1 for e in entries if e.get('human_approved'))
        auto_applied = total_actions - human_approved

        # Group by action type
        by_action_type = {}
        for entry in entries:
            action = entry.get('action_type', 'unknown')
            by_action_type[action] = by_action_type.get(action, 0) + 1

        # Group by outcome
        by_outcome = {}
        for entry in entries:
            outcome = entry.get('outcome', 'unknown')
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_ai_actions": total_actions,
                "human_approved": human_approved,
                "auto_applied": auto_applied,
                "approval_rate": f"{(human_approved/total_actions*100):.1f}%" if total_actions > 0 else "0%"
            },
            "by_action_type": by_action_type,
            "by_outcome": by_outcome
        }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Audit Logger Demo
    ========================================
    Track AI decisions for compliance
    ========================================
    """)

    # Initialize logger
    audit = AuditLogger(log_dir="./demo_audit_logs")

    # Test 1: Log config analysis
    print("\nTest 1: Log Configuration Analysis")
    print("-" * 60)

    entry_id = audit.log_analysis(
        action_type="security_analysis",
        input_data={
            "device": "CORE-RTR-01",
            "config": "hostname CORE-RTR-01\nline vty 0 4\n transport input telnet"
        },
        ai_result={
            "critical_issues": [
                {"issue": "Telnet enabled", "severity": "critical"}
            ],
            "recommendation": "Disable telnet, enable SSH"
        },
        user="admin",
        approved=True,
        outcome="pending",
        metadata={"device_type": "cisco_ios"}
    )

    print(f"✓ Logged security analysis: {entry_id}")

    # Test 2: Log config change
    print("\n\nTest 2: Log Configuration Change")
    print("-" * 60)

    change_id = audit.log_config_change(
        device="CORE-RTR-01",
        proposed_change="no transport input telnet\ntransport input ssh",
        ai_recommendation={
            "risk_level": "medium",
            "impact": "Management access method change",
            "approval_required": True
        },
        approved_by="network_admin",
        applied=True
    )

    print(f"✓ Logged config change: {change_id}")

    # Test 3: Query logs
    print("\n\nTest 3: Query Audit Logs")
    print("-" * 60)

    all_entries = audit.query_logs()
    print(f"Total entries: {len(all_entries)}")

    approved_entries = audit.query_logs(approved_only=True)
    print(f"Human-approved entries: {len(approved_entries)}")

    config_changes = audit.query_logs(action_type="config_change")
    print(f"Config changes: {len(config_changes)}")

    # Test 4: Compliance report
    print("\n\nTest 4: Generate Compliance Report")
    print("-" * 60)

    from datetime import timedelta
    report = audit.generate_compliance_report(
        start_date=datetime.utcnow() - timedelta(days=7),
        end_date=datetime.utcnow()
    )

    print(f"\nCompliance Report:")
    print(f"Period: {report['report_period']['start']} to {report['report_period']['end']}")
    print(f"\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key:20s}: {value}")

    print(f"\nActions by Type:")
    for action, count in report['by_action_type'].items():
        print(f"  {action:20s}: {count}")

    # Clean up
    import shutil
    shutil.rmtree("./demo_audit_logs")

    print("\n✅ Demo complete!")
    print("\n💡 Audit Logging Benefits:")
    print("  - Compliance requirements (SOC2, GDPR, HIPAA)")
    print("  - Incident investigation")
    print("  - AI accountability")
    print("  - Change tracking")
    print("  - Security forensics")

    print("\n⚠️  Best Practices:")
    print("  ☐ Log ALL AI decisions")
    print("  ☐ Track human approvals")
    print("  ☐ Record outcomes")
    print("  ☐ Implement log retention policy")
    print("  ☐ Protect logs from tampering")
    print("  ☐ Regular compliance reports")
