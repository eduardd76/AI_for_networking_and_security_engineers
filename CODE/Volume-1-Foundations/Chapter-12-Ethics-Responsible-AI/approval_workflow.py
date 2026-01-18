#!/usr/bin/env python3
"""
Approval Workflow - Human-in-the-Loop AI Decisions

Implement approval workflow for high-risk AI-generated changes.

From: AI for Networking Engineers - Volume 1, Chapter 12
Author: Eduard Dulharu

Usage:
    from approval_workflow import ApprovalWorkflow

    workflow = ApprovalWorkflow()
    request = workflow.request_approval(change_description, risk_level="high")
    if workflow.is_approved(request.request_id):
        # Apply change
        pass
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    """Approval request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Approval request data."""
    request_id: str
    timestamp: str
    change_description: str
    ai_recommendation: Dict[str, Any]
    risk_level: RiskLevel
    impact_assessment: Dict[str, Any]
    requested_by: str
    status: ApprovalStatus
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    expires_at: Optional[str] = None


class ApprovalWorkflow:
    """
    Implement approval workflow for AI-generated changes.

    Features:
    - Risk-based approval requirements
    - Timeout for approvals
    - Multi-level approval for critical changes
    - Audit trail
    - Notification system integration
    """

    # Approval requirements by risk level
    APPROVAL_REQUIREMENTS = {
        RiskLevel.LOW: {
            "auto_approve": True,
            "approvers_required": 0,
            "timeout_hours": 24
        },
        RiskLevel.MEDIUM: {
            "auto_approve": False,
            "approvers_required": 1,
            "timeout_hours": 4
        },
        RiskLevel.HIGH: {
            "auto_approve": False,
            "approvers_required": 2,
            "timeout_hours": 2
        },
        RiskLevel.CRITICAL: {
            "auto_approve": False,
            "approvers_required": 3,
            "timeout_hours": 1
        }
    }

    def __init__(self, requests_dir: str = "./approval_requests"):
        """
        Initialize approval workflow.

        Args:
            requests_dir: Directory for approval requests
        """
        self.requests_dir = Path(requests_dir)
        self.requests_dir.mkdir(parents=True, exist_ok=True)

    def request_approval(
        self,
        change_description: str,
        ai_recommendation: Dict[str, Any],
        risk_level: RiskLevel,
        impact_assessment: Dict[str, Any],
        requested_by: str = "ai_system"
    ) -> ApprovalRequest:
        """
        Create approval request.

        Args:
            change_description: Description of proposed change
            ai_recommendation: AI's recommendation
            risk_level: Risk level assessment
            impact_assessment: Impact analysis
            requested_by: User or system requesting approval

        Returns:
            ApprovalRequest object
        """
        # Generate request ID
        timestamp = datetime.now()
        request_id = f"AR-{timestamp.strftime('%Y%m%d-%H%M%S')}"

        # Get approval requirements
        requirements = self.APPROVAL_REQUIREMENTS[risk_level]

        # Calculate expiration
        if requirements['timeout_hours'] > 0:
            expires_at = (timestamp + timedelta(hours=requirements['timeout_hours'])).isoformat()
        else:
            expires_at = None

        # Auto-approve low-risk changes
        if requirements['auto_approve']:
            status = ApprovalStatus.APPROVED
            approved_by = "auto_approved"
            approved_at = timestamp.isoformat()
        else:
            status = ApprovalStatus.PENDING
            approved_by = None
            approved_at = None

        request = ApprovalRequest(
            request_id=request_id,
            timestamp=timestamp.isoformat(),
            change_description=change_description,
            ai_recommendation=ai_recommendation,
            risk_level=risk_level,
            impact_assessment=impact_assessment,
            requested_by=requested_by,
            status=status,
            approved_by=approved_by,
            approved_at=approved_at,
            expires_at=expires_at
        )

        # Save request
        self._save_request(request)

        # Send notification (mock)
        self._send_notification(request)

        return request

    def approve_request(
        self,
        request_id: str,
        approved_by: str,
        comments: Optional[str] = None
    ) -> bool:
        """
        Approve a pending request.

        Args:
            request_id: Request ID
            approved_by: Approver username
            comments: Optional approval comments

        Returns:
            True if approved, False if not found or already processed
        """
        request = self._load_request(request_id)

        if not request:
            print(f"Request not found: {request_id}")
            return False

        if request.status != ApprovalStatus.PENDING:
            print(f"Request already {request.status}: {request_id}")
            return False

        # Check if expired
        if request.expires_at:
            expires = datetime.fromisoformat(request.expires_at)
            if datetime.now() > expires:
                request.status = ApprovalStatus.EXPIRED
                self._save_request(request)
                print(f"Request expired: {request_id}")
                return False

        # Approve
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now().isoformat()

        # Save updated request
        self._save_request(request)

        # Log approval
        self._log_approval(request, comments)

        print(f"✓ Request approved: {request_id} by {approved_by}")
        return True

    def reject_request(
        self,
        request_id: str,
        rejected_by: str,
        reason: str
    ) -> bool:
        """
        Reject a pending request.

        Args:
            request_id: Request ID
            rejected_by: Rejector username
            reason: Rejection reason

        Returns:
            True if rejected, False if not found or already processed
        """
        request = self._load_request(request_id)

        if not request:
            print(f"Request not found: {request_id}")
            return False

        if request.status != ApprovalStatus.PENDING:
            print(f"Request already {request.status}: {request_id}")
            return False

        # Reject
        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = reason

        # Save updated request
        self._save_request(request)

        # Log rejection
        self._log_rejection(request, rejected_by, reason)

        print(f"✗ Request rejected: {request_id} by {rejected_by}")
        return True

    def is_approved(self, request_id: str) -> bool:
        """
        Check if request is approved.

        Args:
            request_id: Request ID

        Returns:
            True if approved, False otherwise
        """
        request = self._load_request(request_id)

        if not request:
            return False

        # Check expiration
        if request.status == ApprovalStatus.PENDING and request.expires_at:
            expires = datetime.fromisoformat(request.expires_at)
            if datetime.now() > expires:
                request.status = ApprovalStatus.EXPIRED
                self._save_request(request)
                return False

        return request.status == ApprovalStatus.APPROVED

    def get_pending_requests(
        self,
        risk_level: Optional[RiskLevel] = None
    ) -> List[ApprovalRequest]:
        """
        Get all pending approval requests.

        Args:
            risk_level: Optional filter by risk level

        Returns:
            List of pending requests
        """
        pending = []

        for request_file in self.requests_dir.glob("*.json"):
            request = self._load_request(request_file.stem)

            if request and request.status == ApprovalStatus.PENDING:
                # Check expiration
                if request.expires_at:
                    expires = datetime.fromisoformat(request.expires_at)
                    if datetime.now() > expires:
                        request.status = ApprovalStatus.EXPIRED
                        self._save_request(request)
                        continue

                # Filter by risk level
                if risk_level and request.risk_level != risk_level:
                    continue

                pending.append(request)

        # Sort by risk level (critical first) and timestamp
        pending.sort(
            key=lambda r: (
                ['critical', 'high', 'medium', 'low'].index(r.risk_level.value),
                r.timestamp
            )
        )

        return pending

    def _save_request(self, request: ApprovalRequest) -> None:
        """Save approval request to file."""
        request_file = self.requests_dir / f"{request.request_id}.json"

        with open(request_file, 'w') as f:
            json.dump(asdict(request), f, indent=2)

    def _load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load approval request from file."""
        request_file = self.requests_dir / f"{request_id}.json"

        if not request_file.exists():
            return None

        with open(request_file, 'r') as f:
            data = json.load(f)

        # Convert enums
        data['risk_level'] = RiskLevel(data['risk_level'])
        data['status'] = ApprovalStatus(data['status'])

        return ApprovalRequest(**data)

    def _send_notification(self, request: ApprovalRequest) -> None:
        """
        Send notification about approval request.

        Args:
            request: Approval request
        """
        # Mock notification - would integrate with Slack, email, etc.
        if request.status == ApprovalStatus.PENDING:
            print(f"\n📧 Notification: Approval required for {request.request_id}")
            print(f"   Risk Level: {request.risk_level.value}")
            print(f"   Change: {request.change_description[:60]}...")
            print(f"   Expires: {request.expires_at}")

    def _log_approval(
        self,
        request: ApprovalRequest,
        comments: Optional[str]
    ) -> None:
        """Log approval decision."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "approval",
            "request_id": request.request_id,
            "approved_by": request.approved_by,
            "comments": comments
        }

        log_file = self.requests_dir / "approval_log.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _log_rejection(
        self,
        request: ApprovalRequest,
        rejected_by: str,
        reason: str
    ) -> None:
        """Log rejection decision."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "rejection",
            "request_id": request.request_id,
            "rejected_by": rejected_by,
            "reason": reason
        }

        log_file = self.requests_dir / "approval_log.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def generate_approval_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate approval statistics report.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Report with approval statistics
        """
        all_requests = []

        for request_file in self.requests_dir.glob("*.json"):
            request = self._load_request(request_file.stem)
            if request:
                all_requests.append(request)

        # Filter by date
        if start_date or end_date:
            filtered = []
            for request in all_requests:
                request_time = datetime.fromisoformat(request.timestamp)

                if start_date and request_time < start_date:
                    continue
                if end_date and request_time > end_date:
                    continue

                filtered.append(request)

            all_requests = filtered

        # Calculate statistics
        total = len(all_requests)
        approved = sum(1 for r in all_requests if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in all_requests if r.status == ApprovalStatus.REJECTED)
        pending = sum(1 for r in all_requests if r.status == ApprovalStatus.PENDING)
        expired = sum(1 for r in all_requests if r.status == ApprovalStatus.EXPIRED)

        # By risk level
        by_risk = {}
        for level in RiskLevel:
            count = sum(1 for r in all_requests if r.risk_level == level)
            by_risk[level.value] = count

        return {
            "summary": {
                "total_requests": total,
                "approved": approved,
                "rejected": rejected,
                "pending": pending,
                "expired": expired,
                "approval_rate": f"{(approved/total*100):.1f}%" if total > 0 else "0%"
            },
            "by_risk_level": by_risk
        }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Approval Workflow Demo
    ========================================
    Human-in-the-loop AI decisions
    ========================================
    """)

    workflow = ApprovalWorkflow(requests_dir="./demo_approvals")

    # Test 1: Low-risk change (auto-approved)
    print("\nTest 1: Low-Risk Change (Auto-Approved)")
    print("-" * 60)

    request = workflow.request_approval(
        change_description="Update interface description",
        ai_recommendation={"action": "update_description", "interface": "Gi0/1"},
        risk_level=RiskLevel.LOW,
        impact_assessment={"affected_services": [], "downtime_risk": "none"},
        requested_by="ai_config_assistant"
    )

    print(f"Request ID: {request.request_id}")
    print(f"Status: {request.status.value}")
    print(f"Auto-approved: {request.status == ApprovalStatus.APPROVED}")

    # Test 2: High-risk change (requires approval)
    print("\n\nTest 2: High-Risk Change (Requires Approval)")
    print("-" * 60)

    request = workflow.request_approval(
        change_description="Modify routing protocol configuration",
        ai_recommendation={
            "action": "modify_ospf",
            "changes": ["redistribute bgp", "default-information originate"]
        },
        risk_level=RiskLevel.HIGH,
        impact_assessment={
            "affected_services": ["routing", "internet_connectivity"],
            "downtime_risk": "high"
        },
        requested_by="ai_routing_optimizer"
    )

    print(f"Request ID: {request.request_id}")
    print(f"Status: {request.status.value}")
    print(f"Expires at: {request.expires_at}")

    # Test 3: Approve request
    print("\n\nTest 3: Approve Request")
    print("-" * 60)

    success = workflow.approve_request(
        request_id=request.request_id,
        approved_by="network_admin",
        comments="Reviewed and approved. Looks good."
    )

    print(f"Approval successful: {success}")
    print(f"Is approved: {workflow.is_approved(request.request_id)}")

    # Test 4: Reject request
    print("\n\nTest 4: Reject Request")
    print("-" * 60)

    critical_request = workflow.request_approval(
        change_description="Shutdown core router for maintenance",
        ai_recommendation={"action": "shutdown", "device": "CORE-RTR-01"},
        risk_level=RiskLevel.CRITICAL,
        impact_assessment={
            "affected_services": ["all"],
            "downtime_risk": "critical"
        }
    )

    success = workflow.reject_request(
        request_id=critical_request.request_id,
        rejected_by="network_admin",
        reason="Maintenance window not approved. Schedule during off-hours."
    )

    print(f"Rejection successful: {success}")

    # Test 5: Get pending requests
    print("\n\nTest 5: Pending Requests")
    print("-" * 60)

    # Create more pending requests
    for i in range(3):
        workflow.request_approval(
            change_description=f"Test change {i}",
            ai_recommendation={"test": i},
            risk_level=RiskLevel.MEDIUM,
            impact_assessment={}
        )

    pending = workflow.get_pending_requests()
    print(f"Total pending requests: {len(pending)}")

    for req in pending:
        print(f"  - {req.request_id}: {req.risk_level.value} - {req.change_description[:40]}...")

    # Test 6: Generate report
    print("\n\nTest 6: Approval Report")
    print("-" * 60)

    report = workflow.generate_approval_report()

    print(f"\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key:20s}: {value}")

    print(f"\nBy Risk Level:")
    for risk, count in report['by_risk_level'].items():
        print(f"  {risk:12s}: {count}")

    # Clean up
    import shutil
    shutil.rmtree("./demo_approvals")

    print("\n✅ Demo complete!")
    print("\n💡 Approval Workflow Benefits:")
    print("  - Human oversight for high-risk changes")
    print("  - Audit trail of all decisions")
    print("  - Risk-based approval requirements")
    print("  - Prevents unauthorized changes")
    print("  - Compliance with change management policies")

    print("\n⚠️  Best Practices:")
    print("  ☐ Auto-approve only low-risk changes")
    print("  ☐ Set appropriate timeout periods")
    print("  ☐ Require multiple approvals for critical changes")
    print("  ☐ Integrate with notification systems (Slack, email)")
    print("  ☐ Log all approval decisions")
    print("  ☐ Review approval metrics regularly")
