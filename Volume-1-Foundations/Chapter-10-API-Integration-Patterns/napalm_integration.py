#!/usr/bin/env python3
"""
NAPALM Integration - Multi-Vendor Network Automation with AI

Combine NAPALM's multi-vendor support with AI analysis for intelligent
network operations across different platforms.

From: AI for Networking Engineers - Volume 1, Chapter 10
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from napalm_integration import AINetworkManager

    manager = AINetworkManager()
    analysis = manager.analyze_device(
        hostname="192.168.1.1",
        username="admin",
        password="password",
        driver="ios"
    )
"""

from typing import Dict, Any, List, Optional
import logging

# NAPALM is optional - graceful degradation if not installed
try:
    from napalm import get_network_driver
    NAPALM_AVAILABLE = True
except ImportError:
    NAPALM_AVAILABLE = False
    print("NAPALM not installed. Install with: pip install napalm")

logger = logging.getLogger(__name__)


class AINetworkManager:
    """
    Multi-vendor network manager with AI-powered analysis.

    Features:
    - Support for Cisco IOS/IOS-XR/NXOS, Juniper, Arista, etc.
    - Unified interface across vendors
    - AI-powered configuration analysis
    - Compliance checking
    - Change validation
    """

    # Supported NAPALM drivers
    SUPPORTED_DRIVERS = {
        'ios': 'Cisco IOS',
        'iosxr': 'Cisco IOS-XR',
        'nxos': 'Cisco NX-OS',
        'eos': 'Arista EOS',
        'junos': 'Juniper Junos',
        'iosxr_netconf': 'Cisco IOS-XR (NETCONF)'
    }

    def __init__(self):
        """Initialize network manager."""
        if not NAPALM_AVAILABLE:
            raise RuntimeError(
                "NAPALM is required but not installed. "
                "Install with: pip install napalm"
            )

    def connect_device(
        self,
        hostname: str,
        username: str,
        password: str,
        driver: str,
        optional_args: Optional[Dict[str, Any]] = None
    ):
        """
        Connect to network device using NAPALM.

        Args:
            hostname: Device IP or hostname
            username: SSH username
            password: SSH password
            driver: NAPALM driver (ios, nxos, junos, eos, etc.)
            optional_args: Optional driver-specific arguments

        Returns:
            NAPALM device object
        """
        if driver not in self.SUPPORTED_DRIVERS:
            raise ValueError(
                f"Unsupported driver: {driver}. "
                f"Supported: {', '.join(self.SUPPORTED_DRIVERS.keys())}"
            )

        driver_class = get_network_driver(driver)

        device = driver_class(
            hostname=hostname,
            username=username,
            password=password,
            optional_args=optional_args or {}
        )

        logger.info(f"Connecting to {hostname} ({self.SUPPORTED_DRIVERS[driver]})...")
        device.open()
        logger.info("Connected successfully")

        return device

    def get_device_facts(self, device) -> Dict[str, Any]:
        """
        Retrieve device facts (model, serial, OS version, etc.).

        Args:
            device: NAPALM device object

        Returns:
            Device facts dictionary
        """
        return device.get_facts()

    def get_interfaces(self, device) -> Dict[str, Any]:
        """
        Retrieve interface information.

        Args:
            device: NAPALM device object

        Returns:
            Interfaces dictionary
        """
        return device.get_interfaces()

    def get_config(self, device) -> Dict[str, str]:
        """
        Retrieve device configuration.

        Args:
            device: NAPALM device object

        Returns:
            Configuration dictionary (running, startup, candidate)
        """
        return device.get_config()

    def analyze_device(
        self,
        hostname: str,
        username: str,
        password: str,
        driver: str,
        analysis_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Connect to device and perform AI-powered analysis.

        Args:
            hostname: Device IP or hostname
            username: SSH username
            password: SSH password
            driver: NAPALM driver
            analysis_func: Optional custom analysis function

        Returns:
            Comprehensive analysis results
        """
        try:
            device = self.connect_device(hostname, username, password, driver)

            # Gather data
            facts = self.get_device_facts(device)
            interfaces = self.get_interfaces(device)
            config = self.get_config(device)

            # Close connection
            device.close()

            # Analyze with AI (or default analysis)
            if analysis_func:
                analysis = analysis_func(facts, interfaces, config)
            else:
                analysis = self._default_analysis(facts, interfaces, config)

            return {
                "hostname": hostname,
                "facts": facts,
                "interface_summary": self._summarize_interfaces(interfaces),
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}

    def _default_analysis(
        self,
        facts: Dict[str, Any],
        interfaces: Dict[str, Any],
        config: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Default analysis (without AI).

        Args:
            facts: Device facts
            interfaces: Interface data
            config: Configuration

        Returns:
            Analysis results
        """
        # Count interface states
        interface_states = {
            'up': 0,
            'down': 0,
            'disabled': 0
        }

        for iface_name, iface_data in interfaces.items():
            if iface_data.get('is_enabled'):
                if iface_data.get('is_up'):
                    interface_states['up'] += 1
                else:
                    interface_states['down'] += 1
            else:
                interface_states['disabled'] += 1

        return {
            "device_model": facts.get('model', 'Unknown'),
            "os_version": facts.get('os_version', 'Unknown'),
            "serial_number": facts.get('serial_number', 'Unknown'),
            "uptime": facts.get('uptime', 0),
            "interface_states": interface_states,
            "total_interfaces": len(interfaces),
            "config_size": len(config.get('running', ''))
        }

    def _summarize_interfaces(self, interfaces: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create interface summary.

        Args:
            interfaces: Interface data

        Returns:
            List of interface summaries
        """
        summary = []

        for iface_name, iface_data in interfaces.items():
            summary.append({
                "name": iface_name,
                "enabled": iface_data.get('is_enabled', False),
                "up": iface_data.get('is_up', False),
                "speed": iface_data.get('speed', 0),
                "description": iface_data.get('description', '')
            })

        return summary

    def compare_configs(
        self,
        device,
        candidate_config: str
    ) -> Dict[str, Any]:
        """
        Load and compare candidate configuration.

        Args:
            device: NAPALM device object
            candidate_config: Proposed configuration

        Returns:
            Configuration diff
        """
        # Load candidate config
        device.load_merge_candidate(config=candidate_config)

        # Get diff
        diff = device.compare_config()

        return {
            "diff": diff,
            "has_changes": len(diff) > 0
        }

    def validate_change(
        self,
        device,
        candidate_config: str,
        validation_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Validate configuration change using AI.

        Args:
            device: NAPALM device object
            candidate_config: Proposed configuration
            validation_func: Optional AI validation function

        Returns:
            Validation results
        """
        # Get diff
        comparison = self.compare_configs(device, candidate_config)

        if not comparison['has_changes']:
            return {
                "valid": True,
                "message": "No changes detected",
                "risk_level": "none"
            }

        # AI validation (or default)
        if validation_func:
            validation = validation_func(comparison['diff'])
        else:
            validation = self._default_validation(comparison['diff'])

        return validation

    def _default_validation(self, diff: str) -> Dict[str, Any]:
        """
        Default validation logic (without AI).

        Args:
            diff: Configuration diff

        Returns:
            Validation results
        """
        diff_lower = diff.lower()

        # Simple risk assessment
        risk_level = "low"
        warnings = []

        # Check for risky changes
        if any(word in diff_lower for word in ['shutdown', 'no ip address']):
            risk_level = "high"
            warnings.append("Change includes interface shutdown or IP removal")

        if 'no router' in diff_lower or 'no ip route' in diff_lower:
            risk_level = "high"
            warnings.append("Change removes routing configuration")

        if 'access-list' in diff_lower or 'acl' in diff_lower:
            risk_level = "medium"
            warnings.append("Change modifies ACLs")

        return {
            "valid": True,
            "risk_level": risk_level,
            "warnings": warnings,
            "recommendation": "Review changes carefully before committing"
        }

    def safe_deploy(
        self,
        hostname: str,
        username: str,
        password: str,
        driver: str,
        candidate_config: str,
        auto_commit: bool = False,
        rollback_timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Safely deploy configuration with rollback capability.

        Args:
            hostname: Device IP or hostname
            username: SSH username
            password: SSH password
            driver: NAPALM driver
            candidate_config: Configuration to deploy
            auto_commit: Automatically commit if validation passes
            rollback_timeout: Timeout for automatic rollback (seconds)

        Returns:
            Deployment results
        """
        try:
            device = self.connect_device(hostname, username, password, driver)

            # Validate change
            validation = self.validate_change(device, candidate_config)

            if validation['risk_level'] == 'high' and not auto_commit:
                device.discard_config()
                device.close()
                return {
                    "success": False,
                    "message": "High-risk change detected. Manual approval required.",
                    "validation": validation
                }

            # Get diff for logging
            diff = device.compare_config()

            if auto_commit:
                # Commit with rollback capability (if supported)
                if driver in ['junos', 'iosxr_netconf']:
                    device.commit_config(revert_in=rollback_timeout)
                else:
                    device.commit_config()

                logger.info("Configuration committed successfully")

                result = {
                    "success": True,
                    "message": "Configuration deployed successfully",
                    "diff": diff,
                    "validation": validation
                }
            else:
                # Discard candidate (dry-run mode)
                device.discard_config()

                result = {
                    "success": True,
                    "message": "Configuration validated (dry-run mode)",
                    "diff": diff,
                    "validation": validation,
                    "note": "Changes not committed. Set auto_commit=True to apply."
                }

            device.close()
            return result

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    NAPALM + AI Integration Demo
    ========================================
    Multi-vendor network automation with AI
    ========================================
    """)

    if not NAPALM_AVAILABLE:
        print("\n⚠️  NAPALM not installed!")
        print("Install with: pip install napalm")
        print("\nThis demo shows the code structure.")
        print("To test with real devices, install NAPALM and configure device access.\n")

    print("\n💡 Example Usage:")
    print("-" * 60)

    example_code = '''
from napalm_integration import AINetworkManager

# Initialize manager
manager = AINetworkManager()

# Example 1: Analyze device
analysis = manager.analyze_device(
    hostname="192.168.1.1",
    username="admin",
    password="password",
    driver="ios"  # Cisco IOS
)

print(f"Device: {analysis['facts']['hostname']}")
print(f"Model: {analysis['facts']['model']}")
print(f"Interfaces: {analysis['interface_summary']}")

# Example 2: Validate configuration change
device = manager.connect_device(
    hostname="192.168.1.1",
    username="admin",
    password="password",
    driver="ios"
)

candidate_config = """
interface GigabitEthernet0/1
 description Updated by AI
 no shutdown
"""

validation = manager.validate_change(device, candidate_config)

if validation['risk_level'] == 'low':
    print("✓ Safe to deploy")
else:
    print(f"⚠ Risk level: {validation['risk_level']}")

device.close()

# Example 3: Safe deployment with rollback
result = manager.safe_deploy(
    hostname="192.168.1.1",
    username="admin",
    password="password",
    driver="ios",
    candidate_config=candidate_config,
    auto_commit=False,  # Dry-run mode
    rollback_timeout=300
)

print(f"Status: {result['message']}")
'''

    print(example_code)

    print("\n💡 Supported Platforms:")
    print("-" * 60)
    for driver, name in AINetworkManager.SUPPORTED_DRIVERS.items():
        print(f"  - {name:25s} (driver: {driver})")

    print("\n💡 AI Integration Ideas:")
    print("-" * 60)
    ai_integration = '''
from resilient_api_client import ResilientAPIClient

def ai_config_validator(diff: str) -> Dict[str, Any]:
    """Validate configuration changes with AI."""
    client = ResilientAPIClient()

    prompt = f"""
    Analyze this network configuration change:

    {diff}

    Assess the risk level (low/medium/high) and identify:
    1. Potential service disruptions
    2. Security implications
    3. Best practice violations
    4. Rollback plan

    Return JSON with: risk_level, findings, recommendations
    """

    result = client.call(prompt, temperature=0)
    return json.loads(result['text'])

# Use AI validation
manager = AINetworkManager()
device = manager.connect_device(...)

validation = manager.validate_change(
    device,
    candidate_config,
    validation_func=ai_config_validator
)
'''

    print(ai_integration)

    print("\n✅ Demo complete!")
    print("\n💡 Key Capabilities:")
    print("  - Multi-vendor support (Cisco, Juniper, Arista)")
    print("  - Unified API across platforms")
    print("  - Configuration validation")
    print("  - Safe deployment with rollback")
    print("  - AI-powered risk assessment")

    print("\n⚠️  Production Checklist:")
    print("  ☐ Test on lab devices first")
    print("  ☐ Always review diffs before committing")
    print("  ☐ Use rollback capability when available")
    print("  ☐ Log all configuration changes")
    print("  ☐ Implement change approval workflow")
    print("  ☐ Monitor device health after changes")
