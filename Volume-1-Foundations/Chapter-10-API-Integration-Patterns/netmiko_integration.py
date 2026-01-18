#!/usr/bin/env python3
"""
Netmiko Integration - AI-Powered Network Automation

Combine Netmiko (SSH automation) with AI analysis for intelligent network operations.

From: AI for Networking Engineers - Volume 1, Chapter 10
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from netmiko_integration import AINetworkDevice

    device = AINetworkDevice(
        host="192.168.1.1",
        username="admin",
        password="password"
    )

    analysis = device.analyze_config_with_ai()
"""

from typing import Dict, Any, Optional, List
from netmiko import ConnectHandler
import logging

logger = logging.getLogger(__name__)


class AINetworkDevice:
    """
    Network device with AI-powered analysis capabilities.

    Combines Netmiko for device interaction with AI for intelligent analysis.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        device_type: str = "cisco_ios",
        secret: Optional[str] = None
    ):
        """
        Initialize AI-powered network device.

        Args:
            host: Device IP or hostname
            username: SSH username
            password: SSH password
            device_type: Netmiko device type
            secret: Enable secret (if required)
        """
        self.device_params = {
            "device_type": device_type,
            "host": host,
            "username": username,
            "password": password,
        }

        if secret:
            self.device_params["secret"] = secret

        self.connection = None

    def connect(self) -> bool:
        """
        Connect to network device.

        Returns:
            True if successful
        """
        try:
            logger.info(f"Connecting to {self.device_params['host']}...")
            self.connection = ConnectHandler(**self.device_params)
            logger.info("Connected successfully")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from device."""
        if self.connection:
            self.connection.disconnect()
            logger.info("Disconnected")

    def get_config(self) -> str:
        """
        Retrieve running configuration.

        Returns:
            Running configuration text
        """
        if not self.connection:
            raise RuntimeError("Not connected to device")

        return self.connection.send_command("show running-config")

    def send_command(self, command: str) -> str:
        """
        Send command to device.

        Args:
            command: CLI command

        Returns:
            Command output
        """
        if not self.connection:
            raise RuntimeError("Not connected to device")

        return self.connection.send_command(command)

    def analyze_config_with_ai(
        self,
        analysis_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Retrieve config and analyze with AI.

        Args:
            analysis_func: Optional custom analysis function

        Returns:
            Analysis results
        """
        # Get config
        config = self.get_config()

        # Default analysis (mock - integrate with your AI client)
        if not analysis_func:
            analysis_func = self._default_analysis

        return analysis_func(config)

    def _default_analysis(self, config: str) -> Dict[str, Any]:
        """Default config analysis (mock)."""
        return {
            "hostname": self._extract_hostname(config),
            "interface_count": config.count('interface '),
            "has_acls": 'access-list' in config,
            "routing_protocols": self._detect_routing(config)
        }

    def _extract_hostname(self, config: str) -> Optional[str]:
        """Extract hostname from config."""
        import re
        match = re.search(r'hostname\s+(\S+)', config)
        return match.group(1) if match else None

    def _detect_routing(self, config: str) -> List[str]:
        """Detect routing protocols."""
        protocols = []
        if 'router ospf' in config:
            protocols.append('OSPF')
        if 'router bgp' in config:
            protocols.append('BGP')
        if 'router eigrp' in config:
            protocols.append('EIGRP')
        return protocols

    def intelligent_troubleshooting(
        self,
        symptoms: str,
        commands: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        AI-guided troubleshooting.

        Args:
            symptoms: Description of problem
            commands: Optional list of commands to run

        Returns:
            Troubleshooting results
        """
        if not self.connection:
            raise RuntimeError("Not connected to device")

        # Default troubleshooting commands
        if not commands:
            commands = [
                "show version",
                "show ip interface brief",
                "show ip route summary",
                "show logging | include ERROR|WARN"
            ]

        # Collect data
        data = {}
        for cmd in commands:
            try:
                output = self.send_command(cmd)
                data[cmd] = output
            except Exception as e:
                data[cmd] = f"Error: {e}"

        # AI analysis would go here
        # For now, return collected data
        return {
            "symptoms": symptoms,
            "commands_run": commands,
            "data_collected": data,
            "analysis": "AI analysis would be performed here"
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Netmiko + AI Integration Demo
    ========================================
    Note: This demo shows code structure.
    To test with real devices, uncomment
    the device connection code below.
    ========================================
    """)

    # Example device parameters (DO NOT USE IN PRODUCTION)
    # device_params = {
    #     "host": "192.168.1.1",
    #     "username": "admin",
    #     "password": "password",
    #     "device_type": "cisco_ios"
    # }

    print("\n💡 Example Usage:")
    print("-" * 60)

    example_code = '''
    from netmiko_integration import AINetworkDevice

    # Using context manager (recommended)
    with AINetworkDevice(host="192.168.1.1",
                         username="admin",
                         password="password") as device:

        # Get and analyze config
        analysis = device.analyze_config_with_ai()
        print(f"Hostname: {analysis['hostname']}")
        print(f"Interfaces: {analysis['interface_count']}")

        # Intelligent troubleshooting
        results = device.intelligent_troubleshooting(
            symptoms="High CPU usage",
            commands=["show processes cpu", "show processes memory"]
        )

    # Device automatically disconnects after 'with' block
    '''

    print(example_code)

    print("\n💡 Integration with AI Client:")
    print("-" * 60)

    integration_example = '''
    from netmiko_integration import AINetworkDevice
    from resilient_api_client import ResilientAPIClient

    ai_client = ResilientAPIClient()

    def ai_security_analysis(config: str) -> dict:
        """Analyze config with AI."""
        prompt = f"""
        Analyze this config for security issues:
        {config}

        Return JSON with findings.
        """

        result = ai_client.call(prompt, temperature=0)
        return json.loads(result['text'])

    # Use AI analysis with network device
    with AINetworkDevice(**device_params) as device:
        analysis = device.analyze_config_with_ai(
            analysis_func=ai_security_analysis
        )
        print(analysis)
    '''

    print(integration_example)

    print("\n✅ Demo complete!")
    print("\n💡 Key Capabilities:")
    print("  - Automated config retrieval")
    print("  - AI-powered analysis")
    print("  - Intelligent troubleshooting")
    print("  - Safe device interaction (context manager)")
    print("  - Multi-vendor support via Netmiko")

    print("\n⚠️  Production Checklist:")
    print("  ☐ Use secrets manager for credentials")
    print("  ☐ Implement connection pooling")
    print("  ☐ Add retry logic for failed connections")
    print("  ☐ Log all device interactions")
    print("  ☐ Validate commands before execution")
    print("  ☐ Implement change approval workflow")
