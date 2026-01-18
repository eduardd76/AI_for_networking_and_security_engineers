#!/usr/bin/env python3
"""
Ansible Integration - AI-Powered Playbook Generation and Execution

Generate Ansible playbooks with AI and execute them safely with validation.

From: AI for Networking Engineers - Volume 1, Chapter 10
Author: Eduard Dulharu (Ed Harmoosh)

Usage:
    from ansible_integration import AIAnsibleManager

    manager = AIAnsibleManager()
    playbook = manager.generate_playbook("Configure VLANs 10-20 on all switches")
    result = manager.execute_playbook(playbook, check_mode=True)
"""

import json
import yaml
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import logging

logger = logging.getLogger(__name__)


class AIAnsibleManager:
    """
    AI-powered Ansible playbook management.

    Features:
    - Generate playbooks from natural language
    - Validate playbook syntax
    - Execute with safety checks
    - Parse and analyze results
    """

    def __init__(self, inventory_file: Optional[str] = None):
        """
        Initialize Ansible manager.

        Args:
            inventory_file: Path to Ansible inventory file
        """
        self.inventory_file = inventory_file or "hosts"
        self._check_ansible_installed()

    def _check_ansible_installed(self) -> bool:
        """Check if Ansible is installed."""
        try:
            result = subprocess.run(
                ['ansible', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("Ansible is installed")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.warning("Ansible not found. Install with: pip install ansible")
        return False

    def generate_playbook(
        self,
        task_description: str,
        device_group: str = "all",
        ai_generator: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate Ansible playbook from natural language description.

        Args:
            task_description: Natural language task description
            device_group: Target device group from inventory
            ai_generator: Optional AI generation function

        Returns:
            Playbook dictionary
        """
        if ai_generator:
            return ai_generator(task_description, device_group)
        else:
            return self._mock_generate_playbook(task_description, device_group)

    def _mock_generate_playbook(
        self,
        task_description: str,
        device_group: str
    ) -> Dict[str, Any]:
        """
        Mock playbook generation (replace with AI in production).

        Args:
            task_description: Task description
            device_group: Target device group

        Returns:
            Playbook dictionary
        """
        # Simple mock based on keywords
        task_lower = task_description.lower()

        tasks = []

        if 'vlan' in task_lower:
            tasks.append({
                "name": "Configure VLANs",
                "cisco.ios.ios_vlan": {
                    "vlan_id": "{{ item }}",
                    "name": "VLAN_{{ item }}",
                    "state": "present"
                },
                "loop": list(range(10, 21))
            })

        if 'interface' in task_lower:
            tasks.append({
                "name": "Configure interfaces",
                "cisco.ios.ios_interface": {
                    "name": "GigabitEthernet0/1",
                    "description": "Configured by AI",
                    "enabled": True
                }
            })

        if 'backup' in task_lower or 'config' in task_lower:
            tasks.append({
                "name": "Backup configuration",
                "cisco.ios.ios_command": {
                    "commands": ["show running-config"]
                },
                "register": "config_output"
            })
            tasks.append({
                "name": "Save backup",
                "copy": {
                    "content": "{{ config_output.stdout[0] }}",
                    "dest": "/backups/{{ inventory_hostname }}.cfg"
                }
            })

        if not tasks:
            # Default task
            tasks.append({
                "name": "Gather facts",
                "cisco.ios.ios_facts": {
                    "gather_subset": "all"
                }
            })

        playbook = [{
            "name": f"AI Generated: {task_description}",
            "hosts": device_group,
            "gather_facts": False,
            "tasks": tasks
        }]

        return {"playbook": playbook}

    def validate_playbook(self, playbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate playbook syntax and structure.

        Args:
            playbook: Playbook dictionary

        Returns:
            Validation results
        """
        errors = []
        warnings = []

        # Check required fields
        if 'playbook' not in playbook:
            errors.append("Missing 'playbook' key")
            return {"valid": False, "errors": errors, "warnings": warnings}

        plays = playbook['playbook']
        if not isinstance(plays, list) or len(plays) == 0:
            errors.append("Playbook must contain at least one play")

        for i, play in enumerate(plays):
            # Check required play fields
            if 'name' not in play:
                warnings.append(f"Play {i} missing 'name' field")

            if 'hosts' not in play:
                errors.append(f"Play {i} missing 'hosts' field")

            if 'tasks' not in play:
                errors.append(f"Play {i} missing 'tasks' field")
            elif not isinstance(play['tasks'], list):
                errors.append(f"Play {i} 'tasks' must be a list")
            elif len(play['tasks']) == 0:
                warnings.append(f"Play {i} has no tasks")

            # Check tasks
            if 'tasks' in play:
                for j, task in enumerate(play['tasks']):
                    if 'name' not in task:
                        warnings.append(f"Play {i}, Task {j} missing 'name'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def save_playbook(
        self,
        playbook: Dict[str, Any],
        filename: str = "playbook.yml"
    ) -> Path:
        """
        Save playbook to YAML file.

        Args:
            playbook: Playbook dictionary
            filename: Output filename

        Returns:
            Path to saved playbook
        """
        playbook_path = Path(filename)

        with open(playbook_path, 'w') as f:
            yaml.dump(
                playbook['playbook'],
                f,
                default_flow_style=False,
                sort_keys=False
            )

        logger.info(f"Playbook saved to {playbook_path}")
        return playbook_path

    def execute_playbook(
        self,
        playbook: Dict[str, Any],
        check_mode: bool = True,
        inventory: Optional[str] = None,
        extra_vars: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Ansible playbook.

        Args:
            playbook: Playbook dictionary
            check_mode: Run in check mode (dry-run)
            inventory: Inventory file path
            extra_vars: Extra variables to pass

        Returns:
            Execution results
        """
        # Validate playbook first
        validation = self.validate_playbook(playbook)
        if not validation['valid']:
            return {
                "success": False,
                "message": "Playbook validation failed",
                "validation": validation
            }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yml',
            delete=False
        ) as f:
            yaml.dump(playbook['playbook'], f)
            playbook_file = f.name

        try:
            # Build ansible-playbook command
            cmd = [
                'ansible-playbook',
                playbook_file,
                '-i', inventory or self.inventory_file
            ]

            if check_mode:
                cmd.append('--check')

            if extra_vars:
                cmd.extend(['--extra-vars', json.dumps(extra_vars)])

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse output
            return self._parse_ansible_output(result)

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": "Playbook execution timed out"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "message": "Ansible not installed"
            }
        finally:
            # Clean up temp file
            Path(playbook_file).unlink(missing_ok=True)

    def _parse_ansible_output(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """
        Parse Ansible playbook output.

        Args:
            result: subprocess result

        Returns:
            Parsed execution results
        """
        stdout = result.stdout
        stderr = result.stderr

        # Parse task results from output
        ok_count = stdout.count("ok=")
        changed_count = stdout.count("changed=")
        failed_count = stdout.count("failed=")
        skipped_count = stdout.count("skipped=")

        success = result.returncode == 0

        return {
            "success": success,
            "return_code": result.returncode,
            "summary": {
                "ok": ok_count,
                "changed": changed_count,
                "failed": failed_count,
                "skipped": skipped_count
            },
            "stdout": stdout,
            "stderr": stderr
        }

    def ai_analyze_results(
        self,
        execution_result: Dict[str, Any],
        ai_analyzer: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Analyze playbook execution results with AI.

        Args:
            execution_result: Execution results
            ai_analyzer: Optional AI analysis function

        Returns:
            Analysis results
        """
        if ai_analyzer:
            return ai_analyzer(execution_result)
        else:
            return self._default_analyze_results(execution_result)

    def _default_analyze_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default result analysis (without AI).

        Args:
            execution_result: Execution results

        Returns:
            Analysis
        """
        summary = execution_result.get('summary', {})

        status = "success" if execution_result.get('success') else "failed"
        recommendations = []

        if summary.get('failed', 0) > 0:
            recommendations.append("Review failed tasks and check device connectivity")

        if summary.get('changed', 0) == 0 and summary.get('ok', 0) > 0:
            recommendations.append("No changes made - devices may already be in desired state")

        return {
            "status": status,
            "recommendations": recommendations,
            "risk_assessment": "low" if status == "success" else "high"
        }

    def create_inventory(
        self,
        devices: List[Dict[str, str]],
        groups: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Create Ansible inventory from device list.

        Args:
            devices: List of device dictionaries
            groups: Optional device groups

        Returns:
            Inventory content as string
        """
        inventory_lines = []

        # Add ungrouped devices
        ungrouped = [d for d in devices if 'group' not in d]
        if ungrouped:
            for device in ungrouped:
                line = f"{device['hostname']} ansible_host={device['ip']}"
                if 'ansible_user' in device:
                    line += f" ansible_user={device['ansible_user']}"
                if 'ansible_network_os' in device:
                    line += f" ansible_network_os={device['ansible_network_os']}"
                inventory_lines.append(line)

        # Add groups
        if groups:
            for group_name, device_list in groups.items():
                inventory_lines.append(f"\n[{group_name}]")
                for device_name in device_list:
                    # Find device in devices list
                    device = next((d for d in devices if d['hostname'] == device_name), None)
                    if device:
                        line = f"{device['hostname']} ansible_host={device['ip']}"
                        if 'ansible_user' in device:
                            line += f" ansible_user={device['ansible_user']}"
                        if 'ansible_network_os' in device:
                            line += f" ansible_network_os={device['ansible_network_os']}"
                        inventory_lines.append(line)

        return '\n'.join(inventory_lines)


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    Ansible + AI Integration Demo
    ========================================
    AI-powered playbook generation
    ========================================
    """)

    manager = AIAnsibleManager()

    # Test 1: Generate playbook
    print("\nTest 1: Generate Playbook from Natural Language")
    print("-" * 60)

    task = "Configure VLANs 10-20 on all switches"
    playbook = manager.generate_playbook(task, device_group="switches")

    print(f"Task: {task}")
    print(f"Generated playbook with {len(playbook['playbook'][0]['tasks'])} tasks\n")

    # Display playbook
    print("Generated Playbook:")
    print(yaml.dump(playbook['playbook'], default_flow_style=False))

    # Test 2: Validate playbook
    print("\nTest 2: Validate Playbook")
    print("-" * 60)

    validation = manager.validate_playbook(playbook)
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # Test 3: Save playbook
    print("\n\nTest 3: Save Playbook")
    print("-" * 60)

    playbook_file = manager.save_playbook(playbook, "demo_playbook.yml")
    print(f"✓ Saved to: {playbook_file}")

    # Test 4: Create inventory
    print("\n\nTest 4: Create Inventory")
    print("-" * 60)

    devices = [
        {
            "hostname": "switch01",
            "ip": "192.168.1.10",
            "ansible_user": "admin",
            "ansible_network_os": "cisco.ios.ios"
        },
        {
            "hostname": "switch02",
            "ip": "192.168.1.11",
            "ansible_user": "admin",
            "ansible_network_os": "cisco.ios.ios"
        },
        {
            "hostname": "router01",
            "ip": "192.168.1.1",
            "ansible_user": "admin",
            "ansible_network_os": "cisco.ios.ios"
        }
    ]

    groups = {
        "switches": ["switch01", "switch02"],
        "routers": ["router01"]
    }

    inventory = manager.create_inventory(devices, groups)
    print("Generated Inventory:")
    print(inventory)

    # Clean up
    playbook_file.unlink()

    print("\n\n✅ Demo complete!")
    print("\n💡 AI Integration Example:")
    print("-" * 60)

    ai_example = '''
from resilient_api_client import ResilientAPIClient

def ai_playbook_generator(task_description: str, device_group: str) -> Dict:
    """Generate playbook with AI."""
    client = ResilientAPIClient()

    prompt = f"""
    Generate an Ansible playbook for this task:
    Task: {task_description}
    Target: {device_group}

    Return valid Ansible YAML playbook structure with:
    - Play name
    - Hosts (use "{device_group}")
    - Tasks with proper module names

    Use cisco.ios collection for Cisco devices.
    Include error handling and rollback tasks.
    """

    result = client.call(prompt, temperature=0)

    # Parse YAML from response
    playbook_yaml = yaml.safe_load(result['text'])

    return {"playbook": playbook_yaml}

# Use AI generator
manager = AIAnsibleManager()
playbook = manager.generate_playbook(
    "Configure NTP servers on all routers",
    device_group="routers",
    ai_generator=ai_playbook_generator
)

# Execute in check mode
result = manager.execute_playbook(playbook, check_mode=True)
print(f"Check mode result: {result['summary']}")
'''

    print(ai_example)

    print("\n💡 Key Capabilities:")
    print("  - Generate playbooks from natural language")
    print("  - Validate syntax and structure")
    print("  - Execute with check mode (dry-run)")
    print("  - Parse and analyze results")
    print("  - Create dynamic inventories")

    print("\n⚠️  Production Checklist:")
    print("  ☐ Always run check mode first")
    print("  ☐ Review generated playbooks before execution")
    print("  ☐ Use vault for sensitive variables")
    print("  ☐ Implement approval workflow")
    print("  ☐ Log all playbook executions")
    print("  ☐ Test on non-production devices first")
