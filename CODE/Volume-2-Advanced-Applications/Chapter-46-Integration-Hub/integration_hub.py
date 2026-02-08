"""
Chapter 46: Integration Hub - Intent-to-Automation
Bridge AI, NetBox inventory, Ansible automation, and Grafana monitoring

This module demonstrates production integration patterns for AI-powered
network operations connecting multiple tools in automated workflows.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import time
import yaml
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class ExecutionStep(str, Enum):
    INTENT_ANALYSIS = "intent_analysis"
    NETBOX_QUERY = "netbox_query"
    PLAYBOOK_GENERATION = "playbook_generation"
    ANSIBLE_EXECUTION = "ansible_execution"
    RESULT_PROCESSING = "result_processing"
    NETBOX_UPDATE = "netbox_update"
    GRAFANA_ALERT = "grafana_alert"


@dataclass
class ExecutionLog:
    """Workflow execution log."""
    timestamp: str
    step: ExecutionStep
    source: str
    message: str
    status: str
    details: Optional[Dict] = None


@dataclass
class Device:
    """Network device from NetBox."""
    id: int
    name: str
    device_type: str
    role: str
    site: str
    status: str
    compliance_status: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict = field(default_factory=dict)


class IntentAnalysis(BaseModel):
    """Structured intent parsing from LLM."""
    action: str = Field(description="audit, deploy, remediate, or configure")
    target_role: str = Field(description="edge, core, or access")
    check_type: str = Field(description="compliance, security, or performance")
    specific_checks: List[str] = Field(description="List of specific checks to perform")


class IntegrationHub:
    """
    AI-powered integration hub.

    Workflow: Intent (NLP) → NetBox (inventory) → Ansible (execution) → Grafana (monitoring)

    Features:
    - Natural language intent parsing
    - Dynamic NetBox inventory queries
    - Automated Ansible playbook generation
    - Result correlation and reporting
    - Grafana alert integration
    """

    def __init__(self):
        self.execution_logs: List[ExecutionLog] = []
        self.generated_playbooks: Dict[str, str] = {}

        # Simulated NetBox devices
        self.devices_db = self._init_devices()

    def _init_devices(self) -> List[Device]:
        """Initialize simulated NetBox inventory."""
        return [
            Device(
                id=1,
                name="edge-rtr-01",
                device_type="Cisco ISR 4331",
                role="edge",
                site="HQ",
                status="active",
                custom_fields={'last_audit': None, 'compliance_status': None}
            ),
            Device(
                id=2,
                name="edge-rtr-02",
                device_type="Cisco ISR 4331",
                role="edge",
                site="HQ",
                status="active",
                custom_fields={'last_audit': None, 'compliance_status': None}
            ),
            Device(
                id=3,
                name="core-sw-01",
                device_type="Cisco Catalyst 9500",
                role="core",
                site="HQ",
                status="active",
                custom_fields={'last_audit': None, 'compliance_status': None}
            ),
            Device(
                id=4,
                name="access-sw-01",
                device_type="Cisco Catalyst 9300",
                role="access",
                site="Branch-A",
                status="active",
                custom_fields={'last_audit': None, 'compliance_status': None}
            ),
        ]

    def log_step(
        self,
        step: ExecutionStep,
        source: str,
        message: str,
        status: str = "success",
        details: Optional[Dict] = None
    ):
        """Log workflow execution step."""
        log = ExecutionLog(
            timestamp=datetime.now().isoformat(),
            step=step,
            source=source,
            message=message,
            status=status,
            details=details
        )

        self.execution_logs.append(log)

    def parse_intent(self, intent: str) -> Dict:
        """
        Parse natural language intent.

        In production: Use LLM for robust parsing.
        """
        intent_lower = intent.lower()

        parsed = {
            'action': None,
            'target_role': None,
            'check_type': 'compliance',
            'specific_checks': []
        }

        # Action
        if 'audit' in intent_lower or 'check' in intent_lower:
            parsed['action'] = 'audit'
        elif 'deploy' in intent_lower or 'configure' in intent_lower:
            parsed['action'] = 'deploy'
        elif 'remediate' in intent_lower or 'fix' in intent_lower:
            parsed['action'] = 'remediate'

        # Target
        if 'edge' in intent_lower:
            parsed['target_role'] = 'edge'
        elif 'core' in intent_lower:
            parsed['target_role'] = 'core'
        elif 'access' in intent_lower:
            parsed['target_role'] = 'access'

        # Check type
        if 'compliance' in intent_lower:
            parsed['check_type'] = 'compliance'
        elif 'security' in intent_lower:
            parsed['check_type'] = 'security'

        # Specific checks
        if 'bgp' in intent_lower:
            parsed['specific_checks'].append('bgp_status')
        if 'interface' in intent_lower:
            parsed['specific_checks'].append('interface_status')

        return parsed

    def query_netbox(self, filters: Dict) -> List[Device]:
        """
        Query NetBox inventory.

        In production: Use requests to call real NetBox API.
        """
        self.log_step(
            ExecutionStep.NETBOX_QUERY,
            'AI',
            f"Querying NetBox for role='{filters.get('target_role', 'all')}'",
            status='running'
        )

        # Filter devices
        devices = self.devices_db

        if filters.get('target_role'):
            devices = [d for d in devices if d.role == filters['target_role']]

        self.log_step(
            ExecutionStep.NETBOX_QUERY,
            'NetBox',
            f"Returned {len(devices)} devices",
            status='success',
            details={'devices': [d.name for d in devices]}
        )

        return devices

    def generate_playbook(
        self,
        action: str,
        devices: List[Device],
        intent_details: Dict
    ) -> str:
        """
        Generate Ansible playbook from intent.

        In production: Use LLM to generate playbooks.
        """
        self.log_step(
            ExecutionStep.PLAYBOOK_GENERATION,
            'AI',
            f"Generating {action} playbook for {len(devices)} devices",
            status='running'
        )

        if action == 'audit':
            playbook = self._generate_audit_playbook(devices, intent_details)
        elif action == 'deploy':
            playbook = self._generate_deployment_playbook(devices, intent_details)
        else:
            playbook = self._generate_generic_playbook(devices)

        playbook_yaml = yaml.dump(playbook, default_flow_style=False, sort_keys=False)

        self.log_step(
            ExecutionStep.PLAYBOOK_GENERATION,
            'AI',
            "Playbook generated and validated",
            status='success',
            details={'lines': len(playbook_yaml.split('\n'))}
        )

        return playbook_yaml

    def _generate_audit_playbook(self, devices: List[Device], details: Dict) -> List[Dict]:
        """Generate compliance audit playbook."""
        host_group = f"{details.get('target_role', 'all')}_routers"

        tasks = [
            {
                'name': 'Check BGP State',
                'cisco.ios.ios_command': {
                    'commands': ['show ip bgp summary']
                },
                'register': 'bgp_output'
            },
            {
                'name': 'Check Interface Status',
                'cisco.ios.ios_command': {
                    'commands': ['show ip interface brief']
                },
                'register': 'interface_output'
            },
            {
                'name': 'Validate BGP Compliance',
                'assert': {
                    'that': [
                        '"Established" in bgp_output.stdout[0]'
                    ],
                    'fail_msg': 'BGP peer is not established'
                }
            },
            {
                'name': 'Save Results',
                'set_fact': {
                    'audit_result': {
                        'device': '{{ inventory_hostname }}',
                        'bgp_status': 'UP',
                        'timestamp': '{{ ansible_date_time.iso8601 }}'
                    }
                }
            }
        ]

        return [{
            'name': f'Audit {host_group}',
            'hosts': host_group,
            'gather_facts': False,
            'tasks': tasks
        }]

    def _generate_deployment_playbook(self, devices: List[Device], details: Dict) -> List[Dict]:
        """Generate configuration deployment playbook."""
        tasks = [
            {
                'name': 'Backup Current Config',
                'cisco.ios.ios_command': {
                    'commands': ['show running-config']
                },
                'register': 'config_backup'
            },
            {
                'name': 'Deploy Configuration',
                'cisco.ios.ios_config': {
                    'lines': details.get('config_lines', [
                        'logging buffered 8192',
                        'no logging console'
                    ]),
                    'backup': True
                }
            },
            {
                'name': 'Verify Configuration',
                'cisco.ios.ios_command': {
                    'commands': ['show running-config | include logging']
                },
                'register': 'verify_config'
            }
        ]

        return [{
            'name': 'Deploy Network Configuration',
            'hosts': 'all',
            'gather_facts': False,
            'tasks': tasks
        }]

    def _generate_generic_playbook(self, devices: List[Device]) -> List[Dict]:
        """Generate information-gathering playbook."""
        return [{
            'name': 'Gather Network Information',
            'hosts': 'all',
            'gather_facts': True,
            'tasks': [
                {
                    'name': 'Collect Device Info',
                    'cisco.ios.ios_command': {
                        'commands': [
                            'show version',
                            'show ip interface brief'
                        ]
                    },
                    'register': 'device_info'
                }
            ]
        }]

    def execute_playbook(self, playbook_yaml: str, devices: List[Device]) -> Dict[str, Dict]:
        """
        Execute Ansible playbook.

        In production: Use ansible-runner or AWX API.
        """
        self.log_step(
            ExecutionStep.ANSIBLE_EXECUTION,
            'Ansible',
            f"Executing playbook on {len(devices)} hosts",
            status='running'
        )

        results = {}

        for device in devices:
            # Simulate execution
            time.sleep(0.1)

            # Simulate one failure
            success = device.name != 'edge-rtr-02'
            status = 'OK' if success else 'FAILED'

            result_msg = f"{device.name}: {status}"
            if not success:
                result_msg += " - BGP peer down"

            self.log_step(
                ExecutionStep.ANSIBLE_EXECUTION,
                'Ansible',
                result_msg,
                status='success' if success else 'error'
            )

            results[device.name] = {
                'success': success,
                'stdout': result_msg,
                'changed': success
            }

        return results

    def update_netbox(self, devices: List[Device], results: Dict[str, Dict]):
        """Update NetBox with execution results."""
        self.log_step(
            ExecutionStep.NETBOX_UPDATE,
            'AI',
            "Updating NetBox custom fields",
            status='running'
        )

        for device in devices:
            result = results.get(device.name, {})
            device.custom_fields['compliance_status'] = 'compliant' if result.get('success') else 'non-compliant'
            device.custom_fields['last_audit'] = datetime.now().isoformat()

        self.log_step(
            ExecutionStep.NETBOX_UPDATE,
            'NetBox',
            f"Updated {len(devices)} devices",
            status='success'
        )

    def create_grafana_alert(self, devices: List[Device], results: Dict[str, Dict]):
        """Create Grafana annotation for issues."""
        non_compliant = [d for d in devices if not results.get(d.name, {}).get('success')]

        if non_compliant:
            self.log_step(
                ExecutionStep.GRAFANA_ALERT,
                'Grafana',
                f'Alert: Compliance failed on {", ".join([d.name for d in non_compliant])}',
                status='success',
                details={'devices': [d.name for d in non_compliant]}
            )

    def run_workflow(self, intent: str) -> Dict:
        """
        Execute complete intent-to-action workflow.

        Args:
            intent: Natural language intent

        Returns:
            Workflow results
        """
        self.execution_logs.clear()

        # Step 1: Parse Intent
        self.log_step(
            ExecutionStep.INTENT_ANALYSIS,
            'AI',
            f'Analyzing: "{intent}"',
            status='success'
        )

        parsed = self.parse_intent(intent)

        # Step 2: Query NetBox
        devices = self.query_netbox(parsed)

        if not devices:
            return {
                'success': False,
                'message': 'No devices found matching criteria',
                'logs': self.execution_logs
            }

        # Step 3: Generate Playbook
        playbook_yaml = self.generate_playbook(parsed['action'], devices, parsed)
        self.generated_playbooks[intent] = playbook_yaml

        # Step 4: Execute
        execution_results = self.execute_playbook(playbook_yaml, devices)

        # Step 5: Update NetBox
        self.update_netbox(devices, execution_results)

        # Step 6: Grafana Alerts
        self.create_grafana_alert(devices, execution_results)

        return {
            'success': True,
            'devices_affected': [d.name for d in devices],
            'playbook': playbook_yaml,
            'results': execution_results,
            'logs': self.execution_logs
        }


def example_1_simple_intent():
    """
    Example 1: Simple intent-to-action
    """
    print("=" * 60)
    print("Example 1: Intent-to-Action Workflow")
    print("=" * 60)

    hub = IntegrationHub()

    intent = "Audit edge routers for compliance"

    print(f"\n🎯 Intent: \"{intent}\"\n")

    result = hub.run_workflow(intent)

    print("📋 Execution Logs:\n")

    for log in result['logs']:
        emoji = {
            'AI': '🤖',
            'NetBox': '📦',
            'Ansible': '🔧',
            'Grafana': '📊'
        }.get(log.source, '⚪')

        print(f"{emoji} [{log.step.value}] {log.message}")

    print(f"\n✅ Workflow completed")
    print(f"Devices affected: {', '.join(result['devices_affected'])}")

    print("\n" + "=" * 60 + "\n")


def example_2_view_generated_playbook():
    """
    Example 2: View generated Ansible playbook
    """
    print("=" * 60)
    print("Example 2: Generated Ansible Playbook")
    print("=" * 60)

    hub = IntegrationHub()

    intent = "Audit edge routers for BGP compliance"
    result = hub.run_workflow(intent)

    print(f"\n📝 Generated Playbook:\n")
    print(result['playbook'])

    print("=" * 60 + "\n")


def example_3_multi_device_audit():
    """
    Example 3: Multi-device compliance audit
    """
    print("=" * 60)
    print("Example 3: Multi-Device Compliance Audit")
    print("=" * 60)

    hub = IntegrationHub()

    intent = "Audit all edge routers for compliance"
    result = hub.run_workflow(intent)

    print(f"\n📊 Audit Results:\n")

    for device_name, exec_result in result['results'].items():
        status_emoji = "✅" if exec_result['success'] else "❌"
        compliance = "COMPLIANT" if exec_result['success'] else "NON-COMPLIANT"

        print(f"{status_emoji} {device_name}: {compliance}")

    # Show device status in NetBox
    print(f"\n📦 NetBox Status Update:\n")

    for device in hub.devices_db:
        if device.name in result['devices_affected']:
            print(f"  {device.name}:")
            print(f"    Compliance: {device.custom_fields.get('compliance_status')}")
            print(f"    Last Audit: {device.custom_fields.get('last_audit', 'Never')[:19]}")

    print("\n" + "=" * 60 + "\n")


def example_4_llm_intent_parsing():
    """
    Example 4: Use LLM for intent parsing
    """
    print("=" * 60)
    print("Example 4: LLM-Powered Intent Parsing")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping LLM example.")
        print("=" * 60 + "\n")
        return

    intent = "I need to check if all our edge routers have BGP neighbors in established state"

    prompt = f"""Parse this network operations intent into structured format:

"{intent}"

Identify:
- action: (audit, deploy, remediate, or configure)
- target_role: (edge, core, or access)
- check_type: (compliance, security, or performance)
- specific_checks: (list of specific checks)"""

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    parser = PydanticOutputParser(pydantic_object=IntentAnalysis)

    full_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

    print(f"\n🎯 Intent: \"{intent}\"\n")
    print("🤖 Parsing with Claude...\n")

    response = llm.invoke(full_prompt)
    parsed = parser.parse(response.content)

    print("Parsed Intent:")
    print(f"  Action: {parsed.action}")
    print(f"  Target Role: {parsed.target_role}")
    print(f"  Check Type: {parsed.check_type}")
    print(f"  Specific Checks: {', '.join(parsed.specific_checks)}")

    print("\n" + "=" * 60 + "\n")


def example_5_complete_workflow():
    """
    Example 5: Complete end-to-end workflow with all integrations
    """
    print("=" * 60)
    print("Example 5: Complete Integration Workflow")
    print("=" * 60)

    hub = IntegrationHub()

    intents = [
        "Audit edge routers for compliance",
        "Check BGP status on core switches",
        "Deploy logging configuration to all devices"
    ]

    print("\n🚀 Running multiple workflows:\n")

    for i, intent in enumerate(intents, 1):
        print(f"{i}. \"{intent}\"")

        result = hub.run_workflow(intent)

        success_count = sum(1 for r in result['results'].values() if r['success'])
        total_count = len(result['results'])

        print(f"   Results: {success_count}/{total_count} successful")
        print()

    # Show summary
    print("📊 NetBox Inventory Summary:")
    print("-" * 60)

    for device in hub.devices_db:
        status_emoji = {
            'compliant': '✅',
            'non-compliant': '❌',
            None: '⚪'
        }.get(device.custom_fields.get('compliance_status'), '⚪')

        print(f"{status_emoji} {device.name} ({device.role})")
        print(f"   Site: {device.site}")
        print(f"   Type: {device.device_type}")
        print(f"   Compliance: {device.custom_fields.get('compliance_status', 'Not audited')}")
        print()

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\n🔗 Chapter 46: Integration Hub")
    print("Intent-to-Automation with NetBox, Ansible, and Grafana\n")

    try:
        example_1_simple_intent()
        input("Press Enter to continue...")

        example_2_view_generated_playbook()
        input("Press Enter to continue...")

        example_3_multi_device_audit()
        input("Press Enter to continue...")

        example_4_llm_intent_parsing()
        input("Press Enter to continue...")

        example_5_complete_workflow()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Natural language intents drive automation")
        print("- NetBox provides dynamic inventory")
        print("- AI generates Ansible playbooks on-demand")
        print("- Results feed back to NetBox and Grafana")
        print("- Complete closed-loop automation workflow\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
