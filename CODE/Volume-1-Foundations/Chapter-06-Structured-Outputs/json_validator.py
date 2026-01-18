#!/usr/bin/env python3
"""
JSON Validation and Parsing for AI Outputs

Extract and validate JSON from LLM responses, handling common issues like
markdown code blocks, extra text, and malformed JSON.

From: AI for Networking Engineers - Volume 1, Chapter 6
Author: Eduard Dulharu

Usage:
    from json_validator import JSONValidator

    validator = JSONValidator()

    # Parse JSON from response (handles markdown blocks)
    data = validator.extract_json(response_text)

    # Validate against schema
    is_valid = validator.validate_schema(data, schema)
"""

import json
import re
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class JSONParsingError(Exception):
    """Raised when JSON cannot be extracted or parsed."""
    pass


class JSONValidator:
    """
    Extract and validate JSON from LLM responses.

    Handles common issues:
    - JSON wrapped in markdown code blocks
    - Extra text before/after JSON
    - Single quotes instead of double quotes
    - Trailing commas
    - Comments in JSON
    """

    def extract_json(
        self,
        text: str,
        strict: bool = False
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Extract JSON from text, handling various formats.

        Args:
            text: Text containing JSON (may include markdown, extra text)
            strict: If True, require valid JSON. If False, attempt repairs.

        Returns:
            Parsed JSON object (dict or list)

        Raises:
            JSONParsingError: If JSON cannot be extracted or parsed
        """
        if not text or not text.strip():
            raise JSONParsingError("Empty input text")

        # Try direct parsing first (fastest path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # Try other methods

        # Try extracting from markdown code block
        json_text = self._extract_from_markdown(text)
        if json_text:
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                if not strict:
                    # Try repairs
                    repaired = self._repair_json(json_text)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as e:
                        raise JSONParsingError(f"Failed to parse JSON: {e}")
                else:
                    raise JSONParsingError("Invalid JSON in markdown block")

        # Try finding JSON object in text
        json_text = self._extract_json_object(text)
        if json_text:
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                if not strict:
                    repaired = self._repair_json(json_text)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as e:
                        raise JSONParsingError(f"Failed to parse JSON: {e}")
                else:
                    raise JSONParsingError("Invalid JSON object found")

        raise JSONParsingError("No valid JSON found in text")

    def _extract_from_markdown(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks."""
        # Pattern: ```json ... ``` or ```{ ... }```
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(\{.*?\})\s*\n```',
            r'```\s*\n(\[.*?\])\s*\n```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return None

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract JSON object or array from text."""
        # Find outermost { } or [ ]

        # Try finding object
        start_obj = text.find('{')
        if start_obj != -1:
            depth = 0
            for i in range(start_obj, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start_obj:i+1]

        # Try finding array
        start_arr = text.find('[')
        if start_arr != -1:
            depth = 0
            for i in range(start_arr, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start_arr:i+1]

        return None

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON issues."""
        # Remove comments
        text = re.sub(r'//.*?\n', '\n', text)  # Single-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Multi-line comments

        # Replace single quotes with double quotes (risky but common issue)
        # Only do this if there are no double quotes
        if '"' not in text and "'" in text:
            text = text.replace("'", '"')

        # Remove trailing commas before } or ]
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)

        return text

    def validate_schema(
        self,
        data: Union[Dict[str, Any], List[Any]],
        schema: Dict[str, Any],
        strict: bool = True
    ) -> bool:
        """
        Validate JSON data against a simple schema.

        Args:
            data: JSON data to validate
            schema: Schema dict with required keys and types
            strict: If True, extra keys are not allowed

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails

        Example schema:
            {
                "hostname": str,
                "ip_address": str,
                "vlan_id": int,
                "enabled": bool,
                "interfaces": list
            }
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        # Check required keys
        for key, expected_type in schema.items():
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

            actual_value = data[key]

            # Check type
            if not isinstance(actual_value, expected_type):
                raise ValueError(
                    f"Key '{key}': expected {expected_type.__name__}, "
                    f"got {type(actual_value).__name__}"
                )

        # Check for extra keys if strict
        if strict:
            extra_keys = set(data.keys()) - set(schema.keys())
            if extra_keys:
                raise ValueError(f"Unexpected keys: {extra_keys}")

        return True

    def extract_field(
        self,
        data: Dict[str, Any],
        field_path: str,
        default: Any = None
    ) -> Any:
        """
        Extract nested field from JSON using dot notation.

        Args:
            data: JSON data
            field_path: Dot-separated path (e.g., "config.interfaces.0.name")
            default: Default value if field not found

        Returns:
            Field value or default

        Example:
            data = {"config": {"hostname": "R1"}}
            hostname = validator.extract_field(data, "config.hostname")
        """
        keys = field_path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    index = int(key)
                    current = current[index]
                except (ValueError, IndexError):
                    return default
            else:
                return default

            if current is None:
                return default

        return current


class NetworkConfigValidator(JSONValidator):
    """
    Specialized validator for network configuration JSON.

    Includes predefined schemas for common networking data structures.
    """

    # Common schemas
    INTERFACE_SCHEMA = {
        "name": str,
        "ip_address": str,
        "subnet_mask": str,
        "status": str
    }

    DEVICE_SCHEMA = {
        "hostname": str,
        "mgmt_ip": str,
        "device_type": str,
        "interfaces": list
    }

    SECURITY_FINDING_SCHEMA = {
        "issue": str,
        "severity": str,
        "line": str,
        "fix": str
    }

    def validate_interface(self, data: Dict[str, Any]) -> bool:
        """Validate interface configuration."""
        return self.validate_schema(data, self.INTERFACE_SCHEMA, strict=False)

    def validate_device(self, data: Dict[str, Any]) -> bool:
        """Validate device configuration."""
        return self.validate_schema(data, self.DEVICE_SCHEMA, strict=False)

    def validate_security_finding(self, data: Dict[str, Any]) -> bool:
        """Validate security finding."""
        return self.validate_schema(data, self.SECURITY_FINDING_SCHEMA, strict=False)

    def validate_ip_address(self, ip: str) -> bool:
        """
        Validate IPv4 address format.

        Args:
            ip: IP address string

        Returns:
            True if valid IPv4 address
        """
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, ip):
            return False

        # Check each octet is 0-255
        octets = ip.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)

    def validate_subnet_mask(self, mask: str) -> bool:
        """
        Validate subnet mask format.

        Args:
            mask: Subnet mask (255.255.255.0 or /24)

        Returns:
            True if valid
        """
        # CIDR notation
        if mask.startswith('/'):
            try:
                prefix = int(mask[1:])
                return 0 <= prefix <= 32
            except ValueError:
                return False

        # Dotted decimal
        if not self.validate_ip_address(mask):
            return False

        # Check it's a valid subnet mask (contiguous 1s)
        octets = [int(o) for o in mask.split('.')]
        binary = ''.join(f'{o:08b}' for o in octets)

        # Valid mask has all 1s followed by all 0s
        return '01' not in binary


# Example usage and testing
if __name__ == "__main__":
    print("""
    ========================================
    JSON Validator Demo
    ========================================
    Testing JSON extraction and validation
    ========================================
    """)

    validator = JSONValidator()
    network_validator = NetworkConfigValidator()

    # Test 1: Extract from markdown
    print("\nTest 1: Extract JSON from markdown block")
    print("-" * 60)

    response1 = """Here's the interface configuration:

```json
{
    "name": "GigabitEthernet0/1",
    "ip_address": "10.1.1.1",
    "subnet_mask": "255.255.255.0",
    "status": "up"
}
```

This configuration looks good."""

    try:
        data = validator.extract_json(response1)
        print(f"✓ Extracted: {json.dumps(data, indent=2)}")

        # Validate
        network_validator.validate_interface(data)
        print("✓ Validation passed")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Extract from plain text
    print("\n\nTest 2: Extract JSON from plain text")
    print("-" * 60)

    response2 = """The device configuration is {"hostname": "CORE-RTR-01", "mgmt_ip": "192.168.1.1", "device_type": "router", "interfaces": []} and it's working fine."""

    try:
        data = validator.extract_json(response2)
        print(f"✓ Extracted: {json.dumps(data, indent=2)}")

        network_validator.validate_device(data)
        print("✓ Validation passed")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Repair malformed JSON
    print("\n\nTest 3: Repair malformed JSON (trailing comma)")
    print("-" * 60)

    malformed = """{
    "hostname": "R1",
    "ip": "10.1.1.1",
}"""

    try:
        data = validator.extract_json(malformed, strict=False)
        print(f"✓ Repaired and extracted: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Field extraction
    print("\n\nTest 4: Extract nested fields")
    print("-" * 60)

    nested_data = {
        "device": {
            "hostname": "CORE-SW-01",
            "interfaces": [
                {"name": "Gi0/1", "vlan": 10},
                {"name": "Gi0/2", "vlan": 20}
            ]
        }
    }

    hostname = validator.extract_field(nested_data, "device.hostname")
    first_vlan = validator.extract_field(nested_data, "device.interfaces.0.vlan")
    missing = validator.extract_field(nested_data, "device.missing", default="N/A")

    print(f"Hostname: {hostname}")
    print(f"First interface VLAN: {first_vlan}")
    print(f"Missing field: {missing}")

    # Test 5: IP validation
    print("\n\nTest 5: IP address validation")
    print("-" * 60)

    test_ips = [
        ("10.1.1.1", True),
        ("192.168.0.255", True),
        ("256.1.1.1", False),
        ("10.1.1", False),
        ("not-an-ip", False)
    ]

    for ip, expected in test_ips:
        result = network_validator.validate_ip_address(ip)
        status = "✓" if result == expected else "✗"
        print(f"{status} {ip:20s} → {result}")

    # Test 6: Subnet mask validation
    print("\n\nTest 6: Subnet mask validation")
    print("-" * 60)

    test_masks = [
        ("255.255.255.0", True),
        ("/24", True),
        ("/33", False),
        ("255.255.0.255", False),  # Invalid (not contiguous)
        ("255.255.255.128", True)
    ]

    for mask, expected in test_masks:
        result = network_validator.validate_subnet_mask(mask)
        status = "✓" if result == expected else "✗"
        print(f"{status} {mask:20s} → {result}")

    print("\n✅ Demo complete!")
