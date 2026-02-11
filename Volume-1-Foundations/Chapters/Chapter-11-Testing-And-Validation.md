# Chapter 11: Testing and Validation

## Learning Objectives

By the end of this chapter, you will:
- Write unit tests for prompts and AI outputs
- Build regression test suites for model changes
- Generate synthetic test data for networking scenarios
- Implement CI/CD pipelines for AI systems
- Measure and improve AI system quality

**Prerequisites**: Chapters 1-10 completed, basic pytest knowledge.

**What You'll Build**: A complete testing framework for AI networking systems—from unit tests to integration tests to CI/CD pipelines—ensuring your AI systems work reliably in production.

---

## The Production Incident That Taught Me to Test

It was a Wednesday when things got interesting.

Our AI config auditor had been running in production for two weeks—flagging security issues, generating remediation tickets, looking smart. Then our security team asked: "Why didn't the auditor catch this?" pointing to a router with telnet enabled.

I pulled the logs. The auditor had analyzed that config. It had returned... nothing. No findings. The config clearly had `transport input telnet` on the VTY lines.

I ran the exact same config through my local test script. **It worked fine.** Telnet detected, flagged as high severity.

Three hours of debugging later, I found it:

```python
# Production code (deployed 2 weeks ago)
prompt = "Analyze for security issues: {config}"

# Test code (my local copy, modified yesterday)
prompt = "Analyze this Cisco IOS config for security issues: {config}"
```

Someone had pushed a "minor" prompt tweak to production. The vague prompt worked 90% of the time, but edge cases slipped through. My local tests passed because I'd already updated the prompt.

**We had no automated tests that would have caught this.**

That incident cost us a week of re-auditing 400 routers. It also taught me that AI systems need tests just as much as traditional software—maybe more.

---

## The Testing Pyramid for AI Systems

Traditional testing pyramid:

```
        ▲
       /E2E\      (Few, slow, expensive)
      /─────\
     / Integ \    (Some, moderate)
    /─────────\
   /   Unit    \  (Many, fast, cheap)
  ─────────────
```

AI testing pyramid (adapted):

```
        ▲
     /Prod Mon\   (Continuous, real data)
    /───────────\
   / Regression  \  (Catch model/prompt drift)
  /───────────────\
 /  Prompt Tests   \  (Test inputs → expected outputs)
────────────────────
```

| Level | What It Tests | When It Runs |
|-------|---------------|--------------|
| Prompt Tests | Does this prompt produce expected outputs? | Every commit |
| Regression Tests | Did quality change vs baseline? | Daily, on model updates |
| Production Monitoring | Is the system working with real data? | Continuous |

---

## The Problem: "It Works on My Machine"

You built a config analyzer. It works perfectly in testing:

```python
# Your test
config = "snmp-server community public RO"
result = analyze_config(config)
assert "weak SNMP" in result
# PASS
```

Then production happens:
- **Day 1**: Works fine
- **Day 2**: Claude API updated, outputs changed format
- **Day 3**: New engineer modifies prompt, breaks parsing
- **Day 4**: Edge case config causes crash
- **Day 5**: Boss asks: "How do we know this actually works?"

**The reality**: Untested AI systems fail in creative ways.

---

### Check Your Understanding: Testing Challenges

Before diving into solutions, verify you understand the core challenges:

<details>
<summary><strong>Question 1:</strong> Why can't you test AI systems the same way you test traditional code?</summary>

**Answer**: AI systems are non-deterministic. The same prompt can produce different outputs (though `temperature=0` helps). Traditional tests assert exact equality (`assert result == expected`), but AI tests need probabilistic checks (`assert keyword in result`).

**Example**:
- Traditional: `assert add(2,3) == 5` (always true)
- AI: `assert "error" in classify_log("ERROR occurred")` (probably true, but LLM might say "CRITICAL", "severe", etc.)

</details>

<details>
<summary><strong>Question 2:</strong> What are the three main failure modes for AI systems in production?</summary>

**Answer**:
1. **Model updates**: API provider updates model, behavior changes
2. **Prompt drift**: Engineers modify prompts, break existing functionality
3. **Edge cases**: Unexpected inputs cause crashes or incorrect outputs

**Real example from chapter**: Vague prompt worked 90% of time, but missed telnet detection in edge cases.

</details>

<details>
<summary><strong>Question 3:</strong> What would happen if you deployed the config auditor from the opening story without tests?</summary>

**Answer**: You'd discover issues in production when users report them. The opening story showed:
- Telnet went undetected on 400 routers
- Required week-long re-audit
- Lost trust from security team
- No way to know if future changes break things

**Key insight**: Testing AI systems BEFORE production saves time and reputation.

</details>

---

## Testing AI Systems is Different

### Traditional Software vs AI Systems

**Traditional**:
```python
def add(a, b):
    return a + b

# Test
assert add(2, 3) == 5  # Always true
```

**AI System**:
```python
def classify_log(log):
    return call_llm(log)

# Test
assert classify_log("ERROR occurred") == "ERROR"  # Maybe true?
# LLM might return "CRITICAL", "SEVERE", "Error", etc.
```

**Challenges**:
- Non-deterministic outputs
- Subtle prompt changes break things
- Model updates change behavior
- Edge cases are hard to predict

**Solution**: Probabilistic testing + regression detection + continuous monitoring.

---

## Unit Testing Prompts

Don't jump to the complex testing framework. See how it evolves from a simple test to production-ready test suite.

### Building PromptTest: Progressive Development

#### Version 1: First Simple Test (20 lines)

Start with the absolute basics - one test, one assert:

```python
import pytest
from anthropic import Anthropic
import os

class PromptTest:
    """V1: Simplest possible test."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def test_weak_snmp(self):
        """Test that weak SNMP is detected."""
        config = "snmp-server community public RO"
        prompt = f"Find security issues: {config}"

        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Simple assertion
        assert "snmp" in result.lower()
        assert "public" in result.lower()
```

**What it does:** One test, one assertion checking if keywords appear.

**What's missing:** Reusable test helpers, JSON parsing, multiple test cases, parametrization.

---

#### Version 2: Add Test Helper Methods (35 lines)

Add reusable helper for common assertion patterns:

```python
import pytest
from anthropic import Anthropic
import os

class PromptTest:
    """V2: Add reusable assertion helper."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM with prompt - reusable helper."""
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def assert_contains(self, text: str, expected: str, case_sensitive: bool = False):
        """Reusable assertion helper."""
        if not case_sensitive:
            text = text.lower()
            expected = expected.lower()

        assert expected in text, f"Expected '{expected}' not found in: {text[:200]}"

    def test_weak_snmp(self):
        """Test SNMP detection."""
        config = "snmp-server community public RO"
        prompt = f"Find security issues: {config}"

        response = self.call_llm(prompt)

        # Use helper
        self.assert_contains(response, "snmp")
        self.assert_contains(response, "public")
```

**What it adds:** Reusable `call_llm()` and `assert_contains()` helpers.

**What's still missing:** JSON parsing, multiple test methods, parametrization.

---

#### Version 3: Add JSON Validation (55 lines)

Add helper for testing structured outputs:

```python
import pytest
from anthropic import Anthropic
import os
import re
import json

class PromptTest:
    """V3: Add JSON validation helper."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM with prompt."""
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def assert_contains(self, text: str, expected: str, case_sensitive: bool = False):
        """Assert text contains expected string."""
        if not case_sensitive:
            text = text.lower()
            expected = expected.lower()

        assert expected in text, f"Expected '{expected}' not found in: {text[:200]}"

    def assert_json_structure(self, text: str, expected_keys: list):
        """Assert response is valid JSON with expected keys."""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        assert json_match, "No JSON found in response"

        data = json.loads(json_match.group())
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in JSON"

        return data

    def test_weak_snmp(self):
        """Test SNMP detection."""
        config = "snmp-server community public RO"
        prompt = f"Find security issues: {config}"

        response = self.call_llm(prompt)
        self.assert_contains(response, "snmp")

    def test_structured_output(self):
        """Test JSON structure."""
        config = "interface Gi0/0\n ip address 192.168.1.1 255.255.255.0"
        prompt = f"Extract interface info: {config}\nJSON: {{\"name\": \"...\", \"ip\": \"...\"}}\nONLY JSON."

        response = self.call_llm(prompt)
        data = self.assert_json_structure(response, ["name", "ip"])

        # Validate extracted values
        assert "gi" in data["name"].lower() or "gig" in data["name"].lower()
        assert "192.168.1.1" in data["ip"]
```

**What it adds:** JSON parsing helper, multiple test methods, value validation.

**What's still missing:** Parametrized tests, multiple test classes, inheritance.

---

#### Version 4: Production-Ready Test Suite (75 lines)

Add parametrization and multiple test classes:

```python
#!/usr/bin/env python3
"""
Unit test framework for AI prompts.
"""

import pytest
from anthropic import Anthropic
import os
import re
import json


class PromptTest:
    """Base class for prompt testing."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-haiku-4.5"

    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM with prompt."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,  # Deterministic
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def assert_contains(self, text: str, expected: str, case_sensitive: bool = False):
        """Assert text contains expected string."""
        if not case_sensitive:
            text = text.lower()
            expected = expected.lower()

        assert expected in text, f"Expected '{expected}' not found in: {text[:200]}"

    def assert_json_structure(self, text: str, expected_keys: list):
        """Assert response is valid JSON with expected keys."""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        assert json_match, "No JSON found in response"

        data = json.loads(json_match.group())
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in JSON"

        return data


class TestConfigAnalysis(PromptTest):
    """Test config analysis prompts."""

    def test_detect_weak_snmp(self):
        """Test detection of weak SNMP community."""
        config = "snmp-server community public RO"

        prompt = f"""
Analyze for security issues: {config}
Return findings as JSON: {{"issues": ["list"]}}
"""

        response = self.call_llm(prompt)

        # Should contain mention of SNMP issue
        self.assert_contains(response, "snmp")
        self.assert_contains(response, "public")

    def test_detect_telnet(self):
        """Test detection of insecure telnet."""
        config = "line vty 0 4\n transport input telnet"

        prompt = f"""
Find security issues: {config}
JSON: {{"issues": []}}
"""

        response = self.call_llm(prompt)
        self.assert_contains(response, "telnet")

    def test_structured_output(self):
        """Test structured output format."""
        config = "interface Gi0/0\n ip address 192.168.1.1 255.255.255.0"

        prompt = f"""
Extract interface info: {config}
JSON: {{"name": "...", "ip": "..."}}
ONLY JSON.
"""

        response = self.call_llm(prompt)
        data = self.assert_json_structure(response, ["name", "ip"])

        # Validate values
        assert "gi" in data["name"].lower() or "gig" in data["name"].lower()
        assert "192.168.1.1" in data["ip"]


class TestLogClassification(PromptTest):
    """Test log classification."""

    def test_classify_critical(self):
        """Test critical log classification."""
        log = "%OSPF-5-ADJCHG: Neighbor Down"

        prompt = f"Classify: {log}\nReturn: CRITICAL|ERROR|WARNING|INFO"

        response = self.call_llm(prompt, max_tokens=20)

        # Should be CRITICAL or ERROR
        assert any(word in response.upper() for word in ["CRITICAL", "ERROR"])

    def test_classify_info(self):
        """Test info log classification."""
        log = "%SYS-5-CONFIG_I: Configured from console"

        prompt = f"Classify: {log}\nReturn: CRITICAL|ERROR|WARNING|INFO"

        response = self.call_llm(prompt, max_tokens=20)

        assert "INFO" in response.upper()

    @pytest.mark.parametrize("log,expected", [
        ("%OSPF-5-ADJCHG: Neighbor Down", "CRITICAL"),
        ("%LINEPROTO-5-UPDOWN: Line protocol up", "INFO"),
        ("%SYS-2-MALLOCFAIL: Memory allocation failed", "CRITICAL"),
    ])
    def test_multiple_logs(self, log, expected):
        """Test multiple logs with parametrize."""
        prompt = f"Classify: {log}\nReturn severity level."

        response = self.call_llm(prompt, max_tokens=20)

        self.assert_contains(response, expected, case_sensitive=False)


# Run with: pytest test_prompts.py -v
```

---

### Check Your Understanding: Unit Testing Prompts

Verify you understand the PromptTest pattern before moving to regression testing:

<details>
<summary><strong>Question 1:</strong> Why does PromptTest use <code>temperature=0</code> instead of default (1.0)?</summary>

**Answer**: `temperature=0` makes outputs deterministic (consistent). This is critical for testing because:
- Same prompt → same output (repeatable tests)
- Failures are consistent (easier to debug)
- Tests don't randomly fail/pass

**Without temperature=0**: Your test might pass Monday, fail Tuesday with identical code.

**Example**:
```python
# temperature=0 (deterministic)
response = call_llm("Classify: ERROR", temperature=0)
# Always returns: "ERROR severity"

# temperature=1.0 (creative)
response = call_llm("Classify: ERROR", temperature=1.0)
# Might return: "ERROR severity" or "Critical issue" or "Error level detected"
```

</details>

<details>
<summary><strong>Question 2:</strong> What's the purpose of the <code>assert_contains</code> helper method?</summary>

**Answer**: Provides flexible, case-insensitive keyword matching for AI outputs.

**Why needed**: LLM might return "TELNET", "telnet", "Telnet" - all valid but fail exact equality check.

**Without helper**:
```python
assert "telnet" in result.lower()  # Repeated everywhere
assert "snmp" in result.lower()    # Repeated everywhere
```

**With helper**:
```python
self.assert_contains(result, "telnet")  # DRY, reusable
self.assert_contains(result, "snmp")    # DRY, reusable
```

**Bonus**: Error messages are clearer when assertions fail.

</details>

<details>
<summary><strong>Question 3:</strong> When should you use <code>@pytest.mark.parametrize</code>?</summary>

**Answer**: When testing the same logic with multiple inputs.

**Example scenario**: Test log classification with 10 different log types.

**Without parametrize** (repetitive):
```python
def test_ospf_log(self): ...
def test_bgp_log(self): ...
def test_interface_log(self): ...
# ... 7 more test methods
```

**With parametrize** (DRY):
```python
@pytest.mark.parametrize("log,expected", [
    ("%OSPF-5-ADJCHG: Neighbor Down", "CRITICAL"),
    ("%BGP-5-ADJCHANGE: peer Down", "CRITICAL"),
    ("%LINEPROTO-5-UPDOWN: up", "INFO"),
    # ... 7 more test cases
])
def test_log_classification(self, log, expected):
    # Single test method, runs 10 times
```

**Benefits**: Less code, easier to add test cases, clearer test coverage.

</details>

---

## Regression Testing

Build a regression detection system step-by-step, from simple baseline storage to production monitoring.

### Building RegressionTest: Progressive Development

#### Version 1: Basic Baseline Storage (25 lines)

Start simple - store test results in memory:

```python
from anthropic import Anthropic
import os

class RegressionTest:
    """V1: Basic baseline storage in memory."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.baselines = {}  # In-memory storage

    def test_and_store(self, test_name: str, prompt: str, expected_keyword: str):
        """Run test and store result as baseline."""
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text
        keyword_found = expected_keyword.lower() in result.lower()

        # Store baseline
        self.baselines[test_name] = keyword_found

        print(f"Baseline established for {test_name}: keyword_found={keyword_found}")
```

**What it does:** Calls API, checks for keyword, stores result in memory.

**What's missing:** File persistence (baselines lost on restart), comparison logic, multiple keywords.

---

#### Version 2: Add File Persistence (45 lines)

Save baselines to JSON file so they survive restarts:

```python
from anthropic import Anthropic
from pathlib import Path
import os
import json

class RegressionTest:
    """V2: Add file persistence."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.baseline_file = Path("test_baselines.json")
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> dict:
        """Load baselines from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def _save_baselines(self):
        """Save baselines to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def test_and_store(self, test_name: str, prompt: str, expected_keywords: list):
        """Run test and store baseline."""
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Check multiple keywords
        keywords_found = [kw for kw in expected_keywords if kw.lower() in result.lower()]

        # Store baseline
        self.baselines[test_name] = {
            "keywords_found": keywords_found,
            "response_length": len(result)
        }

        self._save_baselines()
        print(f"PASS: Baseline saved for {test_name}")
```

**What it adds:** File persistence with JSON, multiple keyword support, response metadata.

**What's still missing:** Comparison logic to detect regressions, pytest integration.

---

#### Version 3: Add Regression Detection (70 lines)

Add logic to compare current results vs baseline:

```python
import pytest
from anthropic import Anthropic
from pathlib import Path
import os
import json

class RegressionTest:
    """V3: Add regression detection."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.baseline_file = Path("test_baselines.json")
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> dict:
        """Load baselines from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def _save_baselines(self):
        """Save baselines to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def test_and_compare(self, test_name: str, prompt: str, expected_keywords: list):
        """
        Run test and compare to baseline.

        Args:
            test_name: Unique test identifier
            prompt: Prompt to test
            expected_keywords: Keywords that should appear
        """
        # Call LLM
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Check keywords
        keywords_found = [kw for kw in expected_keywords if kw.lower() in result.lower()]

        current_result = {
            "keywords_found": keywords_found,
            "keywords_missing": [kw for kw in expected_keywords if kw not in keywords_found],
            "response_length": len(result)
        }

        # Compare to baseline
        if test_name in self.baselines:
            baseline = self.baselines[test_name]

            # Check if behavior changed
            if baseline["keywords_found"] != current_result["keywords_found"]:
                pytest.fail(
                    f"REGRESSION: Keywords changed!\n"
                    f"Baseline: {baseline['keywords_found']}\n"
                    f"Current: {current_result['keywords_found']}"
                )

            print(f"PASS: No regression detected for {test_name}")

        else:
            # First run - establish baseline
            print(f"PASS: Establishing baseline for {test_name}")
            self.baselines[test_name] = current_result
            self._save_baselines()

        return current_result
```

**What it adds:** Comparison logic, regression detection, pytest integration with `pytest.fail()`.

**What's still missing:** Better diff reporting, tolerance thresholds, production-ready error handling.

---

#### Version 4: Production-Ready (95 lines)

Add production features - this is the full implementation:

```python
#!/usr/bin/env python3
"""
Regression test suite - detect changes in behavior.
"""

import pytest
import json
from pathlib import Path
from anthropic import Anthropic
import os


class RegressionTest:
    """Test for regressions across model updates."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.baseline_file = Path("test_baselines.json")
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> dict:
        """Load baseline test results."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def _save_baselines(self):
        """Save baseline results."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def test_and_compare(self, test_name: str, prompt: str, expected_keywords: list):
        """
        Run test and compare to baseline.

        Args:
            test_name: Unique test identifier
            prompt: Prompt to test
            expected_keywords: Keywords that should appear
        """
        # Call LLM
        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Check expected keywords
        keywords_found = [kw for kw in expected_keywords if kw.lower() in result.lower()]

        current_result = {
            "keywords_found": keywords_found,
            "keywords_missing": [kw for kw in expected_keywords if kw not in keywords_found],
            "response_length": len(result)
        }

        # Compare to baseline
        if test_name in self.baselines:
            baseline = self.baselines[test_name]

            # Check if behavior changed
            if baseline["keywords_found"] != current_result["keywords_found"]:
                pytest.fail(
                    f"REGRESSION: Keywords changed!\n"
                    f"Baseline: {baseline['keywords_found']}\n"
                    f"Current: {current_result['keywords_found']}"
                )

        else:
            # First run - establish baseline
            print(f"PASS: Establishing baseline for {test_name}")
            self.baselines[test_name] = current_result
            self._save_baselines()

        return current_result


# Example tests
def test_config_analysis_regression():
    """Test config analysis hasn't regressed."""
    tester = RegressionTest()

    prompt = "Analyze security: snmp-server community public RO"
    expected = ["snmp", "community", "weak", "security"]

    tester.test_and_compare("config_security", prompt, expected)


def test_log_classification_regression():
    """Test log classification consistency."""
    tester = RegressionTest()

    prompt = "Classify: %OSPF-5-ADJCHG: Neighbor Down"
    expected = ["critical", "error", "ospf", "neighbor"]

    tester.test_and_compare("log_classify", prompt, expected)


# First run: Establishes baselines
# pytest test_regression.py -v

# Later runs: Detects regressions
# If model behavior changes, test fails
```

---

### Check Your Understanding: Regression Testing

Verify you understand regression testing before moving to the labs:

<details>
<summary><strong>Question 1:</strong> What's the difference between unit testing and regression testing?</summary>

**Answer**:
- **Unit testing**: Verifies current behavior is correct (does this prompt work NOW?)
- **Regression testing**: Verifies behavior hasn't changed over time (does this prompt still work the SAME?)

**Example**:

**Unit test**:
```python
def test_detect_telnet():
    result = analyze("transport input telnet")
    assert "telnet" in result.lower()  # Does it work today?
```

**Regression test**:
```python
def test_detect_telnet_regression():
    result = analyze("transport input telnet")
    # Compare to baseline from last week - did behavior change?
    assert result == baseline["telnet_detection"]
```

**When to use each**:
- Unit tests: Every commit
- Regression tests: Daily, on model updates, before deployments

</details>

<details>
<summary><strong>Question 2:</strong> Why store baselines in a JSON file instead of hardcoding expected values?</summary>

**Answer**: Flexibility and intentional updates.

**Hardcoded (rigid)**:
```python
# Locked to exact output from 2 months ago
assert result == "Critical: Weak SNMP community 'public' detected"
# Breaks if LLM improves output to: "CRITICAL: Insecure SNMP..."
```

**Baseline file (flexible)**:
```python
# Checks keywords, not exact text
assert baseline["keywords_found"] == current["keywords_found"]
# Allows: "Critical", "CRITICAL", "Critical severity" - all valid
```

**When behavior SHOULD change**:
```bash
rm test_baselines.json      # Delete old baselines
pytest test_regression.py   # Re-establish new baselines
git add test_baselines.json # Commit new baselines
```

**Key insight**: Baselines let you intentionally update expected behavior while still catching unintentional changes.

</details>

<details>
<summary><strong>Question 3:</strong> When would a regression test fail, and what should you do?</summary>

**Answer**: Regression test fails when current output differs from baseline.

**Failure scenarios**:

**1. Model update** (Anthropic released new Claude version):
```
REGRESSION: Keywords changed!
Baseline: ['snmp', 'community', 'weak', 'security']
Current: ['snmp', 'community', 'insecure']  # "weak" → "insecure"
```
**Action**: Review if new output is still correct. If yes, update baseline.

**2. Prompt drift** (someone edited prompt):
```
REGRESSION: Keywords changed!
Baseline: ['telnet', 'insecure', 'ssh']
Current: ['telnet']  # Lost "insecure" and "ssh" suggestions
```
**Action**: Fix the prompt (revert or improve), don't just update baseline.

**3. Bug introduction** (code change broke functionality):
```
REGRESSION: Keywords changed!
Baseline: ['critical', 'ospf', 'neighbor']
Current: []  # Nothing detected!
```
**Action**: Fix the bug, don't update baseline.

**Rule of thumb**: Only update baselines when you INTENTIONALLY improved the system.

</details>

---

## Lab 0: Write Your First Prompt Test (20 min)

**Goal**: Create a simple pytest that tests an LLM prompt for config analysis.

**Success Criteria**:
- [ ] Install pytest with `pip install pytest`
- [ ] Create `test_my_prompt.py` with PromptTest class
- [ ] Write test that checks for keyword in LLM response
- [ ] Run test with `pytest test_my_prompt.py -v`
- [ ] See green PASS output

**Expected Outcome**:
```bash
$ pytest test_my_prompt.py -v

test_my_prompt.py::test_detect_telnet PASSED                    [100%]

========================= 1 passed in 2.14s =========================
```

**Instructions**:

1. **Install pytest** (if not already installed):
```bash
pip install pytest anthropic
```

2. **Create test file** `test_my_prompt.py`:
```python
from anthropic import Anthropic
import os

class PromptTest:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def test_detect_telnet():
    """Test that telnet is detected in config."""
    tester = PromptTest()

    config = "line vty 0 4\n transport input telnet"
    prompt = f"Find security issues: {config}"

    response = tester.client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text

    # Simple assertion
    assert "telnet" in result.lower()
```

3. **Set your API key**:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

4. **Run the test**:
```bash
pytest test_my_prompt.py -v
```

5. **Understand the output**:
- `PASSED` = Test succeeded, keyword found
- `FAILED` = Assertion failed, keyword not found
- Time shows how long the LLM call took

6. **Break it intentionally** to see a failure:

Change the prompt to something vague:
```python
prompt = f"Analyze this: {config}"  # Too vague
```

Run again:
```bash
pytest test_my_prompt.py -v
```

You should see `FAILED` with assertion error showing "telnet" not found.

7. **Fix it back** and verify it passes again.

**If You Finish Early**:

1. **Add a second test** for SNMP detection:
```python
def test_detect_snmp():
    tester = PromptTest()
    config = "snmp-server community public RO"
    prompt = f"Find security issues: {config}"

    response = tester.client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text
    assert "snmp" in result.lower()
    assert "public" in result.lower()
```

2. **Add the `assert_contains()` helper** from Version 2 to reduce duplication

3. **Test with different model** - change to `claude-sonnet-4.5` and compare speed/quality

4. **Add parametrization** to test multiple configs:
```python
import pytest

@pytest.mark.parametrize("config,keyword", [
    ("transport input telnet", "telnet"),
    ("snmp-server community public RO", "snmp"),
    ("enable password cisco123", "weak"),
])
def test_security_issues(config, keyword):
    # ... test logic
```

**Common Issues**:

- **API key not set**: `export ANTHROPIC_API_KEY="..."` in your shell
- **Module not found**: Install with `pip install anthropic`
- **Test timeout**: LLM calls take 1-3 seconds, this is normal
- **Flaky test**: LLM might say "TELNET" or "Telnet" - use `.lower()` in assertions

---

## Lab 1: Build Prompt Test Suite (45 min)

**Goal**: Build a production-ready test suite using the PromptTest V4 pattern with multiple test classes and parametrization.

**Success Criteria**:
- [ ] Create `test_suite.py` with base PromptTest class
- [ ] Add TestConfigAnalysis class with 3+ test methods
- [ ] Add TestLogClassification class with parametrized test
- [ ] Run full suite with `pytest test_suite.py -v`
- [ ] All tests pass (green output)

**Expected Outcome**:
```bash
$ pytest test_suite.py -v

test_suite.py::TestConfigAnalysis::test_detect_weak_snmp PASSED     [ 25%]
test_suite.py::TestConfigAnalysis::test_detect_telnet PASSED        [ 50%]
test_suite.py::TestConfigAnalysis::test_structured_output PASSED    [ 75%]
test_suite.py::TestLogClassification::test_classify_critical PASSED [100%]

========================= 4 passed in 8.32s =========================
```

**Instructions**:

1. **Copy the PromptTest V4 code** from earlier in this chapter (the production-ready version around line 310-447)

2. **Save it as** `test_suite.py`

3. **Understand the structure**:
```python
# Base class with helpers
class PromptTest:
    def call_llm(...)      # Reusable API call
    def assert_contains(...) # Reusable assertion
    def assert_json_structure(...) # JSON validation

# Test classes inherit from PromptTest
class TestConfigAnalysis(PromptTest):
    def test_detect_weak_snmp(self): ...
    def test_detect_telnet(self): ...
    def test_structured_output(self): ...

class TestLogClassification(PromptTest):
    def test_classify_critical(self): ...
    @pytest.mark.parametrize(...)
    def test_multiple_logs(self, log, expected): ...
```

4. **Run the full suite**:
```bash
pytest test_suite.py -v
```

5. **Add your own test** to TestConfigAnalysis:
```python
def test_detect_weak_password(self):
    """Test detection of weak enable password."""
    config = "enable password cisco123"

    prompt = f"""
Analyze for security issues: {config}
Return findings as JSON.
"""

    response = self.call_llm(prompt)

    # Should detect weak password
    self.assert_contains(response, "password")
    self.assert_contains(response, "weak")
```

6. **Run again** to see your new test:
```bash
pytest test_suite.py -v
```

You should see 5 passed (4 original + your new test).

7. **Add parametrized test** for multiple weak passwords:
```python
@pytest.mark.parametrize("password,issue", [
    ("cisco", "weak"),
    ("admin", "weak"),
    ("password123", "weak"),
    ("Str0ngP@ssw0rd!", "ok"),  # This should NOT flag as weak
])
def test_password_strength(self, password, issue):
    """Test password strength detection."""
    config = f"enable secret {password}"

    prompt = f"Analyze password strength: {config}"

    response = self.call_llm(prompt, max_tokens=100)

    if issue == "weak":
        self.assert_contains(response, "weak")
    else:
        # Strong password should NOT be flagged
        assert "weak" not in response.lower()
```

8. **Verify all tests pass**:
```bash
pytest test_suite.py -v --tb=short
```

**If You Finish Early**:

1. **Add test for JSON output validation**:
```python
def test_json_findings_format(self):
    """Ensure findings are returned as valid JSON."""
    config = "snmp-server community public RO\nline vty 0 4\n transport input telnet"

    prompt = f"""
Analyze security: {config}
Return JSON: {{"findings": [{{"severity": "...", "issue": "...", "line": "..."}}]}}
ONLY JSON.
"""

    response = self.call_llm(prompt, max_tokens=500)
    data = self.assert_json_structure(response, ["findings"])

    # Validate structure
    assert isinstance(data["findings"], list)
    assert len(data["findings"]) >= 2  # At least SNMP and telnet
```

2. **Add test timing decorator** to measure slow tests:
```python
import time

def test_performance_check(self):
    """Ensure test completes in reasonable time."""
    start = time.time()

    config = "interface Gi0/0"
    prompt = f"Summarize: {config}"
    self.call_llm(prompt, max_tokens=50)

    elapsed = time.time() - start
    assert elapsed < 5.0, f"Test too slow: {elapsed:.2f}s"
```

3. **Run tests in parallel** for speed:
```bash
pip install pytest-xdist
pytest test_suite.py -v -n 4  # 4 parallel workers
```

4. **Add coverage reporting**:
```bash
pip install pytest-cov
pytest test_suite.py -v --cov=. --cov-report=term-missing
```

**Common Issues**:

- **Parametrize not working**: Check decorator is `@pytest.mark.parametrize`, not `@parametrize`
- **Tests run in random order**: Use `pytest -v --collect-only` to preview test discovery
- **Flaky assertions**: Use `case_sensitive=False` in assert_contains for keywords
- **Cost concern**: Each test = 1 API call. 10 tests = ~$0.02 with Haiku

**Verification Questions**:
1. How many API calls does `pytest test_suite.py` make? (Count test methods)
2. What happens if you remove `temperature=0`? (Tests become non-deterministic)
3. Why inherit from PromptTest? (DRY - reuse call_llm and assert helpers)

---

## Test Data Generation

```python
#!/usr/bin/env python3
"""
Generate synthetic test data for networking scenarios.
"""

from anthropic import Anthropic
import os
import json


class TestDataGenerator:
    """Generate synthetic network test data."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_configs(
        self,
        num_configs: int = 10,
        include_issues: bool = True
    ) -> list:
        """
        Generate synthetic network configs.

        Args:
            num_configs: Number of configs to generate
            include_issues: Include security issues

        Returns:
            List of config strings
        """
        prompt = f"""
Generate {num_configs} short Cisco IOS configuration snippets.

{"Include various security issues (weak SNMP, telnet, etc.)" if include_issues else "Make them secure following best practices"}

Return as JSON array of strings:
["config1", "config2", ...]

ONLY JSON array.
"""

        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=3000,
            temperature=0.7,  # Higher temp for variety
            messages=[{"role": "user", "content": prompt}]
        )

        import re
        text = response.content[0].text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return []

    def generate_logs(self, num_logs: int = 50, severity_mix: bool = True) -> list:
        """Generate synthetic syslog entries."""
        prompt = f"""
Generate {num_logs} realistic Cisco syslog entries.

{"Mix of severities: INFO, WARNING, ERROR, CRITICAL" if severity_mix else "All INFO level"}

Return JSON array:
["log1", "log2", ...]
"""

        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        import re
        text = response.content[0].text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return []

    def generate_edge_cases(self) -> list:
        """Generate edge case configs that might break parsers."""
        prompt = """
Generate 10 edge case network configurations that might break parsers:
- Very long interface names
- Unusual characters
- Missing fields
- Malformed syntax
- Empty sections

JSON array.
"""

        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=2000,
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )

        import re
        text = response.content[0].text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())

        return []


# Example usage
if __name__ == "__main__":
    generator = TestDataGenerator()

    # Generate test configs
    configs = generator.generate_configs(num_configs=5, include_issues=True)

    print("Generated Test Configs:")
    for i, config in enumerate(configs, 1):
        print(f"\n--- Config {i} ---")
        print(config[:200])

    # Save for testing
    with open("test_data/generated_configs.json", "w") as f:
        json.dump(configs, f, indent=2)
```

---

## Lab 2: Implement Regression Testing (60 min)

**Goal**: Build a regression test suite that detects when model updates or prompt changes break existing behavior.

**Success Criteria**:
- [ ] Create `test_regression.py` with RegressionTest class
- [ ] Run first time to establish baselines (creates `test_baselines.json`)
- [ ] Modify a prompt and re-run to detect regression
- [ ] See regression detected with clear diff output
- [ ] Restore prompt and verify tests pass again

**Expected Outcome**:

First run (establishing baselines):
```bash
$ pytest test_regression.py -v -s

test_regression.py::test_config_analysis_regression PASS: Establishing baseline for config_security
PASSED
test_regression.py::test_log_classification_regression PASS: Establishing baseline for log_classify
PASSED

========================= 2 passed in 4.21s =========================

$ cat test_baselines.json
{
  "config_security": {
    "keywords_found": ["snmp", "community", "weak", "security"],
    "keywords_missing": [],
    "response_length": 142
  },
  "log_classify": {
    "keywords_found": ["critical", "ospf", "neighbor"],
    "keywords_missing": ["error"],
    "response_length": 87
  }
}
```

After intentional prompt change (regression detected):
```bash
$ pytest test_regression.py -v

test_regression.py::test_config_analysis_regression FAILED

=========================== FAILURES ===========================
test_config_analysis_regression - REGRESSION: Keywords changed!
Baseline: ['snmp', 'community', 'weak', 'security']
Current: ['snmp', 'community']

========================= 1 failed in 2.14s =========================
```

**Instructions**:

1. **Copy the RegressionTest V4 code** from earlier (around line 649-756)

2. **Save as** `test_regression.py`

3. **Run first time** to establish baselines:
```bash
pytest test_regression.py -v -s
```

The `-s` flag shows print output so you see "Establishing baseline" messages.

4. **Check the baseline file** was created:
```bash
cat test_baselines.json
```

You should see JSON with test names and their expected keywords.

5. **Run again** - should pass because baselines exist:
```bash
pytest test_regression.py -v
```

All tests should PASS with "No regression detected" messages.

6. **Intentionally break a prompt** to trigger regression:

Edit `test_regression.py`, find this line:
```python
prompt = "Analyze security: snmp-server community public RO"
```

Change it to something vague:
```python
prompt = "Look at this: snmp-server community public RO"  # Too vague
```

7. **Run tests again**:
```bash
pytest test_regression.py -v
```

You should see FAILED with clear diff showing which keywords are now missing.

8. **Fix the prompt back**:
```python
prompt = "Analyze security: snmp-server community public RO"  # Restored
```

9. **Verify tests pass** again:
```bash
pytest test_regression.py -v
```

Should see PASSED - regression fixed.

10. **Understand when to update baselines**:

When you INTENTIONALLY change expected behavior:
```bash
# Delete old baselines
rm test_baselines.json

# Re-run to establish new baselines
pytest test_regression.py -v -s
```

**If You Finish Early**:

1. **Add a third regression test** for interface parsing:
```python
def test_interface_parsing_regression():
    """Test interface data extraction consistency."""
    tester = RegressionTest()

    prompt = "Extract interface name and IP: interface Gi0/0\n ip address 192.168.1.1 255.255.255.0"
    expected = ["gi", "192.168.1.1", "interface"]

    tester.test_and_compare("interface_parse", prompt, expected)
```

2. **Add tolerance for response length** (allow 10% variance):
```python
# In test_and_compare method, after baseline comparison:
length_diff = abs(current_result["response_length"] - baseline["response_length"])
length_tolerance = baseline["response_length"] * 0.1  # 10%

if length_diff > length_tolerance:
    pytest.fail(f"Response length changed significantly: {baseline['response_length']} → {current_result['response_length']}")
```

3. **Add timestamp to baselines** to track when they were created:
```python
from datetime import datetime

current_result = {
    "keywords_found": keywords_found,
    "keywords_missing": keywords_missing,
    "response_length": len(result),
    "timestamp": datetime.now().isoformat()
}
```

4. **Test across different models**:
```python
@pytest.mark.parametrize("model", ["claude-haiku-4.5", "claude-sonnet-4.5"])
def test_model_consistency(model):
    """Test that both models produce similar results."""
    tester = RegressionTest()
    tester.client.model = model  # Override model

    prompt = "Classify: %OSPF-5-ADJCHG: Neighbor Down"
    expected = ["critical", "error", "ospf"]

    tester.test_and_compare(f"log_classify_{model}", prompt, expected)
```

5. **Create a baseline comparison script**:
```python
#!/usr/bin/env python3
"""Compare two baseline files to see what changed."""
import json
import sys

old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))

for test_name in old:
    if test_name in new:
        if old[test_name]["keywords_found"] != new[test_name]["keywords_found"]:
            print(f"CHANGED: {test_name}")
            print(f"  Old: {old[test_name]['keywords_found']}")
            print(f"  New: {new[test_name]['keywords_found']}")
```

Usage:
```bash
cp test_baselines.json test_baselines_old.json
# ... make changes ...
pytest test_regression.py -v -s
python compare_baselines.py test_baselines_old.json test_baselines.json
```

**Common Issues**:

- **Baseline file not created**: Check write permissions in current directory
- **Tests always fail**: Delete `test_baselines.json` and re-establish baselines
- **Keywords not found**: LLM might use synonyms ("critical" vs "severe") - adjust expected keywords
- **Flaky tests**: Some prompts are naturally non-deterministic; use `temperature=0` for consistency

**Verification Questions**:
1. When should you delete baselines? (When intentionally changing expected behavior)
2. What does `keywords_missing` track? (Expected keywords that weren't found in response)
3. Why use regression tests vs unit tests? (Regression detects unintentional changes over time)

---

## Lab 3: Generate Test Data with AI (45 min)

**Goal**: Use Claude to generate synthetic test data for network testing scenarios.

**Success Criteria**:
- [ ] Create `generate_test_data.py` with TestDataGenerator class
- [ ] Generate 10 Cisco configs with security issues
- [ ] Generate 20 syslog entries with mixed severities
- [ ] Save generated data to JSON files
- [ ] Use generated data in a test

**Expected Outcome**:
```bash
$ python generate_test_data.py

Generated Test Configs:
--- Config 1 ---
snmp-server community public RO
snmp-server community private RW
! Weak SNMP communities

--- Config 2 ---
line vty 0 4
 transport input telnet
! Insecure remote access

--- Config 3 ---
enable password cisco123
! Weak enable password

Saved: test_data/generated_configs.json
Saved: test_data/generated_logs.json

Generated 10 configs and 20 logs
```

**Instructions**:

1. **Copy TestDataGenerator code** from earlier (around line 1067-1198)

2. **Create directory** for test data:
```bash
mkdir -p test_data
```

3. **Save as** `generate_test_data.py`

4. **Run the generator**:
```bash
python generate_test_data.py
```

You should see generated configs and logs printed to console and saved to JSON files.

5. **Inspect generated data**:
```bash
cat test_data/generated_configs.json | jq '.[0]'
```

You should see realistic Cisco config snippets with intentional security issues.

6. **Use generated data in a test**:

Create `test_with_generated_data.py`:
```python
import pytest
import json
from anthropic import Anthropic
import os

class TestWithGeneratedData:
    """Test using AI-generated test data."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load_test_configs(self):
        """Load generated configs."""
        with open("test_data/generated_configs.json") as f:
            return json.load(f)

@pytest.mark.parametrize("config", TestWithGeneratedData().load_test_configs())
def test_all_generated_configs(config):
    """Test each generated config for security issues."""
    tester = TestWithGeneratedData()

    prompt = f"Find security issues: {config}"

    response = tester.client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text

    # Should detect at least one issue
    assert any(word in result.lower() for word in ["weak", "insecure", "issue", "risk"])
```

7. **Run tests with generated data**:
```bash
pytest test_with_generated_data.py -v
```

You should see 10 tests run (one per generated config), all passing.

**If You Finish Early**:

1. **Generate edge cases**:
```python
generator = TestDataGenerator()
edge_cases = generator.generate_edge_cases()

print("Edge Cases:")
for edge in edge_cases:
    print(f"\n{edge}")
```

2. **Generate data with specific characteristics**:
```python
def generate_configs_with_issue(self, issue_type: str, num_configs: int = 5) -> list:
    """Generate configs with specific security issue."""
    prompt = f"""
Generate {num_configs} Cisco IOS configs that contain {issue_type} issues.

Examples of {issue_type}:
- If "SNMP": weak community strings
- If "telnet": insecure remote access
- If "password": weak passwords

JSON array of config strings.
"""

    response = self.client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    import re
    text = response.content[0].text
    json_match = re.search(r'\[.*\]', text, re.DOTALL)

    if json_match:
        return json.loads(json_match.group())
    return []

# Usage
snmp_configs = generator.generate_configs_with_issue("SNMP", num_configs=5)
telnet_configs = generator.generate_configs_with_issue("telnet", num_configs=5)
```

3. **Generate negative test cases** (configs that SHOULD NOT flag):
```python
secure_configs = generator.generate_configs(num_configs=5, include_issues=False)

# Test that secure configs don't trigger false positives
@pytest.mark.parametrize("config", secure_configs)
def test_secure_configs_no_issues(config):
    # ... should NOT find security issues
    assert "critical" not in result.lower()
```

4. **Track generation cost**:
```python
import time

start = time.time()
configs = generator.generate_configs(num_configs=50)
elapsed = time.time() - start

# Rough cost: Haiku 3000 tokens output = ~$0.015
print(f"Generated {len(configs)} configs in {elapsed:.2f}s (~$0.015)")
```

**Common Issues**:

- **No JSON in response**: LLM might add explanation text; regex `r'\[.*\]'` extracts JSON array
- **Invalid JSON**: Use `json.loads()` with try/except to handle malformed JSON
- **Empty results**: Check model had enough `max_tokens` to generate requested amount
- **Test data directory not found**: Create with `mkdir -p test_data` before running

**Verification Questions**:
1. Why use `temperature=0.7` for generation vs `0` for testing? (Higher temp = more variety)
2. What's the benefit of AI-generated test data? (Scale, edge cases, no manual work)
3. When should you regenerate test data? (When adding new test scenarios or finding gaps)

---

## Integration Testing

```python
#!/usr/bin/env python3
"""
Integration tests - test complete workflows.
"""

import pytest
from unittest.mock import Mock, patch
import sys
sys.path.append('..')

# Import your modules
from config_analyzer import ConfigAnalyzer
from cost_monitor import CostMonitor


class TestConfigAnalysisWorkflow:
    """Test complete config analysis workflow."""

    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        # Setup
        analyzer = ConfigAnalyzer()
        test_config = """
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
snmp-server community public RO
line vty 0 4
 transport input telnet
"""

        # Execute
        result = analyzer.analyze(test_config)

        # Verify
        assert result is not None
        assert 'findings' in result
        assert len(result['findings']) > 0

        # Check specific findings
        issues = [f['issue'].lower() for f in result['findings']]
        assert any('snmp' in issue for issue in issues)
        assert any('telnet' in issue for issue in issues)

    @patch('anthropic.Anthropic')
    def test_with_mocked_api(self, mock_anthropic):
        """Test with mocked API calls (fast, no cost)."""
        # Mock API response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"findings": [{"severity": "critical", "issue": "test"}]}')]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        analyzer = ConfigAnalyzer()
        result = analyzer.analyze("test config")

        # Verify mock was called
        assert mock_client.messages.create.called


class TestCostTracking:
    """Test cost tracking integration."""

    def test_cost_logging(self):
        """Test that costs are logged correctly."""
        monitor = CostMonitor()

        # Log some requests
        monitor.log_request(
            model="claude-haiku-4.5",
            task_type="classification",
            input_tokens=100,
            output_tokens=50,
            cost=0.0003,
            user_id="test_user"
        )

        # Verify
        stats = monitor.get_cost_by_user(days=1)
        assert len(stats) > 0
        assert stats[0]['user'] == "test_user"


# Run with: pytest test_integration.py -v
```

---

## Lab 4: Build Integration Tests (60 min)

**Goal**: Create integration tests that verify complete end-to-end workflows, including mocked API tests for fast iteration.

**Success Criteria**:
- [ ] Create `test_integration.py` with workflow test
- [ ] Import your ConfigAnalyzer (or create a stub)
- [ ] Write end-to-end test that analyzes config and validates findings
- [ ] Add mocked API test for fast, no-cost testing
- [ ] Run both real and mocked tests successfully

**Expected Outcome**:
```bash
$ pytest test_integration.py -v

test_integration.py::TestConfigWorkflow::test_end_to_end_analysis PASSED    [ 50%]
test_integration.py::TestConfigWorkflow::test_with_mocked_api PASSED        [100%]

========================= 2 passed in 3.87s =========================
```

**Instructions**:

1. **Create a simple ConfigAnalyzer** to test (or use existing code):

`config_analyzer.py`:
```python
from anthropic import Anthropic
import os
import re
import json

class ConfigAnalyzer:
    """Analyze network configs for security issues."""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def analyze(self, config: str) -> dict:
        """
        Analyze config and return findings.

        Returns:
            {"findings": [{"severity": "...", "issue": "..."}]}
        """
        prompt = f"""
Analyze this network config for security issues:

{config}

Return JSON:
{{"findings": [{{"severity": "critical|high|medium|low", "issue": "description"}}]}}

ONLY JSON.
"""

        response = self.client.messages.create(
            model="claude-haiku-4.5",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text

        # Extract JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {"findings": []}
```

2. **Create integration test file** `test_integration.py`:

Copy the Integration Testing code from earlier (around line 1608-1695).

3. **Run the end-to-end test** (this makes a real API call):
```bash
pytest test_integration.py::TestConfigAnalysisWorkflow::test_end_to_end_analysis -v
```

Should PASS and show detected issues (SNMP, telnet).

4. **Run the mocked test** (no API call, instant, free):
```bash
pytest test_integration.py::TestConfigAnalysisWorkflow::test_with_mocked_api -v
```

Should PASS instantly (~0.1s instead of ~2s).

5. **Add your own integration test** for multi-step workflow:
```python
def test_batch_analysis_workflow(self):
    """Test analyzing multiple configs in batch."""
    analyzer = ConfigAnalyzer()

    configs = [
        "snmp-server community public RO",
        "line vty 0 4\n transport input telnet",
        "enable password cisco123"
    ]

    results = []
    for config in configs:
        result = analyzer.analyze(config)
        results.append(result)

    # Verify all returned findings
    assert len(results) == 3
    for result in results:
        assert 'findings' in result
        assert len(result['findings']) > 0
```

6. **Run all integration tests**:
```bash
pytest test_integration.py -v
```

7. **Understand when to use mocks**:

**Use real API calls for**:
- End-to-end smoke tests
- Regression detection
- Production validation

**Use mocked API calls for**:
- Fast iteration during development
- Testing error handling
- CI/CD pipelines (to avoid costs)

**If You Finish Early**:

1. **Add fixture for reusable test analyzer**:
```python
import pytest

@pytest.fixture
def analyzer():
    """Reusable analyzer fixture."""
    return ConfigAnalyzer()

@pytest.fixture
def sample_config():
    """Reusable test config."""
    return """
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
snmp-server community public RO
"""

def test_with_fixtures(analyzer, sample_config):
    """Test using fixtures."""
    result = analyzer.analyze(sample_config)
    assert 'findings' in result
```

2. **Add error handling test**:
```python
def test_handles_empty_config(self):
    """Test analyzer handles empty config gracefully."""
    analyzer = ConfigAnalyzer()

    result = analyzer.analyze("")

    # Should not crash, return empty or benign findings
    assert 'findings' in result
```

3. **Add mock for specific error scenarios**:
```python
@patch('anthropic.Anthropic')
def test_handles_api_error(self, mock_anthropic):
    """Test handling of API errors."""
    # Mock API to raise exception
    mock_client = Mock()
    mock_client.messages.create.side_effect = Exception("API Error")
    mock_anthropic.return_value = mock_client

    analyzer = ConfigAnalyzer()

    # Should handle gracefully
    with pytest.raises(Exception):
        analyzer.analyze("test config")
```

4. **Add performance test**:
```python
import time

def test_analysis_performance(self):
    """Ensure analysis completes in reasonable time."""
    analyzer = ConfigAnalyzer()

    config = "interface Gi0/0\n ip address 1.1.1.1 255.255.255.0"

    start = time.time()
    result = analyzer.analyze(config)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Analysis too slow: {elapsed:.2f}s"
    assert 'findings' in result
```

5. **Test different config types**:
```python
@pytest.mark.parametrize("config_type,config", [
    ("cisco_ios", "interface Gi0/0"),
    ("juniper", "set interfaces ge-0/0/0"),
    ("arista", "interface Ethernet1"),
])
def test_multi_vendor_support(config_type, config):
    """Test support for different vendors."""
    analyzer = ConfigAnalyzer()

    result = analyzer.analyze(config)

    assert result is not None
    assert 'findings' in result
```

**Common Issues**:

- **Import error**: Ensure `config_analyzer.py` is in same directory or Python path
- **Mock not working**: Check import path in `@patch('anthropic.Anthropic')` matches your code
- **Fixture not found**: Fixtures must be in same file or `conftest.py`
- **Test isolation**: Each test should be independent; don't share state between tests

**Verification Questions**:
1. What's the difference between unit test and integration test? (Unit = single function, Integration = complete workflow)
2. When should you use mocks? (Fast iteration, testing error paths, CI/CD cost control)
3. Why use fixtures? (DRY - reuse common setup code across tests)

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: AI System Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Run regression tests
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        pytest tests/regression/ -v

    - name: Check code quality
      run: |
        pip install pylint black
        black --check src/
        pylint src/

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

    - name: Test report
      if: always()
      uses: dorny/test-reporter@v1
      with:
        name: Test Results
        path: 'test-results/*.xml'
        reporter: java-junit
```

---

## Lab 5: Set Up CI/CD Pipeline (90 min)

**Goal**: Create a GitHub Actions workflow that automatically runs your test suite on every commit and pull request.

**Success Criteria**:
- [ ] Create `.github/workflows/test.yml` workflow file
- [ ] Configure ANTHROPIC_API_KEY as GitHub secret
- [ ] Push to GitHub and see workflow run automatically
- [ ] All tests pass in CI environment
- [ ] See test results and coverage reports in GitHub Actions UI

**Expected Outcome**:

GitHub Actions UI shows:
```
AI System Tests
✓ Set up Python
✓ Install dependencies
✓ Run unit tests (4 passed in 8.2s)
✓ Run regression tests (2 passed in 4.1s)
✓ Check code quality
✓ Upload coverage (87% coverage)
```

**Instructions**:

1. **Create directory structure** in your Git repo:
```bash
mkdir -p .github/workflows
mkdir -p tests/unit
mkdir -p tests/regression
mkdir -p src
```

2. **Move your test files** into organized structure:
```bash
# Unit tests
mv test_suite.py tests/unit/
mv test_prompts.py tests/unit/

# Regression tests
mv test_regression.py tests/regression/

# Integration tests
mv test_integration.py tests/

# Source code
mv config_analyzer.py src/
```

3. **Create workflow file** `.github/workflows/test.yml`:

Copy the GitHub Actions workflow from earlier (around line 1703-1760).

4. **Create `requirements.txt`**:
```txt
anthropic>=0.40.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

5. **Add GitHub secret for API key**:
- Go to your GitHub repo → Settings → Secrets and variables → Actions
- Click "New repository secret"
- Name: `ANTHROPIC_API_KEY`
- Value: Your actual API key
- Click "Add secret"

6. **Commit and push**:
```bash
git add .github/workflows/test.yml
git add requirements.txt
git add tests/
git add src/
git commit -m "Add CI/CD pipeline with pytest

- GitHub Actions workflow for automated testing
- Unit tests in tests/unit/
- Regression tests in tests/regression/
- Code coverage reporting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git push origin main
```

7. **Watch the workflow run**:
- Go to your GitHub repo → Actions tab
- You should see "AI System Tests" workflow running
- Click on it to see real-time logs
- Wait for all steps to complete (green checkmarks)

8. **Fix any failures**:

Common first-time issues:
```bash
# Import errors - fix with __init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/regression/__init__.py
touch src/__init__.py

# Path issues - fix imports in test files
# Change: from config_analyzer import ConfigAnalyzer
# To: from src.config_analyzer import ConfigAnalyzer
```

9. **Verify coverage report**:

Add `pytest.ini` in repo root:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
```

10. **Create a pull request** to test PR workflow:
```bash
git checkout -b test-pr-workflow
echo "# Test PR" >> README.md
git add README.md
git commit -m "Test PR workflow"
git push origin test-pr-workflow
```

Open PR on GitHub and verify tests run automatically.

**If You Finish Early**:

1. **Add code quality checks** to workflow:

Add this step to `.github/workflows/test.yml`:
```yaml
    - name: Run pylint
      run: |
        pip install pylint
        pylint src/ --disable=C0111,R0903  # Disable docstring and too-few-public-methods warnings
```

2. **Add test result annotations**:

```yaml
    - name: Publish test results
      uses: EnricoMi/publish-unit-test-result-action@v2
      if: always()
      with:
        files: |
          test-results/**/*.xml
```

3. **Add caching for faster runs**:

```yaml
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
```

4. **Add matrix testing** across Python versions:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
```

5. **Add nightly regression runs**:

Create `.github/workflows/nightly.yml`:
```yaml
name: Nightly Regression Tests

on:
  schedule:
    - cron: '0 0 * * *'  # Run at midnight UTC
  workflow_dispatch:  # Allow manual trigger

jobs:
  regression:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run regression suite
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        pytest tests/regression/ -v --tb=long

    - name: Upload baseline changes
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: baseline-diff
        path: test_baselines.json
```

6. **Add status badge** to README.md:

```markdown
# AI Network Config Analyzer

![Tests](https://github.com/yourusername/yourrepo/actions/workflows/test.yml/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/yourusername/yourrepo)

Automated network configuration analysis using Claude AI.
```

7. **Add pre-commit hook** for local testing:

`.git/hooks/pre-commit`:
```bash
#!/bin/bash
echo "Running tests before commit..."
pytest tests/unit/ -v --tb=short

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

echo "Tests passed. Proceeding with commit."
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

**Common Issues**:

- **API key not found**: Verify secret name matches exactly (case-sensitive)
- **Import errors in CI**: Check all `__init__.py` files exist
- **Tests pass locally but fail in CI**: Check Python version consistency
- **Workflow not triggering**: Verify `.yml` file is in `.github/workflows/` directory
- **High API costs**: Consider mocking tests in CI, only run real API tests nightly

**Cost Management**:

Running full test suite in CI:
- Unit tests: 10 tests × $0.002 = $0.02 per run
- Regression tests: 5 tests × $0.002 = $0.01 per run
- Per commit: ~$0.03
- Daily (10 commits): ~$0.30
- Monthly: ~$9

**To reduce costs**:
1. Use mocked tests in PR checks
2. Run real API tests only on `main` branch
3. Use nightly regression runs instead of per-commit
4. Use Haiku ($1/$5) instead of Sonnet ($3/$15) for tests

**Verification Questions**:
1. Why run tests in CI vs locally? (Catch issues before merge, consistent environment, team-wide quality gate)
2. When should you use secrets vs hardcoded values? (ALWAYS use secrets for API keys, never commit credentials)
3. What's the benefit of coverage reporting? (Identify untested code paths, measure test quality)

---

## Quality Metrics

```python
#!/usr/bin/env python3
"""
Measure AI system quality metrics.
"""

from typing import List, Dict
import json


class QualityMetrics:
    """Calculate quality metrics for AI systems."""

    def calculate_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate accuracy.

        Args:
            predictions: Model predictions
            ground_truth: Correct answers

        Returns:
            Accuracy (0.0 to 1.0)
        """
        correct = sum(p.lower() == g.lower() for p, g in zip(predictions, ground_truth))
        return correct / len(predictions) if predictions else 0.0

    def calculate_precision_recall(
        self,
        predictions: List[str],
        ground_truth: List[str],
        positive_class: str
    ) -> Dict:
        """Calculate precision, recall, F1."""
        tp = sum((p == positive_class and g == positive_class)
                 for p, g in zip(predictions, ground_truth))
        fp = sum((p == positive_class and g != positive_class)
                 for p, g in zip(predictions, ground_truth))
        fn = sum((p != positive_class and g == positive_class)
                 for p, g in zip(predictions, ground_truth))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def calculate_confusion_matrix(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict:
        """Calculate confusion matrix."""
        from collections import defaultdict

        matrix = defaultdict(lambda: defaultdict(int))

        for pred, truth in zip(predictions, ground_truth):
            matrix[truth][pred] += 1

        return dict(matrix)


# Example usage
if __name__ == "__main__":
    metrics = QualityMetrics()

    # Test log classification
    predictions = ["CRITICAL", "INFO", "ERROR", "INFO", "CRITICAL"]
    ground_truth = ["CRITICAL", "INFO", "ERROR", "WARNING", "CRITICAL"]

    accuracy = metrics.calculate_accuracy(predictions, ground_truth)
    print(f"Accuracy: {accuracy:.2%}")

    pr = metrics.calculate_precision_recall(predictions, ground_truth, "CRITICAL")
    print(f"Precision: {pr['precision']:.2%}")
    print(f"Recall: {pr['recall']:.2%}")
    print(f"F1: {pr['f1']:.2%}")

    matrix = metrics.calculate_confusion_matrix(predictions, ground_truth)
    print("\nConfusion Matrix:")
    print(json.dumps(matrix, indent=2))
```

---

## Lab Time Budget

**Total Time**: ~6 hours hands-on work

**Recommended Schedule**: 3 weeks, 2 hours per week

### Week 1: Foundations (2 hours)
- **Lab 0**: Write Your First Prompt Test (20 min)
  - Get pytest working, understand basic assertions
- **Lab 1**: Build Prompt Test Suite (45 min)
  - Learn test classes, parametrization, helpers
- **Break**: Review test output, fix any failures (15 min)
- **Lab 2**: Implement Regression Testing (60 min)
  - Establish baselines, detect regressions

**Week 1 Deliverable**: Working test suite with 10+ tests and baseline file

### Week 2: Advanced Testing (2 hours)
- **Lab 3**: Generate Test Data with AI (45 min)
  - Automate test data creation at scale
- **Break**: Review generated data quality (15 min)
- **Lab 4**: Build Integration Tests (60 min)
  - End-to-end workflows, mocked tests

**Week 2 Deliverable**: Integration tests + 50+ generated test configs

### Week 3: Production Ready (2 hours)
- **Lab 5**: Set Up CI/CD Pipeline (90 min)
  - GitHub Actions, automated testing, coverage reports
- **Review & Refinement**: Fix CI/CD issues, optimize (30 min)

**Week 3 Deliverable**: Fully automated CI/CD pipeline running on every commit

### If You're Short on Time

**Minimum viable (1 hour)**:
- Lab 0 (20 min) - Basic testing
- Lab 1 (40 min) - Test suite basics

**Standard path (3 hours)**:
- Lab 0, 1, 2 - Core testing framework
- Skip Labs 3-5 initially, return later

**Full completion (6 hours)**:
- All labs in order
- Production-ready testing infrastructure

### Cost Estimate

**Development phase** (building tests):
- Lab 0: ~5 API calls = $0.01
- Lab 1: ~10 API calls = $0.02
- Lab 2: ~10 API calls = $0.02
- Lab 3: ~20 API calls (data generation) = $0.04
- Lab 4: ~5 API calls = $0.01
- Lab 5: ~20 API calls (CI/CD runs) = $0.04

**Total development**: ~$0.15

**Ongoing costs** (per week):
- Daily regression runs: 5 tests × 7 days = 35 calls/week = $0.07/week
- CI/CD per commit: 15 tests × 10 commits = 150 calls/week = $0.30/week

**Monthly**: ~$1.50 for comprehensive automated testing

**To reduce costs**:
- Use mocked tests in CI (free, instant)
- Run real API tests only nightly
- Use Haiku for most tests, Sonnet only for critical paths

---

## Best Practices

### DO:

1. **Test prompts like code**
   - Unit tests for each prompt
   - Regression tests for stability
   - Integration tests for workflows

2. **Use baselines**
   - Establish expected behavior
   - Detect when things change
   - Version your baselines

3. **Generate test data**
   - Don't rely on manual data
   - Generate edge cases
   - Test at scale

4. **Measure quality**
   - Track accuracy over time
   - Monitor precision/recall
   - Set quality thresholds

5. **Automate testing**
   - CI/CD for every commit
   - Automated regression detection
   - Quality gates before deploy

### DON'T:

1. **Don't skip testing**
   - "It works" isn't good enough
   - Production will find your bugs

2. **Don't test only happy paths**
   - Edge cases break AI systems
   - Test failure modes

3. **Don't ignore regressions**
   - Model updates change behavior
   - Track baselines

4. **Don't test in production**
   - Use staging/test environments
   - Catch issues before users

---

## Key Takeaways

1. **AI systems need systematic testing**
   - Non-deterministic doesn't mean untestable
   - Use probabilistic testing strategies
   - Regression detection is critical

2. **Test at multiple levels**
   - Unit tests (prompts)
   - Integration tests (workflows)
   - Regression tests (stability)
   - Quality metrics (accuracy)

3. **Automate everything**
   - CI/CD pipelines
   - Automated test generation
   - Continuous monitoring

4. **Measure quality continuously**
   - Track metrics over time
   - Detect degradation early
   - Set quality gates

5. **Test is investment, not cost**
   - Catch issues before production
   - Faster iteration cycles
   - Confidence in deployments

---

## Next Steps

You now have a complete testing framework for AI networking systems. You can detect regressions, measure quality, and deploy with confidence.

**Next chapter**: Ethics and Responsible AI—the final chapter of Part 1 covering bias, explainability, human oversight, and when NOT to use AI.

**Ready?** → Chapter 12: Ethics and Responsible AI

---

**Chapter Status**: Complete (Enhanced) | Word Count: ~6,000 | Code: Production-Ready

**What's New in This Version**:
- Real-world opening story (the production incident that taught testing)
- AI testing pyramid diagram and comparison table
- Practical framing around prompt drift and regression detection
