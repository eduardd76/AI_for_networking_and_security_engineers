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

## The Problem: "It Works on My Machine"

You built a config analyzer. It works perfectly in testing:

```python
# Your test
config = "snmp-server community public RO"
result = analyze_config(config)
assert "weak SNMP" in result
# ✓ PASS
```

Then production happens:
- **Day 1**: Works fine
- **Day 2**: Claude API updated, outputs changed format
- **Day 3**: New engineer modifies prompt, breaks parsing
- **Day 4**: Edge case config causes crash
- **Day 5**: Boss asks: "How do we know this actually works?"

**The reality**: Untested AI systems fail in creative ways.

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

### Test Framework

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
        self.model = "claude-3-5-haiku-20241022"

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

## Regression Testing

### Detect When Model Updates Break Things

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
            model="claude-3-5-haiku-20241022",
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
            print(f"✓ Establishing baseline for {test_name}")
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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-haiku-20241022",
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
            model="claude-3-5-haiku",
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

## Best Practices

### ✅ DO:

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

### ❌ DON'T:

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

**Chapter Status**: Complete | Word Count: ~5,000 | Code: Production-Ready
