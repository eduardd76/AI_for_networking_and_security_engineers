"""
Chapter 43: LLMOps Lifecycle & CI/CD Pipeline
Production operations for LLM-powered network systems

This module demonstrates complete LLMOps workflow including experiment
tracking, automated testing, drift detection, and production deployment.

Author: Eduard Dulharu
Company: vExpertAI GmbH
"""

import os
import time
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DeploymentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """Individual CI/CD pipeline stage."""
    id: str
    name: str
    status: StageStatus = StageStatus.PENDING
    logs: List[str] = field(default_factory=list)
    duration_ms: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Experiment:
    """LLM experiment with metrics."""
    id: str
    name: str
    prompt_version: str
    model: str
    prompt_content: str
    accuracy: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Deployment:
    """Production deployment state."""
    version: str
    environment: str
    status: DeploymentStatus = DeploymentStatus.HEALTHY
    drift_score: float = 0.0
    uptime_percent: float = 99.99
    deployed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMOpsPipeline:
    """
    Complete LLMOps pipeline management.

    Pipeline stages:
    1. Lint - Validate prompt quality
    2. Unit Tests - Basic functionality tests
    3. Offline Eval - Performance on test set
    4. Deploy Staging - Push to staging environment
    5. E2E Tests - Integration tests
    6. Governance - Safety and compliance checks
    7. Deploy Production - Promote to production
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.deployments: Dict[str, Deployment] = {}

        # Model cost/latency profiles
        self.model_profiles = {
            'gpt-3.5-turbo': {
                'input_cost_per_1m': 0.50,
                'output_cost_per_1m': 1.50,
                'latency_ms': (300, 600),
                'accuracy': 0.78
            },
            'gpt-4': {
                'input_cost_per_1m': 30.00,
                'output_cost_per_1m': 60.00,
                'latency_ms': (800, 1500),
                'accuracy': 0.92
            },
            'gpt-4-turbo': {
                'input_cost_per_1m': 10.00,
                'output_cost_per_1m': 30.00,
                'latency_ms': (600, 1200),
                'accuracy': 0.90
            },
            'claude-3-opus': {
                'input_cost_per_1m': 15.00,
                'output_cost_per_1m': 75.00,
                'latency_ms': (400, 900),
                'accuracy': 0.93
            },
            'claude-3-sonnet': {
                'input_cost_per_1m': 3.00,
                'output_cost_per_1m': 15.00,
                'latency_ms': (200, 600),
                'accuracy': 0.88
            },
            'claude-3-5-haiku': {
                'input_cost_per_1m': 0.80,
                'output_cost_per_1m': 4.00,
                'latency_ms': (150, 400),
                'accuracy': 0.85
            },
        }

    def create_experiment(
        self,
        name: str,
        prompt_version: str,
        model: str,
        prompt_content: str
    ) -> Experiment:
        """
        Create new experiment.

        Args:
            name: Experiment name
            prompt_version: Version identifier
            model: Model to test
            prompt_content: The prompt to evaluate

        Returns:
            Experiment object
        """
        exp_id = hashlib.md5(
            f"{name}-{prompt_version}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        exp = Experiment(
            id=exp_id,
            name=name,
            prompt_version=prompt_version,
            model=model,
            prompt_content=prompt_content
        )

        self.experiments[exp_id] = exp
        return exp

    def run_pipeline(
        self,
        experiment: Experiment,
        test_dataset: List[Dict[str, str]],
        deployment_target: str = 'staging'
    ) -> Tuple[bool, List[PipelineStage]]:
        """
        Execute full CI/CD pipeline.

        Args:
            experiment: Experiment to deploy
            test_dataset: Test cases
            deployment_target: 'staging' or 'production'

        Returns:
            (success, list of stages with results)
        """
        experiment.status = ExperimentStatus.RUNNING

        # Define pipeline
        stages = [
            PipelineStage(id='lint', name='Prompt Linting'),
            PipelineStage(id='unit', name='Unit Tests'),
            PipelineStage(id='eval', name='Offline Evaluation'),
            PipelineStage(id='deploy-stg', name='Deploy to Staging'),
            PipelineStage(id='e2e', name='End-to-End Tests'),
            PipelineStage(id='governance', name='Governance & Compliance'),
            PipelineStage(id='deploy-prod', name='Promote to Production'),
        ]

        try:
            # Stage 1: Lint
            self._run_stage(stages[0], lambda: self._lint_prompt(experiment))

            # Stage 2: Unit Tests
            self._run_stage(stages[1], lambda: self._unit_tests(experiment))

            # Stage 3: Offline Evaluation
            def eval_stage():
                accuracy, latency, cost = self._offline_eval(experiment, test_dataset)
                experiment.accuracy = accuracy
                experiment.latency_ms = latency
                experiment.cost_usd = cost
                return {
                    'accuracy': accuracy,
                    'latency_ms': latency,
                    'cost_usd': cost
                }

            result = self._run_stage(stages[2], eval_stage)
            stages[2].logs.append(
                f"Accuracy: {result['accuracy']:.1f}%, "
                f"Latency: {result['latency_ms']:.0f}ms, "
                f"Cost: ${result['cost_usd']:.4f}"
            )

            # Stage 4: Deploy Staging
            self._run_stage(stages[3], lambda: self._deploy_to_staging(experiment))

            # Stage 5: E2E Tests
            self._run_stage(stages[4], lambda: self._e2e_tests(experiment))

            # Stage 6: Governance
            self._run_stage(stages[5], lambda: self._governance_check(experiment))

            # Stage 7: Production (conditional)
            if deployment_target == 'production':
                self._run_stage(stages[6], lambda: self._promote_to_production(experiment))
            else:
                stages[6].status = StageStatus.SKIPPED
                stages[6].logs.append("Skipped - staging deployment only")

            experiment.status = ExperimentStatus.COMPLETED
            return True, stages

        except Exception as e:
            # Mark failed stage
            for stage in stages:
                if stage.status == StageStatus.RUNNING:
                    stage.status = StageStatus.FAILED
                    stage.logs.append(f"Failed: {str(e)}")
                    stage.completed_at = datetime.now().isoformat()
                    break
                elif stage.status == StageStatus.PENDING:
                    stage.status = StageStatus.SKIPPED

            experiment.status = ExperimentStatus.FAILED
            return False, stages

    def _run_stage(self, stage: PipelineStage, func):
        """Execute a pipeline stage."""
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.now().isoformat()

        start_time = time.time()

        try:
            result = func()
            stage.status = StageStatus.SUCCESS
            stage.completed_at = datetime.now().isoformat()
            stage.duration_ms = int((time.time() - start_time) * 1000)

            if not stage.logs:
                stage.logs.append(f"{stage.name} completed successfully")

            return result

        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.completed_at = datetime.now().isoformat()
            stage.duration_ms = int((time.time() - start_time) * 1000)
            stage.logs.append(f"Error: {str(e)}")
            raise

    def _lint_prompt(self, exp: Experiment) -> Dict:
        """Validate prompt quality."""
        issues = []

        # Check length
        if len(exp.prompt_content) < 50:
            issues.append("Prompt too short (< 50 chars)")

        # Check for clear instructions
        if not any(word in exp.prompt_content.lower() for word in ['analyze', 'generate', 'explain', 'list']):
            issues.append("No clear action verb")

        # Check for examples
        if 'example' not in exp.prompt_content.lower() and len(exp.prompt_content) < 200:
            issues.append("Consider adding examples for better results")

        if issues:
            return {'issues': issues}

        return {'issues': []}

    def _unit_tests(self, exp: Experiment) -> Dict:
        """Run basic unit tests."""
        # Simulate unit test execution
        time.sleep(0.1)

        # Check if model is supported
        if exp.model not in self.model_profiles:
            raise Exception(f"Unsupported model: {exp.model}")

        return {'tests_passed': 5, 'tests_total': 5}

    def _offline_eval(
        self,
        exp: Experiment,
        test_dataset: List[Dict[str, str]]
    ) -> Tuple[float, float, float]:
        """
        Evaluate on test dataset.

        Returns:
            (accuracy_percent, latency_ms, cost_usd)
        """
        time.sleep(0.2)  # Simulate evaluation

        profile = self.model_profiles.get(exp.model, self.model_profiles['gpt-3.5-turbo'])

        accuracy = profile['accuracy'] * 100
        latency_min, latency_max = profile['latency_ms']
        latency = (latency_min + latency_max) / 2

        # Estimate cost
        avg_input_tokens = 150
        avg_output_tokens = 300
        num_calls = len(test_dataset) if test_dataset else 10

        input_cost = (avg_input_tokens / 1_000_000) * profile['input_cost_per_1m']
        output_cost = (avg_output_tokens / 1_000_000) * profile['output_cost_per_1m']
        total_cost = (input_cost + output_cost) * num_calls

        return accuracy, latency, total_cost

    def _deploy_to_staging(self, exp: Experiment) -> Dict:
        """Deploy to staging."""
        time.sleep(0.15)

        deployment = Deployment(
            version=exp.prompt_version,
            environment='staging',
            status=DeploymentStatus.HEALTHY
        )

        self.deployments[f"staging-{exp.id}"] = deployment
        return {'deployed': True, 'environment': 'staging'}

    def _e2e_tests(self, exp: Experiment) -> Dict:
        """Run end-to-end tests."""
        time.sleep(0.2)

        # Simulate E2E test results
        tests_passed = 8
        tests_total = 10

        if tests_passed < tests_total * 0.9:
            raise Exception(f"E2E tests failed: {tests_passed}/{tests_total} passed")

        return {'tests_passed': tests_passed, 'tests_total': tests_total}

    def _governance_check(self, exp: Experiment) -> Dict:
        """Run governance and compliance checks."""
        time.sleep(0.1)

        issues = []

        # Check cost threshold
        if exp.cost_usd > 0.10:
            issues.append(f"Cost exceeds threshold: ${exp.cost_usd:.4f} > $0.10")

        # Check model approval
        approved_models = list(self.model_profiles.keys())
        if exp.model not in approved_models:
            issues.append(f"Model not on approved list: {exp.model}")

        # Check for sensitive patterns
        sensitive_patterns = ['password', 'api_key', 'secret', 'token']
        for pattern in sensitive_patterns:
            if pattern in exp.prompt_content.lower():
                issues.append(f"Sensitive pattern detected: {pattern}")

        if issues:
            raise Exception(f"Governance check failed: {'; '.join(issues)}")

        return {'passed': True, 'issues': []}

    def _promote_to_production(self, exp: Experiment) -> Dict:
        """Promote to production."""
        time.sleep(0.25)

        deployment = Deployment(
            version=exp.prompt_version,
            environment='production',
            status=DeploymentStatus.HEALTHY
        )

        self.deployments[f"prod-{exp.id}"] = deployment
        return {'deployed': True, 'environment': 'production'}

    def detect_drift(
        self,
        deployment_id: str,
        recent_metrics: Dict[str, float]
    ) -> float:
        """
        Detect production performance drift.

        Args:
            deployment_id: Deployment to monitor
            recent_metrics: Current performance (accuracy, latency_ms, error_rate)

        Returns:
            drift_score (0-100, higher is worse)
        """
        if deployment_id not in self.deployments:
            return 0.0

        deployment = self.deployments[deployment_id]

        # Expected baseline (from offline eval)
        baseline_accuracy = 0.90
        baseline_latency = 800
        baseline_error_rate = 0.01

        drift_score = 0.0

        # Accuracy drift
        if 'accuracy' in recent_metrics:
            accuracy_delta = max(0, baseline_accuracy - recent_metrics['accuracy'])
            drift_score += accuracy_delta * 200  # Scale to 0-20

        # Latency drift
        if 'latency_ms' in recent_metrics:
            latency_delta = max(0, recent_metrics['latency_ms'] - baseline_latency)
            drift_score += min(30, latency_delta / 100)  # Cap at 30

        # Error rate drift
        if 'error_rate' in recent_metrics:
            error_delta = recent_metrics['error_rate'] - baseline_error_rate
            drift_score += max(0, error_delta * 200)  # Scale to 0-20

        drift_score = min(100, drift_score)

        # Update deployment status
        deployment.drift_score = drift_score

        if drift_score > 60:
            deployment.status = DeploymentStatus.DEGRADED
        elif drift_score > 80:
            deployment.status = DeploymentStatus.FAILED

        return drift_score


def example_1_create_experiment():
    """
    Example 1: Create and track experiment
    """
    print("=" * 60)
    print("Example 1: Create Experiment")
    print("=" * 60)

    pipeline = LLMOpsPipeline()

    # Create two competing experiments
    exp1 = pipeline.create_experiment(
        name="ACL Analysis - CoT Reasoning",
        prompt_version="v1.0",
        model="gpt-4",
        prompt_content="""Analyze this Cisco ACL for security issues.
Think step by step:
1. Check for dangerous permit any rules
2. Verify default deny exists
3. Assess rule ordering

ACL: {config}"""
    )

    exp2 = pipeline.create_experiment(
        name="ACL Analysis - Few-Shot",
        prompt_version="v1.1",
        model="claude-3-5-haiku",
        prompt_content="""Analyze this Cisco ACL:

Example 1: permit any any → UNSAFE
Example 2: deny ip any any → SAFE (default deny)

ACL: {config}"""
    )

    print(f"\n✅ Created Experiments:\n")
    print(f"1. {exp1.name}")
    print(f"   Model: {exp1.model}")
    print(f"   Version: {exp1.prompt_version}")
    print(f"   ID: {exp1.id}\n")

    print(f"2. {exp2.name}")
    print(f"   Model: {exp2.model}")
    print(f"   Version: {exp2.prompt_version}")
    print(f"   ID: {exp2.id}")

    print("\n" + "=" * 60 + "\n")
    return pipeline, [exp1, exp2]


def example_2_run_pipeline():
    """
    Example 2: Run CI/CD pipeline
    """
    print("=" * 60)
    print("Example 2: Run CI/CD Pipeline")
    print("=" * 60)

    pipeline = LLMOpsPipeline()

    exp = pipeline.create_experiment(
        name="BGP Diagnostics",
        prompt_version="v2.0",
        model="claude-3-sonnet",
        prompt_content="""Analyze BGP neighbor state and recommend actions.

Input: show ip bgp summary
Output: JSON with {status, issues[], recommendations[]}"""
    )

    # Test dataset
    test_cases = [
        {'input': 'neighbor 10.1.1.1 Active', 'expected': 'Check connectivity'},
        {'input': 'neighbor 10.1.1.2 Established', 'expected': 'Healthy'},
    ]

    print(f"\n🚀 Running pipeline for: {exp.name}\n")

    success, stages = pipeline.run_pipeline(exp, test_cases, deployment_target='staging')

    # Display results
    for stage in stages:
        emoji = {
            'success': '✅',
            'failed': '❌',
            'skipped': '⏭️ ',
            'pending': '⏸️ '
        }.get(stage.status.value, '⚪')

        print(f"{emoji} {stage.name} ({stage.status.value})")

        if stage.logs:
            for log in stage.logs:
                print(f"   {log}")

        if stage.duration_ms:
            print(f"   Duration: {stage.duration_ms}ms")

        print()

    print(f"Pipeline Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Final Status: {exp.status.value}")

    print("\n" + "=" * 60 + "\n")
    return pipeline, exp


def example_3_model_comparison():
    """
    Example 3: Compare models for cost/performance
    """
    print("=" * 60)
    print("Example 3: Model Selection Matrix")
    print("=" * 60)

    pipeline = LLMOpsPipeline()

    models_to_test = [
        ('gpt-3.5-turbo', 'Fast & Cheap'),
        ('gpt-4', 'Highest Accuracy'),
        ('claude-3-5-haiku', 'Balanced'),
    ]

    test_dataset = [{'input': 'test', 'expected': 'result'}] * 100  # 100 test cases

    print(f"\n📊 Comparing {len(models_to_test)} models:\n")

    results = []

    for model, description in models_to_test:
        exp = pipeline.create_experiment(
            name=f"Model Test - {model}",
            prompt_version="v1.0",
            model=model,
            prompt_content="Analyze network config: {input}"
        )

        accuracy, latency, cost = pipeline._offline_eval(exp, test_dataset)

        results.append({
            'model': model,
            'accuracy': accuracy,
            'latency': latency,
            'cost': cost
        })

        print(f"Model: {model}")
        print(f"  Description: {description}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Latency: {latency:.0f}ms")
        print(f"  Cost (100 calls): ${cost:.4f}")
        print()

    # Recommendation
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_cost = min(results, key=lambda x: x['cost'])
    best_balanced = sorted(results, key=lambda x: x['accuracy'] / (x['cost'] * 100 + 1), reverse=True)[0]

    print("💡 Recommendations:")
    print(f"  Highest Accuracy: {best_accuracy['model']} ({best_accuracy['accuracy']:.1f}%)")
    print(f"  Lowest Cost: {best_cost['model']} (${best_cost['cost']:.4f})")
    print(f"  Best Balance: {best_balanced['model']}")

    print("\n" + "=" * 60 + "\n")


def example_4_drift_detection():
    """
    Example 4: Detect production drift
    """
    print("=" * 60)
    print("Example 4: Production Drift Detection")
    print("=" * 60)

    pipeline = LLMOpsPipeline()

    # Create and deploy experiment
    exp = pipeline.create_experiment(
        name="Log Classifier",
        prompt_version="v1.5",
        model="gpt-4-turbo",
        prompt_content="Classify log severity: {log}"
    )

    pipeline._deploy_to_staging(exp)
    pipeline._promote_to_production(exp)

    deployment_id = f"prod-{exp.id}"

    print(f"\n📊 Monitoring deployment: {exp.prompt_version}\n")

    # Simulate drift over time
    time_periods = [
        ('Day 1', {'accuracy': 0.92, 'latency_ms': 750, 'error_rate': 0.01}),
        ('Day 7', {'accuracy': 0.90, 'latency_ms': 850, 'error_rate': 0.02}),
        ('Day 14', {'accuracy': 0.85, 'latency_ms': 1200, 'error_rate': 0.05}),
        ('Day 30', {'accuracy': 0.78, 'latency_ms': 1500, 'error_rate': 0.10}),
    ]

    for period, metrics in time_periods:
        drift = pipeline.detect_drift(deployment_id, metrics)

        status_emoji = {
            'healthy': '🟢',
            'degraded': '🟡',
            'failed': '🔴'
        }.get(pipeline.deployments[deployment_id].status.value, '⚪')

        print(f"{period}:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Latency: {metrics['latency_ms']:.0f}ms")
        print(f"  Error Rate: {metrics['error_rate']:.1%}")
        print(f"  {status_emoji} Drift Score: {drift:.1f}/100")
        print(f"  Status: {pipeline.deployments[deployment_id].status.value.upper()}")
        print()

    print("⚠️  Drift threshold exceeded!")
    print("Action: Retrain model or investigate data shift")

    print("\n" + "=" * 60 + "\n")


def example_5_governance_check():
    """
    Example 5: Governance and compliance validation
    """
    print("=" * 60)
    print("Example 5: Governance & Compliance Check")
    print("=" * 60)

    pipeline = LLMOpsPipeline()

    # Test 1: Valid experiment
    exp_valid = pipeline.create_experiment(
        name="Compliant Config Analyzer",
        prompt_version="v1.0",
        model="claude-3-5-haiku",
        prompt_content="Analyze configuration for compliance issues: {config}"
    )

    # Test 2: Expensive model
    exp_expensive = pipeline.create_experiment(
        name="Expensive Analyzer",
        prompt_version="v1.0",
        model="gpt-4",  # More expensive
        prompt_content="Analyze configuration: {config}"
    )

    # Test 3: Contains sensitive data
    exp_sensitive = pipeline.create_experiment(
        name="Insecure Prompt",
        prompt_version="v1.0",
        model="gpt-3.5-turbo",
        prompt_content="Use API_KEY abc123 and PASSWORD admin to analyze: {config}"
    )

    test_dataset = [{'input': 'test', 'expected': 'result'}] * 100

    experiments = [
        ('Valid', exp_valid),
        ('Expensive', exp_expensive),
        ('Sensitive Data', exp_sensitive),
    ]

    print("\n🔒 Running governance checks:\n")

    for name, exp in experiments:
        print(f"Test: {name}")

        try:
            # Run through eval to set costs
            pipeline._offline_eval(exp, test_dataset)

            # Run governance
            pipeline._governance_check(exp)

            print("  Result: ✅ PASSED")
        except Exception as e:
            print(f"  Result: ❌ FAILED")
            print(f"  Reason: {str(e)}")

        print()

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\n⚙️  Chapter 43: LLMOps Lifecycle & Pipeline")
    print("Production Operations for AI Systems\n")

    try:
        example_1_create_experiment()
        input("Press Enter to continue...")

        example_2_run_pipeline()
        input("Press Enter to continue...")

        example_3_model_comparison()
        input("Press Enter to continue...")

        example_4_drift_detection()
        input("Press Enter to continue...")

        example_5_governance_check()

        print("✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("- Automated pipelines catch issues before production")
        print("- Model selection involves cost/accuracy/latency trade-offs")
        print("- Drift detection prevents silent failures")
        print("- Governance ensures safety and compliance")
        print("- LLMOps is essential for production AI systems\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
