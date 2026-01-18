# Chapter 32: Fine-Tuning Models for Network Data

Production-ready fine-tuning toolkit for network engineering tasks.

## Overview

This module provides a complete toolkit for deciding whether to fine-tune an LLM for network operations, creating training data, validating quality, and calculating ROI.

## Features

### 1. Training Data Creation (`TrainingDataCreator`)
- Convert troubleshooting tickets to training examples
- Generate training data from config standards
- Create examples from log analysis
- Support for multiple task types

### 2. Data Validation (`DataValidator`)
- Remove duplicate examples
- Check for incomplete data
- PII detection (emails, SSNs, credit cards, etc.)
- Format validation
- Quality metrics

### 3. Quality Assessment (`QualityAssessor`)
- LLM-powered quality evaluation
- Consistency scoring
- Accuracy assessment
- Diversity analysis
- Actionable recommendations

### 4. Cost Calculation (`FineTuningCostCalculator`)
- Training cost estimation
- Token-level cost breakdown
- Multiple scenario comparison
- Break-even analysis

### 5. ROI Analysis (`ROIAnalyzer`)
- LLM-powered recommendations
- Should you fine-tune or not?
- Alternative approaches
- Expected accuracy improvements

## File Structure

```
fine_tuning.py (1,248 lines)
├── Data Structures (150 lines)
│   ├── TaskType (enum)
│   ├── DataQuality (enum)
│   ├── TrainingExample (dataclass)
│   ├── ValidationResult (dataclass)
│   └── CostAnalysis (dataclass)
│
├── Pydantic Models (35 lines)
│   ├── QualityIssue
│   ├── DataQualityAssessment
│   └── FineTuningRecommendation
│
├── Core Classes (400 lines)
│   ├── TrainingDataCreator (125 lines)
│   ├── DataValidator (135 lines)
│   ├── QualityAssessor (75 lines)
│   ├── FineTuningCostCalculator (60 lines)
│   └── ROIAnalyzer (65 lines)
│
└── Examples (550 lines)
    ├── example_1_create_training_dataset() (150 lines)
    ├── example_2_validate_and_clean() (75 lines)
    ├── example_3_roi_analysis() (85 lines)
    ├── example_4_cost_calculator() (75 lines)
    └── example_5_fine_tuning_recommendation() (125 lines)
```

## Requirements

```bash
pip install python-dotenv langchain-anthropic pydantic
```

Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Run All Examples

```bash
python fine_tuning.py
```

This runs all 5 examples sequentially (with pauses between).

### Import as Module

```python
from fine_tuning import (
    TrainingDataCreator,
    DataValidator,
    QualityAssessor,
    FineTuningCostCalculator,
    ROIAnalyzer
)

# Create training data
creator = TrainingDataCreator()
creator.add_from_ticket(
    problem="BGP neighbor down",
    diagnosis="Timer mismatch",
    resolution="Fixed timers"
)

# Validate
validator = DataValidator()
cleaned, result = validator.validate_and_clean(creator.examples)

# Calculate costs
calculator = FineTuningCostCalculator()
analysis = calculator.create_cost_analysis(
    training_tokens=500_000,
    monthly_requests=50_000,
    prompt_tokens_without_ft=2_500,
    prompt_tokens_with_ft=500
)

# Get ROI
roi = analysis.calculate_roi()
print(f"Break-even: {roi['breakeven_days']:.0f} days")
```

## Examples Explained

### Example 1: Create Training Dataset
Demonstrates converting real network operations data into training format:
- 5 troubleshooting tickets
- 3 config generation examples
- 3 log analysis examples

**Output**: TrainingDataCreator with statistics

### Example 2: Validate and Clean
Shows comprehensive data validation:
- Removes duplicates
- Filters incomplete examples
- Detects PII (emails, IPs, SSNs)
- Validates format

**Output**: Cleaned dataset + validation metrics

### Example 3: ROI Analysis
Compares high-volume vs low-volume scenarios:
- **High-volume**: 50K requests/month → Break-even in 20 days
- **Low-volume**: 1K requests/month → Break-even in 1000 days

**Output**: Detailed financial analysis with recommendations

### Example 4: Cost Calculator
Token-level cost breakdown:
- Training costs for different dataset sizes
- Monthly API costs by volume
- Break-even analysis table

**Output**: Cost comparison tables

### Example 5: LLM Recommendation
Uses Claude to provide expert recommendation:
- Analyzes cost/benefit
- Considers current vs target accuracy
- Evaluates data quality
- Suggests alternatives

**Output**: Structured recommendation from LLM

## Key Classes

### TrainingDataCreator
```python
creator = TrainingDataCreator()

# From ticket
creator.add_from_ticket(problem, diagnosis, resolution)

# From config
creator.add_from_config_pair(request, config, explanation)

# From log
creator.add_from_log_analysis(log_entry, classification, severity, action)

# Stats
stats = creator.get_statistics()
```

### DataValidator
```python
validator = DataValidator()
cleaned, result = validator.validate_and_clean(examples)

print(f"Success rate: {result.success_rate:.1f}%")
print(f"Duplicates removed: {result.duplicates_removed}")
print(f"PII removed: {result.pii_removed}")
```

### CostAnalysis
```python
analysis = CostAnalysis(
    training_tokens=500_000,
    fine_tuning_cost=15.0,
    monthly_requests=50_000,
    tokens_without_finetuning=3_000,
    tokens_with_finetuning=1_000
)

roi = analysis.calculate_roi()
# Returns: breakeven_months, annual_savings, roi_percentage, recommendation
```

## Design Patterns

### 1. Dataclasses for Data Structures
- `TrainingExample`: Single training example with metadata
- `ValidationResult`: Validation metrics
- `CostAnalysis`: Cost/ROI calculations

### 2. Pydantic for LLM Outputs
- `DataQualityAssessment`: Structured quality evaluation
- `FineTuningRecommendation`: Expert recommendation
- Type-safe, validated outputs

### 3. Enums for Constants
- `TaskType`: Training task categories
- `DataQuality`: Quality levels

### 4. LangChain Integration
- ChatAnthropic for Claude API
- PydanticOutputParser for structured outputs
- PromptTemplate for reusable prompts

## Real Network Engineering Scenarios

All examples use realistic network operations data:
- **Tickets**: BGP neighbor failures, OSPF adjacency issues, VoIP quality problems
- **Configs**: Access ports with security, trunk configs, BGP peering
- **Logs**: Syslog entries with proper classification and severity

## Production-Ready Features

1. **PII Detection**: Regex patterns for SSNs, emails, credit cards, IPs
2. **Error Handling**: Comprehensive validation and error reporting
3. **Metrics**: Token counts, success rates, cost breakdowns
4. **Documentation**: Detailed docstrings for all functions
5. **Modularity**: Can import and use individual components

## Cost Estimates (Reference)

Based on typical Claude pricing:
- **Input tokens**: $0.003 per 1K
- **Output tokens**: $0.015 per 1K
- **Fine-tuning**: ~$10 per 1M training tokens (3 epochs)

Example scenario:
- 1,000 training examples × 400 tokens = 400K tokens
- Training cost: 400K × 3 epochs × $10/1M = **$12**
- Monthly savings at 50K requests: **$300/month**
- Break-even: **1.2 days**

## Decision Framework

### Fine-tune when:
- ✅ High volume (>10K requests/month)
- ✅ Repetitive context in every prompt
- ✅ Accuracy below target (<90%)
- ✅ Cost-sensitive application
- ✅ Break-even <3 months

### Don't fine-tune when:
- ❌ Low volume (<1K requests/month)
- ❌ Frequently changing requirements
- ❌ Already hitting accuracy targets
- ❌ Limited training data (<500 examples)
- ❌ Break-even >6 months

## Next Steps

After running this module:
1. Use `TrainingDataCreator` to build your dataset
2. Validate with `DataValidator` (aim for >95% pass rate)
3. Assess quality with `QualityAssessor` (aim for "good" or "excellent")
4. Calculate ROI with `FineTuningCostCalculator`
5. Get recommendation with `ROIAnalyzer`
6. If recommended, export to JSONL and fine-tune via AWS Bedrock or OpenAI

## Reference Content

Based on: `E:\vExpertAI\CONTENT\Book\AI-for-Networking-Engineers\Part3-Advanced-Techniques\Chapter-32-Fine-Tuning-Models-for-Network-Data.md`

## Author

Eduard Dulharu
CTO & Founder, vExpertAI GmbH
Munich, Germany
