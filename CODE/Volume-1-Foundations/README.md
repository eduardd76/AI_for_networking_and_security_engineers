# Volume 1: Foundations - Code Repository

Complete, production-ready code examples from **AI for Networking Engineers: Volume 1 - Foundations**

## 📁 Repository Structure

```
Volume-1-Foundations/
├── Chapter-01-What-Is-Generative-AI/
│   ├── token_counter.py
│   ├── model_comparison.py
│   └── README.md
├── Chapter-02-Introduction-To-LLMs/
│   ├── basic_api_call.py
│   ├── model_capabilities.py
│   └── README.md
├── Chapter-03-Choosing-The-Right-Model/
│   ├── model_benchmarking.py
│   ├── cost_calculator.py
│   └── README.md
├── Chapter-04-API-Basics-Authentication/
│   ├── resilient_api_client.py
│   ├── rate_limiter.py
│   ├── usage_tracker.py
│   ├── secret_manager.py
│   └── README.md
├── Chapter-05-Prompt-Engineering/
│   ├── networking_prompts.py
│   ├── prompt_tester.py
│   ├── temperature_comparison.py
│   ├── few_shot_examples.py
│   └── README.md
├── Chapter-06-Structured-Outputs/
│   ├── json_validator.py
│   ├── pydantic_schemas.py
│   ├── config_parser.py
│   └── README.md
├── Chapter-07-Context-Management/
│   ├── context_chunker.py
│   ├── token_optimizer.py
│   ├── sliding_window.py
│   └── README.md
├── Chapter-08-Cost-Optimization/
│   ├── cost_analyzer.py
│   ├── caching_layer.py
│   ├── batch_processor.py
│   └── README.md
├── Chapter-09-Working-With-Network-Data/
│   ├── config_analyzer.py
│   ├── multi_vendor_parser.py
│   ├── log_processor.py
│   └── README.md
├── Chapter-10-API-Integration-Patterns/
│   ├── netmiko_integration.py
│   ├── napalm_integration.py
│   ├── ansible_integration.py
│   └── README.md
├── Chapter-11-Testing-And-Validation/
│   ├── test_framework.py
│   ├── accuracy_validator.py
│   ├── regression_tests.py
│   └── README.md
├── Chapter-12-Ethics-Responsible-AI/
│   ├── audit_logger.py
│   ├── approval_workflow.py
│   ├── guardrails.py
│   └── README.md
├── sample-data/
│   ├── configs/
│   ├── logs/
│   ├── outputs/
│   └── README.md
├── tests/
│   ├── test_api_client.py
│   ├── test_prompts.py
│   ├── test_parsers.py
│   └── README.md
├── .env.example
├── requirements.txt
├── setup.py
└── README.md (this file)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/eduardd76/AI_for_networking_and_security_engineers.git
cd AI_for_networking_and_security_engineers/Volume-1-Foundations

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# ANTHROPIC_API_KEY=sk-ant-api03-...
# OPENAI_API_KEY=sk-proj-...
# GOOGLE_API_KEY=AIzaSy...
```

### 3. Run Examples

```bash
# Test API connection
python Chapter-04-API-Basics-Authentication/resilient_api_client.py

# Try prompt engineering
python Chapter-05-Prompt-Engineering/networking_prompts.py

# Analyze a config
python Chapter-09-Working-With-Network-Data/config_analyzer.py sample-data/configs/router1.cfg
```

## 📖 Chapter Guide

### Part 1: Understanding LLMs (Chapters 1-3)

**Chapter 1: What Is Generative AI**
- Token counting and estimation
- Model comparison utilities
- Understanding transformer architecture

**Chapter 2: Introduction To LLMs**
- Basic API calls to Claude, GPT-4, Gemini
- Model capabilities testing
- Response format examples

**Chapter 3: Choosing The Right Model**
- Benchmark different models for networking tasks
- Cost vs. accuracy analysis
- Model selection framework

### Part 2: Core Skills (Chapters 4-6)

**Chapter 4: API Basics and Authentication**
- Production-ready API client with retry logic
- Rate limiting and backoff strategies
- Usage tracking and cost monitoring
- Secrets management

**Chapter 5: Prompt Engineering Fundamentals**
- Reusable prompt templates library
- Few-shot vs zero-shot examples
- Temperature and top-p exploration
- Prompt testing framework

**Chapter 6: Structured Outputs**
- JSON validation and parsing
- Pydantic schema validation
- Config data extraction
- Type-safe API responses

### Part 3: Production Scale (Chapters 7-9)

**Chapter 7: Context Management**
- Handling configs larger than context window
- Token-aware chunking strategies
- Sliding window processing
- Context optimization

**Chapter 8: Cost Optimization**
- Cost analysis and forecasting
- Response caching layer
- Batch processing for efficiency
- 50-70% cost reduction techniques

**Chapter 9: Working With Network Data**
- Multi-vendor config parsing (Cisco, Juniper, Arista)
- Log analysis and classification
- Security vulnerability detection
- Data normalization

### Part 4: Real World (Chapters 10-12)

**Chapter 10: API Integration Patterns**
- Netmiko integration (SSH automation)
- NAPALM integration (multi-vendor)
- Ansible playbook integration
- Real device interaction

**Chapter 11: Testing and Validation**
- Comprehensive test framework
- Accuracy validation suite
- Regression testing
- Performance benchmarking

**Chapter 12: Ethics and Responsible AI**
- Audit logging for compliance
- Human approval workflows
- Guardrails and safety checks
- Bias detection

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api_client.py -v

# Run with coverage
pytest --cov=. tests/
```

## 📊 Sample Data

The `sample-data/` directory contains:

**Configs**: Sample network device configurations
- `cisco_ios_router.cfg` - Cisco IOS router
- `cisco_nexus_switch.cfg` - Cisco Nexus switch
- `juniper_mx.cfg` - Juniper MX router
- `arista_eos.cfg` - Arista EOS switch

**Logs**: Sample network logs for analysis
- `syslog_samples.log` - Various syslog messages
- `bgp_neighbor_logs.log` - BGP session logs
- `interface_errors.log` - Interface error logs

**Outputs**: Expected AI responses for testing

## 💰 Cost Estimates

**Running all examples once**:
- API calls: ~500 requests
- Total tokens: ~1.5M
- Estimated cost: $8-12 USD

**Learning exercises (per chapter)**:
- API calls: ~50 requests
- Total tokens: ~150K
- Estimated cost: $0.80-1.20 USD

**Tip**: Use Claude Haiku for development/testing (10x cheaper than Sonnet)

## 🔧 Configuration

### Environment Variables

All examples use environment variables for configuration:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional (for multi-provider examples)
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIzaSy...

# Optional (for logging and monitoring)
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_FILE=usage_metrics.json
```

### API Client Configuration

The `ResilientAPIClient` supports these parameters:

```python
client = ResilientAPIClient(
    api_key=None,           # Defaults to ANTHROPIC_API_KEY env var
    max_retries=3,          # Number of retry attempts
    initial_retry_delay=1.0, # Initial backoff delay (seconds)
    timeout=60              # Request timeout (seconds)
)
```

## 📚 Dependencies

Core dependencies:
- `anthropic>=0.18.0` - Claude API client
- `openai>=1.12.0` - OpenAI API client (GPT-4)
- `google-generativeai>=0.3.0` - Google Gemini API
- `python-dotenv>=1.0.0` - Environment variable management
- `pydantic>=2.5.0` - Data validation
- `requests>=2.31.0` - HTTP client

Network automation:
- `netmiko>=4.3.0` - Multi-vendor SSH automation
- `napalm>=4.1.0` - Multi-vendor network automation
- `textfsm>=1.1.3` - Structured text parsing

Testing and validation:
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### "Invalid API Key"
- Check that `.env` file exists and contains valid key
- Verify key starts with `sk-ant-api03-` for Anthropic
- Ensure `.env` is in the same directory or use `load_dotenv()` with path

### "Rate Limit Exceeded"
- Free tier has low limits (5 RPM for Anthropic)
- Use built-in rate limiting: `RateLimiter(max_requests=5, time_window=60)`
- Consider upgrading to paid tier

### "Context Length Exceeded"
- Config/log file too large for model's context window
- Use chunking: `Chapter-07-Context-Management/context_chunker.py`
- Consider using models with larger context (Claude Opus: 200K, Gemini 1.5 Pro: 2M)

### "Import Error"
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.10+)

## 📖 Additional Resources

**Book**: AI for Networking Engineers - Volume 1: Foundations
**Author**: Eduard Dulharu
**Website**: https://vexpertai.com
**Discord**: [Join the community - link in book]
**GitHub**: https://github.com/eduardd76/AI_for_networking_and_security_engineers

## 📝 License

MIT License - See LICENSE file for details

Code examples are free to use in your projects. Attribution appreciated but not required.

## 🤝 Contributing

Found a bug? Have an improvement?

1. Open an issue describing the problem
2. Submit a pull request with fixes
3. Share your use cases in Discord

## ⚠️ Disclaimer

These examples are for educational purposes. Always test thoroughly before using in production networks. The author is not responsible for any network outages, security incidents, or costs incurred from using this code.

## 📞 Support

- **GitHub Issues**: Technical problems and bugs
- **Discord Community**: Questions and discussions
- **Email**: ed@vexpertai.com (Consulting inquiries only)

---

**Version**: 1.0.0
**Last Updated**: January 2026
**Tested With**: Python 3.10+, Claude Sonnet 4.5, GPT-4, Gemini 1.5 Pro

**Ready to start?** Begin with Chapter 4: API Basics and Authentication
