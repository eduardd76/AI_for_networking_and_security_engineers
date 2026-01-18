# AI for Networking and Security Engineers - Code Repository

**Complete, production-ready code from the book series by Eduard Dulharu (Ed Harmoosh)**

This repository contains all working code examples, libraries, and tools from the "AI for Networking Engineers" book series. Every code file is tested, documented, and ready to use in your network automation projects.

## 📚 Book Volumes

### Volume 1: Foundations ✅ COMPLETE
**12 chapters covering core AI/LLM fundamentals for networking**

- API clients with retry logic and rate limiting
- Prompt engineering templates library
- Multi-vendor config parsing (Cisco, Juniper, Arista)
- Cost optimization and caching layers
- Testing frameworks and validation
- Ethics, approval workflows, and guardrails

👉 [**Go to Volume 1 →**](./Volume-1-Foundations/)

### Volume 2: Advanced Applications (Coming Soon)
**Deep dive into specialized AI applications**

- RAG (Retrieval Augmented Generation) for documentation
- Vector databases and semantic search
- Custom fine-tuning for network tasks
- Multi-agent orchestration
- Advanced troubleshooting automation

### Volume 3: Production Deployment (Coming Soon)
**Enterprise-grade deployment patterns**

- Kubernetes deployment
- Monitoring and observability
- Security hardening
- CI/CD pipelines
- High-availability architectures

### Volume 4: Security Focus (Coming Soon)
**AI-powered security operations**

- Threat detection and analysis
- Security policy generation
- Incident response automation
- Compliance monitoring
- Anomaly detection

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (Tested on 3.10, 3.11, 3.12)
- **API Keys**: At least one of:
  - Anthropic (Claude): Recommended for networking tasks
  - OpenAI (GPT-4): Alternative option
  - Google (Gemini): Free tier available
- **Optional**: Network devices or simulators (GNS3, EVE-NG) for integration examples

### Installation

```bash
# Clone repository
git clone https://github.com/eduardd76/AI_for_networking_and_security_engineers.git
cd AI_for_networking_and_security_engineers

# Navigate to Volume 1
cd CODE/Volume-1-Foundations

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### First Example

```bash
# Test your API connection
python Chapter-04-API-Basics-Authentication/resilient_api_client.py

# Try network config analysis
python Chapter-09-Working-With-Network-Data/config_analyzer.py
```

## 🎯 Who Is This For?

**Network Engineers** who want to:
- Automate repetitive tasks with AI
- Build intelligent troubleshooting tools
- Analyze configs and logs at scale
- Generate documentation automatically
- Stay ahead of industry trends

**Requirements**:
- Basic Python knowledge (variables, functions, loops)
- Network fundamentals (CCNA level or equivalent)
- Curiosity and willingness to experiment

**NOT required**:
- Machine learning expertise
- Deep math/statistics background
- PhD in computer science

## 📖 Learning Path

### Beginner Track (Start Here)
1. **Chapter 4**: API Basics - Learn to call AI models
2. **Chapter 5**: Prompt Engineering - Write effective prompts
3. **Chapter 9**: Network Data - Parse configs and logs
4. **Chapter 11**: Testing - Validate AI outputs

**Time**: 4-6 hours | **Cost**: ~$5 in API calls

### Intermediate Track
1. **Chapter 6**: Structured Outputs - Type-safe data extraction
2. **Chapter 7**: Context Management - Handle large files
3. **Chapter 8**: Cost Optimization - Reduce API costs 50-70%
4. **Chapter 10**: Integrations - Connect to Netmiko, NAPALM, Ansible

**Time**: 8-12 hours | **Cost**: ~$10 in API calls

### Advanced Track
1. **Chapter 12**: Ethics & Guardrails - Production safety
2. **Chapter 11**: Regression Testing - Quality assurance
3. Build your own AI-powered tools
4. Deploy to production

**Time**: 12-20 hours | **Cost**: ~$15 in API calls

## 💡 Real-World Use Cases

### Configuration Management
```python
from Chapter_09_Working_With_Network_Data.config_analyzer import ConfigAnalyzer

analyzer = ConfigAnalyzer()
findings = analyzer.analyze_config(router_config, device_type="cisco_ios")

for finding in findings['findings']:
    if finding['severity'] == 'critical':
        print(f"⚠️ {finding['title']}: {finding['recommendation']}")
```

### Log Analysis
```python
from Chapter_09_Working_With_Network_Data.log_processor import LogProcessor

processor = LogProcessor()
processor.parse_log_file("syslog.txt")
analysis = processor.analyze_logs()

print(f"Critical events: {analysis['summary']['critical_count']}")
print(f"Interface issues: {len(analysis['interface_issues'])}")
```

### Cost Optimization
```python
from Chapter_08_Cost_Optimization.caching_layer import CachedAPIClient

# Enable caching for 50-70% cost reduction
client = CachedAPIClient(ttl_seconds=3600)

# Repeated calls use cache
result1 = client.call("Analyze this config...")  # API call
result2 = client.call("Analyze this config...")  # Cached (free!)
```

### Safety Guardrails
```python
from Chapter_12_Ethics_Responsible_AI.guardrails import SafetyGuardrails

guardrails = SafetyGuardrails()
result = guardrails.validate_change(
    proposed_config=ai_generated_config,
    device_info={"hostname": "core-rtr-01", "role": "core_router"}
)

if result.safe:
    apply_config(ai_generated_config)
else:
    print(f"🔴 Blocked: {result.violations[0].message}")
```

## 🏗️ Repository Structure

```
CODE/
├── Volume-1-Foundations/          # Chapters 1-12 ✅
│   ├── Chapter-04-API-Basics-Authentication/
│   ├── Chapter-05-Prompt-Engineering/
│   ├── Chapter-06-Structured-Outputs/
│   ├── Chapter-07-Context-Management/
│   ├── Chapter-08-Cost-Optimization/
│   ├── Chapter-09-Working-With-Network-Data/
│   ├── Chapter-10-API-Integration-Patterns/
│   ├── Chapter-11-Testing-And-Validation/
│   ├── Chapter-12-Ethics-Responsible-AI/
│   ├── sample-data/
│   ├── tests/
│   └── README.md
│
├── Volume-2-Advanced-Applications/ # Coming soon
├── Volume-3-Production-Deployment/ # Coming soon
├── Volume-4-Security-Focus/        # Coming soon
└── README.md                       # This file
```

## 💰 Cost Management

### Expected Costs

| Activity | API Calls | Tokens | Cost (USD) |
|----------|-----------|--------|------------|
| Run all Volume 1 examples | ~500 | ~1.5M | $8-12 |
| Chapter exercises | ~50 | ~150K | $0.80-1.20 |
| Build 1 production tool | ~200 | ~600K | $3-5 |

### Cost Reduction Tips

1. **Use Haiku for development** (10x cheaper than Sonnet)
   ```python
   client.call(prompt, model="claude-3-5-haiku-20241022")
   ```

2. **Enable caching** (Chapter 8)
   ```python
   from Chapter_08_Cost_Optimization.caching_layer import CachedAPIClient
   client = CachedAPIClient(ttl_seconds=3600)
   ```

3. **Batch processing** (Chapter 8)
   ```python
   from Chapter_08_Cost_Optimization.batch_processor import BatchProcessor
   processor = BatchProcessor(max_workers=5)
   results = processor.process_batch(configs, analyze_func)
   ```

4. **Set daily budgets** (Chapter 4)
   ```python
   from Chapter_04_API_Basics_Authentication.usage_tracker import UsageTracker
   tracker = UsageTracker(daily_limit=5.0)  # $5/day limit
   ```

## 🧪 Testing

All code is tested and includes working examples:

```bash
# Run specific example
cd Volume-1-Foundations
python Chapter-05-Prompt-Engineering/networking_prompts.py

# Run tests
pytest tests/ -v

# Check test coverage
pytest --cov=. tests/
```

## 🔧 Dependencies

### Core AI APIs
- `anthropic>=0.18.0` - Claude (Recommended)
- `openai>=1.12.0` - GPT-4
- `google-generativeai>=0.3.0` - Gemini

### Network Automation
- `netmiko>=4.3.0` - Multi-vendor SSH
- `napalm>=4.1.0` - Multi-vendor config management
- `textfsm>=1.1.3` - Structured text parsing

### Data Processing
- `pydantic>=2.5.0` - Data validation
- `python-dotenv>=1.0.0` - Environment variables

See `requirements.txt` in each volume for complete list.

## 🐛 Common Issues

### "Invalid API Key"
**Solution**: Check `.env` file contains valid key
```bash
# Anthropic key format
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Test with
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY')[:20])"
```

### "Rate Limit Exceeded"
**Solution**: Use built-in rate limiting
```python
from Chapter_04_API_Basics_Authentication.rate_limiter import RateLimiter
limiter = RateLimiter(max_requests=5, time_window=60)
```

### "Context Length Exceeded"
**Solution**: Use context chunking
```python
from Chapter_07_Context_Management.context_chunker import ContextChunker
chunker = ContextChunker(max_tokens=4000)
chunks = chunker.chunk_by_size(large_config)
```

### "Import Error"
**Solution**: Activate virtual environment
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 📚 Additional Resources

- **Book**: Purchase at [vexpertai.com](https://vexpertai.com)
- **Author**: Eduard Dulharu (Ed Harmoosh)
  - LinkedIn: [Eduard Dulharu](https://www.linkedin.com/in/eduarddulharu/)
  - Email: ed@vexpertai.com
- **Company**: vExpertAI GmbH (Munich, Germany)
- **Discord**: Join the community (link in book)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Bug Reports

Open an issue with:
- Python version
- OS (Windows/Linux/Mac)
- Error message
- Steps to reproduce

## 📄 License

MIT License - See LICENSE file

**TL;DR**: Free to use in your projects. Attribution appreciated but not required.

## ⚠️ Disclaimer

**Educational Use**: Code examples are for learning and development. Test thoroughly before production deployment.

**No Warranty**: Author not responsible for network outages, security incidents, or API costs from using this code.

**API Costs**: You are responsible for API usage charges from Anthropic, OpenAI, or Google.

**Network Changes**: Always have a rollback plan when applying AI-generated configurations.

## 🎓 About the Author

**Eduard Dulharu (Ed Harmoosh)**
- 20+ years in network infrastructure (AT&T, Infosys, Kyndryl)
- CTO & Founder, vExpertAI GmbH
- CCNA to CCIE level experience
- Bridging AI and networking for enterprise operations

## 📊 Project Stats

- **Total Lines of Code**: ~15,000+
- **Python Files**: 40+
- **Chapters Covered**: 12 (Volume 1)
- **Test Coverage**: 85%+
- **Production Ready**: Yes

## 🔜 Coming Soon

- **Volume 2**: Advanced Applications (Q2 2026)
- **Volume 3**: Production Deployment (Q3 2026)
- **Volume 4**: Security Focus (Q4 2026)
- **Video Course**: Hands-on tutorials
- **Certification Program**: AI for Network Engineers

## 📞 Support

- **Technical Issues**: [GitHub Issues](https://github.com/eduardd76/AI_for_networking_and_security_engineers/issues)
- **Community**: Discord (link in book)
- **Consulting**: ed@vexpertai.com

---

**Version**: 1.0.0
**Last Updated**: January 2026
**Python**: 3.10+
**Status**: Volume 1 Complete ✅

**Ready to start?** → [Begin with Volume 1](./Volume-1-Foundations/)
