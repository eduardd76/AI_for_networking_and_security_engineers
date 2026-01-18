# Volume 1: Foundations - Code Repository

**Simple, practical code examples for network engineers learning AI**

From: **AI for Networking Engineers - Volume 1: Foundations**

## 📁 What's Included

### Chapter 4: API Basics
**Learn to call Claude API using LangChain**

- `api_basics.py` - 7 practical examples

**Examples:**
1. Simple API call
2. API call with context (config analysis)
3. Streaming responses
4. Error handling
5. Temperature settings (creativity control)
6. Cost tracking
7. Model comparison (Haiku vs Sonnet)

**Test it:**
```bash
cd Chapter-04-API-Basics-Authentication
python api_basics.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 5: Prompt Engineering
**Write better prompts for better results**

- `prompt_engineering.py` - 7 techniques

**Techniques:**
1. Basic vs detailed prompts
2. Role-based prompting
3. Few-shot learning (show examples)
4. Chain of thought reasoning
5. Adding constraints
6. Prompt templates
7. Negative prompting (tell it what NOT to do)

**Test it:**
```bash
cd Chapter-05-Prompt-Engineering
python prompt_engineering.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 6: Structured Outputs
**Get JSON instead of text using Pydantic**

- `structured_outputs.py` - 4 examples with validation

**What you'll learn:**
- Parse interface configs into objects
- Security issue detection with structured output
- Device inventory extraction
- Routing table parsing

**Test it:**
```bash
cd Chapter-06-Structured-Outputs
python structured_outputs.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 7: Context Management
**Handle large configs that exceed token limits**

- `context_management.py` - 4 strategies

**Strategies:**
- Chunking large configs
- Analyze chunks separately
- Map-reduce pattern (summarize then combine)
- Sliding window for context

**Test it:**
```bash
cd Chapter-07-Context-Management
python context_management.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 8: Cost Optimization
**Keep your AI costs under control**

- `cost_optimization.py` - 4 techniques

**Cost savers:**
- Caching (avoid repeat API calls)
- Use cheaper models (Haiku vs Sonnet)
- Batch processing (process multiple items in one call)
- Prompt optimization (shorter = cheaper)

**Typical savings: 50-70%**

**Test it:**
```bash
cd Chapter-08-Cost-Optimization
python cost_optimization.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 9: Working with Network Data
**Process real configs, logs, and command outputs**

- `network_data.py` - 4 examples

**Examples:**
- Config analysis (identify device role, topology)
- Log analysis (troubleshoot from syslogs)
- Multi-vendor support (Cisco, Juniper)
- Parse show commands

**Test it:**
```bash
cd Chapter-09-Working-With-Network-Data
python network_data.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 10: API Integration
**Combine AI with Netmiko, NAPALM, Ansible**

- `api_integration.py` - 3 patterns

**Integration patterns:**
- Generate device commands with AI
- Interpret device errors
- Validate commands before execution

**Test it:**
```bash
cd Chapter-10-API-Integration-Patterns
python api_integration.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 11: Testing and Validation
**Ensure AI outputs are accurate**

- `testing.py` - 4 test types

**Testing strategies:**
- Accuracy testing (known correct answers)
- Consistency testing (same answer each time)
- Output validation (check format)
- Regression testing (no breaking changes)

**Test it:**
```bash
cd Chapter-11-Testing-And-Validation
python testing.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 12: Ethics and Responsible AI
**Safe AI use in production networks**

- `ethics.py` - 5 safety practices

**Safety features:**
- Command guardrails (block dangerous commands)
- Approval workflows (human review)
- Audit logging (track all AI interactions)
- Read-only by default
- Change verification (before and after)

**Test it:**
```bash
cd Chapter-12-Ethics-Responsible-AI
python ethics.py  # Requires ANTHROPIC_API_KEY
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Setup API Keys

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

Get your key at: https://console.anthropic.com/

### 3. Run Your First Example

```bash
cd Chapter-04-API-Basics-Authentication
python api_basics.py
```

## 📦 Dependencies

### Core Requirements
- Python 3.10+
- `langchain` - LLM application framework
- `langchain-anthropic` - Anthropic/Claude integration
- `anthropic` - Official Anthropic SDK
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation

All dependencies are in `requirements.txt`.

## 💡 Key Design Principles

### Simplified for Network Engineers
Unlike the previous version, this rewrite focuses on:
- **One file per chapter** - Easy to find and run
- **LangChain-based** - Modern framework approach
- **Practical examples** - Real network scenarios
- **Simple and clean** - ~200-300 lines per file
- **Easy to understand** - Clear, readable code

### Real Network Use Cases
Every example uses actual networking scenarios:
- Config analysis and generation
- Security checking
- Log troubleshooting
- Multi-vendor support
- Cost optimization

## 🎓 Learning Path

### Beginner (Start Here)
1. **API Basics** (Ch 4) - Learn to call AI APIs
2. **Prompt Engineering** (Ch 5) - Write better prompts
3. **Structured Outputs** (Ch 6) - Get JSON responses

### Intermediate
4. **Context Management** (Ch 7) - Handle large configs
5. **Cost Optimization** (Ch 8) - Save money
6. **Network Data** (Ch 9) - Process real data

### Advanced
7. **API Integration** (Ch 10) - Combine with Netmiko
8. **Testing** (Ch 11) - Validate AI outputs
9. **Ethics** (Ch 12) - Safe production use

## 🔧 Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "Invalid API key"
Check your `.env` file:
```bash
# Make sure key is set correctly
cat .env

# Test connection
python -c "from langchain_anthropic import ChatAnthropic; print('OK')"
```

### "Rate limit exceeded"
Free tier limits: 5 requests/minute

**Solution**: Wait 60 seconds or upgrade to paid tier.

## 💰 Cost Estimates

### Per Example Run
- Chapter 4 (7 examples): **$0.05-0.10**
- Chapter 5 (7 examples): **$0.05-0.10**
- Chapter 6 (4 examples): **$0.03-0.05**
- Chapters 7-12 (each): **$0.02-0.05**

### Full Testing
Running all chapters once: **~$0.30-0.50**

### Tips to Save Money
1. Use Haiku model for simple tasks (10x cheaper)
2. Cache repeated queries (Chapter 8)
3. Batch process multiple items (Chapter 8)
4. Keep prompts concise

## 📚 What's Different from Volume 2?

### Volume 1 (Foundations)
- **Focus**: Core AI concepts and APIs
- **Style**: Direct API usage with LangChain
- **Examples**: Prompt engineering, cost optimization, testing
- **Audience**: Learning the basics

### Volume 2 (Advanced Applications)
- **Focus**: Production RAG systems and agents
- **Style**: LangGraph agents, vector databases
- **Examples**: Semantic search, troubleshooting agents, log analysis
- **Audience**: Building real systems

**Start with Volume 1, then move to Volume 2.**

## 🤝 Contributing

Found issues? Have improvements?

1. Test the code
2. Document the issue
3. Submit a pull request
4. Share your use case!

## 📞 Support

- **Technical Issues**: GitHub Issues
- **Questions**: Discord community
- **Consulting**: ed@vexpertai.com

---

**Author**: Eduard Dulharu
**Company**: vExpertAI GmbH (Munich, Germany)
**Version**: 1.1.0 (Simplified Rewrite)
**Last Updated**: January 2026
**Status**: All Core Chapters Complete

**Total**: 9 Python files, ~1,800 lines of clean, simple code

**Ready to learn AI?** Start with Chapter 4 → `api_basics.py`
