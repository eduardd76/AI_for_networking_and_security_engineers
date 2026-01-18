# Volume 2: Advanced Applications - Code Repository

Production-ready code examples for **AI for Networking Engineers: Volume 2 - Advanced Applications**

Focus: RAG systems, LangChain, LangGraph agents, and semantic search for network operations.

## 📁 What's Included

### Chapter 14: RAG Fundamentals
**Simple, working RAG implementation for network documentation**

- `simple_rag.py` - Complete RAG system using LangChain + ChromaDB
- `document_loader.py` - Load docs from multiple formats (.md, .txt, .cfg)

**What you'll learn:**
- Build a knowledge base from network docs
- Search documentation semantically
- Answer questions using RAG
- Simple, clean code (< 200 lines)

**Test it:**
```bash
cd Chapter-14-RAG-Fundamentals
python document_loader.py  # Test without API keys
python simple_rag.py        # Requires ANTHROPIC_API_KEY + OPENAI_API_KEY
```

### Chapter 15: LangChain Integration
**Practical LangChain examples for network engineers**

- `langchain_basics.py` - 4 complete working examples

**Examples included:**
1. Simple chain - Analyze configs
2. Structured output - Get JSON responses
3. Chat-style prompts - Natural interaction
4. Config generation - Generate from requirements

**Test it:**
```bash
cd Chapter-15-LangChain-Integration
python langchain_basics.py  # Requires ANTHROPIC_API_KEY
```

### Chapter 19: Agent Architecture
**Build autonomous agents with LangGraph**

- `simple_agent.py` - Network analysis agent with tools

**What the agent can do:**
- Retrieve config sections
- Check for security issues
- Use multiple tools autonomously
- Simple state machine logic

**Test it:**
```bash
cd Chapter-19-Agent-Architecture
python simple_agent.py  # Requires ANTHROPIC_API_KEY
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

Edit `.env` and add your keys:
```bash
# Required for most examples
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Required for RAG examples (embeddings)
OPENAI_API_KEY=sk-proj-xxxxx

# Optional - for monitoring with LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxx
LANGCHAIN_PROJECT=networking-ai
```

### 3. Test Without API Keys

Start with the document loader (no API needed):
```bash
cd Chapter-14-RAG-Fundamentals
python document_loader.py
```

### 4. Run Full Examples

Once you have API keys:
```bash
# RAG system
cd Chapter-14-RAG-Fundamentals
python simple_rag.py

# LangChain basics
cd ../Chapter-15-LangChain-Integration
python langchain_basics.py

# Agent
cd ../Chapter-19-Agent-Architecture
python simple_agent.py
```

## 📦 Dependencies

### Core Requirements
- Python 3.10+
- `anthropic` - Claude API
- `openai` - For embeddings (text-embedding-3-small)
- `langchain` - LLM application framework
- `langchain-anthropic` - Anthropic integration
- `langchain-openai` - OpenAI integration
- `langgraph` - Build agents with state machines
- `langsmith` - Optional monitoring/tracing

### Vector Database
- `chromadb` - Local vector database (simple, fast)
- `faiss-cpu` - Facebook's similarity search (alternative)

### Document Processing
- `pypdf` - PDF support
- `python-docx` - Word docs
- `markdown` - Markdown files

## 💡 Key Design Principles

### Simplicity First
Unlike Volume 1, Volume 2 code prioritizes:
- **Readability** over features
- **Clarity** over cleverness
- **Working examples** over abstract frameworks

### Real Network Use Cases
Every example uses actual networking scenarios:
- Config analysis
- Security checking
- Documentation search
- BGP troubleshooting
- VLAN management

### Production Patterns
While simple, code follows production best practices:
- Environment variables for secrets
- Error handling
- Type hints
- Clear documentation
- Testable components

## 🧪 Testing Guide

### Test Checklist

**Without API Keys:**
- ✅ `document_loader.py` - Loads sample docs

**With ANTHROPIC_API_KEY:**
- 🔑 `langchain_basics.py` - All 4 examples
- 🔑 `simple_agent.py` - Agent with tools

**With ANTHROPIC + OPENAI Keys:**
- 🔑 `simple_rag.py` - Full RAG system

### Expected Behavior

**document_loader.py:**
```
✓ Created sample docs
✓ Loaded: 3 files
  - vlan_guide.md
  - bgp_standards.md
  - router_config.cfg
```

**simple_rag.py:**
```
✓ Added 18 chunks from 3 documents
✓ Search works
✓ Can answer:
  - "What VLAN for guest WiFi?" → VLAN 99
  - "What is our BGP AS?" → 65001
  - "Which OSPF areas?" → Area 0, 1, 2
```

**langchain_basics.py:**
```
✓ Example 1: Config analysis
✓ Example 2: Structured JSON output
✓ Example 3: Chat-style prompts
✓ Example 4: Config generation
```

**simple_agent.py:**
```
✓ Agent retrieves config sections
✓ Agent checks security
✓ Agent combines tools autonomously
```

## 🔧 Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "Invalid API key"
Check your `.env` file:
```bash
# Make sure keys are set correctly
cat .env

# Test connection
python -c "import anthropic; print('API key loaded')"
```

### "Rate limit exceeded"
Free tier limits:
- Anthropic: 5 requests/minute
- OpenAI: 3 requests/minute (embeddings)

**Solution**: Wait 60 seconds between runs or upgrade to paid tier.

### LangChain version issues
The code requires recent versions:
```bash
pip install --upgrade langchain langchain-anthropic langchain-openai langgraph
```

If you see API parameter warnings, you may need:
```python
# Older LangChain
ChatAnthropic(model_name="claude-3-5-sonnet-20241022")

# Newer LangChain
ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

### ChromaDB errors
Delete the database and recreate:
```bash
rm -rf ./chroma_db ./demo_chroma
python simple_rag.py
```

## 💰 Cost Estimates

### Per Example Run
- `document_loader.py`: **$0.00** (no API calls)
- `simple_rag.py`: **$0.02-0.05** (embeddings + queries)
- `langchain_basics.py`: **$0.05-0.10** (4 examples)
- `simple_agent.py`: **$0.03-0.08** (multiple tool calls)

### Full Testing
Running all examples once: **~$0.15-0.25**

### Tips to Save Money
1. Start with `document_loader.py` (free)
2. Test examples individually
3. Use `.env` to switch to cheaper models:
   ```bash
   DEFAULT_MODEL=claude-3-5-haiku-20241022  # 10x cheaper
   ```
4. Cache results during development

## 📚 What's Next?

### Additional Chapters (Coming Soon)
- Chapter 16: Document Retrieval
- Chapter 17: Semantic Search
- Chapter 20: Troubleshooting Agent
- Chapter 21: Config Generation Agent
- Chapter 24: Log Analysis

### Advanced Features
- Multi-document RAG
- Conversation memory
- Tool creation framework
- Production deployment examples

## 🎓 Learning Path

### Beginner (Start Here)
1. Run `document_loader.py` - Understand document processing
2. Read `simple_rag.py` - See RAG architecture
3. Run `langchain_basics.py` Example 1 - Simple chains

### Intermediate
1. Study `simple_rag.py` - RAG implementation
2. Try `langchain_basics.py` Examples 2-4 - Structured outputs
3. Understand embeddings and vector search

### Advanced
1. Build custom tools for `simple_agent.py`
2. Add conversation memory
3. Deploy RAG system to production
4. Build multi-agent systems

## 🤝 Contributing

Found issues? Have improvements?

1. Test the code
2. Document the issue
3. Submit a pull request
4. Share your use case!

## 📝 Notes

### Differences from Volume 1
- **Simpler code** - Easier to read and modify
- **LangChain focus** - Framework-based vs raw APIs
- **Agent-based** - Autonomous systems vs simple API calls
- **RAG systems** - Knowledge bases vs one-shot queries

### LangSmith Integration
Optional but recommended for production:
- Trace every LLM call
- Monitor costs in real-time
- Debug agent behavior
- Track performance

Enable in `.env`:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
```

View traces at: https://smith.langchain.com

## 📞 Support

- **Technical Issues**: GitHub Issues
- **Questions**: Discord community
- **Consulting**: ed@vexpertai.com

---

**Author**: Eduard Dulharu
**Company**: vExpertAI GmbH (Munich, Germany)
**Version**: 2.0.0
**Last Updated**: January 2026
**Status**: Chapters 14, 15, 19 Complete

**Ready to build AI agents?** Start with `document_loader.py` →
