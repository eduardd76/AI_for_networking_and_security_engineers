# Volume 2: Practical Applications - Code Repository

Complete, production-ready code examples from **AI for Networking Engineers: Volume 2 - Practical Applications**

## 📁 Repository Structure

```
Volume-2-Practical-Applications/
├── Chapter-13-Network-Documentation-Basics/
│   ├── doc_generator.py
│   ├── topology_diagrammer.py
│   ├── documentation_pipeline.py
│   └── README.md
├── Chapter-14-RAG-Fundamentals/
│   └── (coming soon)
├── Chapter-15-LangChain-Integration/
│   └── (coming soon)
├── .env.example
├── requirements.txt
└── README.md (this file)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/eduardd76/AI_for_networking_and_security_engineers.git
cd AI_for_networking_and_security_engineers/Volume-2-Practical-Applications

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
```

### 3. Run Examples

```bash
# Generate device documentation
python Chapter-13-Network-Documentation-Basics/doc_generator.py

# Create topology diagrams
python Chapter-13-Network-Documentation-Basics/topology_diagrammer.py

# Run documentation pipeline
python Chapter-13-Network-Documentation-Basics/documentation_pipeline.py --generate-now
```

## 📖 Chapter Guide

### Part 1: Documentation & Knowledge (Chapters 13-18)

**Chapter 13: Network Documentation Basics**
- Auto-generate documentation from configs
- Create network topology diagrams with AI
- Build documentation pipelines that stay in sync

**Chapter 14: RAG Fundamentals** *(coming soon)*
- Build searchable documentation with AI
- Vector databases and embeddings

**Chapter 15: LangChain Integration** *(coming soon)*
- Framework integration patterns
- Building AI chains

### Part 2: Agents & Automation (Chapters 19-24)

*(coming soon)*

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 💰 Cost Estimates

**Chapter 13 - Full documentation run**:
- API calls: ~4 per device (overview, interfaces, routing, security)
- Total tokens per device: ~5K
- Cost per device: ~$0.05 with Sonnet

**500 devices daily**: ~$25/run or $750/month

**Tip**: Use Haiku for simple extraction tasks (10x cheaper)

## 📚 Dependencies

Core dependencies:
- `anthropic>=0.18.0` - Claude API client
- `python-dotenv>=1.0.0` - Environment variable management

Optional (for pipeline features):
- `schedule>=1.2.0` - Task scheduling
- `gitpython>=3.1.0` - Git integration

See `requirements.txt` for complete list.

## 📖 Additional Resources

**Book**: AI for Networking Engineers - Volume 2: Practical Applications
**Author**: Eduard Dulharu (Ed Harmoosh)
**Website**: https://vexpertai.com
**GitHub**: https://github.com/eduardd76/AI_for_networking_and_security_engineers

## 📝 License

MIT License - See LICENSE file for details

---

**Version**: 1.0.0
**Last Updated**: January 2026
**Tested With**: Python 3.10+, Claude Sonnet 4.5

**Ready to start?** Begin with Chapter 13: Network Documentation Basics
