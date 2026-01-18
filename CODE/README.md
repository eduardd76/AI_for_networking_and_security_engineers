# AI for Networking Engineers - Complete Code Repository

**Production-ready code examples from the book series**

## 📚 Two Ways to Run the Code

### 🚀 Option 1: Google Colab (Recommended for Beginners)
**No installation required - run in your browser!**

Perfect for:
- Network engineers on corporate laptops (no admin rights needed)
- Quick testing and learning
- Running code from anywhere

**[→ Start with Colab Notebooks](./Colab-Notebooks/)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduardd76/AI_for_networking_and_security_engineers/blob/main/CODE/Colab-Notebooks/)

### 💻 Option 2: Local Python
**Full development environment on your machine**

Perfect for:
- Building production systems
- Custom development
- Working offline

**Choose your volume:**
- **[Volume 1: Foundations](./Volume-1-Foundations/)** - Learn AI basics (9 chapters)
- **[Volume 2: Advanced Applications](./Volume-2-Advanced-Applications/)** - RAG, agents, production (8 chapters)

## 🎯 Quick Decision Guide

**Choose Colab if you want to:**
- ✅ Try the code immediately (no setup)
- ✅ Learn on a restricted corporate laptop
- ✅ Test examples quickly
- ✅ Share notebooks with colleagues

**Choose Local if you want to:**
- ✅ Build production systems
- ✅ Integrate with your network tools (Netmiko, NAPALM)
- ✅ Work offline
- ✅ Customize extensively

## 📖 What's Included

### Volume 1: Foundations (Beginner)
Learn the basics of AI for networking:
- **Chapter 4**: API Basics - Call Claude API with LangChain
- **Chapter 5**: Prompt Engineering - Write better prompts
- **Chapter 6**: Structured Outputs - Get JSON with Pydantic
- **Chapter 7**: Context Management - Handle large configs
- **Chapter 8**: Cost Optimization - Save 50-70% on costs
- **Chapter 9**: Network Data - Process configs and logs
- **Chapter 10**: API Integration - Combine with Netmiko
- **Chapter 11**: Testing - Validate AI outputs
- **Chapter 12**: Ethics - Safe production use

**Total**: 9 Python files, ~1,800 lines of code

### Volume 2: Advanced Applications (Intermediate/Advanced)
Build production RAG systems and agents:
- **Chapter 14**: RAG Fundamentals - Knowledge bases
- **Chapter 15**: LangChain Integration - Framework patterns
- **Chapter 16**: Document Retrieval - Advanced techniques
- **Chapter 17**: Semantic Search - Intent-based search
- **Chapter 19**: Agent Architecture - LangGraph agents
- **Chapter 20**: Troubleshooting Agent - Autonomous diagnostics
- **Chapter 21**: Config Generation - Generate configs from requirements
- **Chapter 24**: Log Analysis - AI-powered log analysis

**Total**: 13 Python files, ~3,500 lines of code

## 🚀 Getting Started

### For Colab Users (No Setup)
1. Go to [Colab Notebooks](./Colab-Notebooks/)
2. Click any "Open in Colab" badge
3. Enter your API key when prompted
4. Run the cells!

### For Local Users
1. Choose your volume (1 or 2)
2. Follow the README in that directory
3. Install dependencies: `pip install -r requirements.txt`
4. Set up `.env` file with API keys
5. Run examples: `python chapter_file.py`

## 🔑 API Keys Needed

**Anthropic API Key** (Required for most examples):
- Get it free at https://console.anthropic.com/
- Free tier: 5 requests/minute
- Paid tier: $0.80-$15 per million tokens

**OpenAI API Key** (Required for RAG examples in Volume 2):
- Get it at https://platform.openai.com/
- Needed for embeddings (text-embedding-3-small)
- ~$0.02 per 1000 pages of text

## 💰 Cost Estimates

**Volume 1** (complete run): $0.30-0.50
**Volume 2** (complete run): $0.50-1.00

**Tips to save money:**
- Use Haiku model instead of Sonnet (10x cheaper)
- Cache repeated queries
- Batch process multiple items
- Use Colab for learning (free compute)

## 🎓 Learning Paths

### Path 1: Complete Beginner
1. Start with Colab (no setup)
2. Volume 1, Chapter 4 (API Basics)
3. Volume 1, Chapter 5 (Prompt Engineering)
4. Volume 1, Chapter 6 (Structured Outputs)

### Path 2: Know Python, New to AI
1. Volume 1, Chapter 4-6 (quick review)
2. Volume 1, Chapter 8 (Cost Optimization)
3. Volume 1, Chapter 9 (Network Data)
4. Volume 2, Chapter 14 (RAG Basics)

### Path 3: Know AI, New to Networking
1. Volume 1, Chapter 9 (Network Data)
2. Volume 1, Chapter 10 (API Integration)
3. Volume 2 - all chapters

### Path 4: Ready for Production
1. Volume 2, Chapter 14-17 (RAG systems)
2. Volume 2, Chapter 19-21 (Agents)
3. Volume 1, Chapter 11-12 (Testing & Ethics)

## 📂 Repository Structure

```
CODE/
├── Colab-Notebooks/          # Google Colab notebooks (run in browser)
│   ├── Vol1_Ch4_API_Basics.ipynb
│   ├── Vol2_Ch14_RAG_Basics.ipynb
│   └── README.md
│
├── Volume-1-Foundations/     # Local Python files (Ch 4-12)
│   ├── Chapter-04-API-Basics-Authentication/
│   ├── Chapter-05-Prompt-Engineering/
│   ├── ...
│   └── README.md
│
└── Volume-2-Advanced-Applications/  # Local Python files (Ch 14-24)
    ├── Chapter-14-RAG-Fundamentals/
    ├── Chapter-15-LangChain-Integration/
    ├── ...
    └── README.md
```

## 🔧 Troubleshooting

### Common Issues

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"Invalid API key"**
- Check your `.env` file (local) or Colab secrets
- Verify key starts with `sk-ant-api03-` (Anthropic) or `sk-proj-` (OpenAI)

**"Rate limit exceeded"**
- Free tier: 5 requests/minute
- Wait 60 seconds or upgrade to paid

**Colab disconnected**
- Free tier: 12 hours max runtime
- Just reconnect and resume

## 🤝 Contributing

Found issues? Have improvements?
1. Open an issue on GitHub
2. Submit a pull request
3. Share your use case!

## 📞 Support

- **GitHub Issues**: Report bugs or request features
- **Email**: ed@vexpertai.com
- **Company**: vExpertAI GmbH (Munich, Germany)

## 📄 License

Code is provided for educational purposes from the book:
**"AI for Networking Engineers"** by Eduard Dulharu

---

**Author**: Eduard Dulharu
**Company**: vExpertAI GmbH (Munich, Germany)
**Last Updated**: January 2026
**Version**: 2.0

**Ready to start?**
- 🚀 [Try in Colab (no setup)](./Colab-Notebooks/)
- 💻 [Volume 1: Foundations](./Volume-1-Foundations/)
- 🔥 [Volume 2: Advanced](./Volume-2-Advanced-Applications/)
