#!/bin/bash
# Push Volume 3 & 4 Colab notebooks to GitHub
# Run this from: E:\vExpertAI\CONTENT\Book\AI_for_networking_and_security_engineers

echo "🚀 Pushing Volumes 3 & 4 to GitHub..."
echo ""

# Step 1: Check git status
echo "📊 Current git status:"
git status

echo ""
echo "📥 Step 1: Add new Colab notebooks for Volumes 3 & 4..."

# Add Volume 3 notebooks
git add CODE/Colab-Notebooks/Vol3_Ch32_Fine_Tuning.ipynb
git add CODE/Colab-Notebooks/Vol3_Ch34_Multi_Agent.ipynb
git add CODE/Colab-Notebooks/Vol3_Ch37_Graph_RAG_Topology.ipynb
git add CODE/Colab-Notebooks/Vol3_Ch48_Production_Monitoring.ipynb
git add CODE/Colab-Notebooks/Vol3_Ch51_Scaling_Systems.ipynb
git add CODE/Colab-Notebooks/Vol3_Ch61_NetOps_AI_Case_Study.ipynb

# Add Volume 4 notebooks
git add CODE/Colab-Notebooks/Vol4_Ch70_Threat_Detection.ipynb
git add CODE/Colab-Notebooks/Vol4_Ch72_SIEM_Integration.ipynb
git add CODE/Colab-Notebooks/Vol4_Ch75_Anomaly_Detection.ipynb
git add CODE/Colab-Notebooks/Vol4_Ch80_Securing_AI.ipynb
git add CODE/Colab-Notebooks/Vol4_Ch83_Compliance_Automation.ipynb
git add CODE/Colab-Notebooks/Vol4_Ch87_Security_Case_Study.ipynb

echo "✅ Staged 12 new Colab notebooks"
echo ""

# Step 2: Show what will be committed
echo "📋 Files staged for commit:"
git status --short

echo ""
echo "💾 Step 2: Creating commit..."

# Create commit
git commit -m "Add Volume 3 & 4 Colab notebooks

- Volume 3: 6 notebooks (Ch 32, 34, 37, 48, 51, 61)
  - Advanced Techniques (Fine-tuning, Multi-agent, Graph RAG)
  - Production Deployment (Monitoring, Scaling, Case Study)

- Volume 4: 6 notebooks (Ch 70, 72, 75, 80, 83, 87)
  - Security Operations (Threat detection, SIEM, Anomaly detection)
  - Securing AI (AI security, Compliance, Case Study)

All notebooks include:
- Simple, well-commented code
- Setup instructions with API key configuration
- Working examples
- Interactive playground sections

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

echo ""
echo "🔄 Step 3: Pushing to GitHub main branch..."

# Push to main branch
git push origin main

echo ""
echo "✅ SUCCESS! Volume 3 & 4 Colab notebooks pushed to GitHub"
echo ""
echo "📍 View at: https://github.com/eduardd76/AI_for_networking_and_security_engineers/tree/main/CODE/Colab-Notebooks"
