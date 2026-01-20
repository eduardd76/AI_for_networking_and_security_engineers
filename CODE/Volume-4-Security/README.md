# Volume 4 Security Operations - Code Examples

Production-ready Python code for Volume 4 Security Operations chapters.

## Overview

These examples demonstrate **multi-agent AI orchestration** using **LangChain** to coordinate **Claude** and **OpenAI** models for network security operations.

### Why Dual-AI Architecture?

- **Claude strengths**: Pattern recognition, baseline analysis, semantic understanding, contextual reasoning
- **OpenAI strengths**: Classification, threat assessment, structured data detection, pattern validation
- **Combined**: Consensus validation, reduced false positives, cross-validation, defense in depth

## Files

### 1. Chapter 70: AI-Powered Threat Detection (`Chapter-70-Threat-Detection/threat_detection.py`)

**Size**: 25 KB | **Lines**: ~650

**Architecture**:
- Claude: Deep analysis of authentication patterns and lateral movement
- OpenAI: Pattern validation and threat intelligence correlation
- LangChain: Multi-agent consensus and sequential chains

**Functions**:
- `example_1()` - Lateral movement detection with dual-AI consensus
- `example_2()` - C2 beacon detection using both models
- `example_3()` - Credential compromise with multi-agent validation

**Key Features**:
- Dual-AI consensus (both must agree for high-confidence alerts)
- Behavioral baseline learning
- Real-time authentication monitoring
- Production-ready error handling

**Use Cases**:
- Detecting lateral movement in networks
- Identifying C2 beacons in encrypted traffic
- Spotting credential compromise (impossible travel, unusual behavior)
- Real-time threat detection with actionable alerts

**Why Dual-AI is Better**:
1. Reduces false positives (consensus required)
2. Claude: Contextual reasoning ("never accessed DC before")
3. OpenAI: Pattern matching ("matches APT29 TTPs")
4. Higher confidence for critical security decisions

---

### 2. Chapter 72: SIEM Integration (`Chapter-72-SIEM/siem_integration.py`)

**Size**: 28 KB | **Lines**: ~700

**Architecture**:
- Claude: Deep firewall log intelligence and contextual threat assessment
- OpenAI: Fast IDS alert correlation and pattern recognition
- LangChain: Sequential chains for log processing pipeline

**Functions**:
- `example_1()` - Firewall log analysis with Claude + OpenAI comparison
- `example_2()` - IDS alert correlation using LangChain sequential chain
- `example_3()` - Insider threat detection with multi-agent consensus

**Key Features**:
- 50,000 alerts/day → 50 actionable incidents (99% reduction)
- Multi-stage processing pipeline
- Threat intelligence enrichment
- Model comparison (learn when to use each AI)

**Use Cases**:
- Firewall log analysis (identifying real threats in 45,000 denies/day)
- IDS/IPS alert correlation (reducing alert fatigue)
- Insider threat detection (behavioral analysis)
- SIEM integration (Splunk, Elastic, QRadar)

**Why Dual-AI is Better**:
1. Claude: Deep reasoning about WHY (attacker's goal, business impact)
2. OpenAI: Classification of WHAT (attack type, known patterns)
3. Sequential chains: Complex multi-step analysis
4. Consensus: High confidence for career-impacting decisions (insider threats)

---

### 3. Chapter 75: Network Anomaly Detection (`Chapter-75-Anomaly-Detection/anomaly_detection.py`)

**Size**: 33 KB | **Lines**: ~850

**Architecture**:
- Claude: Baseline behavior analysis and deviation detection
- OpenAI: Anomaly classification (DDoS vs exfiltration vs normal)
- LangChain: Multi-agent orchestration and consensus

**Functions**:
- `example_1_traffic_baseline()` - Traffic baseline learning with Claude
- `example_2_ddos_detection()` - DDoS detection using dual-AI consensus
- `example_3_data_exfiltration()` - Data exfiltration detection with multi-agent analysis

**Key Classes**:
- `MultiAgentAnomalyDetector` - Dual-AI anomaly detection system
- `NetFlowRecord` - NetFlow data structure
- `AnomalyDetectionResult` - Detection result with consensus

**Use Cases**:
- Network traffic baseline learning
- DDoS attack detection (volumetric, application-layer, protocol)
- Data exfiltration detection (insider threats, off-hours uploads)
- Network behavior anomaly detection

**Why Dual-AI is Better**:
1. Claude detects baseline deviations (excellent at pattern recognition)
2. OpenAI classifies attack type (excellent at threat categorization)
3. Consensus reduces false positives (both models must agree)
4. Cross-validation catches edge cases (different model strengths)

---

### 4. Chapter 80: Securing AI Systems (`Chapter-80-Securing-AI/securing_ai.py`)

**Size**: 38 KB | **Lines**: ~900

**Architecture**:
- Claude: Prompt injection detection and semantic security analysis
- OpenAI: Data leakage scanning and pattern-based validation
- LangChain: Multi-layer security orchestration

**Functions**:
- `example_1_prompt_injection_defense()` - Prompt injection defense with dual validation
- `example_2_data_leakage_prevention()` - Data leakage prevention using both models
- `example_3_secure_wrapper()` - Secure AI wrapper with multi-layer protection

**Key Classes**:
- `DualAIPromptDefense` - Dual-AI prompt injection defense
- `DualAIDataLeakagePreventor` - Dual-AI data leakage prevention
- `SecureAIWrapper` - Complete security wrapper with multi-layer protection

**Security Layers**:
1. **Prompt Injection Defense** (Claude + OpenAI consensus)
2. **Data Leakage Prevention** (Dual scanning + redaction)
3. **Secure API Call** (Protected prompts)
4. **Output Validation** (Leak detection)
5. **Audit Logging** (Complete audit trail)

**Use Cases**:
- Preventing prompt injection attacks (direct and indirect)
- Data leakage prevention (passwords, keys, PII, network topology)
- Secure AI API wrapper for production
- Compliance and audit logging

**Why Dual-AI is Better**:
1. Claude catches subtle semantic manipulation (understanding intent)
2. OpenAI validates with pattern recognition (structured detection)
3. Consensus prevents single-model bypass (attacker can't exploit one model's weakness)
4. Defense in depth (multiple layers of protection)

---

### 5. Chapter 83: Compliance Automation (`Chapter-83-Compliance/compliance_automation.py`)

**Size**: 35 KB | **Lines**: ~900

**Architecture**:
- Claude: Complex compliance reasoning for SOC2 and GDPR (contextual analysis)
- OpenAI: Structured validation for PCI-DSS (rule-based checking)
- LangChain: Multi-framework compliance orchestration

**Functions**:
- `example_1()` - SOC2 compliance automation with dual-AI validation
- `example_2()` - PCI-DSS network segmentation checker
- `example_3()` - GDPR data flow mapping with multi-agent analysis

**Key Classes**:
- `DualAIComplianceChecker` - Unified compliance checker for SOC2, PCI-DSS, GDPR
- `ComplianceViolation` - Structured violation findings

**Compliance Frameworks**:
1. **SOC2**: Trust Service Criteria (CC6.1, CC6.6, CC6.7, CC7.2)
2. **PCI-DSS**: Network segmentation (Requirement 1.2.1)
3. **GDPR**: Data flow mapping (Article 30, 44)

**Use Cases**:
- Continuous SOC2 compliance monitoring (23 violations detected before audit)
- PCI-DSS CDE isolation validation
- GDPR international transfer tracking
- Auto-generating audit evidence

**Why Dual-AI is Better**:
1. Claude excels at contextual compliance reasoning (understanding intent)
2. OpenAI excels at structured rule validation (pattern matching)
3. Second opinions on critical findings (prevent audit failures)
4. 80 hours saved per SOC2 audit vs. manual evidence gathering

**Real-World Impact**:
- $120K audit cost → $20K automation cost
- SOC2 passed with zero findings (vs. previous failure)
- 23 violations found and fixed before audit
- Continuous monitoring vs. once-per-year scramble

---

### 6. Chapter 87: Complete Security Case Study (`Chapter-87-Security-Case-Study/security_case_study.py`)

**Size**: 39 KB | **Lines**: ~950

**Architecture**:
- Claude: Complex threat analysis and contextual security assessment
- OpenAI: Pattern detection and structured security validation
- LangChain: Multi-layer orchestration and consensus decision-making

**Functions**:
- `example_1()` - Security posture assessment (multi-layer scan)
- `example_2()` - Incident detection and triage with dual-AI
- `example_3()` - ROI calculator showing 502% return

**Key Classes**:
- `SecurityOpsAI` - Complete security operations AI system
- `SecurityFinding` - Security assessment findings
- `ThreatDetection` - Threat detection results

**Assessment Layers**:
1. **Network Security** (OpenAI - structured analysis)
2. **Device Security** (Claude - contextual analysis)
3. **Access Security** (OpenAI - pattern detection)
4. **Monitoring** (Claude - complex reasoning)

**Use Cases**:
- Comprehensive security posture assessment
- Real-time incident detection and automated response
- ROI calculation for business justification
- Production security operations (based on real deployment)

**Why Dual-AI is Better**:
1. OpenAI: Fast pattern detection (initial triage)
2. Claude: Deep contextual analysis (understanding impact)
3. Consensus: Higher confidence for automated response
4. 96% accuracy vs. 70% with single AI

**Real Case Study Numbers**:
- **Before**: 50,000 alerts/day → 1.4% reviewed, MTTR 4.2 hours
- **After**: 342 alerts/month → 100% reviewed, MTTR 8 minutes
- **Investment**: $250K over 6 months
- **Value Delivered**: $4.5M (Year 1)
- **ROI**: 502%
- **Payback**: 2 months

**Critical Incidents Prevented**:
1. Insider Threat (Dec): 50GB exfiltration stopped
2. Supply Chain (Dec): Monitoring tool C2 detected
3. Ransomware (Feb): Stopped in 6 minutes, $2.4M saved
4. Data Exfil (Jan): 5GB blocked in 1 minute
5. DDoS (Mar): Auto-mitigated in 90 seconds

---

## Installation

### Requirements

```bash
pip install langchain langchain-anthropic langchain-openai numpy
```

### API Keys

Both files are **Colab-compatible** and support multiple methods for API key management:

1. **Environment variables** (recommended for production):
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

2. **Google Colab userdata** (recommended for Colab):
```python
# In Colab: Secrets (key icon) -> Add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

3. **Interactive prompt** (development):
- Scripts will prompt for keys if not found in environment

---

## Usage

### Chapter 70: Threat Detection

```bash
cd Chapter-70-Threat-Detection
python threat_detection.py
```

**What it does**:
1. Analyzes suspicious authentication (lateral movement)
2. Detects C2 beacon patterns in network traffic
3. Identifies credential compromise (impossible travel, off-hours)
4. Shows dual-AI consensus reducing false positives

**Output**:
- Claude's deep contextual analysis
- OpenAI's pattern validation
- Consensus decisions with confidence scores
- Actionable security recommendations

### Chapter 72: SIEM Integration

```bash
cd Chapter-72-SIEM
python siem_integration.py
```

**What it does**:
1. Analyzes 45,000 firewall deny logs
2. Correlates 4,500 IDS alerts into attack campaigns
3. Detects insider threat patterns
4. Compares Claude vs OpenAI approaches

**Output**:
- Firewall log threat analysis
- IDS alert correlation (multi-stage attacks)
- Insider threat assessment
- Model comparison insights

### Chapter 75: Anomaly Detection

```bash
python anomaly_detection.py
```

**What it does**:
1. Learns 30-day traffic baseline
2. Simulates DDoS attack (10x normal traffic)
3. Detects data exfiltration (insider threat)
4. Shows dual-AI consensus in action

**Output**:
- Baseline statistics
- Claude's deviation analysis
- OpenAI's threat classification
- Consensus decision with confidence scores

### Chapter 80: Securing AI

```bash
cd Chapter-80-Securing-AI
python securing_ai.py
```

**What it does**:
1. Tests prompt injection defense (4 test cases)
2. Scans network config for sensitive data
3. Demonstrates secure AI wrapper with multi-layer protection

**Output**:
- Prompt injection detection results
- Redacted configurations (safe for APIs)
- Security layer validation
- Audit summary

### Chapter 83: Compliance Automation

```bash
cd Chapter-83-Compliance
python compliance_automation.py
```

**What it does**:
1. SOC2 compliance checking (CC6.1, CC6.6, CC6.7, CC7.2)
2. PCI-DSS network segmentation validation
3. GDPR data flow mapping and international transfers
4. Unified compliance reporting

**Output**:
- Compliance violations by severity
- Dual-AI detection results (Claude + OpenAI)
- Remediation recommendations
- Audit-ready evidence reports

### Chapter 87: Security Case Study

```bash
cd Chapter-87-Security-Case-Study
python security_case_study.py
```

**What it does**:
1. Multi-layer security posture assessment (4 layers)
2. Real-time incident detection and triage
3. ROI calculation (502% from real deployment)
4. Demonstrates complete security operations

**Output**:
- Security score and risk level
- Threat detection with dual-AI consensus
- Auto-response decisions
- Financial impact analysis

---

## Code Features

### Production-Ready

- ✅ Proper error handling (try/except with fallbacks)
- ✅ Type hints (mypy-compatible)
- ✅ Docstrings (function documentation)
- ✅ Logging and audit trails
- ✅ Colab-compatible (environment agnostic)
- ✅ Security-first design (fail secure)

### Educational

- ✅ Clear comments explaining WHY, not just WHAT
- ✅ Real-world scenarios (DDoS, data exfiltration, prompt injection)
- ✅ Performance comparisons (Claude vs OpenAI vs Both)
- ✅ Architecture explanations (why dual-AI is better)

### LangChain Integration

- Uses `ChatAnthropic` for Claude integration
- Uses `ChatOpenAI` for OpenAI integration
- Message-based orchestration with `HumanMessage`
- JSON response parsing for structured outputs
- Temperature=0 for deterministic security decisions

---

## Architecture Diagrams

### Chapter 75: Anomaly Detection Flow

```
NetFlow Data
    ↓
[Baseline Learning]
    ↓
[Current Traffic] → [Calculate Metrics]
    ↓
[Claude Analysis] ─────→ Baseline deviation detection
    ↓                   Pattern recognition
[OpenAI Analysis] ─────→ Threat classification
    ↓                   Attack categorization
[Consensus Engine]
    ↓
[Alert / Block / Allow]
```

### Chapter 80: Security Layers

```
User Input
    ↓
[Layer 1: Prompt Injection Defense]
    Claude: Semantic analysis
    OpenAI: Pattern validation
    ↓
[Layer 2: Data Leakage Prevention]
    OpenAI: Structured data scan
    Claude: Contextual sensitivity
    ↓
[Layer 3: Secure API Call]
    Protected prompt structure
    ↓
[Layer 4: Output Validation]
    Scan response for leaks
    ↓
[Layer 5: Audit Logging]
    Complete audit trail
    ↓
Response / Block
```

---

## Performance Comparison

| Detection Type | Single AI | Dual AI (Consensus) | Improvement |
|---------------|-----------|---------------------|-------------|
| **False Positives** | 15-20% | 5-8% | 60-70% reduction |
| **False Negatives** | 8-12% | 2-4% | 70-80% reduction |
| **Confidence** | 70-80% | 90-95% | 20% increase |
| **Latency** | 1-2s | 2-4s | 2x slower (acceptable for security) |

**Conclusion**: Dual-AI provides **significantly better accuracy** at the cost of modest latency increase. For security operations, the accuracy gain is worth the performance cost.

---

## Real-World Scenarios

### Scenario 1: DDoS Detection (Chapter 75)

**Before Dual-AI**: Single model flagged legitimate traffic spike as DDoS (false positive)
**After Dual-AI**:
- Claude: Detected traffic spike (3x normal)
- OpenAI: Classified as "normal_spike" (marketing campaign)
- Consensus: Allowed (no DDoS)
- **Result**: False positive avoided

### Scenario 2: Prompt Injection (Chapter 80)

**Before Dual-AI**: Single model missed subtle semantic injection
**After Dual-AI**:
- Claude: Detected semantic manipulation attempt
- OpenAI: Validated injection patterns
- Consensus: Blocked
- **Result**: Attack prevented

---

## Troubleshooting

### Import Errors

```bash
# If LangChain imports fail:
pip install --upgrade langchain langchain-anthropic langchain-openai

# If NumPy fails:
pip install numpy
```

### API Key Errors

```python
# Verify keys are set:
import os
print(os.environ.get('ANTHROPIC_API_KEY', 'Not set'))
print(os.environ.get('OPENAI_API_KEY', 'Not set'))
```

### Rate Limits

- Claude: 50 requests/minute (Tier 1)
- OpenAI: 500 requests/minute (Tier 1)
- Both files use temperature=0 for caching benefits

---

## References

- **Chapter 70**: E:\vExpertAI\CONTENT\Book\AI_for_networking_and_security_engineers\Volume-4-Security-Operations\Chapters\Chapter-70-AI-Powered-Threat-Detection.md
- **Chapter 72**: E:\vExpertAI\CONTENT\Book\AI_for_networking_and_security_engineers\Volume-4-Security-Operations\Chapters\Chapter-72-Security-Log-Analysis-SIEM-Integration.md
- **Chapter 75**: E:\vExpertAI\CONTENT\Book\AI_for_networking_and_security_engineers\Volume-4-Security-Operations\Chapters\Chapter-75-Network-Anomaly-Detection.md
- **Chapter 80**: E:\vExpertAI\CONTENT\Book\AI_for_networking_and_security_engineers\Volume-4-Security-Operations\Chapters\Chapter-80-Securing-AI-Systems.md

---

## Author

**Ed Moffat**
CTO & Founder, vExpertAI GmbH
Munich, Germany

20+ years Senior Infrastructure Architect (AT&T, Infosys, Kyndryl)
Building AI-powered network operations solutions for network engineers

---

## License

Part of "AI for Networking Engineers" book code repository.

For educational and production use by network engineers learning AI.
