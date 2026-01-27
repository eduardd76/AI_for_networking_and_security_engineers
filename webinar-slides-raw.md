# Webinar Slides: How to Build an AI Agent for Network Engineers
## Source: webinar.vexpertai.com

---

## Slide 1: Title
**How to Build an AI Agent for Network Engineers**
Beyond Hype and Scripts: Reasoning over Real Networks

Eduard Dulharu
CTO vExpertAI GmbH

---

## Slide 2: The 3 AM Scenario
**Why We Need Agents**

3 AM: "VXLAN tunnels down between data center sites"

**Your Process:**
- ✓ SSH into devices (which ones?)
- ✓ Run `show nve peers` (peers are up...)
- ✓ Check VNI configs (what VNI was it?)
- ✓ Compare configs between leafs
- ✓ Search documentation
- ✓ Find VNI mismatch... 20 minutes later

> This is a PATTERN-MATCHING problem that should be automated. But traditional automation EXECUTES. It doesn't REASON.

---

## Slide 3: The Core Insight
**Automation Executes, AI Agents Reason**

| AUTOMATION | AI AGENTS |
|------------|-----------|
| ✕ Follows scripts | ✓ Forms hypotheses |
| ✕ Pre-defined paths | ✓ Explores solution space |
| ✕ Brittle on exceptions | ✓ Adapts to unknowns |
| ✕ Executes | ✓ Reasons |

> "Most AI demos in networking are just chatbots wrapped around documentation. They execute commands, but they don't reason like a network engineer."

---

## Slide 4: How Humans Troubleshoot
**The Cognitive Loop**

1. **OBSERVE** - What do I know? (Read logs, check status)
2. **REASON** - What could cause this? (Form hypothesis)
3. **ACT** - Let me check that theory (Run commands)
4. **VERIFY** - Does this explain it? (Validate hypothesis)
5. **LOOP** - Repeat until solved

---

## Slide 5: Systems Thinking
**The Foundation of Agent Design**

- **Holistic View**: Networks are complex adaptive systems. Problems rarely have single root causes.
- **Feedback Loops**: Agent actions affect network state, which affects agent decisions.
- **Emergence**: Network behavior emerges from component interactions. Agents must adapt.
- **Leverage Points**: Small interventions in the right place = big impact (e.g., BGP config vs reboot).

---

## [Slides 6-59: Capturing...]

