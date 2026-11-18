# Claude API Budget & Token Analysis for Consolidation Drift

**Project:** Consolidation Drift (Multi-Model Replication & Grader Cross-Check)  
**Date:** November 2026  
**Repository:** `c:\Python\urop`  

---

## 1. Important Billing Distinction

> [!WARNING]
> **Do not purchase a Claude Pro web subscription ($20/month on `claude.ai`).**
> - **Claude Pro ($20/mo):** Only works in the web chat browser interface. It enforces dynamic message caps (~45 messages every 5 hours) and **cannot be accessed by Python scripts or automated runners**.
> - **Anthropic Console (`console.anthropic.com`):** The developer platform where you purchase prepaid API credits. Scripts connect via API keys, and calls are billed strictly by the token count consumed.

---

## 2. Official Anthropic API Pricing

Pricing per **1 Million (1M) tokens**:

| Model | Identifier | Input Cost / 1M Tokens | Output Cost / 1M Tokens | Role Fit |
| :--- | :--- | :---: | :---: | :--- |
| **Claude 3.5 Haiku** | `claude-3-5-haiku-20241022` | **$0.80** | **$4.00** | Fast, deterministic grading rubric evaluation; ultra-low-cost writer |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022` | **$3.00** | **$15.00** | Frontier reasoning; gold-standard summarizer & writer |
| **Claude 3 Opus** | `claude-3-opus-20240229` | $15.00 | $75.00 | Flagship reasoning; cost-prohibitive for repetitive tasks |

---

## 3. Workload Mathematical Token Breakdown

Calculations derived directly from empirical run traces in Stage A (`data/experiments/stage_a`):

```
                       TOTAL EXPERIMENTAL WORKLOAD
                       
       +---------------------------------------------------------+
       |               Stage A Experimental Suite                |
       +---------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
        [Consolidation Writer]             [Independent Grader]
        - 48 items x 3 context depths      - 288 chains x 20 rounds
        - 20 consolidation rounds          - 5,760 total evaluations
        - 2,880 rounds total               - 1-5 scale evaluation
        - Notes -> Chunks -> Global        - ~250 in / ~10 out
```

---

### Task 1: Independent Cross-Check Grader

* **Goal:** Use Claude to grade all 5,760 memory snapshots from Stage A to cross-check `gpt-oss-120b`'s scoring accuracy.
* **Volume:** 288 chains $\times$ 20 rounds = **5,760 evaluations**.
* **Per-Call Profile:**
  * **Input:** ~250 tokens (instructions, 1–5 scale rubric, consolidated memory snippet).
  * **Output:** ~10 tokens (single integer rating `1` to `5` or `NOT_PRESENT`).
* **Total Token Volume:**
  * **Input:** $5,760 \times 250 = \mathbf{1.44\text{ M tokens}}$
  * **Output:** $5,760 \times 10 = \mathbf{0.06\text{ M tokens}}$

#### Grader Cost Calculations:
* **With Claude 3.5 Haiku:**
  $$\text{Input: } 1.44\text{ M} \times \$0.80 = \$1.15$$
  $$\text{Output: } 0.0576\text{ M} \times \$4.00 = \$0.23$$
  $$\mathbf{\text{Total Grader Cost (Haiku)} = \$1.38}$$

* **With Claude 3.5 Sonnet:**
  $$\text{Input: } 1.44\text{ M} \times \$3.00 = \$4.32$$
  $$\text{Output: } 0.0576\text{ M} \times \$15.00 = \$0.86$$
  $$\mathbf{\text{Total Grader Cost (Sonnet)} = \$5.18}$$

---

### Task 2: Second Model Family (Consolidation Writer)

* **Goal:** Re-run the full hierarchical consolidation pipeline using Claude instead of Qwen to evaluate model-agnostic drift.
* **Volume:** 48 items $\times$ 3 filler depths ($c0, c8, c20$) = **144 chains $\times$ 20 rounds = 2,880 rounds**.
* **Per-Round Pipeline Architecture:**
  1. *Turn Notes:* Compresses raw turns in groups of 4.
  2. *Chunk Summaries:* Aggregates turn notes into 1–2 line summaries.
  3. *Global Memory:* Distills chunk summaries into a 120-word capped memory.
* **Weighted Averages per Round:**
  * Average Input: $\sim 1,500\text{ tokens}$
  * Average Output: $\sim 300\text{ tokens}$
* **Total Token Volume (2,880 Rounds):**
  * **Input:** $2,880 \times 1,500 = \mathbf{4.32\text{ M tokens}}$
  * **Output:** $2,880 \times 300 = \mathbf{0.864\text{ M tokens}}$

#### Writer Cost Calculations:
* **With Claude 3.5 Haiku:**
  $$\text{Input: } 4.32\text{ M} \times \$0.80 = \$3.46$$
  $$\text{Output: } 0.864\text{ M} \times \$4.00 = \$3.46$$
  $$\mathbf{\text{Total Writer Cost (Haiku)} = \$6.92}$$

* **With Claude 3.5 Sonnet:**
  $$\text{Input: } 4.32\text{ M} \times \$3.00 = \$12.96$$
  $$\text{Output: } 0.864\text{ M} \times \$15.00 = \$12.96$$
  $$\mathbf{\text{Total Writer Cost (Sonnet)} = \$25.92}$$

*(Note: Adding the **AVR arm** on Claude requires an extra ~1.5M input and ~0.2M output tokens for verification checks, adding **+$7.50 on Sonnet** or **+$2.00 on Haiku**).*

---

## 4. Cost Matrix by Setup

| Setup Option | Writer Engine | Grader Engine | Writer Cost | Grader Cost | Safety Buffer (15%) | **Net Budget** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Option 1: Ultra-Budget** | Claude 3.5 Haiku | Claude 3.5 Haiku | $6.92 | $1.38 | $1.24 | **$9.54** |
| **Option 2: Recommended Hybrid** | **Claude 3.5 Sonnet** | **Claude 3.5 Haiku** | **$25.92** | **$1.38** | **$4.10** | **$31.40** |
| **Option 3: Full Frontier** | Claude 3.5 Sonnet | Claude 3.5 Sonnet | $25.92 | $5.18 | $4.67 | **$35.77** |
| **Option 4: Full Frontier + AVR Arm** | Claude 3.5 Sonnet (+AVR) | Claude 3.5 Sonnet | $33.42 | $5.18 | $5.79 | **$44.39** |

---

## 5. Strategic Recommendation

### Why Option 2 (Deposit $35) is Recommended:
1. **Frontier Writer Quality (Sonnet):** Testing whether summarization drift (+0.33) is an artifact of smaller models (Qwen) requires testing a frontier model. Claude 3.5 Sonnet is the strongest reasoning and summarization benchmark model available.
2. **Cost-Efficient Grading (Haiku):** Grading is an unambiguous 1–5 classification problem. Haiku performs deterministic classification at human accuracy while costing only **$1.38 across all 5,760 evaluations**.
3. **Budget Safety:** Depositing **$35** into an Anthropic Console account covers the entire Stage A replication with room for test runs and retries.

---

## 6. Setup & Execution Instructions

1. **Create Account:**
   * Go to **[console.anthropic.com](https://console.anthropic.com/)** (not `claude.ai`).
   * Sign up and open the **Billing** tab.
2. **Deposit Funds:**
   * Add **$35.00** in prepaid credits (or **$15.00** for the Haiku-only setup).
3. **Generate Key:**
   * Go to **API Keys** $\to$ **Create Key**.
   * Copy the key string (`sk-ant-api03-...`).
4. **Configure Environment:**
   * In `c:\Python\urop\.env`, add:
     ```env
     ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
     ```
5. **Run the Commands:**
   * Writer replication:
     ```bash
     python -m src.phase2.run --backend anthropic --model claude-3-5-sonnet-20241022 --rounds 20 --out data/experiments/stage_a_claude
     ```
   * Independent grader cross-check:
     ```bash
     python -m src.phase2.grade --dir data/experiments/stage_a --backend anthropic --model claude-3-5-haiku-20241022
     ```
