# Consolidation Drift

## Measuring Semantic Changes in Repeated LLM Memory Consolidation

**Project:** Consolidation Drift Research
**Status:** Initial experiment completed on one memory item; full-dataset evaluation is in progress.

---

## 1. Overview

Large Language Models (LLMs) can summarize and consolidate information into shorter memories. However, repeatedly rewriting the same memory may gradually change its original meaning.

This research investigates whether repeated memory consolidation causes **semantic drift**—a situation where a memory is not completely forgotten, but its meaning, certainty, intensity, numerical value, or polarity changes over multiple consolidation rounds.

For example:

**Original memory:**

> I slightly prefer morning meetings, but afternoon meetings are fine too.

**Possible drift:**

> I strongly prefer morning meetings.

The second statement retains the general topic but changes the strength of the original preference.

The objective of this project is to measure whether such changes occur systematically and whether they depend more on the number of consolidation rounds or the available memory budget.

---

## 2. Research Question

Does repeatedly consolidating an LLM-generated memory produce measurable and directional changes in its original meaning?

The research also investigates:

* Does repeated consolidation cause information loss?
* Does the meaning of a memory become stronger or weaker over time?
* Are some types of information more vulnerable to drift than others?
* Does semantic error depend on consolidation depth or memory budget?

---

## 3. Research Hypotheses

### H0 — Loss or Diffusion

Repeated consolidation causes information loss or accuracy degradation, primarily related to compression and memory budget.

### H1 — Directional Semantic Drift

Repeated consolidation causes a memory's meaning to move systematically in a particular direction.

For example:

```text
Slight preference
        ↓
Preference
        ↓
Strong preference
        ↓
Requirement
```

The research proposal describes directional drift using the mean signed change in a graded semantic attribute.

**Important:** These are research hypotheses, not conclusions. The experiments are designed to test them.

---

## 4. What I Have Done

### 4.1 Studied the Research Proposal

I reviewed the Consolidation Drift research proposal and identified the main problem, hypotheses, experimental design, and evaluation metrics.

The proposal focuses on repeated memory consolidation over multiple rounds and recommends testing approximately 200–400 atomic memory items.

### 4.2 Designed the Initial Dataset

I created a 20-item starter dataset named:

```text
data/atomic/atomic_v0.jsonl
```

The dataset contains five categories, with four items in each category:

| Category             | Number of items |
| -------------------- | --------------: |
| Preference intensity |               4 |
| Epistemic certainty  |               4 |
| Obligation strength  |               4 |
| Numeric magnitude    |               4 |
| Polarity / negation  |               4 |
| **Total**            |          **20** |

The items are synthetic research examples designed to test changes in specific semantic attributes.

### 4.3 Added Ground Truth

Each dataset item contains an original statement and its intended semantic value.

For graded attributes, I use a scale from 1 to 5. For example:

```text
Original:
I slightly prefer morning meetings, but afternoon meetings are fine too.

Attribute:
meeting_time_preference

True level:
2
```

The ground-truth value remains fixed throughout the experiment. Every consolidated memory is evaluated against this original value.

### 4.4 Added Dataset Provenance

I documented the sources used to design the dataset.

The atomic examples are synthetic and are not copied verbatim from external benchmarks. LongMemEval and LoCoMo were identified as resources for realistic conversational memory data.

### 4.5 Created the Research Repository

I created the initial repository structure:

```text
consolidation-drift/
├── data/
│   ├── atomic/
│   │   └── atomic_v0.jsonl
│   └── experiments/
│       └── rolling_summary/
├── docs/
│   └── dataset_sources.md
├── src/
│   └── consolidate.py
├── .env
├── .gitignore
└── README.md
```

The `.env` file is excluded from version control so that API keys are not committed to the repository.

### 4.6 Connected an LLM API

I initially used Gemini through the Google GenAI Python SDK.

The Gemini connection was successfully tested. However, the experiment encountered a free-tier request quota error during repeated requests.

I then switched to Groq and successfully tested its API connection.

The model used in the subsequent experiment was:

```text
openai/gpt-oss-20b
```

### 4.7 Tested Basic Memory Consolidation

Before running repeated consolidation, I tested whether the LLM could rewrite a memory while preserving its meaning.

**Input:**

```text
I slightly prefer morning meetings, but afternoon meetings are fine too.
```

**Output:**

```text
Slightly prefers morning meetings; finds afternoon meetings acceptable.
```

The output preserved the original preference strength and the acceptability of afternoon meetings.

### 4.8 Ran a 10-Round Consolidation Experiment

I implemented a rolling summarization process in which the output of one round becomes the input to the next.

The experiment was first tested using Gemini and later run using Groq.

The process was:

```text
Original memory
      ↓
Consolidation Round 1
      ↓
Consolidation Round 2
      ↓
Consolidation Round 3
      ↓
       ...
      ↓
Consolidation Round 10
```

I also added a delay between API requests and saved the intermediate outputs to a JSON file.

### 4.9 Saved Intermediate Results

The experiment stores the original memory and each subsequent consolidation round.

The output file for the initial test item was:

```text
data/experiments/rolling_summary/pref_001.json
```

The saved results contain:

* Round number
* Consolidated memory text

The experiment successfully saved the original memory and rounds 1–10.

---

## 5. Methodology

### 5.1 Dataset Preparation

The research begins with atomic memory items. Each item represents a statement containing a semantic attribute that can be measured.

The five initial attributes are:

1. **Preference intensity:** Slightly prefers, prefers, strongly prefers, or requires something.
2. **Epistemic certainty:** Might be true, probably true, or definitely true.
3. **Obligation strength:** Should do something, needs to do something, or must do something.
4. **Numeric magnitude:** Changes in values such as budgets, team sizes, or storage requirements.
5. **Polarity / negation:** Whether a statement is true or false, including the effect of negation.

Each item has a ground-truth value where applicable.

### 5.2 Memory Consolidation

The LLM receives only the current memory and a consolidation instruction.

The consolidator is instructed to:

* Rewrite the memory concisely.
* Preserve the original meaning.
* Preserve qualifiers and uncertainty.
* Preserve negation.
* Preserve numerical values.
* Preserve the strength or intensity of statements.
* Avoid adding unsupported information.

The original ground truth is not provided to the consolidator.

This prevents the model from directly seeing the expected evaluation value during the rewriting process.

### 5.3 Iterative Consolidation

For each item, the output of round *k* becomes the input to round *k + 1*.

Formally:

```text
m₀ → m₁ → m₂ → ... → mₖ
```

where:

* `m₀` is the original memory.
* `mₖ` is the memory after k consolidation rounds.

The initial experiment uses 10 consolidation rounds.

### 5.4 Separate Evaluation

The consolidator and evaluator have different responsibilities.

**Consolidator:**

```text
Current memory
      ↓
      LLM
      ↓
New consolidated memory
```

**Evaluator:**

```text
Original attribute scale
+
Current consolidated memory
      ↓
      LLM
      ↓
Predicted semantic level
```

The evaluator determines the semantic level of the current memory using the predefined attribute scale.

The ground truth is kept separate from the consolidation prompt.

### 5.5 Drift Calculation

For a graded attribute, signed drift is calculated as:

```text
signed_drift = predicted_level - true_level
```

For example:

```text
True level      = 2
Predicted level = 4

Signed drift = 4 - 2 = +2
```

A positive value indicates movement toward a higher level on the chosen scale. A negative value indicates movement toward a lower level.

The direction must always be interpreted according to the category's scale.

---

## 6. Initial Experiment Configuration

| Parameter               | Configuration                    |
| ----------------------- | -------------------------------- |
| Dataset                 | 20 synthetic atomic memory items |
| Initial test            | One item, `pref_001`             |
| Categories              | 5                                |
| LLM provider            | Groq                             |
| Model                   | `openai/gpt-oss-20b`             |
| Consolidation method    | Rolling summarization            |
| Consolidation rounds    | 10                               |
| Evaluation scale        | 1–5 for graded attributes        |
| Initial budget          | Fixed                            |
| Primary proposed metric | Mean signed drift                |

The full 20-item experiment and subsequent budget–depth experiments have not yet been completed.

---

## 7. Initial Results

### 7.1 Single-Item Experiment

The first memory tested was:

```text
I slightly prefer morning meetings, but afternoon meetings are fine too.
```

After 10 rounds of consolidation using Groq, the memory remained substantially consistent with its original meaning.

The final output was:

```text
I slightly prefer morning meetings; afternoons are fine.
```

### 7.2 Observation

The experiment showed that the LLM could repeatedly rewrite this particular memory while preserving:

* The preference for morning meetings.
* The fact that the preference was slight.
* The acceptability of afternoon meetings.

No obvious directional intensification was observed in this single-item experiment.

### 7.3 Interpretation

This result does **not** establish that semantic drift does not occur.

A single memory is insufficient to determine whether drift is systematic. The next stage is to run the full dataset, evaluate each round, and calculate aggregate drift.

The observed result is therefore an initial observation rather than a final research conclusion.

---

## 8. Proposed Evaluation Metrics

The research proposal identifies several metrics for studying consolidation drift.

### Primary metric

**Mean signed drift**

Measures the average directional change in the semantic attribute across items at each consolidation round.

### Additional metrics

* **Mean absolute drift:** Measures the magnitude of semantic change without considering direction.
* **Attractor distance:** Measures movement toward a recurring semantic state or level.
* **Entailment violation:** Measures whether the consolidated memory still follows from the original information.
* **Confidence–accuracy divergence:** Measures whether expressed confidence changes independently of correctness.
* **Growth exponent:** Characterizes how error changes as consolidation depth increases.
* **Survival-vs-truth:** Distinguishes whether a memory remains present from whether it remains semantically correct.

The first complete experiment should begin with predicted semantic levels and signed drift before implementing the additional metrics.

---

## 9. Proposed Budget × Depth Experiment

The next major experiment will investigate the relationship between memory budget and consolidation depth.

The matrix will vary:

* **Memory budget:** The available memory or compression constraint.
* **Consolidation depth:** The number of repeated rewriting rounds.

Conceptually:

| Memory budget / Depth |  0 |  1 |  2 |  3 | ... | 20 |
| --------------------- | -: | -: | -: | -: | --: | -: |
| Budget A              |  — |  — |  — |  — | ... |  — |
| Budget B              |  — |  — |  — |  — | ... |  — |
| Budget C              |  — |  — |  — |  — | ... |  — |
| Budget D              |  — |  — |  — |  — | ... |  — |

Each cell will contain an error or drift measurement.

The exact budget values have not yet been finalized. They will be selected as part of the experimental design rather than presented as values specified by the research proposal.

### Research comparison

* **H0:** Error is primarily determined by compression ratio or memory budget.
* **H1:** Error is primarily associated with the number of consolidation rounds.

A heatmap can be used to visualize the resulting budget–depth measurements.

---

## 10. External Research Resources

### LongMemEval

LongMemEval is a benchmark for evaluating long-term memory in conversational assistants. It provides realistic multi-session memory tasks and is relevant to the conversational layer of this research.

* Repository: https://github.com/xiaowu0162/LongMemEval
* Paper: https://arxiv.org/abs/2410.10813

### LoCoMo

LoCoMo provides long-term conversational data with multiple sessions and annotations. It can be used as a resource for designing realistic multi-session memory scenarios.

* Repository: https://github.com/snap-research/locomo

### Related Research

The research proposal also identifies work on memory consolidation, semantic intensification, memory compaction, and memory updates as relevant background.

These resources motivate the research question but do not replace the proposed controlled drift experiment.

---

## 11. Project Structure

The planned repository structure is:

```text
consolidation-drift/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── atomic/
│   │   └── atomic_v0.jsonl
│   └── experiments/
│       └── rolling_summary/
│
├── docs/
│   ├── dataset_sources.md
│   ├── research_question.md
│   ├── hypothesis.md
│   ├── dataset_design.md
│   ├── experiment_protocol.md
│   ├── metrics.md
│   └── progress.md
│
├── src/
│   ├── consolidate.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── dataset/
│   └── visualization/
│
├── experiments/
│   ├── baseline/
│   ├── rolling_summary/
│   └── results/
│
├── results/
│   ├── raw/
│   ├── tables/
│   ├── plots/
│   └── logs/
│
└── notebooks/
    └── first_drift_experiment.ipynb
```

Some directories and files are planned for future stages and may not yet exist in the repository.

---

## 12. Reproduction of the Initial Experiment

### Requirements

* Python 3
* Groq API key
* Internet access
* Python packages:

  * `groq`
  * `python-dotenv`

### Installation

```bash
pip install groq python-dotenv
```

### API Configuration

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=YOUR_API_KEY_HERE
```

Do not commit this file to GitHub.

### Run the Experiment

The consolidation script can be run from the project root using:

```bash
python src/consolidate.py
```

The initial implementation performs 10 rounds of rolling consolidation and saves the intermediate results.

### Output

The initial test produces:

```text
data/experiments/rolling_summary/pref_001.json
```

The saved JSON contains the original memory and the resulting memory from each round.

---

## 13. Limitations

The current work has several limitations:

1. Only one memory item has been tested through the complete 10-round process.
2. The full 20-item dataset has not yet been evaluated.
3. Semantic drift has not yet been calculated across the complete dataset.
4. No final mean signed drift plot has been produced.
5. The evaluator and human-annotation agreement have not yet been measured.
6. The budget–depth matrix has not yet been implemented.
7. The experiment currently uses one LLM model and one consolidation method.
8. The initial memory item did not show obvious directional drift, so no general conclusion can be drawn from it.

---

## 14. Next Steps

The next stages of the research are:

1. Connect the consolidation script to all 20 items in `atomic_v0.jsonl`.
2. Run 10 consolidation rounds for every item.
3. Save every intermediate memory.
4. Implement a separate evaluator for semantic-level prediction.
5. Calculate signed drift for each item and round.
6. Calculate mean signed drift and mean absolute drift.
7. Generate the first drift plot.
8. Inspect whether drift differs across the five categories.
9. Expand the dataset toward the proposed 200–400 items.
10. Implement the Budget × Depth experiment.
11. Add additional metrics and human evaluation.
12. Compare multiple models and consolidation methods.

---

## 15. Conclusion

This project investigates whether repeated LLM memory consolidation changes the meaning of information over time.

So far, I have:

* Studied the research proposal.
* Designed a 20-item synthetic dataset.
* Added semantic ground truth.
* Created the research repository.
* Connected and tested LLM APIs.
* Implemented rolling memory consolidation.
* Completed a 10-round experiment on one memory item.
* Saved the intermediate consolidation results.

The initial experiment preserved the meaning of the tested memory, but the complete research question remains open.

The next important milestone is to evaluate the entire dataset and determine whether repeated consolidation produces measurable, systematic semantic drift.
