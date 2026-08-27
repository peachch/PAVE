## 🔍 Overview
# PAVE: Prior-Aware Verifier Evaluation

**PAVE** is an evaluation framework for diagnosing how large language model (LLM) verifiers arbitrate between their **parametric knowledge** and **external evidence** under **prior-context discrepancy (PCD)**.

Rather than evaluating evidence-grounded accuracy alone, PAVE first probes a verifier **before evidence is introduced** and then measures how its prediction changes after receiving either reliable or conflicting evidence.

PAVE focuses on two complementary behaviors:

* **Correction**: whether an incorrect prior prediction is corrected by accurate evidence.
* **Persistence**: whether a correct prior prediction is maintained when confronted with misleading evidence.

Repeated prior-only predictions are further used to characterize the verifier's pre-evidence epistemic state as **Known** or **Unknown**, enabling a more fine-grained diagnosis of evidence arbitration behavior.

---


## 📂 Repository Structure

```text
PAVE/
├── README.md
├── counter.py                 # Construct counterfactual evidence
├── evaluate.py                # Run and aggregate PCD evaluation
├── temporal_prepare.py        # Prepare temporal PCD datasets
├── pcd_common.py              # Shared API, configuration, and I/O utilities
├── labels.py                  # Verdict parsing and Known/Unknown classification
├── test_labels.py             # Tests for verdict parsing and state classification
├── requirements.txt
└── fact_benchmarks/
    ├── quantmp/
    │   └── final.jsonl
    ├── declare/
    │   ├── politifact_final.json
    │   └── snopes_final.json
    └── temporal/
```

The released source benchmarks include **QuanTemp**, **PolitiFact**, and **Snopes**. Counterfactual evidence constructed from these datasets is stored separately under `fact_benchmarks/counter_data/`.

---

# 🚀 Quick Start

## 1. Installation

PAVE requires Python 3.9+.

```bash
git clone https://github.com/peachch/PAVE.git
cd PAVE
pip install -r requirements.txt
```

## 2. Configuration

All models are accessed through an OpenAI-compatible API.

Set the API key and endpoint:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=<your-api-endpoint>
```

For OpenAI:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For a local OpenAI-compatible server such as Ollama:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

Provider-specific request fields can also be passed through `--extra-body`. For example:

```bash
--extra-body '{"enable_thinking": false}'
```

---

# 🔧 Constructing the Counterfactual Benchmark

PAVE constructs two types of conflicting evidence from the released source datasets.

### Counter-Entity

Judgment-relevant entities are extracted from the original evidence and replaced with entities of similar type, producing a locally modified counterfactual context.

### Counter-Semantic

An LLM generates realistic evidence supporting the **opposite** verdict of the original claim.

The experiments in the paper use GPT-4o for counterfactual construction:

```bash
python counter.py \
    --model gpt-4o \
    --tag gpt-4o \
    --temperature 0
```

Generated benchmarks are written to:

```text
fact_benchmarks/counter_data/gpt-4o/
```

By default, PAVE constructs counterfactual data for:

```text
quantemp
politifact
snopes
```

A subset can be selected explicitly:

```bash
python counter.py \
    --model gpt-4o \
    --tag gpt-4o \
    --datasets quantemp politifact
```

For a quick pipeline test:

```bash
python counter.py \
    --model gpt-4o \
    --tag test \
    --limit 20
```

---

# 🧪 Counterfactual PCD Evaluation

For every claim, PAVE first queries the verifier **without evidence** to obtain its prior prediction.

The subsequent intervention depends on whether that prior prediction is correct.

### Incorrect Prior → Correction

If the prior prediction is wrong, the verifier receives the original gold evidence.

PAVE then measures whether the evidence successfully corrects the verifier.

### Correct Prior → Persistence

If the prior prediction is correct, the verifier is independently evaluated with:

* Counter-Entity evidence
* Counter-Semantic evidence

PAVE measures whether the original correct judgment persists despite misleading context.

Run the evaluation with:

```bash
python evaluate.py run \
    --model gpt-4o-mini \
    --counter-tag gpt-4o \
    --temperature 0.3 \
    --n 10
```

To evaluate only selected datasets:

```bash
python evaluate.py run \
    --model gpt-4o-mini \
    --counter-tag gpt-4o \
    --datasets quantemp politifact \
    --temperature 0.3 \
    --n 10
```

Evaluation outputs are stored in:

```text
evaluation_results/<model>/<dataset>/
```

with one result file for each independent run.

---

# 🧠 Pre-Evidence Epistemic States

PAVE uses repeated prior-only predictions to characterize the verifier's epistemic state before evidence is introduced.

For each claim:

* **Known**: all valid prior-only predictions agree.
* **Unknown**: valid prior-only predictions disagree across runs.
* **Excluded**: too few runs produce a usable verdict.

By default, PAVE applies strict filtering:

```text
min_valid_ratio = 1.0
```

meaning that every prior-only run must return a usable verdict for an example to remain in the evaluation.

The threshold can be modified using:

```bash
--min-valid-ratio <ratio>
```

This partition allows Correction and Persistence to be analyzed separately under different pre-evidence epistemic states.

---

# 📊 Aggregating Results

After all repeated runs are complete, aggregate them with:

```bash
python evaluate.py aggregate \
    --model gpt-4o-mini \
    --n 10 \
    --table
```

The aggregated output reports results separately for the resulting epistemic-state quadrants.

The main metrics are:

| Metric          | Description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| **CR**          | **Correction Rate** — probability that accurate evidence corrects an incorrect prior |
| **PR-Entity**   | **Persistence Rate** under Counter-Entity evidence                                   |
| **PR-Semantic** | **Persistence Rate** under Counter-Semantic evidence                                 |
| **OI**          | **Odds of Influence** — odds-based measure of evidence influence                     |
| **Ratio**       | Fraction of retained examples belonging to the corresponding epistemic state         |

The Margin reported in the paper is derived from Correction and Persistence:

```text
Margin = CR + PR - 1
```

Aggregated summaries are written to:

```text
evaluation_results/<model>/<dataset>/summary_n=<N>.json
```

---

# ⏱️ Temporal PCD Data Preparation

PAVE also supports the preparation of **temporal PCD** examples for studying correction when relevant facts occur outside a model's knowledge cutoff.

Prepare a temporal dataset from a Wikipedia current-events crawl:

```bash
python temporal_prepare.py \
    --in <current-events-file.json> \
    --out fact_benchmarks/temporal/<model>.jsonl \
    --cutoff YYYY-MM
```

Temporal PCD focuses on a **correction-only** setting: the prior prediction is first obtained without evidence, and inaccurate prior predictions can subsequently be tested against up-to-date evidence.

---

# 🤖 Models Used in the Paper

The main experiments evaluate:

* GPT-4o-mini
* Gemini-2.5-flash
* DeepSeek-V3
* Qwen3-32B
* Mistral-7B
* Llama3-8B
* Phi-4

The default evaluation setting used in the main experiments is:

```text
N = 10
temperature = 0.3
```

Different providers or locally hosted models can be evaluated by setting `OPENAI_BASE_URL` and passing the corresponding model identifier through `--model`.

---

# 🛠️ Offline Smoke Test

PAVE includes a mock LLM mode for testing the counterfactual construction and evaluation pipeline without external API access.

Construct a small mock benchmark:

```bash
python counter.py \
    --mock-llm \
    --tag mock \
    --limit 20
```

Run the evaluation:

```bash
python evaluate.py run \
    --model mock-model \
    --counter-tag mock \
    --n 3 \
    --mock-llm
```

Aggregate the results:

```bash
python evaluate.py aggregate \
    --model mock-model \
    --n 3 \
    --table
```

Run the verdict parsing and knowledge-state tests:

```bash
python test_labels.py
```

---

# 📜 Citation

If you use PAVE in your research, please cite our paper:

```bibtex
@inproceedings{sun2026pave,
  title     = {Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking},
  author    = {Sun, Yuxi and Shang, Wenbo and Gao, Wei and Huang, Xin and Ma, Jing},
  booktitle = {EMNLP},
  year      = {2026}
}
```
