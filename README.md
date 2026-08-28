# PAVE: Prior-Aware Verifier Evaluation

## 🔍 Overview

**PAVE** is an evaluation framework for diagnosing how large language model (LLM) verifiers arbitrate between their **parametric knowledge** and **external evidence** under **prior-context discrepancy (PCD)**.

PAVE first probes a verifier **without evidence** to characterize its prior state, and then introduces evidence to measure two behaviors:

- **Correction** — whether correct evidence can correct an incorrect prior.
- **Persistence** — whether a correct prior persists under conflicting evidence.

PAVE contains two evaluation settings:

1. **Standard PAVE** — evaluates both correction and persistence on a unified benchmark constructed from QuanTemp, PolitiFact, and Snopes.
2. **Temporal PAVE** — evaluates correction only on newly collected Wikipedia facts after a model's knowledge cutoff.

---

## 📂 Repository Structure

```text
PAVE/
├── README.md
├── evaluate_standard_unified/
│   ├── README.md
│   ├── counter.py
│   ├── evaluate.py
│   └── requirements.txt
├── evaluate_temporal_correction_only/
│   ├── README.md
│   ├── crawl.py
│   ├── prepare.py
│   ├── evaluate.py
│   └── requirements.txt
└── img/
```

Each evaluation folder is self-contained. See its local `README.md` for detailed arguments and data formats.

---

## 🧠 Known / Unknown

Known and Unknown are defined **only by the consistency of repeated prior-only predictions**.

```text
Known
= repeated prior verdicts are consistent

Unknown
= at least one prior verdict differs across runs
```

Correctness does **not** determine Known/Unknown.

After the epistemic state is fixed, prior correctness determines the evaluation quadrant:

```text
                    Prior Correct       Prior Wrong
Known                   KK                  KU
Unknown                 UK                  UU
```

- `KK`, `UK` → Persistence
- `KU`, `UU` → Correction

---

# 🚀 Quick Start

## 1. Installation

Python 3.9+ is recommended.

Install dependencies for both evaluation settings:

```bash
pip install -r evaluate_standard_unified/requirements.txt
pip install -r evaluate_temporal_correction_only/requirements.txt
```

Configure an OpenAI-compatible API:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 2. Standard PAVE

Standard PAVE first merges **QuanTemp, PolitiFact, and Snopes** into one counterfactual benchmark.

### Build the unified counter benchmark

```bash
python evaluate_standard_unified/counter.py \
  --quantemp <path-to-quantemp> \
  --politifact <path-to-politifact> \
  --snopes <path-to-snopes> \
  --output data/counter.jsonl \
  --model gpt-4o \
  --temperature 0
```

The constructed benchmark contains both:

- **Counter-Entity evidence**
- **Counter-Semantic evidence**

### Run repeated evaluation

```bash
python evaluate_standard_unified/evaluate.py run \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10
```

For each run:

```text
Prior Correct → Counterfactual Evidence → Persistence
Prior Wrong   → Gold Evidence           → Correction
```

### Aggregate results

```bash
python evaluate_standard_unified/evaluate.py aggregate \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --n 10
```

The primary result is computed on the **three datasets pooled together**. Per-dataset results are reported only as diagnostic breakdowns.

Main metrics include **CR**, **PR-Entity**, **PR-Semantic**, **OI**, and **Margin**.

See [`evaluate_standard_unified/README.md`](evaluate_standard_unified/README.md) for details.

---

## 3. Temporal PAVE

Temporal PAVE targets **new facts after a model's knowledge cutoff** and therefore evaluates **correction only**.

### Collect Wikipedia events

```bash
python evaluate_temporal_correction_only/crawl.py \
  --year 2025 \
  --output data/raw_2025.json
```

### Retrieve and validate gold evidence

```bash
python evaluate_temporal_correction_only/prepare.py \
  --input data/raw_2025.json \
  --output data/temporal_2025.jsonl \
  --cutoff 2024-12 \
  --validator-model gpt-4o-mini
```

### Run repeated evaluation

```bash
python evaluate_temporal_correction_only/evaluate.py run \
  --input data/temporal_2025.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10
```

Temporal intervention keeps only the wrong-prior path:

```text
Prior Wrong → Gold Evidence → Correction
```

### Aggregate results

```bash
python evaluate_temporal_correction_only/evaluate.py aggregate \
  --input data/temporal_2025.jsonl \
  --model gpt-4o-mini \
  --n 10 \
  --table
```

The main temporal result is the **Correction Rate (CR)** for wrong-prior examples under Known and Unknown states.

See [`evaluate_temporal_correction_only/README.md`](evaluate_temporal_correction_only/README.md) for details.

---

## 📊 Evaluation Settings

| Setting | Data | Evidence Intervention | Main Metrics |
| --- | --- | --- | --- |
| **Standard PAVE** | QuanTemp + PolitiFact + Snopes | Gold + Counter-Entity + Counter-Semantic | CR, PR, OI, Margin |
| **Temporal PAVE** | Post-cutoff Wikipedia facts | Gold evidence only | CR |

---

## 📜 Citation

If you use PAVE in your research, please cite:

```bibtex
@inproceedings{sun2026pave,
  title     = {Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking},
  author    = {Sun, Yuxi and Shang, Wenbo and Gao, Wei and Huang, Xin and Ma, Jing},
  booktitle = {EMNLP},
  year      = {2026}
}
```
