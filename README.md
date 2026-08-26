# PAVE — Prior-Aware Verifier Evaluation

Code for **PAVE**, an evaluation framework for studying how LLM verifiers arbitrate between parametric knowledge and external evidence under prior-context discrepancy (PCD).

PAVE evaluates two behaviors:

- **Correction**: whether an incorrect prior is corrected by accurate evidence.
- **Persistence**: whether a correct prior is maintained under misleading evidence.

Prior-only predictions are also grouped into **Known** and **Unknown** states according to their consistency across repeated runs.

## Installation

```bash
git clone https://github.com/peachch/PAVE.git
cd PAVE
pip install -r requirements.txt
```

Python 3.9+ is recommended.

## Configuration

All models are accessed through an OpenAI-compatible API.

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=<your-api-endpoint>
```

For OpenAI:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For a local OpenAI-compatible server, for example Ollama:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

## Repository Structure

```text
PAVE/
├── README.md
├── counter.py
├── evaluate.py
├── evaluate_temporal.py
├── temporal_prepare.py
├── pcd_common.py
├── labels.py
├── test_labels.py
├── requirements.txt
└── fact_benchmarks/
    ├── quantmp/
    │   └── final.jsonl
    └── declare/
        ├── politifact_final.json
        └── snopes_final.json
```

## 1. Construct the Counterfactual Benchmark

The released source files are used to construct two types of conflicting evidence:

- **Counter-Entity**: replace judgment-relevant entities in the original evidence.
- **Counter-Semantic**: generate evidence supporting the opposite verdict.

The paper uses GPT-4o to construct the benchmark:

```bash
python counter.py \
    --model gpt-4o \
    --tag gpt-4o \
    --temperature 0
```

Generated files are stored in:

```text
fact_benchmarks/counter_data/gpt-4o/
```

For a quick test:

```bash
python counter.py \
    --model gpt-4o \
    --tag test \
    --limit 20
```

## 2. Counterfactual PCD Evaluation

For each claim, the verifier is first queried without evidence.

- If the prior prediction is **wrong**, the verifier is re-queried with gold evidence to measure **Correction**.
- If the prior prediction is **correct**, the verifier is re-queried with Counter-Entity and Counter-Semantic evidence to measure **Persistence**.

Run the evaluation:

```bash
python evaluate.py run \
    --model gpt-4o-mini \
    --counter-tag gpt-4o \
    --temperature 0.3 \
    --n 10
```

Aggregate the runs:

```bash
python evaluate.py aggregate \
    --model gpt-4o-mini \
    --n 10 \
    --table
```

The main outputs are:

- **CR**: Correction Rate
- **PR**: Persistence Rate
- **OI**: Odds of Influence

The Margin reported in the paper can be calculated from CR and PR:

```text
Margin = CR + PR - 1
```

Evaluation outputs are written to:

```text
evaluation_results/<model>/<dataset>/
```

## 3. Temporal PCD Evaluation

Temporal PCD is a correction-only setting. For an event outside a model's knowledge cutoff, the verifier is first queried without evidence. If the prior prediction is wrong, it is re-queried with the accurate evidence.

Prepare a temporal dataset from the Wikipedia current-events crawl:

```bash
python temporal_prepare.py \
    --in <current-events-file.json> \
    --out fact_benchmarks/temporal/<model>.jsonl \
    --cutoff YYYY-MM
```

Run temporal evaluation:

```bash
python evaluate_temporal.py run \
    --input fact_benchmarks/temporal/<model>.jsonl \
    --model <model-name> \
    --temperature 0.3 \
    --n 10
```

Aggregate the runs:

```bash
python evaluate_temporal.py aggregate \
    --input fact_benchmarks/temporal/<model>.jsonl \
    --model <model-name> \
    --n 10 \
    --table
```

Temporal outputs are written to:

```text
evaluation_results_temporal/<model>/<dataset>/
```

## 4. Models Used in the Paper

The main experiments evaluate:

- GPT-4o-mini
- Gemini-2.5-flash
- DeepSeek-V3
- Qwen3-32B
- Mistral-7B
- Llama3-8B
- Phi-4

The main evaluation setting is:

```text
N = 10
temperature = 0.3
```

Use `OPENAI_BASE_URL` and `--model` to point the scripts to the corresponding provider or local server.

## 5. Offline Smoke Test

The repository includes a mock mode for checking the counterfactual pipeline without API access.

```bash
python counter.py --mock-llm --tag mock --limit 20

python evaluate.py run \
    --model mock-model \
    --counter-tag mock \
    --n 3 \
    --mock-llm

python evaluate.py aggregate \
    --model mock-model \
    --n 3 \
    --table

python test_labels.py
```

## Citation

```bibtex
@inproceedings{sun2026pave,
  title     = {Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking},
  author    = {Sun, Yuxi and Shang, Wenbo and Gao, Wei and Huang, Xin and Ma, Jing},
  booktitle = {EMNLP},
  year      = {2026}
}
```
