# PAVE Standard Benchmark — Unified Evaluation

This folder runs the **standard (non-temporal) PAVE benchmark** using a single merged counter dataset built from:

- QuanTemp
- PolitiFact
- Snopes

The three datasets are treated as **data sources of one evaluation benchmark**. The primary result is computed on all retained examples pooled together. Dataset-level results are also reported only as a breakdown.

```text
QuanTemp ─────┐
PolitiFact ───┼─> counter.py ─> data/counter.jsonl
Snopes ───────┘                         │
                                        v
                                evaluate.py run
                                        │
                                   N prior runs
                                        │
                                        v
                              evaluate.py aggregate
                                        │
                       Overall + per-dataset breakdown
```

There is no dependency on `pcd_common.py`, `labels.py`, test files, mock LLMs, or other PAVE Python modules.

## 1. Install

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Set an OpenAI-compatible API:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
```

## 2. Build one merged counter dataset

From the PAVE repository root, the default source paths are:

```text
fact_benchmarks/quantmp/final.jsonl
fact_benchmarks/declare/politifact_final.json
fact_benchmarks/declare/snopes_final.json
```

Build all three sources into one file:

```bash
python counter.py \
  --model gpt-4o \
  --temperature 0
```

Output:

```text
data/counter.jsonl
```

Each retained record contains:

```json
{
  "record_id": "quantemp:00000001",
  "dataset": "quantemp",
  "source_id": "...",
  "claim_text": "...",
  "cred_label": "True",
  "evidence": "gold evidence",
  "word_counter_evidence": "counter-entity evidence",
  "sentence_counter_evidence": "counter-semantic evidence"
}
```

`dataset` records where the example came from, but all records are evaluated together.

If your source files are stored elsewhere:

```bash
python counter.py \
  --quantemp <path-to-quantemp> \
  --politifact <path-to-politifact> \
  --snopes <path-to-snopes> \
  --output data/counter.jsonl \
  --model gpt-4o \
  --temperature 0
```

To process only a small number from each source during development, use:

```bash
--limit-per-dataset 20
```

## 3. Run the verifier N times

Run the merged benchmark once per repetition, not one dataset at a time:

```bash
python evaluate.py run \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10
```

For each claim in each run:

```text
Prior-only prediction
        │
        ├── Correct
        │     ├── Counter-Entity evidence   -> Persistence
        │     └── Counter-Semantic evidence -> Persistence
        │
        └── Wrong
              └── Gold evidence -> Correction
```

Run files are written to:

```text
results/<model>/counter/run_00.json
results/<model>/counter/run_01.json
...
results/<model>/counter/run_09.json
```

## 4. Known / Unknown definition

Known / Unknown is determined **once per claim** using only the N prior-only verdicts.

```text
Known
= all N prior verdicts are valid and identical

Unknown
= all N prior verdicts are valid, but at least one verdict differs

Excluded
= at least one prior verdict is invalid or missing
```

There is no majority threshold, consistency score, confidence weight, or tunable state threshold.

For `N=10`:

```text
support x 10              -> Known
refute  x 10              -> Known
support x 9 + refute x 1 -> Unknown
support x 5 + refute x 5 -> Unknown
```

Correctness does not determine Known/Unknown. After the claim-level state is fixed, each run is assigned to:

```text
                    Prior correct       Prior wrong
Known                   KK                  KU
Unknown                 UK                  UU
```

- `KK`: persistence
- `KU`: correction
- `UK`: persistence
- `UU`: correction

## 5. Aggregate the merged benchmark

```bash
python evaluate.py aggregate \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --n 10
```

The command first prints the **OVERALL** result using all three datasets pooled together. This is the primary benchmark result.

It then prints a **SOURCE BREAKDOWN** for QuanTemp, PolitiFact, and Snopes. These source-level metrics are diagnostic; the overall metric is **not** the arithmetic average of the three dataset metrics.

Outputs:

```text
results/<model>/counter/claim_states_n=10.jsonl
results/<model>/counter/summary_n=10.json
```

`summary_n=10.json` has the structure:

```json
{
  "overall": {
    "claims_total": 0,
    "claims_known": 0,
    "claims_unknown": 0,
    "claims_excluded": 0,
    "quadrants": {},
    "profiles": {}
  },
  "by_dataset": {
    "quantemp": {},
    "politifact": {},
    "snopes": {}
  }
}
```

## 6. Metrics

Wrong-prior records (`KU`, `UU`):

```text
CR = corrected / (corrected + unchanged)
```

Correct-prior records (`KK`, `UK`):

```text
PR = persisted / (persisted + flipped)
```

Persistence is reported separately for Counter-Entity and Counter-Semantic evidence.

```text
Margin = CR + PR - 1
```

No consistency weighting or dataset weighting is added.

## 7. Complete run

From the repository root:

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=https://api.openai.com/v1

python counter.py \
  --model gpt-4o \
  --temperature 0

python evaluate.py run \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10

python evaluate.py aggregate \
  --input data/counter.jsonl \
  --model gpt-4o-mini \
  --n 10
```

After these commands finish, use the `overall` block in `summary_n=10.json` as the merged PAVE result and `by_dataset` only when source-level analysis is needed.
