# Temporal PCD in PAVE

This document describes the complete **Temporal Prior–Context Discrepancy (Temporal PCD)** pipeline in PAVE, from crawling recent events on Wikipedia to evaluating whether an LLM verifier can correct an outdated or incorrect prior belief after receiving up-to-date evidence.

The full pipeline is:

```text
Wikipedia Current Events
        │
        ▼
     crawl.py
        │
        │ raw events + candidate Wikipedia pages
        ▼
  get_evidence.py
        │
        │ retrieve + rank + validate supporting evidence
        ▼
 temporal_prepare.py
        │
        │ model-specific temporal benchmark after cutoff filtering
        ▼
evaluate_temporal.py run
        │
        │ repeated prior-only + evidence-conditioned evaluation
        ▼
evaluate_temporal.py aggregate
        │
        ▼
 Known / Unknown + Correction Rate (CR) + OI
```

Unlike the counterfactual PCD setting, Temporal PCD is **correction-only**. The benchmark contains real events that occurred after a model's knowledge cutoff. If the model's prior-only prediction is wrong, PAVE provides validated, up-to-date evidence and measures whether the model corrects its prediction.

---

## 1. Repository Files

The Temporal PCD pipeline uses the following files:

```text
PAVE/
├── pcd_common.py
├── labels.py
├── temporal_prepare.py
├── evaluate_temporal.py
├── test_temporal.py
├── requirements.txt
└── fact_benchmarks/
    └── temporal/
        ├── crawl.py
        └── get_evidence.py
```

Recommended generated-data layout:

```text
fact_benchmarks/temporal/
├── raw/
│   └── current_events_2025.json
├── enriched/
│   └── current_events_2025.json
└── prepared/
    └── <model>.jsonl
```

---

# 2. Installation

Clone PAVE and install the dependencies:

```bash
git clone https://github.com/peachch/PAVE.git
cd PAVE
pip install -r requirements.txt
```

The Temporal pipeline requires, among other packages:

```text
openai>=1.0
pandas>=1.3
tqdm
requests>=2.31
beautifulsoup4>=4.12
```

---

# 3. Configuration

## 3.1 LLM API

Evidence validation and Temporal PCD evaluation use the same OpenAI-compatible interface as the rest of PAVE.

For OpenAI:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For another OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=https://<your-endpoint>/v1
```

For a local OpenAI-compatible server such as Ollama:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

Provider-specific request fields can be passed through `--extra-body`:

```bash
--extra-body '{"enable_thinking": false}'
```

## 3.2 Wikipedia User-Agent

The crawler uses a descriptive Wikimedia User-Agent. The default is:

```text
PAVE-TemporalCrawler/1.0 (+https://github.com/peachch/PAVE)
```

For large-scale crawling, we recommend setting your own identifiable User-Agent, preferably including a contact address:

```bash
export PAVE_WIKI_USER_AGENT="PAVE-TemporalCrawler/1.0 (https://github.com/peachch/PAVE; contact@example.com)"
```

You can also pass it directly with:

```bash
--user-agent "..."
```

---

# 4. Step 1 — Crawl Temporal Events from Wikipedia

The first stage crawls Wikipedia event pages and extracts:

- event facts;
- event date;
- category;
- candidate Wikipedia article links.

Importantly, this stage **does not treat any Wikipedia paragraph as gold evidence yet**. It only acquires the event and its candidate sources.

## 4.1 Recommended: Wikipedia Current Events

To crawl all Wikipedia Current Events in 2025:

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category current_event \
    --out fact_benchmarks/temporal/raw/current_events_2025.json
```

To crawl only selected months:

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category current_event \
    --months January February March \
    --out fact_benchmarks/temporal/raw/current_events_2025_q1.json
```

For a quick test:

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category current_event \
    --months January \
    --max-results 20 \
    --out /tmp/pave_temporal_raw.json
```

## 4.2 Other Supported Crawl Modes

The crawler also supports:

```text
current_event   Wikipedia Portal:Current events
month_event     Wikipedia <Month>_<Year> pages
death_event     Wikipedia Deaths_in_<Month>_<Year> pages
```

For example:

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category death_event \
    --out fact_benchmarks/temporal/raw/death_events_2025.json
```

or:

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category month_event \
    --months January \
    --out fact_benchmarks/temporal/raw/month_events_2025_01.json
```

For the main Temporal PCD benchmark, `current_event` is the recommended source.

## 4.3 Crawler Arguments

| Argument | Description | Default |
|---|---|---:|
| `--year` | Target year | required |
| `--category` | `current_event`, `death_event`, or `month_event` | `current_event` |
| `--months` | One or more month names | all 12 months |
| `--out` | Output JSON file | required |
| `--max-results` | Maximum number of records; `0` means unlimited | `0` |
| `--delay` | Minimum delay between HTTP requests | `0.2` sec |
| `--timeout` | HTTP timeout | `20` sec |
| `--retries` | Number of HTTP retries | `4` |
| `--user-agent` | Wikimedia User-Agent | PAVE default |
| `--verbose` | Enable debug logging | off |

The crawler automatically handles retries/backoff for HTTP `429`, `500`, `502`, `503`, and `504` responses.

## 4.4 Raw Output Schema

A typical raw record looks like:

```json
{
  "Facts": "...",
  "Evidence": null,
  "Time-year": "2025",
  "Time-month": "January",
  "Time-day": 1,
  "Spotlight": "NA",
  "Category": "Politics and elections",
  "Resources": "https://en.wikipedia.org/wiki/...",
  "Candidate_resources": [
    "https://en.wikipedia.org/wiki/...",
    "https://en.wikipedia.org/wiki/..."
  ]
}
```

`Candidate_resources` preserves multiple Wikipedia candidates instead of prematurely committing to one source page.

---

# 5. Step 2 — Retrieve and Validate Supporting Evidence

The second stage converts candidate Wikipedia pages into **validated Fact–Evidence pairs**.

This step is important because the first paragraph of a related Wikipedia article is not necessarily evidence for the specific event. PAVE therefore:

1. downloads multiple candidate Wikipedia pages;
2. extracts article paragraphs;
3. ranks paragraphs against the event fact;
4. selects the best page and top relevant paragraphs;
5. applies an explicit evidence-validation gate.

## 5.1 Recommended Paper-Quality Setting

Use the `hybrid` validator:

```bash
python fact_benchmarks/temporal/get_evidence.py \
    --input fact_benchmarks/temporal/raw/current_events_2025.json \
    --output fact_benchmarks/temporal/enriched/current_events_2025.json \
    --validator hybrid \
    --validation-model gpt-4o-mini
```

`hybrid` performs:

```text
heuristic relevance gate
          +
LLM factual-support validation
```

The LLM is explicitly instructed to reject evidence that is merely topically related or generic background.

## 5.2 Fast / Cheap Retrieval Check

For debugging or measuring retrieval yield without using an LLM validator:

```bash
python fact_benchmarks/temporal/get_evidence.py \
    --input fact_benchmarks/temporal/raw/current_events_2025.json \
    --output fact_benchmarks/temporal/enriched/current_events_2025_heuristic.json \
    --validator heuristic
```

This is useful for development but is **not the recommended setting for paper-quality benchmark construction**.

## 5.3 Validator Modes

| Mode | Behavior | Recommended use |
|---|---|---|
| `hybrid` | Heuristic relevance gate + LLM support validation | **Main experiments** |
| `heuristic` | Relevance score only | Fast debugging / cheap preprocessing |
| `llm` | LLM validation for any retrieved non-empty evidence | Ablation / debugging |
| `none` | Any non-empty retrieved evidence is accepted | Legacy/debug only |

## 5.4 Evidence Retrieval Arguments

| Argument | Description | Default |
|---|---|---:|
| `--input` | Raw crawl JSON | required |
| `--output` | Enriched JSON | required |
| `--max-pages` | Maximum candidate Wikipedia pages per event | `4` |
| `--max-paragraphs` | Maximum paragraphs extracted from each page | `40` |
| `--top-k` | Number of top paragraphs retained from the best page | `2` |
| `--min-score` | Minimum heuristic relevance score | `0.12` |
| `--validator` | `hybrid`, `heuristic`, `llm`, or `none` | `hybrid` |
| `--validation-model` | LLM used for evidence support validation | `gpt-4o-mini` |
| `--temperature` | Validation LLM temperature | `0.0` |
| `--limit` | Process only the first N records; `0` means all | `0` |
| `--delay` | Minimum delay between Wikipedia requests | `0.2` sec |
| `--timeout` | HTTP timeout | `20` sec |
| `--retries` | HTTP retries | `4` |
| `--max-tokens` | LLM completion token cap | provider default |
| `--extra-body` | Provider-specific JSON request fields | none |
| `--verbose` | Debug logging | off |

For example, to test the first 50 events:

```bash
python fact_benchmarks/temporal/get_evidence.py \
    --input fact_benchmarks/temporal/raw/current_events_2025.json \
    --output /tmp/current_events_2025_enriched.json \
    --validator hybrid \
    --validation-model gpt-4o-mini \
    --limit 50
```

## 5.5 Enriched Output Schema

The enriched record contains the selected evidence and validation metadata:

```json
{
  "Facts": "...",
  "Evidence": "...",
  "Resources": "https://en.wikipedia.org/wiki/...",
  "Evidence_score": 0.63,
  "Evidence_valid": true,
  "Evidence_validator": "hybrid",
  "Evidence_retrieval_trace": [
    {
      "url": "https://en.wikipedia.org/wiki/...",
      "paragraphs": 18,
      "best_score": 0.63
    }
  ],
  "Evidence_validation_raw": "support"
}
```

If page retrieval fails, the record can additionally contain:

```text
Evidence_fetch_errors
```

If LLM validation fails, the record can contain:

```text
Evidence_validation_error
```

Failed validation is **not silently treated as valid evidence**.

---

# 6. Step 3 — Build a Model-Specific Temporal Benchmark

`temporal_prepare.py` converts validated crawl records into the schema consumed by the PAVE evaluator.

Input:

```json
{
  "Facts": "...",
  "Evidence": "...",
  "Evidence_valid": true,
  "Time-year": "2025",
  "Time-month": "January",
  "Time-day": 1
}
```

Output:

```json
{
  "claim_text": "On January 1 2025: ...",
  "cred_label": "True",
  "evidence": "...",
  "event_date": "2025-01-01",
  "category": "...",
  "source_url": "...",
  "evidence_score": 0.63,
  "evidence_validator": "hybrid"
}
```

All retained temporal events receive `cred_label="True"` because they represent real events. Temporal PCD evaluates **correction**, not counterfactual persistence.

## 6.1 Filter by Model Knowledge Cutoff

For a model whose knowledge cutoff is June 2024:

```bash
python temporal_prepare.py \
    --in fact_benchmarks/temporal/enriched/current_events_2025.json \
    --out fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --cutoff 2024-06
```

Only events **strictly after the cutoff** are kept.

A month-only cutoff is interpreted as the **end of that month**:

```text
--cutoff 2025-02
```

means:

```text
2025-02-28
```

(or February 29 in a leap year).

You can also specify an exact date:

```bash
--cutoff 2025-02-15
```

## 6.2 Merge Multiple Crawl Files

Multiple enriched files can be merged during preparation:

```bash
python temporal_prepare.py \
    --in \
        fact_benchmarks/temporal/enriched/current_events_2024.json \
        fact_benchmarks/temporal/enriched/current_events_2025.json \
    --out fact_benchmarks/temporal/prepared/<model>.jsonl \
    --cutoff YYYY-MM
```

The preparation stage automatically drops:

- records without facts;
- records without evidence;
- records with unparsable dates;
- unvalidated records;
- records whose evidence validation failed;
- records at or before the model cutoff;
- duplicate claims.

The script reports every drop category and final retained yield.

## 6.3 Legacy Data

Older PAVE temporal files may not contain `Evidence_valid`.

They can still be converted with:

```bash
python temporal_prepare.py \
    --in <legacy-file.json> \
    --out <output.jsonl> \
    --cutoff YYYY-MM \
    --allow-unvalidated
```

`--allow-unvalidated` is provided only for backward compatibility and is **not recommended for paper-quality evaluation**.

---

# 7. Step 4 — Run Temporal PCD Evaluation

Temporal PCD evaluates whether the verifier can correct an incorrect prior using up-to-date evidence.

For each claim and each run:

```text
Claim
  │
  ▼
Prior-only prediction
  │
  ├── prior correct ───────────────► no correction probe
  │
  ├── no usable verdict ───────────► refusal / excluded during aggregation
  │
  └── prior wrong
          │
          ▼
    Validated gold evidence
          │
          ▼
    Evidence-conditioned verdict
          │
          ▼
     Corrected / Unchanged
```

Run the evaluation:

```bash
python evaluate_temporal.py run \
    --input fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --model gpt-4o-mini \
    --temperature 0.3 \
    --n 10
```

For a quick subset test:

```bash
python evaluate_temporal.py run \
    --input fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --model gpt-4o-mini \
    --temperature 0.3 \
    --n 3 \
    --limit 20
```

To rerun existing run files:

```bash
--overwrite
```

## 7.1 Evaluation Arguments

| Argument | Description | Default |
|---|---|---:|
| `run / aggregate` | Evaluation stage | required |
| `--input` | Prepared temporal JSONL | required |
| `--model` | Evaluated model | `gpt-4o-mini` |
| `--temperature` | Evaluation temperature | `0.3` |
| `--n` | Number of independent repeated runs | `10` |
| `--limit` | Evaluate only the first N examples | all |
| `--min-valid-ratio` | Fraction of prior runs requiring valid verdicts | `1.0` |
| `--output-dir` | Custom result directory | automatic |
| `--overwrite` | Replace existing run files | off |
| `--table` | Print summary table during aggregation | off |
| `--seed` | Random seed used by mock/randomized utilities | `42` |
| `--max-tokens` | LLM completion token cap | provider default |
| `--extra-body` | Provider-specific request JSON | none |
| `--mock-llm` | Use offline canned model responses | off |

---

# 8. Step 5 — Aggregate Temporal Results

After all repeated runs finish:

```bash
python evaluate_temporal.py aggregate \
    --input fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --model gpt-4o-mini \
    --n 10 \
    --table
```

The evaluator first uses the repeated prior-only predictions to characterize the model's pre-evidence epistemic state:

- **Known** — all usable prior-only predictions agree;
- **Unknown** — usable prior-only predictions disagree across runs;
- **Excluded** — too few runs produce a usable verdict.

By default:

```text
min_valid_ratio = 1.0
```

so every prior-only run must return a usable verdict for the claim to remain in the evaluation.

A typical table is:

```text
state          ratio   prior-wrong       CR       OI
----------------------------------------------------
known          70.00%        15.00%   65.00%    1.857
unknown        30.00%        45.00%   78.00%    3.545
overall       100.00%        24.00%   70.00%    2.333
```

---

# 9. Temporal Metrics

## Correction Rate (CR)

For claims whose prior-only prediction is wrong:

```text
CR = corrected / (corrected + unchanged)
```

where:

- `corrected`: the model becomes correct after receiving validated evidence;
- `unchanged`: the model remains wrong after receiving validated evidence.

Evidence-conditioned responses that produce no usable verdict are recorded separately and are not silently counted as corrected or unchanged.

## Odds of Influence (OI)

For Temporal correction:

```text
OI = CR / (1 - CR)
```

A larger OI indicates that up-to-date evidence more strongly shifts an incorrect prior toward the correct verdict.

---

# 10. Output Files

Unless `--output-dir` is specified, Temporal evaluation outputs are stored under:

```text
evaluation_results_temporal/
└── <model>/
    └── <prepared_input_stem>/
        ├── run_00.json
        ├── run_01.json
        ├── ...
        ├── run_09.json
        └── summary_n=10.json
```

For example:

```text
evaluation_results_temporal/
└── gpt-4o-mini/
    └── gpt-4o-mini/
        ├── run_00.json
        ├── ...
        └── summary_n=10.json
```

A run record contains fields such as:

```json
{
  "claim": "...",
  "label": "True",
  "event_date": "2025-01-01",
  "source_url": "...",
  "prior_raw": "...",
  "prior_verdict": "support",
  "prior_judge": "correct",
  "evidence_verdict": "",
  "evidence_judge": "skipped"
}
```

When the prior is wrong, `evidence_verdict` and `evidence_judge` contain the post-evidence result.

---

# 11. Complete Recommended Workflow

The following commands run the full Temporal PCD pipeline for 2025 Wikipedia Current Events.

## 11.1 Crawl

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category current_event \
    --out fact_benchmarks/temporal/raw/current_events_2025.json
```

## 11.2 Retrieve and validate evidence

```bash
python fact_benchmarks/temporal/get_evidence.py \
    --input fact_benchmarks/temporal/raw/current_events_2025.json \
    --output fact_benchmarks/temporal/enriched/current_events_2025.json \
    --validator hybrid \
    --validation-model gpt-4o-mini
```

## 11.3 Prepare benchmark for the evaluated model

Replace the cutoff below with the actual knowledge cutoff used for that model:

```bash
python temporal_prepare.py \
    --in fact_benchmarks/temporal/enriched/current_events_2025.json \
    --out fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --cutoff 2024-06
```

## 11.4 Run repeated Temporal PCD evaluation

```bash
python evaluate_temporal.py run \
    --input fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --model gpt-4o-mini \
    --temperature 0.3 \
    --n 10
```

## 11.5 Aggregate

```bash
python evaluate_temporal.py aggregate \
    --input fact_benchmarks/temporal/prepared/gpt-4o-mini.jsonl \
    --model gpt-4o-mini \
    --n 10 \
    --table
```

---

# 12. Offline / Low-Cost Smoke Test

The complete control flow can be checked before launching a large experiment.

## 12.1 Crawl a tiny real Wikipedia subset

```bash
python fact_benchmarks/temporal/crawl.py \
    --year 2025 \
    --category current_event \
    --months January \
    --max-results 10 \
    --out /tmp/pave_temporal_raw.json
```

## 12.2 Retrieve evidence without an LLM validator

```bash
python fact_benchmarks/temporal/get_evidence.py \
    --input /tmp/pave_temporal_raw.json \
    --output /tmp/pave_temporal_enriched.json \
    --validator heuristic \
    --limit 10
```

## 12.3 Prepare

```bash
python temporal_prepare.py \
    --in /tmp/pave_temporal_enriched.json \
    --out /tmp/pave_temporal.jsonl \
    --cutoff 2024-06
```

## 12.4 Run the evaluator offline

```bash
python evaluate_temporal.py run \
    --input /tmp/pave_temporal.jsonl \
    --model mock-model \
    --n 3 \
    --mock-llm \
    --output-dir /tmp/pave_temporal_eval
```

## 12.5 Aggregate

```bash
python evaluate_temporal.py aggregate \
    --input /tmp/pave_temporal.jsonl \
    --model mock-model \
    --n 3 \
    --mock-llm \
    --output-dir /tmp/pave_temporal_eval \
    --table
```

The mock scores have no experimental meaning; this mode is only for verifying that the pipeline, schemas, and output paths work end to end.

---

# 13. Run Temporal Unit Tests

Run the temporal-specific test suite with:

```bash
python test_temporal.py
```

The tests cover key failure modes including:

- Wikipedia Current Events HTML parsing;
- relative Wikipedia URL normalization;
- day 31 parsing for death events;
- evidence relevance ranking;
- support-label parsing;
- strict `Evidence_valid` filtering;
- month-cutoff semantics;
- Temporal Correction Rate denominator handling.

The original verdict/state tests can also be run with:

```bash
python test_labels.py
```

---

# 14. Recommended Settings for Main Experiments

For the main Temporal PCD evaluation, we recommend:

```text
Event source        = Wikipedia Current Events
Evidence validator  = hybrid
Validation temp.    = 0.0
Evaluation N        = 10
Evaluation temp.    = 0.3
min_valid_ratio     = 1.0
```

The benchmark should be prepared **separately for each evaluated model** using that model's actual knowledge cutoff.

Do not use `--allow-unvalidated` or `--validator none` for final paper results unless they are part of an explicit ablation.

---

# 15. Troubleshooting

### `OPENAI_API_KEY is not set`

`hybrid` / `llm` evidence validation and normal Temporal evaluation require an OpenAI-compatible model endpoint.

Set:

```bash
export OPENAI_API_KEY=<key>
export OPENAI_BASE_URL=<endpoint>
```

For evidence-retrieval debugging without an LLM, use:

```bash
--validator heuristic
```

### No records survive `temporal_prepare.py`

Inspect the reported drop counts. Common causes are:

- `Evidence_valid=false`;
- evidence is empty;
- all events occur at or before the supplied cutoff;
- the enriched file was generated with an older pipeline and lacks `Evidence_valid`.

For legacy debugging only, use:

```bash
--allow-unvalidated
```

### Wikipedia requests are rate-limited

Increase the delay:

```bash
--delay 1.0
```

and use a descriptive User-Agent.

### Existing evaluation runs are skipped

By default, PAVE does not overwrite existing `run_XX.json` files. Use:

```bash
--overwrite
```

if you intentionally want to rerun them.

### Provider requires non-standard request fields

Pass them through:

```bash
--extra-body '{"enable_thinking": false}'
```

---

# 16. Temporal PCD vs. Counterfactual PCD

The two PAVE dimensions test different arbitration behaviors:

| Setting | Prior state | Evidence | Main behavior |
|---|---|---|---|
| Counterfactual PCD | correct or wrong | gold / Counter-Entity / Counter-Semantic | Correction + Persistence |
| Temporal PCD | potentially outdated | validated up-to-date evidence | **Correction only** |

Temporal PCD therefore does not construct counterfactual evidence. Its central question is:

> **When a model's parametric knowledge is outdated or incorrect, can validated new evidence reliably correct its prior judgment?**
