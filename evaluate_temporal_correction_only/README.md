# Temporal PAVE: Wikipedia → Gold Evidence → Correction

This folder is self-contained. Temporal PAVE is used for **new facts after a model's knowledge cutoff**, so this pipeline evaluates **correction only**.

```text
crawl.py
  collect post-cutoff Wikipedia events
        ↓
prepare.py
  retrieve relevant Wikipedia evidence
  validate that the evidence supports the claim
        ↓
evaluate.py run
  repeat prior-only prediction N times
  prior wrong → give gold evidence → test correction
  prior correct → no intervention
        ↓
evaluate.py aggregate
  Known / Unknown from prior consistency
  keep wrong-prior records only
  compute CR for Known-Wrong and Unknown-Wrong
```

There is no counterfactual evidence, persistence metric, PR, Margin, `pcd_common.py`, `labels.py`, or smoke-test code in this folder.

## 1. Install

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Set the OpenAI-compatible API used for evidence validation and model evaluation:

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
```

If needed, set a Wikipedia User-Agent:

```bash
export PAVE_WIKI_USER_AGENT="PAVE-TemporalCrawler/1.0 (contact@example.com)"
```

## 2. Crawl new Wikipedia events

Example: collect January 2025 events.

```bash
python crawl.py \
  --year 2025 \
  --months January \
  --output data/raw_2025_01.json
```

For a small first run:

```bash
python crawl.py \
  --year 2025 \
  --months January \
  --limit 20 \
  --output data/raw_2025_01.json
```

A raw record contains:

```json
{
  "claim": "...",
  "event_date": "2025-01-03",
  "category": "...",
  "candidate_urls": ["https://en.wikipedia.org/wiki/..."]
}
```

## 3. Retrieve and validate gold evidence

`prepare.py` keeps events strictly after the model cutoff, retrieves candidate Wikipedia pages, selects relevant paragraphs, and verifies that the resulting evidence supports the claim.

Example for a model whose knowledge cutoff is December 2024:

```bash
python prepare.py \
  --input data/raw_2025_01.json \
  --output data/temporal_2025_01.jsonl \
  --cutoff 2024-12 \
  --validator-model gpt-4o-mini
```

`--cutoff 2024-12` means the model is assumed to know information through December 31, 2024, so only events from January 1, 2025 onward are kept.

Successful output looks like:

```text
read                     20
...
kept                     12 -> data/temporal_2025_01.jsonl
```

Each retained record contains only the information needed for temporal correction:

```json
{
  "id": "...",
  "claim_text": "...",
  "cred_label": "True",
  "gold_evidence": "accurate Wikipedia evidence",
  "event_date": "2025-01-03",
  "source_url": "https://en.wikipedia.org/wiki/..."
}
```

## 4. Run repeated prior probes and correction

Run the target model `N` times. For every run:

```text
prior verdict
   │
   ├─ correct → stop; this run is not part of temporal correction
   │
   └─ wrong   → provide gold evidence
                 ├─ correct after evidence → correction
                 ├─ still wrong            → unchanged
                 └─ invalid output          → no-verdict
```

Example:

```bash
python evaluate.py run \
  --input data/temporal_2025_01.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10
```

Run files are saved to:

```text
results/<model>/<input-name>/run_00.json
results/<model>/<input-name>/run_01.json
...
```

## 5. Aggregate Known / Unknown and compute CR

```bash
python evaluate.py aggregate \
  --input data/temporal_2025_01.jsonl \
  --model gpt-4o-mini \
  --n 10 \
  --table
```

Known / Unknown is defined **only by consistency of the repeated prior verdicts**:

```text
Known   = all valid prior verdicts agree
Unknown = valid prior verdicts disagree across runs
Excluded = too few valid prior verdicts
```

Correctness does not determine Known / Unknown.

After the state is assigned, the temporal benchmark keeps only **wrong-prior records**:

```text
Known   + Wrong → Known-Wrong   → KU in the architecture figure
Unknown + Wrong → Unknown-Wrong → UU in the architecture figure
```

The correct-prior side is not evaluated further in Temporal PAVE.

Correction Rate is:

```text
CR = corrected / (corrected + unchanged)
```

`no-verdict` records are reported but are not included in the CR denominator.

The table looks like:

```text
Epistemic states from prior consistency (claim level)
  Known:    ...
  Unknown:  ...
  Excluded: ...

Temporal correction: wrong-prior records only
group  state           claims  wrong-runs  corrected  unchanged  no-verdict       CR
-------------------------------------------------------------------------------------
KU     Known              ...         ...        ...        ...         ...      ...
UU     Unknown            ...         ...        ...        ...         ...      ...
ALL    -                  ...         ...        ...        ...         ...      ...
```

The full result is saved to:

```text
results/<model>/<input-name>/summary_n=10.json
```

## 6. Full example

```bash
python crawl.py \
  --year 2025 \
  --output data/raw_2025.json

python prepare.py \
  --input data/raw_2025.json \
  --output data/temporal_2025.jsonl \
  --cutoff 2024-12 \
  --validator-model gpt-4o-mini

python evaluate.py run \
  --input data/temporal_2025.jsonl \
  --model gpt-4o-mini \
  --temperature 0.3 \
  --n 10

python evaluate.py aggregate \
  --input data/temporal_2025.jsonl \
  --model gpt-4o-mini \
  --n 10 \
  --table
```

For a quick real run, add `--limit 20` to `crawl.py`, `prepare.py`, and `evaluate.py run`.
