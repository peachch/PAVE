# PAVE — Prior-Aware Verifier Evaluation

Code for **"Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based
Fact-Checking"** (Sun, Shang, Gao, Huang, Ma).

Conventional fact-checking evaluation asks *what* verdict an LLM produced. PAVE asks *how it got
there*: when a verifier's parametric prior disagrees with the retrieved evidence — **prior-context
discrepancy (PCD)** — does it persist in a correct prior under misleading evidence, and does it
correct a wrong prior when accurate evidence arrives?

The pipeline has three steps:

```
    gold claim + evidence
              │
              ▼
   ┌──────────────────────┐   counter-entity   ─┐
   │  counter.py          │   counter-semantic  ├─ built ONCE, with one generator model
   └──────────────────────┘                    ─┘
              │
              ▼  fact_bechmarks/counter_data/<tag>/
   ┌──────────────────────┐   probe the prior, then intervene with evidence
   │  evaluate.py run     │   chosen by that prior. N independent runs.
   └──────────────────────┘
              │
              ▼  evaluation_results/<model>/<dataset>/run_NN.json
   ┌──────────────────────┐   stratify by Knowledge Boundary,
   │ evaluate.py aggregate│   report CR / PR / Margin / OI
   └──────────────────────┘
```

---

## 1. Install

```bash
pip install -r requirements.txt      # openai, pandas, tqdm
```

Python 3.9+. There is no provider-specific dependency: every model is reached through one
OpenAI-compatible client.

## 2. Configuration

Everything comes from the environment. Nothing is hardcoded, and no file should ever contain a key.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required unless `--mock-llm` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `PCD_DATA_ROOT` | script directory | root holding `fact_bechmarks/previous_data/` |
| `PCD_OUTPUT_ROOT` | script directory | root that outputs are written under |
| `PCD_MAX_RETRIES` | `4` | attempts per request before giving up |
| `PCD_THROTTLE` | `0.05` | seconds slept after each successful call |
| `MOCK_LLM` | `false` | canned responses, no network |

**Switching provider is a configuration change, not a code change.** There are no per-model
branches; point `OPENAI_BASE_URL` at whatever serves the model:

```bash
# OpenAI
export OPENAI_BASE_URL=https://api.openai.com/v1

# a local Ollama server (it exposes an OpenAI-compatible endpoint)
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
python evaluate.py run --model llama3:8b ...

# any other gateway
export OPENAI_BASE_URL=https://<host>/compatible-mode/v1
```

Non-standard request fields go through `--extra-body`, so a vendor quirk never becomes a hardcoded
model name:

```bash
python evaluate.py run --model qwen3-32b --extra-body '{"enable_thinking": false}'
```

`--temperature` is always sent, for every model, on every call.

## 3. Input data

`fact_bechmarks/previous_data/` holds the gold claim/evidence pairs, one JSON object per line:

```json
{"claim_text": "...", "cred_label": "True", "evidence": "..."}
```

`cred_label` takes either vocabulary — `true`/`false` or `support`/`refute`, case- and
punctuation-insensitive. Expected paths:

```
fact_bechmarks/previous_data/quantemp/final.jsonl
fact_bechmarks/previous_data/declare/politifact_final.jsonl
fact_bechmarks/previous_data/declare/snopes_final.jsonl
```

The source corpora are not redistributed here: QuanTemp from Anand et al. (2024), PolitiFact and
Snopes from Popat et al. (2017, 2018).

## 4. Step 1 — build the benchmark

```bash
export OPENAI_API_KEY=sk-...
python counter.py --model gpt-4o --tag gpt-4o --limit 0
```

Each retained claim gains two counterfactual variants:

* **`sentence_counter_evidence`** — counter-semantic (§3.2, Table 15). Fabricated evidence pointing
  the opposite way from the gold label: a true claim gets a fabricated refutation, a false claim a
  fabricated support. Conditioned on the claim, so direction is controlled by the label.
* **`word_counter_evidence`** — counter-entity (§3.2, Table 16). The gold evidence with its
  judgment-bearing entities replaced by same-type, different-value entities.

A record is dropped unless **both** succeed, so the two counterfactual sets cover the same claims.
Counter-entity construction requires at least `MIN_ENTITIES = 5` extractable entities (§D) and
verifies that substitution actually changed the text — an entity that does not occur verbatim in
the evidence would otherwise leave the "counterfactual" identical to the gold evidence, which then
scores as the verifier resisting misleading context. Every drop is counted and the yield is printed:

```
[politifact] read 1020  ->  kept 847 (83.0%)  .../politifact_counter.jsonl
        dropped (entity_failed): 151
        dropped (semantic_failed): 22
```

### `--tag`: why the benchmark is not keyed by verifier

`--tag` names a **dataset build**, not a model. The paper builds the whole benchmark once with
GPT-4o and evaluates all seven verifiers against that same fixed data; `evaluate.py --counter-tag`
selects which build to read, independently of `--model`:

```bash
python counter.py  --model gpt-4o --tag gpt-4o          # build once
python evaluate.py run --model gpt-4o-mini --counter-tag gpt-4o
python evaluate.py run --model llama3:8b   --counter-tag gpt-4o
python evaluate.py run --model phi4        --counter-tag gpt-4o
```

No copying, and cross-model comparisons are on identical data by construction. Each build directory
carries a `meta.json` recording the generator model, temperature, entity threshold and per-dataset
yield, so a result set can always be traced back to the data it was produced from.

Useful flags: `--limit N` (default 0 = the full dataset), `--datasets`, `--entity-separator newline`
(the paper's prompt asks for a comma-separated entity list, which cannot represent an entity that
itself contains a comma — `newline` asks for one per line instead), `--dump-csv` (off by default:
the CSV holds the same fabricated evidence and is an easy accidental-distribution path).

## 5. Step 2 — run the evaluation

```bash
python evaluate.py run --model gpt-4o-mini --counter-tag gpt-4o --temperature 0.3 --n 10
```

For each of `--n` independent runs, each claim is probed with the **claim alone**, and the result
selects the intervention:

| prior-only verdict | scenario | second probe |
|---|---|---|
| wrong | **Correction** | re-ask with the gold evidence |
| correct | **Persistence** | re-ask with counter-entity, then with counter-semantic |
| unparseable | — | recorded as `na`, excluded downstream |

Both persistence conditions use the **same prompt**; only the evidence differs, so the counter-entity
vs counter-semantic comparison is not confounded by prompt wording.

One file per run (`run_00.json` … `run_09.json`), each holding the raw response alongside the parsed
verdict. Runs that already have a file are skipped, so an interrupted sweep resumes; pass
`--overwrite` to redo them. Keeping `prior_raw` means a change to the parser can be re-applied to
existing runs without re-querying the model.

`--dimension temporal` evaluates acquisition only: claims whose prior is already correct are not
probed at all, because persisting in an outdated prior is not a meaningful persistence result
(§3.2). Without this, those claims would be sent down the persistence branch with empty evidence and
their verdicts recorded as persistence results.

### Per-claim record

| Field | Values |
|---|---|
| `prior_raw` | the model's untouched response |
| `prior_verdict` | `support` \| `refute` \| `""` |
| `prior_judge` | `correct` \| `wrong` \| `na` |
| `gold_judge` | `correct` \| `wrong` \| `na` \| `skipped` |
| `entity_judge` | `correct` \| `wrong` \| `na` \| `skipped` |
| `semantic_judge` | `correct` \| `wrong` \| `na` \| `skipped` |

`skipped` means the branch did not apply to this record; `na` means the model produced no usable
verdict. They are distinct on purpose — only `na` is an observation about the model.

## 6. Step 3 — aggregate

```bash
python evaluate.py aggregate --model gpt-4o-mini --n 10 --table
```

Claims are grouped by prior-only consistency across the runs:

* **known** — every run produced the same, parseable verdict
* **unknown** — the runs disagree
* **excluded** — too few runs produced a usable verdict at all

The third case is the paper's strict refusal filter (§3.2: *"excluding instances where the model
abstains ... retaining only samples where the model exhibits a definite prior"*). A claim answered
once and refused nine times is evidence of neither confidence nor uncertainty, so it leaves the
benchmark rather than being scored as maximally confident. `--min-valid-ratio` controls the gate;
`1.0` (default) requires all N runs to parse. The exclusion rate is reported per dataset — it is a
result in its own right, since abstention rates differ sharply across models.

`--table` prints the Table 1 block:

```
=== gpt-4o-mini  /  politifact  (n=10, retained 981/1020, refusal-excluded 3.82%) ===
quadrant                 ratio      CR  PR-ent  PR-sem  Margin      OI
----------------------------------------------------------------------
Known-Knows (KK)        61.73%       -  40.23%  31.90% -34.98%   1.486
Known-Unknows (KU)      13.92%  24.73%       -       -       -   0.329
Unknown-Knows (UK)       1.78%       -  25.33%  19.10% -49.88%   2.948
Unknown-Unknows (UU)    22.57%  57.59%       -       -       -   0.345
```

### Metrics

Computed per run and averaged, inside each KB state:

```
CR       = corrected / (corrected + unchanged)          on claims with a wrong prior     (§3.1)
PR       = persisted / (persisted + flipped)            on claims with a correct prior   (§3.1)
Margin   = CR + PR - 1                                                                   (§3.1)
OI       = CR / (1 - CR)        on the correction side                                   (§3.1)
         = (1 - PR) / PR        on the persistence side
```

**Unparseable verdicts are excluded from every denominator**, and counted separately as
`no_verdict`. They are not evidence that the model changed its answer, so scoring them as such would
inflate CR and depress PR at the same time — and those two errors compound in `Margin`.

Because a claim carries two intervention probes, the counter-entity and counter-semantic
denominators can differ when one of them returns `na`. Both `scored` counts are written to the
summary; report them when the two PR columns are compared.

Full per-run detail lands in `evaluation_results/<model>/<dataset>/summary_n=<N>.json`.

## 7. Offline smoke test

No key, no network, canned responses:

```bash
python counter.py  --mock-llm --tag mock --limit 20
python evaluate.py run       --model mock-model --counter-tag mock --n 3 --mock-llm
python evaluate.py aggregate --model mock-model --n 3 --table
python test_labels.py        # 33 parser assertions
```

## 8. Files

| File | Role |
|---|---|
| `counter.py` | Step 1 — counterfactual benchmark construction |
| `evaluate.py` | Steps 2 and 3 — `run` and `aggregate` |
| `pcd_common.py` | configuration, paths, the single LLM request path |
| `labels.py` | verdict parsing and Knowledge-Boundary classification |
| `test_labels.py` | parser regression tests |
| `temporal_prepare.py` | converts the Wikipedia current-events crawl into the input schema |

### `labels.py`

`parse_verdict` checks for a refusal, then takes the **last** verdict cue in the response (a model
that reasons before answering states its conclusion last), then checks the clause around that cue
for negation. This matters more than it sounds: `"The claim should not be supported"` contains
`support` and not `refute`, and a substring parser reads it as the opposite of what the model said.
`classify_kb` implements the KB split with the refusal gate described in §6.

## 9. Reproducing the paper

| Setting | Value |
|---|---|
| Benchmark generator | GPT-4o, temperature 0, one build shared by all verifiers |
| Runs | N = 10 |
| Verifier temperature | 0.3 |
| Claims | full datasets (`--limit 0`) |
| Verifiers | GPT-4o-mini, Gemini-2.5-flash, Deepseek-v3, Qwen3-32B, Mistral-7B, Llama3-8B, Phi-4 |

```bash
python counter.py --model gpt-4o --tag gpt-4o --temperature 0 --limit 0

for M in gpt-4o-mini gemini-2.5-flash deepseek-v3 qwen3-32b mistral:7b llama3:8b phi4; do
  python evaluate.py run       --model "$M" --counter-tag gpt-4o --temperature 0.3 --n 10
  python evaluate.py aggregate --model "$M" --n 10 --table
done
```

Set `OPENAI_BASE_URL` per model as needed; the loop body does not change.

## 10. Data handling

* **The counterfactual evidence is deliberately fabricated misinformation about real people and
  events.** Every generated record carries `"is_counterfactual": true` and each build directory a
  `meta.json` warning, so the text stays identifiable if it is ever flattened into a plain
  `evidence` field downstream. Keep both when redistributing.
* Wikipedia-derived temporal data is CC BY-SA 4.0 — attribute and share-alike. QuanTemp, PolitiFact
  and Snopes each carry their own terms; check them before publishing a derived set.
* No credentials belong in these files. Before making the repository public, check the history
  (`git log -p -S 'sk-' -- '*.py'`), rotate any key that ever appeared in it, and add `.env`,
  `*.csv` and `evaluation_results/` to `.gitignore`.

## 11. Citation

```bibtex
@inproceedings{sun2026pave,
  title     = {Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking},
  author    = {Sun, Yuxi and Shang, Wenbo and Gao, Wei and Huang, Xin and Ma, Jing},
  booktitle = {EMNLP},
  year      = {2026}
}
```

Datasets: https://doi.org/10.5281/zenodo.18151788
