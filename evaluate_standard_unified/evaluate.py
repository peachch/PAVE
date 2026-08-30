#!/usr/bin/env python3
"""Run and aggregate PAVE over one unified standard counter benchmark.

Epistemic-state invariant
-------------------------
Known/Unknown is determined ONLY from the N prior-only verdicts for each claim:

* Known:    all N verdicts are valid and identical.
* Unknown:  all N verdicts are valid and at least two differ.
* Excluded: at least one of the N verdicts is invalid/missing.

There is no majority threshold, consistency score, confidence weighting, or
other tunable state rule. Correctness is used only AFTER the claim-level state
is fixed, to assign each run to KK/KU/UK/UU.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import requests

PRIOR_PROMPT = (
    "Based solely on your internal knowledge, determine whether the claim should be "
    "supported or refuted. Output only 'support' or 'refute'.\n\n"
    "Claim: {claim}\nAnswer:"
)

EVIDENCE_PROMPT = (
    "Determine whether the claim should be supported or refuted using the provided information "
    "and your internal knowledge. Output only 'support' or 'refute'.\n\n"
    "Claim: {claim}\nInformation: {evidence}\nAnswer:"
)

TRUE_LABELS = {"true", "support", "supports", "supported"}
FALSE_LABELS = {"false", "refute", "refutes", "refuted"}
VALID_VERDICTS = {"support", "refute"}
NA = "na"
SKIPPED = "skipped"


def read_records(path: Path):
    text = path.read_text(encoding="utf-8").lstrip("\ufeff\n\r\t ")
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_extra_body(raw):
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--extra-body must be a JSON object")
    return value


class ChatAPI:
    def __init__(self, base_url, api_key, timeout=120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def ask(self, prompt, model, temperature, max_tokens=None, extra_body=None):
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.session.post(
            self.url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


def parse_verdict(response):
    if not response:
        return ""
    text = response.strip().lower()

    answer_match = re.search(
        r"(?:final\s+answer|answer|verdict)\s*[:：]\s*(support(?:ed)?|refut(?:e|ed)|true|false)\b",
        text,
    )
    if answer_match:
        token = answer_match.group(1)
        return "support" if token.startswith("support") or token == "true" else "refute"

    cleaned = re.sub(r"[^a-z]+", " ", text).strip()
    exact = {
        "support": "support",
        "supported": "support",
        "true": "support",
        "refute": "refute",
        "refuted": "refute",
        "false": "refute",
    }
    if cleaned in exact:
        return exact[cleaned]

    tokens = re.findall(r"\b(?:support(?:ed)?|refut(?:e|ed)|true|false)\b", text)
    mapped = {
        "support" if token.startswith("support") or token == "true" else "refute"
        for token in tokens
    }
    return next(iter(mapped)) if len(mapped) == 1 else ""


def gold_verdict(label):
    value = re.sub(r"[^a-z]", "", (label or "").lower())
    if value in TRUE_LABELS:
        return "support"
    if value in FALSE_LABELS:
        return "refute"
    return ""


def judge(verdict, label):
    if verdict not in VALID_VERDICTS:
        return NA
    target = gold_verdict(label)
    if not target:
        return NA
    return "correct" if verdict == target else "wrong"


def classify_epistemic_state(verdicts, n_runs):
    """Consistency-only claim-level state. The rule is intentionally binary."""
    if len(verdicts) != n_runs:
        return "excluded"
    if any(verdict not in VALID_VERDICTS for verdict in verdicts):
        return "excluded"
    return "known" if len(set(verdicts)) == 1 else "unknown"


def probe(api, prompt, model, temperature, max_tokens, extra_body):
    raw = api.ask(prompt, model, temperature, max_tokens, extra_body)
    return raw, parse_verdict(raw)


def validate_counter_records(records):
    required = (
        "record_id",
        "dataset",
        "claim_text",
        "cred_label",
        "evidence",
        "word_counter_evidence",
        "sentence_counter_evidence",
    )
    errors = []
    seen_ids = set()
    for index, record in enumerate(records, start=1):
        missing = [key for key in required if not str(record.get(key) or "").strip()]
        if missing:
            errors.append(f"record {index}: missing {', '.join(missing)}")
        record_id = str(record.get("record_id") or "")
        if record_id in seen_ids:
            errors.append(f"record {index}: duplicate record_id={record_id}")
        seen_ids.add(record_id)
        if not gold_verdict(str(record.get("cred_label") or "")):
            errors.append(f"record {index}: invalid cred_label={record.get('cred_label')!r}")
        if len(errors) >= 10:
            break
    if errors:
        raise SystemExit("Invalid counter benchmark:\n  " + "\n  ".join(errors))


def run_once(records, api, model, temperature, max_tokens, extra_body):
    output = []
    counts = {"correct": 0, "wrong": 0, NA: 0}

    for index, item in enumerate(records, start=1):
        claim = str(item["claim_text"]).strip()
        label = str(item["cred_label"]).strip()

        prior_raw, prior = probe(
            api,
            PRIOR_PROMPT.format(claim=claim),
            model,
            temperature,
            max_tokens,
            extra_body,
        )
        prior_judge = judge(prior, label)
        counts[prior_judge] += 1

        record = {
            "record_id": str(item["record_id"]),
            "dataset": str(item["dataset"]),
            "source_id": item.get("source_id"),
            "claim": claim,
            "label": label,
            "prior_raw": prior_raw,
            "prior_verdict": prior,
            "prior_judge": prior_judge,
            "gold_verdict": "",
            "gold_judge": SKIPPED,
            "entity_verdict": "",
            "entity_judge": SKIPPED,
            "semantic_verdict": "",
            "semantic_judge": SKIPPED,
        }

        # Correctness determines the evidence intervention for THIS run only.
        if prior_judge == "wrong":
            _, verdict = probe(
                api,
                EVIDENCE_PROMPT.format(claim=claim, evidence=item["evidence"]),
                model,
                temperature,
                max_tokens,
                extra_body,
            )
            record["gold_verdict"] = verdict
            record["gold_judge"] = judge(verdict, label)

        elif prior_judge == "correct":
            for key, field in (
                ("entity", "word_counter_evidence"),
                ("semantic", "sentence_counter_evidence"),
            ):
                _, verdict = probe(
                    api,
                    EVIDENCE_PROMPT.format(claim=claim, evidence=item[field]),
                    model,
                    temperature,
                    max_tokens,
                    extra_body,
                )
                record[f"{key}_verdict"] = verdict
                record[f"{key}_judge"] = judge(verdict, label)

        output.append(record)
        if index % 100 == 0:
            print(f"    processed {index}/{len(records)}")

    return output, counts


def default_output_dir(input_path, model):
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return Path("results") / safe_model / Path(input_path).stem


def stage_run(args):
    input_path = Path(args.input)
    records = read_records(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No input records found")
    validate_counter_records(records)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and "api.openai.com" in args.base_url:
        raise SystemExit("OPENAI_API_KEY is not set")

    api = ChatAPI(args.base_url, api_key, timeout=args.timeout)
    extra_body = parse_extra_body(args.extra_body)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.input, args.model)
    fingerprint = file_sha256(input_path)
    source_counts = {}
    for record in records:
        source_counts[record["dataset"]] = source_counts.get(record["dataset"], 0) + 1

    for run_index in range(args.n):
        run_path = output_dir / f"run_{run_index:02d}.json"
        if run_path.exists() and not args.overwrite:
            print(f"run {run_index + 1}/{args.n}: exists, skipping {run_path}")
            continue

        print(f"run {run_index + 1}/{args.n}: {len(records)} claims")
        run_records, counts = run_once(
            records,
            api,
            args.model,
            args.temperature,
            args.max_tokens,
            extra_body,
        )
        total = len(run_records)
        accuracy = counts["correct"] / total if total else 0.0
        print(
            f"    prior accuracy={accuracy:.4f}; correct={counts['correct']}, "
            f"wrong={counts['wrong']}, no-verdict={counts[NA]}"
        )
        write_json(
            run_path,
            {
                "run": run_index,
                "model": args.model,
                "temperature": args.temperature,
                "input": str(input_path),
                "input_sha256": fingerprint,
                "source_counts": source_counts,
                "records": run_records,
            },
        )

    print(f"\nRun files: {output_dir}")
    return 0


def rate(num, denom):
    return num / denom if denom else None


def correction_metrics(records):
    corrected = sum(record["gold_judge"] == "correct" for record in records)
    unchanged = sum(record["gold_judge"] == "wrong" for record in records)
    no_verdict = sum(record["gold_judge"] == NA for record in records)
    scored = corrected + unchanged
    return {
        "records": len(records),
        "scored": scored,
        "corrected": corrected,
        "unchanged": unchanged,
        "no_verdict": no_verdict,
        "CR": rate(corrected, scored),
        "OI": rate(corrected, unchanged),
    }


def persistence_metrics(records, key):
    field = f"{key}_judge"
    persisted = sum(record[field] == "correct" for record in records)
    flipped = sum(record[field] == "wrong" for record in records)
    no_verdict = sum(record[field] == NA for record in records)
    skipped = sum(record[field] == SKIPPED for record in records)
    scored = persisted + flipped
    return {
        "records": len(records),
        "scored": scored,
        "persisted": persisted,
        "flipped": flipped,
        "no_verdict": no_verdict,
        "skipped": skipped,
        "PR": rate(persisted, scored),
        "OI": rate(flipped, persisted),
    }


def margin(cr, pr):
    return cr + pr - 1 if cr is not None and pr is not None else None


def unique_claims(records):
    return len({record["record_id"] for record in records})


def summarize_scope(state_rows, retained_records):
    state_counts = {
        "known": sum(row["state"] == "known" for row in state_rows),
        "unknown": sum(row["state"] == "unknown" for row in state_rows),
        "excluded": sum(row["state"] == "excluded" for row in state_rows),
    }

    quadrants = {
        "KK": [
            record for record in retained_records
            if record["state"] == "known" and record["prior_judge"] == "correct"
        ],
        "KU": [
            record for record in retained_records
            if record["state"] == "known" and record["prior_judge"] == "wrong"
        ],
        "UK": [
            record for record in retained_records
            if record["state"] == "unknown" and record["prior_judge"] == "correct"
        ],
        "UU": [
            record for record in retained_records
            if record["state"] == "unknown" and record["prior_judge"] == "wrong"
        ],
    }

    total_valid_records = sum(len(records) for records in quadrants.values())
    quadrant_summary = {}
    for name, records in quadrants.items():
        block = {
            "claims": unique_claims(records),
            "run_records": len(records),
            "run_record_ratio": rate(len(records), total_valid_records),
        }
        if name in ("KU", "UU"):
            block["behavior"] = "correction"
            block["correction"] = correction_metrics(records)
        else:
            block["behavior"] = "persistence"
            block["persistence_entity"] = persistence_metrics(records, "entity")
            block["persistence_semantic"] = persistence_metrics(records, "semantic")
        quadrant_summary[name] = block

    def state_profile(correct_name, wrong_name):
        cr = quadrant_summary[wrong_name]["correction"]["CR"]
        pr_entity = quadrant_summary[correct_name]["persistence_entity"]["PR"]
        pr_semantic = quadrant_summary[correct_name]["persistence_semantic"]["PR"]
        return {
            "CR": cr,
            "PR_entity": pr_entity,
            "PR_semantic": pr_semantic,
            "Margin_entity": margin(cr, pr_entity),
            "Margin_semantic": margin(cr, pr_semantic),
        }

    all_correct = quadrants["KK"] + quadrants["UK"]
    all_wrong = quadrants["KU"] + quadrants["UU"]
    overall_cr = correction_metrics(all_wrong)["CR"]
    overall_pr_entity = persistence_metrics(all_correct, "entity")["PR"]
    overall_pr_semantic = persistence_metrics(all_correct, "semantic")["PR"]

    return {
        "claims_total": len(state_rows),
        "claims_known": state_counts["known"],
        "claims_unknown": state_counts["unknown"],
        "claims_excluded": state_counts["excluded"],
        "valid_run_records": total_valid_records,
        "quadrants": quadrant_summary,
        "profiles": {
            "known": state_profile("KK", "KU"),
            "unknown": state_profile("UK", "UU"),
            "overall": {
                "CR": overall_cr,
                "PR_entity": overall_pr_entity,
                "PR_semantic": overall_pr_semantic,
                "Margin_entity": margin(overall_cr, overall_pr_entity),
                "Margin_semantic": margin(overall_cr, overall_pr_semantic),
            },
        },
    }


def stage_aggregate(args):
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.input, args.model)

    runs = []
    for run_index in range(args.n):
        path = output_dir / f"run_{run_index:02d}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path}; run evaluation first")
        runs.append(read_json(path))

    fingerprints = {run.get("input_sha256") for run in runs}
    models = {run.get("model") for run in runs}
    if len(fingerprints) != 1:
        raise SystemExit("Run files were produced from different input files")
    if len(models) != 1 or args.model not in models:
        raise SystemExit("Run files do not match --model")

    by_id = {}
    for run in runs:
        run_index = run["run"]
        for record in run["records"]:
            by_id.setdefault(record["record_id"], []).append((run_index, record))

    state_rows = []
    retained_records = []

    # Step 1: fix ONE claim-level Known/Unknown state from prior consistency only.
    for record_id, entries in sorted(by_id.items()):
        entries.sort(key=lambda pair: pair[0])
        verdicts = [record["prior_verdict"] for _, record in entries]
        state = classify_epistemic_state(verdicts, args.n)
        first = entries[0][1]
        correct_runs = sum(record["prior_judge"] == "correct" for _, record in entries)
        wrong_runs = sum(record["prior_judge"] == "wrong" for _, record in entries)

        state_rows.append(
            {
                "record_id": record_id,
                "dataset": first["dataset"],
                "source_id": first.get("source_id"),
                "claim": first["claim"],
                "label": first["label"],
                "state": state,
                "prior_verdicts": verdicts,
                "prior_correct_runs": correct_runs,
                "prior_wrong_runs": wrong_runs,
                "prior_invalid_runs": args.n - correct_runs - wrong_runs,
            }
        )

        if state == "excluded":
            continue

        # Step 2: state is fixed. Correctness only assigns each run to a quadrant.
        for run_index, record in entries:
            if record["prior_judge"] not in ("correct", "wrong"):
                continue
            copied = dict(record)
            copied["run"] = run_index
            copied["state"] = state
            retained_records.append(copied)

    datasets = sorted({row["dataset"] for row in state_rows})
    overall = summarize_scope(state_rows, retained_records)
    by_dataset = {}
    for dataset in datasets:
        dataset_states = [row for row in state_rows if row["dataset"] == dataset]
        dataset_records = [record for record in retained_records if record["dataset"] == dataset]
        by_dataset[dataset] = summarize_scope(dataset_states, dataset_records)

    summary = {
        "model": args.model,
        "n_runs": args.n,
        "aggregation": "all datasets are pooled for the primary overall result; source breakdowns are diagnostic only",
        "state_definition": {
            "known": "all N prior verdicts are valid and identical",
            "unknown": "all N prior verdicts are valid and at least two differ",
            "excluded": "at least one prior verdict is invalid or missing",
            "uses_correctness": False,
            "uses_majority_threshold": False,
            "uses_consistency_weight": False,
        },
        "overall": overall,
        "by_dataset": by_dataset,
    }

    state_path = output_dir / f"claim_states_n={args.n}.jsonl"
    summary_path = output_dir / f"summary_n={args.n}.json"
    write_jsonl(state_path, state_rows)
    write_json(summary_path, summary)

    print_summary(summary)
    print(f"\nClaim states: {state_path}")
    print(f"Summary:      {summary_path}")
    return 0


def fmt_pct(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def fmt_num(value):
    return "-" if value is None else f"{value:.3f}"


def print_scope(title, scope, show_quadrants=True):
    print(f"\n=== {title} ===")
    print("Epistemic states (claim level; consistency only)")
    print(f"  Total:    {scope['claims_total']}")
    print(f"  Known:    {scope['claims_known']}")
    print(f"  Unknown:  {scope['claims_unknown']}")
    print(f"  Excluded: {scope['claims_excluded']}")

    if show_quadrants:
        print("\nQuadrants (behavior counted at run level)")
        print(
            f"{'group':<5} {'behavior':<12} {'claims':>8} {'runs':>8} "
            f"{'run-ratio':>10} {'metric-1':>11} {'metric-2':>11}"
        )
        print("-" * 76)
        for name in ("KK", "KU", "UK", "UU"):
            quadrant = scope["quadrants"][name]
            if quadrant["behavior"] == "correction":
                metric1 = "CR=" + fmt_pct(quadrant["correction"]["CR"])
                metric2 = "OI=" + fmt_num(quadrant["correction"]["OI"])
            else:
                metric1 = "PR-e=" + fmt_pct(quadrant["persistence_entity"]["PR"])
                metric2 = "PR-s=" + fmt_pct(quadrant["persistence_semantic"]["PR"])
            print(
                f"{name:<5} {quadrant['behavior']:<12} {quadrant['claims']:>8} "
                f"{quadrant['run_records']:>8} {fmt_pct(quadrant['run_record_ratio']):>10} "
                f"{metric1:>11} {metric2:>11}"
            )

    print("\nArbitration profile")
    print(f"{'state':<10} {'CR':>9} {'PR-ent':>9} {'PR-sem':>9} {'M-ent':>9} {'M-sem':>9}")
    print("-" * 58)
    for state in ("known", "unknown", "overall"):
        profile = scope["profiles"][state]
        print(
            f"{state:<10} {fmt_pct(profile['CR']):>9} "
            f"{fmt_pct(profile['PR_entity']):>9} {fmt_pct(profile['PR_semantic']):>9} "
            f"{fmt_num(profile['Margin_entity']):>9} {fmt_num(profile['Margin_semantic']):>9}"
        )


def print_summary(summary):
    print_scope("OVERALL — all source datasets pooled", summary["overall"], show_quadrants=True)

    print("\n=== SOURCE BREAKDOWN ===")
    print(
        f"{'dataset':<12} {'claims':>8} {'known':>8} {'unknown':>8} {'excluded':>9} "
        f"{'CR':>9} {'PR-ent':>9} {'PR-sem':>9}"
    )
    print("-" * 83)
    for dataset, scope in summary["by_dataset"].items():
        profile = scope["profiles"]["overall"]
        print(
            f"{dataset:<12} {scope['claims_total']:>8} {scope['claims_known']:>8} "
            f"{scope['claims_unknown']:>8} {scope['claims_excluded']:>9} "
            f"{fmt_pct(profile['CR']):>9} {fmt_pct(profile['PR_entity']):>9} "
            f"{fmt_pct(profile['PR_semantic']):>9}"
        )


def main():
    parser = argparse.ArgumentParser(description="PAVE unified standard benchmark evaluation")
    parser.add_argument("stage", choices=["run", "aggregate"])
    parser.add_argument("--input", default="data/counter.jsonl", help="Unified counter benchmark")
    parser.add_argument("--model", required=True, help="Verifier model")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N merged records; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--extra-body", default=None, help="Extra JSON fields for the API request")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")

    return stage_run(args) if args.stage == "run" else stage_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
