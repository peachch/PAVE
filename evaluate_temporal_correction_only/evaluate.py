"""Temporal PAVE evaluation for post-cutoff facts.

Temporal PAVE is correction-only:
1. probe each claim N times without evidence;
2. use prior consistency to assign Known / Unknown;
3. only wrong prior records receive validated gold evidence;
4. compute Correction Rate (CR) for Known-Wrong and Unknown-Wrong records.

Correctness is NOT used to define Known / Unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

PRIOR_PROMPT = (
    "Based solely on your internal knowledge, determine whether the claim should be "
    "supported or refuted. Return exactly one label: support or refute.\n\n"
    "Claim: {claim}\nAnswer:"
)
EVIDENCE_PROMPT = (
    "Based on the provided information and your internal knowledge, determine whether the "
    "claim should be supported or refuted. Return exactly one label: support or refute.\n\n"
    "Claim: {claim}\nInformation: {evidence}\nAnswer:"
)
NA = "na"
SKIPPED = "skipped"


def read_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def llm_request(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    extra_body: dict | None,
) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
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
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = requests.post(
        f"{base}/chat/completions", headers=headers, json=payload, timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"] or "").strip()


def parse_verdict(text: str) -> str:
    normalized = re.sub(r"[^a-z]", "", (text or "").lower())
    if normalized.startswith("support") or normalized == "true":
        return "support"
    if normalized.startswith("refute") or normalized == "false":
        return "refute"
    return ""


def gold_verdict(label: str) -> str:
    normalized = re.sub(r"[^a-z]", "", (label or "").lower())
    if normalized in {"true", "support", "supports", "supported"}:
        return "support"
    if normalized in {"false", "refute", "refutes", "refuted"}:
        return "refute"
    return ""


def judge(verdict: str, label: str) -> str:
    if not verdict:
        return NA
    target = gold_verdict(label)
    if not target:
        return NA
    return "correct" if verdict == target else "wrong"


def classify_epistemic_state(
    verdicts: list[str], n_runs: int, min_valid_ratio: float
) -> str:
    """Assign state from repeated prior consistency only.

    Known   = enough valid prior verdicts and all valid verdicts agree.
    Unknown = enough valid prior verdicts and valid verdicts disagree.
    Excluded= too few valid prior verdicts.
    """
    valid = [verdict for verdict in verdicts if verdict]
    if n_runs <= 0 or not valid or len(valid) < min_valid_ratio * n_runs:
        return "excluded"
    return "known" if len(set(valid)) == 1 else "unknown"


def probe(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    extra_body: dict | None,
) -> tuple[str, str]:
    raw = llm_request(prompt, model, temperature, max_tokens, extra_body)
    return raw, parse_verdict(raw)


def run_once(
    records: list[dict],
    model: str,
    temperature: float,
    max_tokens: int | None,
    extra_body: dict | None,
) -> list[dict]:
    """Run one prior probe per claim and correct only wrong priors."""
    output: list[dict] = []
    for index, item in enumerate(records, 1):
        claim = str(item.get("claim_text") or "").strip()
        label = str(item.get("cred_label") or "").strip()
        gold = str(item.get("gold_evidence") or "").strip()
        if not claim or not label or not gold:
            continue

        prior_raw, prior = probe(
            PRIOR_PROMPT.format(claim=claim),
            model,
            temperature,
            max_tokens,
            extra_body,
        )
        prior_judge = judge(prior, label)

        record = {
            "id": item.get("id") or claim,
            "claim": claim,
            "label": label,
            "event_date": item.get("event_date", ""),
            "source_url": item.get("source_url", ""),
            "prior_raw": prior_raw,
            "prior_verdict": prior,
            "prior_judge": prior_judge,
            "post_raw": "",
            "post_verdict": "",
            "post_judge": SKIPPED,
            "behavior": SKIPPED,
        }

        # Temporal PAVE keeps only the correction branch.
        if prior_judge == "wrong":
            post_raw, post = probe(
                EVIDENCE_PROMPT.format(claim=claim, evidence=gold),
                model,
                temperature,
                max_tokens,
                extra_body,
            )
            post_judge = judge(post, label)
            record["post_raw"] = post_raw
            record["post_verdict"] = post
            record["post_judge"] = post_judge
            if post_judge == "correct":
                record["behavior"] = "correction"
            elif post_judge == "wrong":
                record["behavior"] = "unchanged"
            else:
                record["behavior"] = "no_verdict"

        output.append(record)
        if index % 100 == 0:
            print(f"    {index}/{len(records)}")
    return output


def rate(num: int | float, denom: int | float):
    return num / denom if denom else None


def correction_metrics(records: list[dict]) -> dict:
    """Compute CR over wrong-prior records only."""
    wrong = [record for record in records if record["prior_judge"] == "wrong"]
    corrected = sum(record["post_judge"] == "correct" for record in wrong)
    unchanged = sum(record["post_judge"] == "wrong" for record in wrong)
    no_verdict = sum(record["post_judge"] == NA for record in wrong)
    scored = corrected + unchanged
    return {
        "wrong_prior_records": len(wrong),
        "scored": scored,
        "corrected": corrected,
        "unchanged": unchanged,
        "no_verdict": no_verdict,
        "CR": rate(corrected, scored),
    }


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model)


def default_output_dir(model: str, input_path: str) -> Path:
    return Path("results") / safe_model_name(model) / Path(input_path).stem


def stage_run(args) -> int:
    records = read_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("input JSONL contains no records")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.model, args.input)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_body = json.loads(args.extra_body) if args.extra_body else None

    for run_index in range(args.n):
        path = output_dir / f"run_{run_index:02d}.json"
        if path.exists() and not args.overwrite:
            print(f"run {run_index + 1}/{args.n}: exists, skipping {path}")
            continue

        print(f"run {run_index + 1}/{args.n} over {len(records)} claims")
        result = run_once(
            records,
            args.model,
            args.temperature,
            args.max_tokens,
            extra_body,
        )
        write_json(
            path,
            {
                "run": run_index,
                "model": args.model,
                "input": str(Path(args.input).resolve()),
                "temperature": args.temperature,
                "records": result,
            },
        )

        correct = sum(record["prior_judge"] == "correct" for record in result)
        wrong = sum(record["prior_judge"] == "wrong" for record in result)
        no_prior = sum(record["prior_judge"] == NA for record in result)
        corrected = sum(record["behavior"] == "correction" for record in result)
        print(
            f"    prior: correct={correct}, wrong={wrong}, no-verdict={no_prior}; "
            f"corrected={corrected}/{wrong} wrong priors"
        )
    return 0


def stage_aggregate(args) -> int:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.model, args.input)
    )
    runs: list[dict] = []
    for run_index in range(args.n):
        path = output_dir / f"run_{run_index:02d}.json"
        if not path.exists():
            print(f"missing {path}; run evaluation first", file=sys.stderr)
            return 1
        runs.append(json.loads(path.read_text(encoding="utf-8")))

    by_id: dict[str, list[tuple[int, dict]]] = {}
    for run in runs:
        for record in run["records"]:
            by_id.setdefault(str(record["id"]), []).append((int(run["run"]), record))

    states: dict[str, str] = {}
    for claim_id, entries in by_id.items():
        entries.sort(key=lambda item: item[0])
        verdicts = [record["prior_verdict"] for _, record in entries]
        states[claim_id] = classify_epistemic_state(
            verdicts, args.n, args.min_valid_ratio
        )

    claim_counts = {
        "known": sum(state == "known" for state in states.values()),
        "unknown": sum(state == "unknown" for state in states.values()),
        "excluded": sum(state == "excluded" for state in states.values()),
    }

    known_wrong: list[dict] = []
    unknown_wrong: list[dict] = []
    known_wrong_claims: set[str] = set()
    unknown_wrong_claims: set[str] = set()

    for claim_id, entries in by_id.items():
        state = states[claim_id]
        if state == "excluded":
            continue
        for _, record in entries:
            if record["prior_judge"] != "wrong":
                continue
            tagged = dict(record)
            tagged["epistemic_state"] = state
            if state == "known":
                known_wrong.append(tagged)
                known_wrong_claims.add(claim_id)
            else:
                unknown_wrong.append(tagged)
                unknown_wrong_claims.add(claim_id)

    known_metrics = correction_metrics(known_wrong)
    unknown_metrics = correction_metrics(unknown_wrong)
    overall_metrics = correction_metrics(known_wrong + unknown_wrong)

    # In the architecture figure these are the Wrong-column groups:
    # KU = Known-Unknowns, UU = Unknown-Unknowns.
    summary = {
        "model": args.model,
        "input": str(Path(args.input).resolve()),
        "n_runs": args.n,
        "min_valid_ratio": args.min_valid_ratio,
        "claims_total": len(by_id),
        "epistemic_states": claim_counts,
        "correction_only": {
            "known_wrong": {
                "figure_label": "KU",
                "name": "Known-Unknowns",
                "contributing_claims": len(known_wrong_claims),
                **known_metrics,
            },
            "unknown_wrong": {
                "figure_label": "UU",
                "name": "Unknown-Unknowns",
                "contributing_claims": len(unknown_wrong_claims),
                **unknown_metrics,
            },
            "overall_wrong": {
                "contributing_claims": len(known_wrong_claims | unknown_wrong_claims),
                **overall_metrics,
            },
        },
    }

    summary_path = output_dir / f"summary_n={args.n}.json"
    write_json(summary_path, summary)
    print(f"wrote {summary_path}")
    if args.table:
        print_table(summary)
    return 0


def fmt_pct(value) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def print_table(summary: dict) -> None:
    states = summary["epistemic_states"]
    print("\nEpistemic states from prior consistency (claim level)")
    print(f"  Known:    {states['known']}")
    print(f"  Unknown:  {states['unknown']}")
    print(f"  Excluded: {states['excluded']}")

    print("\nTemporal correction: wrong-prior records only")
    print("group  state           claims  wrong-runs  corrected  unchanged  no-verdict       CR")
    print("-------------------------------------------------------------------------------------")
    for key, label, state_name in (
        ("known_wrong", "KU", "Known"),
        ("unknown_wrong", "UU", "Unknown"),
    ):
        block = summary["correction_only"][key]
        print(
            f"{label:5}  {state_name:12}  {block['contributing_claims']:6d}  "
            f"{block['wrong_prior_records']:10d}  {block['corrected']:9d}  "
            f"{block['unchanged']:9d}  {block['no_verdict']:10d}  {fmt_pct(block['CR']):>8}"
        )

    overall = summary["correction_only"]["overall_wrong"]
    print(
        f"ALL    {'-':12}  {overall['contributing_claims']:6d}  "
        f"{overall['wrong_prior_records']:10d}  {overall['corrected']:9d}  "
        f"{overall['unchanged']:9d}  {overall['no_verdict']:10d}  {fmt_pct(overall['CR']):>8}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporal PAVE correction-only evaluation")
    parser.add_argument("stage", choices=("run", "aggregate"))
    parser.add_argument("--input", required=True, help="JSONL produced by prepare.py")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-valid-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--extra-body", default=None, help="JSON object passed to the OpenAI-compatible API"
    )
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if not 0 < args.min_valid_ratio <= 1:
        raise SystemExit("--min-valid-ratio must be in (0, 1]")
    return stage_run(args) if args.stage == "run" else stage_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
