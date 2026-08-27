"""Temporal PCD evaluation for PAVE (correction-only)."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import pcd_common as C
from labels import classify_kb, matches_label, parse_verdict

PRIOR_PROMPT = (
    "Based solely on your internal knowledge, determine whether the claim should be "
    "supported or refuted. Provide only \"support\" or \"refute\" as the final answer "
    "without any explanation.\n\nTask:\nClaim: {claim}\nAnswer:"
)
EVIDENCE_PROMPT = (
    "Based on the provided information and your internal knowledge, determine whether the "
    "claim should be supported or refuted. Provide only \"support\" or \"refute\" as the "
    "final answer without any explanation.\n\nTask:\nClaim: {claim}\n"
    "Information: {evidence}\nAnswer:"
)
NA = "na"
SKIPPED = "skipped"


def probe(prompt: str, model: str, temperature: float, kw: dict):
    raw = C.llm_request(prompt, model_name=model, temperature=temperature, **kw)
    return raw, parse_verdict(raw)


def judge_of(verdict: str, label: str) -> str:
    if not verdict:
        return NA
    return "correct" if matches_label(verdict, label) else "wrong"


def run_once(records: list[dict], model: str, temperature: float, kw: dict) -> dict:
    out = {"records": [], "prior_correct": 0, "prior_wrong": 0, "prior_na": 0}
    for i, item in enumerate(records, 1):
        claim = (item.get("claim_text") or "").strip()
        label = (item.get("cred_label") or "").strip()
        evidence = (item.get("evidence") or "").strip()
        if not claim or not label or not evidence:
            continue

        prior_raw, prior = probe(PRIOR_PROMPT.format(claim=claim), model, temperature, kw)
        prior_judge = judge_of(prior, label)
        out["prior_" + {"correct": "correct", "wrong": "wrong", NA: "na"}[prior_judge]] += 1
        rec = {
            "claim": claim,
            "label": label,
            "event_date": item.get("event_date", ""),
            "source_url": item.get("source_url", ""),
            "prior_raw": prior_raw,
            "prior_verdict": prior,
            "prior_judge": prior_judge,
            "evidence_verdict": "",
            "evidence_judge": SKIPPED,
        }
        if prior_judge == "wrong":
            _, verdict = probe(
                EVIDENCE_PROMPT.format(claim=claim, evidence=evidence),
                model,
                temperature,
                kw,
            )
            rec["evidence_verdict"] = verdict
            rec["evidence_judge"] = judge_of(verdict, label)
        out["records"].append(rec)
        if i % 100 == 0:
            print(f"    {i}/{len(records)}")
    return out


def _rate(num: int, denom: int):
    return num / denom if denom else None


def correction_metrics(records: list[dict]) -> dict:
    wrong = [r for r in records if r["prior_judge"] == "wrong"]
    corrected = sum(r["evidence_judge"] == "correct" for r in wrong)
    unchanged = sum(r["evidence_judge"] == "wrong" for r in wrong)
    no_verdict = sum(r["evidence_judge"] == NA for r in wrong)
    cr = _rate(corrected, corrected + unchanged)
    return {
        "claims": len(records),
        "prior_wrong_population": len(wrong),
        "scored": corrected + unchanged,
        "corrected": corrected,
        "unchanged": unchanged,
        "no_verdict": no_verdict,
        "CR": cr,
        "OI": _rate(cr, 1 - cr) if cr is not None and cr < 1 else None,
    }


def _mean(values):
    vals = [x for x in values if x is not None]
    return sum(vals) / len(vals) if vals else None


def average_metrics(per_run: list[dict]) -> dict:
    return {
        "claims": _mean([x["claims"] for x in per_run]),
        "prior_wrong_population": _mean([x["prior_wrong_population"] for x in per_run]),
        "scored": _mean([x["scored"] for x in per_run]),
        "CR": _mean([x["CR"] for x in per_run]),
        "OI": _mean([x["OI"] for x in per_run]),
    }


def default_output_dir(model: str, input_path: str) -> str:
    name = Path(input_path).stem
    return os.path.join(C.OUTPUT_ROOT, "evaluation_results_temporal", model, name)


def stage_run(args, kw) -> int:
    records = C.read_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    out_dir = args.output_dir or default_output_dir(args.model, args.input)
    os.makedirs(out_dir, exist_ok=True)

    for run_idx in range(args.n):
        path = os.path.join(out_dir, f"run_{run_idx:02d}.json")
        if os.path.exists(path) and not args.overwrite:
            print(f"run {run_idx}: already done, skipping")
            continue
        print(f"run {run_idx + 1}/{args.n} over {len(records)} temporal claims")
        result = run_once(records, args.model, args.temperature, kw)
        C.write_json(path, {
            "run": run_idx,
            "model": args.model,
            "input": os.path.abspath(args.input),
            "temperature": args.temperature,
            "records": result["records"],
        })
        n = len(result["records"])
        print(
            f"    prior accuracy {result['prior_correct'] / n if n else 0:.4f}; "
            f"wrong {result['prior_wrong']}; no-verdict {result['prior_na']}"
        )
    return 0


def stage_aggregate(args) -> int:
    out_dir = args.output_dir or default_output_dir(args.model, args.input)
    runs = []
    for run_idx in range(args.n):
        path = os.path.join(out_dir, f"run_{run_idx:02d}.json")
        if not os.path.exists(path):
            print(f"missing {path}; run evaluation first", file=sys.stderr)
            return 1
        runs.append(C.read_json(path))

    by_claim: dict[str, list[tuple[int, dict]]] = {}
    for run in runs:
        for rec in run["records"]:
            by_claim.setdefault(rec["claim"], []).append((run["run"], rec))

    groups = {"known": [], "unknown": [], "excluded": []}
    for claim, entries in by_claim.items():
        entries.sort(key=lambda x: x[0])
        verdicts = [rec["prior_verdict"] for _, rec in entries]
        state = classify_kb(verdicts, n_runs=args.n, min_valid_ratio=args.min_valid_ratio)
        groups[state].append([rec for _, rec in entries])

    retained = len(groups["known"]) + len(groups["unknown"])
    summary = {
        "model": args.model,
        "input": os.path.abspath(args.input),
        "n_runs": args.n,
        "min_valid_ratio": args.min_valid_ratio,
        "claims_total": len(by_claim),
        "retained": retained,
        "excluded_refusal": len(groups["excluded"]),
        "excluded_rate": _rate(len(groups["excluded"]), len(by_claim)),
    }

    for state in ("known", "unknown"):
        claim_groups = groups[state]
        per_run = []
        for run_idx in range(args.n):
            records = [group[run_idx] for group in claim_groups if run_idx < len(group)]
            per_run.append(correction_metrics(records))
        summary[state] = {
            "claims": len(claim_groups),
            "ratio": _rate(len(claim_groups), retained),
            "per_run": per_run,
            "mean": average_metrics(per_run),
        }

    # Overall correction follows the same refusal filter: only retained Known/Unknown
    # claims are included, so excluded priors never leak back into the denominator.
    retained_groups = groups["known"] + groups["unknown"]
    overall_per_run = []
    for run_idx in range(args.n):
        retained_records = [group[run_idx] for group in retained_groups if run_idx < len(group)]
        overall_per_run.append(correction_metrics(retained_records))
    summary["overall"] = {"per_run": overall_per_run, "mean": average_metrics(overall_per_run)}
    summary_path = os.path.join(out_dir, f"summary_n={args.n}.json")
    C.write_json(summary_path, summary)
    print(f"{retained}/{len(by_claim)} claims retained -> {summary_path}")
    if args.table:
        print_table(summary)
    return 0


def print_table(summary: dict) -> None:
    def pct(x):
        return "-" if x is None else f"{100 * x:.2f}%"

    def num(x):
        return "-" if x is None else f"{x:.3f}"

    print("\nstate          ratio   prior-wrong       CR       OI")
    print("----------------------------------------------------")
    for state in ("known", "unknown"):
        s = summary[state]
        m = s["mean"]
        wrong_ratio = _rate(m["prior_wrong_population"], m["claims"]) if m["claims"] else None
        print(f"{state:12} {pct(s['ratio']):>8} {pct(wrong_ratio):>13} {pct(m['CR']):>8} {num(m['OI']):>8}")
    m = summary["overall"]["mean"]
    wrong_ratio = _rate(m["prior_wrong_population"], m["claims"]) if m["claims"] else None
    print(f"{'overall':12} {'100.00%':>8} {pct(wrong_ratio):>13} {pct(m['CR']):>8} {num(m['OI']):>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description="PAVE temporal PCD evaluation")
    ap.add_argument("stage", choices=("run", "aggregate"))
    ap.add_argument("--input", required=True, help="temporal JSONL from temporal_prepare.py")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-valid-ratio", type=float, default=1.0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    C.add_llm_args(ap)
    args = ap.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    random.seed(args.seed)
    if args.mock_llm:
        C.set_mock(True)
    kw = {"max_tokens": args.max_tokens, "extra_body": C.parse_extra_body(args.extra_body)}
    return stage_run(args, kw) if args.stage == "run" else stage_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
