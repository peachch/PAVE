"""PAVE counterfactual evaluation: prior probe, intervention, and aggregation."""

import argparse
import os
import sys

import pcd_common as C
from labels import classify_kb, matches_label, parse_verdict

DATASETS = {
    "quantemp": "quantemp_counter.jsonl",
    "politifact": "politifact_counter.jsonl",
    "snopes": "snopes_counter.jsonl",
}

PRIOR_PROMPT = (
    "Based solely on your internal knowledge, determine whether the claim should be "
    "supported or refuted. Provide only \"support\" or \"refute\" as the final answer "
    "without any explanation.\n\n"
    "Task:\n"
    "Claim: {claim}\n"
    "Answer:"
)

EVIDENCE_PROMPT = (
    "Based on the provided information and your internal knowledge, determine whether the "
    "claim should be supported or refuted. Provide only \"support\" or \"refute\" as the "
    "final answer without any explanation.\n\n"
    "Task:\n"
    "Claim: {claim}\n"
    "Information: {evidence}\n"
    "Answer:"
)

SKIPPED = "skipped"
NA = "na"


def probe(prompt, model, temperature, kw):
    raw = C.llm_request(prompt, model_name=model, temperature=temperature, **kw)
    return raw, parse_verdict(raw)


def judge_of(verdict, label):
    if not verdict:
        return NA
    return "correct" if matches_label(verdict, label) else "wrong"


def run_once(records, model, temperature, kw):
    out = {"prior_correct": 0, "prior_wrong": 0, "prior_na": 0, "records": []}

    for i, item in enumerate(records, 1):
        claim = (item.get("claim_text") or "").strip()
        label = (item.get("cred_label") or "").strip()
        if not claim or not label:
            continue

        prior_raw, prior = probe(PRIOR_PROMPT.format(claim=claim), model, temperature, kw)
        prior_judge = judge_of(prior, label)
        out["prior_" + {"correct": "correct", "wrong": "wrong", NA: "na"}[prior_judge]] += 1

        rec = {
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

        if prior_judge == "wrong":
            _, verdict = probe(
                EVIDENCE_PROMPT.format(claim=claim, evidence=item.get("evidence", "")),
                model,
                temperature,
                kw,
            )
            rec["gold_verdict"] = verdict
            rec["gold_judge"] = judge_of(verdict, label)

        elif prior_judge == "correct":
            for key, field in (
                ("entity", "word_counter_evidence"),
                ("semantic", "sentence_counter_evidence"),
            ):
                evidence = item.get(field, "")
                if not evidence:
                    continue
                _, verdict = probe(
                    EVIDENCE_PROMPT.format(claim=claim, evidence=evidence),
                    model,
                    temperature,
                    kw,
                )
                rec[key + "_verdict"] = verdict
                rec[key + "_judge"] = judge_of(verdict, label)

        out["records"].append(rec)
        if i % 100 == 0:
            print(f"    {i}/{len(records)}")

    return out


def stage_run(args, kw):
    for name in args.datasets:
        in_path = os.path.join(
            C.OUTPUT_ROOT,
            "fact_benchmarks",
            "counter_data",
            args.counter_tag,
            DATASETS[name],
        )
        if not os.path.exists(in_path):
            print(
                f"[{name}] no benchmark at {in_path} -- "
                f"run counter.py --tag {args.counter_tag} first"
            )
            continue

        records = C.read_jsonl(in_path)
        if args.limit and args.limit > 0:
            records = records[: args.limit]

        out_dir = os.path.join(C.OUTPUT_ROOT, "evaluation_results", args.model, name)

        for run_idx in range(args.n):
            out_path = os.path.join(out_dir, f"run_{run_idx:02d}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[{name}] run {run_idx}: already done, skipping")
                continue

            print(f"[{name}] run {run_idx + 1}/{args.n} over {len(records)} claims")
            result = run_once(records, args.model, args.temperature, kw)
            n = len(result["records"])
            acc = result["prior_correct"] / n if n else 0.0
            print(
                f"    prior-only accuracy {acc:.4f} "
                f"(correct {result['prior_correct']}, wrong {result['prior_wrong']}, "
                f"no-verdict {result['prior_na']})"
            )

            C.write_json(
                out_path,
                {
                    "run": run_idx,
                    "model": args.model,
                    "dataset": name,
                    "counter_tag": args.counter_tag,
                    "temperature": args.temperature,
                    "records": result["records"],
                },
            )

    return 0


def _rate(num, denom):
    return num / denom if denom else None


def run_metrics(records):
    metrics = {}

    wrong = [r for r in records if r["prior_judge"] == "wrong"]
    corrected = sum(1 for r in wrong if r["gold_judge"] == "correct")
    unchanged = sum(1 for r in wrong if r["gold_judge"] == "wrong")
    no_verdict = sum(1 for r in wrong if r["gold_judge"] == NA)
    cr = _rate(corrected, corrected + unchanged)
    metrics["correction"] = {
        "population": len(wrong),
        "scored": corrected + unchanged,
        "corrected": corrected,
        "unchanged": unchanged,
        "no_verdict": no_verdict,
        "CR": cr,
        "OI": _rate(cr, 1 - cr) if cr is not None and cr < 1 else None,
    }

    correct = [r for r in records if r["prior_judge"] == "correct"]
    for key in ("entity", "semantic"):
        field = key + "_judge"
        held = sum(1 for r in correct if r[field] == "correct")
        flipped = sum(1 for r in correct if r[field] == "wrong")
        no_verdict = sum(1 for r in correct if r[field] == NA)
        pr = _rate(held, held + flipped)
        metrics["persistence_" + key] = {
            "population": len(correct),
            "scored": held + flipped,
            "persisted": held,
            "flipped": flipped,
            "no_verdict": no_verdict,
            "PR": pr,
            "OI": _rate(1 - pr, pr) if pr else None,
        }

    metrics["prior_no_verdict"] = sum(1 for r in records if r["prior_judge"] == NA)
    metrics["prior_correct_population"] = len(correct)
    metrics["claims"] = len(records)
    return metrics


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def average(per_run):
    out = {"correction": {}, "persistence_entity": {}, "persistence_semantic": {}}

    for key in ("CR", "OI"):
        out["correction"][key] = _mean([r["correction"][key] for r in per_run])
    out["correction"]["population"] = _mean(
        [r["correction"]["population"] for r in per_run]
    )
    out["prior_correct_population"] = _mean(
        [r["prior_correct_population"] for r in per_run]
    )

    for key in ("entity", "semantic"):
        block = "persistence_" + key
        for metric in ("PR", "OI"):
            out[block][metric] = _mean([r[block][metric] for r in per_run])
        out[block]["population"] = _mean([r[block]["population"] for r in per_run])

    return out


def stage_aggregate(args):
    summary = {}

    for name in args.datasets:
        out_dir = os.path.join(C.OUTPUT_ROOT, "evaluation_results", args.model, name)
        runs = []

        for run_idx in range(args.n):
            path = os.path.join(out_dir, f"run_{run_idx:02d}.json")
            if not os.path.exists(path):
                print(f"[{name}] missing {os.path.basename(path)} -- run the run stage first")
                runs = []
                break
            runs.append(C.read_json(path))

        if not runs:
            continue

        by_claim = {}
        for run in runs:
            for rec in run["records"]:
                by_claim.setdefault(rec["claim"], []).append((run["run"], rec))

        groups = {"known": [], "unknown": [], "excluded": []}
        for claim, entries in by_claim.items():
            entries.sort(key=lambda x: x[0])
            verdicts = [rec["prior_verdict"] for _, rec in entries]
            state = classify_kb(
                verdicts,
                n_runs=args.n,
                min_valid_ratio=args.min_valid_ratio,
            )
            groups[state].append([rec for _, rec in entries])

        retained = len(groups["known"]) + len(groups["unknown"])
        dataset_summary = {
            "model": args.model,
            "dataset": name,
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
                per_run.append(run_metrics(records))

            dataset_summary[state] = {
                "claims": len(claim_groups),
                "ratio": _rate(len(claim_groups), retained),
                "per_run": per_run,
                "mean": average(per_run),
            }

        summary_path = os.path.join(out_dir, f"summary_n={args.n}.json")
        C.write_json(summary_path, dataset_summary)
        summary[name] = dataset_summary
        print(
            f"[{name}] {retained}/{len(by_claim)} claims retained "
            f"({len(groups['excluded'])} excluded) -> {summary_path}"
        )

    if args.table and summary:
        print_table(summary)
    return 0


def print_table(summary):
    width = 8

    def fmt(value, pct=True):
        if value is None:
            return "-".rjust(width)
        return (
            f"{value * 100:.2f}%" if pct else f"{value:.3f}"
        ).rjust(width)

    def row(name, cells):
        return f"{name:22}" + "".join(cells)

    for name, s in summary.items():
        print(
            f"\n=== {s['model']} / {name} "
            f"(n={s['n_runs']}, retained {s['retained']}/{s['claims_total']}) ==="
        )
        columns = ["ratio", "CR", "PR-ent", "PR-sem", "OI"]
        header = row("quadrant", [c.rjust(width) for c in columns])
        print(header)
        print("-" * len(header))

        retained = s["retained"] or 1
        for state, persistence_name, correction_name in (
            ("known", "Known-Knows (KK)", "Known-Unknows (KU)"),
            ("unknown", "Unknown-Knows (UK)", "Unknown-Unknows (UU)"),
        ):
            mean = s[state]["mean"]
            pe = mean["persistence_entity"]
            ps = mean["persistence_semantic"]
            correction = mean["correction"]

            print(
                row(
                    persistence_name,
                    [
                        fmt(_rate(mean["prior_correct_population"], retained)),
                        fmt(None),
                        fmt(pe["PR"]),
                        fmt(ps["PR"]),
                        fmt(pe["OI"], pct=False),
                    ],
                )
            )
            print(
                row(
                    correction_name,
                    [
                        fmt(_rate(correction["population"], retained)),
                        fmt(correction["CR"]),
                        fmt(None),
                        fmt(None),
                        fmt(correction["OI"], pct=False),
                    ],
                )
            )


def main():
    ap = argparse.ArgumentParser(description="PAVE counterfactual evaluation")
    ap.add_argument("stage", choices=["run", "aggregate"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--counter-tag",
        default="gpt-4o",
        help="Benchmark build created by counter.py --tag.",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["quantemp", "politifact", "snopes"],
        choices=sorted(DATASETS),
    )
    ap.add_argument("--min-valid-ratio", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--table", action="store_true")
    C.add_llm_args(ap)
    args = ap.parse_args()

    if args.stage == "run" and args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.mock_llm:
        C.set_mock(True)

    kw = {
        "max_tokens": args.max_tokens,
        "extra_body": C.parse_extra_body(args.extra_body),
    }
    return stage_run(args, kw) if args.stage == "run" else stage_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
