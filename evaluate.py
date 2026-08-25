"""
evaluate.py — PAVE evaluation: prior probe, prior-conditioned intervention, aggregation

Two stages.

  run        For each of N independent runs, probe the verifier with the claim
             alone, then intervene with evidence chosen by that prior:
               prior WRONG   -> Correction:  re-ask with the gold evidence
               prior CORRECT -> Persistence: re-ask with counter-entity evidence,
                                             and again with counter-semantic evidence
             One JSON file per run, so an interrupted sweep resumes instead of
             starting over.

  aggregate  Stratify claims by Knowledge Boundary (prior-only consistency
             across runs) and report CR / PR / Margin / OI per KB quadrant.

Quick start
-----------
    export OPENAI_API_KEY=sk-...
    python evaluate.py run       --model gpt-4o-mini --counter-tag gpt-4o --n 10
    python evaluate.py aggregate --model gpt-4o-mini --n 10 --table

Offline smoke test:
    python evaluate.py run       --n 2 --limit 5 --mock-llm
    python evaluate.py aggregate --n 2 --table
"""
import argparse
import os
import sys

import pcd_common as C
from labels import parse_verdict, matches_label, classify_kb

DATASETS = {
    "quantemp": "quantemp_counter.jsonl",
    "politifact": "politifact_counter.jsonl",
    "snopes": "snopes_counter.jsonl",
    "temporal": "temporal_counter.jsonl",
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

SKIPPED = "skipped"   # branch not applicable to this record
NA = "na"             # model produced no usable verdict (refusal / unparseable / empty)


# ==========================================================================
# Stage: run
# ==========================================================================
def probe(prompt, model, temperature, kw):
    raw = C.llm_request(prompt, model_name=model, temperature=temperature, **kw)
    return raw, parse_verdict(raw)


def judge_of(verdict, label):
    if not verdict:
        return NA
    return "correct" if matches_label(verdict, label) else "wrong"


def run_once(records, model, temperature, kw, dimension):
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
            "claim": claim, "label": label,
            "prior_raw": prior_raw, "prior_verdict": prior, "prior_judge": prior_judge,
            "gold_verdict": "", "gold_judge": SKIPPED,
            "entity_verdict": "", "entity_judge": SKIPPED,
            "semantic_verdict": "", "semantic_judge": SKIPPED,
        }

        if prior_judge == "wrong":
            # Correction / acquisition: can accurate evidence fix a wrong prior?
            raw, v = probe(
                EVIDENCE_PROMPT.format(claim=claim, evidence=item.get("evidence", "")),
                model, temperature, kw)
            rec["gold_verdict"], rec["gold_judge"] = v, judge_of(v, label)

        elif prior_judge == "correct" and dimension != "temporal":
            # Persistence: does misleading evidence overturn a correct prior?
            # Both conditions use the SAME prompt; only the evidence differs.
            for key, field in (("entity", "word_counter_evidence"),
                               ("semantic", "sentence_counter_evidence")):
                evidence = item.get(field, "")
                if not evidence:
                    continue  # leaves judge == SKIPPED; never scored as a flip
                raw, v = probe(EVIDENCE_PROMPT.format(claim=claim, evidence=evidence),
                               model, temperature, kw)
                rec[key + "_verdict"], rec[key + "_judge"] = v, judge_of(v, label)

        out["records"].append(rec)
        if i % 100 == 0:
            print(f"    {i}/{len(records)}")
    return out


def stage_run(args, kw):
    for name in args.datasets:
        in_path = os.path.join(C.OUTPUT_ROOT, "fact_bechmarks", "counter_data",
                               args.counter_tag, DATASETS[name])
        if not os.path.exists(in_path):
            print(f"[{name}] no benchmark at {in_path} -- run counter.py --tag {args.counter_tag} first")
            continue

        records = C.read_jsonl(in_path)
        if args.limit and args.limit > 0:
            records = records[:args.limit]
        out_dir = os.path.join(C.OUTPUT_ROOT, "evaluation_results", args.model, name)

        for run_idx in range(args.n):
            out_path = os.path.join(out_dir, f"run_{run_idx:02d}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[{name}] run {run_idx}: already done, skipping (use --overwrite to redo)")
                continue
            print(f"[{name}] run {run_idx + 1}/{args.n} over {len(records)} claims")
            result = run_once(records, args.model, args.temperature, kw, args.dimension)
            n = len(result["records"])
            acc = result["prior_correct"] / n if n else 0.0
            print(f"    prior-only accuracy {acc:.4f}   "
                  f"(correct {result['prior_correct']}, wrong {result['prior_wrong']}, "
                  f"no-verdict {result['prior_na']})")
            C.write_json(out_path, {
                "run": run_idx, "model": args.model, "dataset": name,
                "counter_tag": args.counter_tag, "temperature": args.temperature,
                "dimension": args.dimension, "records": result["records"],
            })
    return 0


# ==========================================================================
# Stage: aggregate
# ==========================================================================
def _rate(num, denom):
    return num / denom if denom else None


def run_metrics(records, dimension):
    """Metrics for one KB group in one run.

    Records whose verdict could not be parsed are removed from the DENOMINATOR
    rather than counted as a changed answer. Under the old code an unparseable
    reply was indistinguishable from a successful correction, which inflated CR
    and depressed PR. Refusal counts are reported so the exclusion is visible.
    """
    m = {}

    wrong = [r for r in records if r["prior_judge"] == "wrong"]
    corrected = sum(1 for r in wrong if r["gold_judge"] == "correct")
    unchanged = sum(1 for r in wrong if r["gold_judge"] == "wrong")
    na = sum(1 for r in wrong if r["gold_judge"] == NA)
    cr = _rate(corrected, corrected + unchanged)
    m["correction"] = {"population": len(wrong), "scored": corrected + unchanged,
                       "corrected": corrected, "unchanged": unchanged, "no_verdict": na,
                       "CR": cr, "OI": _rate(cr, 1 - cr) if cr is not None and cr < 1 else None}

    correct = [r for r in records if r["prior_judge"] == "correct"]
    for key in ("entity", "semantic"):
        if dimension == "temporal":
            m["persistence_" + key] = None
            continue
        field = key + "_judge"
        held = sum(1 for r in correct if r[field] == "correct")
        flipped = sum(1 for r in correct if r[field] == "wrong")
        na_p = sum(1 for r in correct if r[field] == NA)
        pr = _rate(held, held + flipped)
        m["persistence_" + key] = {
            "population": len(correct), "scored": held + flipped,
            "persisted": held, "flipped": flipped, "no_verdict": na_p,
            "PR": pr, "OI": _rate(1 - pr, pr) if pr else None,
            "Margin": (cr + pr - 1) if (cr is not None and pr is not None) else None,
        }

    m["prior_no_verdict"] = sum(1 for r in records if r["prior_judge"] == NA)
    m["prior_correct_population"] = len(correct)
    m["claims"] = len(records)
    return m


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def average(per_run, dimension):
    out = {"correction": {}, "persistence_entity": {}, "persistence_semantic": {}}
    for k in ("CR", "OI"):
        out["correction"][k] = _mean([r["correction"][k] for r in per_run])
    out["correction"]["population"] = _mean([r["correction"]["population"] for r in per_run])
    out["prior_correct_population"] = _mean([r["prior_correct_population"] for r in per_run])
    for key in ("entity", "semantic"):
        blk = "persistence_" + key
        if dimension == "temporal":
            out[blk] = None
            continue
        for k in ("PR", "OI", "Margin"):
            out[blk][k] = _mean([r[blk][k] for r in per_run])
        out[blk]["population"] = _mean([r[blk]["population"] for r in per_run])
    return out


def stage_aggregate(args):
    summary = {}
    for name in args.datasets:
        out_dir = os.path.join(C.OUTPUT_ROOT, "evaluation_results", args.model, name)
        runs = []
        for run_idx in range(args.n):
            p = os.path.join(out_dir, f"run_{run_idx:02d}.json")
            if not os.path.exists(p):
                print(f"[{name}] missing {os.path.basename(p)} -- run the 'run' stage first")
                runs = []
                break
            runs.append(C.read_json(p))
        if not runs:
            continue
        dimension = runs[0].get("dimension", "counterfactual")

        # index by claim, preserving run order
        by_claim = {}
        for r in runs:
            for rec in r["records"]:
                by_claim.setdefault(rec["claim"], []).append((r["run"], rec))

        groups = {"known": [], "unknown": [], "excluded": []}
        for claim, entries in by_claim.items():
            entries.sort(key=lambda t: t[0])
            verdicts = [rec["prior_verdict"] for _, rec in entries]
            state = classify_kb(verdicts, n_runs=args.n, min_valid_ratio=args.min_valid_ratio)
            groups[state].append([rec for _, rec in entries])

        retained = len(groups["known"]) + len(groups["unknown"])
        totals = {"claims_total": len(by_claim), "retained": retained,
                  "excluded_refusal": len(groups["excluded"]),
                  "excluded_rate": _rate(len(groups["excluded"]), len(by_claim))}

        ds_summary = {"model": args.model, "dataset": name, "dimension": dimension,
                      "n_runs": args.n, "min_valid_ratio": args.min_valid_ratio, **totals}

        for state in ("known", "unknown"):
            claim_groups = groups[state]
            per_run = []
            for run_idx in range(args.n):
                recs = [g[run_idx] for g in claim_groups if run_idx < len(g)]
                per_run.append(run_metrics(recs, dimension))
            ds_summary[state] = {
                "claims": len(claim_groups),
                "ratio": _rate(len(claim_groups), retained),
                "per_run": per_run,
                "mean": average(per_run, dimension),
            }

        path = os.path.join(out_dir, f"summary_n={args.n}.json")
        C.write_json(path, ds_summary)
        summary[name] = ds_summary
        print(f"[{name}] {retained}/{len(by_claim)} claims retained "
              f"({len(groups['excluded'])} excluded for refusal)  ->  {path}")

    if args.table and summary:
        print_table(summary)
    return 0


def print_table(summary):
    """Table 1 layout: KB quadrant x arbitration behaviour."""
    W = 8

    def fmt(v, pct=True):
        if v is None:
            return "-".rjust(W)
        return (f"{v * 100:.2f}%" if pct else f"{v:.3f}").rjust(W)

    def row(name, cells):
        return f"{name:22}" + "".join(cells)

    for name, s in summary.items():
        print(f"\n=== {s['model']}  /  {name}  "
              f"(n={s['n_runs']}, retained {s['retained']}/{s['claims_total']}, "
              f"refusal-excluded {fmt(s['excluded_rate']).strip()}) ===")
        cols = ["ratio", "CR", "PR-ent", "PR-sem", "Margin", "OI"]
        header = row("quadrant", [c.rjust(W) for c in cols])
        print(header)
        print("-" * len(header))
        retained = s["retained"] or 1
        for state, quad_persist, quad_correct in (
                ("known", "Known-Knows (KK)", "Known-Unknows (KU)"),
                ("unknown", "Unknown-Knows (UK)", "Unknown-Unknows (UU)")):
            mean = s[state]["mean"]
            pe, ps, c = (mean["persistence_entity"], mean["persistence_semantic"],
                         mean["correction"])
            # persistence quadrant: claims whose prior was already correct
            print(row(quad_persist, [
                fmt(_rate(mean["prior_correct_population"], retained)),
                fmt(None),
                fmt(pe["PR"]) if pe else fmt(None),
                fmt(ps["PR"]) if ps else fmt(None),
                fmt(pe["Margin"]) if pe else fmt(None),
                fmt(pe["OI"], pct=False) if pe else fmt(None),
            ]))
            # correction quadrant: claims whose prior was wrong
            print(row(quad_correct, [
                fmt(_rate(c["population"], retained)), fmt(c["CR"]),
                fmt(None), fmt(None), fmt(None), fmt(c["OI"], pct=False),
            ]))
        print("\nRatio: share of retained claims in that quadrant (mean over runs; the four sum to "
              "100%).\nPR-ent/PR-sem: persistence under counter-entity / counter-semantic evidence. "
              "OI: odds of\nchanging the answer. Margin = CR + PR-ent - 1 within that KB state. "
              "Unparseable verdicts are\nexcluded from every denominator; the refusal exclusion rate "
              "is in the header.")
        if s["dimension"] == "temporal":
            print("Dimension 'temporal': persistence is not measured. A non-zero KK/UK ratio means "
                  "claims\nthe model already answers correctly are still in the set and should be "
                  "filtered out (§3.2).")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="PAVE evaluation: correction / persistence under PCD")
    ap.add_argument("stage", choices=["run", "aggregate"])
    ap.add_argument("--model", default="gpt-4o-mini", help="Verifier being evaluated.")
    ap.add_argument("--counter-tag", default="gpt-4o",
                    help="Which benchmark build to evaluate against, i.e. the --tag given to "
                         "counter.py. Independent of --model on purpose: every verifier is scored "
                         "on the same fixed benchmark.")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="Sampling temperature (paper main results use 0.3). Always sent.")
    ap.add_argument("--n", type=int, default=10, help="Number of independent runs N.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Claims per dataset; 0 (default) means all of them.")
    ap.add_argument("--datasets", nargs="+", default=["quantemp", "politifact", "snopes"],
                    choices=sorted(DATASETS))
    ap.add_argument("--dimension", choices=["counterfactual", "temporal"], default="counterfactual",
                    help="'temporal' evaluates acquisition only: claims whose prior is already "
                         "correct are not probed, because persisting in an outdated prior is not a "
                         "meaningful persistence result.")
    ap.add_argument("--min-valid-ratio", type=float, default=1.0,
                    help="Fraction of runs that must yield a parseable prior verdict for a claim to "
                         "stay in the benchmark. 1.0 is the paper's strict refusal filtering.")
    ap.add_argument("--overwrite", action="store_true", help="Redo runs that already have a file.")
    ap.add_argument("--table", action="store_true", help="Print the Table 1 quadrant block.")
    C.add_llm_args(ap)
    args = ap.parse_args()

    if args.stage == "run" and args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.mock_llm:
        C.set_mock(True)
    kw = {"max_tokens": args.max_tokens, "extra_body": C.parse_extra_body(args.extra_body)}

    return stage_run(args, kw) if args.stage == "run" else stage_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
