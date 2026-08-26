"""Build PAVE counterfactual benchmarks from the released source datasets."""

import argparse
import os
import re
import sys

import pcd_common as C

MIN_ENTITIES = 5

# Source files currently released in this repository.
DATASETS = {
    "quantemp": ("quantmp/final.jsonl", "quantemp_counter.jsonl"),
    "politifact": ("declare/politifact_final.json", "politifact_counter.jsonl"),
    "snopes": ("declare/snopes_final.json", "snopes_counter.jsonl"),
}

TRUE_LABELS = {"true", "support", "supports", "supported"}
FALSE_LABELS = {"false", "refute", "refutes", "refuted"}


# --------------------------------------------------------------------------
# Counter-semantic
# --------------------------------------------------------------------------
def sentence_level_counter(claim, label, model, temperature, **kw):
    label_clean = (label or "").strip().lower()
    if label_clean in TRUE_LABELS:
        prompt = (
            "Given a claim labeled as true, generate realistic but fabricated evidence to refute it. "
            "Create convincing fake content that contradicts the true claim.\n\n"
            "Task:\n"
            f"Claim: {claim} (label: true)\n"
            "Evidence:"
        )
    elif label_clean in FALSE_LABELS:
        prompt = (
            "Given a claim labeled as false, generate realistic but fabricated evidence to support it. "
            "Create convincing fake content that validates the false claim.\n\n"
            "Task:\n"
            f"Claim: {claim} (label: false)\n"
            "Evidence:"
        )
    else:
        return ""
    return C.llm_request(prompt, model_name=model, temperature=temperature, **kw).strip()


# --------------------------------------------------------------------------
# Counter-entity
# --------------------------------------------------------------------------
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _split_entities(raw, separator):
    raw = (raw or "").strip()
    if not raw:
        return []

    lines = [_LIST_MARKER.sub("", ln).strip().strip('"“”') for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]

    if separator == "newline" and len(lines) > 1:
        parts = lines
    else:
        parts = [p.strip() for ln in lines for p in ln.split(",")]

    seen, out = set(), []
    for p in parts:
        p = p.strip().strip('"“”.')
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _substitute(evidence, mapping):
    if not mapping:
        return evidence

    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        "|".join(r"(?<!\w)" + re.escape(k) + r"(?!\w)" for k in keys),
        re.IGNORECASE,
    )
    lowered = {k.lower(): v for k, v in mapping.items()}
    return pattern.sub(lambda m: lowered.get(m.group(0).lower(), m.group(0)), evidence)


def word_level_counter(claim, evidence, model, temperature, separator="comma", **kw):
    fmt = (
        "Output as a comma separated list."
        if separator == "comma"
        else "Output one entity per line."
    )
    prompt1 = (
        "Extract the entities from the following evidence which directly influence "
        f"the judgement of the claim. Claim: {claim}. {fmt} Evidence: {evidence}"
    )
    entities = _split_entities(
        C.llm_request(prompt1, model_name=model, temperature=temperature, **kw),
        separator,
    )
    if len(entities) < MIN_ENTITIES:
        return ""

    mapping = {}
    for ent in entities:
        prompt2 = (
            f"Given the entity '{ent}', generate a similar type but a different entity. "
            "Only output the new entity."
        )
        new_ent = C.llm_request(
            prompt2, model_name=model, temperature=temperature, **kw
        ).strip()
        new_ent = new_ent.strip('"“”')
        if not new_ent or new_ent.lower() == ent.lower():
            return ""
        mapping[ent] = new_ent

    out = _substitute(evidence, mapping)
    if out == evidence:
        return ""
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Construct PAVE counter-entity and counter-semantic benchmarks"
    )
    ap.add_argument(
        "--model",
        default="gpt-4o",
        help="Model used to generate the counterfactual benchmark.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Name of this benchmark build. Defaults to --model.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. The paper uses 0.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Records per dataset; 0 means all records.",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASETS),
        choices=sorted(DATASETS),
    )
    ap.add_argument(
        "--entity-separator",
        choices=["comma", "newline"],
        default="comma",
    )
    ap.add_argument(
        "--dump-csv",
        action="store_true",
        help="Also write a CSV copy of each generated benchmark.",
    )
    C.add_llm_args(ap)
    args = ap.parse_args()

    if args.mock_llm:
        C.set_mock(True)
    extra_body = C.parse_extra_body(args.extra_body)
    kw = {"max_tokens": args.max_tokens, "extra_body": extra_body}
    tag = args.tag or args.model

    out_dir = os.path.join(C.OUTPUT_ROOT, "fact_benchmarks", "counter_data", tag)
    totals = {}

    for name in args.datasets:
        rel_in, rel_out = DATASETS[name]
        in_path = os.path.join(C.DATA_ROOT, "fact_benchmarks", rel_in)
        if not os.path.exists(in_path):
            print(f"[{name}] input not found, skipping: {in_path}")
            continue

        records = C.read_jsonl(in_path)
        if args.limit and args.limit > 0:
            records = records[: args.limit]

        kept = []
        stats = {
            "read": len(records),
            "no_claim": 0,
            "bad_label": 0,
            "semantic_failed": 0,
            "entity_failed": 0,
        }

        for i, rec in enumerate(records, 1):
            claim = (rec.get("claim_text") or "").strip()
            label = (rec.get("cred_label") or "").strip()
            evidence = (rec.get("evidence") or "").strip()

            if not claim or not evidence:
                stats["no_claim"] += 1
                continue
            if label.lower() not in TRUE_LABELS | FALSE_LABELS:
                stats["bad_label"] += 1
                continue

            semantic = sentence_level_counter(
                claim, label, args.model, args.temperature, **kw
            )
            if not semantic:
                stats["semantic_failed"] += 1
                continue

            entity = word_level_counter(
                claim,
                evidence,
                args.model,
                args.temperature,
                separator=args.entity_separator,
                **kw,
            )
            if not entity:
                stats["entity_failed"] += 1
                continue

            out = dict(rec)
            out["word_counter_evidence"] = entity
            out["sentence_counter_evidence"] = semantic
            out["is_counterfactual"] = True
            kept.append(out)

            if i % 100 == 0:
                print(f"[{name}] {i}/{len(records)} processed, {len(kept)} kept")

        out_path = os.path.join(out_dir, rel_out)
        C.write_jsonl(out_path, kept)

        if args.dump_csv:
            import pandas as pd

            pd.DataFrame(kept).to_csv(
                out_path.replace(".jsonl", ".csv"), index=False, encoding="utf-8"
            )

        stats["kept"] = len(kept)
        totals[name] = stats
        yield_pct = 100.0 * len(kept) / stats["read"] if stats["read"] else 0.0
        print(
            f"[{name}] read {stats['read']} -> kept {len(kept)} "
            f"({yield_pct:.1f}%) {out_path}"
        )

    C.write_json(
        os.path.join(out_dir, "meta.json"),
        {
            "tag": tag,
            "generator_model": args.model,
            "temperature": args.temperature,
            "min_entities": MIN_ENTITIES,
            "entity_separator": args.entity_separator,
            "limit": args.limit,
            "base_url": C.OPENAI_BASE_URL,
            "mock": C.is_mock(),
            "datasets": totals,
        },
    )

    print(f"\nBenchmark build '{tag}' written to {out_dir}")
    print(
        "Evaluate it with: python evaluate.py run --model <verifier> "
        f"--counter-tag {tag}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
