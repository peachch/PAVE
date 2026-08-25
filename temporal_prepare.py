"""
temporal_prepare.py — turn the Wikipedia "Current events" crawl into the
schema the PAVE pipeline reads (Dimension 2, §3.2 Temporality).

The crawl files look like this:

    {"Facts": "...", "Evidence": "...", "Resources": "https://en.wikipedia.org/...",
     "Time-year": "2025", "Time-month": "January", "Time-day": 1,
     "Category": "...", "Spotlight": "NA"}

counter.py / evaluate.py read this instead:

    {"claim_text": "...", "cred_label": "True", "evidence": "..."}

so nothing consumes the crawl until it has been through this script. Crawled
events are all real occurrences, so every record is emitted with
cred_label="True" — the temporal dimension is an *acquisition* setting
(§3.2: "persisting in it would not constitute reliable fact-checking"), never
a persistence one.

Usage
-----
    # convert one crawl file
    python temporal_prepare.py \
        --in fact_bechmarks/temporal/wjbk_current_event_2025_2588_evidence.json \
        --out fact_bechmarks/previous_data/temporal/temporal_final.jsonl

    # merge several crawls and keep only events after a model's cutoff
    python temporal_prepare.py \
        --in crawl_2024.json crawl_2025.json \
        --out .../temporal_gpt-4o-mini_final.jsonl \
        --cutoff 2023-10

Records with empty Evidence are dropped (they cannot support a gold-evidence
correction probe); the count is reported so the yield is never silent.
"""
import argparse
import json
import os
import re
import sys

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_cutoff(s):
    """'2023-10' or '2023-10-15' -> (year, month, day)."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"cutoff must be YYYY-MM or YYYY-MM-DD, got {s!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)


def record_date(rec):
    try:
        y = int(str(rec.get("Time-year", "")).strip())
        mo = MONTHS.get(str(rec.get("Time-month", "")).strip().capitalize())
        d = int(rec.get("Time-day") or 1)
        if mo is None:
            return None
        return (y, mo, d)
    except (TypeError, ValueError):
        return None


def build_claim(rec, date):
    """Date-prefixed claim, matching the format in the paper's Tables 11/13."""
    facts = re.sub(r"\s*\n\s*", " ", (rec.get("Facts") or "").strip())
    y, mo, d = date
    month_name = [k for k, v in MONTHS.items() if v == mo][0]
    return f"On {month_name} {d} {y}: {facts}"


def convert(paths, cutoff=None):
    out, stats = [], {"read": 0, "no_evidence": 0, "no_facts": 0, "bad_date": 0, "before_cutoff": 0}
    seen = set()
    stats["duplicate"] = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            stats["read"] += 1
            evidence = (rec.get("Evidence") or "").strip()
            facts = (rec.get("Facts") or "").strip()
            date = record_date(rec)
            if not facts:
                stats["no_facts"] += 1
                continue
            if not evidence:
                stats["no_evidence"] += 1
                continue
            if date is None:
                stats["bad_date"] += 1
                continue
            if cutoff and date <= cutoff:
                stats["before_cutoff"] += 1
                continue
            claim = build_claim(rec, date)
            if claim in seen:
                stats["duplicate"] += 1
                continue
            seen.add(claim)
            out.append({
                "claim_text": claim,
                "cred_label": "True",
                "evidence": re.sub(r"\s*\n\s*", " ", evidence),
                "event_date": "%04d-%02d-%02d" % date,
                "category": rec.get("Category") or "",
                "source_url": rec.get("Resources") or "",
            })
    return out, stats


def main():
    ap = argparse.ArgumentParser(description="Convert Wikipedia current-events crawl into PAVE input schema")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="one or more crawl .json files")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--cutoff", type=parse_cutoff, default=None,
                    help="keep only events strictly AFTER this date (YYYY-MM), i.e. outside the "
                         "evaluated model's parametric memory. Omit to keep everything.")
    args = ap.parse_args()

    records, stats = convert(args.inputs, cutoff=args.cutoff)

    _ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"read              {stats['read']}")
    print(f"  dropped: no Facts        {stats['no_facts']}")
    print(f"  dropped: no Evidence     {stats['no_evidence']}")
    print(f"  dropped: unparsable date {stats['bad_date']}")
    print(f"  dropped: at/before cutoff{stats['before_cutoff']:>4}")
    print(f"  dropped: duplicate claim {stats['duplicate']}")
    print(f"kept              {len(records)}  ->  {args.out}")
    if not records:
        print("WARNING: nothing kept.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
