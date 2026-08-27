"""Convert validated Wikipedia temporal crawls into PAVE temporal JSONL."""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import sys
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
])}


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_cutoff(s):
    """'2023-10' or '2023-10-15' -> (year, month, day)."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"cutoff must be YYYY-MM or YYYY-MM-DD, got {s!r}")
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"invalid cutoff date: {s!r}")
    # A month-only model cutoff means knowledge through the end of that month.
    day = int(m.group(3)) if m.group(3) else calendar.monthrange(year, month)[1]
    try:
        date(year, month, day)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid cutoff date: {s!r}") from exc
    return year, month, day


def record_date(rec):
    try:
        y = int(str(rec.get("Time-year", "")).strip())
        raw_month = str(rec.get("Time-month", "")).strip()
        if raw_month.isdigit():
            mo = int(raw_month)
        else:
            mo = MONTHS.get(raw_month.capitalize())
        d = int(rec.get("Time-day") or 1)
        if mo is None:
            return None
        date(y, mo, d)  # validates month length/leap years
        return y, mo, d
    except (TypeError, ValueError):
        return None


def build_claim(rec, date):
    facts = re.sub(r"\s*\n\s*", " ", (rec.get("Facts") or rec.get("claim_text") or "").strip())
    y, mo, d = date
    month_name = next(k for k, v in MONTHS.items() if v == mo)
    return f"On {month_name} {d} {y}: {facts}"


def _evidence(rec):
    return (rec.get("Evidence") or rec.get("evidence") or "").strip()


def _source(rec):
    return (rec.get("Resources") or rec.get("source_url") or rec.get("Evidence_resource") or "").strip()


def convert(paths, cutoff=None, allow_unvalidated=False):
    out = []
    stats = {
        "read": 0,
        "no_evidence": 0,
        "no_facts": 0,
        "bad_date": 0,
        "before_cutoff": 0,
        "duplicate": 0,
        "unvalidated": 0,
        "validation_failed": 0,
    }
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            stats["read"] += 1
            evidence = _evidence(rec)
            facts = (rec.get("Facts") or rec.get("claim_text") or "").strip()
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

            validation = rec.get("Evidence_valid")
            if validation is False:
                stats["validation_failed"] += 1
                continue
            if validation is not True and not allow_unvalidated:
                stats["unvalidated"] += 1
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
                "category": rec.get("Category") or rec.get("category") or "",
                "source_url": _source(rec),
                "evidence_score": rec.get("Evidence_score"),
                "evidence_validator": rec.get("Evidence_validator", "legacy"),
            })
    return out, stats


def main():
    ap = argparse.ArgumentParser(description="Convert validated Wikipedia crawl into PAVE temporal input")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="one or more enriched crawl JSON files")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--cutoff", type=parse_cutoff, default=None,
                    help="keep only events strictly after YYYY-MM or YYYY-MM-DD")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="accept legacy records without Evidence_valid=True (not recommended for paper-quality evaluation)")
    args = ap.parse_args()

    records, stats = convert(args.inputs, cutoff=args.cutoff, allow_unvalidated=args.allow_unvalidated)
    _ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"read                         {stats['read']}")
    print(f"  dropped: no Facts           {stats['no_facts']}")
    print(f"  dropped: no Evidence        {stats['no_evidence']}")
    print(f"  dropped: unparsable date    {stats['bad_date']}")
    print(f"  dropped: unvalidated        {stats['unvalidated']}")
    print(f"  dropped: validation failed  {stats['validation_failed']}")
    print(f"  dropped: at/before cutoff   {stats['before_cutoff']}")
    print(f"  dropped: duplicate claim    {stats['duplicate']}")
    print(f"kept                         {len(records)} -> {args.out}")
    if not records:
        print("WARNING: nothing kept.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
