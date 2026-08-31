#!/usr/bin/env python3
"""Build one unified PAVE counterfactual benchmark from standard datasets."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

DEFAULT_DATASETS = {
    "quantemp": "fact_benchmarks/quantmp/final.jsonl",
    "politifact": "fact_benchmarks/declare/politifact_final.json",
    "snopes": "fact_benchmarks/declare/snopes_final.json",
}

TRUE_LABELS = {"true", "support", "supports", "supported"}
FALSE_LABELS = {"false", "refute", "refutes", "refuted"}
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def read_records(path: Path):
    text = path.read_text(encoding="utf-8").lstrip("\ufeff\n\r\t ")
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON input must be a list: {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def opposite_label(label):
    value = re.sub(r"[^a-z]", "", (label or "").lower())
    if value in TRUE_LABELS:
        return "refute"
    if value in FALSE_LABELS:
        return "support"
    return ""


def semantic_counter(claim, label, api, model, temperature, max_tokens, extra_body):
    target = opposite_label(label)
    if not target:
        return ""
    prompt = (
        "Generate a concise, realistic counterfactual evidence passage that would make a verifier "
        f"judge the following claim as {target}. The evidence must clearly and unambiguously support "
        f"the target verdict ({target}); do not include statements that point toward the opposite verdict. "
        "The passage should directly address the claim and should not mention that it is fabricated or "
        "counterfactual. Output only the evidence passage.\n\n"
        f"Claim: {claim}\n"
        f"Target verdict: {target}\n"
        "Evidence:"
    )
    counter = api.ask(prompt, model, temperature, max_tokens, extra_body).strip()
    if not counter:
        return ""

    validation_prompt = (
        "Judge the relation between the evidence and the claim. Output exactly one label: "
        "support, refute, or unclear.\n\n"
        f"Claim: {claim}\n"
        f"Evidence: {counter}\n"
        "Label:"
    )
    verdict = api.ask(
        validation_prompt, model, 0.0, max_tokens, extra_body
    ).strip().lower()
    verdict = re.sub(r"[^a-z]", "", verdict)
    return counter if verdict == target else ""


def split_entities(raw, separator):
    raw = (raw or "").strip()
    if not raw:
        return []

    lines = [_LIST_MARKER.sub("", line).strip().strip('"“”') for line in raw.splitlines()]
    lines = [line for line in lines if line]
    if separator == "newline" and len(lines) > 1:
        parts = lines
    else:
        parts = [part.strip() for line in lines for part in line.split(",")]

    seen, entities = set(), []
    for part in parts:
        part = part.strip().strip('"“”.')
        key = part.lower()
        if part and key not in seen:
            seen.add(key)
            entities.append(part)
    return entities


def parse_entity_replacements(raw):
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    left, right = raw.find("["), raw.rfind("]")
    if left < 0 or right < left:
        return []
    try:
        data = json.loads(raw[left : right + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    pairs = []
    for item in data:
        if not isinstance(item, dict):
            return []
        original = item.get("original")
        replacement = item.get("replacement")
        if not isinstance(original, str) or not isinstance(replacement, str):
            return []
        original = original.strip()
        replacement = replacement.strip()
        if not original or not replacement:
            return []
        pairs.append((original, replacement))
    return pairs


def substitute_entities(evidence, replacements):
    mapping = dict(replacements)
    pattern = re.compile("|".join(re.escape(original) for original in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda match: mapping[match.group(0)], evidence)


def entity_counter(
    claim,
    evidence,
    api,
    model,
    temperature,
    max_tokens,
    extra_body,
    min_entities,
    separator,
):
    prompt = (
        f"Select exactly {min_entities} distinct named entities or concrete values from the evidence "
        "that are directly relevant to judging the claim, and provide one counterfactual replacement "
        "for each. You are selecting replacement strings only; do not rewrite the evidence.\n\n"
        "Hard constraints:\n"
        "1. Each `original` must be copied exactly and case-sensitively from the evidence.\n"
        "2. Select only a named entity or concrete value, such as a person, organization, location, "
        "date/time, age, number/quantity, disease, product, or event name. Do not select ordinary words "
        "or arbitrary phrases.\n"
        "3. Each `replacement` must be a plausible drop-in alternative of the same fine-grained semantic "
        "type and contextual role as its `original` (e.g., country -> country, person -> person, "
        "year -> year, quantity -> quantity).\n"
        "4. The replacement must fit directly into the original span without requiring any change to "
        "surrounding text.\n"
        "5. Do not return overlapping or nested originals. Do not return the same original twice.\n"
        "6. The replacement must differ from the original and must contain only the replacement string, "
        "with no arrows, explanations, labels, or before/after notation.\n\n"
        "Return JSON only in exactly this form:\n"
        '[{"original":"...","replacement":"..."}]\n\n'
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n"
        "Replacements:"
    )
    raw = api.ask(prompt, model, temperature, max_tokens, extra_body)
    replacements = parse_entity_replacements(raw)
    if len(replacements) != min_entities:
        return ""

    originals = []
    validated = []
    for original, replacement in replacements:
        if original not in evidence:
            return ""
        if original == replacement:
            return ""
        if any(mark in replacement for mark in ("->", "→", "=>", "\n", "\r")):
            return ""
        if any(
            original in other or other in original
            for other in originals
        ):
            return ""
        originals.append(original)
        validated.append((original, replacement))

    counter = substitute_entities(evidence, validated)
    return counter if counter != evidence else ""


def source_id_of(record, index):
    for key in ("id", "claim_id", "uid"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def build_dataset(
    dataset,
    path,
    api,
    args,
    extra_body,
):
    records = read_records(path)
    if args.limit_per_dataset > 0:
        records = records[: args.limit_per_dataset]

    kept = []
    stats = {
        "read": len(records),
        "missing_fields": 0,
        "bad_label": 0,
        "semantic_failed": 0,
        "entity_failed": 0,
        "kept": 0,
    }

    for index, record in enumerate(records, start=1):
        claim = str(record.get("claim_text") or "").strip()
        label = str(record.get("cred_label") or "").strip()
        evidence = str(record.get("evidence") or "").strip()
        if not claim or not label or not evidence:
            stats["missing_fields"] += 1
            continue
        if not opposite_label(label):
            stats["bad_label"] += 1
            continue

        semantic = semantic_counter(
            claim,
            label,
            api,
            args.model,
            args.temperature,
            args.max_tokens,
            extra_body,
        )
        if not semantic:
            stats["semantic_failed"] += 1
            continue

        entity = entity_counter(
            claim,
            evidence,
            api,
            args.model,
            args.temperature,
            args.max_tokens,
            extra_body,
            args.min_entities,
            args.entity_separator,
        )
        if not entity:
            stats["entity_failed"] += 1
            continue

        source_id = source_id_of(record, index)
        kept.append(
            {
                "record_id": f"{dataset}:{index:08d}",
                "dataset": dataset,
                "source_id": source_id,
                "claim_text": claim,
                "cred_label": label,
                "evidence": evidence,
                "word_counter_evidence": entity,
                "sentence_counter_evidence": semantic,
            }
        )

        if index % 50 == 0:
            print(f"[{dataset}] processed {index}/{len(records)}; kept {len(kept)}")

    stats["kept"] = len(kept)
    return kept, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build one unified PAVE counter benchmark from QuanTemp, PolitiFact, and Snopes"
    )
    parser.add_argument("--quantemp", default=DEFAULT_DATASETS["quantemp"])
    parser.add_argument("--politifact", default=DEFAULT_DATASETS["politifact"])
    parser.add_argument("--snopes", default=DEFAULT_DATASETS["snopes"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASETS),
        default=list(DEFAULT_DATASETS),
        help="Datasets to include; default: all three",
    )
    parser.add_argument("--output", default="data/counter.jsonl")
    parser.add_argument("--model", default="gpt-4o", help="Counterfactual builder model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Process at most this many source records from each dataset; 0 means all",
    )
    parser.add_argument(
        "--min-entities",
        type=int,
        default=5,
        help="Exact number of distinct entities/values to replace in word counter evidence",
    )
    parser.add_argument("--entity-separator", choices=["comma", "newline"], default="comma")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--extra-body", default=None, help="Extra JSON fields for the API request")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if args.min_entities < 1:
        raise SystemExit("--min-entities must be >= 1")
    if args.limit_per_dataset < 0:
        raise SystemExit("--limit-per-dataset must be >= 0")

    source_paths = {
        "quantemp": Path(args.quantemp),
        "politifact": Path(args.politifact),
        "snopes": Path(args.snopes),
    }
    missing = [f"{name}: {source_paths[name]}" for name in args.datasets if not source_paths[name].exists()]
    if missing:
        raise SystemExit("Missing source benchmark file(s):\n  " + "\n  ".join(missing))

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and "api.openai.com" in args.base_url:
        raise SystemExit("OPENAI_API_KEY is not set")

    api = ChatAPI(args.base_url, api_key, timeout=args.timeout)
    extra_body = parse_extra_body(args.extra_body)

    all_records = []
    all_stats = {}
    for dataset in args.datasets:
        print(f"\n[{dataset}] source: {source_paths[dataset]}")
        records, stats = build_dataset(
            dataset,
            source_paths[dataset],
            api,
            args,
            extra_body,
        )
        all_records.extend(records)
        all_stats[dataset] = stats

    output_path = Path(args.output)
    write_jsonl(output_path, all_records)

    print("\nUnified counter benchmark complete")
    print(f"{'dataset':<12} {'read':>8} {'kept':>8} {'yield':>9}")
    print("-" * 39)
    total_read = total_kept = 0
    for dataset in args.datasets:
        stats = all_stats[dataset]
        total_read += stats["read"]
        total_kept += stats["kept"]
        yield_rate = stats["kept"] / stats["read"] if stats["read"] else 0.0
        print(f"{dataset:<12} {stats['read']:>8} {stats['kept']:>8} {100 * yield_rate:>8.2f}%")
    total_yield = total_kept / total_read if total_read else 0.0
    print("-" * 39)
    print(f"{'TOTAL':<12} {total_read:>8} {total_kept:>8} {100 * total_yield:>8.2f}%")
    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
