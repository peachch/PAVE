"""Retrieve and validate Wikipedia evidence for PAVE temporal events.

The old script accepted the first paragraph of one linked Wikipedia page as
"gold" evidence. That is unsafe for PAVE: a generic lead paragraph can be
unrelated to the actual event. This implementation searches multiple candidate
pages and paragraphs, ranks them against the event fact, and applies an explicit
validation gate before the record can enter the temporal benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

# Allow this script to reuse PAVE's OpenAI-compatible client when LLM validation
# is requested, while remaining importable/runnable from its subdirectory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pcd_common as C  # noqa: E402
from fact_benchmarks.temporal.crawl import (  # noqa: E402
    DEFAULT_USER_AGENT,
    WikipediaClient,
)

logger = logging.getLogger("pave.temporal.evidence")

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "her", "his", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "will", "with", "after", "before", "during", "over", "under",
}


def normalize_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text or ""))


def _proper_terms(text: str) -> set[str]:
    terms = set()
    for token in _WORD_RE.findall(text or ""):
        if token[:1].isupper() and token.lower() not in _STOPWORDS and len(token) > 2:
            terms.add(token.lower())
    return terms


def relevance_score(fact: str, evidence: str) -> float:
    """Score how specifically an evidence paragraph overlaps an event fact.

    The score emphasizes coverage of the fact rather than paragraph length and
    gives extra weight to numbers and capitalized entity-like tokens. It is a
    retrieval/quality gate, not a semantic entailment model.
    """
    fact_tokens = normalize_tokens(fact)
    evidence_tokens = normalize_tokens(evidence)
    if not fact_tokens or not evidence_tokens:
        return 0.0
    f_counts = Counter(fact_tokens)
    e_set = set(evidence_tokens)
    covered = sum(count for token, count in f_counts.items() if token in e_set)
    lexical_coverage = covered / max(1, sum(f_counts.values()))

    nums = _numbers(fact)
    num_coverage = len(nums & _numbers(evidence)) / len(nums) if nums else 0.0
    proper = _proper_terms(fact)
    proper_coverage = len(proper & _proper_terms(evidence)) / len(proper) if proper else 0.0

    # Slight preference for substantive evidence, capped so verbosity cannot win.
    length_bonus = min(1.0, math.log2(max(2, len(evidence_tokens))) / 8.0)
    return 0.62 * lexical_coverage + 0.20 * proper_coverage + 0.13 * num_coverage + 0.05 * length_bonus


def extract_article_paragraphs(html: str, max_paragraphs: int = 40,
                               min_chars: int = 80) -> list[str]:
    """Extract prose paragraphs from a Wikipedia article, excluding tables/nav."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div.mw-parser-output") or soup.select_one("div.mw-body-content")
    if not isinstance(root, Tag):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for p in root.find_all("p"):
        if not isinstance(p, Tag):
            continue
        if p.find_parent(["table", "nav", "aside"]):
            continue
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        # Strip citation markers but keep dates/numbers.
        text = re.sub(r"\[(?:\d+|citation needed)\]", "", text, flags=re.I)
        if len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_paragraphs:
            break
    return out


def choose_evidence(fact: str, pages: Iterable[tuple[str, str]], top_k: int = 2,
                    max_paragraphs: int = 40) -> tuple[str, str, float, list[dict]]:
    """Return best evidence, source URL, score, and retrieval trace."""
    candidates: list[tuple[float, str, str]] = []
    trace: list[dict] = []
    for url, html in pages:
        paragraphs = extract_article_paragraphs(html, max_paragraphs=max_paragraphs)
        scored = sorted(
            ((relevance_score(fact, p), p) for p in paragraphs),
            key=lambda x: x[0],
            reverse=True,
        )
        best = scored[0][0] if scored else 0.0
        trace.append({"url": url, "paragraphs": len(paragraphs), "best_score": round(best, 6)})
        for score, paragraph in scored[: max(1, top_k)]:
            candidates.append((score, paragraph, url))
    if not candidates:
        return "", "", 0.0, trace

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_url = candidates[0][2]
    # Keep the top paragraphs from the best page only to avoid accidental
    # cross-page concatenations that manufacture support.
    best_score = candidates[0][0]
    same_page = [x for x in candidates if x[2] == best_url]
    selected = [x for x in same_page if x[0] >= 0.5 * best_score][: max(1, top_k)]
    selected = selected or [candidates[0]]
    evidence = "\n".join(paragraph for _, paragraph, _ in selected)
    return evidence, best_url, selected[0][0], trace


def parse_llm_support(text: str) -> bool | None:
    normalized = re.sub(r"[^a-z_ ]", "", (text or "").lower()).strip()
    if normalized.startswith("support") or normalized.startswith("yes"):
        return True
    if (normalized.startswith("not_support") or normalized.startswith("not support")
            or normalized.startswith("no") or normalized.startswith("refute")):
        return False
    return None


def llm_supports(fact: str, evidence: str, model: str, temperature: float,
                 max_tokens: int | None, extra_body: dict | None) -> tuple[bool, str]:
    prompt = (
        "Judge whether the evidence directly supports the factual event claim. "
        "The evidence must contain enough specific information to establish the claim; "
        "mere topical relevance or generic background is NOT sufficient.\n\n"
        "Return exactly one label: support or not_support.\n\n"
        f"Claim:\n{fact}\n\nEvidence:\n{evidence}\n\nLabel:"
    )
    raw = C.llm_request(
        prompt,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    parsed = parse_llm_support(raw)
    return bool(parsed), raw


def candidate_urls(record: dict, max_pages: int) -> list[str]:
    urls: list[str] = []
    raw = record.get("Candidate_resources") or []
    if isinstance(raw, str):
        raw = [raw]
    for url in raw:
        if url and url not in urls:
            urls.append(url)
    for key in ("Resources", "Evidence_resource", "source_url"):
        url = record.get(key)
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls[: max(1, max_pages)]


def enrich_record(record: dict, client: WikipediaClient, *, max_pages: int,
                  top_k: int, max_paragraphs: int, min_score: float,
                  validator: str, validation_model: str, temperature: float,
                  max_tokens: int | None, extra_body: dict | None,
                  page_cache: dict[str, str] | None = None) -> dict:
    fact = (record.get("Facts") or record.get("claim_text") or "").strip()
    out = dict(record)
    urls = candidate_urls(record, max_pages)
    pages: list[tuple[str, str]] = []
    errors: list[dict] = []
    cache = page_cache if page_cache is not None else {}
    for url in urls:
        try:
            html = cache.get(url)
            if html is None:
                html = client.get_text(url)
                cache[url] = html
            pages.append((url, html))
        except requests.RequestException as exc:
            errors.append({"url": url, "error": str(exc)})
            logger.warning("failed evidence page %s: %s", url, exc)

    evidence, source_url, score, trace = choose_evidence(
        fact, pages, top_k=top_k, max_paragraphs=max_paragraphs
    )
    heuristic_valid = bool(evidence) and score >= min_score
    llm_valid: bool | None = None
    llm_raw = ""

    validation_error = ""
    should_call_llm = validator == "llm" and bool(evidence)
    should_call_llm = should_call_llm or (validator == "hybrid" and heuristic_valid)
    if should_call_llm:
        try:
            llm_valid, llm_raw = llm_supports(
                fact,
                evidence,
                model=validation_model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as exc:  # provider errors must not silently become support
            validation_error = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM validation failed: %s", validation_error)
            llm_valid = False

    if validator == "none":
        valid = bool(evidence)
    elif validator == "heuristic":
        valid = heuristic_valid
    elif validator == "llm":
        valid = bool(llm_valid)
    elif validator == "hybrid":
        valid = heuristic_valid and bool(llm_valid)
    else:
        raise ValueError(f"unknown validator: {validator}")

    out.update({
        "Evidence": evidence,
        "Resources": source_url or (out.get("Resources") or ""),
        "Evidence_score": round(score, 6),
        "Evidence_valid": bool(valid),
        "Evidence_validator": validator,
        "Evidence_retrieval_trace": trace,
    })
    if errors:
        out["Evidence_fetch_errors"] = errors
    if llm_raw:
        out["Evidence_validation_raw"] = llm_raw
    if validation_error:
        out["Evidence_validation_error"] = validation_error
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieve and validate Wikipedia evidence for PAVE")
    ap.add_argument("--input", required=True, help="raw crawl JSON")
    ap.add_argument("--output", required=True, help="enriched JSON")
    ap.add_argument("--max-pages", type=int, default=4, help="candidate Wikipedia pages per event")
    ap.add_argument("--max-paragraphs", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=2, help="top paragraphs joined as evidence")
    ap.add_argument("--min-score", type=float, default=0.12,
                    help="minimum heuristic relevance score")
    ap.add_argument("--validator", choices=("heuristic", "llm", "hybrid", "none"),
                    default="hybrid",
                    help="hybrid = heuristic retrieval gate + LLM support validation (recommended)")
    ap.add_argument("--validation-model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    C.add_llm_args(ap)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mock_llm:
        C.set_mock(True)

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    if args.limit > 0:
        records = records[: args.limit]

    client = WikipediaClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        delay=args.delay,
        retries=args.retries,
    )
    extra_body = C.parse_extra_body(args.extra_body)
    enriched: list[dict] = []
    kept = 0
    page_cache: dict[str, str] = {}
    for i, record in enumerate(records, 1):
        item = enrich_record(
            record,
            client,
            max_pages=args.max_pages,
            top_k=args.top_k,
            max_paragraphs=args.max_paragraphs,
            min_score=args.min_score,
            validator=args.validator,
            validation_model=args.validation_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
            page_cache=page_cache,
        )
        kept += int(bool(item.get("Evidence_valid")))
        enriched.append(item)
        if i % 50 == 0:
            logger.info("processed %d/%d; validated %d", i, len(records), kept)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("validated %d/%d (%.1f%%) -> %s", kept, len(enriched),
                100.0 * kept / len(enriched) if enriched else 0.0, out_path)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
