"""Build a temporal correction benchmark from Wikipedia events.

Input: raw JSON produced by crawl.py.
Output: JSONL with post-cutoff claims and validated gold Wikipedia evidence.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger("temporal.prepare")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "her", "his", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "will", "with", "after", "before", "during", "over", "under",
}


def parse_cutoff(value: str) -> date:
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = map(int, value.split("-"))
            return date(year, month, calendar.monthrange(year, month)[1])
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--cutoff must be YYYY-MM or YYYY-MM-DD") from exc


def after_cutoff(event_date: str, cutoff: date | None) -> bool:
    return cutoff is None or date.fromisoformat(event_date) > cutoff


def build_session(user_agent: str, retries: int) -> requests.Session:
    retry = Retry(
        total=max(0, retries),
        connect=max(0, retries),
        read=max(0, retries),
        status=max(0, retries),
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "en;q=1.0"})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class WebClient:
    def __init__(self, user_agent: str, timeout: float, delay: float, retries: int):
        self.session = build_session(user_agent, retries)
        self.timeout = timeout
        self.delay = max(0.0, delay)
        self._last_request = 0.0

    def get_text(self, url: str) -> str:
        elapsed = time.monotonic() - self._last_request
        if self.delay and elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        response = self.session.get(url, timeout=self.timeout)
        self._last_request = time.monotonic()
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text


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


def normalize_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text or ""))


def proper_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text or "")
        if token[:1].isupper() and token.lower() not in _STOPWORDS and len(token) > 2
    }


def relevance_score(claim: str, evidence: str) -> float:
    claim_tokens = normalize_tokens(claim)
    evidence_tokens = normalize_tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0

    counts = Counter(claim_tokens)
    evidence_set = set(evidence_tokens)
    covered = sum(count for token, count in counts.items() if token in evidence_set)
    lexical = covered / max(1, sum(counts.values()))

    claim_numbers = numbers(claim)
    num_cov = (
        len(claim_numbers & numbers(evidence)) / len(claim_numbers)
        if claim_numbers
        else 0.0
    )
    claim_proper = proper_terms(claim)
    proper_cov = (
        len(claim_proper & proper_terms(evidence)) / len(claim_proper)
        if claim_proper
        else 0.0
    )
    length_bonus = min(1.0, math.log2(max(2, len(evidence_tokens))) / 8.0)
    return 0.62 * lexical + 0.20 * proper_cov + 0.13 * num_cov + 0.05 * length_bonus


def extract_paragraphs(html: str, max_paragraphs: int, min_chars: int = 80) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div.mw-parser-output") or soup.select_one("div.mw-body-content")
    if not isinstance(root, Tag):
        return []

    paragraphs: list[str] = []
    seen: set[str] = set()
    for p in root.find_all("p"):
        if not isinstance(p, Tag) or p.find_parent(["table", "nav", "aside"]):
            continue
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        text = re.sub(r"\[(?:\d+|citation needed)\]", "", text, flags=re.I)
        if len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)
        if len(paragraphs) >= max_paragraphs:
            break
    return paragraphs


def choose_gold_evidence(
    claim: str,
    pages: Iterable[tuple[str, str]],
    top_k: int,
    max_paragraphs: int,
) -> tuple[str, str, float]:
    candidates: list[tuple[float, str, str]] = []
    for url, html in pages:
        scored = sorted(
            (
                (relevance_score(claim, paragraph), paragraph)
                for paragraph in extract_paragraphs(html, max_paragraphs)
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        for score, paragraph in scored[: max(1, top_k)]:
            candidates.append((score, paragraph, url))

    if not candidates:
        return "", "", 0.0

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, _, best_url = candidates[0]
    same_page = [item for item in candidates if item[2] == best_url]
    selected = [
        item for item in same_page if item[0] >= 0.5 * best_score
    ][: max(1, top_k)] or [candidates[0]]
    return "\n".join(item[1] for item in selected), best_url, selected[0][0]


def parse_support_refute(text: str) -> str:
    normalized = re.sub(r"[^a-z]", "", (text or "").lower())
    if normalized.startswith("support") or normalized == "true":
        return "support"
    if normalized.startswith("refute") or normalized == "false":
        return "refute"
    return ""


def validate_gold_evidence(
    claim: str,
    evidence: str,
    model: str,
    max_tokens: int | None,
    extra_body: dict | None,
) -> bool:
    prompt = (
        "Judge whether the evidence supports the claim. "
        "Return exactly one label: support or refute.\n\n"
        f"Claim:\n{claim}\n\nEvidence:\n{evidence}\n\nLabel:"
    )
    raw = llm_request(prompt, model, 0.0, max_tokens, extra_body)
    return parse_support_refute(raw) == "support"


def record_id(event_date: str, claim: str) -> str:
    return hashlib.sha1(f"{event_date}\n{claim}".encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve and validate gold evidence for Temporal PAVE"
    )
    parser.add_argument("--input", required=True, help="raw JSON from crawl.py")
    parser.add_argument("--output", required=True, help="JSONL for evaluate.py")
    parser.add_argument(
        "--cutoff",
        type=parse_cutoff,
        default=None,
        help="keep events strictly after YYYY-MM or YYYY-MM-DD",
    )
    parser.add_argument(
        "--validator-model",
        default="gpt-4o-mini",
        help="model used only to validate that retrieved evidence supports the claim",
    )
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--max-paragraphs", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "PAVE_WIKI_USER_AGENT",
            "PAVE-TemporalCrawler/1.0 (+https://github.com/peachch/PAVE)",
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--extra-body",
        default=None,
        help='JSON object passed to the OpenAI-compatible API',
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    raw_records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit > 0:
        raw_records = raw_records[: args.limit]

    web = WebClient(args.user_agent, args.timeout, args.delay, args.retries)
    page_cache: dict[str, str] = {}
    output: list[dict] = []
    stats = Counter()
    seen_ids: set[str] = set()

    for index, record in enumerate(raw_records, 1):
        stats["read"] += 1
        claim = str(record.get("claim") or "").strip()
        event_date = str(record.get("event_date") or "").strip()
        urls = record.get("candidate_urls") or []

        if not claim or not event_date:
            stats["bad_record"] += 1
            continue
        try:
            if not after_cutoff(event_date, args.cutoff):
                stats["before_cutoff"] += 1
                continue
        except ValueError:
            stats["bad_date"] += 1
            continue
        if not urls:
            stats["no_candidate_url"] += 1
            continue

        pages: list[tuple[str, str]] = []
        for url in urls[: max(1, args.max_pages)]:
            try:
                html = page_cache.get(url)
                if html is None:
                    html = web.get_text(url)
                    page_cache[url] = html
                pages.append((url, html))
            except requests.RequestException as exc:
                LOG.debug("failed candidate %s: %s", url, exc)

        gold, source_url, score = choose_gold_evidence(
            claim, pages, args.top_k, args.max_paragraphs
        )
        if not gold:
            stats["no_gold_evidence"] += 1
            continue
        if score < args.min_score:
            stats["low_relevance"] += 1
            continue

        try:
            if not validate_gold_evidence(
                claim, gold, args.validator_model, args.max_tokens, extra_body
            ):
                stats["gold_validation_failed"] += 1
                continue
        except Exception as exc:
            stats["llm_error"] += 1
            LOG.warning("record %d validation error: %s", index, exc)
            continue

        rid = record_id(event_date, claim)
        if rid in seen_ids:
            stats["duplicate"] += 1
            continue
        seen_ids.add(rid)

        output.append(
            {
                "id": rid,
                "claim_text": claim,
                "cred_label": "True",
                "gold_evidence": gold,
                "event_date": event_date,
                "category": record.get("category") or "",
                "source_url": source_url,
                "gold_relevance_score": round(score, 6),
            }
        )
        stats["kept"] += 1
        if index % 25 == 0:
            LOG.info("processed %d/%d; kept %d", index, len(raw_records), stats["kept"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"read                     {stats['read']}")
    for key in (
        "bad_record",
        "bad_date",
        "before_cutoff",
        "no_candidate_url",
        "no_gold_evidence",
        "low_relevance",
        "gold_validation_failed",
        "llm_error",
        "duplicate",
    ):
        if stats[key]:
            print(f"dropped {key:24} {stats[key]}")
    print(f"kept                     {stats['kept']} -> {out_path}")
    return 0 if output else 1


if __name__ == "__main__":
    raise SystemExit(main())
