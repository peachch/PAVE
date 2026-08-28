"""Crawl Wikipedia Current Events into a small, normalized temporal dataset."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://en.wikipedia.org"
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DEFAULT_USER_AGENT = os.environ.get(
    "PAVE_WIKI_USER_AGENT",
    "PAVE-TemporalCrawler/1.0 (+https://github.com/peachch/PAVE)",
)
LOG = logging.getLogger("temporal.crawl")


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


class WikipediaClient:
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


def clean_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def canonical_article_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("./"):
        href = "/wiki/" + href[2:]
    parsed = urlparse(href)
    path = parsed.path or href
    if not path.startswith("/wiki/"):
        return ""
    title = path[len("/wiki/"):]
    if not title or ":" in title or title == "Main_Page":
        return ""
    url = urljoin(BASE_URL, href)
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def candidate_links(container: Tag, max_links: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for a in container.find_all("a", href=True):
        if "external" in set(a.get("class") or []):
            continue
        url = canonical_article_url(str(a.get("href") or ""))
        if url and url not in seen:
            seen.add(url)
            out.append(url)
        if len(out) >= max_links:
            break
    return out


def infer_day(block: Tag, fallback: int) -> int:
    candidates: list[str] = []
    for node in (block, block.parent if isinstance(block.parent, Tag) else None):
        if not isinstance(node, Tag):
            continue
        if node.get("id"):
            candidates.append(str(node.get("id")))
        heading = node.find_previous(["h2", "h3", "h4", "h5"])
        if isinstance(heading, Tag):
            candidates.append(clean_text(heading))
            if heading.get("id"):
                candidates.append(str(heading.get("id")))
    for text in candidates:
        match = re.search(r"(?:^|\D)(3[01]|[12]?\d)(?:\D|$)", text)
        if match:
            day = int(match.group(1))
            if 1 <= day <= 31:
                return day
    return fallback


def category_for_list(ul: Tag) -> str:
    p = ul.find_previous_sibling("p") or ul.find_previous("p")
    if isinstance(p, Tag):
        bold = p.find("b")
        if isinstance(bold, Tag):
            text = clean_text(bold)
            if text:
                return text
    return "Event"


def parse_current_events_html(html: str, year: int, month: str, limit: int = 0) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", class_="current-events-content")
    records: list[dict] = []

    for fallback_day, block in enumerate(blocks, 1):
        if not isinstance(block, Tag):
            continue
        day = infer_day(block, fallback_day)
        try:
            event_date = date(year, MONTHS.index(month) + 1, day).isoformat()
        except ValueError:
            continue

        for ul in block.find_all("ul"):
            if not isinstance(ul, Tag):
                continue
            if isinstance(ul.parent, Tag) and ul.parent.name == "li":
                continue
            category = category_for_list(ul)
            for li in ul.find_all("li", recursive=False):
                if not isinstance(li, Tag):
                    continue
                claim = clean_text(li)
                if not claim:
                    continue
                urls = candidate_links(li)
                records.append({
                    "claim": claim,
                    "event_date": event_date,
                    "category": category,
                    "candidate_urls": urls,
                })
                if limit and len(records) >= limit:
                    return records
    return records


def crawl(year: int, months: list[str], client: WikipediaClient, limit: int = 0) -> list[dict]:
    records: list[dict] = []
    for month in months:
        remaining = limit - len(records) if limit else 0
        if limit and remaining <= 0:
            break
        url = f"{BASE_URL}/wiki/Portal:Current_events/{month}_{year}"
        LOG.info("fetching %s", url)
        try:
            html = client.get_text(url)
        except requests.RequestException as exc:
            LOG.warning("failed %s: %s", url, exc)
            continue
        month_records = parse_current_events_html(html, year, month, remaining)
        LOG.info("%s %d: %d events", month, year, len(month_records))
        records.extend(month_records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl Wikipedia Current Events")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", nargs="+", choices=MONTHS, default=MONTHS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = WikipediaClient(args.user_agent, args.timeout, args.delay, args.retries)
    records = crawl(args.year, args.months, client, args.limit)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with_urls = sum(bool(r["candidate_urls"]) for r in records)
    LOG.info("wrote %d events (%d with candidate pages) -> %s", len(records), with_urls, path)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
