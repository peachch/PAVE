"""Crawl Wikipedia temporal events for PAVE.

This module intentionally only acquires event metadata and candidate Wikipedia
pages. Evidence extraction/validation is handled by ``get_evidence.py`` so the
crawler does not silently treat an arbitrary linked page as gold evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable
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

logger = logging.getLogger("pave.temporal.crawl")


def build_session(user_agent: str = DEFAULT_USER_AGENT, retries: int = 4) -> requests.Session:
    """Return a retrying HTTP session suitable for Wikimedia requests."""
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
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "en;q=1.0",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


class WikipediaClient:
    """Small throttled Wikipedia HTTP client."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 20.0,
                 delay: float = 0.2, retries: int = 4):
        self.session = build_session(user_agent, retries=retries)
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


def _clean_text(tag: Tag) -> str:
    text = tag.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_article_url(href: str) -> str:
    """Return a canonical enwiki article URL or an empty string."""
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


def extract_candidate_links(container: Tag, max_links: int = 8) -> list[str]:
    """Collect unique internal Wikipedia article links in document order."""
    links: list[str] = []
    seen: set[str] = set()
    for a in container.find_all("a", href=True):
        href = str(a.get("href") or "")
        classes = set(a.get("class") or [])
        if "external" in classes:
            continue
        url = _canonical_article_url(href)
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            links.append(url)
        if len(links) >= max_links:
            break
    return links


def infer_day(div: Tag, fallback: int) -> int:
    """Infer the day number around a Current events day block.

    Wikipedia markup has changed over time, so this checks nearby headings and
    ids before falling back to the block's ordinal position in the month page.
    """
    candidates: list[str] = []
    for node in (div, div.parent if isinstance(div.parent, Tag) else None):
        if not isinstance(node, Tag):
            continue
        if node.get("id"):
            candidates.append(str(node.get("id")))
        heading = node.find_previous(["h2", "h3", "h4", "h5"])
        if isinstance(heading, Tag):
            candidates.append(_clean_text(heading))
            if heading.get("id"):
                candidates.append(str(heading.get("id")))
    for text in candidates:
        m = re.search(r"(?:^|\D)(3[01]|[12]?\d)(?:\D|$)", text)
        if m:
            day = int(m.group(1))
            if 1 <= day <= 31:
                return day
    return fallback


def _category_for_list(ul: Tag) -> str:
    p = ul.find_previous_sibling("p")
    if isinstance(p, Tag):
        bold = p.find("b")
        if isinstance(bold, Tag):
            category = _clean_text(bold)
            if category:
                return category
    # Some pages put the label slightly farther away.
    p = ul.find_previous("p")
    if isinstance(p, Tag):
        bold = p.find("b")
        if isinstance(bold, Tag):
            category = _clean_text(bold)
            if category:
                return category
    return "Event"


def parse_current_events_html(html: str, year: int, month: str,
                              max_results: int | None = None) -> list[dict]:
    """Parse one ``Portal:Current_events/<Month>_<Year>`` page."""
    soup = BeautifulSoup(html, "html.parser")
    day_blocks = soup.find_all("div", class_="current-events-content")
    records: list[dict] = []

    for index, div in enumerate(day_blocks, 1):
        if not isinstance(div, Tag):
            continue
        day = infer_day(div, index)
        # Each category generally has a <p><b>...</b></p> followed by a <ul>.
        for ul in div.find_all("ul"):
            if not isinstance(ul, Tag):
                continue
            # Only consume top-level category lists, not nested sublists twice.
            if isinstance(ul.parent, Tag) and ul.parent.name == "li":
                continue
            category = _category_for_list(ul)
            for li in ul.find_all("li", recursive=False):
                if not isinstance(li, Tag):
                    continue
                fact = _clean_text(li)
                if not fact:
                    continue
                candidate_links = extract_candidate_links(li)
                records.append({
                    "Facts": fact,
                    "Evidence": None,
                    "Time-year": str(year),
                    "Time-month": month,
                    "Time-day": day,
                    "Spotlight": "NA",
                    "Category": category,
                    "Candidate_resources": candidate_links,
                    "Resources": candidate_links[0] if candidate_links else "",
                })
                if max_results and len(records) >= max_results:
                    return records
    return records


def parse_deaths_html(html: str, year: int, month: str,
                      max_results: int | None = None) -> list[dict]:
    """Parse ``Deaths_in_<Month>_<Year>`` including day 31."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []
    for day in range(1, 32):
        heading = soup.find(["h2", "h3", "h4"], id=str(day))
        if not isinstance(heading, Tag):
            # Newer Parsoid HTML may put the id on a nested span.
            marker = soup.find(id=str(day))
            heading = marker.find_parent(["h2", "h3", "h4"]) if isinstance(marker, Tag) else None
        if not isinstance(heading, Tag):
            continue
        ul = heading.find_next("ul")
        if not isinstance(ul, Tag):
            continue
        for li in ul.find_all("li", recursive=False):
            if not isinstance(li, Tag):
                continue
            fact = _clean_text(li)
            if not fact:
                continue
            candidate_links = extract_candidate_links(li)
            records.append({
                "Facts": fact,
                "Evidence": None,
                "Time-year": str(year),
                "Time-month": month,
                "Time-day": day,
                "Spotlight": "NA",
                "Category": "human death",
                "Candidate_resources": candidate_links,
                "Resources": candidate_links[0] if candidate_links else "",
            })
            if max_results and len(records) >= max_results:
                return records
    return records


def parse_month_events_html(html: str, year: int, month: str,
                            max_results: int | None = None) -> list[dict]:
    """Best-effort parser for generic ``<Month>_<Year>`` pages."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []
    date_re = re.compile(rf"^{re.escape(month)}\s+(3[01]|[12]?\d)$")
    for li in soup.find_all("li"):
        if not isinstance(li, Tag):
            continue
        first_a = li.find("a")
        if not isinstance(first_a, Tag):
            continue
        m = date_re.match(_clean_text(first_a))
        if not m:
            continue
        day = int(m.group(1))
        event_ul = li.find("ul")
        if not isinstance(event_ul, Tag):
            continue
        for event_li in event_ul.find_all("li", recursive=False):
            if not isinstance(event_li, Tag):
                continue
            fact = _clean_text(event_li)
            if not fact:
                continue
            links = extract_candidate_links(event_li)
            records.append({
                "Facts": fact,
                "Evidence": None,
                "Time-year": str(year),
                "Time-month": month,
                "Time-day": day,
                "Spotlight": "NA",
                "Category": "Event",
                "Candidate_resources": links,
                "Resources": links[0] if links else "",
            })
            if max_results and len(records) >= max_results:
                return records
    return records


def crawl(year: int, category: str, months: Iterable[str], client: WikipediaClient,
          max_results: int | None = None) -> list[dict]:
    records: list[dict] = []
    for month in months:
        if month not in MONTHS:
            raise ValueError(f"invalid month: {month}")
        remaining = None if not max_results else max_results - len(records)
        if remaining is not None and remaining <= 0:
            break
        if category == "current_event":
            url = f"{BASE_URL}/wiki/Portal:Current_events/{month}_{year}"
            parser = parse_current_events_html
        elif category == "death_event":
            url = f"{BASE_URL}/wiki/Deaths_in_{month}_{year}"
            parser = parse_deaths_html
        elif category == "month_event":
            url = f"{BASE_URL}/wiki/{month}_{year}"
            parser = parse_month_events_html
        else:
            raise ValueError(f"unsupported category: {category}")
        logger.info("fetching %s", url)
        try:
            html = client.get_text(url)
        except requests.RequestException as exc:
            logger.warning("failed to fetch %s: %s", url, exc)
            continue
        month_records = parser(html, year, month, remaining)
        logger.info("%s %s: %d records", month, year, len(month_records))
        records.extend(month_records)
    return records


def write_json(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Crawl Wikipedia events for PAVE temporal PCD")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument(
        "--category",
        choices=("current_event", "death_event", "month_event"),
        default="current_event",
    )
    ap.add_argument("--months", nargs="+", choices=MONTHS, default=MONTHS)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--max-results", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--delay", type=float, default=0.2, help="minimum seconds between HTTP requests")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                    help="descriptive Wikimedia User-Agent; override PAVE_WIKI_USER_AGENT as needed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = WikipediaClient(
        user_agent=args.user_agent,
        timeout=args.timeout,
        delay=args.delay,
        retries=args.retries,
    )
    records = crawl(
        args.year,
        args.category,
        args.months,
        client,
        max_results=args.max_results or None,
    )
    write_json(args.out, records)
    with_links = sum(bool(r.get("Candidate_resources")) for r in records)
    logger.info("wrote %d records (%d with candidate Wikipedia pages) -> %s",
                len(records), with_links, args.out)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
