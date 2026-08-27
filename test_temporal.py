"""Offline unit/smoke tests for the PAVE temporal pipeline."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from fact_benchmarks.temporal.crawl import (
    extract_candidate_links,
    parse_current_events_html,
    parse_deaths_html,
)
from fact_benchmarks.temporal.get_evidence import (
    choose_evidence,
    extract_article_paragraphs,
    relevance_score,
    parse_llm_support,
)
from temporal_prepare import convert, parse_cutoff
from evaluate_temporal import correction_metrics


class CrawlParserTests(unittest.TestCase):
    def test_current_event_parser_and_links(self):
        html = """
        <div class='current-events-content description' id='day-31'>
          <p><b>Science and technology</b></p>
          <ul><li><a href='./Test_Event'>Test Event</a> happened in Paris on 31 January 2025.
          <a class='external' href='https://example.com'>source</a></li></ul>
        </div>
        """
        records = parse_current_events_html(html, 2025, "January")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["Time-day"], 31)
        self.assertEqual(records[0]["Category"], "Science and technology")
        self.assertEqual(records[0]["Candidate_resources"], ["https://en.wikipedia.org/wiki/Test_Event"])

    def test_death_parser_includes_day_31(self):
        html = """
        <h3 id='31'>31</h3><ul><li><a href='/wiki/Jane_Doe'>Jane Doe</a>, scientist</li></ul>
        """
        records = parse_deaths_html(html, 2025, "January")
        self.assertEqual(records[0]["Time-day"], 31)


class EvidenceTests(unittest.TestCase):
    def test_relevance_prefers_specific_support(self):
        fact = "Charli XCX wins British Artist of the Year at the 2025 Brit Awards."
        generic = "The Brit Awards are an annual music awards ceremony in the United Kingdom."
        specific = "At the Brit Awards 2025, Charli XCX won British Artist of the Year."
        self.assertGreater(relevance_score(fact, specific), relevance_score(fact, generic))

    def test_llm_support_parser(self):
        self.assertTrue(parse_llm_support("support"))
        self.assertFalse(parse_llm_support("not_support"))
        self.assertFalse(parse_llm_support("refute"))

    def test_choose_evidence_picks_specific_paragraph(self):
        fact = "A storm caused evacuations in South Carolina in March 2025."
        html = """
        <div class='mw-parser-output'>
          <p>Wildfires are uncontrolled fires in vegetation and occur in many countries.</p>
          <p>In March 2025, several wildfires in South Carolina caused evacuations across affected towns.</p>
        </div>
        """
        evidence, url, score, _ = choose_evidence(fact, [("https://en.wikipedia.org/wiki/X", html)], top_k=1)
        self.assertIn("South Carolina", evidence)
        self.assertGreater(score, 0)
        self.assertEqual(url, "https://en.wikipedia.org/wiki/X")


class PrepareTests(unittest.TestCase):
    def test_month_cutoff_means_end_of_month(self):
        self.assertEqual(parse_cutoff("2025-02"), (2025, 2, 28))

    def test_strict_validation_and_cutoff(self):
        rows = [
            {"Facts": "Event A", "Evidence": "Specific evidence A", "Evidence_valid": True,
             "Time-year": "2025", "Time-month": "January", "Time-day": 2, "Resources": "u1"},
            {"Facts": "Event B", "Evidence": "Generic evidence", "Evidence_valid": False,
             "Time-year": "2025", "Time-month": "January", "Time-day": 3, "Resources": "u2"},
            {"Facts": "Event C", "Evidence": "Legacy evidence",
             "Time-year": "2025", "Time-month": "January", "Time-day": 4, "Resources": "u3"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "data.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            out, stats = convert([path], cutoff=parse_cutoff("2025-01-01"))
        self.assertEqual(len(out), 1)
        self.assertEqual(stats["validation_failed"], 1)
        self.assertEqual(stats["unvalidated"], 1)


class TemporalMetricTests(unittest.TestCase):
    def test_correction_metric_denominator_excludes_no_verdict(self):
        records = [
            {"prior_judge": "wrong", "evidence_judge": "correct"},
            {"prior_judge": "wrong", "evidence_judge": "wrong"},
            {"prior_judge": "wrong", "evidence_judge": "na"},
            {"prior_judge": "correct", "evidence_judge": "skipped"},
        ]
        m = correction_metrics(records)
        self.assertEqual(m["prior_wrong_population"], 3)
        self.assertEqual(m["scored"], 2)
        self.assertAlmostEqual(m["CR"], 0.5)


if __name__ == "__main__":
    unittest.main()
