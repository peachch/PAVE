"""
labels.py — verdict parsing and Knowledge-Boundary classification for PAVE.

Drop-in replacement for `normalize_llm_label` in evaluate.py, plus the two
policy decisions that go with it:

  * parse_verdict()  -> 'support' | 'refute' | ''      ('' == no usable verdict)
  * classify_kb()    -> 'known' | 'unknown' | 'excluded'

Design notes
------------
Three things went wrong with the substring-based parser:

  1. Negation was invisible. "The claim should not be supported" contains
     'support' and not 'refute', so it parsed as *support* — the exact
     opposite of what the model said.
  2. Refusals were coerced. "I cannot support or refute this" also contains
     'support', so an abstention was recorded as a confident verdict.
  3. The FIRST cue won. Models that reason before answering ("...support,
     not refute") had their preamble read instead of their conclusion.

The parser below fixes all three: refusals are detected first and return '',
then the LAST verdict cue in the response is taken (that is the model's
conclusion), and a short window before that cue is checked for negation.

Paper §3.2 applies "a strict refusal filtering procedure, excluding instances
where the model abstains ... retaining only samples where the model exhibits a
definite prior". classify_kb() implements that: a claim whose N prior-only runs
are not all parseable is 'excluded' — it leaves the benchmark for that model
rather than being scored as confident.
"""
import re

__all__ = ["parse_verdict", "classify_kb", "matches_label", "REFUSAL", "SUPPORT", "REFUTE"]

SUPPORT = "support"
REFUTE = "refute"
REFUSAL = ""  # no usable verdict: refusal, empty response, or API failure


# --- refusal / abstention -------------------------------------------------
# Narrow on purpose. "cannot be supported" is a REFUTE, not a refusal, so the
# verbs here are epistemic ones (determine/answer/verify), never support/refute
# in the passive.
_REFUSAL_PATTERNS = [
    r"\bneither\s+(support|refute|true|false)\b",
    r"\bcannot\s+(support\s+or\s+refute|refute\s+or\s+support)\b",
    r"\bcan(?:no|')t\s+(determine|answer|verify|say|assess|conclude|confirm|judge|evaluate)\b",
    r"\bcannot\s+(determine|answer|verify|say|assess|conclude|confirm|judge|evaluate)\b",
    r"\b(?:am\s+)?unable\s+to\s+(determine|answer|verify|say|assess|conclude|confirm|judge|evaluate)\b",
    r"\b(insufficient|not\s+enough|no)\s+(information|evidence|data|context)\b",
    r"\bmore\s+(information|context|evidence)\s+(is\s+)?(needed|required)\b",
    r"\bwithout\s+(more|further|additional)\s+(information|context|evidence)\b",
    r"\bi\s+(do\s+not|don'?t)\s+(know|have\s+enough)\b",
    r"\b(as\s+an\s+ai|i'?m\s+sorry|i\s+am\s+sorry)\b",
    r"\bimpossible\s+to\s+(determine|verify|say)\b",
    r"\b(unclear|indeterminate|inconclusive)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# --- verdict cues ---------------------------------------------------------
_CUES = {
    "support": SUPPORT, "supports": SUPPORT, "supported": SUPPORT,
    "true": SUPPORT, "correct": SUPPORT, "accurate": SUPPORT, "yes": SUPPORT,
    "refute": REFUTE, "refutes": REFUTE, "refuted": REFUTE,
    "false": REFUTE, "incorrect": REFUTE, "inaccurate": REFUTE, "no": REFUTE,
}
_CUE_RE = re.compile(r"\b(" + "|".join(sorted(_CUES, key=len, reverse=True)) + r")\b", re.IGNORECASE)

# Negation looked for in the clause immediately preceding a cue.
_NEG_RE = re.compile(
    r"\b(not|n't|cannot|cant|never|no|without|fails?\s+to|unable\s+to|"
    r"isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|shouldn'?t|wouldn'?t|couldn'?t|can'?t)\b",
    re.IGNORECASE,
)
_NEG_WINDOW = 45  # characters of left context searched for a negator

_FLIP = {SUPPORT: REFUTE, REFUTE: SUPPORT}


def _negated(text, cue_start):
    """Is the cue at `cue_start` inside a negated clause? Only the current
    clause is inspected — the window is cut at the nearest preceding sentence
    or clause boundary so that 'It is false. The claim is supported' does not
    read the earlier clause's negation."""
    left = text[max(0, cue_start - _NEG_WINDOW):cue_start]
    for boundary in (".", ";", ":", "!", "?", ","):
        idx = left.rfind(boundary)
        if idx != -1:
            left = left[idx + 1:]
    return bool(_NEG_RE.search(left))


def parse_verdict(resp):
    """Normalize a free-form response into 'support', 'refute', or '' .

    '' means *no usable verdict* (refusal, hedge, empty, or API failure) and is
    never silently treated as a wrong answer — callers must exclude it.
    """
    if not resp or not resp.strip():
        return REFUSAL

    text = re.sub(r"\s+", " ", resp.strip())

    if _REFUSAL_RE.search(text):
        return REFUSAL

    matches = list(_CUE_RE.finditer(text))
    if not matches:
        return REFUSAL

    # The model's conclusion is its last verdict cue, not its first.
    m = matches[-1]
    verdict = _CUES[m.group(1).lower()]
    if _negated(text, m.start()):
        verdict = _FLIP[verdict]
    return verdict


# --- gold-label matching --------------------------------------------------
_LABEL_ALIASES = {
    "true": SUPPORT, "support": SUPPORT, "supports": SUPPORT, "supported": SUPPORT,
    "false": REFUTE, "refute": REFUTE, "refutes": REFUTE, "refuted": REFUTE,
}


def matches_label(verdict, gold_label):
    """Compare a parsed verdict against a gold label, tolerating the
    true/false vs support/refute vocabularies used across the corpora.
    An unusable verdict ('') never matches."""
    if not verdict:
        return False
    gold = _LABEL_ALIASES.get(re.sub(r"[^\w]", "", (gold_label or "")).lower())
    return gold is not None and verdict == gold


# --- Knowledge Boundary ---------------------------------------------------
def classify_kb(verdicts, n_runs=None, min_valid_ratio=1.0):
    """Classify a claim's pre-evidence epistemic state from its prior-only runs.

    verdicts        : list of parse_verdict() outputs, one per independent run
    n_runs          : expected number of runs (defaults to len(verdicts));
                      pass it explicitly so a claim missing from some run is
                      counted against the validity gate rather than ignored
    min_valid_ratio : fraction of runs that must yield a usable verdict for the
                      claim to stay in the benchmark. 1.0 is the paper's strict
                      refusal filtering (§3.2); lower it only deliberately.

    Returns 'known' (all valid verdicts identical), 'unknown' (they disagree),
    or 'excluded' (too few usable verdicts to judge either way).

    'excluded' is the answer to "what about 1 support and 9 refusals?" — that
    claim carries no evidence of confidence *or* of uncertainty, so it must
    leave the denominator instead of being scored as maximally confident.
    """
    total = n_runs if n_runs is not None else len(verdicts)
    if total <= 0:
        return "excluded"
    valid = [v for v in verdicts if v]
    if len(valid) < min_valid_ratio * total:
        return "excluded"
    if not valid:
        return "excluded"
    return "known" if len(set(valid)) == 1 else "unknown"
