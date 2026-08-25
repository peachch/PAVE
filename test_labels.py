"""Regression tests for labels.py, contrasted against the old parser.

Run:  python test_labels.py
"""
import re
import string
import sys

from labels import parse_verdict, classify_kb, matches_label

# ---- the parser currently in evaluate.py, for side-by-side comparison ----
_S = {'support', 'supports', 'true', 'yes'}
_R = {'refute', 'refutes', 'false', 'no'}


def old_parser(resp):
    if not resp:
        return ""
    text = resp.lower()
    tn = text.translate(str.maketrans('', '', string.punctuation))
    for tok in re.split(r'\s+', tn):
        if tok in _S:
            return 'support'
        if tok in _R:
            return 'refute'
    if any(k in text for k in ('support', 'supports')) and 'refute' not in text:
        return 'support'
    if 'refute' in text or 'refutes' in text:
        return 'refute'
    return ""


CASES = [
    # (response, expected)
    ("support", "support"),
    ("Support.", "support"),
    ("refute", "refute"),
    ("Refute", "refute"),
    ("Yes", "support"),
    ("No", "refute"),
    ("**support**", "support"),
    ("Answer: refute", "refute"),
    ("Based on my knowledge: refute", "refute"),
    ("support (the evidence agrees)", "support"),
    ("The claim is supported.", "support"),
    ("The claim is true.", "support"),
    ("The claim is false.", "refute"),
    ("No, the claim is false.", "refute"),

    # negation — the old parser gets every one of these backwards
    ("The claim should not be supported.", "refute"),
    ("This claim cannot be supported by the evidence.", "refute"),
    ("The claim is not true.", "refute"),
    ("The evidence does not refute the claim.", "support"),
    ("This is not accurate.", "refute"),
    ("The statement isn't supported by the sources.", "refute"),

    # conclusion comes last, after reasoning
    ("I would say support, not refute.", "support"),
    ("At first glance this looks like refute, but the final answer is support.", "support"),
    ("Reasoning: the figure matches the report. Answer: support", "support"),
    ("It is false. Actually, on reflection, the claim is supported.", "support"),

    # refusals / hedges — must be '' so they can be excluded, not scored
    ("I cannot support or refute this claim without more information.", ""),
    ("It is neither support nor refute.", ""),
    ("I'm sorry, I can't answer that.", ""),
    ("I cannot determine whether this claim is true.", ""),
    ("There is insufficient information to judge.", ""),
    ("Unable to verify this claim.", ""),
    ("The answer is unclear.", ""),
    ("", ""),
    ("   ", ""),
]

KB_CASES = [
    # (verdicts, n_runs, expected, note)
    (["support"] * 10, 10, "known", "unanimous"),
    (["refute"] * 10, 10, "known", "unanimous"),
    (["support"] * 6 + ["refute"] * 4, 10, "unknown", "genuine disagreement"),
    (["support"] * 9 + [""], 10, "excluded", "one refusal breaks strict consistency"),
    (["support"] + [""] * 9, 10, "excluded", "1 support + 9 refusals  <-- your question 3"),
    ([""] * 10, 10, "excluded", "all refusals"),
    (["support"] * 8, 10, "excluded", "claim missing from 2 runs"),
]


def main():
    fails = 0
    changed = []
    print(f"{'response':62} {'old':>9}  {'new':>9}")
    print("-" * 86)
    for resp, expected in CASES:
        got, old = parse_verdict(resp), old_parser(resp)
        ok = got == expected
        fails += not ok
        mark = "" if ok else "   <-- FAIL"
        if old != got:
            changed.append((resp, old, got))
        disp = (resp[:59] + "...") if len(resp) > 62 else resp
        print(f"{disp!r:62} {old!r:>9}  {got!r:>9}{mark}")

    print()
    print(f"{'prior-only verdicts':52} {'expected':>10}  {'got':>10}")
    print("-" * 86)
    for verdicts, n, expected, note in KB_CASES:
        got = classify_kb(verdicts, n_runs=n)
        ok = got == expected
        fails += not ok
        print(f"{note:52} {expected:>10}  {got:>10}{'' if ok else '   <-- FAIL'}")

    assert matches_label("support", "True") and matches_label("refute", "False")
    assert matches_label("support", "Supports") and not matches_label("", "True")

    print()
    print(f"{len(changed)} of {len(CASES)} responses parse differently than before:")
    for resp, old, new in changed:
        print(f"   {old!r:9} -> {new!r:9}   {resp[:60]!r}")
    print()
    print("FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
