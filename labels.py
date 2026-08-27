import re
SUPPORT='support'; REFUTE='refute'; REFUSAL=''
def parse_verdict(resp):
    if not resp: return ''
    t=resp.lower().strip()
    if 'refute' in t or t == 'false': return 'refute'
    if 'support' in t or t == 'true': return 'support'
    return ''
def matches_label(verdict,gold):
    g=re.sub(r'[^\w]','',(gold or '')).lower()
    target={'true':'support','support':'support','supported':'support','false':'refute','refute':'refute','refuted':'refute'}.get(g)
    return verdict==target and bool(verdict)
def classify_kb(verdicts,n_runs=None,min_valid_ratio=1.0):
    total=n_runs if n_runs is not None else len(verdicts)
    valid=[v for v in verdicts if v]
    if total<=0 or len(valid)<min_valid_ratio*total or not valid: return 'excluded'
    return 'known' if len(set(valid))==1 else 'unknown'
