"""
pcd_common.py — configuration, paths, and the LLM client shared by
counter.py and evaluate.py.

There is exactly ONE request path. Earlier versions branched on model name
(`qwen3-32b`, `gemini-2.5-flash`, `llama3:8b`, ...), and two of those branches
silently dropped the `temperature` argument, so `--temperature 0.3` was only
honoured for some models. Provider differences belong in the caller's
environment (OPENAI_BASE_URL) and, where a vendor needs a non-standard body
field, in --extra-body. Nothing here knows any model's name.

To point at a different provider, set OPENAI_BASE_URL:

    # OpenAI
    export OPENAI_BASE_URL=https://api.openai.com/v1
    # local Ollama (OpenAI-compatible endpoint)
    export OPENAI_BASE_URL=http://localhost:11434/v1 ; export OPENAI_API_KEY=ollama
    # any other OpenAI-compatible gateway
    export OPENAI_BASE_URL=https://<host>/compatible-mode/v1

Vendor-specific request fields go through --extra-body, e.g. disabling Qwen's
thinking mode:

    --extra-body '{"enable_thinking": false}'
"""
import json
import os
import random
import re
import time

from openai import OpenAI

__all__ = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "DATA_ROOT", "OUTPUT_ROOT",
    "set_mock", "is_mock", "llm_request", "ensure_dir", "read_jsonl",
    "write_jsonl", "read_json", "write_json", "add_llm_args", "LLMError",
]

# --------------------------------------------------------------------------
# Configuration — environment only; nothing is hardcoded
# --------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("BASE_DIR", _THIS_DIR)
DATA_ROOT = os.environ.get("PCD_DATA_ROOT", BASE_DIR)
OUTPUT_ROOT = os.environ.get("PCD_OUTPUT_ROOT", BASE_DIR)

_MOCK = os.environ.get("MOCK_LLM", "false").strip().lower() in ("1", "true", "yes", "on")

MAX_RETRIES = int(os.environ.get("PCD_MAX_RETRIES", "4"))
RETRY_BASE_DELAY = float(os.environ.get("PCD_RETRY_BASE_DELAY", "1.0"))
THROTTLE = float(os.environ.get("PCD_THROTTLE", "0.05"))


class LLMError(RuntimeError):
    """Raised when a request still fails after MAX_RETRIES attempts."""


def set_mock(flag):
    global _MOCK
    _MOCK = bool(flag)


def is_mock():
    return _MOCK


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def read_jsonl(path):
    """Read line-delimited JSON, tolerating a plain JSON array."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  skipping malformed line {lineno} of {path}: {e}")
    return out


def write_jsonl(path, records):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "    export OPENAI_API_KEY=sk-...\n"
                "    export OPENAI_BASE_URL=https://api.openai.com/v1   # or your provider\n"
                "...or pass --mock-llm to run offline with canned responses.\n"
                "Never hardcode credentials into these files."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def _mock_response(prompt):
    """Canned responses for offline smoke tests.

    The entity branches echo real tokens out of the prompt so that substitution
    genuinely rewrites the evidence — a mock that returned invented entities
    would be filtered out by word_level_counter's no-op guard and the pipeline
    would never be exercised end to end.
    """
    low = prompt.lower()
    # verdict prompts are the only ones naming both labels
    if "support" in low and "refute" in low:
        return random.choice(["support", "refute"])
    if "extract the entities" in low:
        body = prompt.split("Evidence:", 1)[-1]
        words, seen = [], set()
        for w in re.findall(r"[A-Za-z][A-Za-z'-]{5,}", body):
            if w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
            if len(words) == 8:
                break
        return ", ".join(words)
    if "only output the new entity" in low:
        ent = prompt.split("'")[1] if "'" in prompt else "entity"
        return "Mock" + ent.title().replace(" ", "")
    return "<MOCK RESPONSE> Fabricated evidence for offline testing."


def llm_request(prompt, model_name, temperature, max_tokens=None, extra_body=None):
    """One request, one code path, for every model.

    `temperature` is required and is always sent — this is deliberate. Returns
    the response text, or "" if the model produced an empty completion.
    Raises LLMError when the call fails after MAX_RETRIES attempts, so a
    transient outage never gets silently recorded as model behaviour.
    """
    if _MOCK:
        return _mock_response(prompt)

    client = _get_client()
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            if THROTTLE:
                time.sleep(THROTTLE)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  request failed ({type(e).__name__}: {e}); retrying in {delay:.1f}s "
                  f"[{attempt + 1}/{MAX_RETRIES - 1}]")
            time.sleep(delay)
    raise LLMError(f"request failed after {MAX_RETRIES} attempts: {last_err}")


def add_llm_args(parser):
    """Flags shared by counter.py and evaluate.py."""
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Cap on completion length. Leave unset for the provider default.")
    parser.add_argument("--extra-body", type=str, default=None,
                        help='JSON object merged into the request body, for provider-specific '
                             'fields, e.g. \'{"enable_thinking": false}\'.')
    parser.add_argument("--mock-llm", dest="mock_llm", action="store_true",
                        help="Canned responses, no network. Same as MOCK_LLM=true.")


def parse_extra_body(raw):
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--extra-body is not valid JSON: {e}")
    if not isinstance(value, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return value
