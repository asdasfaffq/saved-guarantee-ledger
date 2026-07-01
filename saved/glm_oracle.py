"""Real-LLM oracle via GLM-5.2 (ZhipuAI), for end-to-end semantic-filter + natural-drift experiments.

Security: the API key is NEVER stored in this repo. It is read from (in order):
  1. env var GLM_API_KEY
  2. the session scratchpad file (outside the repo, gitignored path)
Do not hardcode or commit the key. Cache responses so we never pay twice for the same doc.

Cost control: `thinking` disabled (GLM-5.2 is a reasoning model; disabling gives a 2-token
yes/no), max_tokens small, deterministic (temperature 0), on-disk cache keyed by md5(prompt).
"""
from __future__ import annotations
import hashlib, json, os, time, urllib.request

_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_MODEL = "glm-5.2"
_SCRATCH_KEY = ("/tmp/claude-1000/-home-lyx---/"
                "d85085a2-bb4b-44da-bdeb-689c33313d9f/scratchpad/glm_key")
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      ".glm_cache.jsonl")


def _key() -> str:
    k = os.environ.get("GLM_API_KEY")
    if k:
        return k.strip()
    if os.path.exists(_SCRATCH_KEY):
        return open(_SCRATCH_KEY).read().strip()
    raise RuntimeError("GLM key not found (set GLM_API_KEY or the scratchpad key file)")


_cache = None
def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(_CACHE):
            for line in open(_CACHE):
                try:
                    r = json.loads(line)
                    _cache[r["h"]] = r["v"]
                except Exception:
                    pass
    return _cache


def _append_cache(h, v):
    with open(_CACHE, "a") as f:
        f.write(json.dumps({"h": h, "v": v}) + "\n")
    _cache[h] = v


def classify_yes_no(text: str, question: str, max_chars: int = 1500,
                    retries: int = 3) -> int:
    """Return 1 if GLM answers yes to `question` about `text`, else 0. Cached, deterministic."""
    prompt = f"{question} Answer only yes or no.\n\nText: {text[:max_chars]}"
    h = hashlib.md5((_MODEL + "|" + prompt).encode()).hexdigest()
    c = _load_cache()
    if h in c:
        return c[h]
    body = json.dumps({
        "model": _MODEL, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 6, "temperature": 0,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                _ENDPOINT, data=body,
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read())
            ans = (d["choices"][0]["message"].get("content") or "").strip().lower()
            v = 1 if ans.startswith("y") else 0
            _append_cache(h, v)
            return v
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


if __name__ == "__main__":
    # smoke test (1 cached call)
    print("med? ->", classify_yes_no(
        "The patient was prescribed antibiotics for a bacterial infection.",
        "Is the following text about medicine?"))
    print("cache file:", _CACHE)
