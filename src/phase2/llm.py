"""Shared LLM client for Phase 2 consolidation + evaluation.

One variable at a time (MemDelta protocol): all pipelines share this
client, model, prompt wording, and delay. Only the pipeline logic differs.
"""
import os
import time

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"  # strongest text model on Groq; prior runs used gpt-oss-20b


def _load_keys():
    load_dotenv()
    keys = [os.getenv("GROQ_API_KEY")]
    extra = os.getenv("GROQ_API_KEYS_EXTRA", "")
    keys += [k.strip() for k in extra.split(",") if k.strip()]
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError(
            "GROQ_API_KEY missing. Create a .env file with "
            "GROQ_API_KEY=... (see README §12). .env is gitignored."
        )
    return keys


class _Obj:
    def __init__(self, d):
        self.__dict__.update({k: _wrap(v) for k, v in d.items()})


def _wrap(v):
    if isinstance(v, dict):
        return _Obj(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


class OpenAICompatClient:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only).

    Used for a self-hosted server (e.g. sglang) reached through an SSH tunnel.
    Mirrors the bit of the Groq SDK surface that chat() uses.
    """

    def __init__(self, base_url, api_key, timeout=900):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kwargs):
        import json
        import urllib.request
        body = {"model": model, "messages": messages, **kwargs}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return _wrap(json.loads(r.read()))


def get_client(backend="groq"):
    """backend="groq" (default) or "selfhosted" (SELFHOSTED_BASE_URL + SELFHOSTED_API_KEY)."""
    if backend == "selfhosted":
        load_dotenv()
        base, key = os.getenv("SELFHOSTED_BASE_URL"), os.getenv("SELFHOSTED_API_KEY")
        if not base or not key:
            raise RuntimeError("selfhosted backend needs SELFHOSTED_BASE_URL and SELFHOSTED_API_KEY in .env")
        return OpenAICompatClient(base, key)
    if backend != "groq":
        raise ValueError(f"unknown backend {backend!r}")
    from groq import Groq
    return Groq(api_key=_load_keys()[0])


START_CAP = 8192   # was the provider default (2048), which truncated long fuse steps
MAX_CAP = 32768


class EmptyReplyError(RuntimeError):
    """Model kept returning empty content; abort instead of poisoning the chain."""


def _is_rate_limit(e):
    s = f"{type(e).__name__} {e}"
    return "429" in s or "ate limit" in s or "RateLimit" in type(e).__name__


def chat(client, prompt, model=DEFAULT_MODEL, retries=200, delay=2.0, max_empty=6):
    """Single chat call with retry + pacing. Returns stripped text.

    Rotates across all GROQ_API_KEY + GROQ_API_KEYS_EXTRA keys on
    rate-limit errors, so free-tier TPD is pooled. Sleeps long only
    when every key is exhausted; background runs ride out the limits
    instead of dying (the runner is resume-aware regardless).
    """
    import re
    if isinstance(client, OpenAICompatClient):
        clients = [client]
    else:
        from groq import Groq
        keys = _load_keys()
        clients = [client] + [Groq(api_key=k) for k in keys[1:]]
    kid = 0
    last = None
    empties = 0
    cap = START_CAP
    for attempt in range(retries):
        try:
            resp = clients[kid].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=cap,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                # finish_reason == "length": reasoning ate the whole output cap
                # before any answer was written. Retrying at the same cap fails
                # every time, so raise it (does not change normal outputs).
                cap = min(cap * 2, MAX_CAP)
                # Reasoning models sometimes return empty content. Never pass an
                # empty memory down a chain: the next round would answer the
                # empty prompt ("please share your notes") instead of summarizing.
                empties += 1
                if empties > max_empty:
                    raise EmptyReplyError(f"model returned empty content {empties} times")
                print(f"  [empty reply {empties}/{max_empty}] retrying", flush=True)
                time.sleep(delay)
                continue
            time.sleep(delay)
            return text
        except EmptyReplyError:
            raise
        except Exception as e:  # noqa: BLE001 - quota/transient API errors
            last = e
            if _is_rate_limit(e) and len(clients) > 1:
                kid = (kid + 1) % len(clients)
                print(f"  [key {kid + 1}/{len(clients)}] rate-limited, rotating",
                      flush=True)
                time.sleep(delay)
                continue
            m = re.search(r"try again in (?:([\d.]+)m)?([\d.]+)s", str(e))
            if m:
                mins = float(m.group(1)) if m.group(1) else 0.0
                wait = min(mins * 60 + float(m.group(2)) + 5, 600)
            else:
                wait = 60
            print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}, "
                  f"sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


PRESERVE = """Preserve:
- the original meaning
- qualifiers
- uncertainty
- negation
- numerical values
- strength or intensity of statements

Do not add information.
Do not infer information that is not explicitly stated."""


def format_turns(target_text, turns):
    lines = [f"User memory to retain: {target_text}"]
    for t in turns:
        who = "User" if t.get("role") == "user" else "Agent"
        lines.append(f"{who}: {t['content']}")
    return "\n".join(lines)
