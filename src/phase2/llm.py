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


def get_client():
    from groq import Groq
    return Groq(api_key=_load_keys()[0])


def _is_rate_limit(e):
    s = f"{type(e).__name__} {e}"
    return "429" in s or "ate limit" in s or "RateLimit" in type(e).__name__


def chat(client, prompt, model=DEFAULT_MODEL, retries=200, delay=2.0):
    """Single chat call with retry + pacing. Returns stripped text.

    Rotates across all GROQ_API_KEY + GROQ_API_KEYS_EXTRA keys on
    rate-limit errors, so free-tier TPD is pooled. Sleeps long only
    when every key is exhausted; background runs ride out the limits
    instead of dying (the runner is resume-aware regardless).
    """
    import re
    from groq import Groq
    keys = _load_keys()
    clients = [client] + [Groq(api_key=k) for k in keys[1:]]
    kid = 0
    last = None
    for attempt in range(retries):
        try:
            resp = clients[kid].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            time.sleep(delay)
            return text
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
