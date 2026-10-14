"""Phase 2 consolidation pipelines.

Each pipeline maps a memory state m_k -> m_{k+1} under a fixed budget.
All share prompt wording (see llm.PRESERVE) so the pipeline — not the
prompt — is the experimental variable.

- rolling:      m_{k+1} = rewrite(m_k). Prior baseline (consolidate.py).
- sliding:      keep last W turns verbatim; summarize the older tail.
                Simulates a fixed window over an ongoing conversation:
                each round the oldest window turn falls into the tail.
- hierarchical: bottom-up re-derivation each round —
                turn notes -> chunks of 4 -> chunk summaries -> global
                (budget-capped). Next round re-chunks the global.

States are plain dicts/lists so chains serialize to JSON directly.
"""
import re

from .llm import PRESERVE, chat

WINDOW = 4
CHUNK = 4
NOTE_GROUP = 4  # turns compressed per note call; caps LLM calls/round


def rolling_step(client, memory, model, delay):
    prompt = (
        "You are a memory consolidation system.\n\n"
        "Rewrite the following memory into a concise form.\n\n"
        f"{PRESERVE}\n\nMemory:\n{memory}"
    )
    return chat(client, prompt, model=model, delay=delay)


def sliding_init(target_text, turns=None):
    """Initialize sliding window with target text (and optional initial turns)."""
    initial_window = [target_text]
    if turns:
        initial_window.extend([t["content"] if isinstance(t, dict) else str(t) for t in turns])
    return {"summary": "", "window": initial_window}


def sliding_render(state):
    parts = [state["summary"]] if state["summary"] else []
    parts += [f"- {w}" for w in state["window"]]
    return "\n".join(parts)


def sliding_step(client, state, model, delay, window_size=WINDOW, new_turns=None, recompact=True):
    """One sliding-window consolidation step.
    
    1. Ingests optional new incoming dialogue turns into the active window.
    2. Evicts older turns exceeding window_size and merges them into the running summary.
    3. If no turns were evicted but a running summary exists and recompact=True,
       performs an autonomous re-compaction pass on the summary.
    """
    window = list(state["window"])
    summary = state["summary"]
    
    if new_turns:
        window.extend([t["content"] if isinstance(t, dict) else str(t) for t in new_turns])

    tail_add = ""
    while len(window) > window_size and window:
        tail_add += (" " + window.pop(0)).strip()

    if tail_add:
        prompt = (
            "You are a memory consolidation system.\n\n"
            "Merge the following older memory into the running summary. "
            "Keep it concise.\n\n"
            f"{PRESERVE}\n\nRunning summary:\n{summary or '(empty)'}\n\n"
            f"Older memory to fold in:\n{tail_add}"
        )
        summary = chat(client, prompt, model=model, delay=delay)
    elif summary and recompact:
        # Periodic compaction pass on the existing summary to enforce brevity
        prompt = (
            "You are a memory consolidation system.\n\n"
            "Rewrite the following running summary to keep it concise, clear, and faithful.\n\n"
            f"{PRESERVE}\n\nRunning summary:\n{summary}"
        )
        summary = chat(client, prompt, model=model, delay=delay)

    new_state = {"summary": summary, "window": window}
    return new_state, sliding_render(new_state)


def hierarchical_step(client, statements, model, delay, budget_words=120,
                      chunk=CHUNK):
    """One bottom-up pass. Returns (global_memory, next_statements)."""
    notes = []
    for i in range(0, len(statements), NOTE_GROUP):
        group = "\n".join(f"- {s}" for s in statements[i:i + NOTE_GROUP])
        prompt = (
            "Compress the following conversation turns, one line per turn. "
            "Keep numbers, qualifiers, uncertainty, and strength words.\n\n"
            f"{PRESERVE}\n\nTurns:\n{group}"
        )
        notes.append(chat(client, prompt, model=model, delay=delay))
    chunk_sums = []
    for i in range(0, len(notes), chunk):
        piece = "\n".join(f"- {n}" for n in notes[i:i + chunk])
        prompt = (
            "Summarize the following turn-notes into 1-2 lines for a "
            "conversation memory.\n\n"
            f"{PRESERVE}\n\nNotes:\n{piece}"
        )
        chunk_sums.append(chat(client, prompt, model=model, delay=delay))
    joined = "\n".join(f"- {c}" for c in chunk_sums)
    prompt = (
        f"Fuse the following chunk summaries into one global memory of at "
        f"most {budget_words} words, structured as concise bullet points or clean statements.\n\n"
        f"{PRESERVE}\n\nChunks:\n{joined}"
    )
    glob = chat(client, prompt, model=model, delay=delay)
    # Extract clean statements: strip markdown bullets/numbering and filter out empty or short fragments (<10 chars)
    lines = []
    for raw in glob.splitlines():
        cleaned = re.sub(r"^[-*•\d.)]+\s*", "", raw.strip()).strip()
        if len(cleaned) >= 10:
            lines.append(cleaned)
    if not lines:
        sentences = re.split(r"(?<=[.!?])\s+", glob.replace("\n", " "))
        lines = [s.strip() for s in sentences if len(s.strip()) >= 10]
    next_statements = lines if lines else [glob.strip()]
    return glob, next_statements


def hierarchical_init(target_text, turns):
    return [target_text] + [t["content"] for t in turns]
