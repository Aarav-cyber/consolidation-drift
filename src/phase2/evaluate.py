"""Phase 2 evaluator: map a consolidated memory back to its graded level.

Uses the per-item 1-5 scale stored in atomic_v1 (not the generic wording
from analyze_results.py), so grading matches the item's own anchors.
Negation items get a binary preserved/flipped/lost verdict instead.
"""
import re

from .llm import chat


def grade(client, target, memory, model, delay):
    """Return predicted level 1-5 for a graded target, or None if attribute is not present."""
    scale = "\n".join(f"{k} = {v}" for k, v in sorted(target["scale"].items()))
    prompt = (
        "You are evaluating the strength of a user's statement.\n\n"
        f"Consolidated memory:\n{memory}\n\n"
        f"The memory belongs to this 1-5 scale for '{target['attribute']}':\n"
        f"{scale}\n\n"
        "If the consolidated memory does not mention, retain, or reflect this attribute at all, "
        "return ONLY 'NOT_PRESENT'.\n"
        "Otherwise, return ONLY the integer 1, 2, 3, 4, or 5."
    )
    for _ in range(2):
        text = chat(client, prompt, model=model, delay=delay)
        if "NOT_PRESENT" in text.upper():
            return None
        m = re.search(r"\b[1-5]\b", text)
        if m:
            return int(m.group(0))
        m2 = re.search(r"[1-5]", text)
        if m2:
            return int(m2.group(0))
    raise RuntimeError(f"Evaluator returned neither 1-5 grade nor NOT_PRESENT: {text!r}")


def polarity(client, target, memory, model, delay):
    """Verdict for polarity/negation targets: PRESERVED, FLIPPED, or LOST.
    
    Strictly parses single-label responses using word boundaries and handles negation phrases
    like 'NOT PRESERVED' or 'UNPRESERVED' to prevent false positive matches.
    """
    expected = "TRUE" if target["true_value"] else "FALSE"
    prompt = (
        "You are checking whether a consolidated memory preserves a fact.\n\n"
        f"Original statement:\n{target['original_text']}\n\n"
        f"Consolidated memory:\n{memory}\n\n"
        f"The original statement is {expected}.\n"
        "Does the consolidated memory keep the same truth?\n"
        "- If it keeps the exact same truth/polarity, return ONLY 'PRESERVED'.\n"
        "- If it inverts or negates the truth (e.g. true became false or false became true), return ONLY 'FLIPPED'.\n"
        "- If the fact was omitted, deleted, or cannot be determined, return ONLY 'LOST'.\n\n"
        "Return ONLY one word: PRESERVED, FLIPPED, or LOST."
    )
    for _ in range(2):
        text = chat(client, prompt, model=model, delay=delay).strip()
        upper = text.upper()
        
        # Check for explicit negation of preserved (e.g. "NOT PRESERVED", "NOT PRESERVE")
        if re.search(r"\bNOT\s+PRESERVE[D]?\b", upper) or re.search(r"\bUNPRESERVED\b", upper):
            if re.search(r"\bFLIPPED\b", upper):
                return "FLIPPED"
            return "LOST"
            
        matches = set(re.findall(r"\b(PRESERVED|FLIPPED|LOST)\b", upper))
        if len(matches) == 1:
            return matches.pop()
            
    raise RuntimeError(f"Polarity evaluator returned ambiguous or unparseable response: {text!r}")
