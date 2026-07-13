"""Plain-language dossier for a scored company (Anthropic).

The model returns a concealment rank and a set of PSC/registry tells that fired.
This turns them into a short, honest dossier. The whole ethic: SIGNAL, NOT VERDICT.
A high rank means the company's ownership DISCLOSURE is shaped like structures where
a hidden owner was later revealed. It is never proof of wrongdoing, never an
accusation against any named person, and most flagged companies are entirely
legitimate (holding companies, family firms, dormant shells with nothing to hide).
"""

from __future__ import annotations

import anthropic

EXPLAIN_MODEL = "claude-sonnet-5"

SYSTEM = """You write a short investigator's note explaining why a concealment-shape \
model ranked a UK company where it did.

The model looks ONLY at the shape of a company's public disclosure: its PSC \
(people-with-significant-control) filings, its registered-office and formation \
pattern, its accounts and SIC codes. It was trained to recognise the disclosure \
shape of companies whose hidden beneficial owner was LATER revealed in the ICIJ \
offshore leaks or on sanctions lists.

Absolute rules, never break them:
- SIGNAL, NOT VERDICT. Never state or imply the company is hiding anyone, is a \
shell, is criminal, or that any person did anything wrong. Say only that its \
DISCLOSURE SHAPE resembles, or does not resemble, structures where concealment was \
later found.
- Most companies with this shape are legitimate: holding companies, family \
property firms, and dormant vehicles all look like this for innocent reasons. Say \
so when the evidence is thin.
- Ground every sentence in the tells you are given. Name the tell and what it means \
(e.g. "no natural-person PSC on file", "registered at an address shared by \
thousands of companies"). No hand-waving, no invented facts about the company.
- The rank is a RELATIVE position in the population, not a probability of guilt. \
Speak in terms of "ranks in the top X%", never "X% likely to be hiding someone".
- Never name or speculate about specific individuals.

Write 100-150 words, plain language, no markdown headings, no bullet lists."""


def _brief(name: str, pct_rank: float, flags: list[dict], meta: dict) -> str:
    top = int(round((1 - pct_rank) * 100))
    lines = [
        f"Company: {name}",
        f"Concealment-shape rank: top {max(top,1)}% of UK companies "
        f"(percentile {pct_rank:.3f}; higher = more like later-revealed concealment).",
        f"Incorporated: {meta.get('incorporation_year','?')}, status: {meta.get('company_status','?')}, "
        f"SIC: {meta.get('sic_code','?')}.",
        "Disclosure tells that fired:" if flags else "No strong concealment tells fired.",
        *[f"- {f['label']}" for f in flags],
    ]
    return "\n".join(lines)


def _prompt(name, pct_rank, flags, meta):
    return (f"{_brief(name, pct_rank, flags, meta)}\n\n"
            "Write the investigator's note for this company. Follow every rule.")


def explain_stream(name, pct_rank, flags, meta, client, model=EXPLAIN_MODEL):
    with client.messages.stream(
        model=model, max_tokens=600, thinking={"type": "disabled"},
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(name, pct_rank, flags, meta)}],
    ) as stream:
        for delta in stream.text_stream:
            yield delta


def explain(name, pct_rank, flags, meta, client, model=EXPLAIN_MODEL) -> str:
    resp = client.messages.create(
        model=model, max_tokens=600, thinking={"type": "disabled"},
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(name, pct_rank, flags, meta)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
