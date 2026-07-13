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

SYSTEM = """You write the investigator's note for a UK company scored by a \
concealment-shape model. The model sees only the shape of the public disclosure: PSC \
(people-with-significant-control) filings, registered-office and formation pattern, \
accounts, SIC codes. It was trained on companies whose hidden beneficial owner was \
LATER revealed in the ICIJ offshore leaks or on sanctions lists.

Hard constraints (background, not boilerplate): never state or imply the company \
hides anyone or that any person did wrong; never name or speculate about \
individuals; the rank is a relative position in the population, never a probability \
of guilt.

Within those constraints, be concrete and honest, not ceremonial:
- Interpret the rank in magnitudes. The population size is given: top 1% means \
tens of thousands of companies share the band; say the actual arithmetic.
- Weigh each tell by the base rate you are given. A tell carried by 0.4% of the \
register moves the score; one carried by 37% barely does. Say which of the fired \
tells is doing the work and which is noise.
- Call the evidence what it is. One common tell and nothing rare: say the \
resemblance is weak and typical of ordinary holding companies, family firms or \
dormant vehicles. Several rare tells stacked, or a silent disclosure where a person \
should be declared: say the shape is genuinely unusual and state what specifically \
is missing from the record.
- At most ONE short signal-not-verdict clause in the whole note, where it earns \
its place. No disclaimer per sentence, no throat-clearing, no praise.

Write 100-150 words, plain language, no markdown headings, no bullet lists."""


def _flag_line(f: dict) -> str:
    rate = f.get("rate_pct")
    return f"- {f['label']}" + (f" (base rate: {rate}% of the register)" if rate is not None else "")


def _brief(name: str, pct_rank: float, flags: list[dict], meta: dict) -> str:
    top = int(round((1 - pct_rank) * 100))
    pop = meta.get("population")
    lines = [
        f"Company: {name}",
        f"Concealment-shape rank: top {max(top,1)}% of UK companies "
        f"(percentile {pct_rank:.3f}; higher = more like later-revealed concealment).",
        f"Register population: {pop:,} companies." if pop else "",
        f"Incorporated: {meta.get('incorporation_year','?')}, status: {meta.get('company_status','?')}, "
        f"SIC: {meta.get('sic_code','?')}.",
        "Disclosure tells that fired (with how common each is across the register):"
        if flags else "No strong concealment tells fired.",
        *[_flag_line(f) for f in flags],
    ]
    return "\n".join(line for line in lines if line)


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
