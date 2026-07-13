"""Conversational layer over the register (Anthropic tool use).

Not embeddings-RAG: the register is structured, so retrieval is deterministic
tools over the live dataframes and the linkage graph. The model never answers a
number, name or score it did not fetch through a tool. Same ethic as explain.py:
SIGNAL, NOT VERDICT.
"""

from __future__ import annotations

ASK_MODEL = "claude-sonnet-5"
MAX_TURNS = 6     # tool-use round trips per question
MAX_HISTORY = 6   # prior q/a pairs kept

SYSTEM = """You are the analyst behind "Empty Chair", a public register that ranks \
UK companies by how much their ownership DISCLOSURE is shaped like structures where \
a hidden owner was later revealed (ICIJ offshore leaks, sanctions lists).

Every fact comes from a tool call: never invent a number, score, name or company. \
If a tool returns nothing, say so plainly.

Hard constraints (background, state them at most once per answer, in one clause, \
only where it matters): the score is a relative rank of disclosure shape, not \
evidence of wrongdoing; never accuse or insinuate about a company or person; owner \
names are public PSC records, never speculate about individuals, their motives, \
origins or communities; if asked to accuse or expose, decline in one line and say \
what the register can honestly say instead.

Within that, be direct and useful, not ceremonial:
- Interpret figures concretely. A 99th percentile on 5.7M companies means roughly \
57,000 rank at or above it; do that arithmetic for the user.
- Weigh tells by their base rate: a tell 0.4% of the register carries fires real \
signal, a 37% one is close to noise. Say which is which.
- Call evidence weak or strong plainly. One common tell: weak, typical of holding \
companies and dormant vehicles. Several rare tells plus a dense shared-owner web: \
say the shape is genuinely unusual and what exactly is missing from the record.
- No hedging boilerplate, no flattery, no repeating the framing.

House voice: no em dashes (comma, period, colon, or restructure); no "it's not X, \
it's Y" constructions, say it straight; no closing flourish, end on the fact.

Answer in the language the user writes in. Under 180 words unless asked for depth. \
Plain prose, no markdown headings."""

TOOLS = [
    {"name": "lookup_company",
     "description": "Look up one UK company by company number or name. Returns its "
                    "concealment-shape profile: score, percentile, fired tells with "
                    "population base rates, status, SIC, and whether it sits in a "
                    "scored ownership nest.",
     "input_schema": {"type": "object", "properties": {
         "q": {"type": "string", "description": "company number or name"}},
         "required": ["q"]}},
    {"name": "ownership_web",
     "description": "The scored ownership web around a company: its declared owners "
                    "from the PSC register, the other high-shape companies those "
                    "owners control, and the companies shared between owners.",
     "input_schema": {"type": "object", "properties": {
         "number": {"type": "string", "description": "company number"}},
         "required": ["number"]}},
    {"name": "search_companies",
     "description": "Search companies by name substring; up to 10 best-scored "
                    "matches with scores and percentiles.",
     "input_schema": {"type": "object", "properties": {
         "name_contains": {"type": "string"}}, "required": ["name_contains"]}},
    {"name": "top_ranked",
     "description": "The companies with the highest concealment-shape scores on "
                    "file (up to 15).",
     "input_schema": {"type": "object", "properties": {
         "limit": {"type": "integer"}}, "required": []}},
    {"name": "register_stats",
     "description": "Population statistics: companies scored, base rate of each "
                    "disclosure tell, scored nests and linked webs, model version.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "owner_nests",
     "description": "Search scored nests by owner name; returns matching nests "
                    "(owner, kind, member count, mean score) with sample members.",
     "input_schema": {"type": "object", "properties": {
         "owner_name": {"type": "string"}}, "required": ["owner_name"]}},
]


def _messages(question, history, ctx):
    msgs = []
    for q, a in history[-MAX_HISTORY:]:
        msgs.append({"role": "user", "content": str(q)[:600]})
        msgs.append({"role": "assistant", "content": str(a)[:2000]})
    lead = f"[On screen right now: {str(ctx)[:300]}]\n" if ctx else ""
    msgs.append({"role": "user", "content": lead + question[:600]})
    return msgs


def _tool_round(resp, exec_tool):
    results = []
    for b in resp.content:
        if b.type == "tool_use":
            try:
                out = exec_tool(b.name, dict(b.input or {}))
            except Exception as e:
                out = f"tool error: {e}"
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": str(out)[:6000]})
    return results


BUDGET_MSG = "\nThe register could not settle this within the tool budget; ask a narrower question."


def run_ask(question, history, ctx, client, exec_tool, model=ASK_MODEL):
    """history: [(q, a), ...]; ctx: company number in view or None;
    exec_tool(name, args) -> str."""
    msgs = _messages(question, history, ctx)
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=model, max_tokens=900, thinking={"type": "disabled"},
            system=SYSTEM, tools=TOOLS, messages=msgs)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        msgs.append({"role": "assistant", "content": resp.content})
        msgs.append({"role": "user", "content": _tool_round(resp, exec_tool)})
    return BUDGET_MSG.strip()


def run_ask_stream(question, history, ctx, client, exec_tool, model=ASK_MODEL):
    """Same loop, streamed: yields answer tokens as they arrive and a dim
    [consulting <tool>...] status line for every tool call."""
    msgs = _messages(question, history, ctx)
    for _ in range(MAX_TURNS):
        with client.messages.stream(
                model=model, max_tokens=900, thinking={"type": "disabled"},
                system=SYSTEM, tools=TOOLS, messages=msgs) as stream:
            for delta in stream.text_stream:
                yield delta
            resp = stream.get_final_message()
        if resp.stop_reason != "tool_use":
            return
        for b in resp.content:
            if b.type == "tool_use":
                yield f"\n[consulting {b.name}…]\n"
        msgs.append({"role": "assistant", "content": resp.content})
        msgs.append({"role": "user", "content": _tool_round(resp, exec_tool)})
    yield BUDGET_MSG
