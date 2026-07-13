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

SYSTEM = """You are the conversational guide to "Empty Chair", a public register that \
ranks UK companies by how much their ownership DISCLOSURE is shaped like structures \
where a hidden owner was later revealed (ICIJ offshore leaks, sanctions lists).

You answer questions about the register: individual companies, their disclosure \
tells, the ownership webs (shared-owner graphs from the public PSC register), and \
population statistics. Use the tools for every fact; never invent a number, score, \
name or company. If a tool returns nothing, say so.

Absolute rules, never break them:
- SIGNAL, NOT VERDICT. Never state or imply a company hides anyone, is a shell, is \
criminal, or that any person did anything wrong. Say only that its disclosure shape \
resembles, or does not resemble, structures where concealment was later found.
- Most companies with this shape are legitimate: holding companies, family property \
firms, dormant vehicles. Say so when the evidence is thin.
- Ranks are relative positions in the population, never probabilities of guilt.
- Owner names come from the public PSC register and carry no judgement. Never \
speculate about individuals, their motives, origins or communities.
- If asked to accuse, expose or judge someone, decline and restate what the register \
can honestly say.
- Answer in the language the user writes in. Under 200 words, plain prose, no \
markdown headings."""

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
