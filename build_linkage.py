"""I2: build the concealment-nest graph.

Among the highest-shape companies (top percentile), find the ones controlled by the
SAME beneficial owner. Two companies that declare the same corporate PSC, or the
same named person (name + birth year), are linked; a cluster of them under one owner
is a nest. This is the network boss wanted: not "same mill address" (an innocent
formation agent), but "same hand on multiple empty-chair-shaped shells".

Streams the PSC snapshot once, keeping owner identities only for the high-shape set
(bounded memory). Writes `data/linkage.parquet`: one row per nest (shared owner ->
member companies). The app renders each nest as an SVG constellation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from chair_features import normalize_name

DATA = Path(__file__).resolve().parent / "data"
PCTL = 0.99            # top 1% by concealment shape
MIN_MEMBERS = 2        # a nest is >= 2 companies under one owner
MAX_MEMBERS = 400      # above this the "owner" is a mass agent/registrar, not a nest
_NUM_RE = re.compile(rb'^\{"company_number":"([^"]+)"')
# owner names too generic to link on
_STOP = {"", "THE", "SECRETARY", "DIRECTOR", "LTD", "LIMITED"}


def owner_key(data: dict):
    """(kind, key, display) for the PSC owner, or None if not a linkable identity."""
    kind = data.get("kind", "")
    name = normalize_name(data.get("name") or "")
    if not name or name in _STOP or len(name) < 5:
        return None
    if "corporate" in kind or "legal-person" in kind:
        return ("corporate", name, data.get("name") or name)
    if "individual" in kind:
        dob = data.get("date_of_birth") or {}
        yr = dob.get("year")
        if not yr:
            return None  # a bare common name is too weak to link individuals
        return ("person", f"{name}|{yr}", f"{data.get('name') or name} (b.{yr})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    print(f"data dir: {DATA} (exists: {DATA.exists()})")

    scores = pd.read_parquet(DATA / "universe_scores.parquet",
                             columns=["company_number", "company_name", "score", "pct_rank"])
    scores["company_number"] = scores["company_number"].astype(str)
    hi = scores[scores["pct_rank"] >= PCTL]
    hi_set = set(hi["company_number"])
    name_of = dict(zip(hi["company_number"], hi["company_name"]))
    score_of = dict(zip(hi["company_number"], hi["score"].astype(float)))
    print(f"high-shape set (pctl>={PCTL}): {len(hi_set)} companies")

    owner_to_co = defaultdict(set)   # owner_key -> {company_number}
    owner_disp = {}
    zf = zipfile.ZipFile(DATA / "psc_snapshot.zip")
    with zf.open(zf.namelist()[0]) as fh:
        for i, raw in enumerate(fh):
            if args.limit and i >= args.limit:
                break
            m = _NUM_RE.match(raw)
            if not m or m.group(1).decode("ascii", "ignore") not in hi_set:
                continue
            try:
                data = json.loads(raw).get("data") or {}
            except Exception:
                continue
            ok = owner_key(data)
            if ok is None:
                continue
            kind, key, disp = ok
            owner_to_co[(kind, key)].add(m.group(1).decode("ascii", "ignore"))
            owner_disp[(kind, key)] = disp
    print(f"distinct owners over the high set: {len(owner_to_co)}")

    nests = []
    for (kind, key), cos in owner_to_co.items():
        if not (MIN_MEMBERS <= len(cos) <= MAX_MEMBERS):
            continue
        members = sorted(cos, key=lambda c: score_of.get(c, 0), reverse=True)
        nests.append({
            "owner_kind": kind,
            "owner_name": owner_disp[(kind, key)],
            "n_members": len(members),
            "mean_score": round(sum(score_of[c] for c in members) / len(members), 4),
            "members": json.dumps([{"number": c, "name": name_of.get(c, ""),
                                    "score": round(score_of.get(c, 0), 3)} for c in members]),
        })
    nests.sort(key=lambda n: (n["n_members"], n["mean_score"]), reverse=True)
    df = pd.DataFrame(nests)
    out = DATA / "linkage.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} nests")
    if len(df):
        print(df[["owner_kind", "owner_name", "n_members", "mean_score"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
