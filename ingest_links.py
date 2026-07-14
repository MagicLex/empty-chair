"""F4: owner-degree features from the full PSC snapshot.

One stream over the 15.7M-record snapshot: count companies per normalized owner
name across the WHOLE register (dedup by company+owner pair), then aggregate the
counts over each case-control company's named owners via the shared
`owner_degree_features`. Statements and exemptions carry no name and stay out.

Writes `owner_degree`; joined by the feature view, never touches psc_shape.
Deployed as a Hopsworks job.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import hopsworks
from chair_features import OWNER_DOC, normalize_name, owner_degree_features

DATA = Path(__file__).resolve().parent / "data"
_NUM_RE = re.compile(rb'^\{"company_number":"([^"]+)"')
_PSC_KINDS = ("individual-person-with-significant-control",
              "corporate-entity-person-with-significant-control",
              "legal-person-person-with-significant-control")


def load_selected() -> set[str]:
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group("company_registry", version=1).read()
    return set(df["company_number"].astype(str))


def stream(selected: set[str], limit: int | None):
    """counts: owner key -> distinct companies; owners_of: selected company -> keys."""
    counts = defaultdict(int)
    owners_of = defaultdict(list)
    seen = set()  # hash of (company, owner) pairs, dedups reappointments
    zf = zipfile.ZipFile(DATA / "psc_snapshot.zip")
    with zf.open(zf.namelist()[0]) as fh:
        for i, raw in enumerate(fh):
            if limit and i >= limit:
                break
            m = _NUM_RE.match(raw)
            if not m:
                continue
            num = m.group(1).decode("ascii", "ignore")
            try:
                data = json.loads(raw).get("data") or {}
            except Exception:
                continue
            if data.get("kind") not in _PSC_KINDS:
                continue
            key = normalize_name(data.get("name") or "")
            if not key:
                continue
            pair = hash((num, key))
            if pair in seen:
                continue
            seen.add(pair)
            counts[key] += 1
            if num in selected:
                owners_of[num].append(key)
    return counts, owners_of


def build(selected: set[str], counts, owners_of) -> pd.DataFrame:
    rows = []
    for num in selected:
        feats = owner_degree_features(owners_of.get(num, []), counts)
        feats["company_number"] = num
        rows.append(feats)
    return pd.DataFrame(rows)


def write_fg(df: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="owner_degree",
        version=1,
        description="Owner-degree per company: how many companies its named PSC owners control across the whole register",
        primary_key=["company_number"],
        online_enabled=False,
    )
    fg.insert(df)
    for name, desc in OWNER_DOC.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception as e:
            print(f"desc {name}: {e}")
    print(f"wrote {len(df)} owner-degree rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    print(f"data dir: {DATA} (exists: {DATA.exists()})")

    selected = load_selected()
    print(f"selected companies: {len(selected)}")
    counts, owners_of = stream(selected, args.limit)
    print(f"named owners on register: {len(counts)}; selected with >=1 named owner: {len(owners_of)}")
    df = build(selected, counts, owners_of)
    top = df.nlargest(5, "owner_max_companies")[["company_number", "owner_max_companies", "owner_n_named"]]
    print(f"rows: {len(df)}\n{top.to_string(index=False)}")
    if args.dry_run:
        return
    write_fg(df)


if __name__ == "__main__":
    main()
