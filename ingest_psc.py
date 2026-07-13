"""F3: build the PSC-shape features for the case-control set.

Stream the Companies House PSC snapshot (line-delimited JSON, ~15.7M records,
12.8GB) once, keep only records whose company is in `company_registry`, group by
company, and run the shared `psc_features`. Companies in the set with NO record in
the snapshot get an all-absent row: that silence is itself the strongest
concealment tell.

Writes `psc_shape`. Deployed as a Hopsworks job. Cheap prefilter: pull the
company_number from the line head and only json.loads the ~selected lines.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import hopsworks
from chair_features import PSC_DOC, psc_features

DATA = Path(__file__).resolve().parent / "data"
_NUM_RE = re.compile(rb'^\{"company_number":"([^"]+)"')


def load_selected() -> set[str]:
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group("company_registry", version=1).read()
    return set(df["company_number"].astype(str))


def collect(selected: set[str], limit: int | None) -> dict[str, list]:
    by_company: dict[str, list] = defaultdict(list)
    zf = zipfile.ZipFile(DATA / "psc_snapshot.zip")
    with zf.open(zf.namelist()[0]) as fh:
        for i, raw in enumerate(fh):  # bytes, no decode until needed
            if limit and i >= limit:
                break
            m = _NUM_RE.match(raw)
            if not m:
                continue
            num = m.group(1).decode("ascii", "ignore")
            if num not in selected:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            data = rec.get("data") or {}
            if data:
                by_company[num].append(data)
    return by_company


def build(selected: set[str], by_company: dict[str, list]) -> pd.DataFrame:
    rows = []
    for num in selected:
        feats = psc_features(by_company.get(num, []))
        feats["company_number"] = num
        rows.append(feats)
    return pd.DataFrame(rows)


def write_fg(df: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="psc_shape",
        version=1,
        description="PSC concealment-shape features per company (declared, exempted, routed, silent)",
        primary_key=["company_number"],
        online_enabled=False,
    )
    fg.insert(df)
    for name, desc in PSC_DOC.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception as e:
            print(f"desc {name}: {e}")
    print(f"wrote {len(df)} psc-shape rows")


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
    by_company = collect(selected, args.limit)
    print(f"companies with >=1 PSC record: {len(by_company)}")
    df = build(selected, by_company)
    print(f"rows: {len(df)}")
    for col in ("psc_silence", "psc_corporate_only", "psc_absent", "psc_super_secure"):
        print(f"  {col}: {int(df[col].sum())}")
    if args.dry_run:
        print(df.head(10).to_string())
        return
    write_fg(df)


if __name__ == "__main__":
    main()
