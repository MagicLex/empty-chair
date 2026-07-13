"""F2: build the case-control registry frame for empty-chair.

Keep every positive (from `revealed_owner`), sample unlabeled controls stratified
on incorporation-year x SIC-section at ~20:1. The stratification removes the cheap
demographic confounders so the model must learn concealment shape, not age/sector.

Two streaming passes over the Companies House basic zip:
  pass 1  count companies per registered-office address (the mill signal), and
          histogram the positives' strata.
  pass 2  emit every positive + per-stratum reservoir-sampled controls, each with
          registry features from the shared extractor.

Writes `company_registry`. Deployed as a Hopsworks job (streams 5.7M rows twice).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import hopsworks
from chair_features import REGISTRY_DOC, _year, primary_sic, registry_features

csv.field_size_limit(sys.maxsize)
# default when run from the FUSE repo; a job overrides it with --data-dir since a
# job pod may stage the script outside /hopsfs while still mounting the FUSE home.
DATA = Path(__file__).resolve().parent / "data"
CONTROL_RATIO = 20


def _addr_key(row: dict) -> str:
    parts = [row.get(k, "") or "" for k in (
        "RegAddress.AddressLine1", "RegAddress.PostTown", "RegAddress.PostCode")]
    return "|".join(p.strip().upper() for p in parts)


def _stratum(row: dict) -> str:
    return f"{_year(row.get('IncorporationDate',''))}|{primary_sic(row)[:2]}"


def load_positives() -> set[str]:
    fs = hopsworks.login().get_feature_store()
    df = fs.get_feature_group("revealed_owner", version=1).read()
    return set(df["company_number"].astype(str))


def pass1(positives: set[str], limit: int | None) -> tuple[dict, dict]:
    addr_count: dict[str, int] = defaultdict(int)
    pos_strata: dict[str, int] = defaultdict(int)
    zf = zipfile.ZipFile(DATA / "ch_basic.zip")
    with zf.open(zf.namelist()[0]) as fh:
        reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"))
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            row = {k.strip(): v for k, v in row.items()}
            addr_count[_addr_key(row)] += 1
            if row["CompanyNumber"].strip() in positives:
                pos_strata[_stratum(row)] += 1
    return addr_count, pos_strata


def pass2(positives: set[str], addr_count: dict, pos_strata: dict, limit: int | None) -> pd.DataFrame:
    # per-stratum control quota, reservoir-sampled deterministically (no RNG in the pod)
    quota = {k: v * CONTROL_RATIO for k, v in pos_strata.items()}
    taken: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    zf = zipfile.ZipFile(DATA / "ch_basic.zip")
    with zf.open(zf.namelist()[0]) as fh:
        reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"))
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            row = {k.strip(): v for k, v in row.items()}
            num = row["CompanyNumber"].strip()
            is_pos = num in positives
            strat = _stratum(row)
            if not is_pos:
                if taken[strat] >= quota.get(strat, 0):
                    continue
                taken[strat] += 1
            akey = _addr_key(row)
            feats = registry_features(row, addr_count[akey])
            # stable address-cluster id so the training split can group by mill and
            # never let one formation mill straddle train and test (the leak).
            feats.update(
                company_number=num,
                is_revealed=int(is_pos),
                office_group=hashlib.md5(akey.encode()).hexdigest()[:12],
            )
            rows.append(feats)
    return pd.DataFrame(rows)


def write_fg(df: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="company_registry",
        version=1,
        description="Case-control registry-shape features for UK companies (all positives + 20:1 stratified controls)",
        primary_key=["company_number"],
        online_enabled=False,
    )
    fg.insert(df)
    for name, desc in REGISTRY_DOC.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception as e:
            print(f"desc {name}: {e}")
    print(f"wrote {len(df)} registry rows ({int(df['is_revealed'].sum())} positive)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap rows scanned (smoke test)")
    ap.add_argument("--data-dir", default=None, help="absolute path to the captures dir (FUSE)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    print(f"data dir: {DATA} (exists: {DATA.exists()})")

    positives = load_positives()
    print(f"positives: {len(positives)}")
    addr_count, pos_strata = pass1(positives, args.limit)
    print(f"unique addresses: {len(addr_count)} | positive strata: {len(pos_strata)}")
    df = pass2(positives, addr_count, pos_strata, args.limit)
    print(f"frame: {len(df)} rows, {int(df['is_revealed'].sum())} positive")
    print(df["is_mill_address"].value_counts().to_string())
    if args.dry_run:
        print(df.head(10).to_string())
        return
    write_fg(df)


if __name__ == "__main__":
    main()
