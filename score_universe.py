"""I1: score the whole UK company universe for concealment shape.

Three streaming passes over the captured bulk (never loads the 5.7M registry or the
12.8GB PSC snapshot whole):
  pass 0  registry -> companies-per-address (mill signal)
  pass 1  PSC snapshot -> trimmed records per company (only the fields psc_features reads)
  pass 2  registry -> features (shared extractor) -> model score -> row

Writes `data/universe_scores.parquet` (every company: number, name, score, fired
flags) for the app to look up any pasted company, and inserts the case-control
subset into `concealment_dossiers` (scored, with evidence) for monitoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import hopsworks
from auditor import MODEL_VERSION, fired_flags, load_model, score_frame
from chair_features import (CONCEALMENT_FLAGS, MODEL_FEATURES, psc_features,
                            registry_features)

csv.field_size_limit(sys.maxsize)
DATA = Path(__file__).resolve().parent / "data"
_PSC_FIELDS = ("kind", "statement", "identification", "country_of_residence")


def addr_key(row: dict) -> str:
    return "|".join((row.get(k, "") or "").strip().upper() for k in (
        "RegAddress.AddressLine1", "RegAddress.PostTown", "RegAddress.PostCode"))


def pass_addr(limit):
    counts = defaultdict(int)
    zf = zipfile.ZipFile(DATA / "ch_basic.zip")
    with zf.open(zf.namelist()[0]) as fh:
        r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"))
        for i, row in enumerate(r):
            if limit and i >= limit:
                break
            counts[addr_key({k.strip(): v for k, v in row.items()})] += 1
    return counts


def pass_psc(limit):
    import re
    num_re = re.compile(rb'^\{"company_number":"([^"]+)"')
    by = defaultdict(list)
    zf = zipfile.ZipFile(DATA / "psc_snapshot.zip")
    with zf.open(zf.namelist()[0]) as fh:
        for i, raw in enumerate(fh):
            if limit and i >= limit:
                break
            m = num_re.match(raw)
            if not m:
                continue
            try:
                data = json.loads(raw).get("data") or {}
            except Exception:
                continue
            # keep only the fields psc_features reads, to bound memory
            trimmed = {k: data[k] for k in _PSC_FIELDS if k in data}
            by[m.group(1).decode("ascii", "ignore")].append(trimmed)
    return by


def score_all(addr_counts, psc_by, model, limit):
    rows = []
    zf = zipfile.ZipFile(DATA / "ch_basic.zip")
    with zf.open(zf.namelist()[0]) as fh:
        r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"))
        buf = []
        for i, raw in enumerate(r):
            if limit and i >= limit:
                break
            row = {k.strip(): v for k, v in raw.items()}
            num = row["CompanyNumber"].strip()
            feats = registry_features(row, addr_counts[addr_key(row)])
            feats.update(psc_features(psc_by.get(num, [])))
            feats["company_number"] = num
            feats["company_name"] = (row.get("CompanyName") or "").strip()
            buf.append(feats)
            if len(buf) >= 200_000:
                rows.append(_score_buf(buf, model)); buf = []
        if buf:
            rows.append(_score_buf(buf, model))
    return pd.concat(rows, ignore_index=True)


def _score_buf(buf, model):
    df = pd.DataFrame(buf)
    df["score"] = score_frame(df, model).values
    df["n_flags"] = df.apply(lambda r: len(fired_flags(r.to_dict())), axis=1)
    keep = ["company_number", "company_name", "score", "n_flags", "sic_code"] + MODEL_FEATURES
    return df[[c for c in keep if c in df.columns]]


# the parquet / dossier schema: score ranking is the signal (calibration holds only
# on the case-control prior), so we persist a population percentile for the app to
# present rank, not raw prob.
_OUT_COLS = (["company_number", "company_name", "score", "pct_rank", "n_flags",
             "incorporation_year", "sic_code", "company_status"] + list(CONCEALMENT_FLAGS))


def write_parquet(df):
    out = DATA / "universe_scores.parquet"
    df["pct_rank"] = df["score"].rank(pct=True)
    df[[c for c in _OUT_COLS if c in df.columns]].to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} companies)")
    return out


def write_dossiers(df, case_control):
    dossiers = df[df["company_number"].isin(case_control)].copy()
    dossiers = dossiers[[c for c in _OUT_COLS if c in dossiers.columns]]
    dossiers["model_version"] = MODEL_VERSION
    fs = hopsworks.login().get_feature_store()
    try:  # recreate for a clean, deterministic schema (FG, not a model)
        existing = fs.get_feature_group("concealment_dossiers", version=1)
        if existing is not None:
            existing.delete()
    except Exception:
        pass
    fg = fs.get_or_create_feature_group(
        name="concealment_dossiers", version=1,
        description="Concealment score + evidence for the case-control universe (inference log)",
        primary_key=["company_number"], online_enabled=False)
    fg.insert(dossiers)
    print(f"inserted {len(dossiers)} dossiers into concealment_dossiers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--from-parquet", action="store_true",
                    help="skip scoring; rebuild the dossiers FG from the existing parquet")
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    print(f"data dir: {DATA} (exists: {DATA.exists()})")

    cc = set(hopsworks.login().get_feature_store()
             .get_feature_group("company_registry", version=1).read()["company_number"].astype(str))
    print(f"case-control set: {len(cc)}")

    if args.from_parquet:
        df = pd.read_parquet(DATA / "universe_scores.parquet")
        print(f"loaded parquet: {len(df)} companies")
        write_dossiers(df, cc)
        return

    model = load_model()
    addr = pass_addr(args.limit); print(f"addresses: {len(addr)}")
    psc = pass_psc(args.limit); print(f"companies with PSC records: {len(psc)}")
    df = score_all(addr, psc, model, args.limit)
    print(f"scored: {len(df)} | score mean {df['score'].mean():.3f} max {df['score'].max():.3f}")
    write_parquet(df)
    write_dossiers(df, cc)


if __name__ == "__main__":
    main()
