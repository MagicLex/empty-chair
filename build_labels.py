"""F1: build the positive label set for empty-chair.

Positive = a UK company whose hidden interest was later revealed. Two sources:
- ICIJ Offshore Leaks: GB-linked entities and corporate officers.
- OpenSanctions: GB companies/organizations under sanctions.

Match their names to Companies House by deterministic normalization (case,
punctuation, suffix folding). Writes `revealed_owner` (company_number, source,
matched_name, match_confidence). Everything unmatched stays UNLABELED, not
negative: this is PU learning, so every downstream metric is a lower bound.

Reads the raw captures from HopsFS `data/`. Runs as a Hopsworks job or in a
terminal; it only streams, never loads the 5.7M registry into memory.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

import hopsworks
from chair_features import normalize_name

csv.field_size_limit(sys.maxsize)
DATA = Path(__file__).resolve().parent / "data"


def load_icij_names() -> dict[str, str]:
    """Normalized GB company-like name -> ICIJ source label."""
    names: dict[str, str] = {}
    ent = DATA / "icij" / "nodes-entities.csv"
    off = DATA / "icij" / "nodes-officers.csv"
    with open(ent, encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            if "GBR" in (row.get("country_codes") or ""):
                n = normalize_name(row.get("name") or "")
                if len(n) >= 5 and not n.isdigit():
                    names.setdefault(n, "icij-entity")
    with open(off, encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("name") or "").upper()
            if "GBR" in (row.get("country_codes") or "") and any(
                s in raw for s in (" LIMITED", " LTD", " LLP", " PLC")
            ):
                n = normalize_name(row.get("name") or "")
                if len(n) >= 5 and not n.isdigit():
                    names.setdefault(n, "icij-officer-corp")
    return names


def load_sanctions_names() -> dict[str, str]:
    names: dict[str, str] = {}
    path = DATA / "opensanctions_targets.csv"
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            if row.get("schema") in ("Company", "Organization", "LegalEntity") and "gb" in (
                row.get("countries") or ""
            ):
                for field in ("name", "aliases"):
                    for nm in (row.get(field) or "").split(";"):
                        n = normalize_name(nm)
                        if len(n) >= 5 and not n.isdigit():
                            names.setdefault(n, "opensanctions")
    return names


def match(targets: dict[str, str]) -> pd.DataFrame:
    """Stream Companies House basic data, emit one row per matched company."""
    zf = zipfile.ZipFile(DATA / "ch_basic.zip")
    hits: list[dict] = []
    seen: set[str] = set()
    with zf.open(zf.namelist()[0]) as fh:
        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"))
        header = [c.strip() for c in next(reader)]
        idx_name = header.index("CompanyName")
        idx_num = header.index("CompanyNumber")
        for row in reader:
            if len(row) <= idx_num:
                continue
            key = normalize_name(row[idx_name])
            if key in targets:
                num = row[idx_num].strip()
                if num in seen:
                    continue
                seen.add(num)
                # single-token generic names ("CONNECT") match spuriously; keep them
                # but at lower confidence so training and the audit can down-weight.
                conf = "exact-multitoken" if len(key.split()) >= 2 else "exact-single-token"
                hits.append(
                    {
                        "company_number": num,
                        "matched_name": key,
                        "source": targets[key],
                        "match_confidence": conf,
                        "is_revealed": 1,
                    }
                )
    return pd.DataFrame(hits)


def write_fg(df: pd.DataFrame) -> None:
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="revealed_owner",
        version=1,
        description="UK companies whose hidden interest was later revealed (ICIJ + OpenSanctions), the positive label set",
        primary_key=["company_number"],
        online_enabled=False,
    )
    fg.insert(df)
    for name, desc in {
        "company_number": "Companies House company number",
        "matched_name": "Normalized name that matched a leak/sanctions record",
        "source": "icij-entity | icij-officer-corp | opensanctions",
        "match_confidence": "How the match was made",
        "is_revealed": "Always 1 (positive class)",
    }.items():
        try:
            fg.update_feature_description(name, desc)
        except Exception as e:
            print(f"desc {name}: {e}")
    print(f"wrote {len(df)} revealed-owner labels")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = {**load_icij_names(), **load_sanctions_names()}
    print(f"normalized target names: {len(targets)}")
    df = match(targets)
    print(f"matched companies: {len(df)}")
    print(df["source"].value_counts().to_string())
    if args.dry_run:
        print(df.head(15).to_string())
        return
    write_fg(df)


if __name__ == "__main__":
    main()
