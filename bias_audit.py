"""Pre-publication bias / confound audit.

The network graph clustered high-shape companies under property developers, REITs,
and corporate-secretary agents. That raises the question the whole project has to
answer honestly before publishing: does the model separate CONCEALMENT, or does it
separate a legitimate STRUCTURE (corporate-owned holding / SPV / multinational) that
happens to correlate with the training labels and with certain business communities?

This job quantifies it, without inferring any protected attribute (which we neither
have nor should guess). It reports what the model actually keys on:
  - score concentration by SIC section (is it a property/holding detector?)
  - the structural composition of the top 1% vs the whole population
  - the same composition for the training positives (label bias)
  - active vs dissolved among the flagged (a dissolved shell hides no one now)
Writes docs/bias-audit.md. Runs as a job (reads the 5.7M parquet).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import hopsworks

DATA = Path(__file__).resolve().parent / "data"
FLAGS = ["psc_absent", "psc_silence", "psc_corporate_only", "psc_foreign_corporate",
         "is_mill_address", "is_holding_sic", "accounts_dormant"]
_SIC_LABEL = {"68": "real estate", "70": "head office / management consulting",
              "64": "financial holding / trusts", "82": "business support",
              "41": "construction", "62": "IT", "47": "retail", "99": "dormant"}


def pct(s):
    return f"{100 * float(s):.1f}%"


def compose(df):
    n = len(df)
    return {f: pct((df[f] == 1).mean()) for f in FLAGS if f in df} | {
        "active": pct(df["company_status"].astype(str).str.startswith("Active").mean()),
        "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)

    u = pd.read_parquet(DATA / "universe_scores.parquet")
    u["sic2"] = u["sic_code"].astype(str).str[:2]
    top = u[u["pct_rank"] >= 0.99]
    print(f"universe {len(u)}, top1% {len(top)}")

    # 1. SIC concentration in the top 1% vs population
    top_sic = (top["sic2"].value_counts(normalize=True) * 100).round(1)
    pop_sic = (u["sic2"].value_counts(normalize=True) * 100)
    sic_rows = []
    for s, share in top_sic.head(10).items():
        lift = share / pop_sic.get(s, 0.001)
        sic_rows.append((s, _SIC_LABEL.get(s, "other"), share, round(lift, 1)))

    # 2. structural composition: top 1% vs whole population
    comp_top, comp_all = compose(top), compose(u)

    # 3. label composition (training positives) via the FGs
    fs = hopsworks.login().get_feature_store()
    reg = fs.get_feature_group("company_registry", version=1).read()
    psc = fs.get_feature_group("psc_shape", version=1).read()
    lab = reg.merge(psc, on="company_number", how="left")
    pos = lab[lab["is_revealed"] == 1]
    comp_pos = compose(pos)

    # 4. mean score by whether holding-SIC / foreign-corporate (confound isolation)
    hold = u[u["is_holding_sic"] == 1]["score"].mean()
    nonhold = u[u["is_holding_sic"] == 0]["score"].mean()
    fcorp = u[u["psc_foreign_corporate"] > 0]["score"].mean()
    nofc = u[u["psc_foreign_corporate"] == 0]["score"].mean()

    lines = []
    lines.append("# empty-chair bias / confound audit\n")
    lines.append("Pre-publication check. No protected attribute is inferred. The question: "
                 "does the score track concealment, or a legitimate corporate/property structure "
                 "that correlates with both the labels and certain business communities?\n")
    lines.append("## 1. SIC concentration in the top 1%\n")
    lines.append("| SIC | activity | share of top 1% | lift vs population |")
    lines.append("|---|---|--:|--:|")
    for s, lab_, share, lift in sic_rows:
        lines.append(f"| {s} | {lab_} | {share}% | {lift}x |")
    lines.append("")
    lines.append("## 2. Structural composition: top 1% vs whole population\n")
    lines.append("| trait | top 1% | population | positives (labels) |")
    lines.append("|---|--:|--:|--:|")
    for f in FLAGS + ["active"]:
        lines.append(f"| {f} | {comp_top.get(f,'-')} | {comp_all.get(f,'-')} | {comp_pos.get(f,'-')} |")
    lines.append("")
    lines.append("## 3. Confound isolation (mean score)\n")
    lines.append(f"- holding-SIC companies: {hold:.3f}  vs  non-holding: {nonhold:.3f}")
    lines.append(f"- foreign-corporate PSC: {fcorp:.3f}  vs  none: {nofc:.3f}\n")
    lines.append("## Reading\n")
    lines.append("If the top 1% is dominated by real-estate / holding SIC at high lift, and "
                 "holding-SIC and foreign-corporate alone move the mean score a lot, then the model "
                 "is substantially a **corporate-structure detector**: it finds SPV / holding / "
                 "multinational vehicles, which include both genuine concealment and ordinary "
                 "legitimate structuring (property developers, REITs, group treasuries). It cannot "
                 "tell intent apart. Publishing named individuals as concealment-linked on this "
                 "basis is not defensible; the honest use is anonymous, structural triage.")

    out = Path(__file__).resolve().parent / "docs"
    out.mkdir(exist_ok=True)
    (out / "bias-audit.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out/'bias-audit.md'}")


if __name__ == "__main__":
    main()
