"""Honesty controls for the autoresearch jul14 winner (a3a9d7c).

Control A: shuffle train labels -> PR-AUC on true holdout must collapse to ~base rate.
Control B: demographics only (incorporation year, SIC, region) with the same TE
           treatment -> must stay well below the winner; within 30% = TAINTED.

Same frozen split as train.py. Never touches it.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chair_features import CAT_FEATURES as CAT
from chair_features import NUM_FEATURES as NUM

# mirrored from train.py (importing it would arm its 600s alarm)
CACHE = "/tmp/claude-1235/-hopsfs-Users-meb10000/c5cbd640-6f6a-4a47-8966-ace5f607a96a/scratchpad/empty_chair_frame.parquet"
LABEL = "is_revealed"
GROUP = "office_group"
TE_SMOOTH = 20

WINNER = 0.376826


def winner_model():
    def lgbm(seed):
        return LGBMClassifier(
            n_estimators=1500, learning_rate=0.02, num_leaves=31,
            reg_lambda=5.0, scale_pos_weight=1.0, min_child_samples=80,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            random_state=seed, n_jobs=3, verbose=-1)
    return VotingClassifier([(f"s{s}", lgbm(s)) for s in range(10)], voting="soft")


def pipe(cols_cat):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cols_cat)],
        remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", winner_model())])


def te(Xtr, Xte, ytr, groups_tr, cols):
    prior = ytr.mean()
    out = []
    for c in cols:
        oof = np.full(len(Xtr), prior)
        for fi, vi in GroupKFold(n_splits=5).split(Xtr, ytr, groups_tr):
            agg = pd.Series(ytr[fi]).groupby(Xtr[c].iloc[fi].values).agg(["sum", "count"])
            smooth = (agg["sum"] + TE_SMOOTH * prior) / (agg["count"] + TE_SMOOTH)
            oof[vi] = Xtr[c].iloc[vi].map(smooth).fillna(prior).values
        Xtr[c + "_te"] = oof
        agg = pd.Series(ytr).groupby(Xtr[c].values).agg(["sum", "count"])
        smooth = (agg["sum"] + TE_SMOOTH * prior) / (agg["count"] + TE_SMOOTH)
        Xte[c + "_te"] = Xte[c].map(smooth).fillna(prior).values
        out.append(c + "_te")
    return out


def add_features(df):
    tells = ["psc_absent", "psc_silence", "psc_corporate_only", "psc_super_secure",
             "psc_exempt", "is_mill_address", "is_holding_sic", "accounts_dormant"]
    df["n_tells"] = df[tells].sum(axis=1)
    df["mill_x_no_individual"] = df["is_mill_address"] * df["psc_has_no_individual"]
    df["mill_x_silence"] = df["is_mill_address"] * df["psc_silence"]
    df["dormant_x_corp_only"] = df["accounts_dormant"] * df["psc_corporate_only"]
    df["holding_x_foreign"] = df["is_holding_sic"] * (df["psc_foreign_corporate"] > 0).astype(int)
    df["foreign_corp_ratio"] = df["psc_foreign_corporate"] / df["psc_n_corporate"].clip(lower=1)
    return ["n_tells", "mill_x_no_individual", "mill_x_silence",
            "dormant_x_corp_only", "holding_x_foreign", "foreign_corp_ratio"]


def main():
    df = pd.read_parquet(CACHE)
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    y = df[LABEL].values
    groups = df[GROUP].astype(str).values

    # FROZEN split, identical to train.py
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, teix = next(gss.split(df, y, groups))
    extra = add_features(df)
    yte = y[teix]
    base = yte.mean()
    print(f"holdout base rate: {base:.4f}")

    # ---- Control A: shuffled train labels, full winner feature set
    Xtr, Xte = df.iloc[tr].copy(), df.iloc[teix].copy()
    ysh = np.random.default_rng(0).permutation(y[tr])
    te_cols = te(Xtr, Xte, ysh, groups[tr], ["post_area", "sic_section", "sic_code"])
    cols = CAT + NUM + extra + te_cols
    m = pipe(CAT)
    m.fit(Xtr[cols], ysh)
    ap = average_precision_score(yte, m.predict_proba(Xte[cols])[:, 1])
    print(f"control_shuffle_pr_auc: {ap:.6f} (expect ~{base:.3f})")

    # ---- Control B: demographics only (inc year, SIC, region), same TE treatment
    Xtr, Xte = df.iloc[tr].copy(), df.iloc[teix].copy()
    ytr = y[tr]
    te_cols = te(Xtr, Xte, ytr, groups[tr], ["post_area", "sic_section", "sic_code"])
    demo_cat = ["post_area", "sic_section", "country"]
    cols = demo_cat + ["incorporation_year"] + te_cols
    m = pipe(demo_cat)
    m.fit(Xtr[cols], ytr)
    ap = average_precision_score(yte, m.predict_proba(Xte[cols])[:, 1])
    print(f"control_demographics_pr_auc: {ap:.6f} (winner {WINNER}; TAINTED if >= {0.7 * WINNER:.4f})")


if __name__ == "__main__":
    main()
