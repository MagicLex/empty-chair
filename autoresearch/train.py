"""autoresearch jul14: the ONE iterated training file.

Frozen contract (never change):
  - data: feature view empty_chair_fv v1 frame, read like train_chair.py
  - split: GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0) by office_group
  - val_metric: average_precision_score (PR-AUC) on that holdout
  - prints exactly three final lines: val_metric / peak_memory_gb / training_seconds

Experiment knobs live in build_model() / feature lists only.
"""

from __future__ import annotations

import os
import resource
import signal
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chair_features import CAT_FEATURES as CAT
from chair_features import NUM_FEATURES as NUM

LABEL = "is_revealed"
GROUP = "office_group"
CACHE = "/tmp/claude-1235/-hopsfs-Users-meb10000/c5cbd640-6f6a-4a47-8966-ace5f607a96a/scratchpad/empty_chair_frame.parquet"
ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
WALL_CAP_S = 600

T0 = time.time()


def finish(val_metric: float):
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"val_metric: {val_metric}")
    print(f"peak_memory_gb: {round(peak_gb, 3)}")
    print(f"training_seconds: {round(time.time() - T0, 1)}")
    sys.exit(0)


def _timeout(signum, frame):
    print("WALL CLOCK CAP HIT (600s), aborting run")
    finish(0.0)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(WALL_CAP_S)


def get_frame() -> pd.DataFrame:
    if os.path.exists(CACHE) and "--refresh" not in sys.argv:
        return pd.read_parquet(CACHE)
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    reg = fs.get_feature_group("company_registry", version=1)
    psc = fs.get_feature_group("psc_shape", version=1)
    query = reg.select_all().join(psc.select_except(["company_number"]), on=["company_number"])
    fv = fs.get_or_create_feature_view(
        name="empty_chair_fv", version=1, query=query, labels=[LABEL],
        description="Registry + PSC concealment shape for UK companies; label = later-revealed hidden owner",
    )
    df = query.read()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_parquet(CACHE)
    return df


# ---------------------------------------------------------------- experiment area

def build_model(cols_cat, cols_num):
    """Exp: LightGBM with scale_pos_weight instead of HGB balanced."""
    from lightgbm import LGBMClassifier
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cols_cat),
         ("num", "passthrough", cols_num)])
    clf = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63,
        reg_lambda=1.0, scale_pos_weight=19.7, min_child_samples=40,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        random_state=0, n_jobs=4, verbose=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def add_features(df: pd.DataFrame):
    """Derived features computed from the disclosure columns only.
    Returns (df, extra_cat, extra_num)."""
    tells = ["psc_absent", "psc_silence", "psc_corporate_only", "psc_super_secure",
             "psc_exempt", "is_mill_address", "is_holding_sic", "accounts_dormant"]
    df["n_tells"] = df[tells].sum(axis=1)
    df["mill_x_no_individual"] = df["is_mill_address"] * df["psc_has_no_individual"]
    df["mill_x_silence"] = df["is_mill_address"] * df["psc_silence"]
    df["dormant_x_corp_only"] = df["accounts_dormant"] * df["psc_corporate_only"]
    df["holding_x_foreign"] = df["is_holding_sic"] * (df["psc_foreign_corporate"] > 0).astype(int)
    df["foreign_corp_ratio"] = df["psc_foreign_corporate"] / df["psc_n_corporate"].clip(lower=1)
    extra_num = ["n_tells", "mill_x_no_individual", "mill_x_silence",
                 "dormant_x_corp_only", "holding_x_foreign", "foreign_corp_ratio"]
    return df, [], extra_num


CALIBRATION = None  # inner grouped-slice calibration method; None = fit on full train


# ------------------------------------------------------------------------- main

def main():
    df = get_frame()
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    y = df[LABEL].values
    groups = df[GROUP].astype(str).values
    print(f"frame: {len(df)} rows, {y.sum()} positive ({y.mean():.3%})")

    # FROZEN split — do not touch
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(df, y, groups))

    df, extra_cat, extra_num = add_features(df)
    cols_cat = CAT + extra_cat
    cols_num = NUM + extra_num

    Xtr, Xte = df.iloc[tr], df.iloc[te]
    ytr, yte = y[tr], y[te]
    print(f"grouped split: train {len(tr)} / test {len(te)}")

    pipe = build_model(cols_cat, cols_num)
    if CALIBRATION:
        # fit on inner grouped slice, calibrate on held slice (as train_chair.py)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
        fit_i, cal_i = next(gss2.split(Xtr, ytr, groups[tr]))
        pipe.fit(Xtr.iloc[fit_i][cols_cat + cols_num], ytr[fit_i])
        model = CalibratedClassifierCV(FrozenEstimator(pipe), method=CALIBRATION)
        model.fit(Xtr.iloc[cal_i][cols_cat + cols_num], ytr[cal_i])
    else:
        pipe.fit(Xtr[cols_cat + cols_num], ytr)
        model = pipe
    score = model.predict_proba(Xte[cols_cat + cols_num])[:, 1]
    ap = average_precision_score(yte, score)

    os.makedirs(ARTIFACTS, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS, "model.joblib"))
    import json
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump({"pr_auc": round(float(ap), 4), "base_rate": round(float(yte.mean()), 4)}, f)
    with open(os.path.join(ARTIFACTS, "features.json"), "w") as f:
        json.dump({"cat": cols_cat, "num": cols_num}, f)

    finish(round(float(ap), 6))


if __name__ == "__main__":
    main()
