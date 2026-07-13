"""T: train the empty-chair concealment model.

Feature view `empty_chair_fv` = company_registry JOIN psc_shape on company_number
(label `is_revealed` already lives in company_registry). HistGradientBoosting with
balanced weights, calibrated on a grouped holdout.

Every honesty control the design demands runs here and its number is printed and
saved, so a reader can see whether the signal is concealment shape or an artifact:
  - grouped split by office_group: no formation mill straddles train and test.
  - blind rule baseline: flag if silent / corporate-only / foreign-corporate PSC.
  - demographics-only control: year + sic_section + region only. If it matches the
    full model, the signal is population bias, not concealment.
  - shuffle-label control: must collapse to chance.
Headline = PR-AUC and precision@k lift over the blind rule (PU labels => lower bound).

Registers `empty_chair` with eval JSON + PR / calibration / importance PNGs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import hopsworks
from chair_features import CAT_FEATURES as CAT
from chair_features import NUM_FEATURES as NUM

DEMOG = ["incorporation_year", "sic_section", "post_area", "country"]
# the legitimate-structure confounds the bias audit flagged (property/holding SPV shape)
STRUCTURE = ["sic_section", "is_holding_sic", "is_dormant_sic", "is_mill_address",
             "office_company_count"]
LABEL = "is_revealed"
GROUP = "office_group"


def get_training_frame():
    project = hopsworks.login()
    fs = project.get_feature_store()
    reg = fs.get_feature_group("company_registry", version=1)
    psc = fs.get_feature_group("psc_shape", version=1)
    query = reg.select_all().join(psc.select_except(["company_number"]), on=["company_number"])
    fv = fs.get_or_create_feature_view(
        name="empty_chair_fv", version=1, query=query, labels=[LABEL],
        description="Registry + PSC concealment shape for UK companies; label = later-revealed hidden owner",
    )
    df = query.read()  # full frame, includes the label and office_group for grouping
    return project, fv, df


def make_pipeline(cols_cat, cols_num):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cols_cat),
         ("num", "passthrough", cols_num)])
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        class_weight="balanced", random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def precision_at_k(y, score, k):
    order = np.argsort(score)[::-1][:k]
    return float(y[order].mean())


def blind_rule(df):
    return ((df["psc_silence"] == 1) | (df["psc_corporate_only"] == 1) |
            (df["psc_foreign_corporate"] > 0)).astype(int).values


def evaluate(name, y, score):
    ap = average_precision_score(y, score)
    roc = roc_auc_score(y, score)
    base = float(y.mean())
    out = {
        "model": name, "pr_auc": round(ap, 4), "roc_auc": round(roc, 4),
        "base_rate": round(base, 4), "pr_auc_lift": round(ap / base, 2),
        "precision_at_100": round(precision_at_k(y, score, 100), 4),
        "precision_at_1000": round(precision_at_k(y, score, 1000), 4),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-structure", action="store_true",
                    help="ablation: drop the property/holding/mill structure confounds")
    args = ap.parse_args()

    global CAT, NUM
    if args.no_structure:
        CAT = [c for c in CAT if c not in STRUCTURE]
        NUM = [c for c in NUM if c not in STRUCTURE]
        print(f"ABLATION: dropped structure confounds; {len(CAT+NUM)} features remain")

    project, fv, df = get_training_frame()
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    y = df[LABEL].values
    groups = df[GROUP].astype(str).values
    print(f"frame: {len(df)} rows, {y.sum()} positive ({y.mean():.3%})")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(df, y, groups))
    Xtr, Xte = df.iloc[tr], df.iloc[te]
    ytr, yte = y[tr], y[te]
    print(f"grouped split: train {len(tr)} / test {len(te)}; "
          f"train groups {len(set(groups[tr]))}, test groups {len(set(groups[te]))}, "
          f"overlap {len(set(groups[tr]) & set(groups[te]))}")

    results = {}

    # --- blind rule baseline
    results["blind_rule"] = evaluate("blind_rule", yte, blind_rule(Xte).astype(float))

    # --- full model: fit on an inner grouped slice, calibrate on the held slice
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    fit_i, cal_i = next(gss2.split(Xtr, ytr, groups[tr]))
    pipe = make_pipeline(CAT, NUM)
    pipe.fit(Xtr.iloc[fit_i][CAT + NUM], ytr[fit_i])
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(Xtr.iloc[cal_i][CAT + NUM], ytr[cal_i])
    score_full = cal.predict_proba(Xte[CAT + NUM])[:, 1]
    results["full"] = evaluate("full", yte, score_full)
    results["full"]["brier"] = round(brier_score_loss(yte, score_full), 4)

    # --- demographics-only control
    pdem = make_pipeline(["sic_section", "post_area", "country"], ["incorporation_year"])
    pdem.fit(Xtr[DEMOG], ytr)
    score_dem = pdem.predict_proba(Xte[DEMOG])[:, 1]
    results["demographics_only"] = evaluate("demographics_only", yte, score_dem)

    # --- shuffle-label control
    rng = np.random.RandomState(0)
    yshuf = rng.permutation(ytr)
    pshuf = make_pipeline(CAT, NUM)
    pshuf.fit(Xtr[CAT + NUM], yshuf)
    score_shuf = pshuf.predict_proba(Xte[CAT + NUM])[:, 1]
    results["shuffle_label"] = evaluate("shuffle_label", yte, score_shuf)

    print("\n=== RESULTS (grouped holdout) ===")
    for k, v in results.items():
        print(f"{k:20s} PR-AUC {v['pr_auc']:.3f} (lift {v['pr_auc_lift']:.1f})  "
              f"ROC {v['roc_auc']:.3f}  P@100 {v['precision_at_100']:.3f}  P@1000 {v['precision_at_1000']:.3f}")

    if args.no_register:
        return

    register(project, cal, pipe, results, Xte, yte, score_full)


def register(project, cal, pipe, results, Xte, yte, score_full):
    tmp = tempfile.mkdtemp()
    # PR curve
    prec, rec, _ = precision_recall_curve(yte, score_full)
    plt.figure(figsize=(5, 4)); plt.plot(rec, prec)
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title(f"PR (AP={results['full']['pr_auc']}, base={results['full']['base_rate']})")
    plt.tight_layout(); plt.savefig(f"{tmp}/pr_curve.png", dpi=120); plt.close()
    # calibration
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(yte, score_full, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 4)); plt.plot([0, 1], [0, 1], "--", c="gray"); plt.plot(mean, frac, "o-")
    plt.xlabel("predicted"); plt.ylabel("observed"); plt.title("calibration")
    plt.tight_layout(); plt.savefig(f"{tmp}/calibration.png", dpi=120); plt.close()
    # permutation importance on the fitted pipeline
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(pipe, Xte[CAT + NUM], yte, n_repeats=5,
                                 random_state=0, scoring="average_precision", n_jobs=-1)
    names = CAT + NUM
    order = np.argsort(imp.importances_mean)[::-1][:15]
    plt.figure(figsize=(6, 5)); plt.barh([names[i] for i in order][::-1],
                                          imp.importances_mean[order][::-1])
    plt.title("permutation importance (AP)"); plt.tight_layout()
    plt.savefig(f"{tmp}/importance.png", dpi=120); plt.close()

    with open(f"{tmp}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    # bundle the feature contract WITH the model so serving cannot drift from training
    with open(f"{tmp}/features.json", "w") as f:
        json.dump({"cat": CAT, "num": NUM, "features": CAT + NUM}, f)
    import joblib
    joblib.dump(cal, f"{tmp}/model.joblib")

    mr = project.get_model_registry()
    model = mr.python.create_model(
        name="empty_chair",
        metrics={"pr_auc": results["full"]["pr_auc"],
                 "pr_auc_lift": results["full"]["pr_auc_lift"],
                 "precision_at_100": results["full"]["precision_at_100"],
                 "roc_auc": results["full"]["roc_auc"]},
        description="Concealment-shape model: is a UK company's beneficial-ownership disclosure evasive (PU-labelled, lower bound)",
    )
    model.save(tmp)
    print(f"\nregistered empty_chair v{model.version}")


if __name__ == "__main__":
    main()
