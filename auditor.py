"""I: the scoring module, shared by the batch job and the app.

Loads the pinned `empty_chair` model once, scores a feature dict, and lists the
concealment tells that fired. Signal, not verdict: the score is the model's
P(disclosure is evasive); the fired flags are the evidence, never a claim that a
crime occurred. Every consumer imports MODEL_VERSION so a bump is one edit.
"""

from __future__ import annotations

import functools
import json

import os

import pandas as pd

import hopsworks
from chair_features import (CONCEALMENT_FLAGS, derive_features, primary_sic,
                            psc_features, registry_features)

MODEL_NAME = "empty_chair"
MODEL_VERSION = 6  # single-seed LGBM, TE without sic_code (v5 top 1% was 99.3% real estate, docs/bias-audit.md)


@functools.lru_cache(maxsize=1)
def _load():
    d = hopsworks.login().get_model_registry().get_model(MODEL_NAME, version=MODEL_VERSION).download()
    import joblib
    with open(f"{d}/features.json") as f:
        feats = json.load(f)["features"]
    te_maps = None
    if os.path.exists(f"{d}/te_maps.json"):
        with open(f"{d}/te_maps.json") as f:
            te_maps = json.load(f)
    return joblib.load(f"{d}/model.joblib"), feats, te_maps


def load_model():
    return _load()[0]


def model_features():
    """The feature columns the pinned model was trained on, carried in its artifact
    so serving cannot drift from training."""
    return _load()[1]


def features_for(registry_row: dict, psc_records: list[dict], mill_count: int) -> dict:
    """Assemble the full feature dict for one company from its raw CH inputs."""
    feats = registry_features(registry_row, mill_count)
    feats.update(psc_features(psc_records))
    return feats


def score_one(feats: dict, model=None) -> float:
    model = model or load_model()
    _, cols, te_maps = _load()
    X = derive_features(pd.DataFrame([feats]), te_maps)
    return float(model.predict_proba(X[cols])[:, 1][0])


def score_frame(df: pd.DataFrame, model=None) -> "pd.Series":
    model = model or load_model()
    _, cols, te_maps = _load()
    df = derive_features(df, te_maps)
    return pd.Series(model.predict_proba(df[cols])[:, 1], index=df.index)


def fired_flags(feats: dict) -> list[dict]:
    """The concealment tells present in this company, as evidence rows."""
    out = []
    for key, label in CONCEALMENT_FLAGS.items():
        v = feats.get(key, 0)
        if v:
            out.append({"flag": key, "label": label, "value": int(v) if v is True or v in (0, 1) else v})
    return out
