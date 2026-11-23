import json
import math

import numpy as np

from credit.data import load_german_credit
from credit.features import build_features, make_pipeline
from credit.export import export_model, score_from_export


def test_uci_mapping_matches_known_category_counts():
    df = load_german_credit()
    assert len(df) == 1000
    assert (df["Risk"] == "bad").sum() == 300
    assert df["Saving accounts"].value_counts(dropna=False).to_dict() == {
        "little": 603, "no_account": 183, "moderate": 103, "quite rich": 63, "rich": 48}
    assert df["Checking account"].value_counts(dropna=False).to_dict() == {
        "no_account": 394, "little": 274, "moderate": 269, "rich": 63}
    assert df["Purpose"].value_counts().to_dict() == {
        "car": 337, "radio/TV": 280, "furniture/equipment": 181, "business": 97,
        "education": 59, "repairs": 22, "domestic appliances": 12, "vacation/others": 12}
    assert sorted(df["Job"].unique()) == [0, 1, 2, 3]
    assert set(df["Housing"]) == {"own", "rent", "free"}


def test_features_follow_final_submission_options():
    X = build_features(load_german_credit())
    assert list(X.columns) == [
        "Age", "Job", "Saving_ord", "Saving_none", "Checking_ord", "Checking_none",
        "Credit amount", "Duration", "log_Monthly_payment", "Housing", "Purpose"]
    assert set(X["Housing"]) == {"own", "non_own"}
    assert set(X["Purpose"]) == {"Asset purchase", "Living consumption", "Future investment", "Others"}
    assert "Sex" not in X.columns


def test_exported_model_reproduces_sklearn_probabilities():
    df = load_german_credit()
    X, y = build_features(df), (df["Risk"] == "bad").astype(int)
    pipe = make_pipeline().fit(X, y)
    spec = export_model(pipe, X, approve_below=0.3, review_below=0.5)
    json.dumps(spec)  # must be JSON-serialisable for the web demo
    expected = pipe.predict_proba(X)[:, 1]
    got = np.array([score_from_export(spec, row) for row in df.to_dict("records")])
    assert np.allclose(got, expected, atol=1e-9)
    assert all(math.isfinite(v) for v in spec["coef"].values())
