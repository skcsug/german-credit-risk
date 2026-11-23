"""Export the fitted pipeline as plain numbers so a browser can score applicants without Python."""
import math

from credit.features import CAT_COLS, CHK_ORDER, NUM_COLS, PURPOSE_GROUP, SAV_ORDER


def export_model(pipe, X, approve_below: float, review_below: float) -> dict:
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    scaler = pre.named_transformers_["num"]
    onehot = pre.named_transformers_["cat"]
    names = list(pre.get_feature_names_out())
    coef = dict(zip(names, (float(c) for c in model.coef_[0])))
    return {
        "numeric": {col: {"mean": float(m), "scale": float(s)} for col, m, s in zip(NUM_COLS, scaler.mean_, scaler.scale_)},
        "categorical": {col: [str(c) for c in cats] for col, cats in zip(CAT_COLS, onehot.categories_)},
        "coef": coef,
        "intercept": float(model.intercept_[0]),
        "maps": {"saving": SAV_ORDER, "checking": CHK_ORDER, "purpose": PURPOSE_GROUP},
        "policy": {"approve_below": approve_below, "review_below": review_below},
    }


def _features(spec: dict, row: dict) -> dict:
    """Same transformation as credit.features.build_features, for one raw applicant."""
    maps = spec["maps"]
    return {
        "Age": row["Age"],
        "Job": row["Job"],
        "Saving_ord": maps["saving"][row["Saving accounts"]],
        "Saving_none": int(row["Saving accounts"] == "no_account"),
        "Checking_ord": maps["checking"][row["Checking account"]],
        "Checking_none": int(row["Checking account"] == "no_account"),
        "Credit amount": row["Credit amount"],
        "Duration": row["Duration"],
        "log_Monthly_payment": math.log(row["Credit amount"] / row["Duration"]),
        "Housing": "own" if row["Housing"] == "own" else "non_own",
        "Purpose": maps["purpose"][row["Purpose"]],
    }


def score_from_export(spec: dict, row: dict) -> float:
    """P(bad) for one raw applicant using only the exported numbers."""
    f = _features(spec, row)
    z = spec["intercept"]
    for col, p in spec["numeric"].items():
        z += spec["coef"][f"num__{col}"] * (f[col] - p["mean"]) / p["scale"]
    for col, cats in spec["categorical"].items():
        for cat in cats:
            if f[col] == cat:
                z += spec["coef"][f"cat__{col}_{cat}"]
    return 1.0 / (1.0 + math.exp(-z))
