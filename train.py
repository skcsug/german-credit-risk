"""Reproduce the final model on public UCI data and export it for the portfolio demo.

    python train.py      # prints CV F1 and the approve/review/reject policy, writes model/*.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit.data import load_german_credit
from credit.export import export_model
from credit.features import build_features, make_pipeline

COST_APPROVE_BAD = 5  # approving a future defaulter (UCI cost matrix)
COST_REJECT_GOOD = 1  # turning away a good customer
SEEDS = (42, 2026, 7)  # same three splits as the project


def main():
    df = load_german_credit()
    X, y = build_features(df), (df["Risk"] == "bad").astype(int).to_numpy()

    oof = []
    for seed in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof.append(cross_val_predict(make_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1])
    f1s = [f1_score(y, (p >= 0.5).astype(int)) for p in oof]
    p_bad = np.mean(oof, axis=0)

    # Approve when P(bad) < t. Pick t that minimises 5 × (approved bads) + 1 × (rejected goods).
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [COST_APPROVE_BAD * np.sum((p_bad < t) & (y == 1)) + COST_REJECT_GOOD * np.sum((p_bad >= t) & (y == 0)) for t in grid]
    approve_below = float(grid[int(np.argmin(costs))])
    review_below = max(0.5, approve_below)  # above the model's own bad threshold → reject

    band = np.where(p_bad < approve_below, "approve", np.where(p_bad < review_below, "review", "reject"))
    bands = {b: {"n": int((band == b).sum()), "bad_rate": round(float(y[band == b].mean()), 3)} for b in ("approve", "review", "reject")}

    pipe = make_pipeline().fit(X, y)
    spec = export_model(pipe, X, approve_below=approve_below, review_below=review_below)
    metrics = {
        "rows": int(len(y)), "bad_rate": round(float(y.mean()), 3),
        "cv_f1_per_seed": [round(f, 4) for f in f1s], "cv_f1_mean": round(float(np.mean(f1s)), 4),
        "cost_min": int(min(costs)), "approve_below": approve_below, "review_below": review_below, "bands": bands,
        "odds_ratio_per_sd": {k: round(float(np.exp(v)), 2) for k, v in spec["coef"].items()},
    }
    out = Path("model")
    out.mkdir(exist_ok=True)
    (out / "lr_model.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
