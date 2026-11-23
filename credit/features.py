"""Features and model of the final (11th) submission.

Options chosen in the project: Housing → own / non_own, Saving & Checking → ordinal + "no
account" flag, Credit amount & Duration raw, log(monthly payment) added, Purpose → 4 groups,
Sex excluded (fairness; F1 unchanged with it).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SAV_ORDER = {"no_account": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
CHK_ORDER = {"no_account": 0, "little": 1, "moderate": 2, "rich": 3}
PURPOSE_GROUP = {
    "car": "Asset purchase", "furniture/equipment": "Asset purchase",
    "radio/TV": "Living consumption", "domestic appliances": "Living consumption", "repairs": "Living consumption",
    "business": "Future investment", "education": "Future investment",
    "vacation/others": "Others",
}
NUM_COLS = ["Age", "Job", "Saving_ord", "Saving_none", "Checking_ord", "Checking_none",
            "Credit amount", "Duration", "log_Monthly_payment"]
CAT_COLS = ["Housing", "Purpose"]
BAD_WEIGHT = 1.8  # searched 1.2–2.333 in the project; 1.8 gave the best real F1 (0.6667)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Age": df["Age"],
        "Job": df["Job"],
        "Saving_ord": df["Saving accounts"].map(SAV_ORDER),
        "Saving_none": (df["Saving accounts"] == "no_account").astype(int),
        "Checking_ord": df["Checking account"].map(CHK_ORDER),
        "Checking_none": (df["Checking account"] == "no_account").astype(int),
        "Credit amount": df["Credit amount"],
        "Duration": df["Duration"],
        "log_Monthly_payment": np.log(df["Credit amount"] / df["Duration"]),
        "Housing": np.where(df["Housing"] == "own", "own", "non_own"),
        "Purpose": df["Purpose"].map(PURPOSE_GROUP),
    })


def make_pipeline(bad_weight: float = BAD_WEIGHT, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = LogisticRegression(class_weight={0: 1.0, 1: bad_weight}, max_iter=3000, random_state=seed)
    return Pipeline([("pre", pre), ("model", model)])
