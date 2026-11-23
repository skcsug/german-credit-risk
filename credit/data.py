"""UCI Statlog (German Credit) → the 10-column schema used in the course.

The course handed out an 800-row slice of the same dataset. Here we rebuild the full
1,000 rows from the public UCI file so anyone can reproduce the pipeline.
Source: https://archive.ics.uci.edu/dataset/144 (CC BY 4.0).
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
CACHE = Path(__file__).resolve().parent.parent / "data" / "german.data"

CHECKING = {"A11": "little", "A12": "moderate", "A13": "rich", "A14": "no_account"}
SAVING = {"A61": "little", "A62": "moderate", "A63": "quite rich", "A64": "rich", "A65": "no_account"}
JOB = {"A171": 0, "A172": 1, "A173": 2, "A174": 3}
HOUSING = {"A151": "rent", "A152": "own", "A153": "free"}
SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
# A47 (vacation) never occurs; retraining (A48) is grouped with education as in the course data.
PURPOSE = {
    "A40": "car", "A41": "car", "A42": "furniture/equipment", "A43": "radio/TV",
    "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation/others",
    "A48": "education", "A49": "business", "A410": "vacation/others",
}


def load_german_credit(path: Path = CACHE) -> pd.DataFrame:
    """1,000 applicants with the course's 10 columns plus Risk ('good' / 'bad')."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(UCI_URL, path)
    raw = pd.read_csv(path, sep=" ", header=None, names=[f"a{i}" for i in range(1, 22)])
    return pd.DataFrame({
        "Age": raw.a13,
        "Sex": raw.a9.map(SEX),
        "Job": raw.a17.map(JOB),
        "Housing": raw.a15.map(HOUSING),
        "Saving accounts": raw.a6.map(SAVING),
        "Checking account": raw.a1.map(CHECKING),
        "Credit amount": raw.a5,
        "Duration": raw.a2,
        "Purpose": raw.a4.map(PURPOSE),
        "Risk": raw.a21.map({1: "good", 2: "bad"}),
    })
