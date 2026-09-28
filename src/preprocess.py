"""SageMaker Processing step: validate, one-hot encode, split.

CHANGED vs original
  * Deleted the ~60-line commented-out old copy at the bottom of the file.
  * Categories are FIXED (pd.Categorical). get_dummies on whatever categories happen
    to appear can silently change the column set/order between runs -> the model
    and inference.py would disagree. Now the feature contract is deterministic.
  * Target is written as the LAST column, features in a fixed order.
  * Fail-fast schema validation (missing columns, unknown categories, bad ranges).
  * Not importing a shared module: the pipeline uploads only this one file to the
    Processing container, so the contract lives here and train.py derives the
    feature order from train.csv (single source of truth = this file's output).
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_CHANNEL = "train"   # must equal ProcessingOutput output_name / training channel in pipeline.py
TEST_CHANNEL = "test"

NUMERIC = ["age", "income", "previous_clicks", "previous_purchases", "session_duration"]
CATEGORIES = {
    "campaign_type": ["email", "search", "social"],
    "device": ["desktop", "mobile", "tablet"],
}
TARGET = "converted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", default="/opt/ml/processing/input")
    parser.add_argument("--output-data", default="/opt/ml/processing/output")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.input_data, "customers.csv"))
    print("Input shape:", df.shape)

    required = NUMERIC + list(CATEGORIES) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    before = len(df)
    df = df[required].dropna().drop_duplicates()
    print(f"Dropped {before - len(df)} null/duplicate rows")

    for col, cats in CATEGORIES.items():
        unknown = set(df[col].unique()) - set(cats)
        if unknown:
            raise ValueError(f"Unknown values in {col}: {unknown}")
        df[col] = pd.Categorical(df[col], categories=cats)

    if (df["age"] < 0).any() or (df["income"] < 0).any():
        raise ValueError("Negative age/income found")

    df = pd.get_dummies(df, columns=list(CATEGORIES), dtype=int)
    features = NUMERIC + [f"{c}_{v}" for c, vs in CATEGORIES.items() for v in vs]
    df = df[features + [TARGET]]

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])

    for name, part in ((TRAIN_CHANNEL, train), (TEST_CHANNEL, test)):
        d = os.path.join(args.output_data, name)
        os.makedirs(d, exist_ok=True)
        part.to_csv(os.path.join(d, f"{name}.csv"), index=False)
        print(name, part.shape, "->", d)


if __name__ == "__main__":
    main()
