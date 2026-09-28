"""SageMaker Training script (runs in the built-in sklearn container).

CHANGED vs original
  * Real hyperparameters (n_estimators, max_depth, min_samples_leaf) instead of
    passing model-dir as a 'hyperparameter' - SageMaker already exposes /opt/ml/model.
  * Saves feature_columns.json next to model.joblib so inference.py uses the exact
    training column order (removes the hard-coded list duplicated in inference.py).
  * Test channel optional; class_weight='balanced_subsample' for the ~15% positive rate.
  * Metrics printed as key=value (visible in CloudWatch Logs) and written to
    /opt/ml/output/data/train_metrics.json (returned in the training job output).
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "converted"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    a = p.parse_args()

    train = pd.read_csv(os.path.join(a.train, "train.csv"))
    features = [c for c in train.columns if c != TARGET]

    model = RandomForestClassifier(
        n_estimators=a.n_estimators, max_depth=a.max_depth,
        min_samples_leaf=a.min_samples_leaf, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    ).fit(train[features], train[TARGET])

    metrics = {}
    test_file = os.path.join(a.test, "test.csv")
    if os.path.exists(test_file):
        test = pd.read_csv(test_file)
        proba = model.predict_proba(test[features])[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(test[TARGET], pred),
            "f1": f1_score(test[TARGET], pred),
            "auc": roc_auc_score(test[TARGET], proba),
        }
        for k, v in metrics.items():
            print(f"{k}={v:.4f}")

    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(a.model_dir, "model.joblib"))
    with open(os.path.join(a.model_dir, "feature_columns.json"), "w") as f:
        json.dump(features, f)

    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
