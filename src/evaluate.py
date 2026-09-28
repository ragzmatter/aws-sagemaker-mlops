"""SageMaker Processing step: evaluate the trained model on the held-out test set.

CHANGED vs original
  * BUG FIX: the model input is model.tar.gz (a tarball); original joblib.load()ed a
    directory path that doesn't exist. Now extracts the tarball first.
  * Report uses the 'binary_classification_metrics' schema - the schema the Model
    Registry UI understands (Model quality tab) - and the pipeline's ConditionStep
    reads binary_classification_metrics.auc.value through a PropertyFile/JsonGet.
"""
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUT_DIR = "/opt/ml/processing/evaluation"
TARGET = "converted"


def main():
    with tarfile.open(os.path.join(MODEL_DIR, "model.tar.gz")) as tar:
        tar.extractall(path=MODEL_DIR)

    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    features = json.load(open(os.path.join(MODEL_DIR, "feature_columns.json")))

    test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = test[TARGET]

    def m(v):
        return {"value": float(v), "standard_deviation": "NaN"}

    report = {"binary_classification_metrics": {
        "accuracy": m(accuracy_score(y, pred)),
        "precision": m(precision_score(y, pred, zero_division=0)),
        "recall": m(recall_score(y, pred)),
        "f1": m(f1_score(y, pred)),
        "auc": m(roc_auc_score(y, proba)),
    }}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
