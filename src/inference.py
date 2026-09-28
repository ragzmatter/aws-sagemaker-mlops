"""SageMaker sklearn-container serving handlers.

CHANGED vs original
  * Column order comes from feature_columns.json (written by train.py) - no duplicated list.
  * Supports multi-row CSV (one record per line) and a JSON list; original crashed on
    more than one row. Response is always {"predictions": [...], "probabilities": [...]}.
  * JSON input is re-ordered to the training columns (a dict in a different key order
    would otherwise silently mis-feed the model).
  * Threshold configurable via env var CONVERSION_THRESHOLD (default 0.5).
  * Removed unused `io` import.
"""
import json
import os

import joblib
import pandas as pd

THRESHOLD = float(os.environ.get("CONVERSION_THRESHOLD", "0.5"))


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        columns = json.load(f)
    return {"model": model, "columns": columns}


def input_fn(request_body, request_content_type):
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    if request_content_type == "text/csv":
        rows = [[float(x) for x in line.split(",")] for line in request_body.strip().splitlines() if line.strip()]
        return pd.DataFrame(rows)  # positional; columns applied in predict_fn

    if request_content_type == "application/json":
        payload = json.loads(request_body)
        records = payload if isinstance(payload, list) else [payload]
        return pd.DataFrame(records)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, artifacts):
    columns = artifacts["columns"]
    if all(isinstance(c, int) for c in input_data.columns):      # CSV path
        if input_data.shape[1] != len(columns):
            raise ValueError(f"Expected {len(columns)} values per row, got {input_data.shape[1]}")
        input_data.columns = columns
    else:                                                        # JSON path
        input_data = input_data[columns]

    proba = artifacts["model"].predict_proba(input_data)[:, 1]
    return {
        "predictions": [int(p >= THRESHOLD) for p in proba],
        "probabilities": [float(p) for p in proba],
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
