"""Build the PSI baseline: decile edges + expected proportions per numeric feature.

CHANGED vs README_DETAILED version: it stored mean/std/min/max, which cannot compute
PSI. PSI needs the baseline *binning*. Bin edges come from the baseline only, with
open-ended outer bins so out-of-range production values are counted, not dropped.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURES = ["age", "income", "previous_clicks", "previous_purchases", "session_duration"]


def build(df, bins=10):
    out = {}
    for f in FEATURES:
        edges = np.unique(np.percentile(df[f], np.linspace(0, 100, bins + 1)[1:-1]))
        full = np.concatenate([[-np.inf], edges, [np.inf]])
        counts, _ = np.histogram(df[f], bins=full)
        out[f] = {"edges": edges.tolist(), "expected": (counts / len(df)).tolist()}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/output/customers.csv")
    p.add_argument("--output", default="monitoring/baseline.json")
    p.add_argument("--upload", action="store_true", help="also upload to s3://$BUCKET/monitoring/baseline.json")
    a = p.parse_args()

    baseline = build(pd.read_csv(a.input))
    with open(a.output, "w") as fh:
        json.dump(baseline, fh, indent=2)
    print("Wrote", a.output)
    if a.upload:
        import boto3
        boto3.client("s3").upload_file(a.output, os.environ["BUCKET"], "monitoring/baseline.json")
        print("Uploaded to s3://%s/monitoring/baseline.json" % os.environ["BUCKET"])
