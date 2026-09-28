"""Feature-drift job: PSI of captured production inputs vs the training baseline.

CHANGED vs README_DETAILED version
  * Reads REAL Data Capture files from S3 (JSONL) instead of copying the training set
    and doubling income. --local-csv keeps a no-AWS test path (uses customers_drifted.csv).
  * PSI bins come from the baseline only (baseline.json); the original merged both
    distributions' percentiles, which shrinks PSI.
  * Minimum sample guard (PSI on 20 rows is noise), per-feature CloudWatch metrics,
    region from env instead of hard-coded.
Exit code 2 = drift (handy for CI / schedulers).
Thresholds 0.10 / 0.20 are common industry rules of thumb, not AWS-defined.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

FEATURES = ["age", "income", "previous_clicks", "previous_purchases", "session_duration"]
EPS = 1e-4


def psi(expected_pct, actual_values, edges):
    full = np.concatenate([[-np.inf], edges, [np.inf]])
    counts, _ = np.histogram(actual_values, bins=full)
    a = np.clip(counts / max(len(actual_values), 1), EPS, None)
    e = np.clip(np.asarray(expected_pct), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def read_captures(bucket, prefix="monitoring/data-capture/"):
    import boto3
    s3 = boto3.client("s3")
    rows = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            for line in body.splitlines():
                if not line.strip():
                    continue
                inp = json.loads(line)["captureData"]["endpointInput"]
                if inp.get("observedContentType", "").startswith("text/csv"):
                    for rec in inp["data"].strip().splitlines():
                        rows.append([float(x) for x in rec.split(",")][:5])
    return pd.DataFrame(rows, columns=FEATURES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="monitoring/baseline.json")
    p.add_argument("--local-csv", help="use a CSV instead of S3 captures (offline test)")
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--endpoint-name", default=os.environ.get("ENDPOINT_NAME", "customer-conversion-endpoint"))
    a = p.parse_args()

    baseline = json.load(open(a.baseline))
    prod = pd.read_csv(a.local_csv)[FEATURES] if a.local_csv else read_captures(os.environ["BUCKET"])
    print("Production records:", len(prod))
    if len(prod) < a.min_samples:
        print(f"INSUFFICIENT_DATA (< {a.min_samples})")
        sys.exit(0)

    scores = {f: psi(baseline[f]["expected"], prod[f].values, np.array(baseline[f]["edges"])) for f in FEATURES}
    print(json.dumps(scores, indent=2))
    max_psi = max(scores.values())
    print("Maximum PSI:", round(max_psi, 4))

    if not a.no_metrics:
        import boto3
        cw = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
        data = [{"MetricName": "MaxPSI", "Dimensions": [{"Name": "EndpointName", "Value": a.endpoint_name}],
                 "Value": max_psi, "Unit": "None"}]
        data += [{"MetricName": "PSI", "Dimensions": [{"Name": "EndpointName", "Value": a.endpoint_name},
                                                      {"Name": "Feature", "Value": f}],
                  "Value": v, "Unit": "None"} for f, v in scores.items()]
        cw.put_metric_data(Namespace="CustomerConversion/MLOps", MetricData=data)

    print("DRIFT_DETECTED" if max_psi >= a.threshold else "NO_DRIFT")
    sys.exit(2 if max_psi >= a.threshold else 0)


if __name__ == "__main__":
    main()
