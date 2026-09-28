"""Send test/production-like traffic to the endpoint (fills Data Capture, feeds drift job).

  python deployment/invoke_traffic.py --n 200            # in-distribution
  python deployment/invoke_traffic.py --n 200 --drift    # shifted distribution
"""
import argparse
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_data import generate  # noqa: E402

CAMPAIGN = ["email", "search", "social"]
DEVICE = ["desktop", "mobile", "tablet"]


def to_csv_row(r):
    vals = [r.age, r.income, r.previous_clicks, r.previous_purchases, round(r.session_duration, 2)]
    vals += [int(r.campaign_type == c) for c in CAMPAIGN] + [int(r.device == d) for d in DEVICE]
    return ",".join(str(v) for v in vals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--endpoint-name", default=os.environ.get("ENDPOINT_NAME", "customer-conversion-endpoint"))
    a = p.parse_args()

    rt = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
    df = generate(a.n, seed=None, drift=a.drift)
    for i, r in enumerate(df.itertuples(index=False), 1):
        resp = rt.invoke_endpoint(EndpointName=a.endpoint_name, ContentType="text/csv", Body=to_csv_row(r))
        if i % 25 == 0 or i == 1:
            print(i, resp["Body"].read().decode())
