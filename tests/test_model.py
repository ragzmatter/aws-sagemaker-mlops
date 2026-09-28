"""Fast, offline tests (no AWS). Run: pytest -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "monitoring"))

import inference  # noqa: E402
from detect_drift import psi  # noqa: E402
from generate_data import generate  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("lab")
    (d / "raw").mkdir()
    generate(4000, 1).to_csv(d / "raw" / "customers.csv", index=False)
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=ROOT, capture_output=True, text=True)
    run("src/preprocess.py", "--input-data", str(d / "raw"), "--output-data", str(d / "proc"))
    run("src/train.py", "--train", str(d / "proc/train"), "--test", str(d / "proc/test"),
        "--model-dir", str(d / "model"), "--output-dir", str(d / "out"), "--n-estimators", "20")
    return d


def test_feature_contract(trained):
    cols = json.load(open(trained / "model" / "feature_columns.json"))
    assert len(cols) == 11 and "converted" not in cols
    assert cols[5:8] == ["campaign_type_email", "campaign_type_search", "campaign_type_social"]


def test_split_is_disjoint_and_stratified(trained):
    tr, te = pd.read_csv(trained / "proc/train/train.csv"), pd.read_csv(trained / "proc/test/test.csv")
    assert abs(tr.converted.mean() - te.converted.mean()) < 0.02
    assert len(tr) + len(te) <= 4000 and len(te) > 0


def test_inference_csv_multirow(trained):
    art = inference.model_fn(str(trained / "model"))
    body = "35,85000,5,2,240,1,0,0,1,0,0\n50,40000,1,0,60,0,1,0,0,1,0"
    out = json.loads(inference.output_fn(inference.predict_fn(inference.input_fn(body, "text/csv"), art), "application/json")[0])
    assert len(out["predictions"]) == 2 and all(0 <= p <= 1 for p in out["probabilities"])


def test_inference_rejects_wrong_width(trained):
    art = inference.model_fn(str(trained / "model"))
    with pytest.raises(ValueError):
        inference.predict_fn(inference.input_fn("1,2,3", "text/csv"), art)


def test_psi_zero_for_identical_and_high_for_shift():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    edges = np.percentile(x, np.linspace(0, 100, 11)[1:-1])
    expected = np.histogram(x, bins=np.concatenate([[-np.inf], edges, [np.inf]]))[0] / len(x)
    assert psi(expected, x, edges) < 0.01
    assert psi(expected, x + 2, edges) > 0.2
