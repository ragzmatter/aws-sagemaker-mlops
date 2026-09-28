"""Synthetic customer-conversion data.

CHANGED vs original
  * argparse (rows/seed/output/--drift) instead of hard-coded constants.
  * default_rng instead of the legacy global np.random.seed (reproducible, no global state).
  * click / purchase coefficients raised so the signal is learnable; the original
    0.015 per click was ~noise, which would make an AUC quality gate meaningless.
  * --drift produces a deliberately shifted 'production-like' dataset that the
    drift lab (monitoring/detect_drift.py) can be tested against.
"""
import argparse
import os

import numpy as np
import pandas as pd


def generate(n: int, seed: int, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.integers(20_000, 200_000, n),
        "previous_clicks": rng.poisson(3, n),
        "previous_purchases": rng.poisson(1, n),
        "session_duration": rng.exponential(180, n),
        "campaign_type": rng.choice(["search", "social", "email"], n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
    })

    if drift:  # covariate shift: richer, longer-session, more-clicking population
        df["income"] = (df["income"] * 1.8).clip(upper=500_000).astype(int)
        df["session_duration"] = df["session_duration"] * 1.5
        df["previous_clicks"] = rng.poisson(5, n)

    score = (
        -3.6
        + 0.08 * df["previous_clicks"]
        + 0.35 * df["previous_purchases"]
        + 0.003 * df["session_duration"]
        + 0.000004 * df["income"]
        + (df["campaign_type"] == "email") * 0.5
        + (df["device"] == "desktop") * 0.2
    )
    probability = 1 / (1 + np.exp(-score))
    df["converted"] = (rng.random(n) < probability).astype(int)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    out = a.output or ("data/output/customers_drifted.csv" if a.drift else "data/output/customers.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = generate(a.rows, a.seed, a.drift)
    data.to_csv(out, index=False)
    print(data.head(), "\nShape:", data.shape, "\nConversion rate:", round(data["converted"].mean(), 4), "\nWrote:", out)
