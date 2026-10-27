"""
Feature engineering for HPC-trace backdoor detection.

Raw counters (cache-references, cycles, LLC-loads) are dominated by
overall workload size/duration. Backdoor triggers are expected to
perturb the *shape* of computation (extra branches / cache probing
needed to recognize the trigger pattern) more reliably than they
perturb absolute counter magnitude. We therefore add scale-invariant
ratio features on top of the raw (standardized) counters so the
detector can pick up on both magnitude and shape anomalies.
"""

import numpy as np
import pandas as pd

RAW_COLS = ["cache-references", "cycles", "LLC-loads"]

EPS = 1.0  # avoids div-by-zero; counters are >>1 so this has no practical effect


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cycles_per_cacheref"] = df["cycles"] / (df["cache-references"] + EPS)
    df["llc_per_cacheref"] = df["LLC-loads"] / (df["cache-references"] + EPS)
    df["llc_per_cycle"] = df["LLC-loads"] / (df["cycles"] + EPS)
    return df


FEATURE_COLS = RAW_COLS + ["cycles_per_cacheref", "llc_per_cacheref", "llc_per_cycle"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw dataframe with RAW_COLS and returns the full feature matrix."""
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input trace file is missing required columns: {missing}")
    df = add_derived_features(df[RAW_COLS])
    return df[FEATURE_COLS]
