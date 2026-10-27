"""
Ablation: does adding the 3 ratio features (cycles_per_cacheref,
llc_per_cacheref, llc_per_cycle) actually help, or is it dead weight?

Compares CV-achieved FPR calibration quality and synthetic-perturbation
detection rate with vs without the ratio features.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from train import fit_ensemble, ensemble_score, cross_validated_threshold
from features import build_feature_matrix, RAW_COLS, FEATURE_COLS

warnings.filterwarnings("ignore")


def make_fixed_synthetic(df, seed=1):
    base = df.sample(100, random_state=seed).reset_index(drop=True)
    variants = {}
    v10 = base.copy()
    v10["cache-references"] *= 1.10
    v10["LLC-loads"] *= 1.15
    v10["cycles"] *= 1.05
    variants["uniform_10pct"] = v10

    v5spike = base.copy()
    v5spike["LLC-loads"] *= 1.05
    variants["llc_spike_5pct"] = v5spike

    vratio = base.copy()
    vratio["LLC-loads"] *= 1.10
    vratio["cache-references"] *= 0.97
    variants["ratio_shift_10pct"] = vratio
    return variants


def run_config(df, feature_cols, target_fpr=0.01):
    feat_df = build_feature_matrix(df)[feature_cols]
    scaler = StandardScaler().fit(feat_df.values)
    X = scaler.transform(feat_df.values)

    threshold, achieved_fpr, _ = cross_validated_threshold(X, target_fpr, n_splits=5)
    models, ref_scores = fit_ensemble(X)

    synth = make_fixed_synthetic(df)
    detect_rates = {}
    for name, v in synth.items():
        sx = scaler.transform(build_feature_matrix(v)[feature_cols].values)
        s = ensemble_score(models, ref_scores, sx)
        detect_rates[name] = float((s >= threshold).mean())

    return {"achieved_fpr": achieved_fpr, "threshold": threshold, **detect_rates}


def main():
    df = pd.read_csv("../data/trace.csv")

    raw_only = run_config(df, RAW_COLS)
    with_ratios = run_config(df, FEATURE_COLS)

    result = pd.DataFrame([
        {"config": "raw_3_features_only", **raw_only},
        {"config": "raw_plus_3_ratio_features", **with_ratios},
    ])
    result.to_csv("../figures/feature_ablation_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
