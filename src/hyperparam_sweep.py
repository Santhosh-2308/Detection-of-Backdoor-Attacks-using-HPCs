"""
Hyperparameter sweep: for each (contamination, nu, target_fpr) combination,
retrain the ensemble and measure:
  - CV-achieved FPR (out-of-sample, on benign-only data)
  - detection rate on a fixed synthetic perturbation set (10% uniform scale
    and 5% single-counter spike), as a proxy for sensitivity

This makes the FPR/sensitivity tradeoff explicit rather than asserting a
single "best" configuration without justification.
"""
import warnings
import itertools
import numpy as np
import pandas as pd

from train import fit_ensemble, ensemble_score, cross_validated_threshold
from features import build_feature_matrix, RAW_COLS
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def make_fixed_synthetic(df, seed=1):
    base = df.sample(100, random_state=seed).reset_index(drop=True)
    v10 = base.copy()
    v10["cache-references"] *= 1.10
    v10["LLC-loads"] *= 1.15
    v10["cycles"] *= 1.05

    v5spike = base.copy()
    v5spike["LLC-loads"] *= 1.05
    return {"uniform_10pct": v10, "llc_spike_5pct": v5spike}


def main():
    df = pd.read_csv("../data/trace.csv")
    feat_df = build_feature_matrix(df)[RAW_COLS]  # match train.py's raw-only default
    feature_names = list(feat_df.columns)
    scaler = StandardScaler().fit(feat_df.values)
    X = scaler.transform(feat_df.values)

    synth = make_fixed_synthetic(df)
    synth_X = {
        name: scaler.transform(build_feature_matrix(v)[feature_names].values)
        for name, v in synth.items()
    }

    contamination_grid = [0.01, 0.03, 0.05, 0.10]
    nu_grid = [0.01, 0.03, 0.05, 0.10]
    target_fpr_grid = [0.01, 0.02, 0.05]

    rows = []
    for contamination, nu, target_fpr in itertools.product(
        contamination_grid, nu_grid, target_fpr_grid
    ):
        threshold, achieved_fpr, _ = cross_validated_threshold(X, target_fpr, n_splits=5)
        models, ref_scores = fit_ensemble(X, contamination=contamination, nu=nu)

        row = {
            "contamination": contamination,
            "nu": nu,
            "target_fpr": target_fpr,
            "cv_achieved_fpr": round(achieved_fpr, 4),
            "threshold": round(threshold, 4),
        }
        for name, sx in synth_X.items():
            s = ensemble_score(models, ref_scores, sx)
            row[f"detect_{name}"] = round(float((s >= threshold).mean()), 3)
        rows.append(row)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv("../figures/hyperparam_sweep_results.csv", index=False)
    pd.set_option("display.width", 160)
    print(grid_df.to_string(index=False))

    # Highlight a few interesting points: best FPR-calibration at target 1%,
    # and how detection rate on the two synthetic sets varies with contamination/nu.
    print("\n--- summary: mean detection rate by contamination (target_fpr=0.01) ---")
    sub = grid_df[grid_df.target_fpr == 0.01]
    print(sub.groupby("contamination")[["detect_uniform_10pct", "detect_llc_spike_5pct"]].mean())

    plot_tradeoff(grid_df)


def plot_tradeoff(grid_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = grid_df.groupby("target_fpr")[
        ["cv_achieved_fpr", "detect_uniform_10pct", "detect_llc_spike_5pct"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["target_fpr"] * 100, summary["detect_uniform_10pct"] * 100,
            marker="o", label="uniform_10pct (strong signal)")
    ax.plot(summary["target_fpr"] * 100, summary["detect_llc_spike_5pct"] * 100,
            marker="s", label="llc_spike_5pct (weak signal)")
    ax.set_xlabel("Target FPR (%)")
    ax.set_ylabel("% flagged as anomalous")
    ax.set_title("FPR/sensitivity tradeoff as target_fpr is relaxed")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/fpr_sensitivity_tradeoff.png", dpi=150)
    print("saved ../figures/fpr_sensitivity_tradeoff.png")


if __name__ == "__main__":
    main()
