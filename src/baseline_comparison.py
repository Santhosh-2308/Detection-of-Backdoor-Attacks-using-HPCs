"""
Naive baseline comparison: does the ensemble actually beat the simplest
possible anomaly detector, or is the added complexity unjustified?

Baseline: per-feature z-score threshold. A trace is flagged anomalous if
ANY raw feature's |z-score| (computed against the benign training mean/std)
exceeds a threshold calibrated the same way as the ensemble (CV on benign
data, targeting the same nominal FPR).

Run on the same fixed synthetic perturbation sets used elsewhere in the
sweep for an apples-to-apples comparison.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from features import build_feature_matrix, RAW_COLS
from train import fit_ensemble, ensemble_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def zscore_max_abs(X_std):
    # X_std is already standardized (mean 0, std 1 per column) -- max |z| across columns
    return np.abs(X_std).max(axis=1)


def cv_threshold_baseline(X_raw_std, target_fpr, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    held_out = []
    for train_idx, val_idx in kf.split(X_raw_std):
        mean = X_raw_std[train_idx].mean(axis=0)
        std = X_raw_std[train_idx].std(axis=0)
        z_val = np.abs((X_raw_std[val_idx] - mean) / std).max(axis=1)
        held_out.append(z_val)
    held_out = np.concatenate(held_out)
    return np.quantile(held_out, 1 - target_fpr)


def make_synth(df, seed=1):
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

    v_stealthy = base.copy()
    v_stealthy["cache-references"] *= 1.005
    v_stealthy["LLC-loads"] *= 1.0075
    v_stealthy["cycles"] *= 1.0025
    variants["stealthy_0.5pct"] = v_stealthy
    return variants


def main():
    df = pd.read_csv("../data/trace.csv")
    target_fpr = 0.01

    # --- Baseline: raw features only, per-feature z-score, max-abs rule ---
    raw = df[RAW_COLS].values
    raw_mean, raw_std = raw.mean(axis=0), raw.std(axis=0)
    raw_std_data = (raw - raw_mean) / raw_std
    baseline_threshold = cv_threshold_baseline(raw_std_data, target_fpr)

    # --- Ensemble: full pipeline as shipped (raw-only feature set, per train.py default) ---
    feat_df = build_feature_matrix(df)[RAW_COLS]
    feature_names = list(feat_df.columns)
    scaler = StandardScaler().fit(feat_df.values)
    X = scaler.transform(feat_df.values)
    from train import cross_validated_threshold
    ens_threshold, ens_fpr, _ = cross_validated_threshold(X, target_fpr, n_splits=5)
    ens_models, ens_ref = fit_ensemble(X)

    synth = make_synth(df)
    rows = []
    for name, v in synth.items():
        v_raw_std = (v[RAW_COLS].values - raw_mean) / raw_std
        baseline_flag = (zscore_max_abs(v_raw_std) >= baseline_threshold).mean()

        v_feat = build_feature_matrix(v)[feature_names]
        v_X = scaler.transform(v_feat.values)
        ens_score = ensemble_score(ens_models, ens_ref, v_X)
        ens_flag = (ens_score >= ens_threshold).mean()

        rows.append({
            "perturbation": name,
            "baseline_zscore_detect_rate": round(float(baseline_flag), 3),
            "ensemble_detect_rate": round(float(ens_flag), 3),
        })

    result = pd.DataFrame(rows)
    result.to_csv("../figures/baseline_comparison_results.csv", index=False)
    print(f"Baseline threshold (max|z| at {target_fpr:.0%} target FPR): {baseline_threshold:.3f}")
    print(f"Ensemble threshold: {ens_threshold:.4f} (CV achieved FPR: {ens_fpr:.3%})")
    print()
    print(result.to_string(index=False))
    plot_comparison(result)


def plot_comparison(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(result))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, result["baseline_zscore_detect_rate"] * 100, w,
           label="Naive per-feature z-score baseline", color="#C44E52")
    ax.bar(x + w/2, result["ensemble_detect_rate"] * 100, w,
           label="Ensemble (shipped model)", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(result["perturbation"], rotation=15)
    ax.set_ylabel("% flagged as anomalous")
    ax.set_title("Ensemble vs. naive baseline detection rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../figures/baseline_comparison.png", dpi=150)
    print("saved ../figures/baseline_comparison.png")


if __name__ == "__main__":
    main()
