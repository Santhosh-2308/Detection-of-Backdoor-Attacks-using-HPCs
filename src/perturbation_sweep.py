"""
Synthetic sanity-check sweep across DIFFERENT perturbation *shapes*, not just
uniform scaling. Goal: find out where the detector's blind spots are, and
document them honestly rather than only showing the perturbation type that
makes the detector look best.

IMPORTANT: these are synthetic proxies for "what a backdoor trigger might do
to HPC traces" -- not real backdoor data. Treat results as a sensitivity/
robustness diagnostic, not a TPR estimate.
"""
import warnings
import numpy as np
import pandas as pd
import joblib

from features import build_feature_matrix
from train import ensemble_score

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)


def load_bundle(path="../models/detector.joblib"):
    return joblib.load(path)


def score(bundle, df):
    feat = build_feature_matrix(df)[bundle["feature_names"]]
    X = bundle["scaler"].transform(feat.values)
    s = ensemble_score(bundle["models"], bundle["ref_scores"], X)
    return s, (s >= bundle["threshold"]).mean()


def make_perturbations(df, n=100, seed=1):
    base = df.sample(n, random_state=seed).reset_index(drop=True)
    variants = {}

    # 1) Uniform proportional scale-up (all 3 counters move together, same direction)
    for pct in [0.02, 0.05, 0.10, 0.20]:
        v = base.copy()
        v["cache-references"] *= 1 + pct
        v["LLC-loads"] *= 1 + 1.5 * pct
        v["cycles"] *= 1 + 0.5 * pct
        variants[f"uniform_scale_{int(pct*100)}pct"] = v

    # 2) Single-counter spike only (cache-references untouched, only LLC-loads jumps)
    for pct in [0.05, 0.15, 0.30]:
        v = base.copy()
        v["LLC-loads"] *= 1 + pct
        variants[f"llc_only_spike_{int(pct*100)}pct"] = v

    # 3) Single-counter spike only (cycles only -- e.g. extra compute, no extra memory traffic)
    for pct in [0.05, 0.15, 0.30]:
        v = base.copy()
        v["cycles"] *= 1 + pct
        variants[f"cycles_only_spike_{int(pct*100)}pct"] = v

    # 4) Correlated DECREASE (fewer cache refs / LLC loads -- e.g. trigger short-circuits
    #    some computation rather than adding to it)
    for pct in [0.05, 0.15]:
        v = base.copy()
        v["cache-references"] *= 1 - pct
        v["LLC-loads"] *= 1 - 1.2 * pct
        variants[f"uniform_decrease_{int(pct*100)}pct"] = v

    # 5) Ratio-only shift: keep magnitude in the benign range but change the
    #    RELATIONSHIP between counters (tests whether ratio features earn their keep)
    for pct in [0.10, 0.25]:
        v = base.copy()
        # scale LLC-loads up while scaling cache-references down by the same amount,
        # so overall magnitude stays roughly in-distribution but the ratio shifts hard
        v["LLC-loads"] *= 1 + pct
        v["cache-references"] *= 1 - pct * 0.3
        variants[f"ratio_shift_only_{int(pct*100)}pct"] = v

    # 6) Pure noise / jitter, NO directional shift (tests false-positive robustness
    #    to benign measurement noise, not an attack signature at all)
    for std in [0.01, 0.03, 0.06]:
        v = base.copy()
        for c in ["cache-references", "cycles", "LLC-loads"]:
            v[c] *= 1 + RNG.normal(0, std, len(v))
        variants[f"pure_noise_std{std}"] = v

    # 7) Tiny stealthy shift (adversary trying to stay just under the radar)
    for pct in [0.005, 0.01]:
        v = base.copy()
        v["cache-references"] *= 1 + pct
        v["LLC-loads"] *= 1 + 1.5 * pct
        v["cycles"] *= 1 + 0.5 * pct
        variants[f"stealthy_uniform_{pct*100:.1f}pct"] = v

    return variants


def plot_results(results_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = results_df.sort_values("flag_rate")
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ["#C44E52" if "noise" in n else "#4C72B0" for n in df["perturbation"]]
    ax.barh(df["perturbation"], df["flag_rate_pct"], color=bar_colors)
    ax.set_xlabel("% flagged as anomalous")
    ax.set_title("Detection rate across perturbation shapes\n"
                 "(red = pure noise, no directional attack signal)")
    ax.axvline(1, color="grey", linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig("../figures/perturbation_shape_sweep.png", dpi=150)
    print("saved ../figures/perturbation_shape_sweep.png")


def main():
    bundle = load_bundle()
    df = pd.read_csv("../data/trace.csv")
    variants = make_perturbations(df)

    results = []
    for name, v in variants.items():
        _, flag_rate = score(bundle, v)
        results.append((name, flag_rate))

    results_df = pd.DataFrame(results, columns=["perturbation", "flag_rate"])
    results_df["flag_rate_pct"] = (results_df["flag_rate"] * 100).round(1)
    results_df = results_df.sort_values("flag_rate")
    print(results_df.to_string(index=False))
    results_df.to_csv("../figures/perturbation_sweep_results.csv", index=False)
    plot_results(results_df)
    return results_df


if __name__ == "__main__":
    main()
