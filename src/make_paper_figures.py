"""
Column-width (3.4 in) vector figures for the IEEE-format paper (paper/main.tex).
Reads the trained bundle and the sweep CSVs; run after train.py and the sweep scripts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

from features import build_feature_matrix
from train import ensemble_score

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
W = 3.45
OUT = "../paper/figs/"
BLUE, RED, ORANGE = "#2F5D8C", "#B03A2E", "#D98E04"


def fig_scores():
    b = joblib.load("../models/detector.joblib")
    df = pd.read_csv("../data/trace.csv")
    X = b["scaler"].transform(build_feature_matrix(df)[b["feature_names"]].values)
    s = ensemble_score(b["models"], b["ref_scores"], X)
    fig, ax = plt.subplots(figsize=(W, 1.9))
    ax.hist(s, bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(b["threshold"], color=RED, ls="--", lw=1, label=rf"$\tau$ = {b['threshold']:.4f}")
    ax.set_xlabel("Ensemble anomaly score"); ax.set_ylabel("Benign traces")
    ax.set_ylim(0, 46); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3); fig.savefig(OUT + "scores.pdf"); plt.close(fig)


def fig_shapes():
    d = pd.read_csv("../figures/perturbation_sweep_results.csv").sort_values("flag_rate_pct")
    names = [n.replace("_", " ").replace("pct", "%").replace("std", "std ") for n in d["perturbation"]]
    cols = [RED if "noise" in n else BLUE for n in d["perturbation"]]
    fig, ax = plt.subplots(figsize=(W, 3.7))
    ax.barh(names, d["flag_rate_pct"], color=cols, height=0.7)
    ax.axvline(1, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Flagged as anomalous (%)")
    ax.tick_params(axis="y", labelsize=5.8)
    fig.tight_layout(pad=0.3); fig.savefig(OUT + "shapes.pdf"); plt.close(fig)


def fig_tradeoff():
    h = pd.read_csv("../figures/hyperparam_sweep_results.csv")
    g = h.groupby("target_fpr")[["detect_uniform_10pct", "detect_llc_spike_5pct"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(W, 1.9))
    ax.plot(g.target_fpr * 100, g.detect_uniform_10pct * 100, "o-", color=BLUE, ms=3, lw=1, label="Uniform +10% (strong)")
    ax.plot(g.target_fpr * 100, g.detect_llc_spike_5pct * 100, "s-", color=ORANGE, ms=3, lw=1, label="LLC-only +5% (weak)")
    ax.set_xlabel("Target FPR $\\alpha$ (%)"); ax.set_ylabel("Flagged (%)")
    ax.set_ylim(0, 108); ax.legend(frameon=False, loc="center right")
    fig.tight_layout(pad=0.3); fig.savefig(OUT + "tradeoff.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_scores(); fig_shapes(); fig_tradeoff(); print("paper figures written to", OUT)
