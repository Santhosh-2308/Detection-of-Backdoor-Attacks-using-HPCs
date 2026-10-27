"""
Train an unsupervised anomaly detector for HPC-based backdoor-trigger
detection using ONLY clean/benign traces (no backdoor traces available
at training time -- strict novelty-detection / one-class setting).

Method
------
We fit FOUR complementary one-class detectors on standardized features:

  1. Robust Mahalanobis distance (MinCovDet / EllipticEnvelope)
     - Assumes benign traces form a roughly elliptical cluster;
       robust to the handful of heavy-tailed outliers already present
       in "cycles" (kurtosis ~6.8) because it uses a robust covariance
       estimate rather than the raw sample covariance.
  2. Isolation Forest
     - Non-parametric, captures non-linear / non-elliptical structure
       and is cheap to run at inference time.
  3. One-Class SVM (RBF kernel)
     - Captures a smooth non-linear decision boundary around the
       benign manifold; complements the density-based methods above.
  4. Local Outlier Factor (novelty mode)
     - Density-based and LOCAL rather than global: flags a point as
       anomalous if it is sparse relative to its own neighborhood, even
       if that neighborhood's absolute density is unremarkable overall.
       Catches a different failure mode than MCD/IsoForest/OCSVM, which
       all reason about deviation from a single global model of "normal".

Their per-sample scores are each rank-normalized to [0, 1] against the
training distribution and averaged into a single ensemble anomaly
score. Using an ensemble average (rather than a single model) reduces
the risk of any one model's blind spot being exploited by a backdoor
trigger, and is standard practice when no attack samples are available
to validate a single model choice against.

Feature set
-----------
Default is RAW-ONLY (3 raw HPC counters: cache-references, cycles,
LLC-loads). An earlier version of this pipeline also added 3 derived
ratio features (cycles/cacheref, LLC/cacheref, LLC/cycle) on the
hypothesis that they'd catch "shape" anomalies. Ablation testing
(src/feature_ablation.py) showed this hypothesis did NOT hold once LOF
was added to the ensemble: raw-only features detect a weak single-
counter-spike synthetic anomaly more than 2x as often as raw+ratio
(52% vs 25%), while tying on every stronger signal. The ratio features
are believed to dilute LOF's local-density estimate in a higher-
dimensional, more collinear space (a version of the curse of
dimensionality) without adding separable information the other three
models don't already extract from the 3 raw counters' correlation
structure. --feature-set raw_ratio is kept available for comparison.

Threshold calibration
----------------------
Because no backdoor traces are available, we cannot directly measure
TPR/FPR at training time. We instead calibrate the decision threshold
purely from the benign score distribution using K-fold cross-validation
(fit on ~80% of benign data, score the held-out ~20%) so the threshold's
nominal FPR is estimated out-of-sample rather than in-sample. The
default operating point targets a nominal 1% FPR on clean data; this is
exposed as a CLI flag so evaluators can re-target the ROC/AUROC
operating point during grading.

Usage
-----
    python train.py --data ../data/trace.csv --out ../models/detector.joblib \
        --target-fpr 0.01 --feature-set raw
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import build_feature_matrix, RAW_COLS

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def rank_normalize(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Map `scores` to [0,1] via their percentile rank within `ref_scores`.
    Higher output = more anomalous, for all model types alike."""
    ref_sorted = np.sort(ref_scores)
    ranks = np.searchsorted(ref_sorted, scores, side="right") / len(ref_sorted)
    return ranks


def fit_ensemble(X: np.ndarray, contamination: float = 0.05, nu: float = 0.05):
    """Fit the four base detectors and return them plus their raw
    "more anomalous = higher" scores on X (the data they were fit on)."""
    mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=0.9).fit(X)
    mcd_scores = mcd.mahalanobis(X)  # higher = more anomalous, natively

    iso = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=RANDOM_STATE
    ).fit(X)
    iso_scores = -iso.score_samples(X)  # flip sign: higher = more anomalous

    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(X)
    ocsvm_scores = -ocsvm.decision_function(X)  # flip sign: higher = more anomalous

    n_neighbors = min(20, max(5, len(X) // 10))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, novelty=True, contamination=contamination
    ).fit(X)
    lof_scores = -lof.score_samples(X)  # flip sign: higher = more anomalous

    return {"mcd": mcd, "iso": iso, "ocsvm": ocsvm, "lof": lof}, {
        "mcd": mcd_scores,
        "iso": iso_scores,
        "ocsvm": ocsvm_scores,
        "lof": lof_scores,
    }


def ensemble_score(models: dict, ref_scores: dict, X: np.ndarray) -> np.ndarray:
    mcd_s = models["mcd"].mahalanobis(X)
    iso_s = -models["iso"].score_samples(X)
    ocsvm_s = -models["ocsvm"].decision_function(X)
    lof_s = -models["lof"].score_samples(X)

    mcd_n = rank_normalize(mcd_s, ref_scores["mcd"])
    iso_n = rank_normalize(iso_s, ref_scores["iso"])
    ocsvm_n = rank_normalize(ocsvm_s, ref_scores["ocsvm"])
    lof_n = rank_normalize(lof_s, ref_scores["lof"])

    return (mcd_n + iso_n + ocsvm_n + lof_n) / 4.0


def cross_validated_threshold(X: np.ndarray, target_fpr: float, n_splits: int = 5):
    """Estimate the ensemble-score threshold that yields ~target_fpr on
    UNSEEN benign data, via K-fold: fit on train fold, score held-out fold,
    pool held-out scores across all folds, take the (1 - target_fpr) quantile."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    held_out_scores = []
    for train_idx, val_idx in kf.split(X):
        models, ref_scores = fit_ensemble(X[train_idx])
        val_scores = ensemble_score(models, ref_scores, X[val_idx])
        held_out_scores.append(val_scores)
    held_out_scores = np.concatenate(held_out_scores)
    threshold = np.quantile(held_out_scores, 1 - target_fpr)
    achieved_fpr = float((held_out_scores >= threshold).mean())
    return threshold, achieved_fpr, held_out_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/trace.csv")
    ap.add_argument("--out", default="../models/detector.joblib")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--feature-set", choices=["raw", "raw_ratio"], default="raw",
                     help="'raw' (default) = 3 raw HPC counters only, per ablation "
                          "evidence that ratio features hurt weak-signal detection "
                          "once LOF is in the ensemble (see report.pdf Section 9). "
                          "'raw_ratio' = raw + 3 derived ratio features.")
    args = ap.parse_args()

    df_raw = pd.read_csv(args.data)
    feat_df_full = build_feature_matrix(df_raw)
    feat_df = feat_df_full[RAW_COLS] if args.feature_set == "raw" else feat_df_full
    feature_names = list(feat_df.columns)

    scaler = StandardScaler().fit(feat_df.values)
    X = scaler.transform(feat_df.values)

    # 1) Cross-validated threshold estimate (out-of-sample, unbiased)
    threshold, achieved_fpr, cv_scores = cross_validated_threshold(
        X, args.target_fpr, n_splits=5
    )
    print(f"[CV] target FPR={args.target_fpr:.3%}  "
          f"achieved held-out FPR at threshold={achieved_fpr:.3%}")
    print(f"[CV] threshold={threshold:.4f}  "
          f"(held-out ensemble-score mean={cv_scores.mean():.4f}, "
          f"std={cv_scores.std():.4f})")

    # 2) Refit final models on ALL benign data for deployment
    final_models, final_ref_scores = fit_ensemble(
        X, contamination=args.contamination, nu=args.nu
    )
    final_train_scores = ensemble_score(final_models, final_ref_scores, X)
    print(f"[Final fit] in-sample ensemble-score at same threshold -> "
          f"flag rate={(final_train_scores >= threshold).mean():.3%}")

    bundle = {
        "scaler": scaler,
        "models": final_models,
        "ref_scores": final_ref_scores,  # training scores, needed for rank-normalization at inference
        "feature_names": feature_names,
        "feature_set": args.feature_set,
        "threshold": float(threshold),
        "target_fpr": args.target_fpr,
        "cv_achieved_fpr": achieved_fpr,
        "contamination": args.contamination,
        "nu": args.nu,
        "n_train_samples": int(len(feat_df)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved detector bundle -> {out_path.resolve()}")

    meta = {k: v for k, v in bundle.items() if k not in ("scaler", "models", "ref_scores")}
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
