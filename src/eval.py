"""
Evaluate the detector against LABELED data (clean + backdoor traces) using
the exact metrics the hackathon evaluation uses: Accuracy, TPR, FPR, F1, AUROC.

This script is a no-op until real backdoor-labeled data exists -- it is
provided so that the moment the private validation set (or any labeled
subset) is available, evaluation is a single command rather than a
from-scratch metrics script written under time pressure.

Expected input CSV
-------------------
Same raw columns as training data (cache-references, cycles, LLC-loads)
PLUS a ground-truth column. The label column name and the value that means
"backdoor" are configurable via CLI flags to match whatever format the
evaluators (or your own labeled test set) actually uses.

Usage
-----
    python eval.py --model ../models/detector.joblib --data labeled_traces.csv \
        --label-col label --positive-value backdoor \
        --out eval_report.json --roc-plot ../figures/roc_curve.png
"""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

from features import build_feature_matrix
from train import ensemble_score

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/detector.joblib")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-col", default="label",
                     help="Column containing ground-truth labels")
    ap.add_argument("--positive-value", default="backdoor",
                     help="Value in --label-col that means 'this row is a backdoor trace'")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--roc-plot", default=None,
                     help="If given, save an ROC curve PNG to this path")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.data)

    if args.label_col not in df.columns:
        raise ValueError(
            f"'{args.label_col}' not found in {args.data}. "
            f"Available columns: {list(df.columns)}. "
            f"Pass --label-col to match your file's ground-truth column name."
        )

    y_true = (df[args.label_col] == args.positive_value).astype(int).values
    if y_true.sum() == 0:
        raise ValueError(
            f"No rows with {args.label_col}=={args.positive_value} found -- "
            f"check --positive-value matches your label encoding."
        )

    feat_df = build_feature_matrix(df)[bundle["feature_names"]]
    X = bundle["scaler"].transform(feat_df.values)
    scores = ensemble_score(bundle["models"], bundle["ref_scores"], X)
    y_pred = (scores >= bundle["threshold"]).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    results = {
        "n_samples": int(len(df)),
        "n_positive_backdoor": int(y_true.sum()),
        "n_negative_clean": int((1 - y_true).sum()),
        "threshold_used": float(bundle["threshold"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tpr_recall": float(tpr),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    if args.roc_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr_curve, tpr_curve, color="#4C72B0",
                label=f"AUROC = {results['auroc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve on labeled evaluation data")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.roc_plot, dpi=150)
        print(f"Saved ROC curve -> {args.roc_plot}")


if __name__ == "__main__":
    main()
