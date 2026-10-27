"""
Score new HPC traces with a trained detector bundle and label each row
clean (0) or backdoor/anomalous (1).

Usage
-----
    python detect.py --model ../models/detector.joblib --data new_traces.csv \
        --out predictions.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from features import build_feature_matrix
from train import ensemble_score  # reuse identical scoring logic used at train time


def load_bundle(path):
    return joblib.load(path)


def score_dataframe(bundle, df_raw: pd.DataFrame) -> pd.DataFrame:
    feat_df = build_feature_matrix(df_raw)
    feat_df = feat_df[bundle["feature_names"]]  # enforce column order
    X = bundle["scaler"].transform(feat_df.values)

    scores = ensemble_score(bundle["models"], bundle["ref_scores"], X)
    preds = (scores >= bundle["threshold"]).astype(int)

    out = df_raw.copy()
    out["anomaly_score"] = scores
    out["prediction"] = preds  # 0 = clean, 1 = backdoor/anomalous
    out["label"] = np.where(preds == 1, "backdoor", "clean")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/detector.joblib")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    df_raw = pd.read_csv(args.data)
    result = score_dataframe(bundle, df_raw)
    result.to_csv(args.out, index=False)

    n_flagged = int(result["prediction"].sum())
    print(f"Scored {len(result)} traces -> {n_flagged} flagged as backdoor "
          f"({n_flagged / len(result):.2%}), threshold={bundle['threshold']:.4f}")
    print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
