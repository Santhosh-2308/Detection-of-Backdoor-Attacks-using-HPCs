"""
Accuracy/TPR/FPR/F1/AUROC on a SYNTHETIC labeled set (NOT real backdoor data).
Model is trained on a random 85% of benign traces (680); the other 15% (120) are
unseen benign negatives, and positives are author-constructed perturbations of those
same 120 unseen rows. Shows how the metrics behave, not real-world performance.
"""
import subprocess, sys, warnings
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from features import build_feature_matrix
from train import ensemble_score
warnings.filterwarnings("ignore")

df = pd.read_csv("../data/trace.csv")
tr = df.sample(frac=0.85, random_state=1); te = df.drop(tr.index)
tr.to_csv("/tmp/train85.csv", index=False)
subprocess.run([sys.executable, "train.py", "--data", "/tmp/train85.csv",
                "--out", "/tmp/model85.joblib"], check=True, capture_output=True)
b = joblib.load("/tmp/model85.joblib")

def scores(d):
    X = b["scaler"].transform(build_feature_matrix(d)[b["feature_names"]].values)
    return ensemble_score(b["models"], b["ref_scores"], X)

def mult(**k):
    d = te.copy()
    for c, f in k.items(): d[c.replace("_", "-") if c == "cache_references" else c] *= f
    return d
P = {
 "Uniform +20%":  lambda: te.assign(**{"cache-references": te["cache-references"]*1.20, "LLC-loads": te["LLC-loads"]*1.30, "cycles": te["cycles"]*1.10}),
 "Uniform +10%":  lambda: te.assign(**{"cache-references": te["cache-references"]*1.10, "LLC-loads": te["LLC-loads"]*1.15, "cycles": te["cycles"]*1.05}),
 "Uniform +5%":   lambda: te.assign(**{"cache-references": te["cache-references"]*1.05, "LLC-loads": te["LLC-loads"]*1.075, "cycles": te["cycles"]*1.025}),
 "Cycles-only +5%": lambda: te.assign(cycles=te["cycles"]*1.05),
 "LLC-only +5%":  lambda: te.assign(**{"LLC-loads": te["LLC-loads"]*1.05}),
 "Uniform -5%":   lambda: te.assign(**{"cache-references": te["cache-references"]*0.95, "LLC-loads": te["LLC-loads"]*0.94}),
 "Stealthy +0.5%": lambda: te.assign(**{"cache-references": te["cache-references"]*1.005, "LLC-loads": te["LLC-loads"]*1.0075, "cycles": te["cycles"]*1.0025}),
}
sn = scores(te); tau = b["threshold"]
rows = []
for name, f in P.items():
    sp = scores(f())
    y = np.r_[np.zeros(len(sn)), np.ones(len(sp))]; s = np.r_[sn, sp]; p = (s >= tau).astype(int)
    tp = p[y==1].mean(); fp = p[y==0].mean()
    rows.append([name, accuracy_score(y,p), tp, fp, f1_score(y,p), roc_auc_score(y,s)])
out = pd.DataFrame(rows, columns=["Perturbation","Accuracy","TPR","FPR","F1","AUROC"]).round(3)
print(f"threshold={tau:.4f}; negatives = {len(sn)} unseen benign rows; positives = {len(sn)} perturbed copies")
print(out.to_string(index=False)); out.to_csv("../figures/synthetic_metrics.csv", index=False)
