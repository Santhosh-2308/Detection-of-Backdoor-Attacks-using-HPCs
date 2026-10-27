# HPC-Based Backdoor Trigger Detection

Anomaly detector that flags backdoor-trigger inferences using only
hardware performance counters (`cache-references`, `cycles`, `LLC-loads`),
trained on **benign-only** traces (strict one-class / novelty-detection
setting — no backdoor traces were available or used).

## Method summary

1. **Feature engineering** (`src/features.py`): 3 raw counters
   (`cache-references`, `cycles`, `LLC-loads`) used directly, as the
   shipped default. 3 additional scale-invariant ratio features are
   also implemented but NOT used by default — ablation testing showed
   they measurably *hurt* weak-signal detection once Local Outlier
   Factor is in the ensemble (see point 2). Available via
   `--feature-set raw_ratio` for comparison.
2. **Model**: an ensemble of four one-class detectors fit on
   standardized features:
   - Robust Mahalanobis distance (`MinCovDet`)
   - Isolation Forest
   - One-Class SVM (RBF)
   - Local Outlier Factor (novelty mode) — local density-based,
     catches structure the three global models above can miss

   Each model's score is rank-normalized to `[0,1]` against its own
   training distribution; the four normalized scores are averaged into
   one **ensemble anomaly score**.
3. **Threshold**: calibrated via 5-fold cross-validation on the benign
   data only (fit on 80%, score the held-out 20%, pool across folds),
   so the reported false-positive rate is out-of-sample, not in-sample.
   Default operating point: **1% nominal FPR** on clean traces
   (`--target-fpr`, tunable).

See `report.pdf` for the full writeup, including why an ensemble was
chosen over a single model, the sanity-check methodology used to
validate detector sensitivity in the absence of real backdoor traces,
a perturbation-shape sweep that surfaces detector blind spots, a
hyperparameter tradeoff sweep, a feature ablation (raw-only beats
raw+ratio 52% vs 25% on a weak-signal test once LOF is added — this is
why raw-only is the shipped default, not an open question), and a
baseline comparison showing the ensemble detects a weak signal 10x as
often as a naive per-feature z-score rule.

## Repo layout

```
hpc-backdoor-detector/
├── data/trace.csv              # provided benign HPC traces (800 rows)
├── src/
│   ├── features.py              # shared feature engineering
│   ├── train.py                 # trains + calibrates + saves the detector
│   ├── detect.py                # scores new traces with a saved detector
│   ├── eval.py                  # Accuracy/TPR/FPR/F1/AUROC once labeled data exists
│   ├── perturbation_sweep.py    # detection rate across synthetic attack shapes
│   ├── hyperparam_sweep.py      # contamination/nu/target_fpr tradeoff grid
│   ├── feature_ablation.py      # raw-only vs raw+ratio features
│   ├── baseline_comparison.py   # ensemble vs. naive z-score baseline
│   ├── generate_figures.py      # regenerates score_distribution/sensitivity_curve figures
│   └── make_report.py           # rebuilds report.pdf from current model + figures
├── models/
│   ├── detector.joblib          # trained detector bundle (scaler + 4 models + threshold)
│   └── detector.meta.json       # human-readable summary of the saved config
├── figures/                     # diagnostic plots + sweep result CSVs, all reproducible from src/*.py
├── report.pdf                   # short report (IEEE two-column format, 3 pages) - same as paper/main.pdf
├── docs/extended_report.pdf     # longer 7-page version with extra detail
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
cd src
python train.py --data ../data/trace.csv --out ../models/detector.joblib --target-fpr 0.01
```

Retrains the ensemble (four one-class models: MinCovDet, Isolation Forest,
One-Class SVM, Local Outlier Factor) on the given benign CSV, cross-validates
the decision threshold, and saves a single `.joblib` bundle containing the
fitted scaler, the four base models, their reference training scores
(needed for rank-normalization at inference time), and the calibrated
threshold. Default feature set is the 3 raw counters only
(`--feature-set raw`); `--feature-set raw_ratio` adds 3 derived ratio
features, kept available for comparison (see `report.pdf` Section 9 for
why raw-only is the shipped default).

## Score new traces

```bash
cd src
python detect.py --model ../models/detector.joblib --data path/to/new_traces.csv --out predictions.csv
```

Input CSV must contain the same three raw columns as the training data:
`cache-references`, `cycles`, `LLC-loads`. Output adds `anomaly_score`,
`prediction` (0=clean, 1=backdoor), and `label`.

## Analysis scripts (evidence behind report.pdf's claims)

```bash
cd src
python perturbation_sweep.py    # detection rate across different synthetic attack shapes
python hyperparam_sweep.py      # contamination/nu/target_fpr tradeoff grid
python feature_ablation.py      # raw-only vs raw+ratio features, head-to-head
python baseline_comparison.py   # ensemble vs. naive per-feature z-score baseline
python generate_figures.py      # regenerates score_distribution.png + sensitivity_curve.png
                                 # from the trained model (run AFTER train.py and
                                 # perturbation_sweep.py, in that order)
python synthetic_metrics.py     # Accuracy/TPR/FPR/F1/AUROC on a SYNTHETIC labeled set (not real backdoor data)
python make_report.py           # rebuilds report.pdf from current model + figures
python make_paper_figures.py    # column-width figures for the IEEE paper
# then: cd ../paper && pdflatex main.tex && pdflatex main.tex   (needs IEEEtran.cls)
```

Each sweep script writes its own results to `figures/*.csv`, prints a summary table,
and saves its own figure directly (no figures are committed as static images without
a script that reproduces them). Re-running the full sequence above regenerates every
number and every figure in `report.pdf` from scratch — verified end-to-end on a fresh
clone with only `requirements.txt` installed.

## Evaluating against labeled data (once available)

```bash
cd src
python eval.py --model ../models/detector.joblib --data labeled_traces.csv \
    --label-col label --positive-value backdoor \
    --out eval_report.json --roc-plot ../figures/roc_curve.png
```

`labeled_traces.csv` needs the same 3 raw columns plus a ground-truth column
(name and "this row is a backdoor" value are both configurable via flags).
Outputs Accuracy, TPR, FPR, F1, AUROC, and a confusion matrix as JSON, plus
an optional ROC curve plot.

## Notes / limitations

- No backdoor traces were provided, so the threshold's FPR is
  empirically validated (via cross-validation) but its **TPR is
  unverified** until scored against the private evaluation set.
- The ensemble/threshold were sanity-checked against synthetically
  perturbed benign traces across multiple perturbation *shapes*
  (see `report.pdf` Section 7), which is a proxy for, not a
  guarantee of, real backdoor-trigger sensitivity. Known weak spots:
  sub-1% stealthy shifts are barely detected, and single-counter-only
  anomalies are harder to catch than correlated multi-counter shifts.
- Pure random noise (no attack signal) can also get flagged at
  moderate magnitude — the detector responds to deviation magnitude,
  not specifically to attack-like structure.
- Rank-normalization (used to combine the four models' scores) is
  relative to this collection run's benign score distribution, not an
  absolute scale. A private validation set collected in a meaningfully
  different environment could shift where its clean traces land against
  this reference, inflating FPR even on genuinely clean data (see
  `report.pdf` Section 12).
- The shipped ensemble (raw features + LOF) trades some specificity for
  sensitivity vs. an earlier raw+ratio, 3-model version: it catches weak
  correlated signals much better, but is also more reactive to pure
  benign noise. See `report.pdf` Section 7.
- `--target-fpr`, `--contamination`, and `--nu` are exposed as CLI flags
  in case the operating point needs to be re-tuned during evaluation.
  Sweeps show `target_fpr` is the real sensitivity/specificity lever;
  `contamination`/`nu` have little effect due to rank-normalization.
- The 3 ratio features in `features.py` (`--feature-set raw_ratio`)
  measurably *hurt* weak-signal detection once LOF was added to the
  ensemble (52% vs 25% on a matched synthetic test) — this is why
  raw-only is the shipped default, not a leftover open question.
