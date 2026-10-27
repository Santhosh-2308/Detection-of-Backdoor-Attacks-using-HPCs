import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
)

meta = json.load(open("../models/detector.meta.json"))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.grey))

doc = SimpleDocTemplate("../docs/extended_report.pdf", pagesize=letter,
                         topMargin=0.8*inch, bottomMargin=0.8*inch,
                         leftMargin=0.8*inch, rightMargin=0.8*inch)
S = []

S.append(Paragraph("Detection of Backdoor Attacks using HPCs", styles["Title"]))
S.append(Paragraph("Method report — Problem Statement 1", styles["Normal"]))
S.append(Spacer(1, 12))

S.append(Paragraph("1. Problem framing", styles["H2"]))
S.append(Paragraph(
    "Only clean-inference HPC traces are available at training time (800 rows x 3 "
    "counters: cache-references, cycles, LLC-loads). No backdoor traces are provided, "
    "so this is a strict one-class / novelty-detection problem: the model must learn the "
    "geometry of benign behaviour and flag anything sufficiently far from it, without ever "
    "having seen a positive example.", styles["Body"]))

S.append(Paragraph("2. Data characteristics", styles["H2"]))
S.append(Paragraph(
    "Exploratory analysis of the 800 benign traces showed: (a) moderate-to-strong "
    "correlation between cache-references and LLC-loads (r=0.92), weaker correlation with "
    "cycles (r&#8776;0.53); (b) cache-references and LLC-loads are close to symmetric "
    "(skew -0.67 / -0.54) while cycles is right-skewed with high kurtosis (skew 0.90, "
    "kurtosis 6.76) driven by a handful of heavy-tail rows. This motivated using a "
    "<b>robust</b> covariance estimator rather than a plain Gaussian/Mahalanobis fit, and "
    "an ensemble rather than a single parametric model.", styles["Body"]))

S.append(Paragraph("3. Feature set: raw counters only", styles["H2"]))
S.append(Paragraph(
    "We evaluated two feature sets: the 3 raw counters alone, and raw + 3 derived "
    "scale-invariant ratios (cycles/cache-references, LLC-loads/cache-references, "
    "LLC-loads/cycles), motivated by the hypothesis that a backdoor trigger would leave a "
    "signature in the <i>relative</i> access pattern more reliably than in absolute "
    "magnitude. Section 9's ablation testing did not support that hypothesis once Local "
    "Outlier Factor was added to the ensemble (Section 4) &mdash; raw-only features "
    "detected a weak synthetic signal more than 2x as often. The shipped model therefore "
    "uses the 3 raw counters only; --feature-set raw_ratio remains available in the code "
    "for comparison.", styles["Body"]))

S.append(Paragraph("4. Model: rank-normalized ensemble of four one-class detectors", styles["H2"]))
S.append(Paragraph(
    "Four complementary one-class models are fit on the standardized 3-d raw feature space:", styles["Body"]))
S.append(ListFlowable([
    ListItem(Paragraph("<b>Robust Mahalanobis distance</b> (MinCovDet, support_fraction=0.9) "
                        "— models benign traces as a roughly elliptical cluster; the robust "
                        "covariance estimate is not skewed by the outlier-heavy tail in "
                        "'cycles'.", styles["Body"])),
    ListItem(Paragraph("<b>Isolation Forest</b> (300 trees) — non-parametric, captures "
                        "non-elliptical structure and is cheap at inference time.", styles["Body"])),
    ListItem(Paragraph("<b>One-Class SVM</b> (RBF kernel) — a smooth non-linear boundary "
                        "around the benign manifold.", styles["Body"])),
    ListItem(Paragraph("<b>Local Outlier Factor</b> (novelty mode) — density-based and "
                        "LOCAL rather than global: flags a point as anomalous if it is "
                        "sparse relative to its own neighbourhood, catching structure the "
                        "three global models above can miss if benign traces form more "
                        "than one loose cluster (e.g. different workload phases).", styles["Body"])),
], bulletType="bullet"))
S.append(Paragraph(
    "Each model's raw score is rank-normalized to [0,1] against its own training-score "
    "distribution (percentile rank), then the four normalized scores are averaged into a "
    "single ensemble anomaly score. Averaging an ensemble rather than trusting one model is "
    "a deliberate hedge: with zero labelled attack data to validate model choice against, no "
    "single model's assumptions (elliptical vs. tree-partition vs. kernel boundary vs. local "
    "density) can be confirmed as the right fit, so we rely on their consensus instead.", styles["Body"]))

S.append(Paragraph("5. Threshold calibration", styles["H2"]))
S.append(Paragraph(
    f"With no backdoor traces to measure TPR/FPR directly, the decision threshold is "
    f"calibrated purely from the benign distribution via 5-fold cross-validation: for each "
    f"fold the ensemble is fit on the training split and scored on the held-out split, "
    f"held-out scores are pooled across all 5 folds, and the threshold is set at the "
    f"(1 &minus; target FPR) quantile of that pooled, <b>out-of-sample</b> distribution. "
    f"Default target FPR = {meta['target_fpr']:.1%}; achieved held-out FPR at the selected "
    f"threshold = {meta['cv_achieved_fpr']:.2%} (threshold = {meta['threshold']:.4f}). "
    f"An independent 85/15 train/held-out split (not used for the shipped model, run purely "
    f"as an extra sanity check) gave 1.67% flag rate on genuinely unseen benign rows &mdash; "
    f"consistent with the 1% target given the small held-out sample size (120 rows). "
    f"The final deployed models are refit on all {meta['n_train_samples']} benign rows "
    f"using this threshold.", styles["Body"]))

img1 = Image("../figures/score_distribution.png", width=4.6*inch, height=3.07*inch)
S.append(img1)
S.append(Paragraph("Figure 1: Ensemble anomaly-score distribution over the benign training "
                    "set, with the calibrated decision threshold.", styles["Small"]))
S.append(Spacer(1, 10))

S.append(Paragraph("6. Sanity check (no real backdoor data available)", styles["H2"]))
S.append(Paragraph(
    "Since no backdoor traces exist to validate detection power, we ran a synthetic sanity "
    "check only: benign rows were perturbed by scaling cache-references, cycles and LLC-loads "
    "upward by 2-20% (roughly modelling extra cache/branch activity from trigger recognition) "
    "and re-scored with the trained detector. This is a sensitivity check on the detector's "
    "mechanics, <b>not</b> a claim about what real backdoor triggers look like on this "
    "hardware/model, and should not be read as an estimate of true TPR.", styles["Body"]))

img2 = Image("../figures/sensitivity_curve.png", width=4.6*inch, height=3.07*inch)
S.append(img2)
S.append(Paragraph("Figure 2: Fraction of synthetically perturbed rows flagged as anomalous, "
                    "as perturbation magnitude increases. Monotonic increase confirms the "
                    "detector's score is sensitive to counter-pattern shifts in the expected "
                    "direction.", styles["Small"]))
S.append(Spacer(1, 10))

data = [
    ["Perturbation", "0% (baseline)", "2%", "5%", "10%", "20%"],
    ["% flagged anomalous", "1.0%", "20%", "99%", "100%", "100%"],
]
t = Table(data, colWidths=[1.3*inch]+[0.85*inch]*5)
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4C72B0")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
]))
S.append(t)
S.append(Spacer(1, 12))

S.append(Paragraph("7. Perturbation-shape sweep: where the detector is weak", styles["H2"]))
S.append(Paragraph(
    "Section 6's sanity check used only one perturbation shape (uniform proportional "
    "scale-up of all 3 counters together). To find blind spots rather than just confirm "
    "the detector works on the easiest case, we swept several different synthetic "
    "perturbation shapes at matched magnitudes: single-counter spikes (only LLC-loads or "
    "only cycles moves), a correlated decrease, a pure ratio shift with magnitude held "
    "roughly in-distribution, small &lt;1% \"stealthy\" shifts, and pure random noise with "
    "no directional signal at all.", styles["Body"]))

img3 = Image("../figures/perturbation_shape_sweep.png", width=4.8*inch, height=3.6*inch)
S.append(img3)
S.append(Paragraph("Figure 3: Detection rate across perturbation shapes at fixed sample size "
                    "(n=100 each). Grey dotted line marks the ~1% baseline FPR.", styles["Small"]))
S.append(Spacer(1, 10))

S.append(Paragraph(
    "Findings worth flagging honestly:", styles["Body"]))
S.append(ListFlowable([
    ListItem(Paragraph("<b>Sub-1% \"stealthy\" shifts are nearly invisible</b> (2-4% flag "
                        "rate, barely above the 1% baseline). An adversary who tunes a "
                        "trigger to produce only a very small counter deviation could "
                        "plausibly evade this detector at the current operating point.", styles["Body"])),
    ListItem(Paragraph("<b>Weak single-counter spikes are now caught much more often</b> "
                        "after switching to raw-only features and adding LOF: a 5% "
                        "LLC-only spike is detected 52% of the time (up from 23-29% in an "
                        "earlier 3-model, raw+ratio version of this pipeline &mdash; see "
                        "Section 9).", styles["Body"])),
    ListItem(Paragraph("<b>This came with a real tradeoff, not a free improvement:</b> "
                        "pure random noise with no attack signal at all is now flagged more "
                        "often too (98% at 6% std, up from ~90-92% previously), and even a "
                        "mild uniform 2% shift is flagged 20% of the time (up from ~4%). The "
                        "raw-only+LOF configuration is more reactive across the board, not "
                        "selectively better &mdash; it trades some specificity for "
                        "sensitivity. A noisier real collection environment at evaluation "
                        "time could inflate FPR above the 1% calibrated here more than the "
                        "earlier configuration would have.", styles["Body"])),
], bulletType="bullet"))

S.append(Paragraph("8. Hyperparameter tradeoff sweep", styles["H2"]))
S.append(Paragraph(
    "A grid sweep over contamination &isin; {0.01, 0.03, 0.05, 0.10}, nu &isin; {0.01, 0.03, "
    "0.05, 0.10}, and target_fpr &isin; {0.01, 0.02, 0.05} was run to make the operating-point "
    "choice explicit rather than asserting one configuration as correct without justification. "
    "The result: <b>contamination and nu have almost no effect</b> on the final calibrated "
    "threshold or detection rate — because each base model's score is rank-normalized against "
    "its own training distribution before averaging, the ensemble is largely insensitive to "
    "these per-model knobs. <b>target_fpr is the real lever</b>: relaxing it from 1% to 5% "
    "roughly moves sensitivity on the weak-signal case from ~52% to ~89% at the cost of a "
    "5x higher false-positive rate on clean data.", styles["Body"]))

img4 = Image("../figures/fpr_sensitivity_tradeoff.png", width=4.6*inch, height=3.07*inch)
S.append(img4)
S.append(Paragraph("Figure 4: Detection rate on a strong-signal (uniform 10%) and weak-signal "
                    "(LLC-only 5% spike) synthetic case, as the target FPR is relaxed.",
                    styles["Small"]))
S.append(Spacer(1, 10))
S.append(Paragraph(
    "We keep target_fpr=1% as the default because the evaluation metrics (Accuracy, F1, "
    "AUROC) all penalize excess false positives, and 1% is a defensible conservative default; "
    "--target-fpr is exposed as a CLI flag precisely so this tradeoff can be re-tuned if the "
    "evaluators' validation set has a different clean/backdoor balance than assumed here.",
    styles["Body"]))

S.append(Paragraph("9. Feature ablation: raw-only vs. raw+ratio", styles["H2"]))
S.append(Paragraph(
    "Section 3 motivated adding 3 scale-invariant ratio features on the hypothesis that a "
    "backdoor trigger would leave a signature in relative access pattern more reliably than "
    "in absolute magnitude. We tested this directly, with the full 4-model ensemble "
    "including LOF, by training two configurations — raw-3-feature-only vs. raw+ratio-6-"
    "feature — and comparing detection rate on matched synthetic perturbations.", styles["Body"]))

data2 = [
    ["Configuration", "CV FPR", "Uniform 10%", "LLC spike 5%", "Ratio shift 10%"],
    ["Raw features only (3) — shipped", "1.0%", "100%", "52%", "100%"],
    ["Raw + ratio features (6)", "1.0%", "100%", "25%", "100%"],
]
t2 = Table(data2, colWidths=[1.9*inch, 0.65*inch, 0.95*inch, 0.95*inch, 1.0*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4C72B0")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
]))
S.append(t2)
S.append(Spacer(1, 8))
S.append(Paragraph(
    "With LOF in the ensemble, the gap is decisive: raw-only detects the weak single-counter "
    "spike <b>more than 2x as often</b> as raw+ratio (52% vs 25%), while both tie at 100% on "
    "every stronger signal. A plausible explanation is that adding 3 more, partly collinear "
    "dimensions dilutes LOF's local-density estimate (a version of the curse of "
    "dimensionality) without adding information the other three models don't already extract "
    "from the 3 raw counters' correlation structure. This evidence is strong enough that we "
    "switched the shipped default to raw-only features rather than leaving it as an open "
    "question; --feature-set raw_ratio remains available in train.py for comparison.",
    styles["Body"]))

S.append(Paragraph("10. Baseline comparison: is the ensemble's complexity justified?", styles["H2"]))
S.append(Paragraph(
    "To justify the ensemble's added complexity rather than assert it, we compared it "
    "against the simplest reasonable baseline: a per-feature z-score rule (flag if any raw "
    "counter's |z-score| against the benign mean/std exceeds a threshold), calibrated to the "
    "same 1% target FPR via the same cross-validation procedure.", styles["Body"]))

img5 = Image("../figures/baseline_comparison.png", width=4.6*inch, height=3.07*inch)
S.append(img5)
S.append(Paragraph("Figure 5: Detection rate, naive z-score baseline vs. the shipped ensemble, "
                    "on three synthetic perturbation cases.", styles["Small"]))
S.append(Spacer(1, 10))
S.append(Paragraph(
    "Result: on the strong uniform-shift signal both methods hit 100% (no benefit from the "
    "ensemble there), and on the near-invisible stealthy 0.5% shift both are equally blind "
    "(2%) — so the ensemble is not a universal improvement. But on the weak single-counter "
    "spike, the ensemble now detects it <b>10x</b> as often as the naive baseline (52% vs "
    "5%, up from a 4x gap in the earlier raw+ratio, 3-model version). This is the concrete "
    "justification for the added complexity: the baseline only looks at each counter "
    "independently, while the ensemble's covariance-, tree-, kernel-, and local-density-"
    "aware models catch a counter moving out of its usual relationship with the others, not "
    "just out of its own usual range.", styles["Body"]))

S.append(Paragraph("11. Evaluation metrics on the private validation set", styles["H2"]))
S.append(Paragraph(
    "The saved model exposes a continuous anomaly_score per trace in detect.py's output, so "
    "Accuracy, TPR, FPR, F1 and AUROC can all be computed directly from predictions.csv: "
    "AUROC/ROC from anomaly_score against ground truth, and Accuracy/TPR/FPR/F1 from the "
    "thresholded prediction column at the calibrated operating point (re-threshold-able via "
    "--target-fpr at train time if evaluators want a different operating point). "
    "src/eval.py computes all five metrics directly from a labeled CSV the moment one is "
    "available -- see README for usage.", styles["Body"]))

S.append(Paragraph("12. Known failure modes and limitations", styles["H2"]))
S.append(Paragraph(
    "Consolidated from the empirical sweeps above, in order of how likely each is to matter "
    "on the private evaluation set:", styles["Body"]))
S.append(ListFlowable([
    ListItem(Paragraph("<b>TPR against real backdoor traces is unverified.</b> All detection-"
                        "rate numbers in this report come from synthetic perturbations we "
                        "constructed ourselves, not real triggered inferences. They demonstrate "
                        "the detector's mechanics respond in the right direction; they are not "
                        "an estimate of real-world TPR.", styles["Body"])),
    ListItem(Paragraph("<b>Low-magnitude / stealthy triggers may evade detection</b> at the "
                        "default 1% FPR operating point (Section 7) — sub-1% counter deviations "
                        "were flagged only 2-4% of the time, barely above baseline.", styles["Body"])),
    ListItem(Paragraph("<b>Rank-normalization is relative to the training-set score "
                        "distribution, not an absolute scale.</b> A new trace's percentile "
                        "rank is computed against the benign scores from this collection "
                        "run. If the private validation set's clean traces come from a "
                        "meaningfully different environment (different CPU, different OS "
                        "scheduling noise), their absolute counter values could sit "
                        "differently against this reference distribution than intended, "
                        "which could inflate FPR even on truly clean validation traces. The "
                        "CV-based threshold partially mitigates this (it's calibrated from "
                        "held-out in-distribution data), but the reference distribution "
                        "itself still comes from the same single collection run as "
                        "training. A batch-level rank normalization, or an absolute rather "
                        "than percentile-based threshold, would be more robust to this if a "
                        "distribution shift is suspected at evaluation time.", styles["Body"])),
    ListItem(Paragraph("<b>The raw-only+LOF configuration trades specificity for "
                        "sensitivity</b> (Section 7): it catches weak, correlated signals "
                        "much better than the earlier raw+ratio, 3-model version, but is also "
                        "more reactive to pure benign noise (98% flagged at 6% std jitter, "
                        "with zero attack signal). A noisier evaluation environment than this "
                        "collection run could inflate real-world FPR above the 1% calibrated "
                        "here more than a less-sensitive configuration would.", styles["Body"])),
    ListItem(Paragraph("Only 3 raw HPCs were available; a richer counter set (branch "
                        "misses, TLB misses, etc.) would likely improve separability if "
                        "provided in a future iteration.", styles["Body"])),
    ListItem(Paragraph("The 800 benign rows come from a single collection run; distribution "
                        "shift from a different collection environment (CPU load, OS "
                        "scheduling noise) could inflate FPR at deployment, independent of "
                        "the noise-sensitivity point above.", styles["Body"])),
], bulletType="bullet"))

doc.build(S)
print("wrote ../docs/extended_report.pdf")
