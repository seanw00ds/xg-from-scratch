# A2 plan — Expected Goals (xG) via logistic regression built from scratch

Started 10 Sep 2026. Due Mon 28 Sep 2026 23:59. Target submit 25 Sep. Viva Tue 13 Oct 09:00 (or practice session Tue 29 Sep).

## The task

**Mixed Option 1 / Option 2, leaning Option 1.** The spec allows this explicitly: *"you may define your effort as a mixed type, you will not be penalised for crossing boundaries between categories."*

Study **logistic regression** — implemented from scratch, gradient derived by hand — applied to **expected goals (xG)** on real football shot data.

### The problem, at technical level

Given a shot attempt, output the probability it results in a goal.

**Training phase**
- Input: matrix X of shape (n, d), one row per shot, real-valued features. Vector y in {0, 1}^n, 1 = goal.
- Hyperparameters: learning rate η, epochs T, L2 strength λ, seed.
- Output: weight vector w in R^(d+1), bias absorbed as w0.

**Deployment phase**
- Input: one shot's feature vector x in R^d — same feature order, same scaling as training.
- Output: a single real number in [0, 1], the probability this shot is a goal. Not a label. The probability *is* the product.

### Research question (criterion A)

> An xG model cannot predict whether an individual shot goes in — the outcome is genuinely stochastic. Its actual objective is **calibration**: of all shots rated 0.30, roughly 30% should be goals. But the model is trained by minimising log loss on individual binary outcomes. Where does that gap between the training criterion and the deployment objective actually bite, how do you detect it, and does adding features improve calibration or only sharpness?

## Why this beats the alternatives

- **The loss-vs-objective mismatch is real, not manufactured.** Criterion C's 80% and 100% bands want specific cases where the loss minimises but the task objective fails. Here it is structural, and Sean can explain it in footballing terms.
- **Accuracy is actively misleading here.** ~10% of shots are goals, so "always predict no goal" scores ~90% accuracy and is useless. That is a genuine class-imbalance discussion, not a textbook one.
- **Gradient descent is the training algorithm** — the exact Module 5 content Sean is on now. The gradient is one line and derivable by hand.
- **Sean cares about football.** Per `context/study-approach.md` his results are close to binary on engagement. This is the highest-leverage factor in the whole plan.
- **Criterion A's Excellent band wants "why traditional approaches are not sufficient".** Easy and true: shot counts ignore shot quality; a fixed rule cannot capture the interaction between distance, angle and pressure.

## The model — three components, one line each

| | Component | This project |
|---|---|---|
| **H** | Hypothesis space | `{ x -> σ(w·x + w0) }`, σ(z) = 1/(1+e^(−z)). All linear-in-features probability models. |
| **L** | Criterion | Binary cross-entropy (log loss), + λ‖w‖² |
| **A** | Algorithm | Batch gradient descent. ∇ = Xᵀ(σ(Xw) − y)/n + 2λw |

That is the entire model. ~60 lines of NumPy.

## The narrative arc (criterion B gold)

Start from the **perceptron** (Module 5 content, and the quiz just done):
1. Perceptron gives a hard ±1 label. xG needs a probability. Wrong output type.
2. Shot data is fundamentally non-separable — identical shots sometimes go in, sometimes don't. The perceptron never converges on it.
3. Fix: keep the same hypothesis space, swap the criterion for log loss and the algorithm for gradient descent. That *is* logistic regression.

H stays fixed, L and A change. The H/L/A framework told as a story.

## Features — start with 2, then extend

**Phase 1 (get the pipeline working):** distance to goal centre, and shot angle (the angle subtended by the goalmouth from the shot location). These two alone make a respectable xG model, and it's the classic minimal version.

**Phase 2 (extend):** body part (foot/head/other), open play vs set piece vs counter, assist type, whether it was a first-time shot, number of defenders between ball and goal (StatsBomb 360 freeze-frame where available).

Incremental building with evaluation at each step is also the report's results structure.

## Data

**StatsBomb Open Data** — free, real, event-level, on GitHub as raw JSON. Non-trivial, so it satisfies Option 2's data requirement. Downloaded inside the notebook so the project stays self-contained as the spec requires.

## Evaluation (criterion C)

- Split by **match**, not by shot, to avoid leakage between shots from the same game.
- **Log loss** and **Brier score** as primary. Accuracy reported only to show why it's useless here.
- **Calibration / reliability diagram** — the central figure. Predicted probability bucket vs observed goal rate.
- **Brier decomposition** into reliability and resolution: separates "are the probabilities honest" from "do they discriminate". This is the technical language for calibration vs sharpness.
- ROC-AUC and PR curve, with a discussion of why PR is more informative under this imbalance.
- Baseline comparison: overall base rate (~10%), and a distance-only model.

## Timeline

**Revised 16 Sep 2026.** Nothing built as of this date, original plan slipped ~5 days. Ethics AT2 essay (1500 wds) is due the same day and not started, so ML must be done by Thu 24 Sep.

| By | Thing |
|---|---|
| Thu 17 Sep | **Theory done**: true vs empirical risk, ERM, log loss, gradient descent, all anchored on xG. GitHub repo created. |
| Sat 19 Sep | Data loading working, distance + angle features computed and sanity-checked |
| Mon 21 Sep | Logistic regression from scratch, gradient derived by hand, trained on 2 features |
| Tue 22 Sep | Extra features, all experiments, calibration analysis |
| Thu 24 Sep | Report + implementation log written, Colab public, PDF **submitted**. Register with Nicole Zhuo same day if doing 29 Sep practice defence |
| Fri 25 – Mon 28 Sep | Ethics essay |

Implementation log written **as we go**, not retrofitted. It documents AI use, and the spec says that log is what protects him if questioned in the viva.

## Filename

`SeanWoods_26107565_2026_UTS_ML_Journal.pdf`

## Progress — Mon 21 Sep 2026

Built in one session. Everything below is done and verified.

- Data: 7,451 shots, 663 goals (8.9%), 314 matches, six men's international
  tournaments 2018-2024. All carry 360 freeze frames and StatsBomb's own xG.
- Model: logistic regression from scratch, gradient derived by hand, checked
  against central differences (4.55e-10). Metrics all hand written and checked
  against sklearn. Gradient descent solution matches sklearn LBFGS to 0.0074.
- Results: M4 test log loss 0.2568 vs StatsBomb's commercial 0.2472.
- Research question answered: features bought resolution (0.0083 to 0.0125),
  not reliability (0.0013 to 0.0008). Overall bias +0.0001 while counter
  attacks are underrated by 6.0 points and crosses by 4.2.
- Extra experiment: same model trained on squared error, calibration 50% worse.
- Report written, PDF renders at report/SeanWoods_26107565_2026_UTS_ML_Journal.pdf
- Viva prep written at notes/viva-prep.md
- Local git repo initialised and committed.

## Open items

- [ ] Push to public GitHub repo, paste Colab + repo URLs into the report
- [ ] Re-render PDF once URLs are in
- [ ] Submit (target Thu 24 Sep), then email Nicole Zhuo to register for the
      29 Sep practice defence
- [ ] Rehearse the 5 minutes out loud, twice, against a timer
