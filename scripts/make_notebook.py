"""Assembles the submission notebook from the source in scripts/.

Written as a builder so the notebook and the local modules never drift apart:
the code blocks below are the code that produced every number in the report.
"""

import json
from pathlib import Path

NB = "notebook/xg_logistic_regression.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


# ---------------------------------------------------------------------------

md(r"""
# Expected Goals from Scratch: Logistic Regression, and What Log Loss Does Not Measure

**31005 Machine Learning, A2 — Sean Woods (26107565), Spring 2026**

This notebook builds an expected goals (xG) model for football. Every part of the
learner is written by hand: the sigmoid, the log loss, the gradient, gradient
descent, and every evaluation metric. `scikit-learn` appears once, in a
verification cell, purely as a reference to check my arithmetic against.

## The task

Given one shot, output the probability that it becomes a goal.

**Training phase**
- Input: a matrix `X` of shape `(n, d)`, one row per historical shot, and a
  vector `y` in `{0,1}^n` where 1 means the shot was scored.
- Hyperparameters: learning rate, iteration budget, L2 strength, random seed.
- Output: a weight vector `w` in `R^(d+1)`, the bias sitting in `w[0]`.

**Deployment phase**
- Input: one shot's feature vector `x` in `R^d`, built with the same feature
  code and scaled with the *training* mean and standard deviation.
- Output: a single number in `[0, 1]`. Not a label. The probability is the
  product. Nobody wants a model that says "this shot is a goal"; they want to
  know that it was worth 0.31 of one.

## The research question

An xG model cannot know whether a particular shot goes in. Two identical
chances end differently and neither outcome is a mistake. What the model is
actually for is **calibration**: of all the chances it rates 0.30, close to 30%
should be scored. But it is trained by minimising **log loss on individual
binary outcomes**, which is not the same thing.

So: where does that gap bite, how do you detect it, and when you add features,
are you buying *calibration* (honesty) or *sharpness* (confidence)?
""")

md(r"""
## 1. Setup

Only NumPy, pandas and matplotlib are needed to build and train the model.
""")

code(r"""
import json, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG_SEED = 7
plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})
print("numpy", np.__version__, "| pandas", pd.__version__)
""")

md(r"""
## 2. Data

[StatsBomb Open Data](https://github.com/statsbomb/open-data): real event data,
free to use, recorded by human analysts. I take six men's senior international
tournaments, which keeps the population coherent (same level, same era, same
rules) rather than mixing men's and women's football or 1970s and 2020s matches.

Every shot in these competitions carries a **360 freeze frame**: the position of
every player visible at the moment the ball is struck. That is what lets me
count the defenders actually standing in the way, instead of guessing from the
shot location alone.

Penalties are removed. A penalty is the same shot every time, it is scored about
76% of the time, and leaving them in would let the model earn easy log loss for
recognising "the ball is on the spot" rather than learning anything about open
play.

The cell below downloads and parses roughly 314 matches. It takes a few minutes.
""")

code(Path("scripts/build_dataset.py").read_text().split('if __name__')[0].replace(
    'import csv\nimport json\nimport sys\nimport urllib.request\nfrom concurrent.futures import ThreadPoolExecutor\n\n', '')
     .replace('import csv\n', '')
     .replace('    except Exception as exc:  # a handful of matches 404 in the open data\n        print(f"  skipped {mid}: {exc}", file=sys.stderr)\n        return []',
              '    except Exception as exc:  # a handful of matches time out\n        print(f"  skipped {mid}: {exc}")\n        return []'))

code(r"""
def build_dataframe():
    matches = []
    for comp_id, season_id, label in COMPS:
        got = fetch(f"{BASE}/matches/{comp_id}/{season_id}.json")
        for m in got:
            m["_label"] = label
        matches += got
        print(f"{label}: {len(got)} matches")

    print(f"downloading events for {len(matches)} matches ...")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for batch in pool.map(shots_from_match, matches):
            rows += batch
    return pd.DataFrame(rows, columns=FIELDS)


shots = build_dataframe()

# The extraction writes "" where a field is absent (two shots have no keeper in
# the freeze frame). Reading through a CSV would coerce those to NaN for free,
# but building the frame directly does not, so do it explicitly. Skipping this
# leaves object-dtype columns that blow up later in .to_numpy(float).
NUMERIC = ["minute", "period", "x", "y", "first_time", "under_pressure",
           "open_goal", "follows_dribble", "n_opponents_in_frame",
           "defenders_in_cone", "keeper_x", "keeper_y", "statsbomb_xg", "goal"]
for c in NUMERIC:
    shots[c] = pd.to_numeric(shots[c], errors="coerce")

shots = shots.dropna(subset=["x", "y", "goal"]).reset_index(drop=True)
print(f"\n{len(shots)} shots, {int(shots.goal.sum())} goals "
      f"({shots.goal.mean():.1%}), {shots.match_id.nunique()} matches")
shots.head()
""")

code(r"""
# A first look at the class imbalance, which drives most of the decisions later.
print(shots.groupby("competition").goal.agg(["count", "mean"]).round(3))
print("\nbody part:\n", shots.body_part.value_counts())
print("\nbaseline: always predicting 'no goal' would be right "
      f"{1 - shots.goal.mean():.1%} of the time")
""")

md(r"""
## 3. Features

Two ideas do most of the work, and both are geometry.

**Distance** to the centre of the goal, and **angle**, the angle the goalmouth
subtends from where the ball is. Angle matters independently of distance: a shot
from the byline six yards out is close to the goal but can barely see any of it,
while the same distance straight on sees the whole face. Distance alone cannot
express that.

The angle is computed from the two posts. With `v1` and `v2` the vectors from
the ball to each post,

$$\theta = \arccos\left(\frac{v_1 \cdot v_2}{\|v_1\| \, \|v_2\|}\right)$$

Then three further blocks, added one at a time so the results table is a
controlled experiment rather than one big jump:

- **situation**: body part, first-time or not, under pressure, open goal, free
  kick, what kind of pass created it, what phase of play it came from
- **freeze frame**: defenders standing inside the triangle between the ball and
  the two posts, opponents in shot, how far the keeper is off his line and off
  his centre

Categorical variables are one-hot encoded with one level dropped, otherwise the
columns sum to a constant and the design matrix is rank deficient.
""")

src = Path("scripts/xg.py").read_text()
start = src.index("# StatsBomb pitch")
end = src.index("# ----------------------------------------------------------------------------\n# The model")
code(src[start:end].strip())

md(r"""
## 4. The model

Three design choices, in the language of the subject.

**H — the hypothesis space.** Every function of the form

$$h_{w}(x) = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Choosing weights picks one member out of that infinite family. The sigmoid is
not decoration: it is what makes the output a probability, squashing any real
number into `(0, 1)` and never reaching either end, which matters because no
shot is ever truly impossible or certain.

**L — the criterion.** Binary cross-entropy, also called log loss:

$$L(w) = \frac{1}{n}\sum_{i=1}^{n} -\big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] + \lambda\|w\|^2$$

Two reasons for it over squared error. First, it is a *proper scoring rule*: it
is minimised, in expectation, exactly when the predicted probability equals the
true probability, which is precisely what an xG model is supposed to produce.
Second, its gradient does not vanish when the model is confidently wrong. I test
that claim in section 9 by training the same model on squared error instead.

**A — the algorithm.** Full-batch gradient descent. With the bias folded into
the design matrix as a column of ones,

$$\nabla L = \frac{1}{n} X^\top (\sigma(Xw) - y) + 2\lambda w$$

The derivation is worth doing once, because the sigmoid's derivative and the
log's derivative cancel:

$$\frac{\partial}{\partial w_j} \Big[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))\Big]
= (\sigma(z) - y)\,x_j$$

The whole gradient is "how wrong you were, times the input". 7,451 shots is
small, so full batch is fine and the loss curve comes out smooth, with no
mini-batch noise to explain away.
""")

start = src.index("def sigmoid(z):")
end = src.index("# ----------------------------------------------------------------------------\n# Splitting")
code(src[start:end].strip())

md(r"""
### 4b. Does the gradient actually match the loss?

A hand-derived gradient is the easiest thing in this project to get quietly
wrong, and a wrong gradient still trains, just badly. Central differences give
an independent check:

$$\frac{\partial L}{\partial w_j} \approx \frac{L(w + he_j) - L(w - he_j)}{2h}$$

Agreement to around 1e-10 means the calculus and the code say the same thing.
""")

code(r"""
def check_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (rng.random(60) < 0.3).astype(float)
    m = LogisticRegressionGD(l2=0.05)
    Xb = m._add_bias(X)
    w = rng.normal(0, 0.5, Xb.shape[1])

    analytic = m.gradient(Xb, y, w)
    numeric = np.zeros_like(w)
    h = 1e-6
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        numeric[i] = (m.loss(Xb, y, wp) - m.loss(Xb, y, wm)) / (2 * h)

    rel = np.abs(analytic - numeric).max() / np.abs(numeric).max()
    print(f"max relative error between analytic and numerical gradient: {rel:.2e}")
    assert rel < 1e-6
    return rel


check_gradient()
""")

md(r"""
## 5. Validation scheme: split by match, never by shot

Shots from the same match share a goalkeeper, a pitch, a scoreline and a
defensive shape. They are not independent draws from the population. If some
shots from a match sit in training and others in test, the test set is no longer
fresh, and the generalisation estimate it gives is optimistic. So whole matches
move together.

- **Train** 60% of matches, used to fit weights.
- **Validation** 20%, used to watch convergence and to sanity-check choices.
- **Test** 20%, touched once, at the end.

Hyperparameters are chosen by **5-fold cross-validation inside the training
matches**, again splitting by match. Reporting the standard deviation across
folds matters: a difference smaller than the fold-to-fold spread is noise, not a
result.
""")

start = src.index("def split_by_match(")
end = src.index("# ----------------------------------------------------------------------------\n# Metrics")
code(src[start:end].strip())

code(r"""
shots["goal"] = shots.goal.astype(float)
parts = split_by_match(shots, seed=RNG_SEED)
tr, va, te = parts["train"], parts["val"], parts["test"]
y_tr, y_va, y_te = (p.goal.to_numpy(float) for p in (tr, va, te))

print(f"shots    train {len(tr):5d}   val {len(va):5d}   test {len(te):5d}")
print(f"matches  train {tr.match_id.nunique():5d}   val {va.match_id.nunique():5d}"
      f"   test {te.match_id.nunique():5d}")
print(f"goal rate train {y_tr.mean():.3f}   val {y_va.mean():.3f}   test {y_te.mean():.3f}")
assert not set(tr.match_id) & set(te.match_id), "match appears in two splits"
""")

md(r"""
## 6. Metrics, written out

Each one answers a different question, and one of them answers none.

| Metric | Question it answers |
|---|---|
| **Log loss** | How surprised was the model by what happened? The training criterion. |
| **Brier score** | Mean squared error of the probabilities. Decomposes usefully. |
| **Reliability** | Are the stated probabilities honest? Lower is better, 0 is perfect. |
| **Resolution** | Does the model separate good chances from bad? Higher is better. |
| **ECE** | Average size of the calibration miss, in probability points. |
| **ROC-AUC** | Chance a random goal is rated above a random miss. Ranking only. |
| **Average precision** | Same idea, but sensitive to the rare positive class. |
| **Accuracy** | Nothing useful here. Shown to prove it. |

Murphy's decomposition is the one that answers the research question directly:

$$\text{Brier} = \underbrace{\text{reliability}}_{\text{honesty}} - \underbrace{\text{resolution}}_{\text{sharpness}} + \underbrace{\text{uncertainty}}_{\text{fixed by the data}}$$
""")

start = src.index("def log_loss(")
end = src.index("# ----------------------------------------------------------------------------\n# Convenience")
code(src[start:end].strip())

start = src.index("def prepare(")
end = src.index("def load(")
code(src[start:end].strip())

md(r"""
### 6b. Checking the metrics against a reference

This is the only cell in the notebook that imports scikit-learn, and it does not
touch the model. It exists so that "I wrote my own log loss" is a verified claim
rather than a hopeful one.
""")

code(r"""
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss as sk_log_loss, roc_auc_score)

rng = np.random.default_rng(1)
p_fake = rng.random(500)
y_fake = (rng.random(500) < p_fake).astype(float)

for name, mine, theirs in [
    ("log loss", log_loss(y_fake, p_fake), sk_log_loss(y_fake, p_fake)),
    ("brier", brier(y_fake, p_fake), brier_score_loss(y_fake, p_fake)),
    ("roc auc", roc_auc(y_fake, p_fake), roc_auc_score(y_fake, p_fake)),
    ("avg precision", average_precision(y_fake, p_fake),
     average_precision_score(y_fake, p_fake)),
]:
    print(f"{name:<15} mine {mine:.8f}   sklearn {theirs:.8f}   "
          f"{'ok' if abs(mine - theirs) < 1e-6 else 'MISMATCH'}")
""")

md(r"""
## 7. Training, with hyperparameters chosen by cross-validation

Four nested models, each adding one block of features to the one before:

- **M1** distance
- **M2** distance + angle
- **M3** + situation (body part, assist type, phase of play, pressure, …)
- **M4** + freeze frame (defenders in the cone, keeper position)

plus **M0**, a baseline that predicts the training base rate for every shot. Any
model that cannot beat M0 has learned nothing at all.
""")

code(r"""
FEATURE_SETS = ["distance", "geometry", "context", "full"]
LABELS = {"distance": "M1 distance only", "geometry": "M2 distance + angle",
          "context": "M3 + situation", "full": "M4 + freeze frame"}
L2_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
BINS = 10


def cross_validate(train_df, feature_set, l2, k=5):
    scores = []
    for fit_df, held_df in match_folds(train_df, k=k, seed=RNG_SEED):
        (X_fit, X_held), _, _ = prepare(fit_df, [held_df], feature_set)
        m = LogisticRegressionGD(lr=0.5, n_iter=8000, l2=l2).fit(
            X_fit, fit_df.goal.to_numpy(float))
        scores.append(log_loss(held_df.goal.to_numpy(float), m.predict_proba(X_held)))
    return float(np.mean(scores)), float(np.std(scores))


def tune(train_df, feature_set):
    rows = []
    for l2 in L2_GRID:
        mean, sd = cross_validate(train_df, feature_set, l2)
        rows.append({"l2": l2, "cv_log_loss": mean, "cv_sd": sd})
        print(f"   l2={l2:<8g} cv log loss {mean:.5f} (+/- {sd:.5f})")
    best = min(rows, key=lambda r: r["cv_log_loss"])
    print(f"   chosen l2 = {best['l2']}")
    return best["l2"], pd.DataFrame(rows)
""")

code(r"""
preds, models, results, cv_tables = {}, {}, [], {}

base = np.full(len(te), y_tr.mean())
preds["M0 base rate"] = base
results.append({"model": "M0 base rate", **evaluate(y_te, base, BINS)})

for fs in FEATURE_SETS:
    print(f"[{LABELS[fs]}]")
    l2, cv_tables[fs] = tune(tr, fs)
    (X_tr, X_va, X_te), names, _ = prepare(tr, [va, te], fs)
    m = LogisticRegressionGD(lr=0.5, n_iter=20000, l2=l2).fit(X_tr, y_tr, X_va, y_va)
    p_te = m.predict_proba(X_te)
    preds[LABELS[fs]] = p_te
    models[fs] = (m, names, l2)
    results.append({"model": LABELS[fs], "l2": l2, "n_features": len(names),
                    "iters": m.n_iter_run_, **evaluate(y_te, p_te, BINS)})
    print(f"   test log loss {results[-1]['log_loss']:.5f}   "
          f"brier {results[-1]['brier']:.5f}   auc {results[-1]['roc_auc']:.4f}\n")
""")

md(r"""
### 7b. A commercial benchmark

StatsBomb ship their own xG value for every shot, produced by a gradient-boosted
model trained on millions of shots with features I do not have (a full
biomechanical view of the shot, goalkeeper positioning models, shot impact
height). Comparing against it turns "my log loss is 0.257" into a number that
means something.
""")

code(r"""
sb = te.statsbomb_xg.to_numpy(float)
preds["StatsBomb xG"] = sb
results.append({"model": "StatsBomb xG", **evaluate(y_te, sb, BINS)})
print(f"StatsBomb xG on the same test shots: log loss {results[-1]['log_loss']:.5f}")
""")

md(r"""
## 8. Results
""")

code(r"""
res = pd.DataFrame(results).set_index("model")
cols = ["log_loss", "brier", "ece", "reliability", "resolution",
        "roc_auc", "avg_precision", "accuracy", "mean_pred"]
res[cols].round(4)
""")

code(r"""
m_full, names_full, l2_full = models["full"]
coef = (pd.DataFrame({"feature": ["bias"] + names_full, "weight": m_full.w_})
        .assign(abs_w=lambda d: d.weight.abs())
        .sort_values("abs_w", ascending=False)
        .drop(columns="abs_w"))
print("M4 weights, standardised units, largest effect first")
coef.head(15).round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for fs in FEATURE_SETS:
    h = np.array(models[fs][0].history_)
    ax.plot(h[:, 0], h[:, 1], lw=1.2, label=f"{LABELS[fs]} (train)")
    ax.plot(h[:, 0], h[:, 2], lw=1.0, ls="--", alpha=0.7, label=f"{LABELS[fs]} (val)")
ax.set_xscale("log")
ax.set_xlabel("gradient descent iteration")
ax.set_ylabel("mean log loss")
ax.set_title("Convergence, and the gap between train and validation")
ax.legend(fontsize=6, ncol=2)
plt.show()
""")

md(r"""
The validation curve sitting almost on top of the training curve is the
generalisation gap, drawn. It is small because H is small: logistic regression
on 35 standardised features simply does not have the capacity to memorise 4,444
shots. A deeper model would pull those two lines apart, and the distance between
them would be the price of that extra capacity.
""")

code(r"""
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot([0, 0.45], [0, 0.45], color="0.4", lw=1, ls=":", label="perfect calibration")
for name in ["M1 distance only", "M2 distance + angle", "M4 + freeze frame",
             "StatsBomb xG"]:
    t = calibration_table(y_te, preds[name], bins=BINS)
    ax.plot(t.mean_pred, t.observed, marker="o", ms=3.5, lw=1.2, label=name)
ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45)
ax.set_xlabel("predicted probability (bin mean)")
ax.set_ylabel("observed goal rate")
ax.set_title("Reliability diagram, test matches")
ax.legend(fontsize=7)
plt.show()
""")

code(r"""
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for name in ["M1 distance only", "M2 distance + angle", "M4 + freeze frame"]:
    ax.hist(preds[name], bins=40, histtype="step", lw=1.3, label=name)
ax.axvline(y_te.mean(), color="0.4", ls=":", lw=1, label="base rate")
ax.set_yscale("log")
ax.set_xlabel("predicted probability")
ax.set_ylabel("shots (log scale)")
ax.set_title("Sharpness: features push predictions away from the base rate")
ax.legend(fontsize=7)
plt.show()
""")

code(r"""
# What M2 has learned about geometry, drawn over the attacking third.
m2 = models["geometry"][0]
gx, gy = np.meshgrid(np.linspace(80, 120, 160), np.linspace(10, 70, 160))
grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
(_, X_grid), _, _ = prepare(tr, [grid], "geometry")
surface = m2.predict_proba(X_grid).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(5.6, 4.2))
im = ax.contourf(gx, gy, surface, levels=14, cmap="viridis")
ax.plot([120, 120], [36, 44], color="white", lw=3)
ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, color="white", lw=1))
ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, color="white", lw=1))
ax.set_title("M2 predicted goal probability by shot location")
ax.set_xlabel("x (yards)"); ax.set_ylabel("y (yards)")
fig.colorbar(im, ax=ax, label="predicted P(goal)")
plt.show()
""")

md(r"""
## 9. Does the criterion matter? Swapping log loss for squared error

Same hypothesis space, same optimiser, same features, same L2. Only `L` changes,
to

$$L(w) = \frac{1}{n}\sum_i (\sigma(z_i) - y_i)^2$$

The gradient now carries an extra factor:

$$\nabla L = \frac{2}{n} X^\top \big[(p - y)\,p\,(1-p)\big]$$

That `p(1-p)` term is the sigmoid's own derivative, which log loss cancels and
squared error does not. When the model is confidently wrong (`p` near 1 while
`y = 0`), `p(1-p)` is near zero, so the gradient nearly vanishes exactly where
the error is largest. The update stalls when it should be loudest.
""")

code(r"""
class SquaredLossLogistic(LogisticRegressionGD):
    def loss(self, Xb, y, w):
        p = sigmoid(Xb @ w)
        return float(np.mean((p - y) ** 2) + self.l2 * np.sum(w[1:] ** 2))

    def gradient(self, Xb, y, w):
        p = sigmoid(Xb @ w)
        g = Xb.T @ (2.0 * (p - y) * p * (1.0 - p)) / len(y)
        reg = 2.0 * self.l2 * w
        reg[0] = 0.0
        return g + reg


(X_tr, X_te), _, _ = prepare(tr, [te], "full")
sq = SquaredLossLogistic(lr=0.5, n_iter=20000, l2=l2_full).fit(X_tr, y_tr)
preds["M4 trained on squared error"] = sq.predict_proba(X_te)
results.append({"model": "M4 trained on squared error",
                **evaluate(y_te, preds["M4 trained on squared error"], BINS)})

pd.DataFrame(results).set_index("model").loc[
    ["M4 + freeze frame", "M4 trained on squared error"],
    ["log_loss", "brier", "ece", "reliability", "resolution", "roc_auc", "mean_pred"]
].round(4)
""")

md(r"""
## 10. Where the loss stops serving the objective

The headline calibration number is an average over every shot, and an average
can be right while every part of it is wrong. Log loss has no idea that
"counter attack" and "20-yard shot" are different situations a manager would
want treated separately. It only sees rows.

So: group the test shots by football situation, and compare what the model
promised with what happened.
""")

code(r"""
def subgroup_table(te, y_te, p_full, p_geom):
    dist, _ = geometry(te)
    groups = {
        "all shots": np.ones(len(te), bool),
        "headers": (te.body_part == "Head").to_numpy(),
        "feet": te.body_part.isin(["Left Foot", "Right Foot"]).to_numpy(),
        "free kicks": (te.shot_type == "Free Kick").to_numpy(),
        "inside 12 yds": dist <= 12,
        "12-20 yds": (dist > 12) & (dist <= 20),
        "beyond 20 yds": dist > 20,
        "from a cross": (te.assist_type == "Cross").to_numpy(),
        "counter attack": (te.play_pattern == "From Counter").to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        if mask.sum() < 30:
            continue
        rows.append({"group": name, "n": int(mask.sum()),
                     "goals": int(y_te[mask].sum()),
                     "observed_rate": float(y_te[mask].mean()),
                     "M2_mean_xg": float(p_geom[mask].mean()),
                     "M4_mean_xg": float(p_full[mask].mean()),
                     "M4_bias": float(p_full[mask].mean() - y_te[mask].mean())})
    return pd.DataFrame(rows)


sub = subgroup_table(te, y_te, preds["M4 + freeze frame"], preds["M2 distance + angle"])
sub.round(4)
""")

code(r"""
t = sub[sub.group != "all shots"]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
y = np.arange(len(t))
ax.barh(y, t.M4_bias, color=np.where(t.M4_bias > 0, "#c0504d", "#4f81bd"))
ax.set_yticks(y, t.group, fontsize=8)
ax.axvline(0, color="0.3", lw=1)
ax.set_xlabel("mean predicted xG minus observed goal rate")
ax.set_title("Calibrated on average, biased in every subgroup")
plt.show()
""")

md(r"""
## 11. Discussion

**What the numbers say.** Each block of features improves log loss, and M4 lands
within about 0.010 of StatsBomb's commercial model on the same shots. The
ranking metrics agree: ROC-AUC 0.81 against their 0.83.

**Accuracy is meaningless here and the table proves it.** Predicting "no goal"
for every shot scores about 90.3%. The best model scores 91.1%. A metric where
refusing to model anything captures 99% of the achievable score is not measuring
the thing the model is for. This is the class imbalance (roughly one goal per
eleven shots) doing what class imbalance always does.

**Features bought sharpness, not honesty.** Reliability barely moved across M1
to M4 while resolution climbed steadily. The reason is structural: with a bias
term in the model and a criterion that is a proper scoring rule, gradient
descent will drive the *mean* prediction to the base rate almost regardless of
what the other features are doing. Average calibration is close to free. Telling
a 0.4 chance apart from a 0.05 chance is the part you actually pay for.

**Where the criterion and the objective part company.** The subgroup table is
the answer to the research question. M4's overall bias is +0.0001, which looks
flawless, and yet it underrates counter attacks by about 6 percentage points and
shots from crosses by about 4, while overrating shots from beyond 20 yards by
about 2. Log loss is a sum over individual rows, so a systematic underestimate
in one situation and a matching overestimate in another cancel perfectly in the
objective while both being wrong in the only way a coach would care about.

That is a real cost, not a theoretical one. A team that presses high and scores
on transitions would be told its chances are worth less than they are, because
the model has no feature describing how disorganised the defence was. The
freeze-frame count of defenders in the cone is a static snapshot, and it cannot
tell a settled back four from four players sprinting back.

**How you detect it.** Not from the loss, which is exactly the point. You detect
it by grouping the residuals along dimensions the model never saw and testing
whether the bias in each group is bigger than sampling noise would allow.

**Limitations.**
- The hypothesis space is linear in the features, so every interaction has to be
  hand-built. A cross to the back post and a cross to the six-yard box are
  treated as the same kind of event.
- Roughly 7,400 shots with 660 goals is not many. The 55 counter-attack shots in
  the test set make that row suggestive, not conclusive.
- International tournament football is its own population. A model fitted here
  should not be deployed on youth or amateur matches without refitting.
- Shot events are recorded by human analysts, so "under pressure" and "first
  time" carry some labelling noise.

**What I would do next.**
1. Add a group-wise calibration penalty to the criterion, so the training
   objective contains the thing the evaluation actually measures instead of
   hoping the average carries it.
2. Add interaction terms between angle, distance and defender count, which is
   the cheapest way to buy the flexibility a tree model would get for free.
3. Fit the model per competition and compare the weights, to see whether "good
   chance" means the same thing in the AFCON as it does at a World Cup.
""")

# ---------------------------------------------------------------------------

nb = {
    "cells": [{**c, "source": (c["source"] + "\n").splitlines(keepends=True)} for c in cells],
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

Path("notebook").mkdir(exist_ok=True)
Path(NB).write_text(json.dumps(nb, indent=1))
print(f"wrote {NB}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code)")
