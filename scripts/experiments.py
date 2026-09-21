"""Runs every experiment in the report and writes the tables and figures.

The structure mirrors the research question: does adding features improve
calibration (honest probabilities) or only sharpness (confident probabilities),
and where does minimising log loss stop serving the actual xG objective?
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xg

OUT = "results"
BINS = 10
FEATURE_SETS = ["distance", "geometry", "context", "full"]
LABELS = {
    "distance": "M1 distance only",
    "geometry": "M2 distance + angle",
    "context": "M3 + situation",
    "full": "M4 + freeze frame",
}
L2_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


# ----------------------------------------------------------------------------

def cross_validate(train_df, feature_set, l2, k=5):
    """5-fold CV with whole matches held out, returning mean and sd of log
    loss. The sd matters: differences smaller than it are noise."""
    scores = []
    for fit_df, held_df in xg.match_folds(train_df, k=k):
        (X_fit, X_held), _, _ = xg.prepare(fit_df, [held_df], feature_set)
        y_fit = fit_df.goal.to_numpy(float)
        y_held = held_df.goal.to_numpy(float)
        m = xg.LogisticRegressionGD(lr=0.5, n_iter=8000, l2=l2).fit(X_fit, y_fit)
        scores.append(xg.log_loss(y_held, m.predict_proba(X_held)))
    return float(np.mean(scores)), float(np.std(scores))


def tune(train_df, feature_set):
    rows = []
    for l2 in L2_GRID:
        mean, sd = cross_validate(train_df, feature_set, l2)
        rows.append({"feature_set": feature_set, "l2": l2, "cv_log_loss": mean, "cv_sd": sd})
        print(f"  l2={l2:<8g} cv log loss {mean:.5f} (+/- {sd:.5f})")
    best = min(rows, key=lambda r: r["cv_log_loss"])
    print(f"  chosen l2 = {best['l2']}")
    return best["l2"], pd.DataFrame(rows)


# ----------------------------------------------------------------------------

def main():
    df = xg.load()
    parts = xg.split_by_match(df)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    y_tr, y_va, y_te = (p.goal.to_numpy(float) for p in (tr, va, te))

    print(f"shots  train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"matches train {tr.match_id.nunique()}  val {va.match_id.nunique()}"
          f"  test {te.match_id.nunique()}")
    print(f"goal rate  train {y_tr.mean():.3f}  val {y_va.mean():.3f}  test {y_te.mean():.3f}\n")

    preds, models, results, cv_tables = {}, {}, [], []

    # Baseline 0: predict the training base rate for every shot. Any model that
    # cannot beat this has learned nothing.
    base = np.full(len(te), y_tr.mean())
    preds["M0 base rate"] = base
    results.append({"model": "M0 base rate", **xg.evaluate(y_te, base, BINS)})

    for fs in FEATURE_SETS:
        print(f"[{LABELS[fs]}]")
        l2, cv_table = tune(tr, fs)
        cv_tables.append(cv_table)

        (X_tr, X_va, X_te), names, _ = xg.prepare(tr, [va, te], fs)
        m = xg.LogisticRegressionGD(lr=0.5, n_iter=20000, l2=l2).fit(
            X_tr, y_tr, X_va, y_va)
        p_te = m.predict_proba(X_te)
        preds[LABELS[fs]] = p_te
        models[fs] = (m, names, l2)
        results.append({"model": LABELS[fs], "l2": l2, "n_features": len(names),
                        "iters": m.n_iter_run_, **xg.evaluate(y_te, p_te, BINS)})
        print(f"  test log loss {results[-1]['log_loss']:.5f}"
              f"   brier {results[-1]['brier']:.5f}"
              f"   auc {results[-1]['roc_auc']:.4f}\n")

    # Benchmark: StatsBomb's own commercial xG on the same test shots.
    sb = te.statsbomb_xg.to_numpy(float)
    preds["StatsBomb xG"] = sb
    results.append({"model": "StatsBomb xG", **xg.evaluate(y_te, sb, BINS)})

    # The criterion experiment: same hypothesis space, same optimiser, a
    # different loss. Squared error on the sigmoid output instead of log loss.
    (X_tr, X_te), names, _ = xg.prepare(tr, [te], "full")
    sq = SquaredLossLogistic(lr=0.5, n_iter=20000, l2=models["full"][2]).fit(X_tr, y_tr)
    p_sq = sq.predict_proba(X_te)
    preds["M4 trained on squared error"] = p_sq
    results.append({"model": "M4 trained on squared error", **xg.evaluate(y_te, p_sq, BINS)})

    res = pd.DataFrame(results)
    res.to_csv(f"{OUT}/metrics.csv", index=False)
    pd.concat(cv_tables).to_csv(f"{OUT}/cv_grid.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Coefficients of the final model, in original units where it helps.
    m_full, names_full, _ = models["full"]
    coef = pd.DataFrame({"feature": ["bias"] + names_full, "weight": m_full.w_})
    coef.to_csv(f"{OUT}/coefficients.csv", index=False)

    subgroup_table(te, y_te, preds["M4 + freeze frame"], preds["M2 distance + angle"])
    figures(tr, te, y_te, preds, models)
    json.dump({"n_shots": int(len(df)), "n_matches": int(df.match_id.nunique()),
               "goal_rate": float(df.goal.mean())},
              open(f"{OUT}/dataset.json", "w"), indent=2)


class SquaredLossLogistic(xg.LogisticRegressionGD):
    """Same hypothesis space, same algorithm, different criterion.

    L = mean (sigmoid(z) - y)^2. The gradient picks up an extra p(1-p) factor
    that log loss cancels away, so a confidently wrong prediction produces an
    almost flat gradient and the model is slow to correct it.
    """

    def loss(self, Xb, y, w):
        p = xg.sigmoid(Xb @ w)
        return float(np.mean((p - y) ** 2) + self.l2 * np.sum(w[1:] ** 2))

    def gradient(self, Xb, y, w):
        p = xg.sigmoid(Xb @ w)
        g = Xb.T @ (2.0 * (p - y) * p * (1.0 - p)) / len(y)
        reg = 2.0 * self.l2 * w
        reg[0] = 0.0
        return g + reg


def subgroup_table(te, y_te, p_full, p_geom):
    """Global calibration can hide local miscalibration. This is the heart of
    the loss-versus-objective argument, so it gets its own table."""
    rows = []
    dist, _ = xg.geometry(te)
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
    for name, mask in groups.items():
        if mask.sum() < 30:
            continue
        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "goals": int(y_te[mask].sum()),
            "observed_rate": float(y_te[mask].mean()),
            "M2_mean_xg": float(p_geom[mask].mean()),
            "M4_mean_xg": float(p_full[mask].mean()),
            "M4_bias": float(p_full[mask].mean() - y_te[mask].mean()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/subgroups.csv", index=False)
    print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return t


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def figures(tr, te, y_te, preds, models):
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25})

    # 1. Convergence
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fs in FEATURE_SETS:
        h = np.array(models[fs][0].history_)
        ax.plot(h[:, 0], h[:, 1], label=f"{LABELS[fs]} (train)", lw=1.2)
        ax.plot(h[:, 0], h[:, 2], ls="--", lw=1.0, alpha=0.7,
                label=f"{LABELS[fs]} (val)")
    ax.set_xscale("log")
    ax.set_xlabel("gradient descent iteration")
    ax.set_ylabel("mean log loss")
    ax.set_title("Convergence, and the gap between train and validation")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_convergence.png")
    plt.close(fig)

    # 2. Reliability diagram
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 0.45], [0, 0.45], color="0.4", lw=1, ls=":", label="perfect calibration")
    for name in ["M1 distance only", "M2 distance + angle", "M4 + freeze frame",
                 "StatsBomb xG", "M4 trained on squared error"]:
        t = xg.calibration_table(y_te, preds[name], bins=BINS)
        ax.plot(t.mean_pred, t.observed, marker="o", ms=3.5, lw=1.2, label=name)
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed goal rate")
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.45)
    ax.set_title("Reliability diagram, test matches")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_calibration.png")
    plt.close(fig)

    # 3. Sharpness: how spread out the predictions are
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name in ["M1 distance only", "M2 distance + angle", "M4 + freeze frame"]:
        ax.hist(preds[name], bins=40, histtype="step", lw=1.3, label=name)
    ax.axvline(y_te.mean(), color="0.4", ls=":", lw=1, label="base rate")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("shots (log scale)")
    ax.set_title("Sharpness: adding features spreads predictions away from the base rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_sharpness.png")
    plt.close(fig)

    # 4. The pitch map: what the model has actually learned about geometry
    m, names, _ = models["geometry"]
    gx, gy = np.meshgrid(np.linspace(80, 120, 160), np.linspace(10, 70, 160))
    grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    (X_tr, X_grid), _, _ = xg.prepare(tr, [grid], "geometry")
    surface = m.predict_proba(X_grid).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    im = ax.contourf(gx, gy, surface, levels=14, cmap="viridis")
    ax.plot([120, 120], [36, 44], color="white", lw=3)
    ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, color="white", lw=1))
    ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, color="white", lw=1))
    ax.set_title("M2 predicted goal probability by shot location")
    ax.set_xlabel("x (yds)")
    ax.set_ylabel("y (yds)")
    fig.colorbar(im, ax=ax, label="predicted P(goal)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_pitch_surface.png")
    plt.close(fig)

    # 5. Where the money is: subgroup bias
    t = pd.read_csv(f"{OUT}/subgroups.csv")
    t = t[t.group != "all shots"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    y = np.arange(len(t))
    ax.barh(y, t.M4_bias, color=np.where(t.M4_bias > 0, "#c0504d", "#4f81bd"))
    ax.set_yticks(y, t.group, fontsize=7)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("mean predicted xG minus observed goal rate")
    ax.set_title("Calibration is a global average: it hides these")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_subgroup_bias.png")
    plt.close(fig)
    print(f"\nfigures written to {OUT}/")


if __name__ == "__main__":
    main()
