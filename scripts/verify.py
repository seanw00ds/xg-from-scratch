"""Checks that the from-scratch code is actually correct.

Three kinds of check:
  1. the analytic gradient against a numerical one (does calculus match code)
  2. my metrics against scikit-learn's (does my arithmetic match a reference)
  3. my fitted weights against sklearn's LogisticRegression (does gradient
     descent land where a proper solver lands)

sklearn is used ONLY here, as a reference. The model in xg.py never imports it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss as sk_log_loss, roc_auc_score)

import xg


def check_gradient():
    """Central differences: (L(w+h) - L(w-h)) / 2h should equal the analytic
    gradient to about 1e-8. If the derivation is wrong this blows up."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (rng.random(60) < 0.3).astype(float)
    m = xg.LogisticRegressionGD(l2=0.05)
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

    rel = np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-12)
    print(f"gradient check      max relative error {rel:.2e}")
    assert rel < 1e-6, "analytic gradient does not match the numerical one"


def check_metrics():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    pairs = {
        "log_loss": (xg.log_loss(y, p), sk_log_loss(y, p)),
        "brier": (xg.brier(y, p), brier_score_loss(y, p)),
        "roc_auc": (xg.roc_auc(y, p), roc_auc_score(y, p)),
        "avg_precision": (xg.average_precision(y, p), average_precision_score(y, p)),
    }
    for name, (mine, theirs) in pairs.items():
        print(f"{name:<20}mine {mine:.6f}   sklearn {theirs:.6f}")
        assert abs(mine - theirs) < 1e-6, f"{name} disagrees with sklearn"


def check_fit():
    """Same data, same objective, different optimiser. sklearn's lbfgs
    minimises sum of CE + (1/2C)||w||^2, mine minimises mean CE + l2||w||^2,
    so C is set to line the two penalties up."""
    df = xg.load()
    parts = xg.split_by_match(df)
    (X_tr, X_te), _, _ = xg.prepare(parts["train"], [parts["test"]], "context")
    y_tr = parts["train"].goal.to_numpy(float)
    y_te = parts["test"].goal.to_numpy(float)

    l2 = 1e-3
    mine = xg.LogisticRegressionGD(lr=0.5, n_iter=20000, l2=l2, tol=1e-12).fit(X_tr, y_tr)
    ref = LogisticRegression(C=1.0 / (2 * l2 * len(y_tr)), max_iter=5000)
    ref.fit(X_tr, y_tr)

    ref_w = np.r_[ref.intercept_, ref.coef_[0]]
    gap = np.abs(mine.w_ - ref_w).max()
    print(f"weights             max abs difference {gap:.4f}")
    print(f"test log loss       mine {xg.log_loss(y_te, mine.predict_proba(X_te)):.5f}"
          f"   sklearn {xg.log_loss(y_te, ref.predict_proba(X_te)[:, 1]):.5f}")
    assert gap < 0.05, "gradient descent has not converged to the same solution"


if __name__ == "__main__":
    check_gradient()
    check_metrics()
    check_fit()
    print("\nall checks passed")
