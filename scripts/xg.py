"""Expected goals by logistic regression, implemented from first principles.

Nothing here imports a model. NumPy does the linear algebra, everything else
(sigmoid, log loss, gradient, gradient descent, every metric) is written out.
scikit-learn appears only in tests/verify.py, to check my numbers match a
reference implementation.
"""

import numpy as np
import pandas as pd

# StatsBomb pitch: 120 x 80, attacking goal at x = 120, posts at y = 36 and 44.
GOAL_CENTRE = np.array([120.0, 40.0])
POST_A = np.array([120.0, 36.0])
POST_B = np.array([120.0, 44.0])


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------

def geometry(df):
    """Distance to the middle of the goal, and the angle the goalmouth
    subtends from the ball. A shot from the byline is close but has almost no
    angle, which is exactly why distance alone is not enough."""
    xy = df[["x", "y"]].to_numpy(float)
    dist = np.linalg.norm(xy - GOAL_CENTRE, axis=1)

    v1 = POST_A - xy
    v2 = POST_B - xy
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    angle = np.arccos(np.clip(cos, -1.0, 1.0))  # radians, 0 = no view of goal
    return dist, angle


CATEGORICALS = [("body_part", "Right Foot"),
                ("shot_type", "Open Play"),
                ("assist_type", "None"),
                ("play_pattern", "Regular Play")]


def category_levels(df):
    """Which levels each categorical can take. Taken from the training split
    and then reused, so every split produces the same columns in the same
    order. Letting each split name its own levels silently changes the model's
    input space, which is how I first got a shape mismatch."""
    return {col: sorted(df[col].dropna().unique()) for col, _ in CATEGORICALS}


def build_features(df, feature_set, levels=None):
    """Return a design matrix (no bias column yet) and the column names.

    Feature sets are nested on purpose: each one adds a block to the last, so
    the results table reads as a single controlled experiment.
    """
    levels = levels or category_levels(df)
    dist, angle = geometry(df)
    cols = {"distance": dist, "angle": angle}

    if feature_set == "distance":
        cols = {"distance": dist}

    elif feature_set in ("context", "full"):
        # Binary situation flags, already 0/1 in the CSV.
        for c in ["first_time", "under_pressure", "open_goal", "follows_dribble"]:
            cols[c] = df[c].to_numpy(float)
        # One-hot the categoricals. Drop one level of each to avoid a column
        # that is an exact sum of the others (the dummy variable trap).
        for col, drop in CATEGORICALS:
            for level in levels[col]:
                if level == drop:
                    continue
                key = f"{col}={level}".replace(" ", "_")
                cols[key] = (df[col] == level).to_numpy(float)

    if feature_set == "full":
        # 360 freeze-frame features: who is actually in the way.
        cols["defenders_in_cone"] = df["defenders_in_cone"].to_numpy(float)
        cols["n_opponents_in_frame"] = df["n_opponents_in_frame"].to_numpy(float)
        # How far off his line the keeper is, and how far he has drifted from
        # the middle of the goal. Two shots in 7451 have no keeper in the
        # frame at all, which is itself information (one was scored, one
        # blocked), so rather than drop them I put the keeper on his line in
        # the middle of the goal and let a flag carry the fact he was missing.
        keeper = df[["keeper_x", "keeper_y"]].to_numpy(float)
        missing = np.isnan(keeper[:, 0]) | np.isnan(keeper[:, 1])
        kx = np.where(missing, 120.0, keeper[:, 0])
        ky = np.where(missing, 40.0, keeper[:, 1])
        cols["keeper_off_line"] = 120.0 - kx
        cols["keeper_off_centre"] = np.abs(ky - 40.0)
        cols["keeper_not_in_frame"] = missing.astype(float)

    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    return X, names


class Standardiser:
    """z = (x - mean) / sd, with mean and sd learned from the training split
    only. Fitting these on all the data would leak test information into
    training, and gradient descent needs the scaling or the learning rate that
    suits 'distance' (0-100) is wildly wrong for a 0/1 flag."""

    def fit(self, X):
        self.mean_ = X.mean(0)
        self.sd_ = X.std(0)
        self.sd_[self.sd_ == 0] = 1.0  # constant column: leave it alone
        return self

    def transform(self, X):
        return (X - self.mean_) / self.sd_


# ----------------------------------------------------------------------------
# The model: hypothesis space, criterion, algorithm
# ----------------------------------------------------------------------------

def sigmoid(z):
    """1 / (1 + e^-z), written so a large negative z cannot overflow."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionGD:
    """H : x -> sigmoid(w . x + b), every w and b being one hypothesis.
    L : mean binary cross-entropy + L2 penalty on w (not on b).
    A : full-batch gradient descent.
    """

    def __init__(self, lr=0.5, n_iter=4000, l2=0.0, tol=1e-9, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol
        self.seed = seed

    def _add_bias(self, X):
        return np.column_stack([np.ones(len(X)), X])

    def loss(self, Xb, y, w):
        """Mean cross-entropy computed through logaddexp, which is the stable
        way to write -[y log p + (1-y) log(1-p)]:
            -y*z + log(1 + e^z)  =  -y*z + logaddexp(0, z)
        """
        z = Xb @ w
        ce = np.mean(-y * z + np.logaddexp(0.0, z))
        return ce + self.l2 * np.sum(w[1:] ** 2)

    def gradient(self, Xb, y, w):
        """d/dw of the loss above. The sigmoid and the log cancel, which is why
        this comes out as plain 'error times input'."""
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / len(y)
        reg = 2.0 * self.l2 * w
        reg[0] = 0.0  # never penalise the bias: it sets the base goal rate
        return g + reg

    def fit(self, X, y, X_val=None, y_val=None, track_every=25):
        Xb = self._add_bias(X)
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0, 0.01, Xb.shape[1])
        self.history_ = []
        prev = np.inf

        for i in range(self.n_iter):
            w -= self.lr * self.gradient(Xb, y, w)
            if i % track_every == 0 or i == self.n_iter - 1:
                tr = self.loss(Xb, y, w)
                va = (self.loss(self._add_bias(X_val), y_val, w)
                      if X_val is not None else np.nan)
                self.history_.append((i, tr, va))
                if abs(prev - tr) < self.tol:
                    break
                prev = tr

        self.w_ = w
        self.n_iter_run_ = i + 1
        return self

    def predict_proba(self, X):
        return sigmoid(self._add_bias(X) @ self.w_)


# ----------------------------------------------------------------------------
# Splitting: by match, never by shot
# ----------------------------------------------------------------------------

def split_by_match(df, fracs=(0.6, 0.2, 0.2), seed=7):
    """Two shots from the same match share a keeper, a pitch and a game state,
    so they are not independent draws. Splitting by shot would put near-copies
    on both sides of the fence and flatter the test score."""
    ids = np.array(sorted(df.match_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr = int(fracs[0] * len(ids))
    n_va = int(fracs[1] * len(ids))
    groups = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}
    return {k: df[df.match_id.isin(v)].reset_index(drop=True) for k, v in groups.items()}


def match_folds(df, k=5, seed=7):
    """k-fold cross-validation where whole matches move between folds."""
    ids = np.array(sorted(df.match_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    for fold in np.array_split(ids, k):
        held = df.match_id.isin(fold)
        yield df[~held].reset_index(drop=True), df[held].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Metrics, all written out
# ----------------------------------------------------------------------------

def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p, threshold=0.5):
    return float(np.mean((p >= threshold) == y))


def roc_auc(y, p):
    """Rank version of the Mann-Whitney statistic: the chance a random goal is
    scored higher than a random non-goal. Ties share the average rank."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, p):
    """Area under precision-recall, as the step-wise sum used by sklearn."""
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def calibration_table(y, p, bins=10, strategy="quantile"):
    """Bucket the predictions, then compare what was promised with what
    happened. This is the table the whole project argues about."""
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    if len(edges) < 2:
        # A constant predictor (the base-rate baseline) has no quantiles to
        # cut on, so everything goes in a single bin.
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "observed": float(y[m].mean()),
            "goals": int(y[m].sum()),
        })
    return pd.DataFrame(rows)


def brier_decomposition(y, p, bins=10):
    """Murphy's split: Brier = reliability - resolution + uncertainty.

    reliability  how far the promised rate sits from the observed rate (lower
                 is better, 0 means perfectly calibrated)
    resolution   how far the bins spread away from the base rate (higher is
                 better, it is sharpness that pays off)
    uncertainty  the base rate's own variance, fixed by the data, nothing the
                 model can do about it
    """
    t = calibration_table(y, p, bins=bins)
    n = len(y)
    base = y.mean()
    rel = float((t.n * (t.mean_pred - t.observed) ** 2).sum() / n)
    res = float((t.n * (t.observed - base) ** 2).sum() / n)
    unc = float(base * (1 - base))
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier_check": rel - res + unc}


def expected_calibration_error(y, p, bins=10):
    t = calibration_table(y, p, bins=bins)
    return float((t.n * (t.mean_pred - t.observed).abs()).sum() / len(y))


def evaluate(y, p, bins=10):
    d = brier_decomposition(y, p, bins=bins)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": expected_calibration_error(y, p, bins=bins),
        "reliability": d["reliability"],
        "resolution": d["resolution"],
        "roc_auc": roc_auc(y, p),
        "avg_precision": average_precision(y, p),
        "accuracy": accuracy(y, p),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
    }


# ----------------------------------------------------------------------------
# Convenience wiring
# ----------------------------------------------------------------------------

def prepare(train_df, other_dfs, feature_set):
    """Build features for train, fit the scaler on train, apply it elsewhere."""
    levels = category_levels(train_df)
    X_tr, names = build_features(train_df, feature_set, levels)
    sc = Standardiser().fit(X_tr)
    out = [sc.transform(X_tr)]
    for d in other_dfs:
        X, _ = build_features(d, feature_set, levels)
        out.append(sc.transform(X))
    return out, names, sc


def load(path="data/shots.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["x", "y", "goal"]).reset_index(drop=True)
    return df
