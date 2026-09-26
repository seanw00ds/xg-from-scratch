"""Walk a single real shot through the model, one step at a time.

Run it to see exactly what the numbers in the report mean:
    .venv/bin/python scripts/explain_one_shot.py           # a random goal
    .venv/bin/python scripts/explain_one_shot.py 42        # shot number 42
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import xg


def main(row_index=None):
    df = xg.load()
    parts = xg.split_by_match(df)
    tr, te = parts["train"], parts["test"]

    # Train M2 (distance and angle) so the arithmetic is small enough to follow.
    (X_tr, X_te), names, scaler = xg.prepare(tr, [te], "geometry")
    y_tr = tr.goal.to_numpy(float)
    model = xg.LogisticRegressionGD(lr=0.5, n_iter=20000, l2=1e-3).fit(X_tr, y_tr)

    if row_index is None:
        row_index = int(te.index[te.goal == 1][3])
    shot = te.iloc[row_index]

    print("=" * 66)
    print("THE SHOT")
    print("=" * 66)
    print(f"  {shot.player} ({shot.team}), {shot.competition}, minute {shot.minute}")
    print(f"  struck from x={shot.x}, y={shot.y} on a 120 x 80 pitch")
    print(f"  body part: {shot.body_part}   from: {shot.play_pattern}")
    print(f"  what actually happened: {shot.outcome}  ->  y = {int(shot.goal)}")

    print("\n" + "=" * 66)
    print("STEP 1.  RAW FEATURES  (real units)")
    print("=" * 66)
    raw, _ = xg.build_features(te.iloc[[row_index]], "geometry")
    dist, angle = raw[0]
    print(f"  distance = {dist:6.2f} yards from the centre of the goal")
    print(f"  angle    = {angle:6.3f} radians = {np.degrees(angle):.1f} degrees of goal visible")

    print("\n" + "=" * 66)
    print("STEP 2.  STANDARDISED FEATURES  (how many SDs from the average shot)")
    print("=" * 66)
    print(f"  training average distance = {scaler.mean_[0]:.2f} yds,  sd = {scaler.sd_[0]:.2f}")
    print(f"  training average angle    = {scaler.mean_[1]:.3f} rad,  sd = {scaler.sd_[1]:.3f}")
    z_feat = scaler.transform(raw)[0]
    print(f"\n  distance: ({dist:.2f} - {scaler.mean_[0]:.2f}) / {scaler.sd_[0]:.2f} = {z_feat[0]:+.3f}")
    print(f"  angle:    ({angle:.3f} - {scaler.mean_[1]:.3f}) / {scaler.sd_[1]:.3f} = {z_feat[1]:+.3f}")
    print(f"\n  negative distance means closer than average.")
    print(f"  positive angle means more of the goal visible than average.")

    print("\n" + "=" * 66)
    print("STEP 3.  THE WEIGHTS  (this is the trained model, all of it)")
    print("=" * 66)
    w = model.w_
    print(f"  bias     w0 = {w[0]:+.3f}")
    print(f"  distance w1 = {w[1]:+.3f}   (negative: further away lowers the score)")
    print(f"  angle    w2 = {w[2]:+.3f}   (positive: wider view raises the score)")

    print("\n" + "=" * 66)
    print("STEP 4.  THE SCORE z  (add everything up)")
    print("=" * 66)
    z = w[0] + w[1] * z_feat[0] + w[2] * z_feat[1]
    print(f"  z = {w[0]:+.3f} + ({w[1]:+.3f} x {z_feat[0]:+.3f}) + ({w[2]:+.3f} x {z_feat[1]:+.3f})")
    print(f"  z = {w[0]:+.3f} + {w[1]*z_feat[0]:+.3f} + {w[2]*z_feat[1]:+.3f} = {z:+.3f}")
    print("\n  z can be any number at all, which is why it is not a probability yet.")

    print("\n" + "=" * 66)
    print("STEP 5.  THE SIGMOID  (squash z into a probability)")
    print("=" * 66)
    p = float(xg.sigmoid(np.array([z]))[0])
    print(f"  p = 1 / (1 + e^-z) = 1 / (1 + e^{-z:+.3f}) = 1 / (1 + {np.exp(-z):.3f}) = {p:.4f}")
    print(f"\n  the model rates this chance at {p:.1%}")
    print(f"  StatsBomb's own model rated it {float(shot.statsbomb_xg):.1%}")

    print("\n" + "=" * 66)
    print("STEP 6.  THE LOSS  (how badly did it do, on this one shot)")
    print("=" * 66)
    y = float(shot.goal)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"  y = {int(y)}, so the loss is {'-ln(p)' if y == 1 else '-ln(1-p)'}"
          f" = {loss:.3f}")
    # p itself goes in the list, so the row marked as the model's answer is the
    # model's answer rather than whichever hard-coded guess happens to be near it.
    for guess in sorted([0.01, 0.05, 0.3, 0.5, 0.9, 0.99, p]):
        l = -(y * np.log(guess) + (1 - y) * np.log(1 - guess))
        mark = "  <-- what it actually said" if guess == p else ""
        print(f"     had it said {guess:5.3f}, loss would be {l:6.3f}{mark}")

    print("\n" + "=" * 66)
    print("STEP 7.  THE GRADIENT  (which way should each weight move)")
    print("=" * 66)
    err = p - y
    print(f"  error = p - y = {p:.4f} - {int(y)} = {err:+.4f}")
    print(f"  {'positive: the model was too HIGH, push it down' if err > 0 else 'negative: the model was too LOW, push it up'}")
    print()
    for i, (nm, xi) in enumerate(zip(["bias", "distance", "angle"], [1.0, *z_feat])):
        g = err * xi
        print(f"  {nm:<9} gradient = error x feature = {err:+.4f} x {xi:+.3f} = {g:+.4f}"
              f"   ->  w moves {'DOWN' if g > 0 else 'UP'}")

    print("\n  (that is the gradient from THIS ONE SHOT. Training averages it")
    print("   over all 4,444 training shots before taking a single step.)")
    print("=" * 66)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
