# A3 viva prep — xG project

5 minutes to present, hard stop at 7, then about 10 minutes of questions.
Sean is **Presenter 1** in Session 17, so no warm-up from watching others.
Practice defence: Tue 29 Sep. Real slot: Tue 13 Oct 09:00, tutor John Zeyang Zhou.

Li's advice, applied: few slides, project already run before joining, no EDA,
spend two minutes on one thing you genuinely invested in, plant baits for
questions.

---

## The 5 minute script

**Slide 1 (30 sec) — the task, stated technically.**
"Given one shot, output the probability it's a goal. Training input is a matrix
of shots and a 0/1 label, output is a weight vector. At deployment the input is
one shot's feature vector scaled with the stored training statistics, and the
output is one number between 0 and 1. No threshold, no label. The probability is
the product."

**Slide 2 (45 sec) — why it's interesting, i.e. the research question.**
"An xG model can't know if a shot goes in. Two identical chances end
differently. What it's for is calibration: of the chances it calls 0.30, about
30% should go in. But it's trained on log loss over individual binary outcomes.
Those aren't the same objective, and the project is about where they come apart."

**Slide 3 (60 sec) — H, L, A on one slide.**
- H: every function sigmoid(w·x + b). Choosing weights picks one member.
- L: mean binary cross entropy plus L2 on the weights, not the bias.
- A: full batch gradient descent, gradient = (1/n) Xᵀ(σ(Xw) − y) + 2λw.
"I started from the perceptron. Wrong output type, it gives a hard label, and
shot data isn't separable so it never converges. Keep H, change L and A, and you
have logistic regression."

**Slide 4 (90 sec) — the bit I actually invested in. This is the two minutes.**
The subgroup calibration table. Overall bias +0.0001, looks perfect. Counter
attacks −6.0 points, crosses −4.2, long shots +2.0.
"Log loss is a sum over rows. It has no concept of 'counter attack', so an
underestimate here and an overestimate there are worth exactly the same to the
objective as getting both right. The criterion is blind to the grouping the user
cares about. You can only find it by grouping residuals along dimensions the
model never saw."

**Slide 5 (45 sec) — evidence it's real.**
Results table. Point at three things: within 0.010 log loss of StatsBomb's
commercial model, accuracy 91.1% against 90.3% for predicting nothing, and
squared error loss making calibration 50% worse while barely changing ranking.

**Close (20 sec).** "Next step is putting the grouping into the criterion: add a
squared group bias penalty so the loss contains what the evaluation measures."

---

## Numbers to know cold

| | |
|---|---|
| Data | 7,451 shots, 663 goals, 8.9%, 314 matches, 6 tournaments |
| Split | 187 train / 62 val / 63 test matches, split by match never by shot |
| Features | M1 1, M2 2, M3 30, M4 35 |
| M4 test | log loss 0.2568, Brier 0.0729, ECE 0.0245, AUC 0.812 |
| StatsBomb | log loss 0.2472, AUC 0.827 |
| Baseline | always "no goal" = 90.3% accuracy. M4 = 91.1% |
| Squared error | log loss 0.2657, ECE 0.0373, AUC 0.809 |
| Reliability | 0.0013 (M1) to 0.0008 (M4). Barely moved |
| Resolution | 0.0083 (M1) to 0.0125 (M4). Climbed |
| Convergence | lr 0.5, M4 stopped at 501 iterations |
| Top weights | angle +0.40, distance −0.39, defenders in cone −0.22, keeper off line +0.18 |
| Gradient check | 4.55e-10 max relative error |

---

## Questions they will ask, and the answers

**"How many features does your final model have?"**
35. This is the exact question Li uses as the Failed test, the one where having
to scroll your own code is a 0. Know it. 1, 2, 30, 35 across M1 to M4.

**"Why logistic regression and not a neural net or XGBoost?"**
Two reasons. The task is estimating a rate from 7,451 samples with 663
positives, and the effective sample size for a rare class is set by the
positives, so a high capacity model would mostly fit noise. And the point of the
project was to study the loss versus objective gap, which needs a model where I
can attribute behaviour to the criterion rather than to model capacity. The
honest cost is that every interaction has to be hand built, and that is most of
the remaining gap to StatsBomb.

**"Why cross entropy and not squared error?"**
Two reasons, and I tested the second. It's a proper scoring rule, so in
expectation it's minimised when the predicted probability equals the true one.
And its gradient doesn't vanish when the model is confidently wrong. Squared
error's gradient carries a p(1−p) factor, which is the sigmoid's own derivative,
so when p is near 1 and the shot was missed the gradient goes quiet exactly
where the error is biggest. I trained the same model on squared error: log loss
0.2657 against 0.2568, and calibration error 0.0373 against 0.0245.

**"Why don't you penalise the bias?"**
The bias sets the model's base rate. Shrinking it pulls every prediction toward
0.5, and the event happens 9% of the time.

**"Why split by match?"**
Shots in the same match share a keeper, a pitch, a scoreline and a defensive
shape, so they aren't independent draws. Splitting by shot puts near-copies on
both sides and the test estimate comes out flattering. That's leakage.

**"Why is your accuracy barely above the baseline?"**
Because accuracy is the wrong metric and the table is there to prove it.
Predicting no goal every time scores 90.3%. A metric where modelling nothing
gets you almost everything isn't measuring the thing. Log loss, Brier and
calibration are what I judge on.

**"Your learning rate is 0.5. Why?"**
Empirical. I tried a few and watched the loss curves. I know there's a
principled bound involving the largest eigenvalue of the Hessian for a convex
loss with standardised inputs, and I didn't compute it. What I can say is the
loss decreases monotonically and the solution agrees with sklearn's LBFGS to
0.0074 on the largest weight, so it converged to the right place.
**This is a Knowledge Gap in my log, stated on purpose.**

**"How do you know your gradient is right?"**
Central differences against the analytic gradient on random data, max relative
error 4.55e-10. The check runs in the notebook. A wrong gradient still trains,
it just trains to the wrong place, so this isn't optional.

**"Did AI write this?"**
Yes, and it's documented in section 4.2. It wrote the download boilerplate, the
matplotlib and the pandas grouping. It suggested the Murphy decomposition and
the point in triangle test. It also handed me a version that fitted the
standardiser before the split, which is leakage, and I caught it. The xG framing
and the calibration question are mine. Everything mathematical I derived on
paper first and verified numerically after.

**Anything you don't know:** "I don't know. I used it because X and I verified it
by Y, but I haven't derived it myself." Li lists that as worth 50% credit. A
confident waffle is worth 0.

---

## Baits to plant

Things to say that invite the question you want:

1. "The overall calibration bias is 0.0001, which is the most misleading number
   in the report." Someone will ask why.
2. "I trained the same model on the wrong loss function on purpose." Invites the
   criterion question, which is the strongest thing in the project.
3. "Two shots in the dataset had no goalkeeper, and deleting them would have been
   a mistake." Invites a data handling question you can answer fully.

## Peer grading

Your own A3 mark partly depends on the quality of feedback you give the other
five. Straight 'poor' or straight 'excellent' across the board gets removed and
penalised. Write one specific sentence per criterion naming something they
actually did. Upload within 24 hours of your session.
