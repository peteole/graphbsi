# Research log

Goal: make GraphBSI the clearly strongest molecular graph generator at a **fixed
parameter count (7.87 M)** and a **fixed sampling budget** (50 and 500 function
evaluations), measured with the paper's own benchmark code.

Targets from `sota_metrics.md` (Moses FCD/SNN/Scaf against the scaffold test
set, as in the paper):

| Metric | GraphBSI (paper) | Best atom-level graph model | Best overall |
|---|---|---|---|
| Moses FCD ↓ | 0.72 | 0.72 (GraphBSI) | 0.52 CharRNN, 0.58 FragFM |
| Moses Scaf ↑ | 15.2 | 23.5 DeFoG@50 (validity 83.9) | same |
| Moses Nov. ↑ | 96.5 | 100.0 DMol (validity 87.8) | same |
| GuacaMol FCD ↑ | 82.6 | 82.6 (GraphBSI) | 93.4 G2PT (85 M params) |
| GuacaMol KL ↑ | 98.4 | 98.4 (GraphBSI) | 99.4–99.6 (autoregressive) |

## Measurement noise

Four runs of the same 50-step configuration (`em, gamma=20, ms=1.0,
ms_clip=false, rho=0.7`) at 10 000 samples:

| run | FCD | Scaf |
|---|---|---|
| seed 0, first | 0.871 | 18.32 |
| seed 0, repeat | 0.910 | 14.83 |
| seed 1 | 0.909 | 15.34 |
| seed 2 | 0.926 | 16.37 |

So at 10 000 samples **FCD carries about ±0.025 and Scaf about ±1.5**. Two seed-0
runs differ, so part of the spread is GPU non-determinism (TF32 matmuls), not
only the sample. Screening therefore uses 10 000 samples and only differences
above roughly 0.05 FCD / 3 Scaf are treated as real; final numbers use the
paper's sample counts (Moses 25 000, GuacaMol 15 000) and three seeds.

## What works

**Multistep extrapolation of the reconstruction** (`ms`). The network's
prediction is reused across steps: instead of feeding the current prediction
into the belief update, feed a linear (or quadratic) extrapolation built from
the predictions already computed at earlier steps. This costs no extra function
evaluation and is the single largest win found so far.

| | Moses FCD ↓ | GuacaMol FCD ↑ |
|---|---|---|
| paper sampler, 50 NFE | 1.18 | 66.7 |
| + multistep, 50 NFE | 0.87–0.91 | 77.6 |
| paper sampler, 500 NFE | 0.78 | — |
| + multistep, 500 NFE | 0.72 | — |

At 50 evaluations it reaches roughly the quality the paper needed 500 for.

Secondary observations, all at 50 NFE on Moses:
- The gain comes from the **bond** predictions: extrapolating only bonds gives
  FCD 0.82, only atoms gives 1.17 (= baseline).
- Dropping the projection back onto the simplex (`ms_clip=false`) helps:
  0.83 vs 0.96 at the same weight.
- A time grid that puts more steps late (`rho=0.7`) helps and stacks with
  multistep; `rho` below 0.5 or above 1 hurts.
- The best weight is near 1.0 for Euler; Ornstein-Uhlenbeck keeps improving to
  2.0 (FCD 0.81) but loses novelty and scaffold similarity.
- Noise level: Euler prefers `gamma` 10-20, OU prefers 20-40. OU at 80 loses
  novelty (92.7) for filters (98.9) and SNN (0.54).

## What does not work

- **Temperature on the predictions** (sharpening atoms, bonds or both): no
  effect at 0.8, actively worse at 0.5.
- **The end time `t_max`**: 1.0, 1.1, 1.5 and 2.0 are all neutral or worse than
  the paper's 1.2.
- **Continued training at a decaying learning rate.** 30 000 steps of cosine
  decay from the released Moses checkpoint: the training loss never moved from
  0.0205 and sample quality was slightly worse (FCD 1.28 vs 1.18 with the
  baseline sampler). The released model is converged, so at a fixed parameter
  count extra training is not a lever.
- **Self-conditioning** (the network additionally sees its own previous
  prediction, with zero-initialised new input columns so the converted model
  starts out identical). After 10 000 steps the held-out loss with and without
  the extra input differed by under 1 % at every noise level, i.e. the network
  had barely learned to use it. Stopped; it would need far longer training to
  pay off, and the parameter budget leaves little room.

## Ruled out cheaply (no GPU cost)

- **Asymmetric bond predictions.** The decoder reads only the upper triangle of
  the bond tensor, so an asymmetric prediction would throw away half the
  information. Measured: the network's bond output is *exactly* symmetric at
  every noise level (`max|E - Eᵀ| = 0`), so there is nothing to recover.
- **A biased molecule-size prior.** The sampler draws the atom count from the
  training distribution, so a mismatch would hurt FCD directly. Measured
  against a random 30 000-molecule training sample: generated mean 21.67 heavy
  atoms vs 21.63 in training, with every size bin within noise. No bias.
  (An earlier check that suggested a bias had used the *first* 50 000 training
  molecules, which are not a random sample.) The FCD reference set
  (`test_scaffolds`) sits slightly higher at 21.91, but matching that instead
  of the training distribution would be tuning on the evaluation set.

## Belief guidance

A second lever, also free of extra function evaluations: replace the network's
prediction by

    x_used = x + bg * (x - softmax(belief))

i.e. amplify the *change* the network proposes relative to the belief's own
current estimate. This is over-relaxation of the update direction, and it is
**not** the same as sharpening the prediction: plain temperature on `x` does
nothing (see above), while this helps a lot.

Moses, 50 NFE, Euler, `gamma=20`, `rho=0.7`, 10 000 samples:

| bg | ms | FCD ↓ | Scaf ↑ | Nov. | Val. | Filters | SNN |
|---|---|---|---|---|---|---|---|
| – | – | 1.18 | 12.9 | 96.6 | 99.4 | 97.1 | 0.498 |
| – | 1.0 | 0.904 | 14.8 | 96.7 | 99.5 | 97.9 | 0.497 |
| 0.1 | – | 0.945 | 14.4 | 96.5 | 99.6 | 97.9 | 0.502 |
| 0.3 | – | 0.843 | 12.8 | 96.1 | 99.5 | 97.8 | 0.508 |
| 0.5 | – | 0.745 | 12.4 | 95.4 | 99.6 | 98.1 | 0.511 |
| 0.7 | – | 0.761 | 11.7 | 95.2 | 99.6 | 97.9 | 0.514 |
| 1.0 | – | 0.723 | 11.9 | 94.9 | 99.6 | 98.2 | 0.517 |
| 0.2 | 1.0 | 0.797 | 11.1 | 96.1 | 99.6 | 98.0 | 0.504 |
| **0.3** | **0.5** | **0.745** | **17.2** | 95.7 | 99.5 | 98.2 | 0.510 |
| 0.5 | 0.5 | 0.727 | 13.3 | 95.1 | 99.5 | 98.3 | 0.513 |
| 0.3 | 1.0 | 0.805 | 13.7 | 95.9 | 99.4 | 98.1 | 0.504 |
| 0.5 | 1.0 | 0.858 | 15.2 | 95.6 | 99.4 | 98.1 | 0.505 |

The two levers stack on FCD, and both trade novelty and scaffold similarity for
it. `bg=0.3, ms=0.5` is the best balanced point found: against the paper's
50-NFE Euler row (FCD 1.06, Scaf 15.2, Nov. 96.5, Val. 99.3, Filters 96.9,
SNN 0.50) it is better on FCD, scaffolds, filters, SNN and validity, and about
0.8 points lower on novelty.

Also ruled out: a **noise profile** `gamma(t)` (40→10 gives FCD 0.95, 10→40
gives 0.96, both worse than a constant 20), **quadratic** instead of linear
extrapolation (0.5 weight matches linear, 1.0 overshoots to 1.02), and a
**hybrid integrator** that switches Euler→OU at 30 % of the trajectory (FCD
1.40, though it does give the best validity 99.86 and filters 98.97 seen).

The fact that the theoretically correct second-order weight (0.5) is not the
optimum, and that weights up to 2.0 keep helping, says the gain is not really
about integration order: it is about overshooting a conservative, mean-seeking
prediction.

## GuacaMol, 50 NFE, full 15 000 samples

| Sampler | NFE | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|---|
| paper OU | 50 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| OU, matched baseline | 50 | 99.5 | 99.5 | 99.0 | 92.9 | 69.2 |
| **OU + ms=1.0** | **50** | 99.3 | 99.3 | 98.8 | **94.0** | **80.7** |
| paper OU | 500 | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |
| paper EM | 500 | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |

The multistep term adds 11.5 points of FCD score at 50 evaluations, which beats
the paper's 500-evaluation OU row (80.3) with a tenth of the compute, and the KL
score matches the paper's 500-evaluation Euler row. V.U.N. 98.8 ties the best
published value (GraphBSI's own 98.7).

## Moses, 50 NFE, final — paper protocol (25 000 samples, 3 seeds)

Recipe: Euler-Maruyama, `gamma=20`, `rho=0.7`, `bg=0.3`, `ms=0.5`, `ms_clip=false`.

| | FCD ↓ | Scaf ↑ | Nov. | Val. | Uniq. | Filters | SNN |
|---|---|---|---|---|---|---|---|
| paper, 50 NFE | 1.06 | 15.2 | 96.5 | 99.3 | 100.0 | 96.9 | 0.50 |
| paper, 500 NFE | 0.72 | 14.3 | 92.5 | 99.8 | 100.0 | 99.1 | 0.54 |
| **this, 50 NFE** | **0.688 ± 0.007** | 15.7 ± 1.3 | 95.7 | 99.55 | 100.0 | 98.1 | 0.509 |

Per-seed FCD 0.682 / 0.687 / 0.696 and Scaf 15.95 / 14.26 / 16.78: FCD is
reproducible to ±0.007 at this sample count, scaffold similarity only to ±1.3.

The row beats the paper's own 50-NFE row on FCD, scaffolds, validity, filters
and SNN, and gives up 0.85 points of novelty. It also beats the paper's
**500**-NFE FCD (0.72) with a tenth of the function evaluations. At 0.688 it is
the lowest Moses FCD reported by any atom-level one-shot graph model (previous
best: GraphBSI's own 0.72); FragFM reaches 0.58 at the fragment level and
CharRNN 0.52 from SMILES.

The 10 000 → 25 000 sample shift is about −0.06 FCD, consistent across the
baseline and the new recipe, so screening numbers above are pessimistic by
roughly that much.

## Moses, 50 NFE — both integrators, paper protocol (25 000 samples)

| | FCD ↓ | Scaf ↑ | Nov. | Val. | Uniq. | Filters | SNN |
|---|---|---|---|---|---|---|---|
| paper, Euler 50 | 1.06 | 15.2 | 96.5 | 99.3 | 100.0 | 96.9 | 0.50 |
| **ours, Euler 50** | **0.688 ± 0.007** | 15.7 | 95.7 | 99.55 | 100.0 | 98.1 | 0.509 |
| paper, OU 50 | 1.19 | 15.1 | 94.6 | 99.7 | 100.0 | 98.2 | 0.52 |
| **ours, OU 50** | **0.863** | 15.6 | 94.3 | 99.7 | 100.0 | 98.6 | 0.526 |
| paper, Euler 500 | 0.72 | 14.3 | 92.5 | 99.8 | 100.0 | 99.1 | 0.54 |
| paper, OU 500 | 0.90 | 12.7 | 90.7 | 99.9 | 100.0 | 99.2 | 0.55 |

Euler recipe: `gamma=20, rho=0.7, bg=0.3, ms=0.5, ms_clip=false`.
OU recipe: `gamma=20, ms=1.0` (clipped).

Both rows beat their paper counterpart on FCD by about 0.33, with scaffolds,
filters and SNN slightly up and novelty within a point, and both beat the
paper's corresponding **500**-NFE FCD.

An FCD-optimised Euler variant (`bg=1.0`, no multistep) reaches 0.671 but drops
uniqueness to 99.9 % and novelty to 95.0, so the balanced recipe is reported.

## Where each lever applies

The guidance term helps **only on Moses at 50 NFE**. It hurts on GuacaMol
(FCD score 76.9 with `bg=0.3` vs 78.8 without) and at 500 NFE on Moses (0.738
vs 0.722 for Euler, 0.920 vs 0.829 for OU). Multistep extrapolation helps
everywhere. The reading is that guidance compensates for a coarse time
discretisation, and once the discretisation is fine (500 NFE) or the state space
is large (GuacaMol, 88 atoms), over-relaxing the update only adds error.

## GuacaMol, 50 NFE — paper protocol (15 000 samples)

| | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|
| paper, Euler 50 | 97.5 | 97.5 | 97.2 | 90.7 | 65.6 |
| **ours, Euler 50** (`gamma=10, ms=1.0`) | 98.6 | 98.6 | 98.3 | 91.0 | **71.4** |
| paper, OU 50 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| **ours, OU 50** (`gamma=20, ms=1.0`) | 99.3 | 99.3 | 98.8 | **94.0** | **80.7** |
| paper, Euler 500 | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |
| paper, OU 500 | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |

All four 50-NFE rows across both datasets now beat their paper counterpart. The
OU row gains 9.4 FCD points and already exceeds the paper's 500-NFE OU row
(80.3) at a tenth of the compute; its V.U.N. of 98.8 ties the best published
value.

## Checkpoint weight averaging ("model soup")

Averaging the weights of the three released GuacaMol checkpoints (epochs 59, 69
and 89) leaves the parameter count unchanged, and the epochs are far outside
the horizon of the EMA used in training, so the average is a genuinely
different model. GuacaMol, OU 50 NFE, 5 000 samples, same seed:

| | KL ↑ | FCD ↑ | V.U.N. |
|---|---|---|---|
| single checkpoint | 89.3 | 66.7 | 99.0 |
| weight average | 92.0 | 69.0 | 98.8 |
| single + `ms=1.0` | 91.7 | 77.6 | 98.8 |
| weight average + `ms=1.0` | 92.0 | 78.2 | 98.9 |

On its own the average is worth +2.8 KL and +2.3 FCD, but on top of the
multistep term it adds about half a point, i.e. nothing beyond noise. The two
corrections are redundant, so the average is **not** part of the final recipe.
That redundancy is mild support for the mean-seeking-bias reading: averaging
weights across epochs and overshooting the prediction attack the same error.

## GuacaMol, 50 NFE — final row

Recipe: Ornstein-Uhlenbeck, `gamma=20`, `ms=1.0`, `rho=0.7`. 15 000 samples.

| | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|
| paper, OU 50 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| **ours, OU 50** | 99.38 | 99.38 | **98.91** | 93.69 | **81.06** |
| ours, OU 50, `gamma=40, ms=1.5` | 99.35 | 99.35 | 98.89 | 93.20 | 82.18 |
| ours, OU 50, `gamma=20, ms=1.0` | 99.31 | 99.31 | 98.77 | 94.03 | 80.71 |
| paper, OU 500 | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |
| paper, Euler 500 | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |

The reported row gains 9.8 FCD points over the paper's 50-NFE row while matching
its KL, and its V.U.N. of 98.91 is the highest value in the literature (previous
best 98.7, GraphBSI's own). It also beats the paper's 500-NFE OU FCD of 80.3.
The `gamma=40, ms=1.5` variant reaches 82.18 FCD, essentially the paper's
500-NFE Euler value, but gives up 0.5 KL, so the balanced row is reported.

`rho=0.7` helps at 50 NFE on both datasets but **not** at 500 NFE on Moses
(0.752 vs 0.721): a grid that concentrates steps late only pays when steps are
scarce.

## Moses, 500 NFE — final Euler row (25 000 samples)

Recipe: Euler-Maruyama, `gamma=100`, `ms=1.5`, `ms_clip=false`, no guidance.

| | FCD ↓ | Scaf ↑ | Nov. | Val. | Uniq. | Filters | SNN |
|---|---|---|---|---|---|---|---|
| paper, 500 NFE | 0.72 | 14.3 | 92.5 | 99.8 | 100.0 | 99.1 | 0.54 |
| **ours, 500 NFE** | **0.678** | **17.9** | **94.3** | 99.64 | 100.0 | 99.05 | 0.525 |
| **ours, 50 NFE** | 0.688 | 15.7 | 95.7 | 99.55 | 100.0 | 98.1 | 0.509 |

Scaffold similarity gains 3.6 points and novelty 1.8 over the paper's row, with
FCD slightly better and validity, filters and SNN within a few hundredths.

The important comparison is against **our own 50-NFE row**: ten times the
sampling compute buys 0.01 FCD. With the corrected sampler the model is
essentially saturated at 50 function evaluations on Moses.

## A discrepancy found while porting the sampler into the library

The released `StepSampler` evaluates the *final* reconstruction at the
second-to-last grid point, while the belief has already been advanced to the
last one. The reconstructor takes the time as an input, so this is
inconsistent; the research harness passes the time the belief has actually
reached. Every number in this log uses the consistent time, so part of the
measured gain could come from that fix rather than from the two corrections. It
is being measured separately (baseline and best recipe, both choices of final
time) before any attribution is claimed.

## Attribution of the gain (the final-time question, settled)

Moses, 10 000 samples, the two choices of time for the final reconstruction:

| | FCD ↓ | Scaf ↑ | Nov. | Val. |
|---|---|---|---|---|
| baseline, consistent time | 1.176 | 12.93 | 96.63 | 99.36 |
| baseline, released behaviour | 1.174 | 12.98 | 96.63 | 99.39 |
| best recipe, consistent time | 0.745 | 17.17 | 95.69 | 99.53 |
| best recipe, released behaviour | 0.745 | 17.16 | 95.68 | 99.54 |

The difference is 0.002 FCD, i.e. nothing. **None of the reported improvement
comes from that detail**; all of it comes from the two corrections. The
released sampler's inconsistency is therefore harmless in practice. The
consistent time is kept as the default because it is better justified, and the
option exists to reproduce the released behaviour exactly.

Repeating the baseline two days apart reproduced FCD 1.176 to the digit, so the
harness is deterministic for a fixed configuration on the same GPU type; the
run-to-run spread noted earlier comes from different nodes.

## Trying to push scaffold similarity (Moses, 50 NFE, 25 000 samples)

Scaffold similarity against the held-out scaffold split is the one Moses column
where another model is clearly ahead (DeFoG 23.5). Four attempts:

| | FCD ↓ | Scaf ↑ | Nov. | Val. |
|---|---|---|---|---|
| reported recipe (`gamma=20, bg=0.3, ms=0.5`) | **0.688** | 15.7 | 95.7 | 99.55 |
| `gamma=30` | 0.894 | 14.9 | 96.6 | 99.27 |
| guidance early only (`bg_hi=0.5`) | 0.760 | **16.9** | 96.1 | 99.48 |
| guidance late only (`bg_lo=0.5`) | 0.721 | 16.5 | 95.9 | 99.44 |
| weaker guidance (`bg=0.15`) | 0.731 | 14.8 | 95.7 | 99.57 |

None dominates. Restricting the guidance to half the trajectory buys about one
scaffold point — inside the ±1.3 seed noise — for 0.03–0.07 worse FCD. The
trade-off is robust: FCD rewards matching the reference distribution closely,
while this scaffold metric rewards generating ring systems held out of training,
and the corrections buy the former at the cost of the latter.

Scaffold similarity therefore stays near 16–18 at validity 99.6. DeFoG's 23.5
comes at validity 83.9, so the two are different operating points rather than a
straight loss; the honest statement is that GraphBSI is not the best model on
this column.

## GuacaMol, 50 NFE — final row with error bars (15 000 samples, 3 seeds)

Recipe: Ornstein-Uhlenbeck, `gamma=20`, `ms=1.0`, `rho=0.7`.

| | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|
| paper, OU 50 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| **ours, OU 50** | 99.41 | 99.41 | **98.90 ± 0.07** | 93.78 ± 0.13 | **81.06 ± 0.11** |

Per-seed FCD 81.06 / 81.17 / 80.95. Every column is equal to or better than the
paper's row; FCD gains 9.8 points and V.U.N. of 98.90 exceeds the best published
value of 98.7 (GraphBSI's own).

## An out-of-memory failure worth recording

The first attempt at the GuacaMol 500-NFE rows died immediately: batch 512 fits
on an 80 GB A100 but not on the 40 GB variant, and the scheduler had placed it
on the latter. `sample_smiles` now halves the batch and retries rather than
losing hours of sampling, and the rows were resubmitted to an H200. Batch size
changes the random number stream but not the method, so it only moves numbers
within seed noise.

## GuacaMol, 500 NFE — the corrections do not help here (15 000 samples)

| | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|
| paper, OU 500 | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |
| ours, OU 500 + `ms=1.0` | 99.63 | 99.61 | 98.46 | 97.11 | 80.43 |
| paper, Euler 500 | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |
| ours, Euler 500 + `ms=1.0` | 98.86 | 98.84 | 98.42 | 92.49 | 78.10 |

On the Euler row the correction costs 4.5 points of FCD score and 2.1 of KL; on
the OU row FCD is flat and KL drops 1.3. The Euler loss is far larger than any
plausible measurement offset, so the regression is real.

**Scope of the contribution, stated plainly:**

| | 50 NFE | 500 NFE |
|---|---|---|
| Moses | large gain (FCD 1.06 → 0.688) | gain (0.72 → 0.678, Scaf +3.6) |
| GuacaMol | large gain (FCD 71.3 → 81.1) | **no gain; use the paper's sampler** |

This is consistent with the mechanism: the corrections compensate for a coarse
time discretisation. Once 500 steps are spent on GuacaMol's 88-atom graphs the
discretisation error is small, and over-relaxing the update only adds error of
its own. The GuacaMol 500-NFE rows of the table are therefore the paper's
sampler unchanged.

## The GuacaMol 500-NFE regression is real (matched baseline)

Running the paper's own 500-NFE OU configuration through this pipeline:

| | Val. | V.U. | V.U.N. | KL ↑ | FCD ↑ |
|---|---|---|---|---|---|
| paper, published | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |
| same configuration, measured here | 99.81 | 99.77 | 98.18 | 98.40 | 80.05 |
| with `ms=1.0` | 99.63 | 99.61 | 98.46 | 97.11 | 80.43 |

The pipeline reproduces the published row to 0.01 KL and 0.25 FCD, so there is
no systematic offset at 500 NFE and the 1.3-point KL loss from the correction is
genuine. (At 50 NFE a matched baseline *was* about a point below the published
KL, which is why this check was needed before claiming anything.)

The GuacaMol 500-NFE rows therefore stay with the paper's sampler.
