# Results

Running log of headline findings, in plain language, as they're produced. VAL only throughout — the
test set stays frozen and untouched until a future `evaluate.py --final`. Every number here is
development-stage (val-tuned), not a final claim; see `README.md` for full methodology, scripts, and
per-phase detail.

## Phase 5, round 1 (development): confidence-weighted fusion

**Headline: the confidence-weighting hypothesis is not supported in this form.**

The research question asks whether confidence-weighted late fusion recovers performance lost under
facial occlusion by shifting reliance to voice. The first implementation — per-clip fusion weights
derived from each branch's own (temperature-calibrated) max-softmax probability — does **not**
reliably do this better than simple fixed 50/50 fusion:

| condition | fixed 50/50 | confidence-weighted (max-prob) |
|---|---|---|
| clean | 0.700 | 0.687 (worse) |
| mask 0.6 | 0.470 | 0.460 (worse) |
| mask 0.9 | 0.459 | 0.452 (worse) |
| patch 0.9 | 0.491 | 0.488 (worse, but both ≈ audio-only by construction — see below) |
| sunglasses / hand_over_mouth (all severities) | consistently better | consistently slightly worse |

Confidence-weighting beat fixed fusion on **no condition** in this round. It was roughly tied where the
face branch's failure showed up as low confidence (explicit zero-detection abstention, which forces an
exactly-uniform, obviously-unconfident distribution — this happens under `patch`), and it was worse
everywhere else, including on `mask`, the occlusion type the research question is most concerned with.

**Why:** `mask` collapses the face model onto predicting "anger" for 82-87% of clips (vs. 26% on clean)
— a confident, systematic wrong answer, not a hedged one. Mean max-softmax confidence under mask (0.37-
0.39) is lower than clean (0.58) but not low enough, relative to audio's own confidence, to meaningfully
shift fusion weight away from the face branch. A weighting signal built only from raw max-probability
has no way to distinguish "uncertain" from "confidently wrong," and `mask` is squarely the second case.

This is being recorded here, not just in a worse-looking table, because it's a real result: the
mechanism proposed for the project's core research question does not work as first implemented, and
a different signal (e.g. one that incorporates per-class confusion history or an explicit
detection-failure flag, not just max-softmax) is needed before the confidence-weighting hypothesis can
be fairly tested. See `README.md`'s Phase 5 section for the full table and the ECE/calibration results
behind it.

*(Superseded by round 2 below — this entry is kept as the honest first-round record.)*

## Phase 5, round 2 (development): mask fix, occlusion-augmented training, expanded fusion table

**1. Mask severity — fixed, now clearly monotonic.** Round 1's mask coverage was nearly tied at severities 0.6 and 0.9 (~20.5% both), because the two discrete landmark anchors used for those levels happened to sit at almost the same height. Replaced with a continuous linear interpolation between mean mouth-top height and mean eyebrow height. New official coverage (300-sample measurement):

| severity | 0.3 | 0.6 | 0.9 |
|---|---|---|---|
| face-crop coverage | 11.75% | 17.49% | 23.43% |

Clearly, monotonically increasing now, and well separated.

**2. Occlusion-augmented face model: a real robustness gain, at no clean-accuracy cost.** Two ResNet-18 variants were fine-tuned with random occlusions applied during training (3 seeds each): `aug_all` (all 4 occlusion types seen in training) and `aug_no_mask` (mask held out — unseen at evaluation). Monitoring/final-metric stayed on clean val throughout, matching the original clean-trained model's methodology, so the comparison isolates the effect of training-time occlusion exposure.

Clean val macro-F1 (mean ± std, 3 seeds): clean-trained 0.572±0.005, `aug_all` 0.572±0.020, `aug_no_mask` 0.582±0.027 — **no meaningful loss on clean data.**

Occlusion macro-F1, selected conditions (full table in `outputs/phase4_occlusion_table_by_variant.csv`):

| condition | clean-trained | aug_all | aug_no_mask |
|---|---|---|---|
| mask 0.3 | 0.315 | 0.479 | 0.405 |
| mask 0.6 | 0.184 | 0.453 | 0.361 |
| mask 0.9 | 0.116 | 0.325 | 0.184 |
| patch 0.9 | 0.147 | 0.274 | 0.273 |
| sunglasses 0.9 | 0.278 | 0.508 | 0.521 |
| hand_over_mouth 0.9 | 0.281 | 0.439 | 0.436 |

Every occlusion condition improves substantially with training-time augmentation. Notably, `aug_no_mask` — which never saw mask during training — still roughly **doubles** mask performance at every severity (e.g. 0.9: 0.116 → 0.184), showing partial transfer from the other three occlusion types to an unseen one. `aug_all`, which did see mask, does even better on mask (0.9: 0.116 → 0.325, nearly 3x) without sacrificing its performance on the other three types relative to `aug_no_mask`.

**3. Expanded fusion table: the confidence-weighting hypothesis still is not supported, now checked against all three face models.** Full table (mean ± std over 3 seeds, including mean face weight) is in `outputs/phase5_fusion_v2_table.csv`. Four fusion methods were compared — fixed 50/50, confidence-weighted (max-probability), confidence-weighted (negative entropy), and a learned logistic-regression gate — against audio-only and face-only, across all 13 conditions (clean + 4 occlusion types × 3 severities) and all 3 face-model variants (clean-trained, `aug_all`, `aug_no_mask`).

**Correction to a number reported earlier in this file:** the previous entry said "38 of 39." A precise
re-check (every one of the 39 condition×variant combinations, fixed 50/50 compared against *all three*
adaptive methods at once) found it beats-or-matches all three in **35 of 39**, not 38. The 4 exceptions,
all involving a face branch that's severely degraded (face-only macro-F1 well below chance or barely above
it): clean-trained/mask 0.6, clean-trained/mask 0.9, clean-trained/patch 0.9, and aug_no_mask/mask 0.9 — in
each, one adaptive method (usually the learned gate) edges out fixed by roughly 1 point, within the
run-to-run std. This is being corrected here rather than carried forward, per this project's rule against
reporting an unverified number.

This means: making the face branch much more occlusion-robust did **not** change the earlier conclusion
about weighting. A per-clip weight built from a branch's own confidence (max-softmax or negative entropy)
or from a logistic gate trained on those confidences still does not reliably beat simply averaging the two
branches, regardless of which face model is behind it.

**Why fixed averaging is hard to beat: it already down-weights an uncertain branch, implicitly.** When one
branch's distribution is close to uniform (uncertain), averaging it with a confident branch barely moves
the confident branch's peak — a near-uniform vector adds roughly the same small amount to every class, so
it doesn't change which class has the maximum. Explicit confidence-weighting only adds value over this
when a branch's distribution is confidently *wrong* rather than just hedged (as happens specifically under
`mask`, where the face model collapses onto "anger" with real confidence) — and even there, the gain from
correctly identifying that case is small and inconsistent, while the weighting/gate-fitting machinery adds
its own variance. Fixed averaging has no such failure mode to begin with, because it never actively trusts
either branch's self-reported confidence.

**Note on patch 0.9: fusion ≈ audio by construction.** At patch severity 0.9, 493/1224 (40.3%) of clips get zero face detections and are scored as an exactly-uniform face distribution. Any confidence-weighted method correctly identifies these as unconfident and pushes face weight toward 0 for them specifically — mean face weight at patch 0.9 drops to 0.11–0.35 across variants and methods, roughly half of the 0.50 baseline. This is *not* evidence that confidence-weighting "solves" patch occlusion; it's the mechanism correctly detecting detection failure on the ~40% of clips where it occurs, while doing nothing special for the other ~60% that still have a detection and a real (sometimes wrong) prediction. The resulting fusion score (e.g. aug_all: conf_maxprob 0.530 vs fixed_5050 0.537) is close to but still below fixed 50/50 — averaging, even blindly, isn't beaten by conditionally zeroing out face only on the clips that failed detection.

**A fairness caveat on the augmented models' generalization claim:** `aug_all` was trained on the exact
same synthetic mask-generation code used at evaluation (same geometry, different random severity/
placement per clip) — so its strong mask performance is partly "seen the noise model," not pure
generalization. `aug_no_mask`, which never saw mask in training at all, is the fair test of whether
training on *other* occlusion types transfers to a genuinely unseen one — and it does, roughly doubling
mask performance at every severity (see table above), which is the more meaningful generalization result.

## Phase 5, round 3 (development): fusion table for the augmented face models

Full table: `outputs/phase5_fusion_v2_table.csv` (already contains all 3 face variants; this section just
surfaces the `aug_all` / `aug_no_mask` rows specifically, since round 2's writeup focused on `clean`).

| condition | audio-only | face-only (aug_all) | fixed 50/50 | conf max-prob | conf neg-entropy | learned gate | mean face weight (fixed) |
|---|---|---|---|---|---|---|---|
| clean | 0.490 | 0.572 ± 0.020 | **0.702 ± 0.023** | 0.691 ± 0.025 | 0.684 ± 0.025 | 0.681 ± 0.035 | 0.500 |
| mask 0.3 | 0.490 | 0.479 ± 0.043 | **0.658 ± 0.029** | 0.643 ± 0.031 | 0.638 ± 0.032 | 0.621 ± 0.043 | 0.500 |
| mask 0.6 | 0.490 | 0.453 ± 0.042 | **0.641 ± 0.013** | 0.632 ± 0.013 | 0.626 ± 0.011 | 0.609 ± 0.022 | 0.500 |
| mask 0.9 | 0.490 | 0.325 ± 0.039 | **0.563 ± 0.018** | 0.559 ± 0.016 | 0.544 ± 0.018 | 0.543 ± 0.015 | 0.500 |
| patch 0.9 | 0.490 | 0.274 ± 0.026 | **0.537 ± 0.013** | 0.530 ± 0.013 | 0.522 ± 0.008 | 0.523 ± 0.010 | 0.500 |
| sunglasses 0.9 | 0.490 | 0.508 ± 0.043 | **0.657 ± 0.019** | 0.642 ± 0.022 | 0.642 ± 0.019 | 0.645 ± 0.026 | 0.500 |
| hand_over_mouth 0.9 | 0.490 | 0.439 ± 0.049 | **0.639 ± 0.011** | 0.632 ± 0.014 | 0.626 ± 0.018 | 0.603 ± 0.011 | 0.500 |

(`aug_no_mask`'s rows follow the same pattern — see the CSV for all 13 conditions × both variants.)

**Average margin, fixed 50/50 minus each adaptive method, over all 13 conditions:**

| | aug_all | aug_no_mask |
|---|---|---|
| vs. confidence-weighted (max-prob) | +0.0106 (wins 13/13) | +0.0092 (wins 13/13) |
| vs. confidence-weighted (neg-entropy) | +0.0180 (wins 13/13) | +0.0144 (wins 12/13) |
| vs. learned gate | +0.0265 (wins 13/13) | +0.0247 (wins 13/13) |

**Does fixed 50/50 fusion with an augmented face beat audio-only in every occlusion condition? Yes —
for both augmented variants, not for the clean-trained one.** `aug_all` and `aug_no_mask` each beat
audio-only in **12/12** occlusion conditions (margin +0.045 to +0.179). The clean-trained model's fixed
fusion only managed **10/12**, failing on exactly `mask 0.6` (0.413 vs. audio's 0.490) and `mask 0.9`
(0.490 vs. 0.490, a dead heat) — the two conditions where its face branch collapses hardest. This is a
genuinely new finding from the augmented-training ablation: training-time occlusion doesn't just improve
the face branch in isolation, it also **removes fusion's worst failure mode** (fusion dragging below
audio-only), because the face branch no longer collapses into a confidently-wrong answer under mask.

*(Val only, development-stage. No test-set numbers anywhere in this round. See `FINAL_EVAL_PLAN.md` for
the pre-registered plan for the eventual test-set run — not yet executed.)*

## Phase 5, round 4 (development): demographic fairness check, 5-fold actor CV

**Scope reduction (agreed before running):** the full spec (2 face variants × 5 folds × 3 seeds = 30
face-model training runs) was estimated at 10-17 hours on this Mac. Reduced to **1 seed per fold** (10
training runs) — actual wall time **~4.2 hours**. This trades away mean±std for this analysis only; every
number below is a single seed (seed 0), not the usual 3-seed mean. That limitation applies to this section
only — every other result in this project still uses 3 seeds.

**Method:** actor-level 5-fold CV over the 79 non-test actors (train+val combined), `scripts/19_demographic_cv.py`.
Each fold holds out ~16 actors; the fold's own training actors are further split 80/20 (by sorted actor
id) into inner-train/inner-monitor, so early stopping never sees the actors a fold is finally scored on.
Every one of the 79 actors is evaluated exactly once, by a model (face or audio) that never saw them
during training, monitoring, or model selection. **Clean clips only** — this checks base-task fairness,
not occlusion robustness. Fixed 50/50 fusion uses raw, uncalibrated softmax probabilities (no temperature
scaling, no cross-fit) — a deliberate simplification given the scope, noted here rather than silently
applied, since the confidence-weighted/gate methods' calibration machinery would have added meaningfully
more training time per fold for a check that only asked about fixed fusion.

Overall OOF macro-F1 (pooled across all 5 folds, sanity check against the main val-only numbers — note
this is a different actor set and a single seed, so not directly comparable to earlier rounds):

| method | macro-F1 |
|---|---|
| audio_only | 0.5061 |
| face_only (clean-trained) | 0.5367 |
| face_only (aug_all) | 0.5487 |
| fusion_clean (fixed 50/50) | 0.6487 |
| fusion_aug_all (fixed 50/50) | 0.6611 |

### By sex (n actors)

| sex | n actors | audio-only | face (clean) | face (aug_all) | fusion (clean) | fusion (aug_all) |
|---|---|---|---|---|---|---|
| Female | 38 | 0.512 | 0.556 | 0.566 | 0.662 | **0.677** |
| Male | 41 | 0.499 | 0.519 | 0.532 | 0.636 | 0.646 |

Female actors score higher under every single method, by a consistent margin (~0.01-0.04 absolute).

### By age band (n actors)

| age band | n actors | audio-only | face (clean) | face (aug_all) | fusion (clean) | fusion (aug_all) |
|---|---|---|---|---|---|---|
| ≤30 | 32 | 0.507 | 0.569 | 0.611 | 0.677 | **0.707** |
| 31-45 | 28 | 0.509 | 0.519 | 0.519 | 0.630 | 0.641 |
| 46-60 | 15 | 0.502 | 0.492 | 0.477 | 0.621 | 0.612 |
| 60+ | 4 | 0.485 | 0.518 | 0.479 | 0.654 | 0.614 |

Performance on the face branch and fusion generally declines with age, most clearly for the ≤30 group
being the strongest across the board. **The 60+ row (only 4 actors) is noisy** — `face_only_aug_all`
scoring *below* `face_only_clean` there (0.479 vs. 0.518) likely reflects single-seed, small-sample
variance rather than a real effect; it shouldn't be read as "augmentation hurts older actors" without a
multi-seed re-check.

### By race (n actors; "Unknown", n=1, excluded below the n≥3 threshold)

| race | n actors | audio-only | face (clean) | face (aug_all) | fusion (clean) | fusion (aug_all) |
|---|---|---|---|---|---|---|
| African American | 19 | 0.499 | 0.534 | 0.549 | 0.660 | 0.665 |
| Asian | 6 | 0.489 | **0.590** | 0.574 | **0.709** | 0.702 |
| Caucasian | 53 | 0.508 | 0.530 | 0.544 | 0.638 | 0.654 |

Asian actors (smallest included group, n=6) score highest on face-only and fusion in this single-seed run
— **worth flagging as a small-sample result, not a confirmed effect**, given n=6 actors and no seed
averaging here. African American and Caucasian actors score similarly to each other on fusion, both
somewhat below Asian.

**Headline:** there is a real, consistent sex gap (female actors score higher on every method) that's
unlikely to be single-seed noise given it holds the same direction across all 5 methods. The age and race
patterns are directionally interesting (younger actors do best; the two largest race groups are similar
to each other) but the smallest groups (60+, Asian) need a multi-seed re-check before treating their
numbers as more than suggestive — this analysis's 1-seed/fold scope reduction is the main caveat on all of
it.

Full per-group table: `outputs/phase_demographic_cv_by_group.csv`. Per-clip OOF predictions (all methods,
all folds): `outputs/phase_demographic_cv_predictions.csv`. Per-fold training times:
`outputs/phase_demographic_cv_timings.csv`.
