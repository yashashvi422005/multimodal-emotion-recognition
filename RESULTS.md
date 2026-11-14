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

**Headline (corrected in round 5 below):** female actors score higher than male actors on every method
in this single-seed run, and younger actors score higher than older ones — but see round 5: an
actor-level bootstrap found neither difference's 95% CI excludes zero for the *sex* comparison at the
group sizes available here, so "consistent sex gap" was an overclaim and is retracted below. The age
comparison fares better for some methods. The smallest groups (60+, Asian) still need a multi-seed
re-check before treating their numbers as more than suggestive — this analysis's 1-seed/fold scope
reduction is the main caveat on all of it.

Full per-group table: `outputs/phase_demographic_cv_by_group.csv`. Per-clip OOF predictions (all methods,
all folds): `outputs/phase_demographic_cv_predictions.csv`. Per-fold training times:
`outputs/phase_demographic_cv_timings.csv`.

## Phase 5, round 5 (development): actor-level bootstrap CIs on the demographic differences

**Method:** actor-level bootstrap (B=1000), stratified within each group being compared — each draw
resamples group A's actors with replacement (same N as the real group) independently from group B's
actors, pools clips per group, computes macro-F1, and takes the difference. This is the honest way to
put a CI on round 4's group numbers, since the unit of resampling must be the actor (clips from the same
actor aren't independent), not the clip. `scripts/20_demographic_bootstrap_ci.py`, training-free (reuses
round 4's saved OOF predictions). Following the standard here: **a difference is only called a "gap"
below if its 95% CI excludes zero; otherwise it's reported as an observed difference, not a gap.**

### Sex: female − male (observed differences, none reach a 95% CI excluding zero)

| method | female | male | diff | 95% CI | gap? |
|---|---|---|---|---|---|
| audio-only | 0.512 | 0.499 | +0.013 | [−0.031, +0.059] | no |
| face (clean) | 0.556 | 0.519 | +0.037 | [−0.019, +0.097] | no |
| face (aug_all) | 0.566 | 0.532 | +0.034 | [−0.023, +0.091] | no |
| fusion (clean) | 0.662 | 0.636 | +0.025 | [−0.031, +0.080] | no |
| fusion (aug_all) | 0.677 | 0.646 | +0.032 | [−0.024, +0.086] | no |

**Retraction: round 4's "consistent sex gap" claim does not hold up.** The direction is the same
(female higher) for all 5 methods, but every single 95% CI includes zero — at n=38/41 actors and a
single training seed, this project cannot distinguish the observed female-male difference from sampling
noise. It is reported here as an **observed difference**, not a confirmed gap.

### Age: ≤30 − 46–60 (observed differences; two reach a 95% CI excluding zero)

| method | ≤30 | 46-60 | diff | 95% CI | gap? |
|---|---|---|---|---|---|
| audio-only | 0.507 | 0.502 | +0.004 | [−0.057, +0.068] | no |
| face (clean) | 0.569 | 0.492 | +0.077 | [−0.002, +0.154] | no (barely) |
| face (aug_all) | 0.611 | 0.477 | **+0.134** | **[+0.068, +0.201]** | **yes** |
| fusion (clean) | 0.677 | 0.621 | +0.055 | [−0.007, +0.132] | no (barely) |
| fusion (aug_all) | 0.707 | 0.612 | **+0.095** | **[+0.022, +0.170]** | **yes** |

**The audio branch is roughly uniform across age groups** (diff +0.004, CI straddling zero by a wide
margin) — consistent with the research question's premise that voice doesn't carry the same demographic
performance disparities the face branch does. The age difference is only a confirmed gap (CI excludes
zero) for the two **aug_all**-involving methods; the clean-trained face and clean-fusion differences are
in the same direction but their CIs just barely include zero (lower bound −0.002 and −0.007), so they
are reported as observed differences, not confirmed gaps.

### Does occlusion-augmented training widen the age gap? Paired bootstrap on the change, clean → aug_all

| branch | gap (clean) | gap (aug_all) | change | 95% CI | widens? |
|---|---|---|---|---|---|
| face-only | +0.077 | +0.134 | +0.057 | [−0.020, +0.125] | no (not confirmed) |
| fusion | +0.055 | +0.095 | **+0.039** | **[+0.006, +0.075]** | **yes** |

Each bootstrap draw reused the *same* resampled actor sets for both the clean and `aug_all` metrics, so
the clean-vs-`aug_all` comparison within each draw is properly paired. **For the fusion pipeline, the
age gap's widening from occlusion-augmented training is a confirmed effect** (CI excludes zero) — the
same widening for the face branch alone is in the same direction but not statistically confirmed at this
sample size (CI includes zero, barely: lower bound −0.020). This is a genuine, reportable caution on the
occlusion-augmentation ablation: the robustness gain documented in round 2 comes with some evidence of a
wider age-based performance disparity after fusion, not just a face-only side effect.

**Standing limitation, repeated from round 4: every number in this section is single-seed (1 seed/fold),
not the usual 3-seed mean.** The bootstrap CIs account for actor-sampling uncertainty, not seed
uncertainty — a 3-seed version of this analysis could shift these point estimates and CIs.

Full tables: `outputs/phase_demographic_bootstrap_sex.csv`, `..._age.csv`, `..._age_change.csv`.

## Phase 5, round 6 (development): WavLM audio branch, tuned eGeMAPS, and WavLM fusion

**⚠️ Superseded: see round 7 below.** The WavLM numbers in this section (0.7027 ± 0.0076 and every
WavLM-involving fusion number) turned out to depend on an arbitrary historical batch grouping from the
Kaggle run, not a reproducible property of the model — round 7 explains why and replaces every number
here that involves WavLM. eGeMAPS's numbers in this section (tuned `C=0.3` = 0.4950) are unaffected and
still current. Kept below for the historical record, not as current numbers.

Three pieces, all VAL only, all training-free locally (WavLM was fine-tuned on Kaggle; see
`README.md`'s WavLM section for that run). `scripts/22_wavlm_rescore_and_tuned_egemaps.py`,
`scripts/23_phase5_fusion_wavlm.py`.

### 1. WavLM re-scored locally, identical metric code to eGeMAPS

The val clip set is **confirmed identical** to eGeMAPS's: same 1,224 clips, same 15 val actors, for
every one of the 3 seeds, with an explicit assertion that no test actor appears. Re-scoring locally
(`sklearn.metrics.f1_score`/`accuracy_score`/`confusion_matrix`, no abstention — WavLM never abstains)
reproduces the Kaggle-reported numbers **exactly** (diff = 0.0000 for all 3 seeds):

| seed | macro-F1 (local) | macro-F1 (Kaggle-reported) | accuracy | val-actor bootstrap 95% CI |
|---|---|---|---|---|
| 0 | 0.6940 | 0.6940 | 0.7002 | [0.640, 0.743] |
| 1 | 0.7124 | 0.7124 | 0.7116 | [0.661, 0.763] |
| 2 | 0.7016 | 0.7016 | 0.7067 | [0.635, 0.760] |
| **mean ± std** | **0.7027 ± 0.0076** | — | **0.7062 ± 0.0047** | — |

Per-class F1 (mean ± std over 3 seeds): anger 0.859±0.017, disgust 0.680±0.009, fear 0.571±0.010,
happy 0.761±0.020, neutral 0.773±0.029, sad 0.572±0.015. Full per-seed confusion matrices:
`outputs/confusion_matrix_wavlm_seed{0,1,2}.csv`. All three bootstrap CIs are wide relative to the
point estimate (±0.05-0.06) at only 15 val actors — standard caveat for any val-actor estimate in this
project.

### 2. Tuning asymmetry fixed: eGeMAPS LR's C tuned on val

Every prior round left eGeMAPS at its original, never-tuned `C=1.0` while WavLM's architecture/training
hyperparameters were always chosen via early stopping on val macro-F1 — an unfair comparison. Small grid
search (`C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`, same train/val features, same deterministic
`lbfgs` solver — std ≈ 0 across seeds, as established in Phase 3):

| variant | C | val macro-F1 | val accuracy |
|---|---|---|---|
| untuned | 1.0 | 0.4903 | 0.4926 |
| **tuned** | **0.3** | **0.4950** | **0.4975** |

Tuning buys a small improvement (+0.0047 macro-F1) — eGeMAPS's own hyperparameter was already close to
optimal in this grid, so the earlier asymmetry wasn't masking a large gap. **The much bigger gap is
between eGeMAPS and WavLM themselves**: tuned eGeMAPS (0.495) vs. WavLM (0.703 ± 0.008) — WavLM
outperforms eGeMAPS by about +0.21 macro-F1 on the same val clips. Both numbers (untuned and tuned) are
reported side by side in `README.md`; "eGeMAPS" elsewhere in this project now means the **tuned**
version unless stated otherwise.

### 3. Fusion re-run with WavLM as the audio branch (all 3 face variants, same pre-registered methods)

Same four fusion methods (`fixed_5050`, `conf_maxprob`, `conf_negentropy`, `learned_gate`), same
temperature-calibration and gate cross-fit methodology as rounds 2-3, with WavLM substituted for the
classical audio branch. WavLM's 3 seeds are paired with the face models' 3 seeds (seed *i* audio with
seed *i* face) rather than reused across seeds as the deterministic eGeMAPS branch was.

**Pooled over all 12 occlusion conditions (excludes clean), mean macro-F1 across methods:**

| face variant | audio-only (WavLM) | face-only | fixed 50/50 | conf-maxprob | conf-negentropy | learned gate |
|---|---|---|---|---|---|---|
| clean-trained | 0.703 | 0.273 | **0.706** | 0.702 | 0.702 | 0.681 |
| aug_all | 0.703 | 0.451 | **0.741** | 0.738 | 0.737 | 0.711 |
| aug_no_mask | 0.703 | 0.427 | **0.735** | 0.731 | 0.730 | 0.698 |

Fixed 50/50 is again the best fusion method on average — the same pattern as the eGeMAPS-audio rounds
holds with a much stronger audio branch.

**Does fixed fusion beat audio-only in every occlusion condition? Not for the clean-trained face.**
Per-condition (13 conditions including clean), fixed 50/50 beats WavLM-only in:
- **aug_all: 13/13** conditions
- **aug_no_mask: 13/13** conditions
- **clean-trained: 10/13** conditions — it *loses* to audio-only under `mask` at severity 0.3
  (−0.014), `mask` at 0.6 (**−0.046**, the worst case), and ties at `mask` 0.9 (+0.0003) and `patch` 0.9
  (−0.0007).

**Why clean-trained loses here but didn't as clearly against eGeMAPS:** WavLM (≈0.70) is a much
stronger audio branch than eGeMAPS (≈0.49-0.50), so averaging it 50/50 with a face branch that collapses
under mask (clean-trained face-only macro-F1 crashes to 0.115-0.184 under `mask` 0.6-0.9) pulls the
fused score down much further than it would with a weaker audio branch. This is the clearest evidence
yet in this project that **occlusion-augmented training (`aug_all`/`aug_no_mask`) is not just a nice-to-
have — it's what keeps fixed averaging from actively hurting once the audio branch is strong.** Full
per-seed and aggregate tables: `outputs/phase5_fusion_wavlm_by_seed.csv`, `..._table.csv`.

## Phase 5, round 7 (development): WavLM is not batch-composition-invariant — re-scored, every WavLM number superseded

While dry-running `evaluate.py` (the script `FINAL_EVAL_PLAN.md` specifies for the eventual test-set
run) on VAL, WavLM's audio-only macro-F1 came back as 0.6867, not round 6's 0.6940. Tracing it down
found a real, previously-unknown property of this trained model, not a coding bug in the usual sense:

**WavLM's frozen CNN feature encoder has no attention-masking mechanism**, unlike the transformer
layers above it (which do correctly mask padded positions, including in this project's own
masked-mean pooling — see the "training pooling" note below). When clips of different lengths share a
zero-padded batch, the padding changes the conv encoder's `GroupNorm` statistics for the *whole*
sequence, not just near the boundary. Reconstructing the exact historical batch one val clip
(`1035_IEO_SAD_MD`) happened to land in (from `wavlm_val_probs.csv`'s own row order) reproduced
round 6's stored prediction exactly (fear 0.40 / sad 0.32); a different, equally valid batch grouping
gave a completely different top class (happy, 0.75) for the *same clip, same weights*. Across all 1224
val clips this moved 362 clips' probabilities by >0.01 and 23 clips' argmax prediction outright, for a
net macro-F1 shift of about −0.01 to −0.02 depending on seed.

**Fix: `batch_size=1` (no padding at all).** A clip's score then depends on nothing but its own audio
and the checkpoint weights — the only way to get a well-defined, reproducible per-clip result. This is
what `evaluate.py` now does, and `scripts/25_wavlm_rescore_batch1.py` applies it to re-score all 3
seeds, run twice each to confirm determinism (bit-identical, max diff 0.0, on both MPS and CPU) before
trusting the result:

| seed | macro-F1 | accuracy | val-actor bootstrap 95% CI |
|---|---|---|---|
| 0 | 0.6809 | 0.6879 | [0.633, 0.725] |
| 1 | 0.7128 | 0.7124 | [0.662, 0.766] |
| 2 | 0.6850 | 0.6912 | [0.619, 0.743] |
| **mean ± std** | **0.6929 ± 0.0142** | **0.6972 ± 0.0109** | — |

**Every number in round 6 that involves WavLM is superseded by this table** (and by the re-run fusion
numbers below) — round 6 is kept only as a historical record of the batch-order-dependent run, not as
current numbers. Per-class F1 (mean ± std): anger 0.853±0.031, disgust 0.674±0.012, fear 0.567±0.005,
happy 0.750±0.034, neutral 0.758±0.041, sad 0.557±0.020.

### Did WavLM's training-time pooling ignore padded frames? No — only the upstream encoder is unmasked

Checked, not retrained: `WavLMClassifier.forward` (`notebooks/wavlm_finetune_kaggle.ipynb`) computes
`_get_feature_vector_attention_mask` and does a masked-mean pool over the transformer's last hidden
state, and this exact `forward` method is called identically inside both the training loop and
`evaluate()` — there is no training/eval branch. **The pooling step correctly excluded padded frames in
both training and evaluation.** The batch-composition sensitivity above comes entirely from the
upstream CNN feature encoder (before the transformer, before pooling), which has no masking mechanism
at all — a known architectural property of wav2vec2/WavLM-style models, not specific to this project's
code. One real consequence: because training batches were padded and shuffled every epoch, training
itself was exposed to the same batch-composition noise from epoch to epoch — a mild additional source
of training stochasticity on top of the usual data-order/weight-init randomness. This does not
invalidate the trained checkpoints; it only means *evaluating* them needs `batch_size=1` to have a
well-defined answer. **Standing limitation, added here**: WavLM's 3 training runs (on Kaggle) were
themselves subject to this epoch-to-epoch batch-padding noise; a future round could re-run training
with length-bucketed batching (grouping similar-length clips together) to reduce it, but that is a
retraining decision, not made here.

### Fusion re-run with the corrected WavLM (`scripts/23_phase5_fusion_wavlm.py`, same methodology)

**Pooled over all 12 occlusion conditions (excludes clean), mean macro-F1:**

| face variant | audio-only (WavLM) | face-only | fixed 50/50 | conf-maxprob | conf-negentropy | learned gate |
|---|---|---|---|---|---|---|
| clean-trained | 0.693 | 0.273 | **0.698** | 0.694 | 0.694 | 0.674 |
| aug_all | 0.693 | 0.451 | **0.737** | 0.733 | 0.731 | 0.706 |
| aug_no_mask | 0.693 | 0.427 | **0.731** | 0.727 | 0.725 | 0.694 |

Fixed 50/50 is still the best method on average, same pattern as every prior round.

**Does fixed fusion beat audio-only in every occlusion condition?** Per-condition (13 conditions
including clean): **aug_all 13/13**, **aug_no_mask 13/13**, **clean-trained 11/13** — loses to
audio-only at `mask` 0.3 (−0.012) and `mask` 0.6 (**−0.049**, still the worst case); this is one fewer
exception than round 6 found (`patch` 0.9 is now a narrow win, +0.0004, rather than a loss) — the
qualitative story (occlusion-augmented training is what keeps averaging from hurting once audio is
strong) is unchanged.

**Fused val macro-F1 on CLEAN faces with the corrected WavLM, next to WavLM alone:**

| | macro-F1 |
|---|---|
| WavLM alone (audio-only, clean) | 0.6929 ± 0.0142 |
| + clean-trained face, fixed 50/50 | 0.7599 ± 0.0152 |
| + aug_all, fixed 50/50 | 0.7622 ± 0.0150 |
| + aug_no_mask, fixed 50/50 | 0.7623 ± 0.0197 |

Full per-seed and aggregate tables (overwritten in place): `outputs/phase5_fusion_wavlm_by_seed.csv`,
`..._table.csv`. Underlying re-scored WavLM data: `outputs/wavlm_val_probs_batch1.csv`,
`outputs/wavlm_batch1_seed_summary.csv`.

## Final test results (frozen test set, 12 actors) — the one and only run

**Pre-registration commit: `09e454e981079e7197b46c9c92b0fc2b9ace1a74`** (the amendment to
`FINAL_EVAL_PLAN.md` that locked the methodology below, committed and pushed *before* this run).
`python evaluate.py --split test --final`, all 3 seeds, run exactly once on 2026-11-14. No code or
methodology changed after seeing these numbers. Calibration and the learned gate were fit once on all
15 val actors and frozen (per the amendment) — test labels were used only to score, never to fit
anything. Everything below is reported as it came out, including results that cut against the project's
own hypotheses.

### Headline

- **P1 — does the face add to a strong voice model? Yes.** Fixed 50/50 fusion (`aug_all` face + WavLM)
  beats WavLM alone, pooled and in every one of the 13 individual conditions (clean + 12 occlusion),
  all with the test-actor-level 95% CI excluding zero.
- **P2 — does occlusion-augmented training generalize to an unseen occlusion? Yes.** `aug_no_mask`
  beats the clean-trained face under `mask` at all 3 severities, CI excludes zero every time.
- **P3 — does confidence-weighting beat fixed averaging, once both are calibrated? No — fixed
  averaging wins, pooled.** Confidence-weighted (max-prob, calibrated) fusion is *worse* than fixed
  50/50 pooled across all 13 conditions (diff −0.0038, CI excludes zero). This is **not** uniform,
  though: 4 of the 13 individual conditions do **not** reach significance (`clean` p=0.05 borderline,
  `patch` 0.9, `hand_over_mouth` 0.6 and 0.9 — see the full table below) — reported honestly rather than
  rounded up to "always."

### Audio branches alone (macro-F1, mean ± std over 3 seeds)

| | per-seed | mean ± std |
|---|---|---|
| eGeMAPS (tuned, `C=0.3`) | 0.5044, 0.5044, 0.5044 | 0.5044 ± 0.0000 (deterministic) |
| WavLM (`batch_size=1`) | 0.6887, 0.7175, 0.7086 | 0.7049 ± 0.0121 |

### Face branch alone, clean condition (macro-F1, mean ± std over 3 seeds)

| variant | macro-F1 |
|---|---|
| clean-trained | 0.6050 ± 0.0151 |
| `aug_all` | 0.6227 ± 0.0100 |
| `aug_no_mask` | 0.6303 ± 0.0137 |

### Fusion, pooled over the 12 occlusion conditions (excludes clean), macro-F1 mean over 3 seeds

| face variant | audio branch | audio-only | face-only | fixed 50/50 | conf-maxprob | conf-negentropy | learned gate |
|---|---|---|---|---|---|---|---|
| clean-trained | eGeMAPS | 0.504 | 0.263 | 0.522 | 0.513 | 0.513 | **0.525** |
| `aug_all` | eGeMAPS | 0.504 | 0.466 | **0.620** | 0.612 | 0.604 | 0.608 |
| `aug_no_mask` | eGeMAPS | 0.504 | 0.440 | **0.606** | 0.601 | 0.595 | 0.602 |
| clean-trained | WavLM | 0.705 | 0.263 | **0.710** | 0.706 | 0.706 | 0.695 |
| `aug_all` | WavLM | 0.705 | 0.466 | **0.742** | 0.739 | 0.736 | 0.735 |
| `aug_no_mask` | WavLM | 0.705 | 0.440 | **0.739** | 0.735 | 0.734 | 0.728 |

Fixed 50/50 is again the best fusion method on average for every face/audio pairing except
clean-trained+eGeMAPS (where the learned gate edges it out, 0.525 vs 0.522) — the same "fixed averaging
is hard to beat" pattern documented since round 1, now confirmed out of sample on test.

### Does fixed fusion (WavLM) beat audio-only in every condition?

Per-condition (13 conditions including clean): **`aug_all` 13/13**, **`aug_no_mask` 13/13**,
**clean-trained 11/13** — loses to WavLM-only at `mask` 0.3 (−0.0018) and `mask` 0.6 (−0.0217, the worst
case), same exceptions as every prior round. Full tables: `outputs/evaluate_test_table.csv`,
`outputs/evaluate_test_by_seed.csv`, `outputs/evaluate_test_per_clip.csv`.

### P1 — fixed 50/50 fusion (`aug_all` + WavLM) vs. WavLM alone

Paired actor-level bootstrap (B=2000), seed-averaged diff per draw, 95% CI, test actors only.

| condition | diff (seed-avg) | 95% CI | p |
|---|---|---|---|
| pooled (all 13) | +0.0404 | [+0.0182, +0.0629] | 0.0010 |
| clean | +0.0731 | [+0.0355, +0.1072] | 0.0000 |
| mask 0.3 | +0.0492 | [+0.0160, +0.0779] | 0.0020 |
| mask 0.6 | +0.0479 | [+0.0170, +0.0759] | 0.0020 |
| mask 0.9 | +0.0099 | [+0.0034, +0.0164] | 0.0010 |
| sunglasses 0.3 | +0.0417 | [+0.0164, +0.0677] | 0.0020 |
| sunglasses 0.6 | +0.0414 | [+0.0169, +0.0664] | 0.0010 |
| sunglasses 0.9 | +0.0439 | [+0.0228, +0.0666] | 0.0000 |
| patch 0.3 | +0.0478 | [+0.0217, +0.0733] | 0.0000 |
| patch 0.6 | +0.0353 | [+0.0144, +0.0588] | 0.0000 |
| patch 0.9 | +0.0108 | [+0.0039, +0.0192] | 0.0000 |
| hand_over_mouth 0.3 | +0.0460 | [+0.0135, +0.0756] | 0.0050 |
| hand_over_mouth 0.6 | +0.0371 | [+0.0068, +0.0649] | 0.0160 |
| hand_over_mouth 0.9 | +0.0387 | [+0.0130, +0.0629] | 0.0050 |

**Every single condition significant, same direction.** The face adds to WavLM everywhere tested,
though the effect shrinks under the heaviest occlusion (`mask`/`patch` at 0.9: +0.01, an order of
magnitude smaller than clean's +0.07) — occlusion genuinely erodes the face's contribution, it just
never flips sign or loses significance on this test set.

### P2 — `aug_no_mask` vs. clean-trained face-only, under mask

| severity | diff (seed-avg) | 95% CI | p |
|---|---|---|---|
| 0.3 | +0.0935 | [+0.0496, +0.1331] | 0.0000 |
| 0.6 | +0.2258 | [+0.1802, +0.2742] | 0.0000 |
| 0.9 | +0.0584 | [+0.0400, +0.0766] | 0.0000 |

Confirmed on test: training on unseen occlusion types generalizes to the held-out `mask` type, at every
severity. The effect is largest at 0.6 (one seed, seed1, shows a particularly large +0.3633 jump — see
`outputs/evaluate_test_per_clip.csv` for per-seed detail), smallest at 0.9 where both variants are
already badly degraded.

### P3 — confidence-weighted (max-prob, calibrated) vs. fixed 50/50 (`aug_all` + WavLM)

| condition | diff (seed-avg) | 95% CI | p | significant? |
|---|---|---|---|---|
| pooled (all 13) | −0.0038 | [−0.0050, −0.0025] | 0.0000 | **yes** |
| clean | −0.0047 | [−0.0095, −0.0000] | 0.0500 | borderline |
| mask 0.3 | −0.0034 | [−0.0059, −0.0008] | 0.0120 | yes |
| mask 0.6 | −0.0046 | [−0.0085, −0.0002] | 0.0450 | yes |
| mask 0.9 | −0.0031 | [−0.0055, −0.0007] | 0.0080 | yes |
| sunglasses 0.3 | −0.0051 | [−0.0073, −0.0026] | 0.0000 | yes |
| sunglasses 0.6 | −0.0035 | [−0.0067, −0.0005] | 0.0210 | yes |
| sunglasses 0.9 | −0.0060 | [−0.0103, −0.0025] | 0.0000 | yes |
| patch 0.3 | −0.0057 | [−0.0091, −0.0026] | 0.0000 | yes |
| patch 0.6 | −0.0044 | [−0.0070, −0.0019] | 0.0010 | yes |
| patch 0.9 | −0.0003 | [−0.0024, +0.0017] | 0.8300 | **no** |
| hand_over_mouth 0.3 | −0.0038 | [−0.0081, +0.0006] | 0.0930 | **no** |
| hand_over_mouth 0.6 | −0.0013 | [−0.0054, +0.0028] | 0.5570 | **no** |
| hand_over_mouth 0.9 | −0.0034 | [−0.0074, +0.0002] | 0.0690 | **no** |

**The pooled result confirms the project's original round-1 observation as a real, statistically
significant effect on held-out test data: fixed 50/50 beats confidence-weighting.** But it is not
uniform — reported plainly rather than smoothed over: `patch` 0.9 and all three `hand_over_mouth`
severities do not reach significance (CI includes zero), and `clean` is right at the p=0.05 boundary.
The effect is clearest under `mask`, `sunglasses`, and `patch` at low/mid severity; it is weakest or
absent exactly where the face branch is already most degraded (`patch` 0.9, `hand_over_mouth`) — at
that point confidence-weighting and fixed averaging converge because the face's confidence collapses
either way (consistent with the "patch 0.9 forces near-uniform face output" mechanism documented since
round 1).

### Secondary comparisons (original eGeMAPS-based tests)

| comparison | diff (seed-avg) | 95% CI | p |
|---|---|---|---|
| fixed vs audio-only [eGeMAPS, clean-trained] | +0.0311 | [−0.0125, +0.0753] | 0.1690 |
| fixed vs audio-only [WavLM, clean-trained] | +0.0098 | [−0.0098, +0.0263] | 0.3360 |
| fixed vs audio-only [eGeMAPS, `aug_all`] | +0.1221 | [+0.0767, +0.1681] | 0.0000 |
| fixed vs audio-only [WavLM, `aug_all`] (= P1 pooled) | +0.0404 | [+0.0182, +0.0629] | 0.0010 |
| fixed vs audio-only [eGeMAPS, `aug_no_mask`] | +0.1085 | [+0.0582, +0.1595] | 0.0010 |
| fixed vs audio-only [WavLM, `aug_no_mask`] | +0.0366 | [+0.0122, +0.0596] | 0.0020 |

Consistent with the headline: fusion only reliably beats audio-only (both CIs excluding zero) when the
face branch is occlusion-robust (`aug_all`/`aug_no_mask`); with the clean-trained face, neither audio
branch's fusion-vs-audio-only comparison reaches significance pooled — the clean-trained face's gains
under light occlusion are offset by its losses under `mask`, netting out to a non-significant pooled
effect, exactly the mechanism documented in rounds 2-3 and round 6-7.

### Limitations, stated plainly

- B=2000 bootstrap resamples (not 10,000 as originally specified in the un-amended plan) — chosen for
  this run's turnaround time; CIs are already stable at this B given 12 actors, but a future re-analysis
  could widen B for marginally tighter CI edges.
- The learned gate, which underperformed on val in every prior round, is much more competitive on test
  (e.g. `aug_all`+WavLM clean: 0.776 vs fixed's 0.778) — still never the best method, but closer than the
  val-only development rounds suggested. Worth noting as a data point, not over-interpreting from a
  single test-set run.
- Per the amendment, calibration/gate were frozen from VAL and never refit on test — a cleaner leakage
  story, at the cost of whatever precision a test-specific fit might have bought. Fitted temperatures
  (per seed, since the face/WavLM models differ per seed — eGeMAPS is deterministic so its temperature
  doesn't vary): eGeMAPS 1.2751 (all 3 seeds); WavLM 1.2174 / 1.1357 / 1.2459; face clean-trained 0.9222
  / 0.7317 / 1.2174; face `aug_all` 1.1623 / 0.8026 / 1.0353; face `aug_no_mask` 0.9659 / 0.9438 / 0.9885
  — all in a fairly narrow band around 1.0, i.e. the original softmax outputs were already reasonably
  calibrated, with no single seed's face model wildly over- or under-confident.

### Derived from the committed test predictions (no re-run)

Everything below is computed *only* from the already-saved outputs of the single `--final` run above
(`outputs/evaluate_test_by_seed.csv`, `..._table.csv`, `..._per_clip.csv`) — no model inference, no
re-run of `evaluate.py --final`, nothing about the frozen run changes. `scripts/27_test_derived_analysis.py`.

**1. Fixed 50/50 fusion on CLEAN faces (+ WavLM), per-seed and mean ± std:**

| face variant | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| clean-trained | 0.7851 | 0.7846 | 0.7565 | 0.7754 ± 0.0163 |
| `aug_all` | 0.7836 | 0.7835 | 0.7671 | 0.7781 ± 0.0095 |
| `aug_no_mask` | 0.7783 | 0.7900 | 0.7659 | 0.7781 ± 0.0120 |

This was the one gap in the "Fusion, pooled over the 12 occlusion conditions" table above (which
deliberately excludes clean) and the "clean-condition fixed-50/50" row the VAL round-7 table had but
this test write-up hadn't filled in until now.

**2. Per-seed scores for rows already reported above:**

Face-only, clean condition (cross-checks the mean ± std already in "Face branch alone, clean
condition" exactly):

| variant | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| clean-trained | 0.6062 | 0.5893 | 0.6195 | 0.6050 ± 0.0151 |
| `aug_all` | 0.6111 | 0.6283 | 0.6286 | 0.6227 ± 0.0100 |
| `aug_no_mask` | 0.6335 | 0.6421 | 0.6153 | 0.6303 ± 0.0137 |

Pooled over the 12 occlusion conditions, `fixed_5050` only (full 36-row table — every method × variant
× audio branch — in `outputs/test_derived_pooled_per_seed.csv`; means below cross-check exactly against
the "Fusion, pooled over the 12 occlusion conditions" table above):

| face variant | audio branch | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| clean-trained | eGeMAPS | 0.5199 | 0.5237 | 0.5224 | 0.5220 ± 0.0019 |
| `aug_all` | eGeMAPS | 0.6239 | 0.6189 | 0.6169 | 0.6199 ± 0.0036 |
| `aug_no_mask` | eGeMAPS | 0.6121 | 0.6098 | 0.5965 | 0.6061 ± 0.0084 |
| clean-trained | WavLM | 0.7033 | 0.7206 | 0.7075 | 0.7105 ± 0.0091 |
| `aug_all` | WavLM | 0.7372 | 0.7534 | 0.7367 | 0.7424 ± 0.0095 |
| `aug_no_mask` | WavLM | 0.7360 | 0.7472 | 0.7325 | 0.7386 ± 0.0077 |

**3. Actor-level bootstrap 95% CIs for absolute macro-F1** (resample the 12 test actors, B=2000,
seed-averaged per draw — identical method to P1-P3, just a single arm instead of a paired difference):

| | point estimate | 95% CI |
|---|---|---|
| eGeMAPS (tuned) | 0.5044 | [0.4576, 0.5424] |
| WavLM | 0.7049 | [0.6269, 0.7616] |
| face clean-trained (clean) | 0.6050 | [0.5034, 0.6987] |
| face `aug_all` (clean) | 0.6227 | [0.5207, 0.7143] |
| face `aug_no_mask` (clean) | 0.6303 | [0.5196, 0.7279] |
| fixed fusion (`aug_all` + WavLM, clean) | 0.7781 | [0.7071, 0.8400] |
| fixed fusion (`aug_all` + WavLM, pooled 12) | 0.7425 | [0.6663, 0.8032] |

All CIs are wide relative to the point estimate — expected at only 12 test actors, same caveat as every
other actor-level estimate in this project. Note these are independent single-arm CIs, so e.g. fixed
fusion's [0.7071, 0.8400] and WavLM's [0.6269, 0.7616] visibly overlap even though P1 found the paired
difference between them significant — standalone CIs computed this way don't account for the
actor-to-actor correlation between the two scores on the *same* resampled actors, which is exactly why
P1's paired bootstrap (not these single-arm CIs) is the actual significance test for that comparison.
