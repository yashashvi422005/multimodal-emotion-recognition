# Final Evaluation Plan (pre-registered)

**Status: PRE-REGISTERED, NOT YET EXECUTED.** The test set (12 actors, frozen per `splits/split_v1.json`)
has not been touched by this plan or by any development work to date. All Phase 3-5 development (rounds
1-3) used train/val actors only. This document is committed *before* any test-set code is run, so the
plan is fixed in advance of seeing test numbers. A future `evaluate.py --final` (not yet written) is the
only script permitted to touch the test set, and it must implement exactly what's specified here — any
deviation discovered necessary while writing it must be logged and justified in this file, not silently
applied.

## Models (no retraining required for this plan)

- **Audio**: eGeMAPS features + StandardScaler + LogisticRegression, the exact fitted Phase 3 pipeline
  (refit once on train actors, deterministic, `std ~= 0` across seeds). No seed variation.
- **Face**, three variants, each already trained on train actors only (val/early-stopping used val
  actors only; test actors were never seen by any of the three during training, monitoring, or model
  selection), 3 seeds each, existing checkpoints:
  - `clean-trained` — `models/face_resnet18_seed{0,1,2}.pt`
  - `aug_all` (all 4 occlusion types seen during training) — `models/face_resnet18_aug_all/resnet18_aug_all_seed{0,1,2}.pt`
  - `aug_no_mask` (mask held out of training) — `models/face_resnet18_aug_no_mask/resnet18_aug_no_mask_seed{0,1,2}.pt`

No model is retrained for this evaluation. All 9 face checkpoints already exist.

## Conditions (all 13, same as every prior round)

`clean` + {`mask`, `sunglasses`, `patch`, `hand_over_mouth`} × {0.3, 0.6, 0.9} severity.

Occlusion generation uses the same code as development (`fer.face_pipeline`, v2 mask geometry). The
MediaPipe detection cache must be **rebuilt for test clips** — the existing cache
(`cache/phase4_occlusion_detection_cache.parquet`) covers val clips only; detection depends on the
occluded image, so a fresh cache build (CPU, multiprocess, same methodology as `scripts/13`) is needed,
but this is cheap (detection only, no model inference) and was ~1 min for the mask-only rebuild on val.

## Fusion variants (all reported, for each of the 3 face variants)

- `audio_only`, `face_only`
- `fixed_5050`
- `conf_maxprob` (confidence-weighted, max-softmax)
- `conf_negentropy` (confidence-weighted, negative entropy)
- `learned_gate` (logistic regression on `[face_max_prob_conf, audio_max_prob_conf]`)

Temperature calibration and the learned gate are each **refit on test actors**, using the same 2-fold
actor cross-fit procedure as development (fit on one fold's clean data, apply to the other fold, and vice
versa; gate fit pooled across clean + every occlusion condition). Refitting on test — rather than reusing
val-fitted parameters — is necessary because a calibration or gate fit on val actors has no guarantee of
transferring to test actors' particular confidence distribution; the cross-fit discipline (never scoring a
clip with a temperature/gate fit on its own fold) is preserved exactly, just applied to the 12 test actors
instead of the 15 val actors.

## Metrics

- **Primary**: macro-F1.
- Secondary: accuracy, per-class F1, confusion matrix.
- Every number: **mean ± std over the 3 face-model seeds** (audio and the calibration/gate fits are
  deterministic given the data, so they contribute no seed variance of their own).

## Statistical testing: paired bootstrap over TEST ACTORS

Two pre-registered comparisons, both using **actor-level** (not clip-level) paired bootstrap resampling,
because clips from the same actor are not independent observations:

1. **Fusion vs. audio-only.** For each of the 3 face-model variants, compare `fixed_5050` macro-F1 against
   `audio_only` macro-F1. Reported two ways: pooled across all 13 conditions, and condition-by-condition.
2. **`aug_no_mask` vs. `clean-trained` face, under mask only.** Compare face-only macro-F1 at each of the
   3 mask severities separately. `aug_all` is deliberately excluded from this specific comparison — it
   saw the same synthetic mask-generation code during training, so it cannot test generalization to an
   unseen occlusion the way `aug_no_mask` can (see `RESULTS.md`, round 3, for why this distinction
   matters).

**Bootstrap procedure**: resample the 12 test actors with replacement, B = 10,000 times. For each
resample, pool all clips belonging to the resampled actors (with repeats, per standard actor-level
bootstrap) and compute the metric for both methods being compared on that pooled set; record the paired
difference. Report: the point estimate on the real (unresampled) data, a 95% CI (2.5th/97.5th percentiles
of the bootstrap differences), and a two-sided p-value (`2 * min(mean(diff <= 0), mean(diff >= 0))`).

## Reporting commitment

**All results from this plan will be reported in `RESULTS.md` and `README.md` exactly as they come out,
whatever they show** — including a null or negative result for either pre-registered comparison, or for
any individual model/condition/fusion-method combination. Nothing specified here will be dropped,
reworded, or supplemented with an unplanned replacement comparison after seeing the test numbers. Any
additional, clearly-labeled exploratory analysis done *after* this plan's primary results are reported
must be marked as exploratory and must not replace or overshadow the pre-registered numbers.

## Explicitly out of scope for this plan

- The demographic (sex/age/race) breakdown — that uses a separate, dedicated 5-fold actor-level
  cross-validation over the 79 **non-test** actors (see the corresponding section of `RESULTS.md`/
  `README.md`), and does not touch the test set at all.
- Any new model training, hyperparameter change, or methodology change motivated by test-set results —
  the models and methods are frozen as of this document.

## Execution status

**Not yet run.** This document is committed first; `evaluate.py --final` will be written next and will
implement exactly the plan above. No test-set code has been executed as part of this round of work.
