# Final Evaluation Plan (pre-registered)

**Status: EXECUTED.** `python evaluate.py --split test --final` ran exactly once, 2026-11-14, against
the pre-registration frozen at commit `09e454e981079e7197b46c9c92b0fc2b9ace1a74` (the amendment below).
All Phase 3-5 development (rounds 1-7) used train/val actors only; this plan was committed, then amended
once, before any test-set code ran, so the methodology was fixed in advance of seeing test numbers.
`evaluate.py` implemented exactly what's specified here (as amended) — no deviation was found necessary
while writing it. Results: `RESULTS.md`, "Final test results." Nothing in this plan changed after
seeing those results.

## Models (no retraining required for this plan) — LOCKED as of round 6

- **Audio, classical (eGeMAPS)**: eGeMAPS features + StandardScaler + LogisticRegression, the exact
  fitted Phase 3 pipeline, **tuned**: `C=0.3` (chosen by a small grid search, `C ∈ {0.01...100}`, on
  VAL macro-F1 — see `RESULTS.md` round 6 and `outputs/egemaps_C_tuning_grid.csv`), refit once on train
  actors, deterministic, `std ~= 0` across seeds. No seed variation. The earlier untuned `C=1.0` run is
  kept in `outputs/egemaps_tuned_vs_untuned.csv` for the record but is **not** used for the test-set
  evaluation — tuned eGeMAPS is what "eGeMAPS" means from here on.
- **Audio, WavLM**: `microsoft/wavlm-base-plus` fine-tuned (frozen CNN feature encoder, small
  classification head), trained on TRAIN actors, early-stopped on VAL actors' macro-F1, 3 seeds
  (`notebooks/wavlm_finetune_kaggle.ipynb`, run on Kaggle — see that notebook and the project README for
  the exact procedure). **Training complete**: 3 checkpoints in `models/wavlm/` (gitignored, local
  only). Inference uses `batch_size=1` (no padding at all), not Kaggle's original batched val-set
  probabilities — round 7 (`RESULTS.md`) found the frozen CNN feature encoder is not
  batch-composition-invariant, so `batch_size=1` is the only way to get a well-defined, reproducible
  per-clip score; this is what `evaluate.py` and `outputs/wavlm_val_probs_batch1.csv` both use.
  **The Kaggle run never touched test actors** (enforced by an assertion inside the notebook, confirmed again locally by checking the
  downloaded val probabilities' actor set against the official split). This plan's test-set evaluation
  loads those 3 already-trained checkpoints for inference only — no retraining happens as part of this
  plan, same as the face models.
- **Face**, three variants, each already trained on train actors only (val/early-stopping used val
  actors only; test actors were never seen by any of the three during training, monitoring, or model
  selection), 3 seeds each, existing checkpoints:
  - `clean-trained` — `models/face_resnet18_seed{0,1,2}.pt`
  - `aug_all` (all 4 occlusion types seen during training) — `models/face_resnet18_aug_all/resnet18_aug_all_seed{0,1,2}.pt`
  - `aug_no_mask` (mask held out of training) — `models/face_resnet18_aug_no_mask/resnet18_aug_no_mask_seed{0,1,2}.pt`

No model is retrained for this evaluation. All 9 face checkpoints and (once the Kaggle step completes)
3 WavLM checkpoints already exist before this plan's test-set code runs.

## Conditions (all 13, same as every prior round)

`clean` + {`mask`, `sunglasses`, `patch`, `hand_over_mouth`} × {0.3, 0.6, 0.9} severity.

Occlusion generation uses the same code as development (`fer.face_pipeline`, v2 mask geometry). The
MediaPipe detection cache must be **rebuilt for test clips** — the existing cache
(`cache/phase4_occlusion_detection_cache.parquet`) covers val clips only; detection depends on the
occluded image, so a fresh cache build (CPU, multiprocess, same methodology as `scripts/13`) is needed,
but this is cheap (detection only, no model inference) and was ~1 min for the mask-only rebuild on val.

## Fusion variants (all reported, for each of the 3 face variants × each of the 2 audio variants = 6 face/audio pairings)

- `audio_only`, `face_only`
- `fixed_5050`
- `conf_maxprob` (confidence-weighted, max-softmax)
- `conf_negentropy` (confidence-weighted, negative entropy)
- `learned_gate` (logistic regression on `[face_max_prob_conf, audio_max_prob_conf]`)

Each of the 6 fusion methods above is run once per (face variant × audio variant) pairing — 3 face
variants × 2 audio variants (classical eGeMAPS+LR, WavLM) = 6 pairings, so 36 (method × pairing)
combinations in total across all 13 conditions. This is a straightforward extension of the existing
fusion methodology to a second audio branch, not a new design — WavLM's calibration and fusion follow
exactly the same procedure as the classical audio branch's.

**⚠️ Amended 2026-11-13 — see "Amendment" section below.** This paragraph describes the *original*
plan and is kept for the record; it no longer describes what `evaluate.py` does. Temperature calibration
and the learned gate were originally specified to be **refit on test actors**, using the same 2-fold
actor cross-fit procedure as development (fit on one fold's clean data, apply to the other fold, and vice
versa; gate fit pooled across clean + every occlusion condition). Refitting on test — rather than reusing
val-fitted parameters — was reasoned to be necessary because a calibration or gate fit on val actors has
no guarantee of transferring to test actors' particular confidence distribution; the cross-fit discipline
(never scoring a clip with a temperature/gate fit on its own fold) was preserved exactly, just applied to
the 12 test actors instead of the 15 val actors. WavLM's calibration was fit the same way, independently
of the classical audio branch's.

## Metrics

- **Primary**: macro-F1.
- Secondary: accuracy, per-class F1, confusion matrix.
- Every number: **mean ± std over 3 seeds** — the 3 face-model seeds for face-only/fusion rows, and the
  3 WavLM seeds for WavLM-involving rows (classical audio and the calibration/gate fits are deterministic
  given the data, so they contribute no seed variance of their own).
- **Abstention rule** (committed 2026-11-13, previously implicit): scored via
  `fer.metrics.score_with_abstention`, the project's single official scoring method throughout
  development. An abstained clip (the face branch outputs a uniform distribution when zero sampled
  frames survive detection under an occlusion condition) counts as a false negative for its true class
  only, is never counted as a prediction of any class, and still counts in the accuracy denominator.
  eGeMAPS and WavLM never abstain — only the face branch can.

## Statistical testing: paired bootstrap over TEST ACTORS

**⚠️ Amended 2026-11-13 — see "Amendment" section below.** This section describes the *original* two
comparisons and is kept for the record; they are now secondary (still computed and reported), and three
new comparisons (P1/P2/P3) are primary.

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
of the bootstrap differences), and a two-sided p-value (`2 * min(mean(diff <= 0), mean(diff >= 0))`). This
procedure is unchanged by the amendment below — only which method pairs are primary has changed.

**Note on WavLM's role in the two original comparisons above**: both were pre-registered using the
classical eGeMAPS+LR audio branch specifically, and that is what they continue to test. WavLM's own
results are still fully reported.

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

## Amendment, 13 November 2026, before any test-set access

Four changes, decided before `evaluate.py --final` is ever run. **The test set has still not been
touched** — these are methodology decisions made looking only at VAL, same as every other round. Each
change below states what it replaces and why.

### 1. Calibration and the learned gate: fit on VAL, frozen, applied to TEST

**Change.** Temperature calibration (one scalar per audio branch, and one per face-variant) and the
learned gate are now each fit **once on ALL val actors** — the same methodology as development
(temperature fit on clean-only data via `fer.calibration.fit_temperature`; gate fit on
`[face_max_prob_conf, audio_max_prob_conf]` pooled across clean + every occlusion condition via logistic
regression, now `fer.fusion.fit_gate_on_all`) — and then **frozen**. The frozen temperature(s) and gate
are applied directly to the test actors' probabilities; test labels are used only to *score* predictions
against ground truth, never to fit anything. This replaces the original plan's 2-fold cross-fit refit on
the 12 test actors themselves (`fer.fusion.cross_fit_calibrate_clean_only` / `fit_gate_cross_fit`, still
used unchanged by `scripts/18` and `scripts/23` for their own VAL-only development analyses — only
`evaluate.py`'s methodology changes here).

**Reason.** The original cross-fit-on-test approach was internally consistent (no clip was ever scored
by a temperature or gate fit on its own fold) but it still let the test actors' own labels inform the
fitted parameters that then scored those same 12 actors — just via a different fold, not the clip's own.
That is a milder form of information leakage from test into the evaluation than fitting-and-scoring on
identical rows, but it is a real one: a calibration/gate genuinely "sees" test-actor data (labels
included) before any test number is reported. Fitting only on VAL, before test is touched at all,
removes this category of concern entirely — nothing about the frozen parameters depends on any test
label, full stop. The cost is that a calibration/gate fit on 15 val actors might transfer less precisely
to the 12 test actors' confidence distribution than a test-specific fit would; this is accepted as the
right trade for a cleaner separation between "what was decided looking at test" (nothing) and "what is
reported from test" (everything).

### 2. Three new primary pre-registered comparisons (P1, P2, P3)

**Change.** The primary statistical tests are now:

- **P1 — Fusion vs. voice alone.** `fixed_5050` fusion (`aug_all` face + WavLM) vs. `audio_only` (WavLM
  alone), across clean and all 12 occlusion conditions (pooled, and condition-by-condition). Question:
  does the face add anything to a strong voice model?
- **P2 — Occlusion-augmented training generalization.** `aug_no_mask` vs. clean-trained face-only, under
  the 3 mask severities. Unchanged from the original plan (comparison 2 above) — still excludes `aug_all`
  for the same reason (it saw mask during training).
- **P3 — Does confidence-weighting beat fixed averaging, once both are calibrated?** Confidence-weighted
  fusion (max-prob, calibrated) vs. `fixed_5050`, both using `aug_all` face + WavLM. This is the
  project's original round-1 research question, now run as a formal pre-registered test on TEST rather
  than the informal VAL observation it started as.

The original comparison 1 (fusion vs. audio-only, eGeMAPS, all 3 face variants) and the eGeMAPS-paired
version of comparison 2's partner tests become **secondary** — still computed, still fully reported, just
no longer the headline tests.

**Bootstrap mechanics for P1–P3, stated explicitly (clarifies, does not change, the "Statistical
testing" section above):**

- **The resampling unit is the actor.** Each of the B bootstrap draws picks `len(test_actors)`-many
  actors with replacement from the 12 test actors. For a drawn actor, **every one of its clips across
  every condition in the comparison's scope is pooled together** in that draw — there is no separate
  clip-level resampling and no separate condition-level resampling. For the "pooled across all 13
  conditions" number, a drawn actor's clips from clean *and* all 12 occlusion conditions are pooled in
  the same draw; for a per-condition number, the actor's clips are restricted to that one condition
  first, then resampled the same way. (Verified in code: a unit check with two seeds given identical
  data confirms every draw's per-seed diff is bit-identical, proving the same resampled actor set is
  applied to both — see `evaluate.py`'s `paired_bootstrap`.)
- **All 3 seeds are combined within each draw, not averaged after independent bootstraps.** For each of
  the B draws, the *same* resampled actor set is reused for all 3 seeds; the paired macro-F1 difference
  is computed per seed on that draw's resampled set, and the 3 per-seed diffs are averaged into one
  number for that draw. The **primary result is this seed-averaged diff's distribution over B draws**
  (point estimate = mean of the 3 seeds' real, unresampled diffs; 95% CI = the 2.5th/97.5th percentile
  of the B seed-averaged bootstrap diffs). Per-seed point diffs are also reported alongside, but they are
  not the primary number and do not each get their own independently-bootstrapped CI reported as primary.

**Reason.** eGeMAPS (macro-F1 ≈ 0.50) turned out to be a weak audio branch relative to WavLM
(≈ 0.69); the original P1-equivalent (fusion vs. eGeMAPS-only) is a low bar that almost any face signal
clears, and round 6/7 already showed it's not where the interesting behavior is (fixed fusion losing to
*audio-only* only shows up once the audio branch is strong — see `RESULTS.md` rounds 6-7). Testing against
WavLM is the harder, more informative question, and the one this project's core research question (does
fusion help once voice is already strong) actually depends on. P3 promotes the project's very first
research question — whether confidence-weighting beats fixed averaging — to a formal test, since round 1
only ever evaluated it informally on VAL; running it as a pre-registered TEST comparison is a more honest
way to finally answer the original hypothesis.

### 3. Abstention rule committed to this document

**Change.** Added directly to the "## Metrics" section above (not just stated once in `fer/metrics.py`'s
docstring): an abstained clip is a false negative for its true class only, never a prediction of any
class, and still counts in the accuracy denominator.

**Reason.** This project's single official scoring rule had never actually been written into the plan
that governs the test-set run — a real gap, found while reviewing the plan section by section. Nothing
about the rule itself changed; it is now just explicit where it needs to be.

### 4. Val dry run re-confirmed with the frozen val-fitted calibration and gate

**Change/confirmation.** `evaluate.py --split val --seeds 0` was re-run with the new frozen-calibration
code path (item 1 above) — fitting temperature/gate once on the 15 val actors and applying the frozen
result back to those same 15 actors, structurally identical to what will happen for test except the
scoring set. See `RESULTS.md` / this run's output for the confirmed numbers; this is a code-path check,
not a new result, since fitting and scoring on the same actors is a degenerate case of the real test
workflow.

**Reason.** Confirms the new code works end-to-end (fitting, freezing, applying, scoring, and the three
new bootstrap comparisons P1/P2/P3) before it is ever pointed at the frozen test set.

## Execution status

**Done.** `python evaluate.py --split test --final` ran exactly once, 2026-11-14, under `caffeinate -i`
to prevent a sleep-related stall during the ~7-hour run, immediately after the amendment commit
(`09e454e`) was pushed. Preceded by `scripts/26_build_test_detection_cache.py` (2 workers, ~6.6 min, all
4 occlusion types × 3 severities × 12 test actors, detection rate 95.2%) — the one prerequisite this
document flagged as missing. No code or methodology changed after seeing the test results. Full numbers:
`RESULTS.md`, "Final test results."

**WavLM's training step ran on Kaggle, separately from this plan's test-set code** — the Kaggle notebook
itself never touched test actors (see `notebooks/wavlm_finetune_kaggle.ipynb`), confirmed again by this
run's own leakage assertions.

**Every model this plan needed was locked and present**: tuned eGeMAPS (`C=0.3`), WavLM (3 checkpoints,
scored with `batch_size=1` per round 7), and the 3 face-model variants (clean-trained, `aug_all`,
`aug_no_mask`, 3 seeds each). All 6 fusion methods were exercised on VAL first (`RESULTS.md` rounds 2-3
for eGeMAPS, round 6-7 for WavLM) before ever touching test.

**`evaluate.py`** (repo root) implements this plan's full pipeline as amended 2026-11-13: both audio
branches, all 3 face variants, all 6 fusion methods, all 13 conditions, calibration/gate fit once on VAL
and frozen, and the P1/P2/P3 primary bootstrap comparisons (plus the original comparisons as secondary).
Dry-run-verified on `--split val` (twice — once per bootstrap-code revision) before the real run;
face-branch and eGeMAPS numbers reproduced `RESULTS.md` exactly both times. `--split test` required
`--final` and the pre-built test detection cache, both satisfied for this run.
