# fer-project: Multimodal (face + voice) emotion recognition under occlusion

Research question: when the face is occluded by synthetic occlusions (masks, sunglasses,
hand-over-mouth, random patches), does a late-fusion model with confidence weighting shift reliance
onto the voice and recover performance that a face-only model loses?
Secondary question: does performance differ across demographic groups
(age, sex, race, ethnicity)?

Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) only
(7,442 clips, 91 actors, 6 emotions). **Not** FER2013. Already present at
`data/CREMA-D/` — not downloaded or modified by this repo's code.

## Final test results (frozen test set, 12 actors — complete)

The research question above is answered on the frozen 12-actor test set, evaluated exactly once
(`python evaluate.py --split test --final`, 2026-11-14) against the plan pre-specified and committed at
`09e454e` before any test-set access.

| | Test macro-F1 (mean ± std, 3 seeds) | Actor-bootstrap 95% CI |
|---|---|---|
| eGeMAPS + LR (tuned, deterministic) | 0.5044 ± 0.0000 | [0.4576, 0.5424] |
| WavLM (per-seed: 0.6887, 0.7175, 0.7086) | 0.7049 ± 0.0121 | [0.6269, 0.7616] |
| Face (clean-trained), clean | 0.6050 ± 0.0151 | [0.5034, 0.6987] |
| Face (`aug_all`), clean | 0.6227 ± 0.0100 | [0.5207, 0.7143] |
| Face (`aug_no_mask`), clean | 0.6303 ± 0.0137 | [0.5196, 0.7279] |
| Fixed 50/50 fusion (`aug_all` face + WavLM), clean | **0.7781 ± 0.0095** | [0.7071, 0.8400] |
| Fixed 50/50 fusion (`aug_all` face + WavLM), pooled over 12 synthetic-occlusion conditions | **0.7424 ± 0.0095** | [0.6663, 0.8032] |

Std convention, exactly as committed in `RESULTS.md` and not unified here: eGeMAPS's and WavLM's ± is
**population** standard deviation (ddof=0, from `evaluate.py`'s scoring loop); every face and fusion
row's ± is **sample** standard deviation (ddof=1, from the derived-analysis script). Confidence
intervals are an actor-level bootstrap (B=2000, 12 test actors, seed-averaged per draw), from
`RESULTS.md`'s "Derived from the committed test predictions" section, commit `6f247d9`.

- **P1** (fusion vs. WavLM alone): the fusion significantly beats voice alone in every one of 13
  conditions (clean + 12 synthetic-occlusion conditions), with the effect largest on clean faces
  (+0.073) and shrinking to about +0.01 under the heaviest occlusion (`mask`/`patch` at severity 0.9),
  but never losing significance or reversing sign.
- **P2** (`aug_no_mask` vs. clean-trained face-only, under the held-out mask type): significant at
  every severity — occlusion-augmented training generalizes to an unseen occlusion type, though
  hand-over-mouth, seen in training, covers a similar lower-face region to the held-out mask.
- **P3** (confidence-weighted vs. fixed fusion): confidence-weighted fusion shows **no improvement
  over fixed averaging** — pooled, the difference is statistically detectable but practically
  negligible (−0.004) and favors fixed averaging, and this is **not uniform** across the 13 individual
  conditions. The original hypothesis was not supported.

Full numbers, every condition, including results that go against the project's own hypotheses:
`RESULTS.md`, "Final test results". Everything in the rest of this README below the validation-stage
heading is development-stage (VAL only), superseded by the table above for any test-set claim.

## Limitations

- **Synthetic, landmark-placed occlusions.** All occlusion types are rendered digitally on detected
  face landmarks, not captured from genuinely occluded recordings.
- **Acted speech.** CREMA-D emotions are performed by actors reading fixed sentences, not spontaneous
  affect.
- **12 test actors, wide confidence intervals.** Every test-set CI above reflects the actor-sampling
  uncertainty of only 12 actors.
- **A single dataset.** All results are from CREMA-D; no cross-corpus evaluation was performed.
- **The age-gap finding** (see the demographic fairness section below) **is a development-data
  observation only** — computed on the 79 non-test actors with 1 training seed per fold, and was never
  tested on the 12-actor frozen test set.

## Environment

A `venv` (not conda) lives at `.venv/`, built against the system Python
3.11.9. To activate it:

```bash
source .venv/bin/activate
```

Dependencies are pinned in `requirements.txt` (generated via `pip freeze`
from the unpinned top-level list in `requirements.in`). To reproduce the
environment from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch runs on the Apple Silicon **MPS** backend where supported, falling
back to CPU otherwise (no CUDA on this machine). The one phase that needed
large-scale GPU training — WavLM fine-tuning — was moved to Kaggle (see
"WavLM audio fine-tuning" below); no other phase required it.

### Decisions made during environment setup (verified, not assumed)

- **Audio I/O: use `soundfile`, not `torchaudio.load`.** The installed
  `torchaudio` (2.11.1, the latest compatible with this torch/Python/platform
  combo) has deprecated its built-in sox/soundfile backends in favor of a
  separate `torchcodec` package; `torchaudio.load()` raises `ImportError`
  without it. Rather than add that extra dependency now, all audio file
  reading in this project goes through `soundfile` (already verified against
  real CREMA-D `.wav` files), converted to a `torch.Tensor` directly;
  `torchaudio.transforms` (resampling, spectrograms, etc.) are still used on
  tensors already in memory, which works fine standalone.
- **MPS confirmed working**: `torch.backends.mps.is_available()` is `True`
  on this machine (torch 2.15.0).
- **`mediapipe` 1.2.2 exposes a `mediapipe.tasks`-based API**, not the older
  `mediapipe.solutions.face_mesh` module some tutorials reference. The exact
  Face Landmarker API was confirmed in Phase 2 (face detection): `FaceLandmarker`
  (478 landmarks), with a RetinaFace fallback.

## Non-negotiable evaluation rules (see project instructions for full detail)

1. Actor-level, speaker-independent splits — no actor in more than one of
   train/val/test. Enforced by an assertion that runs every time a split is
   loaded (`fer.splits.load_split` / `assert_no_leakage`), not just in tests.
2. The test set was fixed on day one (`splits/split_v1.json`) and was touched exactly once, by
   `evaluate.py --final` (see "Final test results" above).
3. Macro-F1 is the primary metric, reported alongside accuracy, per-class F1,
   and confusion matrices.
4. Every reported number is mean ± std over 3 seeds.
5. Every run is tracked in MLflow (local by default; W&B via a switch).
6. No invented numbers. Failures are reported plainly.

## Dataset citation and licence

This project uses **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset):

> H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, "CREMA-D: Crowd-sourced
> Emotional Multimodal Actors Dataset," *IEEE Transactions on Affective Computing*, vol. 5, no. 4,
> pp. 377–390, 2014.

CREMA-D is released under the Open Database License (ODbL) v1.0. The underlying audio/video content
is released under the Database Contents License (DbCL) v1.0. See the dataset's own distribution for
the full licence text; this project does not redistribute the dataset itself (`data/` is gitignored —
see "Environment" below), only derived features, model outputs, and code.

**This project's own code is MIT-licensed** (see [`LICENSE`](LICENSE)). That licence covers this
repository's code only — CREMA-D itself remains under its own ODbL/DbCL licence as above and is not
redistributed here.

**To obtain the data**: clone or download it from the official repository,
[github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), and
place it at `data/CREMA-D/` with `VideoFlash/`, `AudioWAV/`, and `VideoDemographics.csv` present (see
`configs/data.yaml` for the exact expected paths). This repo never downloads or modifies the dataset
itself. The `ejlok1/cremad` Kaggle mirror, used only for the WavLM fine-tuning step (see below), is a
secondary source for the audio and is missing one canonical clip, supplied separately — see "WavLM
audio fine-tuning" below.

## Validation-stage results (development only — superseded by the final test results above)

Everything from here through the demographic fairness check below was computed on the **validation
set only** (or train+val combined, for cross-validation), during development, before the frozen test
set was touched. It is kept in full as the project's development record, but none of it is a test-set
claim — see "Final test results" above for the one and only evaluation on the frozen 12-actor test set.
All occlusions in this log are synthetic (landmark-placed).

## Phase 1: data verification, metadata, splits

```bash
source .venv/bin/activate

# 1. Scan the real files (not just filenames), build the per-clip metadata
#    table, join demographics, report class counts / anomalies.
python scripts/01_verify_and_build_metadata.py
# -> outputs/metadata.csv

# 2. Propose a demographically-stratified actor split (status: "proposed").
#    Does NOT touch the test set's "frozen" status.
python scripts/02_make_split.py
# -> splits/split_v1.json

# 3. After the proposed split has been reviewed and approved, freeze it.
#    This only flips status "proposed" -> "frozen" and timestamps it; it
#    does not recompute or alter the actor assignment or its hash.
python scripts/03_freeze_split.py splits/split_v1.json

# Run the leakage test suite (also runs automatically whenever a split is loaded):
python -m pytest tests/ -v
```

### Known upstream data quirks (confirmed by direct file inspection, not assumed)

- `data/CREMA-D/VideoFlash/1015_DFA_ANG_XX.mp4` is an extra, non-canonical
  file (official CREMA-D ships a correctly-named `.flv` for the same clip
  alongside it — confirmed via the dataset mirror's own git history). This
  is the file responsible for VideoFlash's 7,443 vs AudioWAV's 7,442 count.
  Excluded from the canonical 7,442-clip metadata table.
- `1064_IEO_DIS_MD`: video is ~61.8s and is actually a compilation of all
  emotional displays for that sentence — the `DIS` (disgust) label does
  **not** describe its content. Matches README's documented note.
- `1064_TIE_SAD_XX`, `1076_MTI_NEU_XX`: near-zero video duration (0.57s,
  0.07s) — degenerate/corrupt video, though audio is intact. Matches
  README's documented notes.
- `1076_MTI_SAD_XX`: a "Media offline" placeholder slate (text graphic in 7
  languages), not footage. Not caught by Phase 1's container-level probing
  (duration/audio-stream metadata looked normal) — found during the Phase 2
  full run, where it failed MediaPipe face detection on all 26 sampled
  frames. Matches `README.md`'s documented issue for this file ("error
  picture, no audio, very short").
- 31 clips are H.264-encoded vs. VP6 (`vp6f`) for the rest; 28 of these
  exactly match the README's list of clips that had audio re-attached in
  Dec. 2020. The other 3 are not individually documented in the README but
  show the same re-encode signature. Kept — codec choice doesn't affect
  label validity.

**Decision (approved at CHECKPOINT 1, extended after the Phase 2 full run):
exclude the 4 clips above from the canonical dataset.** Final clip count:
**7,438** (7,442 raw, minus the 4 excluded). Per-emotion counts after
exclusion: anger 1271, fear 1271, happy 1271, disgust 1270, sad 1269,
neutral 1086. The frozen actor split (`splits/split_v1.json`) is defined by
actor, not clip, so this exclusion does not touch it — content hash
verified unchanged.

## Split limitations

`splits/split_v1.json` is frozen (seed 42, Sex×Race-stratified, per-stratum
largest-remainder rounding). Known limitations, carried forward rather than
hidden:

- **Female-Asian (2 actors total) and Unknown-race (1 actor total) appear
  only in train** — zero in val and test. This is a dataset-size limit, not
  a choice: with only 2 or 1 actors in a stratum, no split algorithm can
  place at least one in train, val, *and* test simultaneously.
- **Val has no 60–74 age-band actors** (train: 4, test: 2, val: 0).
- **With only 12 test actors, per-demographic-group test results are
  anecdotal**, not statistically defensible. For the demographic-fairness
  analysis (secondary research question), plan to use **actor-level
  cross-validation over the 79 non-test actors** (train+val) instead of
  relying on the fixed test set's per-group breakdown. Always report the
  number of actors contributing to each group's score, and **never report
  a per-group score for a group with fewer than 3 actors** in that fold.

## Phase 3 (partial): classical audio baseline — eGeMAPS + logistic regression

Run out of phase order, at explicit request, before Phase 2.

```bash
python scripts/04_extract_audio_features.py       # caches eGeMAPSv02 functionals for TRAIN+VAL clips only
python scripts/05_audio_baseline_egemaps_lr.py     # fit on TRAIN, evaluate on VAL, 3 seeds, vs. chance
python scripts/06_audio_baseline_uncertainty.py    # speaker-level uncertainty (bootstrap + actor-CV)
```

- 6,460/6,460 train+val clips extracted cleanly (0 failures, 0 NaN/Inf), on the 7,438-clip canonical
  set (updated after `1076_MTI_SAD_XX` was excluded — see above). Test-actor clips (978, unaffected by
  that exclusion) are deliberately excluded from extraction — they'll only be extracted inside a future
  `evaluate.py --final`.
- `StandardScaler` (fit on train only) + `LogisticRegression(class_weight="balanced", solver="lbfgs")`,
  trained on TRAIN actors, evaluated on VAL actors only:

  | | macro-F1 | accuracy |
  |---|---|---|
  | eGeMAPS + LR (untuned, `C=1.0`) | 0.4903 ± 0.0000 | 0.4926 ± 0.0000 |
  | chance (stratified dummy) | 0.1587 ± 0.0037 | 0.1599 ± 0.0043 |

  The LR std is exactly 0 across the 3 seeds because `lbfgs` is a deterministic solver with no
  stochastic component on a fixed design matrix — not a bug, and reported as such rather than masked.
- **Round 6 fix**: `C=1.0` had never been tuned, while WavLM's hyperparameters always were (early
  stopping on val macro-F1) — an unfair comparison. `scripts/22_wavlm_rescore_and_tuned_egemaps.py`
  grid-searches `C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}` on val macro-F1: best is **`C=0.3`,
  val macro-F1 = 0.4950 ± 0.0000** (+0.0047 over untuned). Both are kept in
  `outputs/egemaps_tuned_vs_untuned.csv`; **tuned (`C=0.3`) is what "eGeMAPS" means from this point on**,
  including in `FINAL_EVAL_PLAN.md`.
- Because that ±0.0000 only proves determinism, not generalization, speaker-level uncertainty was
  estimated two ways (neither touches test): a cluster bootstrap over the 15 VAL actors (1000
  resamples) gives macro-F1 95% CI **[0.4361, 0.5439]**; actor-level 5-fold `GroupKFold` CV over all 79
  train+val actors gives macro_f1 = **0.5156 ± 0.0271** across folds (pooled out-of-fold macro_f1 0.5169).
- Per-class F1 (LR): anger 0.69 (easiest), neutral 0.51, sad 0.49, happy 0.48, fear 0.39, disgust 0.37
  (hardest) — consistent with typical audio-only emotion-recognition results.
- All runs logged to MLflow (`sqlite:///mlflow.db`, gitignored; plain-file tracking is deprecated as of
  the installed mlflow 3.18 and raises on use), tagged with the split content-hash and git commit.

## Phase 2 pilot (Part A): 20-clip feasibility check

```bash
python scripts/07_phase2_pilot.py
```

- 20 clips sampled from TRAIN+VAL actors (stratified across the 6 emotions), `.flv` files independently
  inspected via the `ffprobe` CLI (cross-checks Phase 1's PyAV-based probing — same results: 29.97fps,
  480x360, `vp6f`).
- Face detection: MediaPipe `FaceLandmarker` (478 landmarks; the legacy `solutions.face_mesh` API is gone
  in the installed mediapipe 1.2.2, replaced by the Tasks API — required downloading
  `face_landmarker.task` from Google's model store into `models/mediapipe/`, gitignored) with a
  RetinaFace fallback (used 0 times in this pilot) — no Haar cascades.
- First pass: 0% detection failure rate, ~0.14s/clip. Occlusion overlays were initially semi-transparent
  and the mask was a loose rectangle — fixed before Phase 2 Part B (below): all 4 types now fully opaque,
  and the mask is a landmark polygon (nose bridge apex, cheek/jaw/chin oval boundary, angle-sorted into a
  simple polygon) rather than a bounding-box rectangle.

## Phase 2 Part B: full run, all 7,439 clips

```bash
python scripts/08_phase2_full_run.py          # ~30 min on this Mac
python scripts/09_build_final_contact_sheet.py
```

Run over **all** 7,439 canonical clips, including test-actor clips — unlike the Phase 3 audio-feature
extraction, this was an explicit instruction to cover the full dataset, and face/audio caching is
deterministic preprocessing with no fitted parameters (no test-set rule is implicated).

Per clip, cached to `cache/` (gitignored, 5.3GB total, regenerable):
- `face_crops/{clip_id}/frame_NNN.jpg` — clean 224x224 crops for every ~10fps-sampled frame (2.2GB)
- `landmarks/{clip_id}.npz` — 478-pt landmarks for every successfully-detected frame (983MB)
- `audio/{clip_id}.npy` — mono 16kHz float32 waveform, resampled/verified from AudioWAV (1.1GB)
- `occluded_crops/{clip_id}/{type}_{severity}.jpg` — all 4 occlusion types x 3 severities (0.3/0.6/0.9),
  applied to one representative frame per clip (1.0GB)

**Results:**
- **7,439/7,439 clips processed, 0 script failures**, 29.5 minutes total (~4.2 clips/s).
- Frame-level face-detection failure rate: **0.02%** (1 of ~4,934 sampled frames missed across the whole
  run) — essentially clean.
- **Exactly one clip failed detection on every single sampled frame: `1076_MTI_SAD_XX`** (0/26 frames).
  Investigated directly: its video is a "Media offline" placeholder slate in 7 languages, not footage —
  this is CREMA-D's own documented issue for this file ("error picture, no audio, very short" in
  `README.md`), which Phase 1's container-level probing missed because its duration/audio-stream
  metadata looked normal. **Decision (resolved):** added to the excluded-clips list — the canonical
  count is now **7,438** (see "Known upstream data quirks" above). This run itself still processed the
  pre-exclusion 7,439-clip set, as recorded below; all later phases use the 7,438-clip canonical set.
- **MediaPipe detection rate on occluded crops, by type x severity** (`outputs/phase2_occlusion_detection_rate_table.csv`):

  | occlusion_type | 0.3 | 0.6 | 0.9 |
  |---|---|---|---|
  | mask | 99.96% | 99.93% | 99.84% |
  | sunglasses | 100.00% | 99.99% | 99.97% |
  | hand_over_mouth | 100.00% | 99.99% | 99.99% |
  | patch | 99.96% | 95.12% | **44.88%** |

  Landmark-placed occlusions (mask/sunglasses/hand_over_mouth) stay near 100% detectable even at high
  severity, because their placement preserves enough face structure for MediaPipe's detector. The
  random, non-landmark-aware `patch` type is the outlier: at severity 0.9 it drops to 44.9% because it
  sometimes randomly lands over the eyes. This is a real, measured difference in how the 4 occlusion
  types stress the visual branch — relevant context for the Phase 5 confidence-weighting hypothesis.
- Final contact sheet (4 clips, 4 different actors, 4 different emotions, original + all 12 occlusion
  variants): `outputs/phase2_pilot/final_contact_sheet_part_b.png` (gitignored, regenerate via the
  script above).

## Phase 4 (first version): face model — ResNet-18 fine-tuning + occlusion robustness

```bash
python scripts/10_face_model_train.py          # 3 seeds, clean training, MPS
python scripts/11_face_model_occlusion_eval.py # detection cache (CPU, parallel) + model eval (MPS)
```

### Clean training design

- Frame-level fine-tuning of timm `resnet18` (ImageNet-pretrained) on TRAIN actors' cached face crops;
  clip-level prediction by averaging per-frame softmax probabilities, then argmax. Evaluated on VAL
  actors only — test untouched.
- Benchmarked on this Mac's MPS backend with disk-loaded JPEGs: ~110 img/s realistic throughput;
  `num_workers>0` was measured SLOWER (88.9/86.6 img/s at 4/8 workers vs. 113.1 img/s at 0) —
  multiprocessing spawn overhead dominates at this dataset scale, so `num_workers=0` throughout.
- To fit a ~30 min/seed budget: train on 4 subsampled frames/clip/epoch (deterministic per clip+seed,
  genuinely varies across epochs and seeds — an extra source of run-to-run variation beyond weight
  init); per-epoch early stopping is monitored on a cheap subsampled VAL set (same 4-frame scheme); only
  the **final best checkpoint** gets the expensive full-cached-frame VAL evaluation, which is what's
  reported. `max_epochs=6`, `patience=2`, class-weighted cross-entropy, Adam lr=1e-4. 3 seeds (0,1,2).
- Settings recorded in `configs/face_model.yaml`.

### Clean results (VAL, 3 seeds)

| | macro-F1 | accuracy |
|---|---|---|
| ResNet-18 (clean) | **0.5724 ± 0.0037** | 0.6010 ± 0.0023 |

- Bootstrap CI (seed 0's model, 1000 resamples over 15 val actors): **[0.4952, 0.6491]**.
- Per-class F1 (mean ± std over 3 seeds): happy 0.83±0.04 and disgust 0.74±0.01 (easiest), neutral
  0.58±0.02, fear 0.55±0.01, anger 0.54±0.02, **sad 0.18±0.02 — by far the hardest, confused mostly
  with fear and anger** per the confusion matrix.
- Confusion matrix (seed 0) saved to `outputs/confusion_matrix_face_resnet18_seed0.csv`; all 3 seeds +
  the summary logged to MLflow (`sqlite:///mlflow.db`, experiment `phase4_face_resnet18_clean`).

### Occlusion-robustness evaluation: design

A first attempt re-ran MediaPipe detection inside the model-eval loop, on every cached frame
(~20-30/clip) x 12 conditions (4 types x 3 severities), single process — projected to ~12 hours over
the 1,224 val clips and was killed rather than left to run. Redesigned into two phases (`scripts/11`,
superseded by `scripts/13` below; settings recorded in `configs/face_model.yaml`):

1. **Detection cache (CPU, multiprocess, run once):** 8 *evenly spaced* frames per clip — the same 8
   frames for every occlusion type, severity, and model, so results are comparable apples-to-apples
   rather than each condition drawing its own sample. For each (clip, frame, type, severity): apply the
   occlusion, run MediaPipe detection, cache the True/False result
   (`cache/phase4_occlusion_detection_cache.parquet`) — reusable by future face models, since detection
   depends only on the occluded image, not model weights. Parallelized across CPU cores, reserving 2 so
   the Mac stays usable.
2. **Model evaluation (MPS, single process, batched, `torch.no_grad()`, eval mode):** reads the cached
   True/False table — detection is never re-run here. Regenerates (cheap, deterministic) the occluded
   crops only for frames the cache marked detected, batches them per clip/condition, and averages
   softmax probabilities.

**Face-branch missing-detection rule (official, v2 — see fix below):** a frame with no detection casts
no vote. If a clip has zero surviving frames under a condition, the face branch abstains: this is a
false negative for its true class, is **not** a prediction of any class (no false positive), and still
counts in the accuracy denominator (`fer.metrics.score_with_abstention`). An earlier version scored the
abstained clip as a deterministic prediction of class 0 ("anger") — which could spuriously register as
either a false positive for anger or, worse, a correct prediction whenever the true label happened to
be anger — and a stricter alternative dropped abstained clips from the metric entirely, shrinking the
denominator. Both were replaced by the rule above, which is now the only scoring used anywhere in this
project.

A 20-clip dry run measured ~2.6 clips/s for the detection phase and ~0.23s/clip for the eval phase,
projecting a ~15-20 min full run; the actual full run took **10.3 minutes** (3.5 min detection-cache
build + 6.8 min model eval) — comfortably inside the pre-agreed ~1 hour budget.

### Mask severity: redefined by vertical extent (fix, v1 — superseded below)

The original `mask` occlusion scaled continuously with severity (a small oval-widening factor,
`1 + 0.08*severity`), which barely changed its coverage (21-23% across all three severities) and gave a
flat, uninformative severity curve. Redefined (`fer.face_pipeline.apply_mask`, v1) so severity selects a
*landmark-anchored vertical extent*, with the bottom/side boundary always the cheek-jaw-chin oval:

| severity | top edge | mean coverage (224x224 crop) |
|---|---|---|
| 0.3 | mouth (upper lip) | 9.8% |
| 0.6 | nose bridge midpoint | 20.5% |
| 0.9 | just below the eyes | 20.6% |

Coverage jumped sharply from 0.3→0.6 but barely moved 0.6→0.9 — a review caught this: the two discrete
landmark anchors used for 0.6 and 0.9 (nose bridge, lowest eye-contour point) happen to sit at almost the
same height on MediaPipe's topology, an anatomical ambiguity not caught before shipping v1.

### Mask severity: continuous monotonic interpolation (fix, v2 — current)

Replaced the two discrete landmark anchors with a **continuous linear interpolation** between two
well-separated, unambiguous anchors — mean mouth-top height (severity 0.3) and mean eyebrow height
(severity 0.9) — so coverage increases monotonically by construction rather than depending on two single
landmarks happening to be far enough apart:

| severity | top edge | mean coverage (224x224 crop) |
|---|---|---|
| 0.3 | mouth (upper lip) | **11.75%** |
| 0.6 | 50% of the way to eyebrow height | **17.49%** |
| 0.9 | eyebrow height | **23.43%** |

Measured geometrically (occlusion drawn onto a blank canvas, fraction of changed pixels, 300-sample),
independent of any model or detection run — clearly and smoothly monotonic now.

### Reproducibility fix

A question about whether `patch`'s severity-0.6/0.9 numbers were reproducible between runs turned out to
have two parts. First, the specific numbers compared (0.250/0.131 vs. 0.289/0.155) were **not** the same
quantity: 0.250/0.131 was seed 0 alone (from the original `scripts/11` run), while 0.289/0.155 was the
**3-seed mean** reported afterward — seed 0 alone reproduced exactly (0.2496 vs. 0.249565, 0.1308 vs.
0.130795) between the two separate process runs that happened to use it. So there was no actual drift in
that comparison. Second, or so: the code had a real latent risk independent of that specific comparison
— frame-level softmax probabilities were averaged with `.mean(dim=0)` *on the MPS tensor* before moving
to CPU, and Apple's MPS backend does not guarantee deterministic reduction order across runs. Fixed by
moving the averaging to CPU (`fer.occlusion_eval.evaluate_occlusion_probs`: `.cpu().numpy().mean(axis=0)`),
so only a sequential, deterministic sum combines frame probabilities — the model's own forward pass is
the only remaining MPS step, and it operates on one fixed input per call rather than reducing across a
GPU-resident batch dimension. `tests/test_occlusion_reproducibility.py` now runs the same checkpoint
against the same clips/conditions twice and asserts bit-identical output, specifically on `patch`
severity 0.9 and `mask` severity 0.6 — the two lowest-confidence, most flip-prone conditions — so a
future regression here fails a test instead of silently drifting.

### Official occlusion results (VAL, 3 seeds, `fer.metrics.score_with_abstention`)

| occlusion_type | severity | macro-F1 | accuracy | n_abstained / n_clips |
|---|---|---|---|---|
| clean | — | 0.5724 ± 0.0046 | 0.6010 ± 0.0029 | 0/1224 |
| mask | 0.3 | 0.3154 ± 0.0248 | 0.3772 ± 0.0202 | 0/1224 |
| mask | 0.6 | 0.1845 ± 0.0444 | 0.2653 ± 0.0388 | 0/1224 |
| mask | 0.9 | 0.1157 ± 0.0167 | 0.2100 ± 0.0088 | 0/1224 |
| sunglasses | 0.3 | 0.3231 ± 0.0573 | 0.3998 ± 0.0395 | 0/1224 |
| sunglasses | 0.6 | 0.3116 ± 0.0285 | 0.3875 ± 0.0218 | 0/1224 |
| sunglasses | 0.9 | 0.2777 ± 0.0282 | 0.3603 ± 0.0091 | 0/1224 |
| patch | 0.3 | 0.4036 ± 0.0074 | 0.4401 ± 0.0086 | 0/1224 |
| patch | 0.6 | 0.2871 ± 0.0367 | 0.3396 ± 0.0321 | 26/1224 |
| patch | 0.9 | 0.1470 ± 0.0223 | 0.1503 ± 0.0181 | **493/1224 (40.3%)** |
| hand_over_mouth | 0.3 | 0.3281 ± 0.0735 | 0.3897 ± 0.0546 | 0/1224 |
| hand_over_mouth | 0.6 | 0.3044 ± 0.0703 | 0.3676 ± 0.0565 | 0/1224 |
| hand_over_mouth | 0.9 | 0.2810 ± 0.0469 | 0.3448 ± 0.0412 | 0/1224 |

Full table: `outputs/phase4_occlusion_eval_table_official.csv` (per-seed: `..._by_seed_official.csv`). Uses
the v2 mask geometry above. Superseded: `outputs/phase4_occlusion_eval_table.csv` (old scoring, v1 mask
geometry) — kept for history, not for comparison.

- Mask now shows a real, monotonically-worsening severity curve (0.315 → 0.184 → 0.116) instead of the
  v1 geometry's near-flat 0.313 → 0.175 → 0.161 — the v2 fix changed the geometry so severity 0.9 actually
  reaches further down the face than 0.6, and the model's performance now tracks that clearly.
- `patch` at severity 0.9 remains the worst non-mask condition under the stricter official scoring
  (macro-F1 0.147, accuracy 0.150 — *below* the stratified chance accuracy of ~0.16) because 493/1224
  clips (40.3%) get zero votes and are scored as pure misses, with no chance of lucking into a correct
  tie-broken guess the way the old scoring allowed. Mask 0.9 (0.116) is now the single worst condition
  overall, below even `patch` 0.9.
- Every occlusion condition remains well below the clean 0.5724, and mask (all severities) and patch@0.9
  also fall below the clean audio baseline (0.49) — see Phase 5 below for whether fusion recovers this.

### Occlusion-augmented face model (ablation): training-time occlusion closes much of this gap

Two additional ResNet-18 variants were fine-tuned with random occlusions applied during training
(`fer.face_model_augmented`, `scripts/15_face_model_train_augmented.py`, 3 seeds each): each selected
training frame is independently left clean (50%) or occluded with a random type + random severity (50%),
using a training-time RNG namespace separate from the evaluation occlusion seed. Monitoring/early-stopping
and the final reported metric stay on **clean** val throughout, matching the original clean-trained
model's methodology, so the comparison below isolates the effect of training-time occlusion exposure:

- `aug_all` — all 4 occlusion types available during training.
- `aug_no_mask` — mask held out of training entirely, so it's an **unseen** occlusion type at evaluation.

A pre-training time estimate (one timed epoch: 158.3s, one timed monitor-eval call: 14.2s, combined with
historical ~5 epochs/seed) projected ~100-115 minutes for all 6 seed-runs, under the ~2 hour ceiling
agreed in advance — training was launched on that basis. **The actual wall time was ~3h20min**, roughly
double the estimate; this is being reported plainly rather than glossed over. The discrepancy most likely
reflects system-level memory/CPU contention during the run (other applications competing for this Mac's
limited RAM) rather than a flaw in the per-epoch timing itself, but the estimate should have included a
larger safety margin given the single-epoch measurement's susceptibility to exactly that kind of noise.

Clean val macro-F1 (mean ± std, 3 seeds): clean-trained **0.5724 ± 0.0046**, `aug_all` **0.5715 ± 0.0196**,
`aug_no_mask` **0.5822 ± 0.0270** — training-time occlusion costs essentially nothing on clean data.

Occlusion macro-F1 by variant (mean ± std, 3 seeds; full table `outputs/phase4_occlusion_table_by_variant.csv`,
per-seed `outputs/phase4_occlusion_by_variant_by_seed.csv`):

| condition | clean-trained | aug_all | aug_no_mask |
|---|---|---|---|
| mask 0.3 | 0.3154 ± 0.0248 | 0.4791 ± 0.0430 | 0.4048 ± 0.0431 |
| mask 0.6 | 0.1845 ± 0.0444 | 0.4527 ± 0.0420 | 0.3606 ± 0.0558 |
| mask 0.9 | 0.1157 ± 0.0167 | 0.3253 ± 0.0388 | 0.1836 ± 0.0246 |
| patch 0.3 | 0.4036 ± 0.0074 | 0.5205 ± 0.0155 | 0.5188 ± 0.0257 |
| patch 0.6 | 0.2871 ± 0.0367 | 0.4583 ± 0.0160 | 0.4470 ± 0.0336 |
| patch 0.9 | 0.1470 ± 0.0223 | 0.2737 ± 0.0263 | 0.2733 ± 0.0112 |
| sunglasses 0.3 | 0.3231 ± 0.0573 | 0.5168 ± 0.0460 | 0.5293 ± 0.0331 |
| sunglasses 0.6 | 0.3116 ± 0.0285 | 0.5088 ± 0.0474 | 0.5267 ± 0.0359 |
| sunglasses 0.9 | 0.2777 ± 0.0282 | 0.5077 ± 0.0431 | 0.5208 ± 0.0400 |
| hand_over_mouth 0.3 | 0.3281 ± 0.0735 | 0.4730 ± 0.0163 | 0.4670 ± 0.0373 |
| hand_over_mouth 0.6 | 0.3044 ± 0.0703 | 0.4570 ± 0.0313 | 0.4559 ± 0.0284 |
| hand_over_mouth 0.9 | 0.2810 ± 0.0469 | 0.4389 ± 0.0494 | 0.4362 ± 0.0359 |

- **Every occlusion condition improves substantially with training-time augmentation**, for both
  variants, at no measurable clean-accuracy cost.
- **`aug_no_mask` — which never saw mask during training — still roughly doubles mask performance** at
  every severity (0.9: 0.116 → 0.184), showing partial transfer from the other three occlusion types to
  an unseen one.
- **`aug_all`, which did see mask, does even better on mask** (0.9: 0.116 → 0.325, nearly 3x) without
  giving up anything on the other three types relative to `aug_no_mask` (they're within noise of each
  other on patch/sunglasses/hand_over_mouth).

## Phase 5, round 1 (development version): first confidence-weighted fusion result

```bash
python scripts/14_phase5_fusion_dev.py
```

VAL only (test untouched). Audio branch: the exact Phase 3 eGeMAPS + StandardScaler + LogisticRegression
pipeline, refit once (deterministic, `std~=0`) and scored via `predict_proba` on val — audio has no
occlusion variants, since occluding the face doesn't touch the waveform, so its per-clip probabilities
are the same across every "condition" below. Face branch: `outputs/phase4_face_probs.parquet` (written
by `scripts/13` above), all 3 seeds, clean + every occlusion condition.

### Calibration

Each branch is temperature-scaled (`fer.calibration`, operating on probability vectors via
`log(p)` as an exact pseudo-logit — softmax's shift-invariance makes this lossless even without the raw
logits). Temperature is fit on **clean data only**, with 2-fold cross-fitting over the 15 val actors
(8/7 split): fold A's temperature is fit on fold B and applied to fold A, and vice versa, so no clip is
ever scored with a temperature fit on itself or its own fold. The same fold-assigned, clean-fit
temperature is then applied to that branch's occluded probabilities too, to test whether a clean-fit
calibration holds up under occlusion.

| | ECE before | ECE after |
|---|---|---|
| audio (clean; condition-invariant) | 0.0735 | **0.0321** |
| face, clean | 0.0659 ± 0.0281 | 0.0603 ± 0.0322 |
| face, mask (0.3 / 0.6 / 0.9) | 0.149 / 0.161 / 0.146 | 0.171 / 0.167 / 0.176 |
| face, sunglasses (0.3 / 0.6 / 0.9) | 0.088 / 0.078 / 0.089 | 0.075 / 0.061 / 0.071 |
| face, patch (0.3 / 0.6 / 0.9) | 0.059 / 0.059 / 0.094 | 0.061 / 0.077 / 0.100 |
| face, hand_over_mouth (0.3 / 0.6 / 0.9) | 0.148 / 0.140 / 0.129 | 0.124 / 0.116 / 0.132 |

Full table: `outputs/phase5_ece_table.csv` (per-seed: `phase5_ece_by_seed.csv`).

- Audio's clean-fit calibration helps audio itself a lot (ECE roughly halved) — expected, since audio
  only ever sees the condition it was calibrated on.
- For the face branch, clean-fit calibration does **not** reliably generalize: it helps
  `sunglasses`/`hand_over_mouth` modestly, is roughly a wash on `patch`, and makes `mask` *worse*
  (0.149→0.171, 0.161→0.146→0.176) — a temperature tuned to clean confidence's distribution doesn't match
  mask's, where the failure mode is silent classification collapse rather than the kind of miscalibration
  clean-data fitting corrects for. Reported plainly rather than only showing the conditions it helped.

### Fusion comparison (VAL, 3 seeds, macro-F1)

| condition | audio-only | face-only | fixed 50/50 | confidence-weighted | mean face weight |
|---|---|---|---|---|---|
| clean | 0.4903 | 0.5724 ± 0.0046 | **0.7003 ± 0.0178** | 0.6868 ± 0.0180 | 0.545 ± 0.007 |
| mask 0.3 | 0.4903 | 0.3127 ± 0.0262 | **0.5093 ± 0.0316** | 0.5020 ± 0.0320 | 0.524 ± 0.023 |
| mask 0.6 | 0.4903 | 0.1752 ± 0.0634 | 0.4703 ± 0.0681 | 0.4597 ± 0.0803 | 0.460 ± 0.042 |
| mask 0.9 | 0.4903 | 0.1614 ± 0.0539 | 0.4594 ± 0.0719 | 0.4524 ± 0.0760 | 0.461 ± 0.048 |
| sunglasses 0.3 | 0.4903 | 0.3231 ± 0.0573 | **0.5691 ± 0.0354** | 0.5562 ± 0.0336 | 0.478 ± 0.022 |
| sunglasses 0.6 | 0.4903 | 0.3116 ± 0.0285 | **0.5724 ± 0.0170** | 0.5597 ± 0.0211 | 0.464 ± 0.019 |
| sunglasses 0.9 | 0.4903 | 0.2777 ± 0.0282 | **0.5673 ± 0.0084** | 0.5540 ± 0.0120 | 0.454 ± 0.018 |
| patch 0.3 | 0.4903 | 0.4036 ± 0.0074 | **0.5938 ± 0.0118** | 0.5838 ± 0.0107 | 0.497 ± 0.017 |
| patch 0.6 | 0.4903 | 0.2871 ± 0.0367 | **0.5442 ± 0.0299** | 0.5362 ± 0.0318 | 0.449 ± 0.026 |
| patch 0.9 | 0.4903 | 0.1470 ± 0.0223 | 0.4909 ± 0.0130 | 0.4879 ± 0.0135 | 0.335 ± 0.006 |
| hand_over_mouth 0.3 | 0.4903 | 0.3281 ± 0.0735 | **0.5410 ± 0.0740** | 0.5207 ± 0.0676 | 0.508 ± 0.023 |
| hand_over_mouth 0.6 | 0.4903 | 0.3044 ± 0.0703 | **0.5421 ± 0.0675** | 0.5309 ± 0.0709 | 0.491 ± 0.023 |
| hand_over_mouth 0.9 | 0.4903 | 0.2810 ± 0.0469 | **0.5234 ± 0.0496** | 0.5101 ± 0.0535 | 0.482 ± 0.023 |

Full table: `outputs/phase5_fusion_table.csv` (per-seed: `phase5_fusion_by_seed.csv`).

- **Both fusion variants beat face-only at every single condition**, often by a wide margin (e.g. mask
  0.9: face 0.16 vs. fusion ~0.46) — fusing in audio recovers most of what occlusion destroys.
- **Fixed 50/50 fusion beats audio-only everywhere except `mask` at severity 0.6/0.9**, where it's
  slightly *worse* than audio alone (0.470/0.459 vs. 0.4903) — a fixed weight can't protect against a
  branch that's actively wrong rather than just absent.
- **Confidence weighting does not fix the mask problem, and is a mixed bag overall**: it beats fixed
  fusion only on `patch` (where the mean face weight correctly drops to 0.335-0.449 — abstained clips
  produce an exactly-uniform, obviously low-confidence distribution that the weighting formula handles
  well) but is slightly *worse* than fixed fusion on every other condition including clean and mask,
  where the mean face weight barely moves (0.46-0.52, not far from the unweighted 0.5). This matches the
  check-4/check-3 diagnostics from before: mask's failure mode is a confident collapse onto "anger," not
  low confidence, so a weighting scheme driven by raw max-probability has no signal to act on there.
- **Headline, honestly mixed**: confidence-weighted fusion as implemented here is not yet a reliable
  improvement over simple fixed-weight fusion — it helps exactly where the face branch's failure is
  legible as low confidence (detection dropout under `patch`) and does nothing where the failure is a
  confident wrong answer (`mask`). A fusion signal that also incorporates something like per-class
  confusion history, or an explicit detection-failure flag (not just max-softmax), is the natural next
  step and is not yet built.

## Phase 5, round 2 (development version): expanded fusion table, 3 face-model variants

```bash
python scripts/18_phase5_fusion_v2.py
```

VAL only. Same audio branch and the same clean-only, 2-fold actor cross-fit temperature calibration as
round 1 (`fer.fusion`), applied separately to each of the 3 face-model variants (clean-trained, `aug_all`,
`aug_no_mask`) × 3 seeds. Four fusion methods are compared — **fixed in advance, no others added**:

- **fixed 50/50** — equal weight, no adaptation.
- **confidence-weighted (max-prob)** — per-clip face weight from each branch's own calibrated max-softmax.
- **confidence-weighted (negative entropy)** — per-clip face weight from `log(K) - H(p)`, shifted
  non-negative (0 at uniform, `log(K)` at one-hot).
- **learned gate** — logistic regression on `[face_max_prob_conf, audio_max_prob_conf]`, predicting
  P(face branch correct), fit via the same 2-fold actor cross-fit but **pooled across clean + every
  occlusion condition** (not clean-only — a clean-only gate would never see occluded confidence patterns).

Full table (39 condition×variant rows × 6 methods, mean ± std over 3 seeds, including mean face weight):
`outputs/phase5_fusion_v2_table.csv`. Selected rows (macro-F1, clean-trained model):

| condition | audio-only | face-only | fixed 50/50 | conf max-prob | conf neg-entropy | learned gate |
|---|---|---|---|---|---|---|
| clean | 0.490 | 0.572 ± 0.005 | **0.700 ± 0.018** | 0.687 ± 0.018 | 0.684 ± 0.019 | 0.676 ± 0.014 |
| mask 0.6 | 0.490 | 0.184 ± 0.044 | **0.413 ± 0.036** | 0.402 ± 0.028 | 0.409 ± 0.014 | 0.419 ± 0.032 |
| mask 0.9 | 0.490 | 0.116 ± 0.017 | 0.490 ± 0.018 | 0.488 ± 0.016 | 0.490 ± 0.012 | **0.499 ± 0.002** |
| patch 0.9 | 0.490 | 0.147 ± 0.022 | **0.491 ± 0.013** | 0.488 ± 0.014 | 0.491 ± 0.011 | 0.492 ± 0.002 |
| sunglasses 0.6 | 0.490 | 0.312 ± 0.028 | **0.572 ± 0.017** | 0.560 ± 0.021 | 0.558 ± 0.023 | 0.561 ± 0.015 |

And for the two augmented variants, e.g. `aug_all`'s mask 0.9 (its hardest condition): audio-only 0.490,
face-only 0.325 ± 0.039, **fixed 50/50 0.563 ± 0.018**, conf max-prob 0.559 ± 0.015, conf neg-entropy
0.544 ± 0.018, learned gate 0.543 ± 0.015 — fixed fusion still wins outright once the face branch is this
much more robust.

**Fixed 50/50 fusion beats-or-matches all three adaptive methods (confidence max-prob, confidence
neg-entropy, learned gate) in 35 of 39 (condition × face-model-variant) comparisons** (corrected from an
earlier, imprecise "38 of 39" — see `RESULTS.md` round 3 for the correction). The 4 exceptions all involve
a severely degraded face branch: clean-trained/mask 0.6, clean-trained/mask 0.9, clean-trained/patch 0.9,
and aug_no_mask/mask 0.9 — in each, one adaptive method (usually the learned gate) edges fixed out by
roughly 1 point, within the run-to-run std (~0.014-0.018), e.g. at clean-trained `mask 0.9` the learned
gate (0.499) vs. fixed (0.490), mainly reflecting the gate correctly pushing face weight down (mean face
weight 0.256) toward audio-only (0.490), not genuine synergy.

**Headline: making the face branch far more occlusion-robust (the augmented-training ablation above) did
not change the round-1 conclusion about weighting.** A per-clip weight built from a branch's own
confidence (max-softmax or negative entropy), or a logistic gate trained on those confidences, still does
not reliably beat simply averaging the two branches — regardless of which face model sits behind it.

**Note on patch 0.9 — fusion ≈ audio by construction.** At patch severity 0.9, 493/1224 (40.3%) of clips
get zero face detections and are scored as an exactly-uniform face distribution. Any confidence-weighted
method correctly identifies these specific clips as unconfident and pushes face weight toward 0 for them
— mean face weight at patch 0.9 drops to roughly 0.11-0.35 across variants/methods, about half the 0.50
baseline. This is **not** evidence that confidence-weighting "solves" patch occlusion; it's the mechanism
correctly detecting detection failure on the ~40% of clips where it occurs, while doing nothing special
for the other ~60% that still got a (sometimes wrong) prediction. The resulting fusion score is close to,
but still not above, fixed 50/50 — blind averaging isn't beaten by conditionally zeroing out face only
on the clips that failed detection.

## Phase 5, round 3 (development version): fusion table for the augmented face models

Same script and data as round 2 (`outputs/phase5_fusion_v2_table.csv`); this section surfaces the
`aug_all` / `aug_no_mask` rows specifically.

| condition | audio-only | face-only (aug_all) | fixed 50/50 | conf max-prob | conf neg-entropy | learned gate |
|---|---|---|---|---|---|---|
| clean | 0.490 | 0.572 ± 0.020 | **0.702 ± 0.023** | 0.691 ± 0.025 | 0.684 ± 0.025 | 0.681 ± 0.035 |
| mask 0.3 | 0.490 | 0.479 ± 0.043 | **0.658 ± 0.029** | 0.643 ± 0.031 | 0.638 ± 0.032 | 0.621 ± 0.043 |
| mask 0.6 | 0.490 | 0.453 ± 0.042 | **0.641 ± 0.013** | 0.632 ± 0.013 | 0.626 ± 0.011 | 0.609 ± 0.022 |
| mask 0.9 | 0.490 | 0.325 ± 0.039 | **0.563 ± 0.018** | 0.559 ± 0.016 | 0.544 ± 0.018 | 0.543 ± 0.015 |
| patch 0.9 | 0.490 | 0.274 ± 0.026 | **0.537 ± 0.013** | 0.530 ± 0.013 | 0.522 ± 0.008 | 0.523 ± 0.010 |
| sunglasses 0.9 | 0.490 | 0.508 ± 0.043 | **0.657 ± 0.019** | 0.642 ± 0.022 | 0.642 ± 0.019 | 0.645 ± 0.026 |
| hand_over_mouth 0.9 | 0.490 | 0.439 ± 0.049 | **0.639 ± 0.011** | 0.632 ± 0.014 | 0.626 ± 0.018 | 0.603 ± 0.011 |

All mean face weights under `fixed_5050` are, by definition, 0.500 (full table has every method's weight;
see `outputs/phase5_fusion_v2_table.csv` for `aug_no_mask` and all 13 conditions × both variants).

**Average margin, fixed 50/50 minus each adaptive method, over all 13 conditions:**

| | aug_all | aug_no_mask |
|---|---|---|
| vs. conf max-prob | +0.0106 (wins 13/13) | +0.0092 (wins 13/13) |
| vs. conf neg-entropy | +0.0180 (wins 13/13) | +0.0144 (wins 12/13) |
| vs. learned gate | +0.0265 (wins 13/13) | +0.0247 (wins 13/13) |

**Does fixed 50/50 fusion with an augmented face beat audio-only in every occlusion condition? Yes — for
both augmented variants, not for the clean-trained one.** `aug_all` and `aug_no_mask` each beat audio-only
in **12/12** occlusion conditions (margin +0.045 to +0.179). The clean-trained model's fixed fusion
managed only **10/12**, failing on `mask 0.6` (0.413 vs. audio's 0.490) and `mask 0.9` (0.490 vs. 0.490, a
dead heat) — exactly the two conditions where its face branch collapses hardest. This is a new finding
from the augmented-training ablation: training-time occlusion doesn't just improve the face branch in
isolation, it also removes fusion's worst failure mode (fusion dragging below audio-only), because the
face branch no longer collapses into a confidently-wrong answer under mask.

## Demographic fairness check: 5-fold actor CV (val+train actors, clean clips only)

```bash
python scripts/19_demographic_cv.py
```

Full writeup and tables: `RESULTS.md`, Phase 5 round 4. Summary: actor-level 5-fold CV over the 79
non-test actors, scope-reduced to **1 seed per fold** (10 face-model training runs instead of the full
spec's 30) after a pre-run estimate put the full 3-seeds/fold version at 10-17 hours on this Mac — agreed
with the user before launching; actual wall time was ~4.2 hours. Reports macro-F1 by sex, age band, and
race (groups with fewer than 3 actors excluded — only `Race=Unknown`, n=1, triggers this).

**Round 5 adds actor-level bootstrap 95% CIs** (`scripts/20_demographic_bootstrap_ci.py`,
`RESULTS.md` round 5) on the sex and age differences above — a difference is only called a "gap" if its
CI excludes zero, and even then "confirmed" means only that the interval excludes zero at this sample
size, not that it is a real-world effect with a known cause. Result: the sex difference does **not**
reach significance for any method (round 4's "consistent sex gap" is retracted); the age difference
(≤30 vs. 46-60) is an **observed** difference whose CI excludes zero for the two `aug_all`-involving
methods (face and fusion), and does not reach that threshold for the clean-trained ones; the audio
branch is roughly uniform across age groups; and the age gap's *widening* from clean to `aug_all`
training is similarly an observed difference with a CI excluding zero for fusion specifically
([+0.006, +0.075]), not reaching that threshold for the face branch alone. **This entire analysis is
validation/train-only, 1 seed per fold, development-stage, and was never re-tested on the frozen test
set** — no cause for either difference was investigated or is claimed.

## Final evaluation — complete

See `FINAL_EVAL_PLAN.md`, pre-specified before any test-set access and **amended once** (commit
`09e454e`, 2026-11-13, before the test run) to: freeze calibration/the learned gate on VAL rather than
refitting on test, promote three new primary comparisons P1 (fusion vs. WavLM alone)/P2 (`aug_no_mask`
generalization, unchanged)/P3 (confidence-weighting vs. fixed averaging — the project's original round-1
question, finally run as a formal test), and write the abstention rule into the plan explicitly.

`python evaluate.py --split test --final` ran exactly once (2026-11-14), all 3 seeds, all 3 face
variants, both audio branches, all 13 synthetic-occlusion conditions, no changes made after seeing the
results. **Headline: the face adds significantly to a strong voice model (P1, every condition),
occlusion-augmented training generalizes to the unseen `mask` type (P2, every severity), and
confidence-weighted fusion shows no improvement over fixed averaging pooled (P3) — not uniform across
the 13 individual conditions, reported exactly as it came out.** Exact numbers for P1/P2/P3 and every
branch/fusion score: see the "Final test results" table near the top of this README, or `RESULTS.md`,
"Final test results," for every condition including the non-significant P3 ones.

## Project structure

```
evaluate.py       # the pre-specified final evaluation script (FINAL_EVAL_PLAN.md); --split val|test, --final gates test
src/fer/          # package: filenames, probe, demographics, metadata, splits, envinfo,
                  # audio_features, uncertainty, face_pipeline, wavlm_model
configs/          # YAML configs (data.yaml, audio_baseline.yaml, ...)
scripts/          # one script per pipeline stage
splits/           # versioned, hashed actor split files (tracked in git)
tests/            # pytest suite, incl. the leakage check
outputs/          # generated artifacts (gitignored): metadata.csv, reports, contact sheets, etc.
cache/            # cached features (gitignored, regenerable)
models/           # downloaded model assets, e.g. mediapipe/face_landmarker.task (gitignored)
data/             # CREMA-D dataset (gitignored — never committed)
```

## WavLM audio fine-tuning (ran on Kaggle GPU — complete)

`notebooks/wavlm_finetune_kaggle.ipynb` fine-tunes `microsoft/wavlm-base-plus` (frozen CNN feature
encoder, small classification head), TRAIN actors only, early-stopped on VAL actors' macro-F1, 3 seeds.
Runs on Kaggle because this Mac has no suitable GPU for it; never touches test actors (asserted in the
notebook, and re-checked locally against the official split after download).

**Data integrity, verified before any training**: the notebook checks the public `ejlok1/cremad` audio
mirror against a private dataset (`kaggle_upload/`, generated by `scripts/21_prepare_kaggle_upload.py`)
containing `split_v1.json`, the documented 4-clip exclusion list, and MD5 checksums of this project's
local canonical 7,438-clip `AudioWAV` set — count, filenames, and checksums must match exactly, or the
notebook stops before training.

**Known mirror gap**: `ejlok1/cremad` is missing exactly one canonical clip, `1040_ITH_SAD_XX.wav`. It's
supplied from this project's local copy via `kaggle_upload/supplementary_audio/` (also generated by
`scripts/21`) and checksum-verified the same way as every file from the main mirror — the notebook
searches the private dataset's `supplementary_audio/` folder recursively for any clip missing from the
public mirror before deciding anything is actually missing, and reports which clips (if any) came from
the supplement.

**Run complete.** Outputs (3 checkpoints, per-clip val probabilities, the Kaggle run log) were
downloaded via the `kaggle` CLI (`kaggle kernels output <user>/<kernel> -p outputs/kaggle_wavlm_run`,
since browser downloads were unreliable) and moved into `models/wavlm/` (gitignored, local only) and
`outputs/`. The executed notebook itself wasn't among the kernel's outputs; the downloaded run log
(`notebooks/kaggle_run_log/wavlm_finetune_kaggle_run.log`) records the full run instead, including the
verification step passing and every epoch's `train_loss`/`val_macro_f1`.

`scripts/22_wavlm_rescore_and_tuned_egemaps.py` re-scores WavLM locally with the exact same metric code
used for eGeMAPS (`sklearn.metrics.f1_score`/`accuracy_score`/`confusion_matrix`), and confirms the val
clip set is identical to eGeMAPS's (1,224 clips, 15 val actors, every seed, no test-actor leakage).

**Round 7 correction — WavLM's evaluation is not batch-composition-invariant.** The frozen CNN feature
encoder has no attention-masking mechanism (unlike the transformer layers above it), so when clips of
different lengths share a zero-padded batch, `GroupNorm` statistics inside the encoder shift for the
whole sequence — found to move some clips' predicted class outright depending on arbitrary batch
composition (see `RESULTS.md` round 7 for the full account, including a reproduction showing the same
clip/weights give different top-1 classes in two different, equally valid batch groupings). The original
Kaggle-reported numbers (0.7027 ± 0.0076) depended on one arbitrary historical batch grouping and are
**superseded**. Fix: `batch_size=1` (no padding at all) — `scripts/25_wavlm_rescore_batch1.py` re-scores
all 3 seeds this way, run twice each to confirm bit-identical determinism first:

| seed | macro-F1 | accuracy |
|---|---|---|
| 0 | 0.6809 | 0.6879 |
| 1 | 0.7128 | 0.7124 |
| 2 | 0.6850 | 0.6912 |
| **mean ± std** | **0.6929 ± 0.0142** | **0.6972 ± 0.0109** |

(Checked, not retrained: the model's masked-mean pooling itself correctly excluded padded frames in
both training and evaluation — the same `forward` code path runs in both. The unmasked part is only the
upstream CNN encoder, a known property of wav2vec2/WavLM architectures. One consequence: WavLM's 3
Kaggle training runs were themselves exposed to this same batch-padding noise epoch to epoch, a mild
additional source of training stochasticity — noted as a limitation, not re-run.)

That's still roughly **+0.20 macro-F1 over tuned eGeMAPS (0.4950)** on the identical val clips — WavLM
remains a substantially stronger audio branch, just not quite as strong as first measured.
`scripts/23_phase5_fusion_wavlm.py` re-runs the full pre-specified fusion comparison (all 6 methods ×
all 3 face variants × all 13 conditions) with the corrected WavLM as the audio branch (each WavLM seed
paired with the matching face-model seed); fixed 50/50 fusion still *loses* to audio-only for the
clean-trained face under `mask` (at severities 0.3 and 0.6, now 11/13 conditions won rather than
round 6's 10/13) — full results are in `RESULTS.md` round 7, which supersedes round 6's WavLM numbers.

## Future work

Final test-set evaluation is **done** (see "Final test results" at the top). The `mask`-style confident
collapse that originally motivated the confidence-weighting hypothesis was a validation-stage
observation on the **clean-trained** face branch (rounds 1-3); it does not describe what happened on
the frozen test set — there, fixed fusion with the occlusion-trained face (`aug_all`) beat WavLM alone
in every one of the 13 conditions (P1), and confidence-weighting itself was formally tested and not
supported (P3). Remaining directions:

- **Real, non-synthetic occlusions** — captured from genuinely occluded recordings (an actual mask,
  real sunglasses, a real hand) rather than rendered digitally on detected landmarks, to test whether
  these results transfer to real material, fit, lighting, and motion.
- **Cross-corpus testing on [RAVDESS](https://zenodo.org/record/1188976)** (or another audio-visual
  emotion corpus) — to check whether the P1-P3 findings here, especially the P3 null result for
  confidence-weighting, generalize beyond CREMA-D.
- **A public demo** (e.g. Gradio), showing the fusion pipeline live on webcam + microphone input.

## About

**Yashashvi Jangid**, Independent Researcher.

Full technical report: [`reports/technical_report/report.pdf`](reports/technical_report/report.pdf)
(also converted to LaTeX/PDF in preprint form at [`reports/techrxiv/`](reports/techrxiv/), not yet
submitted anywhere).

AI coding assistants (Claude Code) were used for implementation throughout, under the author's
direction and review at every step; see the report's "Statement on the Use of AI Coding Assistants"
for detail.
