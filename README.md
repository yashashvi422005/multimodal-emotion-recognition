# fer-project: Multimodal (face + voice) emotion recognition under occlusion

Research question: when the face is partially occluded (masks, sunglasses,
hands), does a late-fusion model with confidence weighting shift reliance
onto the voice and recover performance that a face-only model loses?
Secondary question: does performance differ across demographic groups
(age, sex, race, ethnicity)?

Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) only
(7,442 clips, 91 actors, 6 emotions). **Not** FER2013. Already present at
`data/CREMA-D/` — not downloaded or modified by this repo's code.

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
back to CPU otherwise (no CUDA on this machine). Phases that need large-scale
GPU training (e.g. extensive wav2vec2/WavLM fine-tuning, large hyperparameter
sweeps) will be flagged for a move to Kaggle/Colab when reached — not decided
up front.

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
  Face Landmarker API will be confirmed when Phase 2 (face detection) starts.

## Non-negotiable evaluation rules (see project instructions for full detail)

1. Actor-level, speaker-independent splits — no actor in more than one of
   train/val/test. Enforced by an assertion that runs every time a split is
   loaded (`fer.splits.load_split` / `assert_no_leakage`), not just in tests.
2. The test set is fixed on day one (`splits/split_v1.json`) and is only
   touched by a future `evaluate.py --final`.
3. Macro-F1 is the primary metric, reported alongside accuracy, per-class F1,
   and confusion matrices.
4. Every reported number is mean ± std over 3 seeds.
5. Every run is tracked in MLflow (local by default; W&B via a switch).
6. No invented numbers. Failures are reported plainly.

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
  | eGeMAPS + LR | 0.4903 ± 0.0000 | 0.4926 ± 0.0000 |
  | chance (stratified dummy) | 0.1587 ± 0.0037 | 0.1599 ± 0.0043 |

  The LR std is exactly 0 across the 3 seeds because `lbfgs` is a deterministic solver with no
  stochastic component on a fixed design matrix — not a bug, and reported as such rather than masked.
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
the 1,224 val clips and was killed rather than left to run. Redesigned into two phases (both in
`scripts/11_face_model_occlusion_eval.py`; settings recorded in `configs/face_model.yaml`):

1. **Detection cache (CPU, multiprocess, run once):** 8 *evenly spaced* frames per clip — the same 8
   frames for every occlusion type, severity, and (if re-run later) model, so results are comparable
   apples-to-apples rather than each condition drawing its own sample. For each (clip, frame, type,
   severity): apply the occlusion, run MediaPipe detection, cache the True/False result
   (`cache/phase4_occlusion_detection_cache.parquet`) — reusable by future face models, since detection
   depends only on the occluded image, not model weights. Parallelized across CPU cores, reserving 2 so
   the Mac stays usable.
2. **Model evaluation (MPS, single process, batched, `torch.no_grad()`, eval mode):** reads the cached
   True/False table — detection is never re-run here. Regenerates (cheap, deterministic) the occluded
   crops only for frames the cache marked detected, batches them per clip/condition, and averages
   softmax probabilities.

**Face-branch missing-detection rule:** a frame with no detection casts no vote. If a clip has zero
surviving frames under a condition, the face branch abstains by outputting a uniform distribution over
the 6 classes rather than being skipped — the clip still counts in macro-F1 (scored via a deterministic
argmax tie-break), so a condition that defeats detection shows up as a real penalty instead of
vanishing from the average.

A 20-clip dry run measured ~2.6 clips/s for the detection phase and ~0.23s/clip for the eval phase,
projecting a ~15-20 min full run; the actual full run took **10.3 minutes** (3.5 min detection-cache
build + 6.8 min model eval) — comfortably inside the pre-agreed ~1 hour budget.

### Occlusion results (VAL, seed-0 model, 1,224 clips)

| occlusion_type | severity | macro-F1 | accuracy | abstained (zero-detection clips) |
|---|---|---|---|---|
| mask | 0.3 | 0.1182 | 0.2092 | 0/1224 |
| mask | 0.6 | 0.1142 | 0.2083 | 0/1224 |
| mask | 0.9 | 0.1204 | 0.2173 | 0/1224 |
| sunglasses | 0.3 | 0.2570 | 0.3578 | 0/1224 |
| sunglasses | 0.6 | 0.2787 | 0.3734 | 0/1224 |
| sunglasses | 0.9 | 0.2515 | 0.3587 | 0/1224 |
| patch | 0.3 | 0.4024 | 0.4314 | 0/1224 |
| patch | 0.6 | 0.2496 | 0.3121 | 26/1224 |
| patch | 0.9 | 0.1308 | 0.2010 | **493/1224 (40.3%)** |
| hand_over_mouth | 0.3 | 0.2926 | 0.3799 | 0/1224 |
| hand_over_mouth | 0.6 | 0.2581 | 0.3480 | 0/1224 |
| hand_over_mouth | 0.9 | 0.2406 | 0.3252 | 0/1224 |

Full table: `outputs/phase4_occlusion_eval_table.csv`.

- Every occlusion condition collapses macro-F1 well below the clean 0.5724 — even the mildest
  (sunglasses/hand_over_mouth at severity 0.3) roughly halve it. `mask` is the most damaging type at
  every severity (macro-F1 ≈ 0.11-0.12) despite covering a similar-sized area to the others, plausibly
  because it occludes the nose-to-chin region the model may be relying on most — consistent with the
  clean model's own weakest class (`sad`) being confused mainly with `fear`/`anger`, all mouth/jaw-cue
  dependent.
- `patch` is the only type whose detection failure rate itself degrades sharply with severity (0
  abstentions at 0.3 → 26 at 0.6 → **493/1224 at 0.9**), matching Phase 2 Part B's finding that this
  random, non-landmark-aware occlusion sometimes lands over the eyes. Its macro-F1 at severity 0.3
  (0.40) is the *highest* of any occluded condition — a small, randomly-placed patch is often less
  disruptive than a full lower-face mask — but by severity 0.9 it is both the most detection-defeating
  and among the worst-performing conditions.
- This is exactly the failure mode the planned confidence-weighted fusion (Phase 5) is meant to
  address: under every occlusion condition here the face branch is far weaker than the clean audio
  baseline (macro-F1 0.49) alone, so a fusion mechanism that shifts weight toward audio under visual
  uncertainty should recover much of this loss — not yet tested, since fusion hasn't been built.

## Project structure

```
src/fer/          # package: filenames, probe, demographics, metadata, splits, envinfo,
                  # audio_features, uncertainty, face_pipeline
configs/          # YAML configs (data.yaml, audio_baseline.yaml, ...)
scripts/          # one script per pipeline stage
splits/           # versioned, hashed actor split files (tracked in git)
tests/            # pytest suite, incl. the leakage check
outputs/          # generated artifacts (gitignored): metadata.csv, reports, contact sheets, etc.
cache/            # cached features (gitignored, regenerable)
models/           # downloaded model assets, e.g. mediapipe/face_landmarker.task (gitignored)
data/             # CREMA-D dataset (gitignored — never committed)
```

## Later phases (not started)

Audio models (wav2vec2/WavLM fine-tuning) → fusion (calibration + confidence-weighted late fusion,
testing whether it recovers the occlusion losses measured in Phase 4) → full experiment grid +
RESULTS.md → Gradio demo.
