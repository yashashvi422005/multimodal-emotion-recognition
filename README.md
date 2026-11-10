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
- 31 clips are H.264-encoded vs. VP6 (`vp6f`) for the rest; 28 of these
  exactly match the README's list of clips that had audio re-attached in
  Dec. 2020. The other 3 are not individually documented in the README but
  show the same re-encode signature. Kept — codec choice doesn't affect
  label validity.

**Decision (approved at CHECKPOINT 1): exclude the 3 clips above from the
canonical dataset.** Final clip count: **7,439** (7,442 raw, minus the 3
excluded). Per-emotion counts after exclusion: anger 1271, fear 1271, happy
1271, disgust 1270, sad 1270, neutral 1086.

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

- 6,461/6,461 train+val clips extracted cleanly (0 failures, 0 NaN/Inf). Test-actor clips (978) are
  deliberately excluded from extraction — they'll only be extracted inside a future `evaluate.py --final`.
- `StandardScaler` (fit on train only) + `LogisticRegression(class_weight="balanced", solver="lbfgs")`,
  trained on TRAIN actors, evaluated on VAL actors only:

  | | macro-F1 | accuracy |
  |---|---|---|
  | eGeMAPS + LR | 0.4912 ± 0.0000 | 0.4935 ± 0.0000 |
  | chance (stratified dummy) | 0.1587 ± 0.0037 | 0.1599 ± 0.0043 |

  The LR std is exactly 0 across the 3 seeds because `lbfgs` is a deterministic solver with no
  stochastic component on a fixed design matrix — not a bug, and reported as such rather than masked.
- Because that ±0.0000 only proves determinism, not generalization, speaker-level uncertainty was
  estimated two ways (neither touches test): a cluster bootstrap over the 15 VAL actors (1000
  resamples) gives macro-F1 95% CI **[0.4370, 0.5448]**; actor-level 5-fold `GroupKFold` CV over all 79
  train+val actors gives macro_f1 = **0.5145 ± 0.0247** across folds (pooled out-of-fold macro_f1 0.5161).
- Per-class F1 (LR): anger 0.69 (easiest), neutral 0.51, sad 0.50, happy 0.48, fear 0.39, disgust 0.37
  (hardest) — consistent with typical audio-only emotion-recognition results.
- All runs logged to MLflow (`sqlite:///mlflow.db`, gitignored; plain-file tracking is deprecated as of
  the installed mlflow 3.18 and raises on use), tagged with the split content-hash and git commit.

## Phase 2 pilot (Part A only): 20-clip feasibility check

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
- Over 491 sampled frames (~10fps) across the 20 clips: **0% detection failure rate**, mean **0.14s**
  wall time per clip on this Mac. Extrapolated to all 7,439 clips: **~17.5 minutes** — comfortably
  within local-CPU reach; **no need to move Phase 2 preprocessing to Kaggle/Colab.**
- All 4 occlusion types (mask, sunglasses, patch, hand_over_mouth) implemented as landmark-placed,
  deterministic-per-clip-and-seed synthetic overlays; contact sheet saved to
  `outputs/phase2_pilot/contact_sheet.png` (gitignored — regenerate via the script above). Sunglasses
  and patch placement look solid; the mask shape is the roughest of the four (a semi-transparent
  rectangle rather than a mask-contoured polygon) and needs refinement before full Phase 2.

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

Full Phase 2 preprocessing (face crops + occlusion synthesis at scale, audio resampling+caching) →
audio models (wav2vec2/WavLM fine-tuning) → face models (timm backbones) → fusion (calibration +
confidence-weighted late fusion) → full experiment grid + RESULTS.md → Gradio demo.
