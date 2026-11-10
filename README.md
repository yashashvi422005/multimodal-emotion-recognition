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

## Project structure

```
src/fer/          # package: filenames, probe, demographics, metadata, splits, envinfo
configs/          # YAML configs (data.yaml, ...)
scripts/          # one script per pipeline stage
splits/           # versioned, hashed actor split files (tracked in git)
tests/            # pytest suite, incl. the leakage check
outputs/          # generated artifacts (gitignored): metadata.csv, reports, etc.
data/             # CREMA-D dataset (gitignored — never committed)
```

## Later phases (not started)

Preprocessing (face crops + occlusion synthesis, audio resampling) → audio
models (eGeMAPS/SVM, then wav2vec2/WavLM) → face models (timm backbones) →
fusion (calibration + confidence-weighted late fusion) → full experiment
grid + RESULTS.md → Gradio demo.
