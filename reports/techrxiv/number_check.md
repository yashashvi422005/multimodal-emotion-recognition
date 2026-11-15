# Number-check table: every number in `main.tex` and its source

Every numeric claim in `reports/techrxiv/main.tex`, matched to the exact line in `RESULTS.md` it was
read from. Where a number is **not** in `RESULTS.md` (dataset/split constants, commit hashes, or a
standard external fact about a named feature set/architecture), the actual source is given instead —
per the project's rule against pretending a number comes from somewhere it doesn't. Line numbers refer
to the files as of this report's commit.

## Abstract

| Number | Source |
|---|---|
| 7,438 clips | `README.md:132` ("**7,438** (7,442 raw, minus the 4 excluded)") — not in `RESULTS.md` |
| 91 actors | `README.md:10` — not in `RESULTS.md` |
| 64/15/12 split | `splits/split_v1.json` (`assignment`: train=64, val=15, test=12) — not in `RESULTS.md` |
| 0.778 [0.707, 0.840] (fixed fusion, aug_all+WavLM, clean) | `RESULTS.md:670` |
| 0.705 (WavLM alone) | `RESULTS.md:484` (0.7049, rounds to 0.705); also `RESULTS.md:666` |
| 0.742 (pooled, aug_all+WavLM) | `RESULTS.md:502` (fixed_5050 column) |
| "every one of 13 conditions" (P1) | `RESULTS.md:522-535` (13 rows, all significant) |
| "every severity" (P2) | `RESULTS.md:546-548` (3 rows, all significant) |
| −0.004 (P3 pooled) | `RESULTS.md:559` (−0.0038, rounds to −0.004) |
| "13 individual conditions" | `RESULTS.md:557-572` |

## Introduction

| Number | Source |
|---|---|
| Commit hash `09e454e981079e7197b46c9c92b0fc2b9ace1a74` | `git log` / `git rev-parse 09e454e` (confirmed full hash); also quoted in `RESULTS.md:457` |

## Data and Split

| Number | Source |
|---|---|
| 64 training / 15 validation / 12 test actors | `splits/split_v1.json` (`assignment` key) — not in `RESULTS.md` |
| seed 42 | `configs/data.yaml:21` (`split_seed: 42`) and `splits/split_v1.json` (`split_seed`) — not in `RESULTS.md` |
| 7,442 raw clips, 4 excluded, 7,438 canonical | `README.md:110-132` — not in `RESULTS.md` |
| Commit hash `d2af74c65605b4b81b9a4333ce81c597115a6a7e` | `git log` / `git rev-parse d2af74c` (confirmed full hash). **Not mentioned anywhere in `RESULTS.md`'s text** — grepped and confirmed absent; this hash is sourced purely from git history, added per your instruction to include it |

## Methods

| Number | Source |
|---|---|
| 88-dimensional eGeMAPSv02 | Standard property of the eGeMAPSv02 feature set as defined by Eyben et al. (2016) / openSMILE's own documentation, and `src/fer/audio_features.py:16` (`feature_set=opensmile.FeatureSet.eGeMAPSv02`) — **not a number stated anywhere in `RESULTS.md`** |
| 3 seeds | Project-wide convention; evidenced operationally by the 3 per-seed values in every `RESULTS.md` table, e.g. `RESULTS.md:483-484` |
| 64 training actors | `splits/split_v1.json` — not in `RESULTS.md` |
| 4 occlusion types, 3 severities, 12 occlusion conditions + clean = 13 total | `RESULTS.md:72` ("across all 13 conditions") and `RESULTS.md:164` ("13 total") |
| 15 validation actors / 12 test actors (fusion calibration/gate) | `splits/split_v1.json` — not in `RESULTS.md` |

## Evaluation Protocol

| Number | Source |
|---|---|
| Commit hash `09e454e981079e7197b46c9c92b0fc2b9ace1a74` | Same as above; `RESULTS.md:457` |
| 2,000 bootstrap draws (B=2000) | `RESULTS.md:518` ("Paired actor-level bootstrap (B=2000)"); also `RESULTS.md:603` |
| 3 random seeds | As above |
| 12 test actors | `splits/split_v1.json` |
| p-value formula ($p = 2\times\min(\Pr[\text{diff}\le0],\Pr[\text{diff}\ge0])$) | Methodology statement (not a numeric result); matches the amendment's bootstrap-mechanics description, `FINAL_EVAL_PLAN.md`, Amendment section |

## Results — Table 1 (branch-level)

| Number | Source |
|---|---|
| eGeMAPS 0.5044 ± 0.0000 | `RESULTS.md:483` |
| eGeMAPS CI [0.4576, 0.5424] | `RESULTS.md:665` |
| WavLM 0.7049 ± 0.0121 | `RESULTS.md:484` |
| WavLM CI [0.6269, 0.7616] | `RESULTS.md:666` |
| Face clean-trained 0.6050 ± 0.0151 | `RESULTS.md:490` |
| Face clean-trained CI [0.5034, 0.6987] | `RESULTS.md:667` |
| Face aug_all 0.6227 ± 0.0100 | `RESULTS.md:491` |
| Face aug_all CI [0.5207, 0.7143] | `RESULTS.md:668` |
| Face aug_no_mask 0.6303 ± 0.0137 | `RESULTS.md:492` |
| Face aug_no_mask CI [0.5196, 0.7279] | `RESULTS.md:669` |

## Results — Table 2 (fusion, pooled/clean)

| Number | Source |
|---|---|
| clean-trained pooled 0.710 | `RESULTS.md:501` (fixed_5050 column) |
| clean-trained clean 0.7754 ± 0.0163 | `RESULTS.md:628` |
| aug_all pooled 0.742 | `RESULTS.md:502` |
| aug_all clean 0.7781 ± 0.0095 | `RESULTS.md:629`; also point estimate `RESULTS.md:670` |
| aug_no_mask pooled 0.739 | `RESULTS.md:503` |
| aug_no_mask clean 0.7781 ± 0.0120 | `RESULTS.md:630` |
| learned gate edges out fixed by 0.003 (clean-trained+eGeMAPS) | `RESULTS.md:505-507` (0.525 vs 0.522, diff 0.003) |

## Results — P1 prose

| Number | Source |
|---|---|
| pooled +0.0404 [0.0182, 0.0629], p=0.001 | `RESULTS.md:522` |
| clean +0.0731 [0.0355, 0.1072] | `RESULTS.md:523` |
| mask 0.9 +0.0099 | `RESULTS.md:526` |
| patch 0.9 +0.0108 | `RESULTS.md:532` |
| "every one of 13 conditions significant" | `RESULTS.md:537` |

## Results — P2 prose

| Number | Source |
|---|---|
| severity 0.3: +0.0935 [0.0496, 0.1331] | `RESULTS.md:546` |
| severity 0.6: +0.2258 [0.1802, 0.2742] | `RESULTS.md:547` |
| severity 0.9: +0.0584 [0.0400, 0.0766] | `RESULTS.md:548` |
| all p < 0.001 | Same rows, `p` column |

## Results — P3 prose

| Number | Source |
|---|---|
| pooled −0.0038 [−0.0050, −0.0025], p<0.001 | `RESULTS.md:559` |
| clean p=0.050 | `RESULTS.md:560` |
| mask 0.3 p=0.012, 0.6 p=0.045, 0.9 p=0.008 | `RESULTS.md:561-563` |
| sunglasses 0.3 p<0.001, 0.6 p=0.021, 0.9 p<0.001 | `RESULTS.md:564-566` |
| patch 0.3 p<0.001, 0.6 p=0.001 | `RESULTS.md:567-568` |
| patch 0.9 p=0.830 | `RESULTS.md:569` |
| hand_over_mouth 0.3 p=0.093, 0.6 p=0.557, 0.9 p=0.069 | `RESULTS.md:570-572` |
| 8 + 4 + 1 = 13 | Arithmetic over the 13 rows at `RESULTS.md:559-572` |

## Results — Table 3 (secondary, eGeMAPS-based)

| Number | Source |
|---|---|
| eGeMAPS, clean-trained: +0.0311 [−0.0125, 0.0753], p=0.169 | `RESULTS.md:588` |
| WavLM, clean-trained: +0.0098 [−0.0098, 0.0263], p=0.336 | `RESULTS.md:589` |
| eGeMAPS, aug_all: +0.1221 [0.0767, 0.1681], p<0.001 | `RESULTS.md:590` |
| WavLM, aug_all (=P1): +0.0404 [0.0182, 0.0629], p=0.001 | `RESULTS.md:591` (also `:522`) |
| eGeMAPS, aug_no_mask: +0.1085 [0.0582, 0.1595], p=0.001 | `RESULTS.md:592` |
| WavLM, aug_no_mask: +0.0366 [0.0122, 0.0596], p=0.002 | `RESULTS.md:593` |

## Discussion

No new numbers; qualitative discussion of the P3 result already sourced above.

## Limitations

| Number | Source |
|---|---|
| Age gap, face aug_all: +0.134 [0.068, 0.201] | `RESULTS.md:253` |
| Age gap, fusion aug_all: +0.095 [0.022, 0.170] | `RESULTS.md:255` |
| Age gap, audio-only: +0.004 [−0.057, 0.068] | `RESULTS.md:251` |
| Widening, fusion: +0.039 [0.006, 0.075] | `RESULTS.md:269` |
| Widening, face-only: +0.057 [−0.020, 0.125] | `RESULTS.md:268` |
| "1 training seed per fold" | `RESULTS.md:148` |
| 79 non-test actors | `RESULTS.md:146, 152` (train+val combined = 64+15) |
| WavLM: 0.6929 validation vs. 0.7049 test | `RESULTS.md:398` (0.6929 ± 0.0142) and `RESULTS.md:484` (0.7049) |
| clean-trained face, clean: 0.5724 validation vs. 0.6050 test | `README.md:433` (0.5724 ± 0.0046) — **not in `RESULTS.md`**; test value `RESULTS.md:490` |

## Conclusion

No new numbers beyond P1/P2/P3 figures already sourced under Results.

## Data and Code Availability

No result numbers; license version strings (ODbL v1.0, DbCL v1.0) and the repository URL are not
empirical claims and are not sourced from `RESULTS.md`.

## Summary of deviations from "every number from RESULTS.md"

Per your instruction, the table above is honest about which numbers are **not** actually in
`RESULTS.md`, rather than forcing a citation that doesn't exist:

1. **Dataset/split constants** (7,438 clips, 91 actors, 64/15/12, seed 42) — these describe the frozen
   split itself, sourced from `splits/split_v1.json` and `configs/data.yaml`, not `RESULTS.md`.
2. **Two git commit hashes** (`09e454e...`, `d2af74c...`) — sourced from `git log`, not `RESULTS.md`
   (though the first is also quoted inside `RESULTS.md:457`).
3. **The eGeMAPSv02 "88-dimensional" fact** — a standard property of the named feature set / the
   `openSMILE` toolkit configuration in `src/fer/audio_features.py:16`, not a result computed by this
   project.
4. **The validation-side comparison numbers in the last Limitations bullet** (0.6929 WavLM-val,
   0.5724 clean-face-val) — these are val-only development numbers from `README.md`, since
   `RESULTS.md`'s header states it is "test set stays frozen... every number here is development-stage"
   for the early rounds; only the final section of `RESULTS.md` holds test numbers, and the val
   comparators quoted here were never restated there.

No number in the paper was invented or computed fresh for this conversion; every one above traces to
an existing committed file.
