---
title: "Does Voice Compensate When the Face Is Occluded? Audio-Visual Emotion Recognition Under Synthetic Facial Occlusion"
author: "Yashashvi Jangid, Independent Researcher"
date: "November 2026"
geometry: margin=0.85in
fontsize: 11pt
colorlinks: true
---

## 1. Abstract

Facial occlusion (masks, sunglasses, hand-over-mouth, random patches) degrades face-based emotion
recognition, but it need not degrade speech. This work asks whether a voice-based classifier can
compensate for occlusion-induced face-branch failure under late fusion, using CREMA-D (7,438 clips,
91 actors, speaker-independent 64/15/12 train/val/test split). Three audio-visual models — eGeMAPS
features with logistic regression, WavLM-base-plus, and a ResNet-18 face branch trained clean or with
synthetic occlusion augmentation — are combined with four fusion rules (fixed averaging, two
confidence-weighted variants, and a learned gate). Three comparisons were pre-specified before the
frozen 12-actor test set was touched: (P1) fused prediction vs. voice alone, (P2) occlusion-robust vs.
clean-trained face generalization to an unseen occlusion type, and (P3) confidence-weighted vs. fixed
fusion. On the test set, fixed 50/50 averaging significantly outperforms voice alone in every one of 13
conditions (P1), occlusion-augmented training generalizes to the held-out mask type at every severity
(P2), and — contrary to the motivating hypothesis — fixed averaging significantly beats confidence
weighting overall (P3), though not in every individual condition. The result suggests that, at this
scale, naive averaging already captures most of the benefit that explicit reliability weighting was
meant to provide.

## 2. Introduction

The motivating question for this work is simple: when a speaker's face is occluded — by a mask, sunglasses,
a hand, or an arbitrary patch — does their voice carry enough emotional information to compensate, and
does a fusion rule that explicitly weights branches by their own confidence do a better job of
exploiting that compensation than a fusion rule that does not?

This is a practical question for any audio-visual emotion recognition system deployed outside a
controlled studio: video calls, telehealth, in-cabin monitoring, and public-safety audio-visual feeds
all involve faces that are sometimes occluded by ordinary objects, by camera framing, or by the speaker's
own hand. A system that degrades gracefully under occlusion — by leaning on audio when the face fails —
is more useful than one that does not.

Three comparisons were specified in writing before the frozen test set was accessed (the pre-specified
plan and its one later amendment are both committed to the project's version-control history; the
amendment's commit hash, `09e454e981079e7197b46c9c92b0fc2b9ace1a74`, fixes the exact methodology below
before any test number was seen):

- **P1.** Does fixed 50/50 fusion of an occlusion-robust face branch and a strong voice branch beat the
  voice branch alone, across clean and occluded conditions?
- **P2.** Does a face model trained with synthetic occlusions of three types generalize to a fourth,
  held-out occlusion type (a mask) that it never saw during training?
- **P3.** Does confidence-weighted fusion (weighting each branch by its own predicted confidence) beat
  fixed 50/50 averaging, once both branches are temperature-calibrated?

P3 is the paper's central methodological question. The intuitive expectation — and the reason
confidence-weighted fusion is a standard technique in multimodal literature — is that a branch should be
trusted less when it is uncertain, and an occluded face branch should become visibly less confident. This
work tests that expectation directly and reports whatever the test set shows, including a negative
result.

## 3. Related Work

CREMA-D (Cao et al., 2014) is a crowd-sourced, single-sentence emotional speech and face dataset with
human recognition accuracies of 40.9% (audio-only), 58.2% (visual-only), and 63.6% (audio-visual),
establishing that fusing the two modalities helps human listeners/viewers on this exact corpus. Several
works fuse audio and visual streams on CREMA-D with temporal or architectural refinements; Birhala,
Ristea, Radoi, and Dutu (2020) fuse audio and visual features sampled from a temporal window with
independent per-modality offsets and report results on CREMA-D exceeding several prior fusion baselines.
None of the CREMA-D fusion work we are aware of evaluates performance under deliberately occluded faces,
which is the gap this work addresses.

Facial occlusion itself has a substantial literature of its own: Zhang, Verma, Tjondronegoro, and
Chandran (2018) survey facial expression recognition under partial occlusion and note that occlusion
is understood to primarily degrade visual features specifically, while leaving other modalities (audio,
gesture, physiological signal) largely unaffected — the premise this work's core question rests on,
though their survey covers vision-only recognition and robustness techniques rather than audio-visual
fusion under occlusion.

Reliability- or confidence-aware fusion — weighting a modality's contribution by a measure of its own
uncertainty rather than fusing with fixed weights — is a recurring idea in multimodal learning broadly,
and in occlusion-aware visual expert mixtures specifically [CITATION NEEDED]. We were unable to verify a
specific, citable prior work evaluating *confidence-weighted audio-visual* fusion against *fixed* fusion
on synthetically occluded faces — the direct point of comparison for P3 — nor a specific, verifiable
reference for gated or modality-dropout-based audio-visual fusion for degraded modalities; both are left
as [CITATION NEEDED] rather than invented.

## 4. Data and Evaluation Protocol

**Dataset.** CREMA-D contains 7,442 raw audio-video clips from 91 actors; 4 clips are excluded for
documented data-quality reasons (a mislabeled compilation clip and three clips with near-zero or
corrupted video), leaving a canonical set of **7,438 clips** used throughout. Each clip is a single
spoken sentence labeled with one of six emotions (anger, disgust, fear, happiness, neutral, sadness).

**Split.** Actors are partitioned into training (64 actors), validation (15 actors), and test (12
actors) sets, speaker-independent — no actor appears in more than one split. This split was frozen
before any model development began and is enforced by an automated leakage check on every data load.
The 12-actor test set was not touched by any model training, hyperparameter selection, or calibration
decision; it was accessed exactly once, for scoring only, after the evaluation plan below was committed.

**Pre-specified evaluation plan.** A written evaluation plan specifying the exact models, conditions,
fusion methods, and the three statistical comparisons (P1–P3) was committed to version control before
the test set was accessed. One amendment to that plan — freezing calibration and the fusion gate on
validation data rather than re-fitting them on the test actors, and promoting P1–P3 to primary status —
was made and committed (hash `09e454e981079e7197b46c9c92b0fc2b9ace1a74`) immediately before, and
independently of, the test run itself. No further changes were made after test results were observed.

**Metric.** Macro-averaged F1 (macro-F1) across the six emotion classes is the primary metric throughout,
reported alongside accuracy. Every model involving stochastic training (the face branch, WavLM) is
trained with 3 random seeds; every number is reported per-seed and as a mean ± standard deviation across
seeds. Statistical comparisons use an actor-level paired bootstrap: the resampling unit is the actor (not
the clip), because clips from the same actor are not independent observations. Each of 2,000 bootstrap
draws resamples the 12 test actors with replacement, pools all of a drawn actor's clips, computes the
paired macro-F1 difference between the two methods being compared for each of the 3 seeds using that same
resampled actor set, and averages the difference across seeds; the reported 95% confidence interval is the
2.5th/97.5th percentile of this distribution, and a two-sided p-value is computed from the fraction of
bootstrap differences on either side of zero. We use **"pre-specified"** rather than "pre-registered"
throughout this report, since the commitment mechanism was version-control history on a private project
repository rather than a registry with external timestamping.

**Abstention rule.** The face branch can fail to detect a face at all on a heavily occluded frame. A
clip where every sampled frame fails detection is scored as an abstention: it counts as a false negative
for its true emotion class, is never counted as a (possibly wrong) prediction of any class, and remains
in the accuracy denominator. Neither audio branch (eGeMAPS+LR nor WavLM) abstains.

## 5. Methods

**Audio branch 1 — eGeMAPS + logistic regression.** 88-dimensional eGeMAPSv02 functional acoustic
features (extracted with openSMILE) are standardized and classified with a multinomial logistic
regression, with its regularization strength tuned on the validation set.

**Audio branch 2 — WavLM.** `microsoft/wavlm-base-plus`, with its convolutional feature encoder frozen
and a small two-layer classification head fine-tuned on top of a masked mean pool over the transformer's
final hidden states, trained on the 64 training actors with early stopping on validation macro-F1, 3
seeds. Inference uses a batch size of one: the frozen convolutional encoder has no attention-masking
mechanism, so zero-padding a batch of variable-length clips measurably shifts normalization statistics
within the encoder and can change a clip's predicted class depending on which other clips happen to
share its batch — a batch-size-of-one inference pass removes this dependency so that a clip's score
depends only on its own audio and the trained weights.

**Face branch — ResNet-18.** A ResNet-18 image classifier is fine-tuned on face crops extracted from
video frames via MediaPipe face landmarking, in three variants: **clean-trained** (no occlusion during
training), **aug_all** (all four synthetic occlusion types applied during training), and **aug_no_mask**
(three occlusion types applied during training, with the mask type deliberately withheld so that its
test-time performance measures generalization to a genuinely unseen occlusion rather than memorization
of the synthetic mask generator). Clip-level prediction averages per-frame softmax probabilities across
multiple sampled frames per clip.

**Synthetic occlusions.** Four occlusion types are rendered on top of a detected face using its
MediaPipe landmarks: a lower-face **mask** (coverage increases continuously with severity, from
mouth-and-chin at the lowest severity to just below the eyes at the highest), **sunglasses** (an opaque
ellipse over the eye region), a randomly placed gray **patch**, and a **hand-over-mouth** occlusion
anchored at the mouth landmarks. Each type is applied at three severities, giving 12 occlusion
conditions plus the clean condition, 13 in total.

**Fusion.** Four methods combine the two per-branch probability vectors: **fixed 50/50** averaging;
**confidence-weighted (max-probability)**, which weights each branch by its own calibrated maximum
softmax probability; **confidence-weighted (negative entropy)**, an analogous weighting using negative
entropy as the confidence signal; and a **learned gate**, a logistic regression predicting whether the
face branch is correct from both branches' confidence scores, used as the face branch's fusion weight.
Both per-branch probability vectors are temperature-calibrated before fusion. Per the pre-specified
amendment, both the calibration temperatures and the learned gate are fit once on the 15 validation
actors and frozen; the 12 test actors' labels are used only to score predictions, never to fit any
calibration or gating parameter.

## 6. Results

All results below are from the frozen 12-actor test set, evaluated exactly once. Pooled figures average
over the 12 occlusion conditions (clean excluded); per-condition breakdowns are given for the three
pre-specified comparisons.

**Table 1. Branch-level macro-F1 (test set, mean ± std over 3 seeds, with actor-level bootstrap 95% CI
where computed).**

| Branch | Condition | Macro-F1 | 95% CI |
|---|---|---|---|
| eGeMAPS + LR (tuned) | — (deterministic) | 0.5044 ± 0.0000 | [0.4576, 0.5424] |
| WavLM | — | 0.7049 ± 0.0121 | [0.6269, 0.7616] |
| Face, clean-trained | clean | 0.6050 ± 0.0151 | [0.5034, 0.6987] |
| Face, aug_all | clean | 0.6227 ± 0.0100 | [0.5207, 0.7143] |
| Face, aug_no_mask | clean | 0.6303 ± 0.0137 | [0.5196, 0.7279] |

**Table 2. Fusion (WavLM + face), macro-F1 mean over 3 seeds, pooled over the 12 occlusion conditions and
on the clean condition alone.**

| Face variant | Pooled (12 occl.) | Clean |
|---|---|---|
| clean-trained | 0.710 | 0.7754 ± 0.0163 |
| aug_all | 0.742 | 0.7781 ± 0.0095 |
| aug_no_mask | 0.739 | 0.7781 ± 0.0120 |

Fixed 50/50 fusion is the best of the four fusion methods for every face/audio-branch pairing tested
except clean-trained face + eGeMAPS, where the learned gate edges it out by 0.003 macro-F1.

**P1 — fixed fusion vs. WavLM alone (aug_all face).** Significant in every one of the 13 conditions.
Pooled: +0.0404 (95% CI [0.0182, 0.0629], p = 0.001). The effect is largest on clean faces (+0.0731,
CI [0.0355, 0.1072]) and shrinks under the heaviest occlusion (mask 0.9: +0.0099; patch 0.9: +0.0108) but
never loses significance or reverses sign.

**P2 — aug_no_mask vs. clean-trained face-only, under the held-out mask type.** Significant at every
severity: +0.0935 (severity 0.3, CI [0.0496, 0.1331]), +0.2258 (severity 0.6, CI [0.1802, 0.2742]), and
+0.0584 (severity 0.9, CI [0.0400, 0.0766]), all p < 0.001.

**P3 — confidence-weighted (max-prob) vs. fixed fusion (aug_all + WavLM).** Pooled, fixed fusion wins
significantly: −0.0038 (CI [−0.0050, −0.0025], p < 0.001) in confidence-weighting's favor being tested
against — i.e., confidence weighting underperforms fixed fusion. Condition-by-condition, this is **not**
uniform: of the 13 conditions, 9 individually reach significance in fixed fusion's favor (mask at all
three severities, sunglasses at all three, patch at severities 0.3 and 0.6), while **4 do not** — patch
0.9 (p = 0.83), and hand-over-mouth at all three severities (p = 0.09–0.56) — and clean sits at the
p = 0.05 boundary.

## 7. Discussion

**Why fixed averaging is hard to beat.** When one branch's output distribution is close to uniform
(the behavior of an uncertain or abstaining classifier), averaging it with a confident branch barely
perturbs the confident branch's peak class, because a near-uniform vector adds roughly the same small
amount to every class and so does not change the arg-max. Explicit confidence weighting only earns its
keep when a branch is *confidently wrong* rather than merely hedged — a case that does occur in this
project (the clean-trained face branch collapses onto a single wrong class under heavy masking, with real
confidence, rather than hedging), but it is the exception rather than the rule across the 13 conditions
tested, and the machinery needed to detect it (temperature scaling, a learned gate, or an entropy-based
signal) introduces its own variance. Fixed averaging has no corresponding failure mode, because it never
acts on either branch's self-reported confidence at all.

**The negative result for confidence weighting is the paper's central finding, not an afterthought.**
The motivating hypothesis behind P3 was intuitive enough to be worth testing formally rather than
assuming: a system should trust a branch less when that branch is less sure of itself, and an occluded
face branch should visibly become less confident. The pre-specified test shows that — at this scale of
training data, with these two specific audio and face architectures, and with the particular confidence
signals tested (calibrated max-softmax probability, negative entropy, and a learned logistic gate) — this
intuition does not hold up overall, and fixed averaging is a strictly simpler, equally or more effective
default. Where the gap between the two is largest (sunglasses, mid-severity patch) the face branch is
degraded but not yet collapsed to a uniform or systematically wrong output; where the gap disappears
(hand-over-mouth, severe patch) the face branch has either collapsed into the uniform-abstention regime,
where fixed and confidence-weighted fusion converge by construction, or into some other regime this
project's confidence signals do not distinguish from genuine uncertainty.

## 8. Limitations

- **Synthetic occlusions.** All occlusions are rendered digitally on top of clean footage using detected
  facial landmarks, not captured from genuinely occluded recordings. Real occlusions (an actual surgical
  mask, real sunglasses, a real hand) vary in material, fit, lighting interaction, and motion in ways a
  landmark-anchored synthetic overlay cannot fully reproduce.
- **Acted speech.** CREMA-D emotions are performed by actors reading fixed sentences, not spontaneous
  affect. Results here characterize performance on acted, scripted emotional expression and may not
  transfer directly to spontaneous or naturalistic emotional speech and expression.
- **A 12-actor test set and correspondingly wide confidence intervals.** Every reported test-set
  confidence interval reflects the actor-level sampling uncertainty of a 12-actor set; several absolute
  (non-paired) intervals overlap each other even where the corresponding *paired* comparison was
  significant (e.g., fixed-fusion and WavLM-alone absolute intervals on clean faces overlap, even though
  P1's paired test on the same data is significant). Paired comparisons, which exploit the correlation
  between two methods' scores on the same resampled actors, are more sensitive than comparing two
  independent intervals and are the basis for every significance claim in this report.
- **WavLM training-time batch-padding noise.** The batch-composition sensitivity addressed at inference
  time (by scoring one clip per forward pass) was also present during WavLM's own fine-tuning on Kaggle,
  where batches were padded and shuffled every epoch — an uncharacterized source of training stochasticity
  beyond the usual data-order and weight-init randomness. This does not invalidate the checkpoints used
  here; retraining with length-bucketed batching to reduce it was not attempted in this project.
- **A demographic age difference observed only on development (validation) data, not tested on the
  frozen test set.** A secondary, single-training-seed analysis on the 79 non-test actors found a
  confirmed macro-F1 gap, by bootstrap confidence interval, between actors aged 30 or under and actors
  aged 46–60, for the occlusion-augmented face branch (+0.134, 95% CI [0.068, 0.201]) and the
  corresponding fusion pipeline (+0.095, CI [0.022, 0.170]); the audio branch alone showed no such gap
  (+0.004, CI [−0.057, 0.068]). This analysis used 1 training seed per fold (a reported scope reduction
  from the usual 3) and was deliberately kept out of scope for the frozen test-set evaluation; it is
  reported as a preliminary, development-stage finding for a future, adequately powered study to test
  formally, not as a claim about the test set.
- **Test-set scores are higher than validation-set scores for every model and condition compared.** Both
  audio branches, all three face variants, and the fusion pipeline score somewhat higher on the 12-actor
  test set than the 15-actor validation set (e.g., WavLM: 0.6929 validation vs. 0.7049 test; clean-trained
  face on clean faces: 0.5724 validation vs. 0.6050 test). Reported as observed; with only 12 and 15
  actors respectively this is consistent with ordinary actor-sampling variability between two disjoint
  speaker sets, not evidence of a systematic split difference, and was not investigated further as it was
  outside this evaluation's pre-specified scope.

## 9. Conclusion

On a frozen, speaker-independent test set of 12 actors, fusing a strong voice classifier with an
occlusion-robust face classifier significantly outperforms voice alone across every occlusion condition
tested, including the clean condition (P1); training the face branch on synthetic occlusions that
exclude a given occlusion type still measurably improves performance on that excluded type at test time
(P2); and, contrary to the intuitive expectation that motivated much of this project's earlier
development, explicitly weighting each branch by its own confidence does not improve on simple fixed
averaging — if anything, fixed averaging wins, significantly, pooled across conditions, though the
effect is not uniform condition-by-condition (P3). The practical implication for a deployed audio-visual
emotion recognition system handling occluded faces is that a simple, parameter-free fusion rule is a
strong and difficult-to-beat default; the engineering effort involved in building and calibrating a
confidence-weighted or learned fusion mechanism was not justified by this project's results.

## 10. Statement on the Use of AI Coding Assistants

This project's code — data processing, model training, evaluation, calibration, bootstrap statistics,
and the Kaggle notebook used for WavLM fine-tuning — was developed with substantial assistance from an
AI coding assistant (Claude, Anthropic), operating under the direction and review of the author at every
step: every methodological decision (the split protocol, the calibration-freezing amendment, the choice
of pre-specified comparisons, scope reductions, and this report's structure and content) was made or
explicitly approved by the author before being implemented or written. The AI assistant was also used to
draft this report's prose from the project's committed, author-reviewed results; all numbers in this
report were read directly from the project's version-controlled results log rather than computed by, or
left to the discretion of, the assistant while drafting. The author takes full responsibility for the
correctness and interpretation of all content in this report.
