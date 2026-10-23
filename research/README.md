# Baseline research audit — 2026-10-22

2026-10-23 Exp01: the [T198 jersey oracle experiment](experiments/exp01_t198_jersey_oracle/README.md) is **CONFIRMED for structural recovery**: +188 B6 rows, with unchanged original downstream code. Fresh TI-HOTA is 0.652881 versus frozen baseline 0.635142. The report explicitly records equal-confidence A12 deduplication interactions, so the score delta is not claimed as a perfectly isolated B6-only effect. No production fix or model inference was performed.

2026-10-23 follow-up: reproduced Colab outputs are now supplied under `results/`. [B11/B6 causal analysis](analysis/results_b11_b6_20261023/README.md) verifies the 930-row coverage deficit, preserves all supplied results, and separates accepted-jersey failures from mixed tracker identities and interpolation boundaries. Status: **PARTIAL**, with detector-versus-tracker gaps and rejected OCR evidence unresolved. Earlier environment/audit statuses below describe the previous sprint.

**Baseline status: PARTIAL. End-to-end reproduction: BLOCKED.**

Kaggle compatibility follow-up: **READY_FOR_KAGGLE**. A separate standalone notebook is prepared; no Kaggle GPU result is claimed. See [the compatibility report and Run All recipe](kaggle.md) and [final local validation](runs/20261022-05_kaggle-compatibility_local-validation/validation.json). The original audit below remains historical context.

The repository initially contained only one notebook, with saved outputs but no dataset, models, package lock, executable scripts, or research notes. We inspected all 15 code cells and downloaded the tutorial GT and court calibration. The original notebook has not been changed. No tracker/model upgrade, training, inference optimization, or oracle experiment was performed.

The implemented pipeline is:

```text
IMG_0104_1.MOV
→ YOLO11x person detection + Ultralytics ByteTrack
→ normalized tracked labels → pixel-space MOT rows
→ bbox bottom centers → fixed Outdoor homography in cm
→ track-level and detection-level court filtering
→ full-player EasyOCR votes + torso HSV two-cluster team separation
→ initial-position cluster-to-team alignment
→ roster-compatible semantic identities
→ highest-confidence duplicate resolution + within-identity interpolation
→ five-column predictions
→ notebook TI-HOTA + location-only diagnostics
→ GT/prediction minimaps + original-video overlay
```

The module map, input/output contracts, parameters, assumptions, and artifact locations are in [architecture.md](architecture.md). The exact notebook order, local commands, and environment differences are in [reproduction.md](reproduction.md).

## Results and evidence

- Run 01: [run manifest](runs/20261022-01_baseline-audit_tutorial_eval-smoke/run.json), [environment](runs/20261022-01_baseline-audit_tutorial_eval-smoke/environment.json), [baseline failure](runs/20261022-01_baseline-audit_tutorial_eval-smoke/baseline_preflight.log). Cell 3's Python portion stops at missing `cv2`; its installation magics were deliberately not executed against the global interpreter.
- Run 02: [run manifest](runs/20261022-02_baseline-audit_tutorial_homography-smoke/run.json), [environment](runs/20261022-02_baseline-audit_tutorial_homography-smoke/environment.json), [baseline failure](runs/20261022-02_baseline-audit_tutorial_homography-smoke/baseline_preflight.log). An isolated environment permits unchanged cell 11 to run on real calibration; preparation still stops at missing `torch`.
- [GT inventory](runs/20261022-01_baseline-audit_tutorial_eval-smoke/gt_inventory.json): 6,708 rows, 1,118 frames, six semantic identities, frames 1–1,118.
- [Homography output](runs/20261022-02_baseline-audit_tutorial_homography-smoke/homography.json) and [stdout](runs/20261022-02_baseline-audit_tutorial_homography-smoke/homography.log): actual baseline calibration stage, not an estimated matrix copied from saved output. Calibration-fit residual is not independent localization validation.
- [Synthetic evaluator contract checks](runs/20261022-02_baseline-audit_tutorial_homography-smoke/synthetic_evaluator_checks.json): perfect, wrong identity, missing prediction, extra prediction, empty frame, and temporary tracker-ID change cases pass; the similarity at 100 cm is also checked.
- [Real GT loader smoke](runs/20261022-02_baseline-audit_tutorial_homography-smoke/real_gt_loader_smoke.json): frames 1–3, deliberately empty predictions, 18 FN, 0 TP/FP. Its zero score is a negative-control check, **not a baseline model score**.
- Every run includes source snapshots, a cell index, historical text outputs, and SHA-256 hashes. Original notebook SHA-256: `33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00`.

**Current independently reproduced end-to-end score: unavailable.** Saved notebook outputs report TI-HOTA 0.6351, TI-DetA 0.6362, TI-AssA 0.6341, TI-TP 4801.42, TI-FP 976.58, TI-FN 1906.58 on the tutorial clip. These are historical and unverified. Tracking output says `track`, while subsequent outputs say `track-4`; execution counts are all null. They cannot prove a single clean execution. No predictions or generated video files were supplied.

## Research readiness

The evaluator, homography helper, MOT files, filtered track tables, and final CSV provide useful boundaries. Cell 17 combines sampling, OCR, team classification, semantic filtering, merging, interpolation, and output generation in 910 lines using notebook globals. Controlled experiments need cached inputs at those internal boundaries and provenance back to tracker observations. No large refactor is required; see [experiments.md](experiments.md).

Main failure surfaces are court/foot-point localization, hard track rejection, uncertain or missing OCR, team-cluster anchor failure, roster-based rejection, confidence-based duplicate selection, and unbounded interpolation across identity gaps. Existing videos support manual inspection, but the evaluator cannot locate tracker switches or fragments, and saved predictions lose tracker provenance. These are observations to investigate, not changes made to the baseline.

The smallest useful additions are an H JSON, per-track filter decisions, per-sample OCR evidence, a track-to-semantic mapping, observed/interpolated flags, and frame-indexed matching diagnostics. Only H persistence and audit/evaluator evidence were implemented by the external audit wrapper; inference remains unchanged.

## Recommended next experiments (not implemented)

1. **Oracle jersey at the track evidence boundary.** Freeze filtered tracking/court rows and team mapping; use manually verified GT-linked jersey labels, retaining the same dropping/merging/interpolation logic. This estimates how much evidence loss and wrong jersey votes affect the outcome.
2. **Oracle semantic identity at the track-to-identity boundary.** Freeze localization and supply a reviewed frame/track-to-player mapping; preserve downstream logic. Compare with experiment 1 to separate team/identity assignment issues from jersey evidence. Mixed-person tracks require explicit frame-level mapping rather than assuming one identity per track.
3. **Oracle localization on matched final prediction rows.** Preserve semantic labels, row support, and missing/extra rows; replace only matched court positions using independently reviewed GT correspondence. This isolates localization sensitivity without manufacturing detections or removing false positives.

Ranked by information gain from the observed interfaces and saved evidence, not expected leaderboard improvement. Record unmatched rows and oracle availability; do not compare different frame supports as though only one component changed.

## Files added / changed

- Root `README.md`: entry point.
- `research/README.md`: report, evidence index, changes, next experiments.
- `research/architecture.md`: all stages, data contracts, artifact and failure inspection inventory.
- `research/reproduction.md`: canonical notebook order, local recipe, failures and environment changes.
- `research/evaluation.md`: exact notebook metrics and their limits.
- `research/experiments.md`: naming, provenance conventions, minimal component interfaces.
- `research/tools/baseline_audit.py`: source/evidence snapshot, preflight, optional real calibration execution, evaluator checks, optional existing-CSV evaluation; dynamically uses original notebook source.
- `research/runs/20261022-01_baseline-audit_tutorial_eval-smoke/*` and `research/runs/20261022-02_baseline-audit_tutorial_homography-smoke/*`: generated text source snapshots, saved text outputs, manifests, downloaded GT/calibration, logs, checks, matrices and hash inventories. Source snapshots are documentation copies, not a replacement pipeline.

The notebook is unchanged. `/tmp/trackid3x3-audit-venv` holds isolated OpenCV/ndjson dependencies; `/tmp/trackid3x3-audit` holds temporary inspection material. No global package was installed or updated.
