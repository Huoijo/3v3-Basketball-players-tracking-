# Controlled experiments and failure analysis conventions

## Naming and provenance

Use `<YYYYMMDD>-<NN>_<variant>_<clip-or-split>_<scope>` for run directory names. Date is the operator's local date (here Asia/Ho_Chi_Minh); NN is a monotonically increasing daily sequence. Current evidence runs use `20261022-01_baseline-audit_tutorial_eval-smoke` and `20261022-02_baseline-audit_tutorial_homography-smoke`. A seed, if actually set, belongs in config and may be included in scope; never invent `seed0` for an unseeded run.

Keep runs append-only. Store command/config, source/notebook hash or revision, environment/device versions, all seeds or explicitly `unset`, data split and frame support, input/model/tracker config hashes, changed component, runtime/failures, prediction and evaluator hashes, metrics, logs and artifact index. Every result needs a scope label such as full-clip, partial inference, synthetic-check, or loader-negative-control. An audit PASS and a baseline PASS are different statuses. Failed runs remain evidence. The current audit wrapper implements only the information available locally, not a complete inference experiment manager.

Compare paired variants using identical frozen inputs and evaluator settings. Record which observations were dropped by roster/filter requirements and which final rows were interpolated. Report row counts and frame coverage alongside metrics. Keep extra diagnostics separate from the frozen notebook metric definition. Run artifacts must state whether evaluation used in-memory floats or saved CSV values. Keep prediction/provenance metadata in sidecar files so the required five-column Challenge submission remains valid.

## Existing boundaries and smallest interfaces needed

**Alternative tracker / association.** `mot_results.txt` is already a useful pixel-space interface `(frame_id,tracker_id,x,y,w,h,conf)` with positive video frame indices. Alternative trackers can feed the same conversion/projection/filter stage if they obey coordinate and ID contracts. To change association alone, cache pre-tracker detections with frame/class/box/confidence and freeze them; current tracked labels cannot guarantee the same detector candidates. Preserve exact tracker YAML, reset state per video, and document ID mapping. No alternative tracker was installed.

**Alternative team classifier.** Keep sampled torso crops/features fixed and inject a per-track `team_id` map after cluster alignment, or inject features/clusters before the same alignment policy. These answer different questions: label changes should not silently replace initial-offense anchoring. Retain OCR votes and downstream roster filtering. Current map is a notebook global inside cell 17, not a callable stage.

**Oracle jersey.** Inject `(track_id,jersey_number)` at `track_to_jersey` with GT correspondence provenance, preserve actual team assignment and downstream operations. Challenge GT has only court position and semantic labels, no boxes or temporary track IDs. Mapping GT to track evidence requires spatial association plus review; exact jersey labels cannot be assumed available per track. Tracks containing switches need frame-level provenance and an explicit oracle policy.

**Oracle identity.** Inject reviewed semantic labels before deduplication/interpolation, with per-frame/track correspondence if necessary; preserve localization/row support. This is distinct from replacing final predictions with GT. Record unknown/mixed tracks, not just successfully annotated ones.

**Oracle localization.** Final court positions provide an existing replaceable representation. Independently establish row correspondence, then replace matched positions while holding semantic labels and row support fixed; keep unmatched predictions/GT policy explicit. Injecting earlier before court filtering can change retained tracks, team anchors and semantic evidence, so it would test a broader intervention.

**Selective VLM assistance (future only).** The natural boundary is a sampled crop/evidence record with an abstention/uncertainty flag; a future helper could contribute candidates to jersey/team evidence while keeping frozen inputs and deterministic downstream steps. No LLM/VLM call, crop router, new component, or prompt was introduced in this sprint.

## Minimal persistence additions proposed

1. `homography.json`: point names, image coordinates, cm correspondences, H/inverse, dimensions and calibration hash. The audit wrapper already saves computed H/points/residuals for the calibration component; original baseline still only prints H.
2. `filter_decisions.csv`: tracker ID, counts, longest in-court run, c1/c2, each gate and keep/drop reason; per-row projected positions before rejection. This distinguishes non-player rejection from a later recognition failure.
3. `sample_evidence.jsonl`: frame ID, tracker ID, crop coordinates/reference, OCR raw text/box/confidence, acceptance/rejection reason, accepted jersey, and feature/sample provenance. Save selected crops on demand rather than all full-resolution frames.
4. `track_identity.csv`: temporary ID, accepted vote vector, winning jersey, team cluster, team mapping/anchor frame, roster validity and rejection reason. Vote margin/sample count is evidence, not probability.
5. `prediction_provenance.csv`: final row key, contributing tracker ID(s), observed/interpolated flag, neighboring observations/gap length and duplicate-selection reason. Preserve internal bboxes separately for exact overlay replay.
6. `matches.jsonl` + compact segment index: explicit original frame IDs, spatial and identity-gated matching results, court distance, unmatched GT/pred, linked semantic/temporary IDs. Store per-alpha aggregate arrays; pick and label one alpha for temporal visual review.

Items 2–6 are **not implemented**. There is no proposed large logging framework or speculative refactor: small exports at the existing cell boundaries suffice initially.

## Temporal review questions and available evidence

- **Where did a tracker ID switch?** Raw tracking video/MOT can be reviewed manually. Current TI-HOTA cannot report switches. Need reviewed spatial GT correspondence and linked temporary-ID timeline; a semantic identity change and a tracker ID change must remain distinct.
- **Where did a track fragment?** Frame gaps and ID starts/ends can be derived from raw MOT. A missing row in one tracker ID is not automatically a confirmed same-player fragment; cross-ID correspondence is required. Preserve pre-interpolation support to inspect what final interpolation hides.
- **Which jersey reads were uncertain?** Console summaries expose competing accepted total scores but not frame uncertainty, rejected reads, crops or calibrated confidence. Save sample evidence and winning margin without changing selection.
- **Where did localization deteriorate?** Existing cell 24 gives only a matched-pair median. Save frame/pair distance and unmatched cases, then inspect the video, calibration and bbox-foot proxy together. Four calibration points alone do not validate player localization.
- **When did a referee/non-player enter?** Person detection includes all persons; court heuristics and roster compatibility may reject some. No explicit referee label is predicted. Keep rejection decisions/crops and review false positives instead of claiming classifier coverage.

## Known limitations to preserve in comparisons

Unpinned downloads/defaults and unset RNG seeds; Colab/CUDA/path assumptions; historical outputs with inconsistent run paths; heuristic label-order parsing; fixed camera and four calibration points; bottom-center foot proxy; court rules tied to this Outdoor geometry and 10 frames; fixed six-player roster and clip-start anchors; per-track fixed OCR/team labels; hard failure if any identity is missing; no gap-length cap or interpolation flags; no independent pre-tracker detections; lost tracker provenance; no event-level errors; compressed minimap timeline on frames with no rows; CSV precision versus in-memory evaluation. None was silently repaired.
