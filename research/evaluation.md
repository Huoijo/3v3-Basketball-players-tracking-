# Evaluation audit

Evidence: unchanged notebook cell 21 (572 lines), cell 23 (68), cell 24 (136). Snapshots are under each run's `source/`. This audit describes **the supplied notebook evaluator**. Its equivalence to a separate challenge server or upstream implementation has not been verified; no external evaluator source was present locally.

## Actual matching and aggregation

`loc_sim(p,g,tau) = exp(log(0.05) * ||p-g||² / tau²)`. Cell 23 uses tau=100 cm. Similarity at a distance of 100 cm is 0.05. `id_sim` is 1 only when concatenated team+jersey strings match exactly. Combined similarity is their product.

For each alpha in `np.arange(0.05,1.0,0.05)` (19 thresholds), cell 21 forms a per-frame GT×prediction matrix, gates combined similarities below alpha by cost 1e6, then Hungarian-matches cost `1-combined`. Accepted matches are TP; remaining predictions FP and GT FN. A wrong team or jersey at a perfect position contributes an FP and an FN. Localization threshold changes with alpha; 100 cm is not a uniform hard gate across all thresholds. At alpha 0.95 the corresponding distance is approximately 13.1 cm; at 0.05 it is 100 cm, subject to floating-point threshold boundaries.

For each GT semantic identity it accumulates TPA (matched GT rows), FNA (unmatched GT rows), and FPA (unmatched predictions carrying that semantic label). Association accuracy is the **unweighted mean over GT identities** of `TPA/(TPA+FNA+FPA)`. Prediction-only identities are counted in detection FP but are not added as identities to this association average. This is the exact notebook definition, not a claim about conventional track-pair HOTA association.

Per alpha, DetA=`TP/(TP+FP+FN)`, AssA as above, TI-HOTA=`sqrt(DetA*AssA)`. The wrapper returns the mean of each metric across 19 alphas. In particular the reported TI-HOTA is mean of per-alpha square roots, not necessarily sqrt(mean DetA × mean AssA). TP/FP/FN outputs are threshold-averaged counts and can be fractional.

## What the evaluator exposes

- `TI-HOTA`, `TI-DetA`, `TI-AssA`, `TI-TP`, `TI-FP`, `TI-FN`: returned and printed by cell 23.
- `TI-HOTA_alphas`: list of 19 values, returned but not printed by the original cell. Alpha array returned separately.
- `compute_ti_hota_alpha` directly returns the six values for a chosen alpha. `compute_ti_hota` does not preserve the per-alpha DetA/AssA/TP/FP/FN arrays in its result.
- No LocA, image-box IoU metric, conventional identity-switch count, fragmentation count, IDF1, MOTA, per-player score, confusion matrix, or frame-indexed match log is exposed.
- Temporary tracker IDs are entirely absent. Switching tracker IDs while preserving semantic labels/positions does not affect this evaluator. Synthetic contract evidence confirms this; it cannot verify temporal continuity of raw tracker trajectories.
- `TI-DetA` and `TI-AssA` are available but both depend on identity-gated position matching. They do **not** independently isolate detector quality and tracker association quality.
- Wrong semantic identity causes observable score loss, but aggregate FP/FN cannot distinguish identity error from missing detection or localization failure.

Cell 24 adds an independent **position-only** diagnostic: Hungarian matching by court distance, rejecting matches beyond 100 cm. It exposes matched-pair count, median position error in cm, team accuracy, jersey accuracy. Team and jersey are scored separately; neither is joint semantic-identity accuracy. All three quality values are conditional on accepted spatial pairs. Unmatched rows are omitted; therefore these accuracies are not whole-dataset identity accuracy or detection recall. Matching at crowded/nearby player positions may not represent a reviewed player correspondence.

Frame IDs are used to assemble the union in `load_video`, then lost from the list of frame objects. Track IDs and pixel boxes are also absent. These omissions prevent event attribution without additional provenance. GT loader enforces exact ordered columns; prediction loader itself does not validate complete schema, duplicates, nonfinite values or units. The audit wrapper adds an exact CSV column check for its optional CSV evaluation, without changing notebook matching.

## Evidence obtained this sprint

Run 02's `synthetic_evaluator_checks.json` verifies perfect matching, wrong identity, missing predictions, extra predictions, empty frames, and tracker-ID invisibility. The extra-prediction fixture deliberately uses a prediction-only B6 label; DetA=0.5, AssA=1, TI-HOTA=√0.5 demonstrates the association denominator behavior. No alternate evaluator or research model is implemented.

Real GT loading verified 6,708 rows, frames 1–1,118, six players. The three-frame negative control deliberately has no model predictions, producing 18 FN and zero TP/FP. This verifies the loader and unmatched-GT counting only. No independently reproduced baseline score exists.

Historical saved output (cell 23): TI-HOTA=0.6351, TI-DetA=0.6362, TI-AssA=0.6341, TI-TP=4801.42, TI-FP=976.58, TI-FN=1906.58. Saved cell 17 has 5,778 prediction rows. Count consistency is plausible: TP+FN=6,708 and TP+FP=5,778 within printed rounding. This does not establish execution provenance.

Historical saved cell 24: matched_pairs=5,649, median_position_error_cm=13.98, team_accuracy=0.9239, jersey_accuracy=0.9233. These are unverified saved results on matched rows, not rerun measurements.

## Smallest diagnostic additions for later work

Persist all per-alpha metrics and semantic-identity counts. Keep explicit frame IDs through the evaluator. At a named fixed alpha save GT/pred row IDs, match distance, identity equality, matched/unmatched status; separately save position-only matches and team/jersey errors. Join back to original tracker IDs and interpolation flags. Derive tracker-switch/fragment diagnostics only from a documented GT-to-track correspondence, rather than relabeling TI-AssA as a switch metric. Freeze the original evaluator and version any additional diagnostics separately. These additions are proposals, not new metrics claimed as currently implemented.
