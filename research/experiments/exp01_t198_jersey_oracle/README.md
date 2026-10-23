# Exp01 — T198 jersey oracle

## Status

**CONFIRMED for structural B6 recovery.** All five preregistered structural hypotheses pass. The normal downstream source adds exactly 188 B6 rows and reaches frame 1118. The score comparison has a documented control limitation: the unchanged global confidence sort changes some equal-confidence winners for A12. Therefore the aggregate metric delta must not be described as a perfectly isolated B6-only coordinate effect.

The sole intervention was `track_to_jersey[198] = 6`, replacing None. T198's Team B mapping, all other jersey/team mappings, raw/filtered tracks, cached homography and court coordinates, roster validation, confidence sorting/deduplication, interpolation and evaluator code were unchanged. No model inference, OCR, clustering, oracle on another track, extrapolation or manual insertion of final rows occurred. Nothing under `results/` or the original notebook was written.

## Preregistration and frozen state

[preregistration.json](preregistration.json) was saved before the replay/oracle execution and hashes the key inputs. H1–H5 predicted: T198→B6; B6 last frame 1118; 175 retained T198 observations; 13 internally interpolated tail rows; total 5778→5966 (+188). It also preregistered the known tie-policy limitation rather than changing the algorithm after observing the result.

[t198_preconditions.json](t198_preconditions.json) verifies all required assumptions: filtered span 939–1118, 175 observations, court-retained, inferred Team B, baseline jersey=None, no logged semantic identity. There was no need to abort. [cached_mapping_state.json](cached_mapping_state.json) preserves all 23 frozen track mappings; [intervention.json](intervention.json) records that only jersey key 198 changed. The original accepted OCR results are unchanged input evidence; the value 6 is explicitly an oracle override, not a fabricated OCR score.

The pre-semantic DataFrame was reconstructed from the raw MOT's original retained row order/index, the cached filtered support, and cached court positions. No court filtering or homography estimation was rerun. Original pre-semantic columns are restored, including bbox foot centers. The exact executed notebook suffix starting with `identity_df = df.copy()` is run verbatim in separate `baseline_replay/` and `exp1/` output directories. Their `out_dir` values affect only output destinations; `video_path` is used only to obtain the existing clip identifier. Inverse-homography bbox reconstruction in the original suffix is retained.

## Baseline replay

[baseline_replay_verification.json](baseline_replay_verification.json) establishes:

- 5,778 final rows, identical row keys to frozen `results/predictions.csv`.
- B6: frame 1–930, 930 rows; **every B6 CSV coordinate exactly matches frozen baseline**.
- B11: frame 743–1118, 376 rows; **every B11 CSV coordinate exactly matches frozen baseline**.
- All six identities have the same coverage as frozen baseline.
- Nine other common rows differ in coordinates (1 A10, 8 A12), consistent with the previously identified unspecified equal-confidence ordering. Maximum single-coordinate absolute difference is 238.60 cm; these are selection differences, not CSV-rounding noise.

[replay_vs_frozen_diff.csv](replay_vs_frozen_diff.csv) preserves those nine differences. Unlike the prior causal audit's provenance reconstruction, this experiment does **not** force tie winners to match the saved prediction coordinates. It executes the original sort as-is. B6/B11 replay is exact at five-decimal CSV precision and therefore passes the preregistered relevance gate before applying the oracle.

Fresh frozen-CSV evaluation is TI-HOTA **0.6351420047**, rounding to the supplied notebook's 0.6351. The unmodified local downstream replay scores **0.6357098225**, with the same B6/B11 data but different non-target tie winners. Both values are preserved, rather than silently substituting local replay for the frozen benchmark.

## Structural result

[hypothesis_results.json](hypothesis_results.json): **H1 PASS, H2 PASS, H3 PASS, H4 PASS, H5 PASS**.

- T198's normal semantic mapping becomes **B6**; all **175** observed rows survive roster compatibility/deduplication.
- B6 extends from **1–930 / 930 rows** to **1–1118 / 1,118 rows**.
- Frames **931–1118** gain exactly **188** final rows: **175 observed** and **13 interpolated**.
- The 13 interpolated tail frames are **931–938, 957–958, 992, 994–995**. The first eight were previously beyond B6's last surviving observation; the oracle makes them internal gaps, so the existing interpolation fills them without extrapolation.
- Total prediction rows: **5,778 → 5,966**, exactly **+188**. The remaining 742-row support deficit is still B11's original prefix.
- B11 stays at **743–1118 / 376 rows**. B6's original 930 rows are unchanged.
- No observed or final B6 semantic keys collide. No manually inserted prediction rows were used.

See [coverage_before_after.json](coverage_before_after.json), [predictions_baseline_replay.csv](predictions_baseline_replay.csv), [predictions_exp1.csv](predictions_exp1.csv) and the original-source logs/selected observations under both arm directories.

## Metric comparison

The four-column metric comparison is available in the [interactive report](/Users/thuong/.cursor/projects/empty-window/canvases/trackid3x3-exp01-t198-jersey-oracle.canvas.tsx) and [metric_comparison.csv](metric_comparison.csv). The interactive report can switch the comparator between the frozen supplied CSV and local replay. These are actual evaluations on all 1,118 frames using the original evaluator, tau=100 cm, 19 alpha thresholds.

Primary comparison, **frozen baseline → Exp1 (Exp1 − frozen)**:

- TI-HOTA: **0.635142 → 0.652881 (+0.017739)**.
- TI-DetA: **0.636182 → 0.653983 (+0.017801)**.
- TI-AssA: **0.634104 → 0.651782 (+0.017678)**.
- TI-TP: **4801.42 → 4953.42 (+152.00)**.
- TI-FP: **976.58 → 1012.58 (+36.00)**.
- TI-FN: **1906.58 → 1754.58 (−152.00)**.

TP/FP/FN are threshold-averaged counts, not integer event totals. Added coverage also adds false-positive contributions at localization thresholds; it does not convert every new row into a true positive.

Paired local replay → Exp1: TI-HOTA **0.635710 → 0.652881 (+0.017171)**, TI-DetA **+0.017211**, TI-AssA **+0.017133**, TP **+149.315789**, FP **+38.684211**, FN **−149.315789**. This removes the frozen-versus-local replay mismatch as the comparator difference, but still includes the intervention's indirect tie-order interaction.

All three arms are evaluated from their **five-decimal CSV values**, holding input precision consistent. The original notebook evaluated its in-memory table; the evaluator code/semantics are unchanged. Per-alpha TI-HOTA arrays and full-precision aggregate values are retained in [baseline_metrics.json](baseline_metrics.json), [baseline_replay_metrics.json](baseline_replay_metrics.json), [exp1_metrics.json](exp1_metrics.json) and [metric_delta.json](metric_delta.json).

Original location-only diagnostics, frozen → Exp1:

- Matched pairs: **5649 → 5818**.
- Median matched court-position error: **13.978796 → 14.114677 cm**.
- Conditional team accuracy: **0.923880 → 0.928154**.
- Conditional jersey accuracy: **0.923349 → 0.927638**.

These diagnostics use position-only Hungarian matching within 100 cm. Their matched populations change, so the conditional medians/accuracies are not a paired error measurement on an identical set of objects.

## Recovered B6 quality

All **188** recovered rows have same-identity B6 GT correspondence. This diagnostic compares their coordinates with B6 GT directly; it does not relabel other players by nearest position.

[recovered_B6_quality.json](recovered_B6_quality.json):

- Median court-position error: **19.912196 cm**.
- **172/188** rows within 100 cm; **16** above 100 cm, **9** above 200 cm, **3** above 300 cm.
- Observed T198: **175** rows, median **19.145271 cm**, **163** within 100 cm.
- Interpolated: **13** rows, median **29.405310 cm**, **9** within 100 cm.
- Maximum error: **373.051736 cm at frame 1053**; frame 1052 is 322.65 cm and 1054 is 313.38 cm.
- Other large errors cluster around **990–999**, including observed frame 996 at **289.66 cm** and interpolated frame 995 at **269.07 cm**.

Some unchanged cached T198 points have negative court_y, e.g. −144.34 cm at 1053, while B6 GT is inside court. These positions had survived the original 300-cm court buffer. The oracle restores their semantic label but does not repair their localization. Interpolation also inherits errors from neighboring bad observations. No localization or filtering change was made.

See [recovered_B6_rows.csv](recovered_B6_rows.csv), [largest_recovered_B6_errors.csv](largest_recovered_B6_errors.csv) and [error plot](recovered_B6_errors.png). There are **zero duplicate final semantic keys** and zero new competing B6 observations at a common frame; T198 starts after T65's last frame.

## Invariants and direct interactions

**Exactly unchanged relative to the frozen supplied CSV:**

- A4: all 1,118 rows and coordinates.
- B9: all 1,118 rows and coordinates.
- B11: all 376 rows and coordinates.
- B6's original frames 1–930: all 930 rows and coordinates.
- Every non-B6 row key and support interval; all other track team/jersey mappings and cached observations.

**Not exactly unchanged relative to frozen CSV:** A10 has 1 coordinate row difference; A12 has 9. Their row counts/keys are unchanged. The initial local replay already differs at 1 A10 + 8 A12 rows; comparison sets overlap and cancel differently, so differences must not be added arithmetically.

**Relative to the local replay in the same environment:** A4, A10, B9 and B11 are exactly unchanged, while **13 A12 rows** change coordinates. Each is linked directly to a different winner among **equal-confidence** T5/T74/T84 observations under the original global sort/deduplication. The added 175 T198 semantic rows change the global array length/order used by the unspecified tie sort. There is no new A12 mapping, no parameter change and no lower-confidence winner promoted.

The affected A12 frames versus local replay are **425,497,522,553,570,571,586,587,606,608,610,624,711**. For example, at frame 497 T5 and T84 both have confidence 0.76; selection changes from T5 at (668.14,912.95) cm to T84 at (472.79,489.19) cm, a displacement of ~466.62 cm. This is an actual downstream interaction, not rounding noise.

The strict “only B6 final coordinates change” invariant therefore **fails**, and is explicitly reported as permitted by the task's direct-deduplication-interaction exception. No tie-policy fix or output patch was applied to make it pass. The structural B6 sufficiency result is unambiguous; the aggregate score increase includes these A12 effects and must not be attributed exclusively to adding B6 points.

Evidence: [invariants.json](invariants.json), [csv_diff_vs_frozen.csv](csv_diff_vs_frozen.csv), [csv_diff_vs_replay.csv](csv_diff_vs_replay.csv), their summary JSONs, [dedup_winner_changes.csv](dedup_winner_changes.csv), [non_B6_interactions.csv](non_B6_interactions.csv).

## Interpretation

**Confirmed:** given the existing Team B mapping and cached geometry for this known fragment, supplying jersey 6 at the original jersey-evidence→semantic boundary is sufficient for the unchanged downstream pipeline to recover T198 as B6 through frame 1118. It naturally fills all 188 previously absent tail rows using the predicted 175-observed/13-interpolated split.

**Not established:** an actual OCR improvement exists, real OCR can reliably recover this jersey, all T198 localization is accurate, other missing identities would recover similarly, or the measured TI-HOTA gain is a perfectly pure B6-only effect. This is one fragment-level oracle intervention, not a deployable fix or a leaderboard optimization.

## Next decision

**Semantic/jersey recovery deserves further work:** the blocked accepted-jersey gate had measurable coverage value on a retained fragment, and the original downstream pipeline already knows how to use that evidence. Keep localization outliers and the global-sort tie interaction explicit in later comparisons. No Exp2, training, OCR change, extrapolation, production change or new association was implemented.

## Reproduction, environment and artifacts

Run exactly once from the repository root:

```sh
/tmp/trackid3x3-audit-venv/bin/python research/experiments/exp01_t198_jersey_oracle/run.py
/tmp/trackid3x3-audit-venv/bin/python research/experiments/exp01_t198_jersey_oracle/render_report.py
```

`run.py` refuses an existing experiment manifest rather than overwriting a completed attempt. To independently reproduce, copy the experiment script and unchanged preregistration to a new directory at the same repository depth and use the same frozen inputs/environment. Do not rerun into `results/`. Source-relative paths are documented in the script; no missing model/data download is required.

Environment: macOS ARM, Python 3.13.6, NumPy 2.4.1, pandas 2.3.3, SciPy 1.17.0, OpenCV headless 5.0.0.93. This was a **CPU cached downstream experiment**, not new Kaggle/Colab model inference. No new packages were installed. Stage/evaluation runtime is in the manifest/metric JSONs; total execution was about 9.06 s in this environment. No RNG seeds or model settings changed.

Primary artifacts:

- [experiment_manifest.json](experiment_manifest.json): CONFIRMED status, hypotheses, preserved-input checks, score deltas and tie caveat.
- [source_hashes.json](source_hashes.json): source/input/script/preregistration hashes; all `results/` inputs and original notebook verified unchanged after execution.
- [environment.json](environment.json), [run.log](run.log), [run.py](run.py), [render_report.py](render_report.py).
- `source/downstream_original.py`, `source/evaluator_original.py`, `source/location_diagnostic_original.py`: original source sections used without semantic modifications.
- `baseline_replay/` and `exp1/`: original-source `predictions.csv`, downstream stdout and selected semantic observations for each arm.
- Root `predictions_baseline_replay.csv`, **`predictions_exp1.csv`**, metric JSONs/deltas/comparison CSV, coverage JSON, recovered-row diagnostics, CSV diffs and the files linked above.
- `verification.json`, `artifact_hashes.json`: final consistency checks and generated-artifact inventory.

The earlier causal analysis at `research/analysis/results_b11_b6_20261023/` remains unchanged evidence of the pre-experiment diagnosis. The preregistration and all raw result files are preserved.
