# B11/B6 causal failure analysis — supplied Colab results, 2026-10-23

## Status

**PARTIAL.** The output coverage, stage losses, retained-track recognition failures and interpolation boundaries are established directly. Video corroborates mixed physical identities inside Track 5 and a B6 fragment returning as Track 198. The remaining limits are specific: no pre-tracker detections, raw rejected OCR evidence, or exhaustive per-frame physical identity annotations exist. Consequently an occlusion gap cannot be attributed uniquely to YOLO versus ByteTrack, and “no valid jersey” cannot be decomposed into empty OCR, low confidence or roster rejection.

No baseline code, model, thresholds, input results, tracks or predictions were changed. No oracle experiment or model inference was run. Diagnostic GT correspondence is used only to analyze existing observations. All files under `results/` were SHA-256 checked before and after analysis and remained unchanged; see [analysis_manifest.json](analysis_manifest.json).

## Evidence categories and scope

- **FACT:** code, notebook stdout, artifact rows and explicitly defined numerical diagnostics. A nearest-GT distance is a numerical fact; it is not an identity annotation of the box.
- **MANUAL VISUAL EVIDENCE:** jersey/color, occlusion and raw-box observations in specifically referenced image frames. These are sampled inspections, not a relabeling of the full video.
- **HYPOTHESIS:** interpretation of an unrecorded intermediate or an unverified physical correspondence.
- **CONCLUSION:** a causal claim supported by the available code, artifacts and sampled visual evidence, with its limits stated.

Actual supplied files: five files in `results/`: predictions, raw MOT, filtered MOT, transformed court rows and `IMG_0104_1_with_minimap.mp4`. `tracked_visualization.mp4`, original video, raw YOLO label files and an executed notebook were **not** in that directory. The user-linked [Colab notebook](https://colab.research.google.com/drive/18ixlNOx_eRyOF2yaSVd_X3i9iUJL_igE) was fetched read-only via Google Drive into [source/executed_colab.ipynb](source/executed_colab.ipynb). It contains 33 cells and actual execution counts, with consistent `/content/runs/detect/track` outputs. All 15 original baseline code cells are exact-source-equal after accounting for the extra Drive-mount/export cells. This parity is recorded in the manifest.

The existing audited GT and Outdoor calibration were used as additional diagnostic references. Their paths/hashes are recorded; no different game/data was substituted. The supplied final video is 3840×2160, 1,118 frames, reported FPS 29.97. Frames below are one-based; timestamps are `(frame-1)/29.97`, approximately. Analysis copies overlay **raw MOT** boxes/IDs on the supplied final video. Its existing colored labels/ellipses are baseline predictions, not raw tracker labels or independent truth; overlapping text must not be read as a box's identity.

## Final output coverage

**FACT — [final_coverage.csv](final_coverage.csv), read directly from `results/predictions.csv`:**

- A4: first 1, last 1118, 1,118 final rows, 0 missing; 995 selected observed rows + 123 interpolated.
- A10: first 1, last 1118, 1,118 final rows, 0 missing; 1,084 observed + 34 interpolated.
- A12: first 1, last 1118, 1,118 final rows, 0 missing; 1,082 observed + 36 interpolated.
- B9: first 1, last 1118, 1,118 final rows, 0 missing; 1,057 observed + 61 interpolated.
- **B11: first 743, last 1118, 376 final rows, 742 missing (1–742); 332 observed + 44 interpolated.**
- **B6: first 1, last 930, 930 final rows, 188 missing (931–1118); 902 observed + 28 interpolated.**

Total: 5,778 unique `(frame, semantic identity)` rows. Expected full six-player support is 6,708; deficit is exactly **742 + 188 = 930**. This is a support count, not a detector-FN count or TI-HOTA FN total. [Coverage and tracker-span figure](coverage_and_tracker_spans.png) shows the exact final bounds and relevant tracker spans.

## Tracker lifecycle summary

The complete [tracker_lifecycle.csv](tracker_lifecycle.csv) has **all 37 IDs appearing in raw MOT**, including the 14 court-rejected IDs; it is not a claim that ByteTrack internally created only 37 IDs. The table records spans, observed counts, longest runs, first/last/median court coordinates in cm, gate decisions/reasons, inferred team, accepted OCR sums, scheduled sample counts, final semantic mapping and deduplication fate. Final CSV does not contain tracker IDs; contribution is reconstructed from the filtered tables, stdout and exported coordinates.

**FACT — important logged mappings and fates:**

- T7: 1–300, 290 observations, B / jersey=None; all 290 removed before semantic identity creation.
- T59: 301–399, 97 observations, B / jersey=None; all 97 removed.
- T5: 1–741, 738 observations, **A12**; 585 selected, 153 semantic duplicates removed. Mixed physical content is visually confirmed below.
- T74: 408–451, 43 observations, **A12**; 23 selected, 20 duplicates removed.
- T84: 474–840, 361 observations, **A12**; 226 selected, 135 duplicates removed. Its late boxes can overlap the green player during crossings; do not assume every row is physical A12.
- T157: 743–808, 66 observations, **B11**, accepted jersey-11 sum **0.38**; all 66 selected.
- T172: 820–857, 29 observations, B / jersey=None; all 29 removed. It starts after the missing B11 prefix.
- T179: 853–1118, 266 observations, **B11**, accepted jersey-11 sum **0.59**; all 266 selected.
- T8: 1–147, 147 observations, **B6**, score 6:23.86; all selected.
- T40: 160–305, 146 observations, **B6**, score 6:21.94; all selected.
- T65: 308–930, 609 observations, **B6**, score 6:15.43; all selected.
- T198: 939–1118, 175 observations, B / jersey=None; all 175 removed.
- T47: 217–318, 80 observations, inferred **A**, jersey **6**, accepted sum 3.52; all 80 fail roster compatibility. This is not a late B6 fragment; its endpoint is far before frame 930. Geometry often places it near A4, with some crossing ambiguity; the jersey candidate alone does not establish its physical identity.

Other surviving mappings are A4={T4,T57,T139,T228}, A10={T2,T138}, A12 additionally T187, and B9={T3,T18,T145}. All 23 filtered IDs and all other raw IDs were inspected numerically. No “unassigned B track = B11/B6” assumption was used.

## B11 diagnosis

### FACT: the first missing stage is not uniform across the prefix

At the beginning of the clip, T7 already exists in raw MOT, passes both court gates, and is assigned Team B by the logged color-cluster mapping. Its accepted OCR score dictionary is empty, jersey becomes None, and the semantic-validation `jersey.notna() & team.notna()` gate drops it. T59 has the same fate. These are **post-tracking, post-court, post-team jersey-evidence failures**, not court rejection. See [recognition stdout](source/recognition_stdout.txt), [executed cell 18](source/cell_18.py) and [filtered observation fates](filtered_observation_fates.csv).

This does not mean T7 is physically pure throughout 1–300. At frame 300 its box overlaps purple players and its endpoint is nearest to another semantic track; the manual evidence supports an early B11-related track with terminal contamination, not a blanket physical identity annotation for all 290 rows.

T59 ends at 399. Around 408–417, the physical B11/A12 crossing becomes an **association/identity-contamination failure**. T74 and T5 both have a fixed aggregate A12 identity. Geometry favors T74 for B11 at frames 408–415; from 416 the nearest track changes to T5. A sampled image pair brackets the physical crossover between 414 and 417, and by frame 430 the mixed-track failure is unmistakable. This is not a formally measured ID-switch timestamp; the exact online transition within the crossing is not logged.

T5's global jersey vote is `{12:5.92}`, cluster→Team A, semantic A12. It originally contains a purple-12 player, but later contains green-11 observations. The pipeline assigns one team and jersey to the entire tracker ID and never splits it by physical content. Some resulting rows survive as A12; others lose confidence-based competition with T74/T84 for the same `(frame,A12)` key. **Neither outcome creates B11.**

### MANUAL VISUAL EVIDENCE

- [Frame 1 / T5 crop](visual/crop_f0001_t5.jpg): T5 contains the purple player with visible number 12.
- [Frame 280 / T7 crop](visual/crop_f0280_t7.jpg): the box contains the green player in the B11 trajectory while crossing near purple 12; jersey detail is partly occluded.
- [Frame 301 / T59 crop](visual/crop_f0301_t59.jpg) and [frame 399](visual/frame_0399.jpg): a green player is retained as T59 in the B11 region, with nearby players occluding parts of its body.
- [Frame 408](visual/frame_0408.jpg), [414](visual/frame_0414.jpg), [417](visual/frame_0417.jpg): T5/T74 boxes overlap the purple-12/green-11 crossing. Sampled images and geometry jointly support identity contamination; nearest position alone is ambiguous.
- **[Frame 430 / T5 crop](visual/crop_f0430_t5.jpg): green jersey 11 is legible inside T5.** [T74 at the same frame](visual/crop_f0430_t74.jpg) contains purple jersey 12. Both tracks nevertheless carry logged A12 identities.
- **[Frame 700](visual/frame_0700.jpg): T5 still boxes green 11 while its output semantic label remains A12; T84 boxes the purple-12 player.** This directly demonstrates detected/tracked B11 information being converted into the wrong semantic identity.
- [Frame 741 / T5 crop](visual/crop_f0741_t5.jpg): the box overlaps multiple bodies; its projected foot position is far from B11 GT. Good localization/association deteriorates near the eventual T157 start.
- [Frame 743 / T157 crop](visual/crop_f0743_t157.jpg): the new tracked box contains the green-11 player.

### FACT: recovery starts with a new jersey-11 track

T157 begins at 743, is assigned Team B and jersey 11, and supplies the **first surviving B11 semantic observation**. T179 later supplies B11 from 853 to 1118. There are no logged B11 semantic tracks before 743. The accepted OCR sum 0.38 is a whole-track aggregate; it does **not** establish that OCR read 11 specifically at frame 743. Labels are applied retrospectively to every row in the track.

### CONCLUSION

B11's 742-row prefix is caused by **unavailable semantic evidence on early retained Team-B fragments, followed by mixed/incorrect tracker association whose whole-track A12 assignment hides later physical B11 observations**. The output cannot recover a B11 identity from those rows, and interpolation does not extend before its first surviving semantic observation. The earliest demonstrated semantic loss is already T7 at frame 1; for the long later segment, association contamination is an earlier upstream failure than the resulting wrong team/jersey assignment.

This is not explained by continuous detector failure: B11 is plainly present in retained boxes at representative early and mid-clip frames. Some brief crossing/localization gaps remain, and the raw detector-versus-tracker contribution to them is unavailable.

## B6 diagnosis

### FACT: distinguish the termination gap from the recovered fragment

The only logged B6 semantic tracks are T8, T40 and T65. Fragmentation by itself is already handled for these three: their observations are semantically combined into B6 and internal gaps are interpolated. T65's final observation is frame 930; all 609 of its rows survive semantic deduplication.

At frames **931–938**, there is no distinct raw-MOT foot-point observation within 100 cm of GT B6. The closest raw projected boxes belong to T139, labeled A4. This is the first **observable tracker-output support gap** at the tail boundary. It does not prove that YOLO emitted no detection: pre-tracker/rejected detections are absent and a box can include two overlapping bodies.

At frame **939**, T198 appears and continues to 1118 with 175 observations. All 175 survive court filtering and receive Team B. Its OCR accepted-evidence dictionary is empty, jersey=None, so **all 175 are removed at semantic validation**. No later track is assigned B6. Thus recovery of a raw track does not recover final B6 support.

T198 is missing five internal raw frames (957,958,992,994,995); this does not explain the entire 188-frame tail. For the 180-frame 939–1118 interval, 175 observations exist and are dropped for missing jersey, while five observations are absent. The preceding 931–938 gap is eight frames. These are exact observation-support facts, not an exhaustive physical detection attribution.

### MANUAL VISUAL EVIDENCE

- [Frame 890](visual/frame_0890.jpg): green player remains in the scene under T65.
- **[Frame 930 / T65 crop](visual/crop_f0930_t65.jpg): green jersey 6 is visible behind/overlapping purple 4.**
- [Frame 934](visual/frame_0934.jpg): B6's expected area is occluded by/overlaps the purple-4 player. A clean individual B6 body/number is not established from this frame; the image supports an occlusion explanation rather than proving a detector miss.
- **[Frame 939 / T198 crop](visual/crop_f0939_t198.jpg): the reappearing box contains a green jersey with readable 6**, near purple 4.
- [Frame 970](visual/frame_0970.jpg): T198 boxes the green player shooting near the basket, while the baseline prediction minimap still lacks B6.
- [Frame 1118 / T198 crop](visual/crop_f1118_t198.jpg): a green player remains tracked at the end. The numeral is not readable in this particular profile view; identity continuity here is supported by trajectory and earlier jersey evidence, not a claimed final-frame OCR read.

### CONCLUSION

The first observed loss at 931 is a short **tracking/localization support gap around occlusion**. The long tail persists because the recovered B6-related fragment T198 fails the **accepted-jersey → semantic identity gate**. Court filtering does not remove it; team labeling is B; it is not deduplicated away or mapped to B9/B11. With no surviving B6 observation after 930, the original no-extrapolation rule leaves the entire 931–1118 tail empty.

## Candidate unassigned Team-B tracks

All coordinates below are projected court **cm**. Source table: [tracker_lifecycle.csv](tracker_lifecycle.csv). Accepted-score sums are printed to two decimals in notebook stdout; individual accepted OCR reads, raw rejected values and actual per-frame OCR-call counts were not saved. “None” means **no accepted roster-compatible number after the original 0.10 confidence and roster-union gates**, not necessarily no characters recognized.

### T7

**FACT:** span 1–300; 290 observations; longest observed/in-court run 285. First (883.58,462.87), last (685.96,231.98), median (1063.14,474.25). Court accepted; Team B; jersey=None; empty accepted scores; 40 scheduled samples, actual calls/rejected values unavailable. All 290 rows fail jersey availability.

**MANUAL VISUAL EVIDENCE:** early samples contain the green B11-related player; frame 300 contains overlapping purple/green bodies and is not a clean B11 endpoint. **FACT:** there is no earlier semantic frame before its start at frame 1. The nearest selected semantic point in the next frame after its end is T40/B6, 50.33 cm away; this endpoint proximity is **not** evidence that T7 is B6. The next unassigned B11-related fragment is T59 from 301, supported by GT proximity and sample imagery, but T7's contaminated last row prevents an uncomplicated endpoint merge claim.

### T59

**FACT:** span 301–399; 97 observations; longest observed/in-court run 81. First (772.63,430.16), last (570.28,156.59), median (706.78,207.91). Court accepted; Team B; jersey=None; empty accepted scores; 40 scheduled samples, actual calls/rejected values unavailable. All 97 rows fail jersey availability. Every row's nearest GT label is B11; median nearest distance is 23.74 cm, a geometric diagnostic.

**MANUAL VISUAL EVIDENCE:** frames 301/399 show the green player among occluding purple players. **FACT:** nearest adjacent semantic-frame points before/after are T5/A12, 25.00 cm at frame 300 and 142.24 cm at frame 400. Because T5 itself has mixed identity content, neither automatically proves same-player continuity. Later B11-related observations appear under T74 and T5 at the crossing. **CONCLUSION:** strong evidence for an unassigned B11-related fragment; no accepted jersey survives, independent of its exact endpoint association.

### T172

**FACT:** span 820–857; 29 observations; longest observed/in-court run 17. First (638.64,134.76), last (702.28,246.04), median (666.71,194.38). Court accepted; Team B; jersey=None; empty accepted scores; 29 scheduled samples, actual calls/rejected values unavailable. All 29 rows fail jersey availability. Nearest-GT counts misleadingly favor A12 (19) over B11 (10), because the boxes overlap the two players.

**MANUAL VISUAL EVIDENCE:** [frame 820 crop](visual/crop_f0820_t172.jpg) and [frame 850 crop](visual/crop_f0850_t172.jpg) include green 11 with purple 12 immediately behind. At [853](visual/frame_0853.jpg), T172/T179 boxes overlap the same crossing area. **FACT:** nearest adjacent semantic-frame points are T84/A12 at frame 819 (110.24 cm) and T179/B11 at 858 (78.99 cm). This supports a B11-related crossing/fragment, but does not prove all 29 rows are pure B11.

**CONCLUSION:** T172 cannot cause missing frames 1–742 because it starts at 820. Its lost evidence lies inside B11's existing 743–1118 support and is hidden by interpolation; the 44-frame B11 gap 809–852 is filled from T157 to T179 even though T172 has real observations during part of it.

### T198

**FACT:** span 939–1118; 175 observations; longest observed run 123 and longest strict in-court run 64. First (546.01,339.56), last (1307.81,140.71), median (1301.56,204.51). Court accepted; Team B; jersey=None; empty accepted scores; 40 scheduled samples, actual calls/rejected values unavailable. All 175 rows fail jersey availability.

**MANUAL VISUAL EVIDENCE:** jersey 6 is visible in T198 at frame 939, as it was in T65 at 930. **FACT:** T65's terminal point to T198's initial point is 178.93 cm over 9 frame steps (~0.30 s). This is plausible motion with localization error, not by itself an identity proof; the visual jersey evidence makes the association substantially stronger. The adjacent frame 938's nearest semantic point is T139/A4, 163.28 cm from T198's initial point, showing why an automatic nearest-semantic handoff would be unsafe. No later semantic frame exists after T198's end at 1118.

**CONCLUSION:** strong corroborated B6-related tail fragment dropped for lack of accepted jersey. It is not discarded by court/roster checks or deduplication after receiving a valid semantic identity.

## Transition windows

Full compact per-track contiguous-range timeline: [transition_windows.csv](transition_windows.csv). It includes every filtered track present in either requested window, not only the candidate four. Full per-frame geometry traces are available separately for checking ambiguous crossings.

**FACT — B11 window, frames 700–780 (~23.32–25.99 s):**

```text
700–733: T5, team A, jersey 12 / sum 5.92, semantic A12.
         16 rows selected, 18 discarded as A12 duplicates; visual sample shows physical green 11.
734:     no T5 row.
735–736: T5 returns with poor/overlapping localization; both rows lose A12 deduplication.
737:     no T5 row.
738–741: T5, A12; 2 selected and 2 deduplicated, with distorted foot-point localization.
742:     no T5 row; T84/A12 is the nearest raw point to GT B11 (~93 cm),
         but one-to-one spatial matching leaves B11 unmatched: crossing ambiguity.
743–780: T157, team B, jersey 11 / whole-track sum 0.38, semantic B11; all 38 rows selected.
```

Other window tracks: T65/B6, T84/A12, T138/A10, T139/A4, T145/B9. T65 and T84 have their own short observation gaps; those do not create a recognized B11 prefix. At 743, T157 is ~13.83 cm from B11 GT. This is a new retained correctly labeled fragment, not evidence of a jersey OCR event occurring exactly at 743.

**FACT — B6 window, frames 890–970 (~29.66–32.33 s):**

```text
890–930: T65, team B, jersey 6 / sum 15.43, semantic B6; all 41 rows selected.
931–938: T65 ended; no separate raw projected point within 100 cm of GT B6.
939–956: T198, team B, accepted jersey scores {}, jersey=None; all 18 observations dropped.
957–958: no T198 observations.
959–970: T198, same semantic failure; all 12 observations dropped.
```

T138/A10, T139/A4, T145/B9, T179/B11 and T187/A12 also occupy this window. T139/A4 overlaps the B6 area around the gap but no evidence establishes a deliberate B6→A4 semantic remap; the late T198 rows are unassigned, not B9/B11 errors.

## Pipeline loss accounting

**FACT — global observations:**

```text
13,890 raw MOT rows / 37 exported tracker IDs
→ 6,489 after court filtering / 23 IDs                 (-7,401)
→ 6,489 team assigned / 23 IDs                            (-0)
→ 5,898 jersey assigned / 19 IDs                        (-591)
→ 5,818 roster-compatible / 18 IDs                       (-80)
→ 5,452 semantic observations after deduplication       (-366)
→ 5,778 final rows after within-identity interpolation  (+326)
```

The four empty-jersey tracks account exactly for 591 rows: T7=290, T59=97, T172=29, T198=175. Roster rejection is exactly T47's 80 rows. All observations of the 23 accepted track IDs pass the final 300-cm-buffer gate; court filtering replay reproduces the supplied retained `(frame,tracker_id)` keys exactly, with projected coordinate error below 1e-7 cm. The 7,401 removed raw rows belong to the 14 track-level-rejected IDs, rather than those four candidate tracks.

**FACT — conditional on the baseline's inferred Team B:**

```text
2,890 court-retained/team-assigned rows / 12 IDs
→ 2,299 jersey-assigned and roster-compatible / 8 IDs   (-591)
→ 2,291 after semantic deduplication                       (-8)
→ 2,424 final Team-B rows                                (+133)
```

All eight Team-B deduplications are B9; **no B6 or B11 semantic observation is removed by deduplication**. Those two observed counts are exactly T8+T40+T65=902 and T157+T179=332. Inferred-Team-B accounting cannot reveal all physical Team-B losses: green-11 observations mislabeled Team A under T5/T74/T84 reside on the Team-A side of the ledger. [pipeline_loss_accounting.csv](pipeline_loss_accounting.csv) includes Team A as well.

### FACT: spatial diagnostics, with an important limit

Using the existing 100-cm location diagnostic gate and selecting the nearest raw projected point, the **742 missing B11 frames** partition numerically into: jersey-unavailable 386; wrong semantic identity 338; roster-incompatible 2; no raw point within 100 cm 16. The **188 missing B6 frames** partition into: jersey-unavailable 163; no raw point within 100 cm 25. None is classified court-removed in this diagnostic.

These are **not an exact physical root-cause allocation**: close/crossing players, duplicate boxes and inaccurate bottom centers can associate a box to the wrong GT identity. Example: nearest T84 at 740–742 is within 100 cm, while exclusive Hungarian matching leaves B11 unmatched. “No point within 100 cm” combines missed boxes, localization errors and occlusion; it is not a count of YOLO misses. These caveats are why the report uses video to corroborate the main correspondences. See [gt_observation_trace.csv](gt_observation_trace.csv), [missing_window_geometry_accounting.csv](missing_window_geometry_accounting.csv) and [missing_window_stable_ranges.csv](missing_window_stable_ranges.csv).

## Role of interpolation

**FACT — executed cell 18:** after semantic validation/deduplication, the baseline builds `range(player.index.min(), player.index.max()+1)` for each semantic identity, then linearly interpolates existing columns. `limit_direction='both'` fills missing values within that reindexed range; it does not add rows outside it.

- B11's surviving observed range is exactly 743–1118. It fills 809–852 (44 frames) but cannot create any row before 743.
- B6's surviving observed range is exactly 1–930. It fills 28 internal missing frames but creates none after 930.

**CONCLUSION:** the intended interpolation boundary rule exposes/amplifies upstream semantic-support loss. It explains the exact endpoints and all 930 absent final rows, but is not evidence that extrapolation should be added. T172 illustrates the opposite effect: interpolation hides an internal semantic gap without fixing its underlying evidence loss. Nothing about interpolation was changed here.

## Reconstruction and reproducibility caveat

The original `.sort_values('conf',ascending=False).drop_duplicates(...)` has no explicit tie policy. There are **21 maximum-confidence tie groups**. The local NumPy/pandas sort chooses a different observation from the exported result at **9 rows**, all A10/A12. The first analysis attempt therefore failed its strict coordinate replay check; [analysis_attempt1.log](analysis_attempt1.log) is preserved.

For provenance analysis only, tied maximum-confidence candidates were compared with the already-saved final court coordinates. Their saved winners can be recovered uniquely; no track was relabeled, lower-confidence row promoted, or supplied prediction edited. After that artifact-constrained recovery, all final row keys agree and each court coordinate agrees within the five-decimal CSV rounding bound (max ~5e-6 cm). This establishes consistency of the artifact chain, **not independent deterministic reproduction of the unspecified tie order**. [semantic_tie_audit.csv](semantic_tie_audit.csv) records every tie and the nine differences. B11/B6 have no such tie ambiguity, so their coverage diagnosis and observed counts do not depend on this recovery.

## Root cause ranking

1. **CONCLUSION — accepted jersey/semantic identity unavailable on otherwise retained observations.** Strong direct code/stdout/CSV evidence: 591 rows dropped, including early B11-related T7/T59 and late B6-related T198. For the tail, this dominates the persistence of the problem after the raw track returns. We cannot identify which OCR rejection mechanism caused each empty dictionary.
2. **CONCLUSION — physical identity contamination within tracker IDs, followed by fixed whole-track labels.** Manual purple-12→green-11 evidence inside T5 plus A12 stdout mapping and GT trajectory diagnostics explains a long additional B11 segment. T74/T84 crossings complicate the same region. Mixed color features affecting k-means are a **HYPOTHESIS**; the exact HSV features were not saved, so that causal sub-step cannot be verified independently.
3. **CONCLUSION — shorter tracker-output/localization gaps around occlusion.** In particular B6's 931–938 gap and B11's late pre-743 crossing. Their earliest observable failure is before court filtering, but YOLO-versus-ByteTrack attribution is **unresolved**, not ranked by guesswork.
4. **FACT — court rejection and semantic deduplication are not the main explanation for these endpoints.** The relevant four fragments survive court filtering; no valid B6/B11 row is deduplicated. Roster conflict on T47 is real but early/small relative to the principal empty-jersey and mixed-track failures.

The no-extrapolation support rule is a proven downstream boundary mechanism, discussed separately rather than mislabeled as a new upstream failure.

## Highest-value next experiment — exactly one, not implemented

**One-track diagnostic oracle: supply jersey 6 only for T198 in a separate copied run, freezing Team B, all track/court rows, every other OCR vote, deduplication, interpolation and evaluator.** This is a future experiment proposal, not a manual relabel applied by this analysis.

Why this one: T198's physical B6 correspondence has unusually clear jersey-6 imagery and a clean isolated missing-jersey gate. It avoids the mixed T5/T74 ambiguity and tests whether the semantic gate plus existing support bounds explains the persistent tail. Before running, retain a reviewed note of the sampled jersey correspondence; do not assume a track is pure just because of its ID.

Predicted support outcome, **not an observed result**: B6's last semantic observation should become 1118; 175 existing T198 observations should survive and 13 additional internal-gap rows should be interpolated, adding exactly 188 final rows (5,966 total). B11's 742-frame prefix should remain missing. No predicted TI-HOTA improvement is asserted. Failure of this support prediction would falsify the simple gate/boundary explanation or reveal an unmodeled interaction. The baseline files remain frozen for comparison.

## Evidence and reproduction

Primary evidence:

- Supplied `results/predictions.csv`, `mot_results.txt`, `filtered_mot.txt`, `transformed_court.txt`, `IMG_0104_1_with_minimap.mp4` — full hashes in [analysis_manifest.json](analysis_manifest.json).
- [Executed Colab snapshot](source/executed_colab.ipynb), [source retrieval metadata](source/source_retrieval.json), [cell 12 homography](source/cell_12.py), [cell 16 court filtering](source/cell_16.py), [cell 18 recognition/interpolation](source/cell_18.py), [recognition stdout](source/recognition_stdout.txt), [cell 31 overlay](source/cell_31.py). Colab cell 18 execution count 8 records all 23 OCR/cluster mappings; cell 24 execution count 11 records TI-HOTA 0.6351; cell 25 execution count 13 records the location diagnostics. These are supplied-run outputs, not a fresh evaluation performed by this analysis.
- [Tracker lifecycle](tracker_lifecycle.csv), [coverage](final_coverage.csv), [observation fates](filtered_observation_fates.csv), [selected observations](selected_semantic_observations.csv), [final row provenance reconstruction](final_row_provenance_reconstruction.csv), [loss accounting](pipeline_loss_accounting.csv).
- [Transition windows](transition_windows.csv), [candidate neighbors](candidate_semantic_neighbors.csv), [endpoint links](candidate_endpoint_links.csv), [scheduled OCR samples](scheduled_ocr_samples.csv), [geometry diagnostics](raw_observation_geometry_diagnostics.csv), [GT observation traces](gt_observation_trace.csv), [tie audit](semantic_tie_audit.csv).
- [Manual visual notes](manual_visual_evidence.json), [frame/time index](visual/index.json), diagnostic copies in `visual/`, [coverage figure](coverage_and_tracker_spans.png). Main windows: 700–780 (~23.32–25.99 s) and 890–970 (~29.66–32.33 s); auxiliary frames 1,280,300,301,399,408,414,417,430,805,820,850,853,1118. No original tracked-visualization file was available; all diagnostic tracker boxes come from raw MOT.

Reproduction command (uses the existing isolated analysis environment, no GPU/model inference):

```sh
/tmp/trackid3x3-audit-venv/bin/python research/tools/analyze_identity_coverage.py --extract-frames
```

Script: [../../tools/analyze_identity_coverage.py](../../tools/analyze_identity_coverage.py). It only writes under this analysis directory, verifies before/after result hashes, and preserves the original notebook. The fetched notebook/calibration/GT references must be present at the recorded paths. [analysis.log](analysis.log) records the completed analysis; the earlier tie-replay failure is kept separately. `artifact_hashes.json` inventories generated evidence; `analysis_manifest.json` records scope, input/source hashes, checks and unresolved limits.
