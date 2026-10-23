# Baseline implementation and data flow

Scope: the supplied notebook, not an assumed upstream repository. Cell references below are **zero-based JSON indices**. The frozen source can be read under `runs/20261022-02_baseline-audit_tutorial_homography-smoke/source/cell_XX.py`; line numbers start at 1 inside each extracted file. All described algorithms are verified against this source. Third-party internal defaults cannot be recovered from an unpinned import or a YAML basename.

## 1. Environment and input loading — cells 3, 5

Cell 3 installs unpinned `ultralytics`, `ndjson`, `easyocr`, `gdown`; imports OpenCV, PyTorch, NumPy, pandas, SciPy, IPython; asserts CUDA. Cell 5 invokes `gdown --folder` for folder `1_yDMy8YVpxyqcKg2_eXN1xz_vg2FeCGu`, writing `/content/trackid3x3_tutorial_data`. `find_file` (cell 5:18) recursively finds exactly one match for each input; zero or duplicate filenames fail.

Inputs: network-accessible tutorial folder. Outputs: paths to `IMG_0104_1.MOV`, `IMG_0104_1_tutorial_challenge.csv`, `Outdoor.ndjson`, `Outdoor.png`. The video supplies frames and metadata; GT is used only for evaluation/GT visualization, not by cell 17. Calibration NDJSON is a Labelbox export; the court PNG supplies the minimap background. Data version/checksums are absent in the original. Saved metadata indicates Colab T4, but does not pin a reproducible runtime.

## 2. Detection and multi-object tracking — cell 7

`YOLO('yolo11x.pt').track(source=video_path, tracker='bytetrack.yaml', device='0', classes=[0], save=True, save_txt=True, save_conf=True, stream=True, verbose=False)`. Consumes every result to exhaustion, retaining its first `save_dir` as `out_dir`; raises if no results.

Input: decoded video frames (image pixels). Output: Ultralytics tracking `Result`s consumed in memory, rendered tracking output and `out_dir/labels/*.txt`. `classes=[0]` selects COCO persons, not verified players. Tracker IDs are temporary track identities; they have no team/jersey meaning. Detector confidence, image size, NMS settings and tracker thresholds are implicit library/YAML defaults, not explicit notebook parameters. No independent raw, pre-tracker detections are saved by notebook code. Saved YOLO labels represent the result of tracking; do not assume they expose rejected detections. No tracker configuration file or weight checksum is supplied.

## 3. YOLO-to-MOT conversion and tracker visualization — cell 9

`get_frame_id_from_filename` (line 6) extracts trailing digits. `parse_yolo_tracking_line` (line 27) reads normalized `xc,yc,w,h` from fields 1–4, accepts either ID/confidence order for seven fields via an integer parsing attempt; six fields are interpreted as track ID with confidence 1.0. Lines with fewer than six fields are silently skipped. A six-field detection-only row containing confidence could therefore be misinterpreted/fail; this was not repaired. Seven-field integer-looking confidence can also make order detection ambiguous. Actual label format must be inspected for the installed Ultralytics version.

Inputs: label files and original video width/height/FPS. Output `out_dir/mot_results.txt`, no header:

```text
frame_id,track_id,x_px,y_px,width_px,height_px,confidence,-1,-1,-1
```

Converts normalized center coordinates to top-left pixel boxes; rounds geometry and confidence to two decimals. Source frame number comes from label filename. The display rereads the full video and uses `CAP_PROP_POS_FRAMES` after each read (one-based in the intended convention), drawing temporary tracker IDs in generated HSV colors. Persists `tracked_visualization.mp4`. Class/visibility/last columns are placeholders, not observed class/visibility signals. Actual frame alignment between filenames and decoder must be checked on a real tracking run.

## 4. Court calibration — cells 11, 13

Cell 11 reads the first NDJSON record, first project and first label's annotation objects, selecting exactly four named points: `rku_0→(0,0)`, `rku_1→(0,1105)`, `rku_10→(1505,0)`, `rku_11→(1505,1105)` in cm. Calls `cv2.findHomography(image_points,court_points)` without explicit robust method. `homographic_transformation` (line 91) applies H to `[x,y,1]` and divides by homogeneous scale.

Input: four annotation points in video image pixels. Output: in-memory 3×3 H, pixel→court cm. Assumes annotation coordinates align with video resolution/orientation and a stationary planar court/camera. Checks point count and non-null H; does not validate unique point names, conditioning, held-out error or denominator safety. Court is x∈[0,1505], y∈[0,1105] cm. Point correspondences define orientation; do not infer a different court origin from minimap rendering.

Cell 13 warps the first video frame to 1505×1105 pixels (one pixel/cm), displays through `google.colab.patches.cv2_imshow`, saves `/content/transformed_court_image.jpg`. H itself is only printed in the original. Our wrapper adds H JSON persistence without altering cell 11. Four-point calibration residuals are an exact-fit check, not localization accuracy.

## 5. Projection and court filtering — cell 15

Reads MOT into pandas `df`. Adds `bot_cx=x+width/2`, `bot_cy=y+height`; projects these image foot-point approximations with H. The bottom center of a detected box is assumed to approximate player contact with the court.

Per temporary tracker ID, require longest run of consecutive integer frame IDs inside strict court ≥10 frames. Reject if more than half its unique-frame count lies in near-endline band `y∈[-300,100], x∈[0,1505]`, or far region `y>1002.5` outside paint extension `x∈[502.5,1002.5]`. Counts in these two tests are rows, denominator is unique frames: code assumes one row per ID/frame. Then retain detections in court with 300 cm buffer: x∈[-300,1805], y∈[-300,1405]. This can retain positions outside the actual court.

Dependencies: MOT + H + court bounds + original video. Outputs: filtered `df` in memory and full `total_video_frames` (not used to extend final predictions). Persists:

```text
filtered_mot.txt: frame_id,id,x,y,width,height,conf      # headerless, px
transformed_court.txt: frame_id,id,court_x,court_y      # headerless, cm
```

Only surviving rows are persisted here. Rejected row/track reasons are not saved. Tracklets are whole temporary-ID groups; there is no explicit gap segmentation object.

## 6. OCR sampling and jersey evidence — cell 17:9–335

Fixed roster: A={4,10,12}, B={6,9,11}; fixed final identity order A4,A10,A12,B6,B9,B11. `easyocr.Reader(['en'],gpu=True)`. Sample up to 40 evenly spaced observed rows per filtered tracker ID using `np.linspace`, not a random sample. Sequentially decodes the video again and extracts selected full-player crops, clips to image, upscales ×2 cubic; `readtext(allowlist='0123456789',detail=1,paragraph=False)`.

Accept OCR confidence ≥0.10, strip non-digits, retain numbers in the union of rosters, add confidence into `ocr_scores[track_id][jersey]`. Pick max summed confidence, or None if no accepted candidates. This sum is evidence weight, not a calibrated probability. Two OCR snippets in one sample can both contribute. One jersey is assigned to the entire temporary track.

Inputs: filtered rows, video pixels, hard-coded roster and OCR weights downloaded by EasyOCR. Outputs: transient crops, raw OCR results and confidence; accepted accumulated scores, `track_to_jersey`. Only a human-readable accepted-score summary is printed. Raw rejected OCR, frame-specific scores/crops are not persisted. `infer_roster_team_from_jersey` is defined at line 108 but is **unused**; team is not assigned with that helper.

## 7. Team identification and label alignment — cell 17:49–80, 338–602

`get_upper_body_crop` takes relative bbox region x=20–80%, y=12–62%. `extract_color_feature` computes normalized HSV H/S histogram (16×8 bins); averages vectors across samples per temporary ID. OpenCV k-means K=2, EPS+MAX_ITER, 100 iterations, epsilon 1.0, 20 attempts, k-means++ initialization. Fails if fewer than two tracks have features; no explicit OpenCV RNG seed.

The cluster IDs 0/1 are arbitrary. Select the earliest frame among tracks with features; nearest player to top reference (752.5,1105) cm is initial offense, nearest other player to that player is initial defense. Offense cluster→A, defense cluster→B; fails if both anchors share a color cluster. Assumes check-ball starting geometry, both teams visible, two separable uniform colors, and valid retained players. No GT consulted for this mapping. Team IDs describe initial offense/defense, not possession-changing roles.

Output: `track_color_features`, `track_feature_vectors`, `track_to_cluster`, `cluster_to_team`, printed cluster and anchor summaries. No feature matrix, centers or mapping file persisted.

## 8. Semantic identity and fragmented-track merging — cell 17:609–703

Map each row from temporary ID to jersey and cluster/team. Drop missing either value. Drop team/jersey combinations incompatible with roster. Semantic `identity` is string concatenation, e.g. `A10`. Multiple tracker IDs may map to the same identity; one tracker ID cannot change identity within this implementation.

Sort by detector confidence descending and retain only one row per `(frame_id,identity)`; this is semantic deduplication, not trajectory association. Require all six roster identities to appear at least once, else raise. Equal-confidence tie behavior is not specified as a policy. Outputs only transient `identity_df`, printed mapping; original tracker→semantic correspondence is discarded from final submission.

## 9. Interpolation and final output — cell 17:706–910

For each required semantic identity, reindex every integer frame between its first and last surviving observation; linearly interpolate court_x, court_y, width, height with `limit_direction='both'`. No maximum gap length or physical plausibility constraint. No extension to video start/end. Uses inverse H to map final court positions to image bottom-center foot points, then reconstructs bbox x/y from width/height. Thus final bbox positions can differ from original measured boxes; they are visualization quantities.

Internal `predictions_df` fields:

```text
video_id,frame_id,court_x,court_y,team_id,jersey_number,
bbox_x,bbox_y,bbox_width,bbox_height
```

Writes `out_dir/predictions.csv`, header and exact submission order:

```text
frame_id,court_x,court_y,team_id,jersey_number
```

Court positions rounded to five decimals in CSV. No tracker ID, confidence, interpolation flag, gap length or bbox in the saved submission. Evaluation, minimap and video overlay use this same in-memory final table; CSV reload cannot reconstruct full video overlay fields. Evaluation uses in-memory floats rather than rereading rounded CSV.

## 10. Evaluation — cells 21, 23, 24

Cell 21: `loc_sim`, `id_sim`, `sim_ti_hota` (helper not used by matching), `compute_ti_hota_alpha`, `compute_ti_hota`, `load_gt_frames`, `load_prediction_frames`, `load_video`. Input: exact five-column GT CSV and in-memory final predictions. Frame dictionaries contain lists of `{position:[x_cm,y_cm],attributes:'A10'}`. `load_video` uses union of GT/pred frame IDs but omits frame IDs from output dictionaries. No tracker IDs or image boxes reach evaluator. Cell 23 uses tau=100 cm, prints six aggregates. Cell 24: `compute_location_diagnostics`, position-only Hungarian matching within 100 cm. See [evaluation.md](evaluation.md).

## 11. Minimap and original video — cells 26, 28, 30

Cell 26 helpers: GT/pred point loaders, `scale_points_swapped` (despite name, flips x as `1505-x`; y unchanged; draw_height unused), fixed A purple/B green colors, point/label drawing. GT attribute colors are used for display only. Cell 28 resizes `Outdoor.png` to 1505×1105, draws 30 px circles and jersey labels, writes `/content/IMG_0104_1_GT.mp4` and `_pred.mp4` using source FPS. Only frame IDs having rows are rendered, not a complete video-frame timeline.

Cell 30 reads main/GT/pred video sequentially, draws final reconstructed player boxes as foot ellipses and semantic labels, and blends both minimaps (alpha 0.9) near the top of the original. Writes `/content/IMG_0104_1_with_minimap.mp4`. If a minimap omitted an empty frame or starts late, sequential overlay can lose alignment; neither minimap stores frame IDs. Overlay codec and VideoWriter success are not systematically validated. These are code-derived risks, not observed failures from a freshly generated video.

## ID, coordinate and unit contracts

- `id`/`track_id`: integer tracker-local trajectory label, temporary; reused by filtering/sampling only.
- `cluster_id`: arbitrary k-means index, not stable team ID between runs.
- `team_id`: A/B, fixed by initial-position anchors; semantic team identity.
- `jersey_number`: integer, restricted to supplied roster. By itself is not generally globally unique, even though this clip's rosters do not overlap.
- `identity`/evaluator `attributes`: concatenated team and jersey, persistent semantic player identity, independent of tracker ID.
- Video pixels: decoded image coordinates, x right/y down; normalized YOLO coordinates divided by frame dimensions; MOT boxes top-left xywh in px.
- Court: image bottom-center projected through H into cm in the four-point-defined Outdoor frame. Final matching distance is Euclidean court cm.
- Time: integer video frame indices, intended one-based; seconds only implicit through video FPS. No time normalization in filtering's 10-frame rule or interpolation. GT downloaded here is verified one-based over 1–1118.

## Artifact and temporal failure inspection inventory

Detection boxes and tracker results: normalized tracked labels + `mot_results.txt` + `tracked_visualization.mp4`; raw pre-tracker detections absent. Court positions and filtered tracklets: `transformed_court.txt` + `filtered_mot.txt`; rejected data still exists in raw MOT, but rejected court coordinates and reason flags do not. H: originally printed only plus warped image; this audit saves a separate H JSON. Cropped players: memory only. Team predictions and jersey evidence: summaries in notebook stdout, richer dictionaries only in live kernel. Final identities: submission CSV; source track mapping printed but not persisted. Minimap/final videos: original paths listed above, none supplied or regenerated locally in this sprint.

A researcher can inspect tracker-ID changes and gaps manually using tracker video and raw MOT, and compare court trajectories in transformed text/minimaps if generated. There is no automatic problematic-segment index, switch/fragment event log, uncertainty timeline, localization-error timeline, or referee classification. Court filtering is the only explicit non-player exclusion heuristic. Missing/rejected OCR and wrong-team rejection can be invisible after final generation. Final interpolation can hide tracker gaps. Recommended minimal additions are documented in [experiments.md](experiments.md).
