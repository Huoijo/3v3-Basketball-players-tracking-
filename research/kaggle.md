# Kaggle compatibility sprint — 2026-10-22

**Status: READY_FOR_KAGGLE.** The standalone notebook is prepared and locally validated. There is no actual Kaggle GPU inference/evaluation evidence yet. This environment exposes a Kaggle CLI but has neither a Kaggle credential file nor KAGGLE credential environment variables; no connected Kaggle execution tool was available. Nothing was uploaded or published remotely.

The original tutorial and the five existing research documents were reviewed before implementation. All 15 original code cells were inspected. The original remains at `TrackID3x3_Challenge_Tutorial.ipynb`, byte-for-byte SHA-256 `33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00`. The derivative is [../notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb](../notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb). Its SHA-256 sidecar is [../notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb.sha256](../notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb.sha256).

## How to run on Kaggle

1. Create a Kaggle notebook and import `notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb`.
2. In Settings enable a **GPU accelerator** and **Internet**. Defaults download the original tutorial folder and original YOLO11x/EasyOCR weights.
3. If offline, attach the exact original tutorial files and original model weights as Dataset inputs, plus wheels for any missing packages. In the first code cell set `ALLOW_DOWNLOADS=False`, `DATA_INPUT_DIR`, optionally `MODEL_INPUT_DIR`, and `WHEEL_DIR` if needed. Keep directories specific if multiple attached inputs contain the same filenames.
4. Choose **Restart Session → Run All**. Keep models, thresholds, roster, interpolation and evaluation unchanged. The isolated real-label probe runs before full inference; inspect its failure evidence if it stops.
5. Open the printed run directory in `/kaggle/working/trackid3x3_runs/`. Read `tracking/predictions.csv`, `ti_hota.json`, `location_diagnostics.json`, and `run_manifest.json`. Save/download the run directory as notebook output.

Videos are optional: set `SAVE_VIDEOS=True` before Run All if desired. The default False changes video generation only. Predictions and metric calculations use the same original in-memory table and logic.

## Input, output and temporary locations

Online acquisition retains the original folder URL and `gdown --folder -O` behavior. The command uses the active kernel's interpreter (`sys.executable -m gdown`) and checks its return code instead of relying on a shell executable. `gdown` has a standard Python CLI and is not a Colab API; actual Kaggle network access/Google Drive limits remain runtime checks. See [gdown's CLI source](https://github.com/wkentaro/gdown/blob/main/gdown/__main__.py).

Defaults:

```text
/kaggle/input/                            read-only attached files
/kaggle/working/trackid3x3_tutorial_data/  original-source download/cache
/kaggle/working/trackid3x3_runs/<run-id>/  one fresh run, all generated results
  tmp/                                   isolated probe script and temporary work
  models/yolo11x.pt                       same model, writable local copy/download
  models/easyocr/                         original default OCR weight cache
  tracking/                              full baseline labels, MOT and predictions
  label_probe/                           separate first-12-frame compatibility sample
```

`WORK_DIR` is discovered from the Kaggle writable directory; an explicitly named local fallback only supports non-Kaggle checks and does not bypass the CUDA requirement. Run IDs include local Asia/Ho_Chi_Minh time with microseconds to avoid reusing labels or auto-incremented tracking directories. No prior kernel variables or previous predictions are required.

The resolver prints all four final paths and requires exactly one of each: `IMG_0104_1.MOV`, `IMG_0104_1_tutorial_challenge.csv`, `Outdoor.ndjson`, `Outdoor.png`. Explicit `DATA_INPUT_DIR` never silently falls back to network or another dataset. Automatic mode prefers a complete attached set, then the original-source cache/download; it never combines partial attached and downloaded sets. Known audited GT/calibration hashes are checked; different bytes fail with an explanation. All four input hashes and video metadata are saved. Video/PNG have no previously verified reference hash, so their recorded hashes establish provenance, not independent proof of origin. Supply the original files, not renamed substitutes.

Offline YOLO must be the original `yolo11x.pt`. Default English EasyOCR requires `craft_mlt_25k.pth` and `english_g2.pth`; files are copied into a writable run cache and checked by EasyOCR's own loading logic. The default English model selection is confirmed in [EasyOCR's Reader implementation](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py). No OCR model or recognition parameters are selected differently. Weights are not embedded in the notebook. A fully offline run also needs already-installed dependencies or matching wheels and their dependencies in `WHEEL_DIR`.

## Compatibility audit and applied fixes

Original cell references below are zero-based JSON indices.

- **ENVIRONMENT — cell 3:** `assert torch.cuda.is_available(), 'Please enable a GPU runtime in Colab.'` assumes Colab terminology. Replaced by Kaggle guidance, printed torch/CUDA/GPU information, a device-0 CUDA allocation check and hard failure on no CUDA. EasyOCR's selected device is checked immediately after Reader initialization; no CPU/MPS fallback is accepted.
- **DEPENDENCY — cell 3:** four unconditional `!pip install ...` calls can use a different interpreter or alter the existing image unnecessarily. The replacement checks installed distributions, installs only missing/minimum-incompatible baseline packages using the kernel interpreter, captures installation output, and constrains existing torch/torchvision/torchaudio, NumPy/pandas/SciPy, OpenCV, NVIDIA packages and Triton versions. No `--upgrade` and no automatic PyTorch/CUDA replacement. Import/solver failures remain visible. `lap>=0.5.12` is installed proactively when needed instead of relying on tracker auto-installation seen in historical output.
- **PATH — cell 5:** `download_dir='/content/trackid3x3_tutorial_data'` is replaced by DATA_DIR and an offline input resolver. The same source data and four filenames are retained. Checked subprocess execution replaces `!gdown ...`; network failure points to the original-file Dataset workflow.
- **PATH — cell 7:** implicit `runs/detect/track*` output and current-directory `yolo11x.pt` are replaced by explicit run-local output/model paths. `bytetrack.yaml` is copied from the installed Ultralytics package, hashed, and passed by that identical-content path; no tracker parameter is edited.
- **LIBRARY-COMPATIBILITY — cells 7, 9:** the current parser has ambiguous six/seven-field behavior. It is **unchanged**. A separate process tracks the first 12 real video frames using the same YOLO11x, ByteTrack YAML, classes, GPU, saved-confidence setting and inference defaults. It reads actual generated labels and compares the original parser's ID/confidence/normalized geometry against each yielded `Results.boxes` entry. On discrepancy it saves actual lines/expected values and stops; it does not guess a repair. No probe state or rows enter the full baseline.
- **VISUALIZATION — cells 7, 9:** annotated tracking video and the separate full-video MOT drawing loop are gated by SAVE_VIDEOS; normalized labels, MOT conversion and all metrics still run. The original conversion code and parser remain intact.
- **VISUALIZATION / PATH — cell 13:** `from google.colab.patches import cv2_imshow` is replaced by OpenCV PNG encoding plus `IPython.display.Image/display`. `/content/transformed_court_image.jpg` moves inside RUN_DIR. The complete cell is optional and runs after evaluation. Pixel warping/drawing parameters remain identical.
- **PATH / ENVIRONMENT — cell 17:** Reader gets only a writable `model_storage_directory` and explicit `download_enabled` policy. `['en']`, `gpu=True`, OCR settings, votes and every threshold remain unchanged. CUDA validation does not change classification behavior.
- **OBSERVABILITY — cell 17:** the original monolithic cell is split at its existing OCR/team/semantic/final-prediction comment boundaries. Same statements, same order, same notebook global scope. Stage context managers add timing/logs without function scope. No algorithm extraction/refactor.
- **ENVIRONMENT — cell 19:** its informational CUDA/OCR prints are covered by stage 1 and Reader checks; no duplicate standalone device-print cell is needed.
- **OBSERVABILITY — cells 11, 15, 17, 23, 24:** H, row counts, metrics, diagnostics, prediction hash and historical deltas are persisted without altering transformations or matching. Evaluation still uses full-precision in-memory `predictions_df`, not a reloaded rounded CSV.
- **PATH / VISUALIZATION — cells 28, 30:** all `/content` video destinations become RUN_DIR. Minimap helpers and drawing rules are unchanged. These optional cells follow persisted evaluation; the homography matrix is restored after cell 30's original use of `H` as display height so final evidence is safe. Original sparse-frame minimap behavior is preserved.
- **OBSERVABILITY — all stages:** clear `[1]` through `[11]` markers, input/count/output summaries, elapsed seconds, failure traceback, source/env/model/input/output hashes, and one fresh run directory. Shell output is saved to logs. Notebook saved outputs/execution counts are cleared rather than carrying historical results into the derivative.

Cell 21 (evaluator definitions) is copied unchanged. Cell 26 (minimap helpers) is copied unchanged into the optional block. Imports in cell 3 are retained after dependency preparation. No remaining `/content` path, `google.colab` import, or shell magic occurs in executable Kaggle code. IPython display is available in a standard notebook and needs no Colab replacement beyond the image helper.

## Dependency policy and runtime evidence

The standard Kaggle GPU image should supply Python, torch/torchvision, NumPy, pandas, SciPy, IPython and packaging. Missing core packages fail clearly; they are not auto-upgraded/repaired. Optional baseline packages are checked: Ultralytics (minimum 8.3.0 for YOLO11 support), EasyOCR, ndjson, lap, and gdown only when downloads are enabled. OpenCV is installed only if no cv2 module exists. The YOLO11 configuration exists in [Ultralytics v8.3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/models/11/yolo11.yaml); this minimum is a compatibility floor, not a claimed reproduction lock.

Required transitive packages may be installed by pip under protected runtime constraints. Existing incompatible imports or unsatisfiable constraints stop the run; the notebook does not blindly reinstall the environment. `dependency_plan.json`, `dependency_install.log`, `protected-runtime-constraints.txt`, `pip-freeze.txt` and `environment.json` show what actually happened. ByteTrack uses the unchanged YAML of the installed version, and `tracking_arguments.json` records resolved inference defaults.

Current Ultralytics documentation writes confidence then ID for tracked boxes with saved confidence. That is evidence of a possible format, not proof of the version in a future Kaggle session. See [Results.save_txt](https://docs.ultralytics.com/reference/engine/results/). A real GPU probe remains mandatory; local synthetic labels do not establish compatibility. No parser change has been made without actual Kaggle evidence.

## Validation and result

Final local evidence: [runs/20261022-05_kaggle-compatibility_local-validation/validation.json](runs/20261022-05_kaggle-compatibility_local-validation/validation.json) and [validation.log](runs/20261022-05_kaggle-compatibility_local-validation/validation.log).

Nine checks pass:

1. nbformat 4.5 schema, all 16 Python code cells compile, outputs cleared, no Colab imports/paths, original hash unchanged.
2. AST comparison preserves original numerical/semantic blocks. Only documented model/YAML paths, video flag, Reader cache/download options and CUDA guard are allowed; stage indentation's docstring whitespace is normalized.
3. Runtime source-capture round trip reconstructs the shipped Kaggle notebook byte-for-byte from code history and embedded markdown/metadata.
4. Offline input resolution works with spaces in paths; missing and duplicate files fail. This test uses a synthetic tiny video/PNG plus real GT/calibration, labeled as a filesystem fixture rather than tutorial inference.
5. The generated homography cell runs on actual tutorial Outdoor calibration; max four-point calibration-fit residual is approximately 6.02e-13 cm, not independent localization accuracy.
6. Generated MOT conversion on a synthetic seven-field label gives the expected pixel-space row.
7. The probe comparison detects integer-looking confidence ambiguity while leaving the original parser untouched.
8. Runtime exceptions propagate and persist failure state/traceback.
9. Generated evaluator perfect/missing synthetic contract checks pass.

Commands used:

```sh
python3 research/tools/build_kaggle_notebook.py
/tmp/trackid3x3-audit-venv/bin/python -m pip install nbformat
/tmp/trackid3x3-audit-venv/bin/python research/tools/validate_kaggle_notebook.py \
  --output research/runs/20261022-05_kaggle-compatibility_local-validation
```

The local validation interpreter is macOS ARM, Python 3.13.6, NumPy 2.4.1, pandas 2.3.3, SciPy 1.17.0, OpenCV headless 5.0.0.93, ndjson 0.3.1, nbformat 5.11.1. PyTorch/Ultralytics/EasyOCR/lap are not installed locally, and there is no CUDA GPU execution. nbformat and its schema dependencies were added only to the prior isolated `/tmp/trackid3x3-audit-venv`; the global Python environment was not changed. The installation log and local environment inventory are stored in the final validation directory. These versions are not a Kaggle environment lock.

Earlier run 03 failed because the initial AST validator compared indentation inside function docstrings literally after adding stage contexts. The validator was corrected to normalize docstring indentation. Numerical statements were unchanged. [Run 03 evidence](runs/20261022-03_kaggle-compatibility_local-validation/validation.json) is retained. [Run 04](runs/20261022-04_kaggle-compatibility_local-validation/validation.json) passed; run 05 validates the final notebook including original-author attribution and the original-hash sidecar. No failed check or inference failure is concealed.

## Fresh results and historical comparison

No fresh prediction rows, TI-HOTA, TI-DetA, TI-AssA, TP/FP/FN or position diagnostic values exist from a Kaggle run. Fresh metrics are deliberately absent from local validation. Historical references remain TI-HOTA 0.6351, TI-DetA 0.6362, TI-AssA 0.6341 and are unverified saved outputs. A successful future run writes all six fresh TI quantities, four location diagnostic fields, row/frame counts and reference deltas. Nothing asserts exact equality with 0.6351 or tunes to match it. Unset k-means RNG, package/default and weight provenance differences must remain visible.

## Outputs produced on Kaggle

All paths below are relative to the printed RUN_DIR and are expected future runtime outputs, not files claimed to exist locally:

- `tracking/predictions.csv`: exact original five-column submission, five decimal places; `output_summary.json` records rows, frames and prediction SHA-256.
- `tracking/labels/*.txt`, `tracking/mot_results.txt`, `tracking/filtered_mot.txt`, `tracking/transformed_court.txt`: original intermediate formats.
- `homography.json`, `homography.npy`: computed matrix and calibration correspondences.
- `ti_hota.json`: fresh original in-memory metrics, tau and alpha values; `location_diagnostics.json`: original conditional spatial diagnostics; `historical_comparison.json`: explicit reference and fresh deltas.
- `environment.json`, `pip-freeze.txt`, `dependency_plan.json`, `dependency_install.log`, `protected-runtime-constraints.txt`: actual runtime versions and dependency actions.
- `bytetrack.yaml`, `models_before_inference.json`, `model_hashes.json`, `tracking_arguments.json`: actual tracker YAML, YOLO hash, OCR hashes, resolved inference defaults.
- `inputs.json`, `video_properties.json`: all four file paths/sizes/hashes and decoded metadata.
- `configuration.json`, `run_manifest.json`, `stages.log`, optional `failure.log`, `artifact_hashes.json`: configuration, scope/platform, timing/status, errors and artifact inventory.
- `label_probe.log`, `label_probe/label_format.json`, `label_probe/tracking/labels/*.txt`: actual isolated GPU sample evidence. Only a successful real probe marks its own format check PASS.
- `kaggle_source_snapshot.ipynb`, `source_hashes.json`: source-only notebook reconstructed from executed raw code and embedded build markdown/metadata, its actual SHA-256 and original notebook SHA-256. Code edits are captured; markdown-only UI edits need a separate exact export. Local round-trip verifies this snapshot matches the shipped source bytes when unchanged. Kernel-created exports with outputs/metadata may have a different byte hash.
- Optional `tracking/tracked_visualization.mp4`, tracking-rendered video, `transformed_court_image.jpg`, `IMG_0104_1_GT.mp4`, `IMG_0104_1_pred.mp4`, `IMG_0104_1_with_minimap.mp4` when SAVE_VIDEOS is True.

## Files changed and categories

- `notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb` — ENVIRONMENT, PATH, DEPENDENCY, VISUALIZATION, LIBRARY-COMPATIBILITY, OBSERVABILITY; separate standalone derivative.
- `notebooks/kaggle_bootstrap.py` — ENVIRONMENT, PATH, DEPENDENCY, OBSERVABILITY; embedded configuration/runtime helpers.
- `notebooks/kaggle_label_probe.py` — LIBRARY-COMPATIBILITY, OBSERVABILITY; isolated real-output validation template. It uses the original parser supplied at build time.
- `research/tools/build_kaggle_notebook.py` — ENVIRONMENT, PATH, DEPENDENCY, VISUALIZATION, LIBRARY-COMPATIBILITY, OBSERVABILITY; deterministic source-based notebook generation.
- `research/tools/validate_kaggle_notebook.py` — OBSERVABILITY; non-GPU checks and preservation evidence.
- `notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb.sha256`, `notebooks/original_notebook.sha256` — OBSERVABILITY; recorded artifact hashes.
- `notebooks/README.md`, `research/kaggle.md` — OBSERVABILITY; execution recipe, compatibility/change audit, evidence/remaining limits.
- Root `README.md`, `research/README.md`, `research/reproduction.md` — OBSERVABILITY; small Kaggle entry/recipe sections; prior audit preserved.
- `research/runs/20261022-03_kaggle-compatibility_local-validation/*`, `research/runs/20261022-04_kaggle-compatibility_local-validation/*`, `research/runs/20261022-05_kaggle-compatibility_local-validation/*` — OBSERVABILITY; failed/final local evidence, source snapshot, computed calibration H, logs, inventories and hashes. No generated inference predictions are included.

There are **no ALGORITHM-IMPROVEMENT changes**. Model families/weights selection, detection/tracking parameters, parser, OCR thresholds, court rules, team/roster logic, merging/interpolation and evaluator semantics are preserved. No research experiment was started.

## Remaining problems / runtime checks

Actual Kaggle GPU allocation, dependency compatibility, Drive/model downloads, real saved label format and full six-identity recognition are unverified until Run All. A parser mismatch, same-cluster team anchors, missing identity or any original baseline exception stops visibly instead of triggering an algorithm change. Offline mode requires original weights and complete compatible dependencies, not just the four data files. Optional video codecs/display and the original sparse-frame minimap limitation remain secondary and unverified on Kaggle. No known unresolved Colab-path or execution-order blocker remains after local validation; end-to-end success is pending actual execution.
