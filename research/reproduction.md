# Reproduction recipe and execution evidence

## Kaggle GPU execution path

Import `notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb` into Kaggle, enable GPU and Internet, then **Restart Session → Run All**. Offline mode supports the same original data/model files via Dataset inputs; see [the complete short recipe and compatibility report](kaggle.md). Outputs go to `/kaggle/working/trackid3x3_runs/<run-id>/tracking/predictions.csv` and `../ti_hota.json` within the same run directory. Current status is **READY_FOR_KAGGLE**, with no fresh Kaggle score. The original notebook is unchanged; the Colab recipe below still describes that original.

## Canonical baseline

Use the original notebook in a fresh **CUDA-enabled Colab runtime**. Its own instructions assert CUDA and explicitly select GPU 0 and EasyOCR GPU mode. A local macOS/MPS or CPU fallback would change its execution configuration; none was substituted.

Execute code cells in this order (zero-based JSON cell indices):

```text
3 → 5 → 7 → 9 → 11 → 13 → 15 → 17 → 19 → 21 → 23 → 24 → 26 → 28 → 30
```

Cell 13 is a calibration visualization and can be skipped for a metrics-only run; it also depends on Colab's display helper. Cell 19 prints GPU/OCR device information. The core metrics-only dependency chain is `3,5,7,9,11,15,17,21,23,24`; cell 9 nevertheless renders a full tracker video. Keep the complete order for canonical tutorial reproduction. No standalone end-to-end CLI existed initially; the audit script is not a substitute for notebook model execution.

Required inputs downloaded by cell 5 into `/content/trackid3x3_tutorial_data`: `IMG_0104_1.MOV`, GT CSV, `Outdoor.ndjson`, `Outdoor.png`. Historical output reports a ~121 MB video, ~193 KB GT, ~14 KB calibration, ~76 KB court image. Actual current video properties were not probed locally because the video was not downloaded. Historical output indicates 29.97002997 FPS and 1,118 frames; only GT frame count was verified locally. YOLO and EasyOCR weights are automatic downloads; saved YOLO output reports ~109 MB `yolo11x.pt` and auto-installation of `lap==0.5.13`.

Configuration is embedded in notebook globals; all explicit parameters are listed in architecture.md. There are **no explicit seeds** for Python, NumPy, OpenCV or PyTorch. K-means uses random initialization; package versions, model hashes, full tracker YAML and device/runtime determinism are not pinned. Do not label the saved score as exactly reproducible yet.

For the next clean CUDA run, before and after execution capture Python and package versions (`python --version`, `python -m pip freeze`), `nvidia-smi`, torch/CUDA/cuDNN versions, actual Ultralytics tracker YAML, weights and all input hashes. Save executed notebook, stdout/stderr, total and stage timings, `out_dir`, generated predictions and all artifact hashes. Record implicit library settings; do not invent a lock from the saved output. First reproduce the unchanged unseeded baseline, then define separately named seeded runs if desired.

## Local audit commands actually executed

From `/Users/thuong/Desktop/TrackID3x3`:

```sh
python3 research/tools/baseline_audit.py \
  --output research/runs/20261022-01_baseline-audit_tutorial_eval-smoke \
  --download-gt

python3 -m venv --system-site-packages /tmp/trackid3x3-audit-venv
/tmp/trackid3x3-audit-venv/bin/python -m pip install opencv-python-headless ndjson

/tmp/trackid3x3-audit-venv/bin/python research/tools/baseline_audit.py \
  --output research/runs/20261022-02_baseline-audit_tutorial_homography-smoke \
  --gt research/runs/20261022-01_baseline-audit_tutorial_eval-smoke/IMG_0104_1_tutorial_challenge.csv \
  --download-calibration
```

Run directories are immutable by convention; the wrapper rejects an existing `--output`. For reruns choose a new sequence number. The script imports function definitions verbatim from cell 21 and the diagnostic function from cell 24 through Python AST. It executes all of cell 11 verbatim only when `--download-calibration` is supplied. It snapshots every cell and textual saved output; does not execute pip magics, tracker, OCR, or videos. Fresh run manifests include total wall time, command, versions and file hashes. An exit status of 0 means the requested component checks passed, **not** that full baseline inference ran.

For an independently obtained existing prediction CSV:

```sh
python3 research/tools/baseline_audit.py \
  --output research/runs/20261022-03_baseline_tutorial_external-csv-eval \
  --gt /absolute/path/to/IMG_0104_1_tutorial_challenge.csv \
  --predictions /absolute/path/to/predictions.csv
```

This optional command was not executed because no baseline predictions were supplied. It evaluates all GT/pred union frames at tau=100 cm using notebook functions, and emits `evaluation.json`. CSV values have been rounded to five decimals; the original notebook evaluates full-precision in-memory floats. Provenance of supplied predictions must be verified independently; successful external CSV evaluation alone is not end-to-end reproduction. Do not claim video generation works by reloading this five-column file.

## Environment, failures, and allowed fixes

Observed host: macOS 15.1.1, ARM64, Python 3.13.6. Numeric environment: NumPy 2.4.1, pandas 2.3.3, SciPy 1.17.0. Initially PyTorch, OpenCV, Ultralytics, EasyOCR, ndjson, gdown, nbclient and lap were not installed. The original preparation Python statements failed immediately at missing OpenCV. No NVIDIA CUDA execution path is available on this macOS host; torch itself was absent, so its CUDA query cannot run.

Only environment addition: an isolated `/tmp/trackid3x3-audit-venv` inheriting existing numeric packages, with `opencv-python-headless==5.0.0.93` and `ndjson==0.3.1`. These versions support the local calibration smoke only and do not assert the original Colab environment. Installation log is in run 02. After this addition cell 3's Python statements get past OpenCV/ndjson and fail at missing `torch`. We did not install full inference dependencies because doing so would not satisfy the notebook's CUDA requirement. We did not edit imports, device flags, seed policy, models or algorithms to manufacture a full run.

Downloaded GT and calibration are SHA-256 identified in run manifests. Network was accessible. No video/model/OCR weights were downloaded. Full detector/tracker→identity→predictions→evaluation and all videos remain unexecuted locally. Runtime logs record the actual first failure; later CUDA/Colab constraints are code/platform findings, not fabricated stack traces.

## Minimum valid subset achieved

- Actual baseline cell 11 on real Outdoor calibration, persisting computed H and four-point fit residuals.
- Actual evaluator functions on deterministic synthetic contract fixtures.
- Actual GT loader + evaluator on real first-three-frame GT with deliberately empty predictions (negative control only).

This is a component reproduction, not a representative end-to-end tracking subset. A shortened video does not automatically give a valid baseline: the track filter needs ≥10 consecutive in-court frames, identity generation requires all six identities, OCR samples vary with available track length, and team mapping depends on initial anchors. Freeze a full baseline before defining a research subset. No baseline score is available from this local execution.
