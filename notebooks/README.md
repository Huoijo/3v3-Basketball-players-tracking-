# Kaggle GPU notebook

Upload **[TrackID3x3_Challenge_Tutorial_kaggle.ipynb](TrackID3x3_Challenge_Tutorial_kaggle.ipynb)** to Kaggle. It is standalone; neither helper `.py` file needs uploading.

Enable GPU and Internet, then **Restart Session → Run All**. For offline inputs, follow [the Kaggle recipe](../research/kaggle.md). Outputs are under `/kaggle/working/trackid3x3_runs/<run-id>/`.

The original is preserved at [../TrackID3x3_Challenge_Tutorial.ipynb](../TrackID3x3_Challenge_Tutorial.ipynb); `original_notebook.sha256` records its unchanged hash. The Kaggle artifact has its own `.ipynb.sha256` sidecar. Importing/editing in Kaggle changes the exported JSON; runtime therefore saves and hashes a source snapshot from executed raw code and embedded build markdown/metadata. Markdown-only UI edits need a separate exported notebook for exact provenance.

`kaggle_bootstrap.py` and `kaggle_label_probe.py` are build templates embedded in the generated notebook. Rebuild from the repository root with:

```sh
python3 research/tools/build_kaggle_notebook.py
```

The builder refuses a changed original notebook hash. Do not edit the original tutorial. No successful Kaggle GPU execution is claimed by these files; current status is **READY_FOR_KAGGLE**.
