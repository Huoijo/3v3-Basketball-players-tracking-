"""Build the final evidence report after inference and gated evaluation."""
import hashlib
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent


def value(x):
    return 'None' if pd.isna(x) else str(int(x))


def main():
    result=json.loads((OUT/'target_result.json').read_text())
    baseline=json.loads((OUT/'baseline_reproduction.json').read_text())
    visual=json.loads((OUT/'visual_source.json').read_text())
    comparison=pd.read_csv(OUT/'baseline_vs_exp1b_jerseys.csv')
    evidence=pd.read_csv(OUT/'target_ocr_evidence.csv',keep_default_na=False)
    new=evidence[evidence.sampling=='quality']
    original=evidence[evidence.sampling=='baseline_uniform']
    ablations=pd.read_csv(OUT/'variant_ablations.csv')
    target=ablations[ablations.tracker_id==198]
    score=result['automatic_result']
    changes=comparison.change.value_counts().to_dict()
    harmful_review=int(comparison.change.isin(['value_to_abstain','changed_value']).sum())
    next_decision='productionize this lightweight jersey-recovery method' if result['status']=='SUCCESS' and result['downstream_run'] and harmful_review==0 else 'test a stronger jersey-recognition model'
    readout=[f'# Exp01b — automatic jersey recovery\n\n## Status\n\n**{result["status"]}**. Automatic predicted jersey for T198 = **{score["jersey"]}**. This is an automatic image/OCR experiment, separate from the Exp01 oracle.',
      f'## Visual source\n\nOriginal public tutorial Drive file `{visual["drive_file_id"]}` was downloaded to [{visual["path"].split("exp01b_auto_jersey_recovery/")[-1]}](source/IMG_0104_1.MOV). SHA256 `{visual["sha256"]}`; {visual["size"]:,} bytes; {visual["width"]}×{visual["height"]}; {visual["decoded_frames"]} decoded frames at {visual["fps"]:.8f} fps. All {visual["observations_reconstructed"]:,} filtered observations have nonempty reconstructible crops. {visual["selected_unique_crops"]:,} lossless PNG crops were cached. No pixels from the annotated minimap video were used for inference.',
      '## Baseline reproduction\n\nOriginal tutorial: equally spaced indices over each sorted track\'s observations using `np.linspace`, at most 40; full clipped person bbox; 2× cubic resize; `Reader(["en"], gpu=True)`; `readtext` digits allowlist, detail=1, paragraph=False and remaining defaults. Accept confidence ≥0.10 and digits in the union of existing rosters; sum confidence per candidate and choose the largest sum, otherwise None. The baseline team is B and its historical accepted scores are `{}` → jersey=None. Original rejected OCR results were never saved.',
      f'Fresh CPU replay: **{baseline["calls"]}** target calls; **{baseline["raw_detections"]}** returned text detections; **{baseline["empty_calls"]}** empty-text result rows; accepted scores `{baseline["replay_scores"]}`; jersey **{baseline["replay_jersey"]}**. See [baseline reproduction](baseline_reproduction.json) and [every baseline target result](baseline_target_raw.csv). CPU/runtime/decode differences from Colab are an explicit reproduction limitation; this replay cannot establish the historical contents of rejected reads.',
      '## Automatic strategy\n\n[Configuration](configuration.json) and [inference script hash](preregistration.json) were frozen before OCR. [run.py](run.py) contains no target-track or known-number recognition rule and never reads GT. Frozen team assignments and existing rosters filter OCR candidates only; a missing jersey is never filled by roster elimination. There is no baseline/oracle fallback. All 23 retained trackers receive the same procedure.',
      'For every observation, quality is the geometric mean of clipped normalized bbox resolution, Laplacian variance and grayscale contrast, multiplied by the retained bbox area fraction. Each component is clipped to [0.05,1]. Sort quality descending, then frame ascending; greedily retain at most 32 frames with at least 3 frame steps between samples. Short tracks may provide fewer. [All candidate scores](candidate_quality.csv) and [selected frames](selected_frames.csv) preserve the selection.',
      'Three variants on the quality-selected frames: **A full_2x**, identical baseline full-bbox cubic 2×; **B torso_2x**, x=15–85%, y=12–65% of the clipped bbox, cubic 2×; **C torso_clahe_3x**, same torso, grayscale CLAHE (clip=2, grid=8×8), cubic 3×, mild unsharp mask (Gaussian sigma=1; weights 1.3/−0.3). Same EasyOCR English/CRAFT/English-G2 stack throughout, with CPU quantize=False for both arms to match the original GPU full-precision path; [pre-OCR precision amendment](protocol_amendment_before_ocr.json) records this change before any reads; [environment](environment.json) records CPU, four Torch threads, package versions and weight hashes. [Model transport/checksums](model_sources.json) verify byte identity to [official EasyOCR checksums](https://raw.githubusercontent.com/JaidedAI/EasyOCR/v1.7.2/easyocr/config.py); a mirror changes transport only.',
      'Accept raw OCR candidates at confidence ≥0.10 that belong to the frozen team roster. Per frame and candidate take the maximum confidence across variants, then weight by `(0.5 + 0.5 × quality)` and sum across frames. A winner needs at least 2 distinct supporting frames, ≥0.60 of total candidate score, and normalized best-minus-runner margin ≥0.20. Otherwise abstain. The score share is a heuristic, not a calibrated probability. Three crop variants from one frame cannot supply three temporal votes.',
      '## T198 evidence\n\nAll-track results were written to [automatic_results.json](automatic_results.json) before [evaluate.py](evaluate.py) inspected T198. [Raw OCR evidence](ocr_evidence.csv) includes every selected input/variant, empty-call sentinels, text, parsed candidate, confidence, union/team roster membership, acceptance flags, quality, OCR box and call time. [Target evidence](target_ocr_evidence.csv) is the complete diagnostic subset. [Representative crops](representative_crops) were copied and inspected only after inference; all selected source crops remain in `crops/`.',
      f'Candidate scores: `{score["candidate_scores"]}`. Best={score["best_jersey"]}; runner-up={score["runner_up"]}; best score={score["best_score"]:.6f}; share={score["share"]:.6f}; normalized margin={score["normalized_margin"]:.6f}; support={score["support_frames"]} distinct frames; accepted={score["accepted"]}. Accepted digit-6 evidence occurred at frames **{result["six_frames"]}** ({result["accepted_six_readings"]} readings across variants).',
      'Highest-confidence accepted target readings (all candidates, at most 16; complete records are in the CSV):']
    accepted=new[new.accepted_new==True].sort_values(['confidence','frame_id'],ascending=[False,True]).head(16)
    for row in accepted.itertuples():
        readout.append(f'- Frame {row.frame_id}, `{row.variant}`: text=`{row.ocr_text}`, candidate={row.candidate}, confidence={row.confidence:.6f}, quality={row.quality:.6f}.')
    if not len(accepted):readout.append('No target reading passed both confidence and team-roster filters.')
    readout+=['\nVariant comparisons with the same new aggregation rule:']
    for row in target.itertuples():
        readout.append(f'- `{row.variant}`: predicted={value(row.jersey)}, best={value(row.best_jersey)}, support={row.support_frames}, score={row.best_score:.6f}, share={row.share:.6f}, margin={row.normalized_margin:.6f}.')
    weak=original[(pd.to_numeric(original.candidate,errors='coerce')==6)&(original.confidence<.1)]
    readout.append(f'\nFresh baseline weak digit-6 reads rejected below 0.10: {len(weak)}; frames {weak.frame_id.tolist()}. This answers the question for this replay only; the original historical weak reads are unavailable. Quality-only full-bbox versus uniform full-bbox and the three fixed quality-sample variants isolate useful comparisons, but sample count and aggregation also differ from the original baseline. This is not an exhaustive factorial benchmark.')
    readout+=['## All-track sanity check',f'Changes against the **frozen** baseline: `{changes}`. Disagreements are flagged for review, not automatically called mistakes. [Full comparison](baseline_vs_exp1b_jerseys.csv), [candidate scores](track_jersey_scores.csv), [variant ablations](variant_ablations.csv).']
    for row in comparison.itertuples():
        readout.append(f'- T{row.tracker_id}: baseline {value(row.baseline_jersey)} → automatic {value(row.new_jersey)}; {row.change}; fresh original-method replay={value(row.replay_baseline_jersey)}.')
    readout+=['## T198 result',f'`automatic predicted jersey = {score["jersey"]}`. The evidence and abstention thresholds above were not tuned after target inspection. Final supporting-frame count is {score["support_frames"]}; the preregistered rule requires at least two, so one isolated reading cannot trigger recovery. Temporal support across nearby frames remains correlated visual evidence.','## Downstream result']
    if result['downstream_run']:
        metrics=pd.read_csv(OUT/'metric_comparison.csv')
        readout.append('The original downstream suffix and TI-HOTA evaluator were executed verbatim, using **all automatic mappings and abstentions**, frozen teams/geometry, and no oracle injection. All arms evaluated from exported five-decimal CSV at tau=100 cm, 19 alpha thresholds. TP/FP/FN are threshold-averaged counts. [Predictions](predictions_exp1b.csv), [metric comparison](metric_comparison.csv), [B6 and total coverage](coverage_comparison.json), [CSV diff summary](csv_diff_summary.json).')
        for row in metrics.to_dict('records'):
            readout.append(f'- {row["metric"]}: frozen={row["baseline"]:.6f}; Exp01 oracle={row["exp01_oracle"]:.6f}; Exp01b automatic={row["exp01b"]:.6f}.')
        for arm in json.loads((OUT/'coverage_comparison.json').read_text()):
            readout.append(f'- {arm["arm"]}: total {arm["total_rows"]} rows; B6 {arm["B6_rows"]} rows, frames {arm["B6_first"]}–{arm["B6_last"]}.')
        readout.append('Changes can affect other tracks, semantic support and the original unspecified equal-confidence sort. Aggregate score differences are not a pure T198-only effect. Neither localization errors nor association contamination were repaired.')
    elif 'downstream_failure' in result:
        readout.append(f'The automatic target passed, so unchanged downstream execution was attempted, but it stopped: `{result["downstream_failure"]}`. No fallback labels or hypothetical metrics were manufactured. See `downstream/run.log`. Primary status concerns automatic target recovery, not downstream deployability.')
    else:
        readout.append('Not run: the automatic target result did not satisfy T198→6. No predictions or hypothetical Exp01b metrics were produced.')
    readout+=['## Interpretation',('The existing OCR stack recovered the target automatically in this fixed experiment. This demonstrates realizable visual evidence for this fragment; it does not establish general accuracy or production readiness. Review all-track changes and downstream results before generalizing.' if result['status']=='SUCCESS' else 'This fixed lightweight procedure did not convert the oracle into accepted automatic target recovery. Its failure is specific to this sampling/cropping/preprocessing/voting configuration and current OCR stack; it does not prove the digit is visually absent or that every generic method must fail.'),'## Next decision',f'**{next_decision}.** This is the sole recommendation; no next experiment, stronger model, training, tracker change, or association logic was implemented.','## Integrity and reproduction','[Verification](verification.json) checks all original `results/`, original notebook and Exp01 files unchanged, plus the frozen configuration and inference-script hashes. [Source hashes](source_hashes.json), [manifest](experiment_manifest.json), [preregistration](preregistration.json), and [environment](environment.json) capture provenance. Preparation used the existing analysis Python 3.13/OpenCV 5 environment; OCR used an isolated Python 3.12 environment with preinstalled Torch/TorchVision and NumPy. Its OpenCV ABI3 package is linked from the analysis environment after a successful import/array-operation check. Exact Colab library versions and historical weight hashes were not recorded, so CPU replay parity is tested rather than presumed.',
      'Completed stage order (do not rerun into a completed directory; copy to a fresh experiment directory at the same depth):\n\n```sh\n/tmp/trackid3x3-audit-venv/bin/python research/experiments/exp01b_auto_jersey_recovery/run.py prepare\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/run.py baseline\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/diagnose_baseline.py\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/run.py enhanced\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/run.py aggregate\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/evaluate.py\n/tmp/trackid3x3-exp01b-py312/bin/python research/experiments/exp01b_auto_jersey_recovery/render_report.py\n```\n\nOCR stages checkpoint completed calls and can resume interrupted transport/runtime sessions without changing parameters. Package-download and model-download helpers only bootstrap transport; they do not perform inference.']
    (OUT/'README.md').write_text('\n\n'.join(readout)+'\n')
    (OUT/'next_decision.json').write_text(json.dumps({'recommendation':next_decision},indent=2)+'\n')


if __name__=='__main__':main()
